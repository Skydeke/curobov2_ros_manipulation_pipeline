#!/usr/bin/env python3
"""
FK services for the unified planner node (v2).

Provides a lazy-initialized FK model plus a collision validator used by the
batch FK endpoint. The `Fk` service needs only the robot's kinematic model;
`FkBatch` additionally validates each configuration (joint limits,
self-collision, scene collision) for its `poses_valid` output.

Services exposed (prefixed with the node name):
  /<node>/warmup_fk  (WarmupFK)     - init FK model with given batch size (default 1)
  /<node>/fk         (Fk)           - joint states → poses
  /<node>/fk_batch   (FkBatch)      - joint states → poses (+ collision validity)

v2 notes:
- CudaRobotModel → Kinematics (curobo.kinematics).
- RobotConfig no longer exists; KinematicsCfg.create accepts the YAML path
  (or dict) via `robot=`.
- Collision validation for FkBatch uses curobo.collision_checking
  RobotCollisionChecker (a RobotSceneCollision), which checks joint bounds,
  self-collision and scene collision.
"""

import torch
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Pose

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.types import DeviceCfg, JointState as CuRoboJS

from isaac_ros_cumotion_interfaces.srv import Fk, FkBatch, WarmupFK


class FKServices:
    """
    Manages the FK model, a collision validator, and their ROS services.

    Depends on:
    - config_wrapper.obstacle_manager (Scene, shared with the planners) — only
      used to keep the FkBatch collision validator in sync with the world.
    - robot's CUDA device/dtype (via tensor_args or config_wrapper).

    The model and validator are created only when warmup_fk is called.
    `Fk` is purely geometric; `FkBatch` also reports per-config validity.
    """

    def __init__(self, node, config_wrapper):
        """
        Args:
            node: ROS2 node.
            config_wrapper: The shared ConfigWrapperMotion (like IKServices),
                supplying `obstacle_manager`, `robot_config_file`, device/dtype.
        """
        self._node = node
        self._config = config_wrapper

        self._robot_config_file = config_wrapper.robot_config_file
        self._obstacle_manager = config_wrapper.obstacle_manager

        self._fk_model: Kinematics | None = None
        self._collision_checker: RobotCollisionChecker | None = None

        # Resolve device/dtype from the node's tensor_args if present, else from
        # the config wrapper, else default to CUDA/float32.
        tensor_args = getattr(node, "tensor_args", None)
        if tensor_args is not None and hasattr(tensor_args, "device"):
            self._device = torch.device(tensor_args.device)
            self._dtype = getattr(tensor_args, "dtype", torch.float32)
        else:
            self._device = getattr(config_wrapper, "_device", torch.device("cuda"))
            self._dtype = getattr(config_wrapper, "_ops_dtype", torch.float32)

        name = node.get_name()
        node.create_service(WarmupFK, f"{name}/warmup_fk", self._warmup_fk_callback)
        node.create_service(Fk, f"{name}/fk", self._fk_callback)
        node.create_service(FkBatch, f"{name}/fk_batch", self._fk_batch_callback)

        node.get_logger().info(
            "FKServices registered (not yet initialized - call warmup_fk)"
        )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup_fk_callback(
        self, request: WarmupFK.Request, response: WarmupFK.Response
    ):
        batch_size = max(1, request.batch_size)
        try:
            self._init(batch_size)
            response.success = True
            response.message = f"FK model ready (batch_size={batch_size})"
        except Exception as e:
            self._node.get_logger().error(f"FK warmup failed: {e}")
            response.success = False
            response.message = str(e)
        return response

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    def _fk_callback(self, request: Fk.Request, response: Fk.Response):
        if self._fk_model is None:
            self._node.get_logger().error("FK not initialized. Call warmup_fk first.")
            return response

        if not request.joint_states:
            self._node.get_logger().error("FK: no joint states provided")
            return response

        qs = [list(js.position) for js in request.joint_states]
        ok, poses = self._compute_poses(qs)
        if not ok:
            return response

        response.poses = poses

        # Fk.srv also declares poses_valid; populate it from the same validator
        # (joint limits, self-collision, scene collision).
        for v in self._validate(qs):
            b = Bool()
            b.data = bool(v)
            response.poses_valid.append(b)
        return response

    def _fk_batch_callback(self, request: FkBatch.Request, response: FkBatch.Response):
        if self._fk_model is None:
            self._node.get_logger().error("FK not initialized. Call warmup_fk first.")
            response.success = False
            response.error_msg = String(
                data="FK not initialized. Call warmup_fk first."
            )
            return response

        if not request.joint_states:
            self._node.get_logger().error("FK batch: no joint states provided")
            response.success = False
            response.error_msg = String(data="FK batch: no joint states provided")
            return response

        qs = [list(js.position) for js in request.joint_states]

        ok, poses = self._compute_poses(qs)
        if not ok:
            response.success = False
            response.error_msg = String(data="FK batch solve failed")
            return response

        # Validate each configuration: joint limits, self-collision, scene
        # collision (see RobotCollisionChecker.validate()).
        valid = self._validate(qs)

        response.poses = poses
        for v in valid:
            b = Bool()
            b.data = bool(v)
            response.poses_valid.append(b)
        response.success = True
        return response

    # ------------------------------------------------------------------
    # World update / rebuild (called by the node when obstacles change)
    # ------------------------------------------------------------------

    def update_world(self):
        """Propagate obstacle changes to the FK collision validator. No-op if
        the validator was never initialized."""
        if self._collision_checker is None:
            return
        # Normalize primitives to solver-supported collision types
        # (sphere/cylinder/capsule -> mesh), or they silently don't collide.
        scene = self._obstacle_manager.collision_world_scene()
        self._collision_checker.update_world(scene)
        self._node.get_logger().info("FKServices: world updated")

    def rebuild(self):
        """Recreate the FK model after a robot-config change. No-op if the
        model was never initialized."""
        if self._fk_model is None:
            return
        self._init(1)
        self._node.get_logger().info("FKServices: model rebuilt")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init(self, batch_size: int):
        """Create the FK model (+ collision validator) and run a warmup batch."""
        self._node.get_logger().info(
            f"Initializing FK model (batch_size={batch_size})..."
        )

        fk_model = Kinematics(
            KinematicsCfg.from_robot_yaml_file(
                self._robot_config_file,
                device_cfg=DeviceCfg(device=self._device, dtype=self._dtype),
            )
        )
        self._fk_model = fk_model

        # Collision validator for FkBatch: built from the same primitives-only
        # Scene the planners are constructed with, so cache is honoured and the
        # ESDF voxel layer is not double-counted. Synced later via update_world().
        scene = self._obstacle_manager.primitives_only_scene()
        robot_cfg_dict = self._config.config_manager.get_robot_config_dict()
        try:
            checker_cfg = RobotCollisionCheckerCfg.load_from_config(
                robot_config=robot_cfg_dict,
                scene_model=scene,
                device_cfg=DeviceCfg(device=self._device, dtype=self._dtype),
                collision_activation_distance=0.001,
            )
            self._collision_checker = RobotCollisionChecker(checker_cfg)
        except Exception as e:
            self._node.get_logger().warn(
                f"FK collision validator init failed ({e}); FkBatch poses_valid "
                "will report all True"
            )
            self._collision_checker = None

        q = torch.rand(
            (batch_size, fk_model.get_dof()),
            dtype=self._dtype,
            device=self._device,
        )
        js = CuRoboJS.from_position(q, joint_names=fk_model.joint_names)
        fk_model.compute_kinematics(js)

        self._node.get_logger().info("FK model ready")

    def _compute_poses(self, qs):
        """
        Compute FK poses for a list of joint-position lists.
        Returns (success: bool, poses: list[geometry_msgs/Pose]).
        """
        if not qs:
            self._node.get_logger().error("FK: empty joint state list")
            return False, []

        if self._fk_model is None:
            self._node.get_logger().error("FK not initialized. Call warmup_fk first.")
            return False, []

        fk_model = self._fk_model
        q = torch.tensor(qs, dtype=self._dtype, device=self._device)
        js = CuRoboJS.from_position(q, joint_names=fk_model.joint_names)
        kin_state = fk_model.compute_kinematics(js)

        # ToolPose.position/quaternion: [B, H=1, L, 3/4]; take first tool frame.
        # v2 quaternion is wxyz; ROS geometry_msgs.Pose.orientation is xyzw.
        positions = kin_state.tool_poses.position[:, 0, 0, :].cpu().numpy()
        quaternions = kin_state.tool_poses.quaternion[:, 0, 0, :].cpu().numpy()

        poses = []
        for pos, ori in zip(positions, quaternions):
            pose = Pose()
            pose.position.x = float(pos[0])
            pose.position.y = float(pos[1])
            pose.position.z = float(pos[2])
            pose.orientation.w = float(ori[0])
            pose.orientation.x = float(ori[1])
            pose.orientation.y = float(ori[2])
            pose.orientation.z = float(ori[3])
            poses.append(pose)
        return True, poses

    def _validate(self, qs):
        """
        Validate a list of joint configurations for collision.
        Returns a list of bool per configuration. When the validator is
        unavailable, all configs are reported valid (True).
        """
        n = len(qs)
        if self._collision_checker is None:
            return [True] * n

        q = torch.tensor(qs, dtype=self._dtype, device=self._device).unsqueeze(
            1
        )  # [B, 1, dof]
        try:
            mask = self._collision_checker.validate(q)  # [B, 1]
            return mask.squeeze(1).cpu().tolist()
        except Exception as e:
            self._node.get_logger().error(f"FK collision validation failed: {e}")
            return [True] * n
