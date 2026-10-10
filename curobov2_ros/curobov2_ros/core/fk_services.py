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

import contextlib

import torch
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Pose

from curobo.kinematics import Kinematics
from curobo.scene import Scene
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.types import DeviceCfg, JointState as CuRoboJS

from curobov2_ros_interfaces.srv import Fk, FkBatch, WarmupFK
from curobov2_ros.core.gpu_lanes import PRIO_KIN


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
                supplying `obstacle_manager`, `robot_model_manager`, device/dtype.
        """
        self._node = node
        self._config = config_wrapper

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

        # Canonicalize an index-less "cuda" to "cuda:0": Warp's
        # wp.device_from_torch indexes its CUDA device list with
        # torch.device(...).index, so the bare device carried by the node's
        # tensor_args (or the fallback above) raises TypeError deep inside
        # SceneData/MeshData construction — historically silently disabling
        # this class's collision validator (poses_valid all True). Upstream
        # reference scripts always use an indexed device (DeviceCfg defaults
        # to torch.device("cuda", 0)).
        if self._device.type == "cuda" and self._device.index is None:
            self._device = torch.device("cuda", 0)

        name = node.get_name()
        node.create_service(WarmupFK, f"{name}/warmup_fk", self._warmup_fk_callback)
        node.create_service(Fk, f"{name}/fk", self._fk_callback)
        node.create_service(FkBatch, f"{name}/fk_batch", self._fk_batch_callback)

        node.get_logger().info(
            "FKServices registered (model allocated by the startup warmup, or "
            "on demand via /warmup_fk when warmup_fk:=false)"
        )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self, batch_size: int) -> str:
        """Build the FK model + collision validator and run a warmup batch.

        Returns the message the ``/warmup_fk`` service reports. Shared with the
        node's startup default warmup (``unified_planner_node``) so both paths
        take the same ``gpu_lock`` discipline around the CUDA work.
        """
        batch_size = max(1, int(batch_size))
        # Model construction + warmup batch run CUDA kernels — must not race a
        # CUDA graph capture (see _gpu_guard).
        with self._gpu_guard():
            self._init(batch_size)
        return f"FK model ready (batch_size={batch_size})"

    def _warmup_fk_callback(
        self, request: WarmupFK.Request, response: WarmupFK.Response
    ):
        try:
            response.message = self.warmup(request.batch_size)
            response.success = True
        except Exception as e:
            self._node.get_logger().error(f"FK warmup failed: {e}")
            response.success = False
            response.message = str(e)
        return response

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _gpu_guard(self):
        """Serialise this callback's CUDA work against CUDA graph captures.

        Every solver/graph capture in the node runs under ``node.gpu_lock``
        (see its docstring in unified_planner_node): a CUDA graph capture is
        process-global, so ANY other CUDA op — including this FK model's
        compute_kinematics kernels and the ``.cpu().numpy()`` host syncs that
        extract the result — issued from another executor thread while a
        capture is in progress invalidates it (cudaErrorStreamCaptureInvalidated).
        Acquired blocking (rather than dropping the request) so on-demand
        service calls stay correct: the caller waits out the brief capture.
        """
        gpu_lock = getattr(self._node, "gpu_lock", None)
        if gpu_lock is None:
            yield
        else:
            with gpu_lock:
                yield

    _LANE_TIMEOUT_S = 120.0

    def _on_lane(self, fn):
        """Run fn on the node's 'kin' lane (sole owner of IK/FK solver state)."""
        lane = getattr(self._node, "lane_kin", None)
        if lane is None:                      # standalone / tests
            with self._gpu_guard():
                return fn()
        return lane.call(fn, timeout=self._LANE_TIMEOUT_S, prio=PRIO_KIN)

    def _fk_callback(self, request: Fk.Request, response: Fk.Response):
        if self._fk_model is None:
            self._node.get_logger().error("FK not initialized. Call warmup_fk first.")
            return response
        if not request.joint_states:
            self._node.get_logger().error("FK: no joint states provided")
            return response
        qs = [self._positions_for_model(js) for js in request.joint_states]

        def _job():
            ok, poses = self._compute_poses(qs)
            if not ok:
                return None
            return poses, [bool(v) for v in self._validate(qs)]

        try:
            out = self._on_lane(_job)
        except Exception as exc:
            self._node.get_logger().warn(
                f"FK failed, returning failure (GPU fault?): {exc}",
                throttle_duration_sec=5.0)
            return response
        if out is None:
            return response
        poses, valid = out
        response.poses = poses
        for v in valid:
            b = Bool()
            b.data = v
            response.poses_valid.append(b)
        return response

    def _fk_batch_callback(self, request: FkBatch.Request, response: FkBatch.Response):
        if self._fk_model is None:
            self._node.get_logger().error("FK not initialized. Call warmup_fk first.")
            response.success = False
            response.error_msg = String(data="FK not initialized. Call warmup_fk first.")
            return response
        if not request.joint_states:
            self._node.get_logger().error("FK batch: no joint states provided")
            response.success = False
            response.error_msg = String(data="FK batch: no joint states provided")
            return response
        qs = [self._positions_for_model(js) for js in request.joint_states]

        def _job():
            ok, poses = self._compute_poses(qs)
            if not ok:
                return None
            return poses, [bool(v) for v in self._validate(qs)]

        try:
            out = self._on_lane(_job)
        except Exception as exc:
            self._node.get_logger().warn(
                f"FK batch failed, returning failure (GPU fault?): {exc}",
                throttle_duration_sec=5.0)
            response.success = False
            response.error_msg = String(data=f"FK batch failed: {exc}")
            return response
        if out is None:
            response.success = False
            response.error_msg = String(data="FK batch solve failed")
            return response
        poses, valid = out
        response.poses = poses
        for v in valid:
            b = Bool()
            b.data = v
            response.poses_valid.append(b)
        response.success = True
        return response

    # ------------------------------------------------------------------
    # World update / rebuild (called by the node when obstacles change)
    # ------------------------------------------------------------------

    def update_world(self):
        """Propagate obstacle changes to the FK collision validator. No-op if
        the validator was never initialized.

        Pushes the PRIMITIVES-ONLY scene, i.e. the same scene the checker was
        constructed with in ``_init`` — NOT ``collision_world_scene()``.
        ``RobotCollisionChecker`` (``curobo.RobotSceneCollision``) is the one
        collision model with no way to pre-allocate a voxel cache:
        ``RobotSceneCollisionCfg.load_from_config`` accepts ``n_meshes`` /
        ``n_cuboids`` but no ``collision_cache``, so its ``VoxelData`` is
        always ``None``. Handing it a scene that carries the perception ESDF
        layer therefore always fails — ``DataScene.add_obstacle`` raises
        "Voxel cache not initialized" for a ``VoxelGrid``. Because the node
        calls this from ``refresh_perception_world()`` inside
        ``_plan_trajectory_goal``, that raise used to abort the whole plan (it
        surfaced as a stage error on any task that refreshed perception).

        Consequence: the validator checks joint bounds, self-collision and
        analytic scene primitives only. Camera/ESDF collision remains the
        planners' job — they own a real cache, sized from
        ``collision_cache['voxel']``.
        """
        if self._collision_checker is None:
            # Rebuild (a failed update disables it below, as does a failed
            # init): the world set that follows a cache change fits the new
            # cache, so the validator heals on the next world event instead
            # of reporting all-True forever.
            self._init(1)
            return
        self._push_world()

    def _push_world(self):
        """Push the primitives-only scene into the existing validator."""
        # Normalize primitives to solver-supported collision types
        # (sphere/cylinder/capsule -> mesh), or they silently don't collide.
        # (primitives_only_scene also applies the attached-object exclusion,
        # the same one the constructor saw.)
        scene = self._obstacle_manager.primitives_only_scene()
        try:
            self._collision_checker.update_world(scene)
        except Exception as e:  # noqa: BLE001
            # The validator is advisory — it only fills poses_valid, and
            # _validate already reports all-True without it. Drop it rather
            # than let a failed refresh abort the plan that triggered it.
            self._node.get_logger().error(
                f"FK collision validator world update failed ({e}); disabling "
                "it — FkBatch poses_valid now reports all True"
            )
            self._collision_checker = None

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

        # Build the FK model from the node's SINGLE shared curobo kinematic
        # (robot_model_manager.robot_cfg), never from the YAML path — otherwise
        # each service instantiates its own robot model (one URDF parse + one
        # set of intermediate buffers per construction; cf.
        # RobotModelManager docstring).
        robot_cfg = self._config.robot_model_manager.robot_cfg
        fk_model = Kinematics(robot_cfg.kinematics)
        self._fk_model = fk_model

        # Collision validator for FkBatch: sized from the shared collision_cache
        # so it holds exactly what the planners' does, and built against an
        # EMPTY scene (never the live one: after a cache shrink the live world
        # may not fit the new cache — same "Cannot load N cuboids" failure as
        # the planners; empty always fits and update_world() pushes the real
        # scene below). Synced later via update_world().
        # NOTE: load_from_config has no voxel-cache knob at all (see
        # update_world), so the ESDF voxel layer is not double-counted here.
        scene = Scene()
        robot_cfg_dict = self._config.config_manager.get_robot_config_dict()
        # Size the validator's obstacle cache from the shared collision_cache so
        # it holds exactly what the planners' does. load_from_config has no
        # voxel-cache knob at all (see update_world), and its n_cuboids/n_meshes
        # defaults (50/50) are unrelated to collision_cache_cuboid/mesh (32/4) —
        # left at the defaults, a world the planners accept can overflow the
        # validator instead, raising "Cuboid cache is full" out of update_world.
        cache = self._obstacle_manager.collision_cache
        try:
            checker_cfg = RobotCollisionCheckerCfg.load_from_config(
                robot_config=robot_cfg_dict,
                scene_model=scene,
                device_cfg=DeviceCfg(device=self._device, dtype=self._dtype),
                collision_activation_distance=0.001,
                n_cuboids=max(1, int(cache.get("cuboid") or 0)),
                n_meshes=max(1, int(cache.get("mesh") or 0)),
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

        # Push the real scene after the empty build (validator is advisory-only
        # so an overflow here must not fail the build — see the try above).
        # Same-cache rebuilds always fit; after a cache shrink the previous
        # world may not fit and the next world event pushes a fitting one.
        # _push_world (not update_world: that would recurse into _init when
        # the checker is missing).
        try:
            self._push_world()
        except Exception as e:
            self._node.get_logger().warn(
                f"FK validator rebuilt empty: current world does not fit the "
                f"new cache ({e}); poses_valid stays all-True until the next "
                f"world update")

        self._node.get_logger().info("FK model ready")

    def _positions_for_model(self, js):
        """Map one JointState's positions onto the FK model's joint order.

        The model is built from the robot config's cspace kinematics (7 DOF
        for franka, fingers excluded), but callers routinely send full robot
        states — e.g. the parity benchmark FK's the planner's returned
        trajectory waypoints, which carry all 9 cspace joints (7 + 2 locked
        fingers). Select by joint name in model order when the incoming state
        carries aligned names (any missing name falls back to a positional
        truncation), else take the first ``dof`` positions. This keeps the
        service contract "joint states → poses" intact for any caller.
        """
        positions = list(js.position)
        names = list(getattr(js, "name", None) or [])
        if len(names) == len(positions):
            by_name = {n: p for n, p in zip(names, positions)}
            try:
                return [by_name[n] for n in self._fk_model.joint_names]
            except KeyError:
                pass
        return positions[: self._fk_model.get_dof()]

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
