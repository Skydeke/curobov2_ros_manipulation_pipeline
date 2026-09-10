import torch
from typing import List, Tuple

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo._src.types.robot import RobotCfg
from curobo.types import JointState
from isaac_ros_cumotion_interfaces.srv import SetLinkCollision


class RobotModelManager:
    """
    Manages the robot kinematics and collision geometry.

    v2 notes:
    - `CudaRobotModel` → `Kinematics` (curobo.kinematics)
    - `RobotConfig` no longer exists; `KinematicsCfg.from_robot_yaml_file`
      parses the robot YAML exactly once here (the URDF config is parsed a
      single time per process).
    - `kin_cfg` is the parsed `KinematicsCfg`, `robot_cfg` the wrapping
      `RobotCfg` passed into `*.cfg.create(robot=…)` by every other solver so
      nothing re-parses the robot config, and `kin_model` the GPU `Kinematics`.
    - KinematicsTensorConfig still lives on the kinematics instance; the
      `enable_link_spheres` / `disable_link_spheres` API is unchanged.
    """

    def __init__(self, robot_config_file: str, robot, base_link: str, node=None):
        """
        Args:
            robot_config_file: Path to the robot YAML config (v2 accepts this directly).
            robot: Robot interface object for accessing joint states.
            base_link: Robot base frame name.
            node: ROS2 node (for logging and service registration).
        """
        self.robot_config_file = robot_config_file
        self.robot = robot
        self.base_link = base_link
        self.node = node

        # `RobotModelManager` owns the SINGLE curobo kinematic model for the
        # node. One `RobotCfg` wraps one parsed `KinematicsCfg` (the URDF is
        # parsed exactly once, here); every other solver built afterwards
        # (MotionPlanner, IK, FK, MPC) constructs its GPU `Kinematics` from
        # THIS `robot_cfg` instead of re-parsing the robot YAML/URDF — that was
        # loading and configuring the robot model multiple times per process
        # (one set of "Converting continuous joint" warnings per construction).
        self.kin_cfg = KinematicsCfg.from_robot_yaml_file(robot_config_file)
        self.robot_cfg = RobotCfg(kinematics=self.kin_cfg)
        self.kin_model = Kinematics(self.robot_cfg.kinematics)

        self._ops_dtype = torch.float32
        self._device = torch.device('cuda')

    def get_kinematics_state(self, joint_positions):
        # v2: kin_model.get_state removed — compute_kinematics takes a JointState.
        js = JointState(
            position=joint_positions if isinstance(joint_positions, torch.Tensor) else torch.tensor(
                joint_positions, dtype=self._ops_dtype, device=self._device
            ),
            joint_names=self.kin_model.joint_names,
        )
        return self.kin_model.compute_kinematics(js)

    def get_collision_spheres(self) -> List[List[float]]:
        q_js = JointState(
            position=torch.tensor(
                self.robot.get_joint_pose(),
                dtype=self._ops_dtype,
                device=self._device,
            ),
            joint_names=self.kin_model.joint_names,
        )

        kinematics_state = self.kin_model.compute_kinematics(q_js)
        # robot_spheres shape: [batch, horizon, num_spheres, 4]
        robot_spheres = kinematics_state.robot_spheres.reshape(-1, 4)
        return robot_spheres.cpu().numpy().tolist()

    def get_collision_spheres_with_attached(
        self, kinematics, attach_link: str = "attached_object"
    ) -> Tuple[List[List[float]], List[bool]]:
        """World collision spheres from an EXTERNAL kinematics, flagging which
        belong to ``attach_link``.

        Our own ``kin_model`` never receives grasp attaches (those live on the
        MotionPlanner). Pass the MotionPlanner's kinematics here to render the
        fitted attached-object spheres. Returns ``(spheres [[x,y,z,r], …],
        attached_mask [bool, …])``; when nothing is attached the attach_link
        spheres carry radius <= 0 and are filtered by the caller.
        """
        q_js = JointState(
            position=torch.tensor(
                self.robot.get_joint_pose(),
                dtype=self._ops_dtype,
                device=self._device,
            ),
            joint_names=kinematics.joint_names,
        )
        spheres = kinematics.compute_kinematics(q_js).robot_spheres.reshape(-1, 4)
        spheres = spheres.cpu().numpy().tolist()

        kc = kinematics.kinematics_config
        attached_idx = kc.link_name_to_idx_map.get(attach_link)
        if attached_idx is None or kc.link_sphere_idx_map is None:
            return spheres, [False] * len(spheres)

        link_ids = kc.link_sphere_idx_map.reshape(-1).cpu().tolist()
        mask = [li == attached_idx for li in link_ids]
        if len(mask) != len(spheres):  # defensive: align to sphere count
            mask = (mask + [False] * len(spheres))[:len(spheres)]
        return spheres, mask

    def set_link_collision(
        self, link_names: List[str], enabled: bool
    ) -> Tuple[List[str], List[str]]:
        kc = self.kin_model.kinematics_config
        applied, unknown = [], []
        for link in link_names:
            if link not in kc.link_name_to_idx_map:
                unknown.append(link)
            else:
                if enabled:
                    kc.enable_link_spheres(link)
                else:
                    kc.disable_link_spheres(link)
                applied.append(link)
        return applied, unknown

    def set_link_collision_callback(
        self,
        request: SetLinkCollision.Request,
        response: SetLinkCollision.Response,
    ) -> SetLinkCollision.Response:
        applied, unknown = self.set_link_collision(list(request.link_names), request.enabled)

        response.applied_links = applied
        response.unknown_links = unknown

        if unknown:
            kc = self.kin_model.kinematics_config
            if self.node:
                self.node.get_logger().warn(
                    f"set_link_collision: unknown links {unknown}. "
                    f"Available: {list(kc.link_name_to_idx_map.keys())}"
                )

        if applied:
            state = "enabled" if request.enabled else "disabled"
            if self.node:
                self.node.get_logger().info(f"Collision spheres {state} for: {applied}")
            response.success = True
            response.message = f"Collision {state} for {applied}"
        else:
            response.success = False
            response.message = f"No valid links found in: {list(request.link_names)}"

        return response

    def get_joint_state(self) -> JointState:
        # Canonical joint names come from the kinematics model (DOF-agnostic).
        # The previous hardcoded ['joint_1'..'joint_6'] did not even match the
        # real m1013 names ('joint1'..'joint6') and broke any non-6-DOF robot.
        return JointState(
            position=torch.tensor(
                self.robot.get_joint_pose(),
                dtype=self._ops_dtype,
                device=self._device,
            ),
            joint_names=self.kin_model.joint_names,
        )
