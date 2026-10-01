import math

import torch
from typing import List, Optional, Sequence, Tuple

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo._src.types.robot import RobotCfg
from curobo.types import JointState
from curobov2_ros_interfaces.srv import SetLinkCollision


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
        # Torque-limited planning: `load_dynamics` (a build-time node param —
        # see docs/concepts/parameters.md) wraps the kinematics config with a
        # pinocchio dynamics config, exactly what curobo's RobotCfg.create
        # does for the upstream benchmark. The robot_cfg is a plain attribute
        # (NOT launch-immutable): set_torque_mode() re-wraps it with/without
        # dynamics on demand, and ConfigWrapperMotion.set_motion_gen_config()
        # syncs it from the node's current `load_dynamics` param before every
        # solver build — so the runtime trigger path (ros2 param set … +
        # /unified_planner/update_motion_gen_config) can switch a running
        # server between the reference page's two tables without a relaunch.
        # The kinematics side is never rebuilt by the torque switch: `kin_cfg`
        # is THE SAME object across it, so every solver built from
        # robot_cfg.kinematics (MotionPlanner, IK, FK) keeps a valid GPU model.
        # set_lock_joints() is the one switch that DOES re-parse `kin_cfg` (the
        # lock is baked into the parsed kinematics config), and it rebuilds
        # `kin_model` with it.
        self._load_dynamics_active = None
        load_dynamics = False
        if node is not None and node.has_parameter("load_dynamics"):
            load_dynamics = bool(
                node.get_parameter("load_dynamics")
                .get_parameter_value()
                .bool_value
            )
        # Active `kinematics.lock_joints` override (None = none; the robot
        # YAML's own values are in force). See set_lock_joints /
        # update_lock_joints.
        self._lock_joints_override = None
        self.set_torque_mode(load_dynamics)
        self.kin_model = Kinematics(self.robot_cfg.kinematics)

        self._ops_dtype = torch.float32
        self._device = torch.device('cuda')

    def _wrap_robot_cfg(self, kin_cfg):
        """Build a ``RobotCfg`` around ``kin_cfg`` and the current torque mode.

        Pure function of its arguments (plus ``_load_dynamics_active``), so a
        caller can assemble the whole replacement model and only commit it once
        every step has succeeded.
        """
        dynamics = None
        if self._load_dynamics_active:
            from curobo._src.types.device_cfg import DeviceCfg

            dynamics = RobotCfg._create_dynamics_config(
                kinematics_config=kin_cfg.kinematics_config,
                device_cfg=DeviceCfg(),
            )
        return RobotCfg(kinematics=kin_cfg, dynamics=dynamics)

    def _rewrap_robot_cfg(self) -> None:
        """Re-wrap ``robot_cfg`` around the current ``kin_cfg`` and torque mode.

        Shared by every path that changes what the robot cfg must contain:
        the torque-mode switch (which swaps ``dynamics``) and the joint-lock
        switch (which swaps ``kin_cfg`` itself). The kinematics side is
        untouched here — ``kin_model`` is rebuilt separately by whoever
        replaced ``kin_cfg``.
        """
        self.robot_cfg = self._wrap_robot_cfg(self.kin_cfg)

    def set_torque_mode(self, load_dynamics: bool) -> bool:
        """Re-wrap ``robot_cfg`` with/without torque-limited dynamics.

        Returns True when the mode actually changed (a switch) and False when
        it already matched — idempotent, so every solver rebuild can call it
        from the node's current ``load_dynamics`` param at no cost when the
        launch mode is unchanged. Only the ``dynamics`` field varies: the
        shared ``kin_cfg`` is reused as-is, so existing GPU Kinematics models
        remain valid and only a MotionPlanner rebuild (via
        ``update_motion_gen_config``) is needed to pick the new mode up.
        """
        load_dynamics = bool(load_dynamics)
        if load_dynamics == self._load_dynamics_active:
            return False
        self._load_dynamics_active = load_dynamics
        self._rewrap_robot_cfg()
        if self.node is not None:
            self.node.get_logger().info(
                f"RobotCfg torque mode "
                f"{'enabled' if load_dynamics else 'disabled'} "
                f"(load_dynamics={str(load_dynamics).lower()}) — rebuild the "
                "MotionPlanner to pick it up"
            )
        return True

    def config_lock_joints(self) -> dict:
        """The robot YAML's own ``kinematics.lock_joints`` as ``{joint: value}``.

        Read straight from the YAML (rather than off ``kin_cfg``) so this is
        the authoritative answer even while an override is in force.
        """
        from curobo._src.util.config_io import load_yaml

        content = load_yaml(self.robot_config_file)
        # Unwrap exactly as KinematicsCfg.from_content_path does, so the values
        # here are the ones a restore re-parses with.
        if "robot_cfg" in content:
            content = content["robot_cfg"]
        if "kinematics" in content:
            content = content["kinematics"]
        return {
            str(name): float(value)
            for name, value in (content.get("lock_joints") or {}).items()
        }

    def lockable_joints(self) -> List[str]:
        """Every joint that can be locked, in the robot model's own order.

        The union of the optimizable joints (``kin_model.joint_names``) and the
        ones already locked (``kin_model.lock_jointstate``) — locking a joint
        takes it out of the cspace, never out of the model, so a caller may
        re-pin a locked joint or pin an active one, and a joint outside this
        set is not part of the robot at all. Order matters only for reporting
        (see :meth:`_in_model_order`); the model, not the caller, is what
        bounds the set.
        """
        names = list(self.kin_model.joint_names)
        lock_state = getattr(self.kin_model, "lock_jointstate", None)
        if lock_state is not None:
            names += list(lock_state.joint_names)
        return names

    def _in_model_order(self, locks: dict) -> dict:
        """``locks`` reordered to follow the model's joint order.

        So the effective lock set reads back the same way whatever order the
        caller or the YAML used — both for the service response and because the
        loader bakes the mapping into the kinematic tree in the given order.
        Names the model does not know keep their relative order at the end.
        """
        rank = {name: index for index, name in enumerate(self.lockable_joints())}
        return {
            name: locks[name]
            for name in sorted(locks, key=lambda n: rank.get(n, len(rank)))
        }

    def lock_joints(self) -> dict:
        """The lock values actually in force, in model joint order.

        The override when one is set, the robot YAML's own values otherwise.
        """
        locks = self._lock_joints_override
        if locks is None:
            locks = self.config_lock_joints()
        return self._in_model_order(locks)

    def set_lock_joints(self, lock_joints) -> bool:
        """Replace ``kinematics.lock_joints`` outright and rebuild the model.

        ``lock_joints`` maps joint name to the value to pin it at; a joint left
        out is released back to the optimizable cspace, so a full mapping is
        what a caller sends to set the whole state in one go (see
        :meth:`update_lock_joints` for the partial form). Pass ``None`` to drop
        the override and reload the YAML's values.

        Names must be joints of the model (see :meth:`lockable_joints`) and
        values finite — a NaN would be baked straight into a fixed transform.
        curobo has the last word: an unlock the config's ``cspace.joint_names``
        does not list is rejected at re-parse, which leaves the model untouched.

        Locked joints are held at their value and excluded from the
        optimizable cspace, and curobo bakes them into
        ``KinematicsConfig.lock_jointstate`` when the kinematics config is
        loaded — so unlike the torque mode (which only re-wraps ``robot_cfg``)
        this one has to re-parse the robot YAML and rebuild ``kin_cfg``,
        ``robot_cfg`` and the GPU ``kin_model``. Callers must then rebuild
        every solver built from the model (see
        ``UnifiedPlannerNode._rebuild_all_solvers``).

        Returns True when the model was actually rebuilt and False when the
        requested locks already match the active ones — idempotent, so a
        rebuild triggered for another reason costs nothing here.
        """
        current = self.lock_joints()
        if lock_joints is None:
            requested = self._in_model_order(self.config_lock_joints())
        else:
            requested = self._in_model_order(
                {str(name): float(value) for name, value in lock_joints.items()}
            )
            lockable = self.lockable_joints()
            unknown = [name for name in requested if name not in set(lockable)]
            if unknown:
                raise ValueError(
                    "lock_joints names joints that are not part of the robot "
                    f"model {lockable}: {sorted(unknown)}"
                )
            non_finite = sorted(
                name for name, value in requested.items() if not math.isfinite(value)
            )
            if non_finite:
                raise ValueError(
                    f"lock_joints values must be finite: {non_finite}"
                )
        # Idempotency: only skip the rebuild when the requested locks name the
        # same joints as the active ones AND match. Key-set equality is checked
        # first, so a config edited under a live override (a lock joint added
        # or removed on disk) always rebuilds instead of indexing a missing key.
        if requested.keys() == current.keys() and all(
            abs(requested[name] - current[name]) < 1e-12 for name in requested
        ):
            return False

        # Re-parse through the same loader as startup (path only, so relative
        # urdf_path / collision_sphere paths still resolve against the config
        # directory) with lock_joints applied as a build-time override, then
        # assemble the whole replacement model before committing it: a failure
        # anywhere here must leave the node on the model it already had.
        kin_cfg = KinematicsCfg.from_robot_yaml_file(
            self.robot_config_file, lock_joints=requested
        )
        robot_cfg = self._wrap_robot_cfg(kin_cfg)
        kin_model = Kinematics(robot_cfg.kinematics)

        self.kin_cfg = kin_cfg
        self.robot_cfg = robot_cfg
        self.kin_model = kin_model
        self._lock_joints_override = None if lock_joints is None else dict(requested)
        if self.node is not None:
            self.node.get_logger().info(
                f"Robot model rebuilt with lock_joints {requested} — rebuild "
                "the solvers to pick it up"
            )
        return True

    def update_lock_joints(
        self,
        lock: Optional[dict] = None,
        unlock: Optional[Sequence[str]] = None,
    ) -> bool:
        """Change *some* of the joint locks and rebuild the model.

        ``lock`` maps a joint to the value to pin it at and ``unlock`` names
        joints to release back to the optimizable cspace; a joint in neither
        keeps the value already in force. That is what lets a caller move one
        joint — or one finger's value — without restating the rest of the
        model. The operations apply to the locks in force *right now*, so
        successive calls compose: pin the gripper, then pin the arm, and the
        gripper keeps its value.

        Releases are idempotent rather than an error (unlocking a joint that is
        already free changes nothing), so a caller can assert a state instead
        of tracking it. Everything else defers to :meth:`set_lock_joints` —
        same validation, same rebuild, same return contract.
        """
        lock = {str(name): float(value) for name, value in (lock or {}).items()}
        unlock = [str(name) for name in (unlock or [])]
        both = sorted(set(lock) & set(unlock))
        if both:
            raise ValueError(
                f"joints cannot be locked and unlocked in the same call: {both}"
            )
        requested = self.lock_joints()
        for name in unlock:
            requested.pop(name, None)
        requested.update(lock)
        return self.set_lock_joints(requested)

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
