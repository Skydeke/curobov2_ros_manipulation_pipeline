"""FixedState — a generator that resolves a named joint configuration.

The name is looked up in the robot's own descriptor YAML's
``named_joint_configs`` section (via ``RobotInterface.get_named_joint_config``
or a ``robot_config_path`` param), *not* in a curobo_task_constructor-owned
file — same descriptor that ``GetRobotStrategies`` reads.
"""

from __future__ import annotations

from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.robot_config import resolve_named_config
from curobo_task_constructor.core.stage import GeneratorStage
from curobo_task_constructor.core.state import InterfaceState, make_joint_state


@register_stage("fixed_state")
class FixedState(GeneratorStage):
    def compute(self) -> None:
        config = self._config
        if config is None:
            self._fail(None, None, f"named joint config '{self._goal_name}' "
                                   "could not be resolved")
            return
        robot = self.robot
        current = robot.get_current_joint_state()
        names = config.names or list(getattr(current, "name", []) or [])
        positions = self._positions_for(names, config)
        joint = make_joint_state(getattr(robot, "joint_state_cls", None),
                                 names, positions)
        self.spawn(InterfaceState(joint_state=joint, scene=self._base_scene),
                   comment=f"named config '{self._goal_name}'")

    def _positions_for(self, names: list, config) -> list:
        if config.names:
            base = dict(zip(getattr(self.robot.get_current_joint_state(), "name", []),
                            getattr(self.robot.get_current_joint_state(), "position", [])))
            base.update(config.as_dict())
            return [base[n] for n in names]
        return list(config.positions)  # flat list: canonical joint order

    def init(self, base_scene, robot) -> None:
        super().init(base_scene, robot)
        self._base_scene = base_scene
        self._goal_name = self.params.get("goal", "")
        # Prefer the robot interface's own resolver (ROS adapter may read the
        # descriptor itself); fall back to reading the YAML here.
        cfg = None
        try:
            cfg = robot.get_named_joint_config(self._goal_name)
        except (KeyError, NotImplementedError):
            cfg = None
        if cfg is None:
            path = self.params.get("robot_config_path")
            if path and self._goal_name:
                cfg = resolve_named_config(path, self._goal_name)
        self._config = cfg