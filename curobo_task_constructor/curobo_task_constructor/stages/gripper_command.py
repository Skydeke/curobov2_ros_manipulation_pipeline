"""GripperCommand — move an end-effector's jaws without planning.

The stage carries a named gripper plus a target and no ``PlanRequest``; the
executor sends it to that gripper's controller action. Direction is irrelevant.
"""

from __future__ import annotations

from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.stage import PropagatingEitherWay, Solution
from curobo_task_constructor.core.state import InterfaceState, make_joint_state


@register_stage("gripper_command")
class GripperCommand(PropagatingEitherWay):
    def __init__(self, name=None, params=None, direction="auto"):
        super().__init__(name, params, direction=direction)

    def command(self) -> dict:
        p = self.params or {}
        goal = p.get("goal") or {}
        position = goal.get("position", p.get("position", 0.0))
        effort = goal.get("effort", p.get("effort", 0.0))
        max_effort = goal.get("max_effort", p.get("max_effort", 0.0))
        return {
            "gripper": str(p.get("gripper", "") or ""),
            "joint": str(p.get("joint", "") or ""),
            "position": float(position),
            "effort": float(effort),
            "max_effort": float(max_effort),
            "controller": str(p.get("controller", "") or ""),
            "action_ns": str(p.get("action_ns", "") or ""),
        }

    def _emit_command(self, state: InterfaceState, write_start: bool) -> None:
        cmd = self.command()
        end = state.clone()
        # Carry the commanded jaw value in the state itself: downstream motion
        # legs then plan (and preview) from the task's actual gripper state,
        # not from the planner's stale lock-joint value. The planner's cspace
        # ignores the extra joint; it is only there for the state chain, the
        # solution's start/end snapshots and the ghost preview.
        joint = cmd.get("joint")
        if joint:
            js = getattr(end, "joint_state", None)
            names = list(getattr(js, "name", []) or [])
            positions = list(getattr(js, "position", []) or [])
            if joint in names:
                index = names.index(joint)
                if index < len(positions):
                    positions[index] = float(cmd["position"])
                else:
                    positions.append(float(cmd["position"]))
            else:
                names.append(joint)
                positions.append(float(cmd["position"]))
            end = state.clone(joint_state=make_joint_state(
                getattr(self.robot, "joint_state_cls", None), names, positions))
        target = cmd["joint"] or cmd["gripper"]
        sol = Solution(
            start=end if write_start else state,
            end=state if write_start else end,
            trajectory=None,
            cost=0.0,
            comment=f"gripper_command {target}={cmd['position']}",
            gripper_command=cmd,
        )
        self._consider(sol)
        # A command is instantaneous: both endpoints are the same state.
        if write_start:
            self._promote_backward(sol.start, state, sol)
        else:
            self._promote_forward(state, sol.end, sol)

    def compute_forward(self, state: InterfaceState) -> None:
        self._emit_command(state, write_start=False)

    def compute_backward(self, state: InterfaceState) -> None:
        self._emit_command(state, write_start=True)
