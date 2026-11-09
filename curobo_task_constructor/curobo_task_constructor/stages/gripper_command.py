"""GripperCommand — move an end-effector's jaws without planning.

The stage carries a named gripper plus a target and no ``PlanRequest``; the
executor sends it to that gripper's controller action. Direction is irrelevant.

The propagated state is the CURRENT robot state merged with the commanded
gripper joint(s): every joint the incoming state already names (the previous
stage's arm pose) wins, every joint it does not name is filled from the live
``/joint_states`` reading, and only the gripper joints are overwritten. That is
what lets a one-stage "open the jaws" plan touch no arm joint while still
chaining after an arm motion.
"""

from __future__ import annotations

from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.stage import PropagatingEitherWay, Solution
from curobo_task_constructor.core.state import InterfaceState
from curobo_task_constructor.stages._util import (
    gripper_joint_names,
    measured_joint_state,
    merged_joint_state,
)


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
        # Start from what the chain knows; fill only what it does not name
        # from the measured reading. Preferring the incoming state is what
        # makes a command after an arm leg sit on that leg's solved pose
        # instead of the plan-time snapshot the measurement still holds.
        measured = measured_joint_state(self.robot)
        incoming = getattr(state, "joint_state", None)
        base_joint_state = merged_joint_state(self.robot, incoming, measured)
        # Carry the commanded jaw value in the state itself: downstream motion
        # legs then plan (and preview) from the task's actual gripper state,
        # not from the planner's stale lock-joint value. The planner's cspace
        # ignores the extra joint; it is only there for the state chain, the
        # solution's start/end snapshots and the ghost preview. Every joint
        # the controller moves is updated (TIAGo has two fingers on one
        # FollowJointTrajectory command).
        joints = gripper_joint_names(
            self.robot, cmd.get("gripper", ""), cmd.get("joint", ""))
        end_joint_state = merged_joint_state(
            self.robot, base_joint_state, measured,
            overrides={name: cmd["position"] for name in joints})
        end = state.clone(joint_state=end_joint_state)
        target = cmd["joint"] or cmd["gripper"]
        # Container chaining is by InterfaceState IDENTITY (see
        # SerialContainer._traces): the endpoint that came from the neighbour
        # must stay the SAME object, so the merged state is only the newly
        # written one. Forward: read ``state``, write ``end``. Backward: read
        # ``state``, write the post-command ``end`` upstream.
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
