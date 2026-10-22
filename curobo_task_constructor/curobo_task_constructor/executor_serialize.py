"""TaskExecutor — MTC's ``Task::init()`` / ``plan()`` / ``execute()`` for cuRobo. Lifecycle (Sec. 3 of the plan):

1. ``build``    — turn the ``StageSpec`` tree into concrete stages via the open
                  ``STAGE_REGISTRY`` (graph.builder.build_tree)."""

from __future__ import annotations

from typing import Any, Optional

from curobo_task_constructor.core.container import GENERATE_INTERFACE
from curobo_task_constructor.core.robot import ObjectSpec, PlanResult, RobotInterface
from curobo_task_constructor.core.stage import InitStageError, Solution
from curobo_task_constructor.core.state import SceneDiff
from curobo_task_constructor.graph.builder import build_tree
from curobo_task_constructor.graph.spec import StageSpec


#: The start-agreement number the whole pipeline shares: 0.01 rad. Kept here
#: as the reference value because it is ALSO what the controller must enforce
#: before it accepts a replayed trajectory (``allowed_start_tolerance`` in
#: iki_kortex_moveit_config/config/moveit_controllers.yaml). The cuRobo
#: executor no longer compares endpoints itself — MTC ``Task::execute`` replays
#: the stored ``RobotTrajectory`` verbatim and lets the controller's start
#: tolerance reject a replay whose first point is not where the arm is. This
#: constant documents that shared number.

from curobo_task_constructor.executor_io import *

class _ExecutorSerializationMixin:
    @staticmethod
    def _planner_id(sol: Solution) -> str:
        req = getattr(sol, "plan_request", None)
        planner = getattr(req, "planner", None)
        return str(planner) if planner is not None else ""

    # ------------------------------------------------------------------
    # Solution serialization (MTC Solution.toMsg / constructMotionPlan)
    # ------------------------------------------------------------------
    def solution_to_dict(self, sol: Solution) -> dict:
        """A solution as ROS-free dicts with MTC ``Solution`` field names.

        ``sub_solution`` mirrors the solution tree (one entry per node, with
        child solution ids); ``sub_trajectories`` carries one entry per
        MOTION leaf in chain order, each with the joint waypoints, the scene
        ops the executor applies before driving it, and the ``replay_goal``
        (the segment's planning goal) the execute server needs to replay the
        validated plan from the curobo_server trajectory cache instead of
        re-solving. ``start_scene`` is the task's base-scene objects.
        """
        sub_solutions = []

        def walk(s: Solution) -> None:
            sub_solutions.append({
                "info": self._solution_info_dict(s),
                "sub_solution_id": [
                    self.register_solution(c) for c in (s.children or [])],
            })
            for ch in (s.children or []):
                walk(ch)

        walk(sol)
        return {
            "task_id": self.task_id,
            "start_scene": [
                _object_spec_to_dict(spec)
                for spec in self.base_scene.objects_added.values()
            ],
            "sub_solution": sub_solutions,
            "sub_trajectories": self._chain_trajectory_dicts(sol),
        }

    def _chain_trajectory_dicts(self, sol: Solution) -> list:
        """One entry per motion leaf, in chain order, with scene ops folded
        in chain position.

        Mutation leaves (scene-only stages) carry no motion, so their ops
        flush into the nearest motion segment: preceding mutations apply
        BEFORE the segment drives (``before``), trailing mutations (e.g. a
        release task's detach) apply AFTER it (``after``). A scene-only
        chain yields one entry with no replay goal that only applies ops.
        """
        entries = []
        pending = []
        for leaf in self.flatten_leaves(sol):
            ops = list(getattr(leaf, "scene_ops", None) or [])
            if getattr(leaf, "gripper_command", None):
                entry = self._leaf_gripper_dict(leaf)
                entry["scene_diff"] = {
                    "before": _scene_ops_to_dict(pending + ops),
                    "after": _scene_ops_to_dict([]),
                }
                pending = []
                entries.append(entry)
                continue
            if getattr(leaf, "plan_request", None) is None:
                pending.extend(ops)
                continue
            entry = self._leaf_trajectory_dict(leaf)
            entry["scene_diff"] = {
                "before": _scene_ops_to_dict(pending + ops),
                "after": _scene_ops_to_dict([]),
            }
            pending = []
            entries.append(entry)
        if pending:
            if entries:
                entries[-1]["scene_diff"]["after"] = _scene_ops_to_dict(
                    pending)
            else:
                entries.append({
                    # No motion leaf: ids stay 0 ("no solution"), never -1 —
                    # uint32 fields reject negatives in C serialization.
                    "info": {"id": 0, "cost": 0.0, "comment": "",
                             "stage_id": 0, "planner_id": ""},
                    "execution_info": {"controller_names": []},
                    "trajectory": {"joint_names": [], "points": []},
                    "scene_diff": {"before": _scene_ops_to_dict(pending),
                                   "after": _scene_ops_to_dict([])},
                    "replay_goal": None,
                })
        return entries

    def _solution_info_dict(self, sol: Solution) -> dict:
        """MTC SolutionInfo fields for one solution.

        ``id`` is the GLOBAL solution id (``register_solution``), exactly the
        number ``StageStatistics.solved``/``failed`` and ``GetSolution`` use —
        never the per-stage ``solution_id``, which restarts at 0 in every
        stage and would make a client join the wrong solution to a stage.
        """
        stage = getattr(sol, "stage", None)
        return {
            "id": self.register_solution(sol),
            "cost": float(getattr(sol, "cost", float("inf"))),
            "comment": str(getattr(sol, "comment", "") or ""),
            "stage_id": int(getattr(stage, "stage_id", 0) or 0),
            "planner_id": self._planner_id(sol),
        }

    def _leaf_gripper_dict(self, leaf: Solution) -> dict:
        """One gripper-command segment (no motion). ``replay_goal`` carries the
        command so the remote execute server can drive it without re-planning."""
        cmd = dict(getattr(leaf, "gripper_command", None) or {})
        controller = str(cmd.get("controller", "") or "")
        return {
            "info": self._solution_info_dict(leaf),
            "execution_info": {
                "controller_names": [controller] if controller else []},
            "trajectory": {"joint_names": [], "points": [], "velocities": []},
            "scene_diff": {"before": _scene_ops_to_dict([]),
                           "after": _scene_ops_to_dict([])},
            "replay_goal": {"gripper_command": cmd},
        }

    def _leaf_trajectory_dict(self, leaf: Solution) -> dict:
        traj = list(getattr(leaf, "trajectory", None) or [])
        names = list(getattr(traj[0], "name", []) or []) if traj else []
        points, velocities = [], []
        for wp in traj:
            aligned = _align_waypoint_positions(wp, names) if names else None
            if aligned is None:
                raw = [float(v) for v in (getattr(wp, "position", []) or [])]
                aligned = raw if len(raw) == len(names) else None
            if aligned is None:
                continue
            points.append([float(v) for v in aligned])
            vel = getattr(wp, "velocity", None) or []
            velocities.append([float(v) for v in vel])
        return {
            "info": self._solution_info_dict(leaf),
            "execution_info": {"controller_names": []},
            # MTC parity: the sub-trajectory carries the waypoints themselves so
            # execute can replay them verbatim. Velocities ride along when the
            # solver produced them (the wire JointState already carries them);
            # the open-loop streamer tapers/clamps them, so an empty list is a
            # benign "derive from positions".
            "trajectory": {
                "joint_names": [str(n) for n in names],
                "points": points,
                "velocities": velocities,
            },
            # Filled by _chain_trajectory_dicts (chain-position folding).
            "scene_diff": {"before": _scene_ops_to_dict([]),
                           "after": _scene_ops_to_dict([])},
            "replay_goal": _plan_request_to_dict(
                getattr(leaf, "plan_request", None)),
        }

    def execute_dict(self, d: dict, progress_callback=None,
                     cancel_event=None) -> tuple:
        """Drive a deserialized solution (see ``solution_to_dict``).

        Rebuilds each motion segment's PlanRequest + scene ops and drives
        them exactly like :meth:`execute`: MTC ``Task::execute`` parity, each
        segment is replayed from the validated trajectory the solution carries
        (never re-solved). Returns (results, failed_stage_name);
        ``failed_stage_name`` is "" on success and names the leaf stage (MTC
        reports per-sub-trajectory progress through the action feedback
        instead).
        """
        self._applied_ops = []
        # MTC maps sub-trajectory ids back to stage names through the task;
        # same here: names come from this executor's own stage tree.
        names = {stg.stage_id: stg.name
                 for stg in self.root.subtree_stages()}

        def _progress(index, total, info):
            if progress_callback is not None:
                stage_name = names.get(int((info or {}).get("stage_id", -1)
                                           or -1), f"segment_{index}")
                progress_callback(stage_name)

        results, failed_index, cancelled = drive_solution_dict(
            self.robot, d, self._applied_ops,
            progress_callback=_progress, cancel_event=cancel_event)
        if cancelled:
            return results, "<cancelled>"
        if failed_index is None:
            return results, ""
        segments = list((d or {}).get("sub_trajectories", None) or [])
        info = (segments[failed_index].get("info", {})
                if 0 <= failed_index < len(segments) else {})
        return results, names.get(int(info.get("stage_id", -1) or -1), "")


__all__ = ["_ExecutorSerializationMixin"]
