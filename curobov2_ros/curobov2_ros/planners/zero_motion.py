# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server-side guard for degenerate joint-space requests.

cuRobo's ``plan_cspace`` is NOT a no-op for a goal that already equals the
start state. Its optimizer re-captures CUDA graphs mid-plan (which is why
``UnifiedPlannerNode._plan_lock`` exists at all — see the comment at
``unified_planner_node.py`` around the ``with self._plan_lock()`` block), and
a ``start == goal`` solve is exactly where that surfaced in the field as::

    CUDA error: an illegal memory access was encountered
      File ".../_src/util/cuda_graph_util.py", in __call__
        self._graph.replay()

reached from the pick task's first stage — an already-open gripper whose
goal equals the start, solved ``planning_attempts`` times.

The guard lives HERE, on the server, deliberately: it protects every client
(MoveIt, the task constructor, a raw ``ros2 service call``) from the same
degenerate request, and it keeps a solver invariant out of client code.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

#: Tolerance for "already there". Positions are compared in the model's
#: active-DOF order; anything above this is a real (if tiny) move that the
#: solver must still handle.
ZERO_DELTA_TOL = 1e-9


def is_zero_delta(start_positions: Any, goal_positions: Any,
                  tol: float = ZERO_DELTA_TOL) -> bool:
    """True when the goal already holds at the start state.

    Both lists are FULL active-cspace length by the time they reach here (the
    caller pads a short goal from the start state, MTC-style), so this is a
    plain element-wise compare. Length disagreement is NOT a zero delta: a
    genuine partial goal is a real request.
    """
    start = list(start_positions or [])
    goal = list(goal_positions or [])
    if len(start) != len(goal) or not start:
        return False
    return all(abs(float(a) - float(b)) <= tol for a, b in zip(start, goal))


class ZeroMotionResult:
    """A solver-shaped result for "no motion needed".

    Implements exactly the surface ``SinglePlanner._finalize_plan_result``
    reads: ``success`` (tensor, so ``hasattr(..., "item")`` takes the value
    branch) and ``get_interpolated_plan()``. Every other attribute the
    finalizer probes is left absent so its ``getattr`` defaults apply — that
    keeps this class as small as the contract (SRP/KISS).
    """

    def __init__(self, trajectory, device, dtype):
        self.success = torch.ones(1, device=device, dtype=torch.bool)
        self._trajectory = trajectory

    def get_interpolated_plan(self):
        """The single-waypoint trajectory: stay where you are."""
        return self._trajectory

    @property
    def status(self) -> str:
        return "no motion needed (goal == start)"
