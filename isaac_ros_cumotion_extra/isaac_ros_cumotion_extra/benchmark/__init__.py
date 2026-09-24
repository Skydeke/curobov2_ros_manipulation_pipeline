# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Planner parity benchmark: ``curobo_core`` native vs the ROS-wrapped planner.

Two legs replay the *identical* robometrics problems (the same datasets the
``curobo_core`` ``motion_plan_benchmark`` uses -- ``demo``,
``motion_benchmaker``, ``mpinets``):

- ``core_runner`` drives ``curobo``'s ``MotionPlanner`` directly, configured
  exactly like the ROS server's internal planner (same robot config YAML, same
  solver knobs, same per-problem world conversion).
- ``ros_runner`` replays the problems through a running server
  (``/unified_planner/generate_trajectory`` with obstacles pushed via
  ``/unified_planner/add_object`` / ``remove_all_objects``).

``compare`` reports per-problem parity (success, path length, motion time,
waypoint count) so wrapping the planner in ROS can be shown to preserve
planning outcomes.
"""