# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M1 skeleton tests for the ROS-free curobo_python_runtime module.

These pin the handle registry and lane binding only; the cuRobo-bearing entry
points are covered by the recorded request/response replays from M2 on.
"""

import curobo_python_runtime


def test_init_returns_unique_handles_and_shutdown_removes_them():
    h1 = curobo_python_runtime.init({"a": 1})
    h2 = curobo_python_runtime.init({"b": 2})
    assert h1 != h2
    assert h1 in curobo_python_runtime.registered_handles()
    assert h2 in curobo_python_runtime.registered_handles()
    curobo_python_runtime.shutdown(h1)
    curobo_python_runtime.shutdown(h2)
    assert h1 not in curobo_python_runtime.registered_handles()
    assert h2 not in curobo_python_runtime.registered_handles()


def test_bind_lane_registers_the_lane():
    curobo_python_runtime.bind_lane("plan")
    assert "plan" in curobo_python_runtime.lane_names()
