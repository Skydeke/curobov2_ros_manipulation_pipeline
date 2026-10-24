# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the perception voxel layer surviving scene edits. These pin the safety property that the solvers are never silently blinded to
camera-observed voxels:

* ``scene_set(clear_first=True)`` (what a ``set_objects`` / clear - e.g. the
  MoveIt side's ``syncPlanningScene`` - calls) must carry the perception voxel
  layer over onto the rebuilt scene. The voxels are not an analytic object; they
  are installed on the scene by the ESDF refresh."""

import sys
import types

import pytest

import curobo_bridge


class _FakeScene:
    """Minimal stand-in for ``curobo.scene.Scene``."""

    def __init__(self, cuboid=None, mesh=None, **_ignored):
        self.cuboid = cuboid
        self.mesh = mesh
        self.sphere = None
        self.capsule = None
        self.cylinder = None
        self.voxel = None


@pytest.fixture(name="fake_scene")
def fixture_fake_scene(monkeypatch):
    module = types.ModuleType("curobo.scene")
    module.Scene = _FakeScene
    package = sys.modules.get("curobo") or types.ModuleType("curobo")
    package.scene = module
    monkeypatch.setitem(sys.modules, "curobo", package)
    monkeypatch.setitem(sys.modules, "curobo.scene", module)
    return module


def test_clear_first_keeps_the_perception_voxel_layer(fake_scene):
    handle = curobo_bridge.init({})
    try:
        marker = object()
        seeded = _FakeScene()
        seeded.voxel = [marker]
        curobo_bridge._REGISTRY[handle]["scene"] = seeded

        # MoveIt syncPlanningScene with no world objects, then a clear.
        curobo_bridge.scene_set(handle, "[]", True)

        scene = curobo_bridge._REGISTRY[handle]["scene"]
        assert scene is not seeded        # the analytic scene was rebuilt
        assert scene.voxel == [marker]    # ...but the camera voxels survived
    finally:
        curobo_bridge.shutdown(handle)


def test_clear_first_without_a_previous_scene_is_safe(fake_scene):
    handle = curobo_bridge.init({})
    try:
        curobo_bridge.scene_set(handle, "[]", True)
        assert curobo_bridge._REGISTRY[handle]["scene"].voxel is None
    finally:
        curobo_bridge.shutdown(handle)


class _Pos:
    def __init__(self):
        self.deltas = []

    def __getitem__(self, _index):
        return 0.0

    def __setitem__(self, _index, value):
        self.deltas.append(value)


class _State:
    def __init__(self):
        self.position = _Pos()

    def clone(self):
        copy = _State()
        copy.position.deltas = list(self.position.deltas)
        return copy

    def unsqueeze(self, _dim):
        return self


class _Planner:
    """Stand-in exposing just what _warmup_cspace touches."""

    def __init__(self, reject_graph_attempt=False):
        self.default_joint_state = _State()
        self.calls = []
        self.resets = 0
        self._reject_graph_attempt = reject_graph_attempt

    def plan_cspace(self, _goal, _current, max_attempts=5, enable_graph_attempt=None):
        if self._reject_graph_attempt and enable_graph_attempt is not None:
            raise TypeError("plan_cspace() got an unexpected keyword argument")
        self.calls.append((max_attempts, enable_graph_attempt))

    def reset_seed(self):
        self.resets += 1


def test_cspace_warmup_records_the_graph_seed_path():
    planner = _Planner()
    curobo_bridge._warmup_cspace(planner, iterations=3)

    # Every iteration exercises the graph-planner seed attempt too, so that path's
    # graph is captured under the exclusive barrier, not mid-plan.
    assert planner.calls == [(2, 1), (2, 1), (2, 1)]
    assert planner.resets == 3


def test_cspace_warmup_falls_back_when_graph_attempt_is_unsupported():
    planner = _Planner(reject_graph_attempt=True)
    curobo_bridge._warmup_cspace(planner, iterations=2)

    assert planner.calls == [(1, None), (1, None)]


def test_cspace_warmup_mirrors_the_requested_attempt_budget():
    """A narrow runtime budget must not be warmed as a wide one.

    The re-capture runs on every perception world push, so warming paths the plan
    requests never replay (an extra attempt, the PRM seed search) is pure GPU
    time spent in the middle of the perception lane."""
    planner = _Planner()
    curobo_bridge._warmup_cspace(
        planner, iterations=1, max_attempts=1, enable_graph_attempt=1)

    assert planner.calls == [(1, 1)]


class _ToolPoses:
    def as_goal(self):
        return "goal"


class _Kin:
    def __init__(self):
        self.tool_poses = _ToolPoses()


class _TrajoptSolver:
    def __init__(self):
        self.resets = 0

    def reset_cuda_graph(self):
        self.resets += 1


class _ModePlanner:
    """Planner stand-in for prepare_solve_mode: records solves + graph resets."""

    def __init__(self):
        self.default_joint_state = _State()
        self.trajopt_solver = _TrajoptSolver()
        self.calls = []

    def compute_kinematics(self, _goal):
        return _Kin()

    def plan_pose(self, _poses, _current, max_attempts=1):
        self.calls.append(("pose", max_attempts))

    def plan_cspace(self, _goal, _current, max_attempts=5, enable_graph_attempt=None):
        self.calls.append(("cspace", max_attempts, enable_graph_attempt))

    def reset_seed(self):
        pass


def test_prepare_solve_mode_resets_and_recaptures_on_change():
    handle = curobo_bridge.init({})
    try:
        planner = _ModePlanner()
        curobo_bridge._REGISTRY[handle]["motion_planner"] = planner

        curobo_bridge.prepare_solve_mode(handle, "cspace")
        assert planner.trajopt_solver.resets == 1
        assert ("cspace", 2, 2) in planner.calls
        assert curobo_bridge._REGISTRY[handle]["_solve_mode"] == "cspace"

        # Same mode again: no reset, no extra capture.
        calls_before = list(planner.calls)
        curobo_bridge.prepare_solve_mode(handle, "cspace")
        assert planner.trajopt_solver.resets == 1
        assert planner.calls == calls_before

        # Switching back to pose resets and captures the pose graphs.
        curobo_bridge.prepare_solve_mode(handle, "pose")
        assert planner.trajopt_solver.resets == 2
        assert ("pose", 1) in planner.calls
        assert curobo_bridge._REGISTRY[handle]["_solve_mode"] == "pose"
    finally:
        curobo_bridge.shutdown(handle)


def test_prepare_solve_mode_without_a_planner_is_a_noop():
    handle = curobo_bridge.init({})
    try:
        curobo_bridge.prepare_solve_mode(handle, "cspace")
        assert "_solve_mode" not in curobo_bridge._REGISTRY[handle]
    finally:
        curobo_bridge.shutdown(handle)


class _GraphPlanner:
    """PRM stand-in: records the explicit re-warmup prepare_solve_mode issues."""

    def __init__(self):
        self.warmups = 0

    def warmup(self, num_warmup_iterations=1, max_batch_size=1):
        self.warmups += 1


def _mode_planner_with_graph():
    planner = _ModePlanner()
    planner.graph_planner = _GraphPlanner()
    return planner


def test_prepare_solve_mode_uses_the_configured_attempt_budget():
    """The re-capture must mirror max_attempts/enable_graph_attempt.

    max_attempts=1 / enable_graph_attempt=1 (the node's configuration) cannot
    reach the PRM seed path, so its warmup is skipped and the cspace capture runs
    a single direct attempt instead of the reference two-plus-graph one."""
    handle = curobo_bridge.init({})
    try:
        planner = _mode_planner_with_graph()
        entry = curobo_bridge._REGISTRY[handle]
        entry["motion_planner"] = planner
        entry["attempt_cfg"] = {"max_attempts": 1, "enable_graph_attempt": 1}

        curobo_bridge.prepare_solve_mode(handle, "cspace")

        assert ("cspace", 1, 1) in planner.calls
        assert planner.graph_planner.warmups == 0
        assert entry["_solve_mode"] == "cspace"
    finally:
        curobo_bridge.shutdown(handle)


def test_prepare_solve_mode_warms_a_direct_solve_and_the_graph_separately():
    """enable_graph_attempt < max_attempts keeps the PRM warmup.

    With max_attempts=2 / enable_graph_attempt=1 the second attempt seeds from
    the graph planner, so its graph has to be captured under this barrier. The
    trajopt warmup itself stays a DIRECT solve: the PRM search is what the
    separate graph-planner warmup below is for."""
    handle = curobo_bridge.init({})
    try:
        planner = _mode_planner_with_graph()
        entry = curobo_bridge._REGISTRY[handle]
        entry["motion_planner"] = planner
        entry["attempt_cfg"] = {"max_attempts": 2, "enable_graph_attempt": 1}

        curobo_bridge.prepare_solve_mode(handle, "cspace")

        assert ("cspace", 2, 2) in planner.calls
        assert planner.graph_planner.warmups == 1
    finally:
        curobo_bridge.shutdown(handle)


def test_world_push_keeps_the_graphs_when_the_buffers_did_not_move():
    """A value-only world update must NOT force a CUDA-graph re-capture.

    ``load_collision_model`` writes in place into the build-time cache tensors, so
    the graphs baked against those buffers still read the freshly pushed world:
    the plan sees current perception for free. Only a buffer that moved (a
    reallocated tensor, a different cache shape) may force a re-capture.
    """
    handle = curobo_bridge.init({})
    try:
        import torch

        entry = curobo_bridge._REGISTRY[handle]
        entry["scene"] = _FakeScene()

        class _Store:
            def __init__(self):
                self.max_n = 2
                self.dims = torch.zeros((2, 4))
                self.pose = torch.zeros((2, 8))
                self.enable = torch.zeros(2, dtype=torch.uint8)

        class _Data:
            def __init__(self):
                self.cuboids = _Store()
                self.meshes = None
                self.voxels = _Store()

        class _Checker:
            def __init__(self, data):
                self.data = data

        class _Planner:
            def __init__(self):
                self.data = _Data()
                self.scene_collision_checker = _Checker(self.data)

            def update_world(self, _scene):
                # In-place value rewrite, exactly like load_collision_model.
                self.data.cuboids.enable[0] = 1
                self.data.voxels.dims[0, 0] = 3.5

        planner = _Planner()
        entry["motion_planner"] = planner
        entry["_solve_mode"] = "cspace"

        # First push records the buffer signature (the mode is already None after
        # a build, so this is a no-op there); the next one must not drop it.
        assert curobo_bridge.world_push(handle) is True
        entry["_solve_mode"] = "cspace"
        assert curobo_bridge.world_push(handle) is True
        assert entry["_solve_mode"] == "cspace", "graphs were dropped for a value-only push"

        # Same push again: still no re-capture.
        assert curobo_bridge.world_push(handle) is True
        assert entry["_solve_mode"] == "cspace"

        # A moved buffer (what a reallocated cache looks like) must drop the mode.
        planner.data.cuboids.dims = torch.zeros((4, 4))
        assert curobo_bridge.world_push(handle) is True
        assert entry["_solve_mode"] is None
    finally:
        curobo_bridge.shutdown(handle)


def test_world_push_reports_a_failed_solver_update(fake_scene, capsys):
    handle = curobo_bridge.init({})
    try:
        entry = curobo_bridge._REGISTRY[handle]
        entry["scene"] = _FakeScene()

        class _Boom:
            def update_world(self, _scene):
                raise RuntimeError("voxel cache not initialized")

        entry["motion_planner"] = _Boom()

        assert curobo_bridge.world_push(handle) is False
        assert "world_push FAILED" in capsys.readouterr().out
    finally:
        curobo_bridge.shutdown(handle)
