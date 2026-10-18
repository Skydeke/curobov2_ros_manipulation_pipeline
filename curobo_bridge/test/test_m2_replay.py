# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M2 parity: replay the recorded kinematics requests through curobo_bridge.

Uses the recordings captured from the (unmodified) Python server
(`recordings/m2_kinematics_franka/`) as the expectation (design section 11.5).
Needs a GPU and the recordings; skips otherwise.

Acceptance asserted here (design section 10, M2):
  * FK within 1e-5 of the recorded response;
  * IK validity flags identical.
"""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import curobo_bridge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RECORDINGS = os.path.join(REPO, "recordings", "m2_kinematics_franka")
SCENE_RECORDINGS = os.path.join(REPO, "recordings", "m2_scene_collision_franka")
FRANKA_CONFIG = os.path.join(REPO, "curobov2_ros", "config", "franka.curobo.yml")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isdir(RECORDINGS),
    reason="needs a GPU and the M2 recordings",
)


def _load(step_file, root=None):
    with open(os.path.join(root or RECORDINGS, step_file)) as fh:
        return json.load(fh)


def _bridge():
    handle = curobo_bridge.init({})
    curobo_bridge.robot_model_build(handle, json.dumps({"robot_config_file": FRANKA_CONFIG}))
    return handle


def test_fk_matches_recorded_response_within_1e5():
    record = _load("004_05_FK_valid_joints.json")
    q = record["request"]["joint_states"][0]["position"]
    expected = record["response"]["poses"][0]

    handle = _bridge()
    poses = curobo_bridge.fk(handle, np.asarray([q], dtype=np.float32))
    assert poses.shape == (1, 1, 7)
    pose = poses[0, 0]
    # bridge order [x,y,z,qw,qx,qy,qz]; recorded geometry_msgs order (x,y,z,w).
    expected_row = np.array([
        expected["position"]["x"], expected["position"]["y"], expected["position"]["z"],
        expected["orientation"]["w"], expected["orientation"]["x"],
        expected["orientation"]["y"], expected["orientation"]["z"],
    ],
        dtype=np.float32)
    np.testing.assert_allclose(pose, expected_row, atol=1e-5)
    curobo_bridge.shutdown(handle)


def test_ik_validity_flags_match_recorded_responses():
    valid_record = _load("002_03_IK_valid_pose.json")
    invalid_record = _load("003_04_IK_invalid_pose.json")

    def _pose(msg):
        return [msg["position"]["x"], msg["position"]["y"], msg["position"]["z"],
                msg["orientation"]["w"], msg["orientation"]["x"],
                msg["orientation"]["y"], msg["orientation"]["z"]]

    handle = _bridge()

    valid_pose = _pose(valid_record["request"]["pose"])
    outcome = curobo_bridge.ik(handle, np.asarray([valid_pose], dtype=np.float32))
    assert outcome["valid"][0] == int(valid_record["response"]["joint_states_valid"]["data"])

    invalid_pose = _pose(invalid_record["request"]["pose"])
    outcome = curobo_bridge.ik(handle, np.asarray([invalid_pose], dtype=np.float32))
    assert outcome["valid"][0] == int(invalid_record["response"]["joint_states_valid"]["data"])

    curobo_bridge.shutdown(handle)


def test_collision_distance_matches_recorded_within_1e4():
    record = _load("001_02_collision_distance_(no_scene).json", root=SCENE_RECORDINGS)
    expected = record["response"]
    assert record["request"] == {}   # the server evaluated at its live (home) pose

    handle = _bridge()
    # franka cspace is 7 joints; the value does not matter for an empty scene.
    distances = curobo_bridge.collision_distance(
        handle, np.zeros((1, 7), dtype=np.float32))

    assert len(distances) == int(expected["nb_sphere"])
    np.testing.assert_allclose(
        np.asarray(distances), np.asarray(expected["data"], dtype=np.float32), atol=1e-4)
    curobo_bridge.shutdown(handle)


def test_scene_listing_matches_recorded_conventions():
    """The scene-listing wire form the C++ ObstacleManager must reproduce.

    get_scene_objects reports a SPHERE as [radius, radius, radius] and the
    messages the C++ side emits were asserted against the recorded strings in
    curobo_core/test/test_obstacle_manager.cpp.
    """
    scene = _load("003_04_get_scene_objects.json", root=SCENE_RECORDINGS)["response"]
    assert scene["names"] == ["cube", "ball"]
    assert scene["types"] == [0, 1]                       # CUBOID, SPHERE
    assert scene["dimensions"][0]["x"] == pytest.approx(0.5)
    # sphere filler is the used value, never 0
    assert [scene["dimensions"][1][k] for k in ("x", "y", "z")] == pytest.approx([0.05] * 3)
    assert scene["attached"] == [False, False]
    assert scene["mesh_file_paths"] == ["", ""]

    assert _load("004_05_get_obstacles.json", root=SCENE_RECORDINGS)["response"]["message"] \
        == "cube\nball\n"
    assert _load("005_06_remove_all_objects.json", root=SCENE_RECORDINGS)["response"]["message"] \
        == "All 2 obstacles removed"


def test_joint_limits_match_recorded_joint_info():
    record = _load("000_01_get_joint_info_(empty_scene,_default_locks).json",
                   root=SCENE_RECORDINGS)
    expected = record["response"]

    handle = _bridge()
    limits = curobo_bridge.joint_limits(handle)

    assert limits["joint_names"] == expected["joint_names"]
    np.testing.assert_allclose(limits["positions"][0], expected["lower_limits"], atol=1e-5)
    np.testing.assert_allclose(limits["positions"][1], expected["upper_limits"], atol=1e-5)
    assert limits["locked_joint_names"] == expected["locked_joint_names"]
    np.testing.assert_allclose(
        limits["locked_positions"], expected["locked_positions"], atol=1e-9)
    curobo_bridge.shutdown(handle)
