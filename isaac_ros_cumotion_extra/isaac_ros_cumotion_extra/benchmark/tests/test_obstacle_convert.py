# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the obstacle -> AddObject payload conversion (pure Python)."""

import pytest

from isaac_ros_cumotion_extra.benchmark.obstacle_convert import (
    OBJECT_CAPSULE,
    OBJECT_CUBOID,
    OBJECT_CYLINDER,
    OBJECT_MESH,
    OBJECT_SPHERE,
    obstacles_dict_to_add_requests,
)

_POSE = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]


def _cuboid(name='box', dims=(1.0, 2.0, 3.0), **extra):
    return {'cuboid': {name: {'pose': _POSE, 'dims': dims, **extra}}}


class TestCuboid:
    def test_basic_cuboid(self):
        reqs = obstacles_dict_to_add_requests(_cuboid())
        assert len(reqs) == 1
        req = reqs[0]
        assert req['type'] == OBJECT_CUBOID
        assert req['name'] == 'cuboid_box'
        assert req['pose'] == _POSE
        assert req['dims'] == [1.0, 2.0, 3.0]

    def test_default_color_is_opaque_red(self):
        req = obstacles_dict_to_add_requests(_cuboid())[0]
        assert req['color'] == [1.0, 0.0, 0.0, 1.0]

    def test_explicit_color(self):
        req = obstacles_dict_to_add_requests(
            _cuboid(color=[0.0, 1.0, 0.0, 0.5])
        )[0]
        assert req['color'] == [0.0, 1.0, 0.0, 0.5]


class TestPrimitives:
    def test_sphere_radius_replicated(self):
        reqs = obstacles_dict_to_add_requests(
            {'sphere': {'ball': {'pose': _POSE, 'radius': 0.4}}}
        )
        req = reqs[0]
        assert req['type'] == OBJECT_SPHERE
        assert req['dims'] == [0.4, 0.4, 0.4]  # all axes positive (server check)

    def test_cylinder_radius_height(self):
        req = obstacles_dict_to_add_requests(
            {'cylinder': {'pipe': {'pose': _POSE, 'radius': 0.2, 'height': 1.5}}}
        )[0]
        assert req['type'] == OBJECT_CYLINDER
        assert req['dims'] == [0.2, 1.5, 1.0]

    def test_capsule_radius_height(self):
        req = obstacles_dict_to_add_requests(
            {'capsule': {'pill': {'pose': _POSE, 'radius': 0.3, 'height': 2.0}}}
        )[0]
        assert req['type'] == OBJECT_CAPSULE
        assert req['dims'] == [0.3, 2.0, 1.0]

    def test_names_are_bucket_prefixed_for_uniqueness(self):
        obstacles = {
            'cuboid': {'a': {'pose': _POSE, 'dims': [1, 1, 1]}},
            'sphere': {'a': {'pose': _POSE, 'radius': 0.5}},
        }
        names = [r['name'] for r in obstacles_dict_to_add_requests(obstacles)]
        assert names == ['cuboid_a', 'sphere_a']


class TestMesh:
    def test_inline_geometry(self):
        req = obstacles_dict_to_add_requests(
            {'mesh': {
                'tri': {
                    'pose': _POSE,
                    'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    'triangles': [0, 1, 2],
                }
            }}
        )[0]
        assert req['type'] == OBJECT_MESH
        assert req['mesh_file_path'] == ''
        assert req['vertices'] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        assert req['triangles'] == [0, 1, 2]

    def test_file_path_with_scale(self):
        req = obstacles_dict_to_add_requests(
            {'mesh': {
                'm': {'pose': _POSE, 'file_path': '/tmp/thing.obj', 'scale': [2, 2, 2]}
            }}
        )[0]
        assert req['mesh_file_path'] == '/tmp/thing.obj'
        assert req['dims'] == [2.0, 2.0, 2.0]

    def test_mesh_without_geometry_raises(self):
        with pytest.raises(ValueError):
            obstacles_dict_to_add_requests(
                {'mesh': {'m': {'pose': _POSE}}}
            )


class TestErrors:
    def test_unknown_bucket_raises(self):
        with pytest.raises(ValueError):
            obstacles_dict_to_add_requests({'blox': {'b': {'pose': _POSE}}})

    def test_empty_scene(self):
        assert obstacles_dict_to_add_requests({}) == []