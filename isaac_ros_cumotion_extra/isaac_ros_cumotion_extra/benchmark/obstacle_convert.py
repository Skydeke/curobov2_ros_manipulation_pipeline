# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert robometrics/cuRobo obstacle dicts to ``AddObject`` request payloads.

Deliberately pure dict -> dict (no ROS message imports): the conversion is
unit-testable in any Python environment, and ``ros_runner`` maps the payloads
onto ``AddObject.Request`` messages at call time.

Payload schema::

    {
        'name': str,               # unique across the world
        'type': int,               # AddObject type constant (see below)
        'pose': [x, y, z, w, x, y, z],
        'dims': [x, y, z],         # interpreted per type (see ObstacleManager)
        'color': [r, g, b, a],     # 0..1 floats; default opaque red
        'mesh_file_path': str,     # MESH only; '' when inline geometry is used
        'vertices': [...],         # MESH inline vertices (unused for primitives)
        'triangles': [...],        # MESH inline flat index buffer
    }
"""

# Standard Library
from typing import Any, Dict, List

# Mirrors AddObject.srv type constants (kept in sync — they are int8 service
# constants, so we re-declare them here to keep this module ROS-free).
OBJECT_CUBOID = 0
OBJECT_SPHERE = 1
OBJECT_CAPSULE = 2
OBJECT_CYLINDER = 3
OBJECT_MESH = 4

# Supported obstacle buckets in a cuRobo/robometrics scene dict.
_SUPPORTED_BUCKETS = frozenset(("cuboid", "sphere", "capsule", "cylinder", "mesh"))

# Deterministic emission order for the payload list.
_BUCKET_ORDER = ("cuboid", "sphere", "capsule", "cylinder", "mesh")

_DEFAULT_COLOR = [1.0, 0.0, 0.0, 1.0]  # opaque red


def _pose(data: Dict[str, Any]) -> List[float]:
    """cuRobo pose list [x, y, z, w, x, y, z] -> floats."""
    return [float(v) for v in data["pose"]]


def _color(data: Dict[str, Any]) -> List[float]:
    color = data.get("color")
    if not color:
        return list(_DEFAULT_COLOR)
    return [float(v) for v in color]


def _base_payload(name: str, obj_type: int, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "type": obj_type,
        "pose": _pose(data),
        "dims": [1.0, 1.0, 1.0],  # replaced per type below; all axes positive
        "color": _color(data),
        "mesh_file_path": "",
        "vertices": [],
        "triangles": [],
    }


def _single_request(obj_type: str, obj_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one obstacle entry to an AddObject request payload."""
    # Names must be unique across the whole world, so bucket-prefix them (the
    # server rejects duplicate names; robometrics names can collide across
    # type buckets within a scene).
    name = f"{obj_type}_{obj_name}"

    if obj_type == "cuboid":
        payload = _base_payload(name, OBJECT_CUBOID, data)
        dims = data["dims"]
        payload["dims"] = [float(d) for d in dims]
        return payload

    if obj_type == "sphere":
        payload = _base_payload(name, OBJECT_SPHERE, data)
        radius = float(data["radius"])
        payload["dims"] = [radius, radius, radius]
        return payload

    if obj_type in ("capsule", "cylinder"):
        payload = _base_payload(
            name, OBJECT_CAPSULE if obj_type == "capsule" else OBJECT_CYLINDER, data
        )
        radius = float(data["radius"])
        height = float(data["height"])
        # dims: [radius, height, _] — z must stay > 0 (server validates).
        payload["dims"] = [radius, height, 1.0]
        return payload

    if obj_type == "mesh":
        payload = _base_payload(name, OBJECT_MESH, data)
        scale = data.get("scale", [1.0, 1.0, 1.0])
        payload["dims"] = [float(v) for v in scale]
        file_path = data.get("file_path") or data.get("mesh_file_path") or ""
        payload["mesh_file_path"] = str(file_path)
        vertices = data.get("vertices")
        triangles = data.get("triangles")
        if vertices and triangles:
            payload["vertices"] = [
                [float(v[0]), float(v[1]), float(v[2])] for v in vertices
            ]
            payload["triangles"] = [int(t) for t in triangles]
        elif not file_path:
            raise ValueError(
                f"MESH obstacle {obj_name!r} carries neither inline vertex/"
                f"triangle data nor a file path"
            )
        return payload

    raise ValueError(f"Unsupported obstacle bucket {obj_type!r}")


def obstacles_dict_to_add_requests(obstacles: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a cuRobo-style obstacle dict to a list of AddObject payloads.

    Args:
        obstacles: ``{bucket: {name: params, ...}}`` exactly as passed to
            ``SceneCfg.create`` by the upstream benchmark (also what
            robometrics problems carry in ``problem['obstacles']``).

    Returns:
        List of AddObject request payload dicts (see module docstring).

    Raises:
        ValueError: on unknown obstacle buckets or MESH entries without
            geometry.
    """
    for bucket in obstacles:
        if bucket not in _SUPPORTED_BUCKETS:
            raise ValueError(
                f"Unsupported obstacle bucket {bucket!r} in scene; "
                f"supported: {sorted(_SUPPORTED_BUCKETS)}. "
                f"Use the curobo_core benchmark world conversion for "
                f"voxel/other types."
            )

    requests: List[Dict[str, Any]] = []
    for bucket in _BUCKET_ORDER:
        bucket_objs = obstacles.get(bucket)
        if not bucket_objs:
            continue
        for obj_name, data in bucket_objs.items():
            requests.append(_single_request(bucket, obj_name, data))
    return requests