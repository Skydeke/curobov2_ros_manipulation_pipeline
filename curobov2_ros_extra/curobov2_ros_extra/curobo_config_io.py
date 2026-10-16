"""Shared cuRobo config IO: load, preserve custom fields, save one file.

Used by :mod:`curobo_config` (CLI) and :mod:`morphit_bubblify` (interactive).
All YAML imports are lazy so this module imports without ROS/cuRobo.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Original kinematics keys the builder rebuilds; every other key
#: (lock_joints, cspace is handled by the builder, extra_links, ...) is
#: carried over on save.
_REBUILT_KEYS = frozenset({
    "collision_spheres",
    "collision_link_names",
    "mesh_link_names",
    "self_collision_ignore",
    "cspace",
})


def load_original_kinematics(config_path: str) -> Dict[str, Any]:
    """Load the raw kinematics dict of an existing cuRobo yml config."""
    import yaml

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    if "robot_cfg" in data:
        data = data["robot_cfg"]
    return dict(data["kinematics"])


def apply_preserved_fields(config: Any, original: Dict[str, Any]) -> None:
    """Copy custom fields from the original yml onto a rebuilt config."""
    for key, value in original.items():
        if key not in _REBUILT_KEYS:
            setattr(config, key, value)


def build_and_save(builder: Any, output: str,
                   original: Optional[Dict[str, Any]]) -> Any:
    """Build, restore original custom fields, and save to one file."""
    config = builder.build()
    if original is not None:
        apply_preserved_fields(config, original)
    builder.save(config, output)
    return config


def pull_store_spheres_from_builder(builder: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Snapshot builder spheres as plain comparable dicts."""
    snapshot: Dict[str, List[Dict[str, Any]]] = {}
    for link, entries in (builder.collision_spheres or {}).items():
        snapshot[link] = [{"center": [float(v) for v in e["center"]],
                           "radius": float(e["radius"])} for e in entries]
    return snapshot
