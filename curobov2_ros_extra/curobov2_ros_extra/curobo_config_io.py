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


def enforce_lock_joints(config: Any, names) -> None:
    """Move ``names`` out of the planning cspace and lock them to ``null``.

    ``null`` is an unpinned lock: the joint is never planned, but its value is
    read from the live joint state. Cspace parallel arrays are filtered by the
    same indices so every list stays aligned with ``joint_names``.
    """
    names = [n for n in (names or []) if n]
    if not names:
        return
    cspace = getattr(config, "cspace", None)
    if cspace is not None:
        joint_names = list(getattr(cspace, "joint_names", []) or [])
        drop = {i for i, n in enumerate(joint_names) if n in names}
        if drop:
            for attr, value in list(vars(cspace).items()):
                if isinstance(value, list) and len(value) == len(joint_names):
                    setattr(cspace, attr,
                            [v for i, v in enumerate(value) if i not in drop])
    locks = dict(getattr(config, "lock_joints", None) or {})
    for n in names:
        locks[n] = None
    config.lock_joints = locks


def build_and_save(builder: Any, output: str,
                   original: Optional[Dict[str, Any]] = None,
                   lock_joints=()) -> Any:
    """Build, restore original custom fields, lock requested joints, save."""
    config = builder.build()
    if original is not None:
        apply_preserved_fields(config, original)
    enforce_lock_joints(config, lock_joints)
    builder.save(config, output)
    return config


def pull_store_spheres_from_builder(builder: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Snapshot builder spheres as plain comparable dicts."""
    snapshot: Dict[str, List[Dict[str, Any]]] = {}
    for link, entries in (builder.collision_spheres or {}).items():
        snapshot[link] = [{"center": [float(v) for v in e["center"]],
                           "radius": float(e["radius"])} for e in entries]
    return snapshot
