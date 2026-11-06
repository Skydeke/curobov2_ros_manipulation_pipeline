#!/usr/bin/env python3
"""Write generated metadata into a cuRobo native artifact bundle.

This is a *build-time* helper.  It reads the robot descriptor (``robots/*.yaml``)
and the cuRobo config YAML that already describe the robot, and writes a
``manifest.json`` next to the exported native runtime files.  C++ never parses
YAML: it reads this generated manifest instead.

The file is generated output.  Do not check it in and do not hand-edit it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


# Methods the native (cpp) bridge implements.  Keep this in sync with the
# methods CppBridge overrides rather than the Python IBridge surface:
# perception/mapper/attachment calls are still Python-only and must not be
# advertised here, otherwise `auto` would pick the native backend for a node
# that needs them.
NATIVE_CAPABILITIES = (
    "robot_model",
    "fk",
    "ik",
    "ik_multi",
    "validate",
    "plan_pose",
    "plan_cspace",
    "plan_pose_waypoints",
    "reset_seed",
    "tool_frames",
    "joint_limits",
    "set_link_collision",
    "scene_set",
    "scene_get",
    # Native mapper (CppBridge owns libcurobo_mapper_runtime).
    "configure_mapper",
    "warmup_mapper",
    "esdf_refresh",
    "voxel_grid",
    "depth_integrate",
    "clear_voxel_map",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> Any:
    import yaml  # imported lazily so --help works without PyYAML

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def joint_limits(cspace: dict[str, Any], joint_names: list[str]) -> dict[str, list[float]]:
    """Return the per-joint limit arrays that are present in the cspace block."""
    limits: dict[str, list[float]] = {}
    for key, value in cspace.items():
        if not isinstance(value, list):
            continue
        if not ("limit" in key or key in ("default_joint_position",)):
            continue
        if len(value) != len(joint_names):
            continue
        if all(isinstance(item, (int, float)) for item in value):
            limits[key] = [float(item) for item in value]
    return limits


def home_state(
    descriptor: dict[str, Any], cspace: dict[str, Any], joint_names: list[str]
) -> tuple[list[str], list[float]]:
    named = descriptor.get("named_joint_configs") or {}
    home = named.get("home")
    if not isinstance(home, dict):
        home = {}
        defaults = as_list(cspace.get("default_joint_position"))
        if len(defaults) == len(joint_names):
            home = dict(zip(joint_names, defaults))
    # Preserve the cspace order; ignore joints the descriptor does not name.
    names = [name for name in joint_names if name in home]
    positions = [float(home[name]) for name in names]
    return names, positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, help="artifact/robot set name")
    parser.add_argument("--artifact-set", required=True, help="artifact set directory name")
    parser.add_argument("--arch", required=True, help="target GPU architecture, e.g. sm_89")
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--curobo-config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--tool-frame", action="append", default=[])
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--exporter-version", default="")
    args = parser.parse_args()

    robot_config = args.robot_config.resolve()
    curobo_config = args.curobo_config.resolve()
    artifact_dir = args.artifact_dir.resolve()

    descriptor = load_yaml(robot_config) or {}
    curobo_data = load_yaml(curobo_config) or {}

    robot_cfg = curobo_data.get("robot_cfg", curobo_data)
    kinematics = robot_cfg.get("kinematics", robot_cfg)
    cspace = kinematics.get("cspace") or {}
    cspace_joints = [str(name) for name in as_list(cspace.get("joint_names"))]

    tool_frames = list(args.tool_frame)
    if not tool_frames:
        tool_frames = [str(name) for name in as_list(kinematics.get("tool_frames"))]

    lock_joints = kinematics.get("lock_joints") or {}
    pinned_locks: dict[str, float] = {}
    unpinned_locks: list[str] = []
    for name, value in lock_joints.items():
        if value is None:
            unpinned_locks.append(str(name))
        else:
            pinned_locks[str(name)] = float(value)

    home_names, home_positions = home_state(descriptor, cspace, cspace_joints)

    source_hash = sha256_file(robot_config)
    curobo_config_hash = sha256_file(curobo_config)

    # Hash every descriptor in the same robots/ directory so the manifest records
    # the local source of truth (robots/*.yaml) without adding a second format.
    robot_sources: dict[str, str] = {}
    for candidate in sorted(robot_config.parent.glob("*.yaml")):
        robot_sources[candidate.name] = sha256_file(candidate)

    generated: dict[str, str] = {}
    for candidate in sorted(artifact_dir.glob("*")):
        if candidate.is_file() and candidate.name != "manifest.json":
            generated[candidate.name] = sha256_file(candidate)

    manifest = {
        "schema_version": 1,
        "generated_by": "docker/write_artifact_metadata.py",
        "exporter_version": args.exporter_version,
        "artifact_set": args.artifact_set,
        "arch": args.arch,
        "robot": {
            "name": descriptor.get("name") or args.robot,
            "display_name": descriptor.get("display_name", ""),
            "base_link": descriptor.get("base_link", kinematics.get("base_link", "")),
            "curobo_config": descriptor.get("curobo_config", ""),
        },
        "files": {
            "robot_config": str(robot_config),
            "curobo_config": str(curobo_config),
            "urdf": str(args.urdf.resolve()) if args.urdf is not None else "",
        },
        # This is the value the native runtime compares against the configured
        # robot YAML.  A mismatch makes `auto` choose Python and `cpp` fail.
        "source_hash": source_hash,
        "curobo_config_hash": curobo_config_hash,
        "robot_sources": robot_sources,
        "tool_frames": tool_frames,
        # Runtime behavior that the native bridge cannot derive from the model
        # artifact (control strategy and its controller routing).
        "strategy": descriptor.get("strategy", ""),
        "strategy_params": descriptor.get("strategy_params") or {},
        "native_capability_profile": {
            # Native planning/kinematics and the native mapper are complete.
            # The hybrid bridge still routes methods that require the Python
            # backend (robot segmentation, attachments) to Python per call, so
            # `auto` prefers CppBridge whenever the artifact is valid.
            "complete": True,
            "required": list(NATIVE_CAPABILITIES),
            **{name: True for name in NATIVE_CAPABILITIES},
            "mapper": True,
            "perception": True,
            # Attachments and lock-joint updates still use the Python fallback.
            "attachments": False,
            "lock_joint_updates": False,
        },
        "home": {
            "joint_names": home_names,
            "positions": home_positions,
        },
        "named_joint_configs": descriptor.get("named_joint_configs") or {},
        "gripper_controllers": descriptor.get("gripper_controllers") or {},
        "lock_joints": {
            "pinned": pinned_locks,
            "unpinned": unpinned_locks,
        },
        "cspace": {
            "joint_names": cspace_joints,
            "limits": joint_limits(cspace, cspace_joints),
        },
        "generated_files": generated,
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
