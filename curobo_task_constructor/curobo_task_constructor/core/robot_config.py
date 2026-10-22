"""Robot descriptor reading for the ROS-free core.

Fixed-state stages (`FixedState`) resolve a named joint configuration against
the **robot's own config YAML** — the same file that already carries
`strategy` / `strategy_params` (read by `GetRobotStrategies`) and the
`attached_object` link config. We do NOT ship a separate
`curobo_task_constructor`-owned config: the descriptor gains a
`named_joint_configs:` section, and this module reads it with exactly the
same `load_yaml(desc_path).get(...)` idiom `robot_description.load_robot_description`
uses for `strategy`/`strategy_params`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:  # ROS deployment: curobo's helper (same idiom robot_description uses)
    from curobo.config_io import load_yaml  # type: ignore
except Exception:  # pragma: no cover - ROS-free core / tests
    import yaml

    def load_yaml(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)


@dataclass
class NamedJointConfig:
    """One named joint configuration.

    ``names``/``positions`` are parallel lists; when the descriptor lists all
    robot joints per config, the plan for the whole robot is given
    explicitly. When it only lists a few joints (e.g. the finger), the robot
    adapter merges the sparse values onto the current joint state.
    """

    name: str
    names: list
    positions: list

    def as_dict(self) -> dict:
        return dict(zip(self.names, self.positions))


def load_robot_descriptor(robot_config_path: str) -> dict:
    """Parse a robot descriptor YAML into a plain dict."""
    return load_yaml(robot_config_path)


def get_named_joint_configs(robot_config_path: str) -> dict:
    """``{name: NamedJointConfig}`` from the descriptor's
    ``named_joint_configs:`` section.

    Each entry maps a name to either a flat list of positions (in canonical
    joint order) or a name→position mapping. A flat list is stored with empty
    ``names`` — the adapter resolves them against the robot's joint order.
    """
    desc = load_robot_descriptor(robot_config_path)
    section = desc.get("named_joint_configs", {}) or {}
    out = {}
    for name, value in section.items():
        if isinstance(value, dict):
            out[name] = NamedJointConfig(name, list(value.keys()),
                                         [float(v) for v in value.values()])
        else:
            out[name] = NamedJointConfig(name, [], [float(v) for v in value])
    return out


def resolve_named_config(robot_config_path: str, name: str) -> Optional[NamedJointConfig]:
    configs = get_named_joint_configs(robot_config_path)
    resolved = configs.get(name)
    if resolved is None:
        raise KeyError(f"named joint config {name!r} not found in "
                       f"{robot_config_path}; available: "
                       f"{', '.join(sorted(configs)) or '<none>'}")
    return resolved


def _resolve_config_path(path: Optional[str]) -> Optional[str]:
    """Resolve a descriptor's ``curobo_config`` (``package://`` or a file)."""
    if not path:
        return None
    if str(path).startswith("package://"):
        try:
            from ament_index_python.packages import get_package_share_directory
        except Exception:
            return None
        pkg, _, rel = str(path)[len("package://"):].partition("/")
        try:
            return os.path.join(get_package_share_directory(pkg), rel)
        except Exception:
            return None
    return str(path) if os.path.exists(str(path)) else None


def _cspace_order_from_config(cfg_path: Optional[str]) -> Optional[list]:
    """``kinematics.cspace.joint_names`` of a cuRobo config, if present."""
    if not cfg_path:
        return None
    try:
        cfg = load_yaml(cfg_path)
        cspace = (cfg.get("kinematics") or {}).get("cspace") or {}
        order = cspace.get("joint_names") or []
    except Exception:
        return None
    return [str(n) for n in order] if order else None


def canonical_joint_order(robot_config_path: str) -> Optional[list]:
    """Canonical joint-name order for the curobo_server wire, if known. The server resolves ``start_pose``/``target_joint_positions`` position lists
    VERBATIM in cspace order, while ``/joint_states`` order is
    publisher-defined, so readings must be reordered by name before they reach
    the wire. Source of preference order:

    1."""
    desc = load_robot_descriptor(robot_config_path)
    cspace = (desc.get("kinematics") or {}).get("cspace") or {}
    order = cspace.get("joint_names") or []
    if order:
        return [str(n) for n in order]
    order = _cspace_order_from_config(
        _resolve_config_path(desc.get("curobo_config")))
    if order:
        return order
    order = []
    seen = set()
    for value in (desc.get("named_joint_configs") or {}).values():
        # dict-form configs carry names (flat lists are bare positions with no
        # names — the adapter cannot order those, matching get_named_joint_configs)
        if isinstance(value, dict):
            for name in value:
                if name not in seen:
                    seen.add(name)
                    order.append(name)
    return order or None


def _normalize_gripper(cfg: dict) -> dict:
    """One gripper-controller entry with every field defaulted."""
    cfg = cfg or {}
    joints = list(cfg.get("joints") or [])
    if not joints and cfg.get("joint"):
        joints = [cfg["joint"]]
    joints = [str(j) for j in joints]
    command_joint = str(cfg.get("command_joint")
                        or (joints[0] if joints else cfg.get("joint", "")))
    return {
        "type": str(cfg.get("type", "GripperCommand")),
        "name": str(cfg.get("name", "gripper")),
        "action_ns": str(cfg.get("action_ns", "gripper_cmd")),
        "joints": joints,
        "command_joint": command_joint,
        "open": float(cfg.get("open", 0.0) or 0.0),
        "close": float(cfg.get("close", 0.0) or 0.0),
        "max_effort": float(cfg.get("max_effort", 0.0) or 0.0),
        "allow_stalling": bool(cfg.get("allow_stalling", False)),
    }


def get_gripper_controllers(robot_config_path: str) -> dict:
    """``{name: normalized-gripper}`` from the robot descriptor.

    The general schema is a ``gripper_controllers:`` map, one entry per end
    effector (so a bimanual robot can name ``left``/``right``). A single
    ``gripper_controller:`` block is accepted as the legacy one-entry form
    under the key ``default``. Returns ``{}`` when the descriptor has neither.
    """
    desc = load_robot_descriptor(robot_config_path) or {}
    section = desc.get("gripper_controllers")
    if isinstance(section, dict) and section:
        return {str(k): _normalize_gripper(v) for k, v in section.items()}
    legacy = desc.get("gripper_controller")
    if legacy:
        return {"default": _normalize_gripper(legacy)}
    return {}


def reorder_joint_vectors(joint_order: list, names: list, position: list,
                          velocity: Optional[list] = None,
                          effort: Optional[list] = None):
    """Reorder parallel joint arrays into ``joint_order`` by name. Only joints present in ``names`` are kept, in canonical order, so a
    7-DOF arm-only reading stays a contiguous canonical prefix — the server
    pads a short trailing DOF (the gripper) from its own current pose. Unknown joints are dropped."""
    by_name = {}
    for i, name in enumerate(names):
        by_name.setdefault(name, i)  # first occurrence wins (defensive)
    out_names = [n for n in joint_order if n in by_name]
    out_position = [position[by_name[n]] for n in out_names]

    def _reordered(src):
        if not src or len(src) != len(names):
            return []
        return [src[by_name[n]] for n in out_names]

    return out_names, out_position, _reordered(velocity), _reordered(effort)