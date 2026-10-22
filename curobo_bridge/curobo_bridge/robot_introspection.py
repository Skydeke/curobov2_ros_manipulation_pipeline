# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

def robot_model_rebuild_locks(handle: int, lock_json: str) -> None:
    """Rebuild the shared Kinematics with a lock_joints override.

    PORT: robot_model_manager.set_lock_joints. The solvers built from the robot
    cfg must be rebuilt afterwards (motion_planner_build).
    """
    import json

    from curobo.kinematics import Kinematics
    from curobo._src.types.robot import RobotCfg

    entry = _REGISTRY[handle]
    locks = json.loads(lock_json)
    from curobo.kinematics import KinematicsCfg
    # NB: entry["robot_config_file"] is NOT recorded (the build stores the
    # RESOLVED path); an eager .get(default) here raised KeyError on every
    # joint-lock change. Resolve lazily and say so if neither is on record.
    resolved = entry.get("resolved_config") or entry.get("robot_config_file")
    if not resolved:
        raise ValueError(
            "robot_model_rebuild_locks: no resolved robot config on record; "
            "build the robot model first")
    # Empty locks -> rebuild from the YAML's OWN lock_joints (restore). A
    # non-empty override REPLACES the YAML's lock_joints, so the YAML's unpinned
    # locks (`lock_joints: {joint: null}`) must be merged back in - they are not
    # part of an override map, but dropping them would leave those joints neither
    # in the cspace nor locked and the model would refuse to build.
    if not locks:
        kin_cfg = KinematicsCfg.from_robot_yaml_file(resolved)
    else:
        _unpinned = {name: None for name in entry.get("unpinned_lock_joints", [])}
        kin_cfg = KinematicsCfg.from_robot_yaml_file(
            resolved, lock_joints={**_unpinned, **locks})
    robot_cfg = RobotCfg(
        kinematics=kin_cfg, dynamics=entry["robot_cfg"].dynamics)
    entry["kin_cfg"] = kin_cfg
    entry["robot_cfg"] = robot_cfg
    entry["kin"] = Kinematics(robot_cfg.kinematics)
    entry["joint_names"] = list(entry["kin"].joint_names)
    entry["dof"] = int(entry["kin"].get_dof())
    entry["unpinned_lock_joints"] = list(
        entry["kin"].config.kinematics_config.unpinned_lock_joints or [])
    if locks:
        entry["lock_joints"] = dict(locks)
    else:
        # Restore: the model now carries the YAML's OWN lock_joints, and
        # joint_limits() reports this map - so report the YAML's values, not
        # an empty override (which would claim nothing is locked while the
        # model has them baked in).
        try:
            import yaml as _yaml
            with open(resolved) as _fh:
                _cfg = _yaml.safe_load(_fh) or {}
            _kin = _robot_kinematics_block(_cfg)
            entry["lock_joints"] = {
                str(k): float(v)
                for k, v in (_kin.get("lock_joints") or {}).items()
                if v is not None
            }
        except Exception:
            entry["lock_joints"] = {}
    entry["motion_planner"] = None
    entry["ik"] = None
def robot_model_update_lock_joints(handle: int, lock_json: str) -> None:
    """Update the values of the robot's unpinned lock joints, in place. PORT: robot_model_manager.update_lock_joints. These are the joints written
    ``lock_joints: {joint: null}`` in the robot YAML: locked out of the planning
    cspace (never planned) but NOT pinned, so their value follows the joint state."""
    import json

    entry = _REGISTRY[handle]
    values = {str(k): float(v) for k, v in json.loads(lock_json).items()}
    if not values:
        return
    # The shared Kinematics drives FK and segmentation. The solvers keep their own
    # kinematics, so push the same rewrite into them where they expose one.
    targets = [entry["kin"]]
    planner = entry.get("motion_planner")
    if planner is not None:
        targets.append(getattr(planner, "kinematics", None))
    ik = entry.get("ik")
    if ik is not None:
        targets.append(getattr(ik, "kinematics", ik))
    updated = set()
    for target in targets:
        config = getattr(getattr(target, "config", None), "kinematics_config", None)
        if config is None or id(config) in updated:
            continue
        updated.add(id(config))
        target.update_lock_joints(values)
def tool_frames(handle: int):
    """The kinematics model's tool frames (kin.tool_frames)."""
    return list(_REGISTRY[handle]["kin"].tool_frames)
def robot_spheres(handle: int, q):
    """Robot collision spheres in the base frame as [S,4] (PORT:
    robot_model_manager.get_collision_spheres)."""
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    kin = entry["kin"]
    js = JointState.from_position(
        torch.tensor([list(np.asarray(q, dtype=np.float32))], dtype=torch.float32,
                     device=torch.device("cuda", torch.cuda.current_device())),
        joint_names=kin.joint_names)
    res = kin.compute_kinematics(js)
    return res.robot_spheres.reshape(-1, 4).detach().cpu().numpy().astype(np.float32)
def _resolve_descriptor(robot_arg: str) -> str:
    """Locate robots/<name>.yaml: an absolute path, package://, or a package
    share lookup."""
    import os

    from ament_index_python.packages import get_package_share_directory

    if not robot_arg:
        return ""
    if robot_arg.startswith("package://"):
        return _resolve_uri(robot_arg, "")
    if os.path.isabs(robot_arg) or os.path.isfile(robot_arg):
        return robot_arg
    # Bare name -> the package's robots/ dir, then the legacy config/robots dirs.
    candidates = []
    try:
        share = get_package_share_directory("curobov2_ros")
        candidates.append(os.path.join(share, "robots", f"{robot_arg}.yaml"))
        candidates.append(os.path.join(share, "config", "robots", f"{robot_arg}.yaml"))
    except Exception:
        pass
    candidates.append(os.path.join(os.getcwd(), "robots", f"{robot_arg}.yaml"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""
def robot_description(robot_arg: str, robot_config_file: str):
    """Load a robot descriptor (PORT: robot_description.load_robot_description).

    Returns the fields the C++ node needs: base_link, the resolved cuRobo config
    (`curobo_config`, a package:// URI resolved to a path), the control strategy
    key + its params, and the `home` named joint config.
    """
    import os

    import yaml

    path = _resolve_descriptor(robot_arg)
    if not path or not os.path.isfile(path):
        return {"found": False, "path": path}
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    curobo_config = data.get("curobo_config") or ""
    if curobo_config:
        curobo_config = _resolve_uri(curobo_config, os.path.dirname(path))
    params = data.get("strategy_params") or {}
    home = (data.get("named_joint_configs") or {}).get("home") or {}
    return {
        "found": True,
        "path": os.path.abspath(path),
        "name": data.get("name", ""),
        "base_link": data.get("base_link", ""),
        "curobo_config": curobo_config or robot_config_file,
        "strategy": data.get("strategy") or data.get("control_strategy") or "",
        "strategy_params": {str(k): str(v) for k, v in params.items()},
        "home_names": [str(k) for k in home.keys()],
        "home_positions": [float(v) for v in home.values()],
    }
def collision_spheres_with_attached(handle: int, q):
    """World collision spheres + per-sphere attached flag.

    PORT: robot_model_manager.get_collision_spheres_with_attached - sourced from
    the MotionPlanner's kinematics (which carries the fitted attached-object
    spheres); our own kin model never receives grasps.
    """
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    mp = entry.get("motion_planner")
    kin = getattr(mp, "kinematics", None) or entry["kin"]
    js = JointState.from_position(
        torch.tensor([list(np.asarray(q, dtype=np.float32))], dtype=torch.float32,
                     device=torch.device("cuda", torch.cuda.current_device())),
        joint_names=kin.joint_names)
    spheres = kin.compute_kinematics(js).robot_spheres.reshape(-1, 4)
    kc = kin.kinematics_config
    attached_idx = kc.link_name_to_idx_map.get("attached_object")
    mask = [False] * int(spheres.shape[0])
    link_map = getattr(kc, "link_sphere_idx_map", None)
    if attached_idx is not None and link_map is not None:
        link_ids = link_map.reshape(-1).cpu().tolist()
        if len(link_ids) == len(mask):
            mask = [li == attached_idx for li in link_ids]
    # Link-relative copy of the attached spheres. The consumer (trajectory
    # preview) re-applies FK of this URDF link at every waypoint, so the
    # payload rides the ghost instead of staying at the live pose. We derive
    # the link frame from the solver's own FK rather than the YAML transform
    # so it is exact for whatever `fixed_transform` the config uses.
    world = spheres.detach().cpu().numpy().astype(np.float32)
    attached_frame = ""
    attached_local = np.zeros_like(world)
    parent = entry.get("attach_parent_link")
    if parent and any(mask):
        try:
            q_t = torch.tensor(
                [list(np.asarray(q, dtype=np.float32))], dtype=torch.float32,
                device=torch.device("cuda", torch.cuda.current_device()))
            link_pose = kin.get_link_poses(q_t, [parent])
            p = link_pose.position[0, 0].detach().cpu().numpy()
            # Pose.get_rotation() only accepts a flat [b, 4] quaternion, and FK
            # returns [b, n_links, 4]; convert the quaternion here instead of
            # relying on that shape contract.
            w, x, y, z = link_pose.quaternion[0, 0].detach().cpu().numpy()
            rot = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x),
                 1 - 2 * (x * x + y * y)],
            ], dtype=np.float64)
            keep = np.asarray(mask, dtype=bool)
            # world -> link-local: R^T (p_world - p_link); v @ R == R^T v.
            attached_local[keep, :3] = (world[keep, :3] - p) @ rot
            attached_local[keep, 3] = world[keep, 3]
            attached_frame = parent
        except Exception:
            attached_frame = ""
            attached_local = np.zeros_like(world)
    return {
        "spheres": world,
        "attached": mask,
        "attached_local": attached_local,
        "attach_frame": attached_frame,
    }

from .runtime import _REGISTRY
from .robot_model import _resolve_uri, _robot_kinematics_block
from .ik import ik
