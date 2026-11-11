# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_python_runtime - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
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
def robot_model_fixed_transforms(handle: int) -> bytes:
    """Return the current robot fixed-transform tensor as raw float32 bytes.

    The native C++ planner owns device copies of the same per-link fixed
    transforms. ``robot_model_update_lock_joints`` updates the Python model;
    this getter lets the C++ bridge upload the resulting tensor into every
    live native robot stage so ``lock_joints: {joint: null}`` follows the
    current/configured joint state there too.
    """
    import numpy as np

    entry = _REGISTRY[handle]
    planner = entry.get("motion_planner")
    kin = getattr(planner, "kinematics", None) if planner is not None else None
    if kin is None:
        kin = entry["kin"]
    tensor = kin.config.kinematics_config.fixed_transforms
    array = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
    return array.tobytes()


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
    key + its params, the `home` named joint config, and the full
    `named_joint_configs` / `gripper_controllers` maps flattened for
    GetJointInfo (so tooling does not need MoveIt's SRDF).
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
    named = data.get("named_joint_configs") or {}
    home = named.get("home") or {}

    # Flatten the named configs for the C++ -> ROS boundary. A dict of dicts
    # cannot cross the pybind bridge without a JSON dependency, so each pose
    # becomes (name, joint names, positions, size). Order is the YAML's own
    # document order, which is the order a UI should show.
    named_config_names = []
    named_config_joint_names = []
    named_config_positions = []
    named_config_sizes = []
    for pose_name, joints in named.items():
        if not isinstance(joints, dict):
            continue
        pairs = [(str(j), float(v)) for j, v in joints.items()]
        named_config_names.append(str(pose_name))
        named_config_sizes.append(len(pairs))
        for joint, value in pairs:
            named_config_joint_names.append(joint)
            named_config_positions.append(value)

    # Gripper controllers: same schema the task constructor consumes. The
    # legacy single `gripper_controller:` block is the one-entry form.
    grippers = data.get("gripper_controllers") or {}
    if not grippers and isinstance(data.get("gripper_controller"), dict):
        grippers = {"default": data["gripper_controller"]}
    gripper_names = []
    gripper_types = []
    gripper_controllers = []
    gripper_action_ns = []
    gripper_joint_lists = []
    gripper_command_joints = []
    gripper_open_positions = []
    gripper_close_positions = []
    gripper_max_efforts = []
    gripper_time_from_start = []
    for gripper_name, cfg in grippers.items():
        if not isinstance(cfg, dict):
            continue
        joints = cfg.get("joints") or []
        gripper_names.append(str(gripper_name))
        gripper_types.append(str(cfg.get("type") or "GripperCommand"))
        gripper_controllers.append(str(cfg.get("name") or ""))
        gripper_action_ns.append(str(cfg.get("action_ns") or "gripper_cmd"))
        gripper_joint_lists.append(" ".join(str(j) for j in joints))
        gripper_command_joints.append(str(cfg.get("command_joint") or ""))
        gripper_open_positions.append(float(cfg.get("open", 0.0) or 0.0))
        gripper_close_positions.append(float(cfg.get("close", 0.0) or 0.0))
        gripper_max_efforts.append(float(cfg.get("max_effort", 0.0) or 0.0))
        gripper_time_from_start.append(
            float(cfg.get("time_from_start", 2.0) or 2.0))

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
        "named_config_names": named_config_names,
        "named_config_joint_names": named_config_joint_names,
        "named_config_positions": named_config_positions,
        "named_config_sizes": named_config_sizes,
        "gripper_names": gripper_names,
        "gripper_types": gripper_types,
        "gripper_controllers": gripper_controllers,
        "gripper_action_ns": gripper_action_ns,
        "gripper_joint_lists": gripper_joint_lists,
        "gripper_command_joints": gripper_command_joints,
        "gripper_open_positions": gripper_open_positions,
        "gripper_close_positions": gripper_close_positions,
        "gripper_max_efforts": gripper_max_efforts,
        "gripper_time_from_start": gripper_time_from_start,
    }
def _self_collision_pair_cost(entry: dict, sc, spheres):
    """Cache one store_pair_distance SelfCollisionCost per collision config.

    The pair buffer keeps the exact kernel pair list and sphere padding, so the
    visualizer does not re-derive a collision model of its own. It is keyed by
    the config object: a rebuild (lock joints, new artifact) creates a new one,
    while link enable/disable only changes sphere radii and reuses this cost.
    """
    from curobo._src.cost.cost_self_collision import SelfCollisionCost
    from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg

    key = (id(sc), str(spheres.device), spheres.dtype)
    cached = entry.get("_self_collision_pair_cost")
    if cached is not None and cached[0] == key:
        return cached[1]
    cost = SelfCollisionCost(
        SelfCollisionCostCfg(
            weight=1.0,
            self_collision_kin_config=sc,
            store_pair_distance=True,
        )
    )
    entry["_self_collision_pair_cost"] = (key, cost)
    return cost


def _self_collision_pair_diagnostics(entry: dict, kin, spheres):
    """Per-sphere self-collision penetration from cuRobo's own pair kernel.

    `store_pair_distance=True` writes every configured pair's squared overlap.
    Converting that with the same per-sphere padding the kernel used gives the
    exact signed penetration (positive = collision, negative = clearance). The
    per-sphere value is the WORST pair it participates in.

    Returns (padding [S], self_distance [S]) as float32 numpy arrays.
    """
    import numpy as np
    import torch

    n_spheres = int(spheres.shape[-2])
    padding = np.zeros(n_spheres, dtype=np.float32)
    self_distance = np.full(n_spheres, -np.inf, dtype=np.float32)

    # Escape hatch for a very large collision topology where even the 1.04x
    # store_pair_distance kernel is too expensive; see
    # curobov2_ros/scripts/bench_self_collision_pairs.py.
    import os
    if os.environ.get("CUROBO_DISABLE_SPHERE_PAIR_DIAG"):
        return padding, self_distance
    sc = getattr(kin.config, "self_collision_config", None)
    if sc is None or getattr(sc, "collision_pairs", None) is None:
        return padding, self_distance
    pairs = sc.collision_pairs
    pad_t = getattr(sc, "sphere_padding", None)
    if pairs.numel() == 0 or pad_t is None:
        return padding, self_distance
    pad = pad_t.detach().reshape(-1)
    if int(pad.shape[0]) != n_spheres or int(getattr(sc, "num_spheres", -1)) != n_spheres:
        return padding, self_distance
    padding = pad.to(dtype=torch.float32).cpu().numpy()

    cost = _self_collision_pair_cost(entry, sc, spheres)
    cost.setup_batch_tensors(int(spheres.shape[0]), int(spheres.shape[1]))
    cost.forward(spheres)
    raw = cost._pair_distance.reshape(-1)
    idx = pairs.to(dtype=torch.int64)
    flat = spheres.reshape(-1, 4)
    a = idx[:, 0]
    b = idx[:, 1]
    radius_a = flat[a, 3] + pad[a]
    radius_b = flat[b, 3] + pad[b]
    radius_sum = radius_a + radius_b
    # raw = radius_sum^2 - centre_distance^2. Recover the signed penetration:
    # positive when the padded spheres overlap.
    centre_distance = torch.sqrt(torch.clamp(radius_sum * radius_sum - raw, min=0.0))
    penetration = radius_sum - centre_distance
    # Disabled spheres (negative radius) are inactive in cuRobo; the kernel
    # masks their pairs to zero, which must not look like a real contact.
    valid = (flat[a, 3] >= 0.0) & (flat[b, 3] >= 0.0)
    penetration = torch.where(
        valid, penetration, torch.full_like(penetration, float("-inf")))

    per_sphere = torch.full(
        (n_spheres,), float("-inf"), dtype=penetration.dtype, device=penetration.device)
    per_sphere.scatter_reduce_(0, a, penetration, reduce="amax", include_self=True)
    per_sphere.scatter_reduce_(0, b, penetration, reduce="amax", include_self=True)
    self_distance = per_sphere.detach().cpu().numpy().astype(np.float32)
    return padding, self_distance


def _sphere_link_names(kin, n_spheres: int):
    """The URDF link owning each sphere, row-aligned with robot_spheres."""
    kc = kin.config.kinematics_config
    idx_map = getattr(kc, "link_sphere_idx_map", None)
    if idx_map is None or int(idx_map.numel()) != n_spheres:
        return []
    idx = idx_map.reshape(-1).detach().cpu().tolist()
    names = {int(v): str(k) for k, v in kc.link_name_to_idx_map.items()}
    return [names.get(int(v), "") for v in idx]


def collision_spheres_with_attached(handle: int, q):
    """World collision spheres + per-sphere attached flag and diagnostics.

    PORT: robot_model_manager.get_collision_spheres_with_attached - sourced from
    the MotionPlanner's kinematics (which carries the fitted attached-object
    spheres); our own kin model never receives grasps.

    The extra `self_padding`, `self_distance`, `world_distance` and `link_names`
    fields describe the SAME FK and collision model the solver uses, so RViz can
    colour the exact model instead of approximating it from sphere centres.
    """
    import numpy as np
    import torch

    entry = _REGISTRY[handle]
    mp = entry.get("motion_planner")
    kin = getattr(mp, "kinematics", None) or entry["kin"]
    # One FK feeds the spheres, the world distance and the self-collision pair
    # query; querying the historical standalone `entry["kin"]` here was the
    # source of sphere/distance array misalignment once a planner was built.
    from .collision import _sphere_snapshot
    world, world_distance, _activation, _checker = _sphere_snapshot(entry, q, kin=kin)

    spheres = torch.as_tensor(
        world, dtype=torch.float32,
        device=torch.device("cuda", torch.cuda.current_device())).reshape(1, 1, -1, 4)
    self_padding, self_distance = _self_collision_pair_diagnostics(
        entry, kin, spheres)

    kc = kin.kinematics_config
    # Only an active attachment counts. The `attached_object*` link always has
    # preallocated (disabled) sphere slots in the artifact; marking them all as
    # attached would put those slots in the payload namespace even with nothing
    # in the gripper.
    used_link = entry.get("attach_link_used") or ""
    attached_idx = kc.link_name_to_idx_map.get(used_link) if used_link else None
    link_map = getattr(kc, "link_sphere_idx_map", None)
    mask = []
    if attached_idx is not None and link_map is not None:
        mask = (link_map.reshape(-1) == attached_idx).detach().cpu().tolist()
    if len(mask) != world.shape[0]:
        mask = [False] * world.shape[0]

    # Link-relative copy of the attached spheres. The consumer (trajectory
    # preview) re-applies FK of this URDF link at every waypoint, so the
    # payload rides the ghost instead of staying at the live pose. We derive
    # the link frame from the solver's own FK rather than the YAML transform
    # so it is exact for whatever `fixed_transform` the config uses.
    attached_frame = ""
    attached_local = np.zeros_like(world)
    parent = entry.get("attach_parent_link_used") or entry.get("attach_parent_link")
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
        "self_padding": np.asarray(self_padding, dtype=np.float32).reshape(-1, 1),
        "self_distance": np.asarray(self_distance, dtype=np.float32).reshape(-1, 1),
        "world_distance": np.asarray(world_distance, dtype=np.float32).reshape(-1, 1),
        "link_names": _sphere_link_names(kin, int(world.shape[0])),
    }

from .runtime import _REGISTRY
from .robot_model import _resolve_uri, _robot_kinematics_block
from .ik import ik
