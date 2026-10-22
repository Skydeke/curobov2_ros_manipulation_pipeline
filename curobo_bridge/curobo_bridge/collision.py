# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

def validate(handle: int, q):
    """Per-config collision validity (PORT: fk_services._validate, advisory)."""
    import numpy as np
    import torch
    from curobo.types import JointState
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from curobo.scene import Scene
    from curobo.types import DeviceCfg

    entry = _REGISTRY[handle]
    kin = entry["kin"]
    device = torch.device("cuda", torch.cuda.current_device())
    q = np.asarray(q, dtype=np.float32)
    n = q.shape[0]

    checker = entry.get("checker")
    if checker is None:
        # PORT: fk_services._init - an EMPTY scene at construction (that is what
        # the reference does: cuRobo raises on a populated one; empty always
        # fits), sized from the collision cache exactly like the planners.
        try:
            cache = entry.get("collision_cache") or {}
            cfg = RobotCollisionCheckerCfg.load_from_config(
                robot_config=entry["robot_cfg"],
                scene_model=Scene(),
                device_cfg=DeviceCfg(device=device, dtype=torch.float32),
                collision_activation_distance=0.001,
                n_cuboids=max(1, int(cache.get("cuboid") or 0)),
                n_meshes=max(1, int(cache.get("mesh") or 0)),
            )
            checker = RobotCollisionChecker(cfg)
            # PORT: fk_services._init's `self._push_world()` right after the
            # empty build. The C++ FKServices hands the validator to the bridge
            # and it is built lazily here, on the FIRST validate() call - AFTER
            # every scene mutation already pushed its world (world_push skips a
            # checker that does not exist yet). Without this the checker keeps
            # the EMPTY construction scene forever and reports every
            # scene-collision config valid: the benchmark's cost leg read
            # 432/500 valid where the reference reads 364/500.
            checker.update_world(_collision_supported(
                _primitives_only_scene(entry), _obstacle_mode(entry)))
        except Exception as exc:
            # PORT: fk_services._init's failure path - say why before degrading
            # to all-True, or a broken validator is invisible.
            print(
                f"[curobo_bridge] FK collision validator init failed ({exc}); "
                "fk_batch poses_valid will report all True", flush=True)
            checker = False   # advisory: unavailable -> all True
        entry["checker"] = checker
    if checker is False or checker is None:
        return [1] * n

    q_t = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(1)  # [B,1,dof]
    mask = checker.validate(q_t)                                            # [B,1]
    return [1 if v else 0 for v in mask.squeeze(1).detach().cpu().tolist()]
def collision_distance(handle: int, q):
    """Flattened per-sphere scene distances (PORT: collision_distance.py)."""
    import numpy as np
    import torch
    from curobo.types import JointState
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer

    entry = _REGISTRY[handle]
    _spheres, dists, _activation, _checker = _sphere_snapshot(entry, q)
    return [float(v) for v in dists]
def _rot_from_quat_wxyz(quat):
    """PORT: collision_distance._rot_from_quat_wxyz - 3x3 rotation (numpy)."""
    import numpy as np

    qw, qx, qy, qz = (float(v) for v in quat)
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ], dtype=np.float64)
def _sphere_snapshot(entry, q, kin=None):
    """PORT: collision_distance._sphere_collision_snapshot. One FK feeds BOTH the
    per-sphere distances and the sphere array they index, so they align 1:1.

    Returns (spheres [N,4] np, dists np, activation, checker).
    """
    import numpy as np
    import torch
    from curobo.types import JointState
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer

    if kin is None:
        kin = entry["kin"]
    dcfg = DeviceCfg(device=torch.device("cuda", torch.cuda.current_device()),
                            dtype=torch.float32)
    q_t = torch.tensor(np.asarray(q, dtype=np.float32), dtype=torch.float32,
                       device=torch.device("cuda", torch.cuda.current_device()))
    js = JointState.from_position(q_t, joint_names=kin.joint_names)
    ks = kin.compute_kinematics(js)
    robot_spheres = ks.robot_spheres

    checker = _scene_collision(entry, dcfg)
    buf = CollisionBuffer.from_shape(robot_spheres.shape, dcfg)
    weight = dcfg.to_device([1.0])
    # PORT: _resolve_activation - the node's collision_activation_distance (same
    # margin the planner's optimizer uses).
    activation = float(entry.get("collision_activation_distance", 0.025))
    dist = checker.get_sphere_distance(
        ks, buf, weight, activation_distance=dcfg.to_device([activation]))
    # Negative-radius spheres are meaningless and would always read as colliding;
    # the mask must be SHAPE-PRESERVING (PORT: _sphere_collision_snapshot).
    neg = robot_spheres[..., 3] < 0
    if bool(neg.any()):
        dist = dist * (~neg)
    spheres = robot_spheres.reshape(-1, 4).detach().cpu().numpy()
    dists = np.asarray([float(v) for v in torch.flatten(dist, start_dim=0).tolist()])
    return spheres, dists, activation, checker
def _link_of_factory(kin):
    """PORT: the sphere-index -> link-name mapping shared by the diagnostics."""
    kp = kin.config.kinematics_config
    idx_map = (kp.link_sphere_idx_map.reshape(-1).detach().cpu().tolist()
               if kp.link_sphere_idx_map is not None else [])
    idx_to_name = {int(v): k for k, v in kp.link_name_to_idx_map.items()}

    def _link_of(sph):
        li = idx_map[sph] if sph < len(idx_map) else None
        return idx_to_name.get(li, f"link#{li}") if li is not None else "unknown-link"

    return _link_of
def collision_attribution(handle: int, q) -> dict:
    """Per-contact collision attribution at joint state `q`. PORT: collision_distance._attributed_collisions +
    _attributed_self_collisions + _attributed_joint_limit_violations. Feeds the
    node's StateCollisions message and the "collision feedback:" failure log."""
    import numpy as np

    entry = _REGISTRY[handle]
    # NB: the Python diagnostic attributes against attach_svc.kinematics() (the
    # MotionPlanner's kin, which carries fitted attached-object spheres). Using
    # that kin here produced non-physical distances (~100 m) against our
    # standalone scene checker, so we use the robot-model kinematics - the same
    # kin the colliding-distance harness (`collision_distance`) uses, which
    # agrees with get_collision_distance. Attached-object spheres are therefore
    # covered via the world scene, not the kin.
    kin = entry["kin"]
    spheres, dists, activation, checker = _sphere_snapshot(entry, q, kin=kin)
    link_of = _link_of_factory(kin)

    scene_data = getattr(checker, "data", None)

    def _active(store):
        if store is None:
            return []
        try:
            count = int(store.count[0].item())
            enable = store.enable[0]
            names = store.names[0][:count]
            dims = store.dims[0].detach().cpu().numpy()
            inv_pose = store.inv_pose[0].detach().cpu().numpy()
        except Exception:
            return []
        out = []
        for j in range(count):
            on = enable[j].item() if hasattr(enable[j], "item") else enable[j]
            if int(on) != 1:
                continue
            name = names[j]
            if isinstance(name, (bytes, bytearray)):
                name = name.decode()
            else:
                name = str(name)
            out.append((name, dims[j], inv_pose[j]))
        return out

    cuboids = _active(getattr(scene_data, "cuboids", None))
    meshes = _active(getattr(scene_data, "meshes", None))
    has_voxels = getattr(scene_data, "voxels", None) is not None
    world = []
    for sph_idx, depth in enumerate(dists):
        if not (depth > 0.0) or sph_idx >= len(spheres):
            continue
        c = spheres[sph_idx]
        center, radius = c[:3].astype(np.float64), float(c[3])
        if radius < 0:
            continue
        hits = []
        for name, dims, inv_pose in cuboids:
            half = dims[:3] * 0.5
            rot = _rot_from_quat_wxyz(inv_pose[3:7])
            local = rot @ center + inv_pose[:3]
            box = np.abs(local) - half
            outside = np.sqrt(np.sum(np.maximum(box, 0.0) ** 2))
            sdf = outside + min(np.max(box), 0.0)
            # Same reach test as the GPU kernel's inflated sphere.
            if sdf < radius + activation:
                hits.append(name)
        note = ""
        if not hits:
            causes = []
            if meshes:
                causes.append("mesh:" + ",".join(n for n, _, _ in meshes))
            if has_voxels:
                causes.append("depth-voxel-layer")
            note = " or ".join(causes)
        world.append({
            "link": link_of(sph_idx),
            "obstacle": ", ".join(hits) if hits else (note or "unknown-obstacle"),
            "depth": float(depth),
        })
    world.sort(key=lambda r: -r["depth"])

    selfs = []
    sc = getattr(kin.config, "self_collision_config", None)
    if sc is not None:
        pairs = getattr(sc, "collision_pairs", None)
        padding = getattr(sc, "sphere_padding", None)
        if pairs is not None and padding is not None and pairs.numel() != 0 \
                and spheres.shape[0] == int(getattr(sc, "num_spheres", -1)):
            pad = padding.detach().cpu().numpy().reshape(-1)
            idx = pairs.detach().cpu().numpy().astype(np.int64)
            ra = spheres[idx[:, 0], 3] + pad[idx[:, 0]]
            rb = spheres[idx[:, 1], 3] + pad[idx[:, 1]]
            cd = np.linalg.norm(spheres[idx[:, 0], :3] - spheres[idx[:, 1], :3], axis=1)
            overlap2 = (ra + rb) ** 2 - cd ** 2
            valid = spheres[:, 3] >= 0
            for k in np.flatnonzero((overlap2 > 0.0) & valid[idx[:, 0]] & valid[idx[:, 1]]):
                a, b = int(idx[k, 0]), int(idx[k, 1])
                selfs.append({"link": link_of(a), "link_b": link_of(b),
                              "depth": float(ra[k] + rb[k] - cd[k])})
            selfs.sort(key=lambda r: -r["depth"])

    cspace = []
    try:
        limits = kin.get_joint_limits()
        pos = limits.position.detach().cpu().numpy()
        lim_names = list(limits.joint_names)
        names = list(kin.joint_names)
        lim_idx = {n: i for i, n in enumerate(lim_names)}
        if q is not None and len(names) == len(q):
            for i, name in enumerate(names):
                if name not in lim_idx:
                    continue
                value = float(q[i])
                lo = float(pos[0, lim_idx[name]])
                hi = float(pos[1, lim_idx[name]])
                if value < lo or value > hi:
                    cspace.append({"joint": name, "value": value,
                                   "lower": lo, "upper": hi})
    except Exception:
        pass

    return {"world": world, "self": selfs, "cspace": cspace}

from .runtime import _REGISTRY
from .robot_introspection import robot_spheres
from .scene import _scene_collision
from .voxel import _collision_supported, _obstacle_mode, _primitives_only_scene
