# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

def _primitives_only_scene(entry):
    """The scene's analytic primitives minus any attached (excluded) obstacle.

    PORT: obstacle_manager.primitives_only_scene (voxel layer excluded).
    """
    from curobo.scene import Scene

    attached = entry.get("attached_name")
    scene = Scene()
    for spec in entry.get("scene_specs", []):
        if attached and spec.get("name") == attached:
            continue
        primitive = _to_scene(spec)
        attr = {0: "cuboid", 1: "sphere", 2: "capsule", 3: "cylinder", 4: "mesh"}[
            int(spec["type"])]
        bucket = getattr(scene, attr)
        if bucket is None:
            setattr(scene, attr, [primitive])
        else:
            bucket.append(primitive)
    return scene
def _obstacle_mode(entry):
    """The node's obstacle_collision_mode (reference default: 'cuboid')."""
    return entry.get("obstacle_collision_mode") or "cuboid"
def _collision_supported(scene, mode="cuboid"):
    """PORT: obstacle_manager._collision_supported. * ``cuboid`` (default, the deployment's): sphere/capsule/cylinder -> OBB via
      ``get_cuboid()`` - the same approximation native cuRobo makes for the
      primitives. Cuboids, USER MESHES (deliberate exact geometry) and the voxel
      layer pass through unchanged."""
    from curobo.scene import Scene

    if mode == "mesh":
        from curobo._src.geom.types import SceneCfg

        if getattr(scene, "mesh", None) is None:  # create_collision_support_world concatenates it
            scene.mesh = []
        return SceneCfg.create_collision_support_world(scene)

    cuboids = list(scene.cuboid) if scene.cuboid is not None else []
    for attr in ("sphere", "capsule", "cylinder"):
        for obstacle in getattr(scene, attr) or []:
            cuboids.append(obstacle.get_cuboid())
    out = Scene(cuboid=cuboids, mesh=scene.mesh or [])
    out.voxel = getattr(scene, "voxel", None)
    return out
def _rasterize_primitives_gpu(entry, voxel_grid, grid_min, voxel_size) -> int:
    """Rasterize the analytic primitives with a pre-allocated GPU SceneCollision
    point-sphere query (r=0), marking cells where distance > 0.

    PORT: obstacle_manager._rasterize_primitives_gpu - the SAME collision checker
    the planner uses, so the published voxels match the solver's geometry.
    """
    import numpy as np
    import torch

    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
    from curobo.types import DeviceCfg

    scene = _collision_supported(_primitives_only_scene(entry), _obstacle_mode(entry))
    if not (list(scene.cuboid or []) or list(scene.mesh or [])):
        return 0
    size = tuple(int(v) for v in np.asarray(voxel_grid).shape)
    device = "cuda"
    cache = entry.setdefault("_voxel_raster_cache", {})
    grid_key = (tuple(float(v) for v in np.asarray(grid_min).tolist()),
                float(voxel_size), size)
    n_cub = len(scene.cuboid or [])
    n_mesh = len(scene.mesh or [])
    caps = cache.get("caps", (0, 0))
    if (cache.get("sc") is None or cache.get("key") != grid_key
            or n_cub > caps[0] or n_mesh > caps[1]):
        ax, ay, az = np.arange(size[0]), np.arange(size[1]), np.arange(size[2])
        gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
        idx = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        centers = (np.asarray(grid_min) + (idx + 0.5) * voxel_size).astype(np.float32)
        n_pts = len(centers)
        dev_cfg = DeviceCfg(device=device)
        cache["sc"] = SceneCollision.from_config(
            SceneCollisionCfg(device_cfg=dev_cfg, scene_model=scene))
        cache["spheres"] = torch.zeros((1, 1, n_pts, 4), device=device, dtype=torch.float32)
        cache["spheres"][0, 0, :, :3] = torch.tensor(centers, device=device)
        cache["buf"] = CollisionBuffer.from_shape(torch.Size([1, 1, n_pts]), dev_cfg)
        cache["weight"] = torch.ones(1, device=device, dtype=torch.float32)
        cache["act"] = torch.zeros(1, device=device, dtype=torch.float32)
        cache["key"] = grid_key
        cache["caps"] = (n_cub, n_mesh)
    sc = cache["sc"]
    sc.load_collision_model(scene)
    buf = cache["buf"]
    buf.zero_()
    sc.get_sphere_distance_raw(cache["spheres"], buf, cache["weight"], cache["act"])
    occ = (buf.distance[0, 0] > 0).cpu().numpy().reshape(size)
    voxel_grid |= occ.astype(np.uint8)
    return int(occ.sum())
def depth_integrate(handle: int, depth, intrinsics, camera_pose) -> None:
    """PORT: cameras/camera_depth_map_strategy._integrate_once."""
    import numpy as np
    import torch
    from curobo.types import CameraObservation, Pose
    from curobo._src.types.device_cfg import DeviceCfg

    entry = _REGISTRY[handle]
    device = torch.device("cuda", torch.cuda.current_device())
    dcfg = DeviceCfg(device=device, dtype=torch.float32)
    mapper = _mapper(entry)

    depth_t = torch.from_numpy(np.asarray(depth, dtype=np.float32)).to(
        device=device, dtype=torch.float32)
    if depth_t.ndim == 2:
        depth_t = depth_t.unsqueeze(0)
    intrinsics_t = torch.tensor(
        np.asarray(intrinsics, dtype=np.float32), dtype=torch.float32, device=device)
    if intrinsics_t.ndim == 2:
        intrinsics_t = intrinsics_t.unsqueeze(0)
    rgb = torch.zeros((depth_t.shape[0], depth_t.shape[1], depth_t.shape[2], 3),
                      dtype=torch.uint8, device=device)
    camera_pose_t = Pose.from_list(list(camera_pose), device_cfg=dcfg)
    mapper.integrate(CameraObservation(
        depth_image=depth_t, rgb_image=rgb, intrinsics=intrinsics_t, pose=camera_pose_t))
def lidar_integrate(handle, range_image, poses, valid_range, elevation_range) -> None:
    """PORT: lasers/laser_context._build_and_integrate."""
    import numpy as np
    import torch
    from curobo.types import LidarObservation, Pose
    from curobo._src.types.device_cfg import DeviceCfg

    entry = _REGISTRY[handle]
    device = torch.device("cuda", torch.cuda.current_device())
    _ = DeviceCfg(device=device, dtype=torch.float32)
    mapper = _mapper(entry)

    range_np = np.asarray(range_image, dtype=np.float32)
    n = int(range_np.shape[0])
    range_t = torch.tensor(range_np, dtype=torch.float32, device=device)
    rgb_t = torch.zeros((n, range_np.shape[-2], range_np.shape[-1], 3),
                        dtype=torch.uint8, device=device)
    poses_np = np.asarray(poses, dtype=np.float32).reshape(n, 7)
    pose = Pose(
        position=torch.tensor(poses_np[:, 0:3], dtype=torch.float32, device=device),
        quaternion=torch.tensor(poses_np[:, 3:7], dtype=torch.float32, device=device),
        normalize_rotation=True)
    valid_t = torch.tensor(np.asarray(valid_range, dtype=np.float32),
                           dtype=torch.float32, device=device)
    elev_t = torch.tensor(np.asarray(elevation_range, dtype=np.float32),
                          dtype=torch.float32, device=device)
    mapper.integrate(LidarObservation(
        range_image=range_t, rgb_image=rgb_t, pose=pose,
        valid_range_m=valid_t, elevation_range_rad=elev_t))
def _quat_to_matrix_wxyz(q):
    """3x3 rotation from a [qw,qx,qy,qz] quaternion (torch).

    ``q`` may be a tensor OR a plain sequence of floats (the caller passes a
    list); coerce so ``torch.stack`` never sees bare floats.
    """
    import torch

    t = torch.as_tensor(q, dtype=torch.float32)
    w, x, y, z = t[0], t[1], t[2], t[3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ])
def _pose_matrix(pos, quat_wxyz):
    """4x4 homogeneous matrix from position + [qw,qx,qy,qz] (torch)."""
    import torch

    device = pos.device
    t = torch.eye(4, dtype=torch.float32, device=device)
    t[:3, :3] = _quat_to_matrix_wxyz(quat_wxyz)
    t[:3, 3] = pos
    return t
def _inside_shape(points_base, mask, margin):
    """Analytic inside test in the mask's local frame (cuboid/sphere/cylinder/capsule).

    PORT: camera_robot_segmentation_strategy._inside_shape. MESH masks are
    skipped (the Python warns); the mask frame is assumed to be the base frame.
    """
    import numpy as np
    import torch

    device = points_base.device
    kind = int(mask.get("type", 0))
    dims = [float(d) for d in mask.get("dims", [0.0, 0.0, 0.0])]
    if kind == 4:
        return None   # MESH: skipped
    pos = torch.tensor(np.asarray(mask.get("pos", [0, 0, 0]), dtype=np.float32), device=device)
    quat = mask.get("quat", [0.0, 0.0, 0.0, 1.0])   # scipy [x,y,z,w]
    q_wxyz = [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]
    t = _pose_matrix(pos, q_wxyz)
    homog = torch.cat(
        [points_base, torch.ones((points_base.shape[0], 1), device=device)], dim=1)
    local = (homog @ torch.inverse(t).T)[:, :3]
    if kind == 0:      # CUBOID: half-extents
        hx, hy, hz = dims[0] / 2.0 + margin, dims[1] / 2.0 + margin, dims[2] / 2.0 + margin
        return (local[:, 0].abs() <= hx) & (local[:, 1].abs() <= hy) & (local[:, 2].abs() <= hz)
    if kind == 1:      # SPHERE
        return torch.norm(local, dim=1) <= dims[0] + margin
    if kind == 2:      # CYLINDER: radius, height (z axis)
        r = torch.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2)
        return (r <= dims[0] + margin) & (local[:, 2].abs() <= dims[1] / 2.0 + margin)
    if kind == 3:      # CAPSULE: radius, height (segment [0,0,0]..[0,0,h])
        dz = torch.clamp(local[:, 2], 0.0, dims[1])
        d = torch.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2 + (local[:, 2] - dz) ** 2)
        return d <= dims[0] + margin
    return None
def robot_segment(handle, depth, intrinsics, cam_pose, q, masks_json):
    """PORT: camera_robot_segmentation_strategy._mask_depth_image."""
    import json
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    kin = entry["kin"]
    device = torch.device("cuda", torch.cuda.current_device())

    depth_t = torch.tensor(np.asarray(depth, dtype=np.float32), device=device)
    h, w = int(depth_t.shape[0]), int(depth_t.shape[1])
    fx, fy, cx, cy = [float(v) for v in np.asarray(intrinsics).reshape(-1)[:4]]

    v, u = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    valid = (depth_t > 0) & ~torch.isnan(depth_t) & ~torch.isinf(depth_t)
    uu = u[valid]
    vv = v[valid]
    z = depth_t[valid]
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    pts = torch.stack([x, y, z], dim=1)

    # camera pose -> base-frame points.
    cam = np.asarray(cam_pose, dtype=np.float32)
    t_cam = _pose_matrix(
        torch.tensor(cam[0:3], dtype=torch.float32, device=device),
        [float(cam[3]), float(cam[4]), float(cam[5]), float(cam[6])])
    homog = torch.cat([pts, torch.ones((pts.shape[0], 1), device=device)], dim=1)
    base = (homog @ t_cam.T)[:, :3]

    # robot spheres (FK) -> keep points farther than threshold from every sphere.
    q_t = torch.tensor([list(np.asarray(q, dtype=np.float32))],
                       dtype=torch.float32, device=device)
    js = JointState.from_position(q_t, joint_names=kin.joint_names)
    spheres = kin.compute_kinematics(js).robot_spheres.reshape(-1, 4)
    threshold = float(entry.get("seg_distance_threshold", 0.05))
    # Distance to the nearest robot sphere. The naive
    # `torch.norm(base[:, None] - spheres[None, :, :3], dim=2)` materialises an
    # [N, S, 3] float32 tensor, and on KATE that is not a rounding error: the
    # collision model carries 592 spheres over 68 links (the simulated chassis
    # wheel hubs, top_plate, screen, router, sonars -- not just an arm like the
    # franka/ur10e demos), so a single 800x600 frame asks for ~886 MiB here. On
    # this 7.7 GiB card, shared with the GPU Gazebo/MoveIt, that allocation
    # fails, the frame is dropped, the masked depth stops and the Mapper freezes.
    # Two guards bound it:
    #   1. only points inside the robot's AABB (+threshold) can be within
    #      `threshold` of a sphere -- everything else is kept with no distance
    #      computation at all (the robot covers a small part of the view);
    #   2. the rest are evaluated in chunks, so no [N, S, 3] intermediate exists.
    sph = spheres[:, :3].contiguous()
    sph_r = spheres[:, 3].contiguous()
    n_pts = int(base.shape[0])
    keep = torch.ones(n_pts, dtype=torch.bool, device=device)
    if n_pts > 0 and sph.numel() > 0:
        reach = float(sph_r.max()) + threshold
        lo = sph.min(dim=0).values - reach
        hi = sph.max(dim=0).values + reach
        near = ((base >= lo) & (base <= hi)).all(dim=1).nonzero(as_tuple=True)[0]
        chunk = 8192
        for start in range(0, int(near.numel()), chunk):
            idx = near[start:start + chunk]
            b = base.index_select(0, idx)                               # [c, 3]
            d = torch.norm(b.unsqueeze(1) - sph.unsqueeze(0), dim=2) - sph_r.unsqueeze(0)
            keep[idx] = d.min(dim=1).values > threshold
    del sph, sph_r

    margin = float(entry.get("seg_mask_margin", 0.0))
    for mask in json.loads(masks_json or "[]"):
        inside = _inside_shape(base, mask, margin)
        if inside is not None:
            keep = keep & ~inside

    keep_img = torch.zeros(h * w, dtype=torch.bool, device=device)
    keep_img[(vv.long() * w) + uu.long()] = keep
    keep_img = keep_img.view(h, w)
    masked = torch.where(keep_img, depth_t, torch.zeros_like(depth_t))
    return masked.cpu().numpy().astype(np.float32)

from .runtime import _REGISTRY
from .scene import _to_scene
from .mapper import _mapper, voxel_grid
