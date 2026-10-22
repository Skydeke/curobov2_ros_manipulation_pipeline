# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

MAPPER_DEFAULTS = {
    "extent_xyz": (2.56, 2.56, 2.56),
    "voxel_size": 0.02,
    "esdf_voxel_size": 0.05,
    "grid_center": (0.0, 0.0, 0.0),
    "image_height": 720,
    "image_width": 1280,
    "depth_min": 0.1,
    "depth_max": 5.0,
    "truncation_distance": 0.04,
    "minimum_tsdf_weight": 0.1,
    "block_size": 8,
    "seeding_method": "gather",
    "max_support_pixels_per_block_camera": 8,
    "decay_factor": 1.0,
}
def configure_mapper(handle: int, cfg_json: str) -> None:
    """Build the Mapper (PORT: obstacle_manager._build_mapper)."""
    import json

    entry = _REGISTRY[handle]
    cfg = dict(MAPPER_DEFAULTS)
    cfg.update(json.loads(cfg_json) if cfg_json else {})
    entry["mapper_cfg"] = cfg
    entry["mapper"] = None   # built lazily on first use
def _mapper(entry):
    mapper = entry.get("mapper")
    if mapper is None:
        import torch
        from curobo.perception import Mapper, MapperCfg

        cfg = entry.get("mapper_cfg") or dict(MAPPER_DEFAULTS)
        entry["mapper_cfg"] = cfg
        mapper = Mapper(MapperCfg(
            extent_meters_xyz=tuple(cfg["extent_xyz"]),
            extent_esdf_meters_xyz=tuple(cfg["extent_xyz"]),
            voxel_size=cfg["voxel_size"],
            esdf_voxel_size=cfg["esdf_voxel_size"],
            grid_center=torch.tensor(cfg["grid_center"], dtype=torch.float32),
            image_height=cfg["image_height"],
            image_width=cfg["image_width"],
            depth_minimum_distance=cfg["depth_min"],
            depth_maximum_distance=cfg["depth_max"],
            truncation_distance=cfg["truncation_distance"],
            minimum_tsdf_weight=cfg["minimum_tsdf_weight"],
            block_size=cfg["block_size"],
            seeding_method=cfg["seeding_method"],
            max_support_pixels_per_block_camera=cfg["max_support_pixels_per_block_camera"],
            decay_factor=cfg["decay_factor"],
            num_cameras=1,
        ))
        entry["mapper"] = mapper
    return mapper
def _inspect_esdf(entry, vg) -> bool:
    """PORT: obstacle_manager._inspect_esdf - reject a DEFINITELY corrupt ESDF
    (non-finite values, or a voxel count beyond the configured grid) before it
    reaches the collision kernels; such a grid makes the kernel read out of
    bounds (GPU illegal access). A merely saturated grid is allowed through.
    """
    import numpy as np
    import torch

    try:
        ft = getattr(vg, "feature_tensor", None)
        if ft is None or ft.numel() == 0:
            return True
        n = int(ft.numel())
        if int((~torch.isfinite(ft)).sum().item()) > 0:
            print(
                f"[curobo_bridge] perception ESDF has non-finite values "
                f"({n} voxels) - rejecting to avoid a collision-kernel crash",
                flush=True)
            return False
        cfg = entry.get("mapper_cfg") or dict(MAPPER_DEFAULTS)
        vsize = float(getattr(vg, "voxel_size", 0.0)) or cfg["esdf_voxel_size"]
        expected = int(np.prod(np.ceil(
            np.asarray(cfg["extent_xyz"], dtype=np.float64) / vsize)))
        if expected and n > int(expected * 1.01) + 1:
            print(
                f"[curobo_bridge] perception ESDF voxel count {n} exceeds "
                f"expected {expected} (voxel_size={vsize}) - rejecting",
                flush=True)
            return False
        return True
    except Exception:
        return True
def esdf_refresh(handle: int) -> bool:
    """PORT: obstacle_manager.refresh_esdf - compute the ESDF and push it into
    the scene's voxel layer. False when the mapper produced nothing."""
    entry = _REGISTRY[handle]
    mapper = _mapper(entry)
    try:
        voxel_grid_result = mapper.compute_esdf()
    except Exception:
        return False
    if voxel_grid_result is None:
        return False
    if not _inspect_esdf(entry, voxel_grid_result):
        return False
    scene = entry.get("scene")
    if scene is None:
        # Defensive: never drop a successfully computed ESDF for lack of a scene
        # (see motion_planner_build - the entry owns a live scene from build on).
        from curobo.scene import Scene

        scene = Scene()
        entry["scene"] = scene
    scene.voxel = [voxel_grid_result]
    entry["scene_collision"] = None
    return True
def clear_voxel_map(handle: int) -> int:
    """PORT: obstacle_manager.clear_dynamic_voxels - clear the full mapper extent."""
    import numpy as np
    import torch

    entry = _REGISTRY[handle]
    cfg = entry.get("mapper_cfg") or dict(MAPPER_DEFAULTS)
    center = np.asarray(cfg["grid_center"], dtype=np.float32)
    half = np.asarray(cfg["extent_xyz"], dtype=np.float32) / 2.0
    mapper = _mapper(entry)
    cleared = mapper.clear_region(
        torch.from_numpy(center - half), torch.from_numpy(center + half))
    return int(cleared) if cleared is not None else 0
def clear_voxel_region(handle: int, bounds_min, bounds_max) -> int:
    """PORT: obstacle_manager.clear_voxel_region."""
    import numpy as np
    import torch

    entry = _REGISTRY[handle]
    mapper = _mapper(entry)
    cleared = mapper.clear_region(
        torch.as_tensor(np.asarray(bounds_min, dtype=np.float32)),
        torch.as_tensor(np.asarray(bounds_max, dtype=np.float32)))
    return int(cleared) if cleared is not None else 0
def voxel_grid(handle: int) -> dict:
    """PORT: obstacle_manager._compute_dense_voxel_grid - dense occupied grid."""
    import numpy as np

    entry = _REGISTRY[handle]
    cfg = entry.get("mapper_cfg") or dict(MAPPER_DEFAULTS)
    mapper = _mapper(entry)
    centers = None
    try:
        occupied = mapper.integrator.extract_occupied_voxels(surface_only=False)
        if occupied is not None:
            centers = occupied.centers.detach().cpu().numpy()
    except Exception:
        centers = None

    extent = np.asarray(cfg["extent_xyz"], dtype=np.float64)
    vsize = float(cfg["esdf_voxel_size"])
    center = np.asarray(cfg["grid_center"], dtype=np.float64)
    shape = tuple(int(np.ceil(e / vsize)) for e in extent)
    origin = center - extent / 2.0
    grid = np.zeros(shape, dtype=np.uint8)
    if centers is not None and centers.size:
        idx = np.floor((centers - origin) / vsize).astype(np.int64)
        valid = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
        idx = idx[valid]
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    # Analytic primitives (AddObject obstacles) - GPU rasterization, exactly the
    # method the oracle uses (same collision checker as the planner).
    try:
        _rasterize_primitives_gpu(entry, grid, origin, vsize)
    except Exception as exc:
        print(f"[curobo_bridge] primitive rasterization failed: {exc}", flush=True)
    return {
        "shape": list(shape),
        "data": grid.reshape(-1),
        "resolution": vsize,
        "origin": origin.tolist(),
    }
def warmup_mapper(handle: int, iterations: int = 2) -> None:
    """Force the Mapper's Warp kernels to compile AND LOAD now. Warp JIT-compiles one module per kernel specialization and loads it lazily
    with ``cuModuleLoadData``. A load that happens while the plan lane is
    capturing/replaying a CUDA graph fails with
    ``Warp CUDA error 716: misaligned address (wp_cuda_load_module)`` and poisons
    the CUDA context - the ``recycle_empty_blocks_kernel_bsN_<hash>`` failure
    seen during depth integration."""
    import numpy as np

    entry = _REGISTRY[handle]
    cfg = entry.get("mapper_cfg") or dict(MAPPER_DEFAULTS)
    h = int(cfg.get("image_height", 600))
    w = int(cfg.get("image_width", 800))
    depth = np.full((h, w), 0.8, dtype="float32")
    intr = [[float(w), 0.0, w / 2.0], [0.0, float(h), h / 2.0], [0.0, 0.0, 1.0]]
    cam = [0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for _ in range(int(iterations)):
        try:
            depth_integrate(handle, depth, intr, cam)
        except Exception as exc:
            # Keep the failure loud but do not block startup: the mapper will
            # simply load the module on its first real frame.
            print(
                f"[curobo_bridge] mapper wakeup integrate failed ({exc!r}); the "
                f"mapper's Warp modules will still load lazily", flush=True)
            return
    try:
        clear_voxel_map(handle)
        esdf_refresh(handle)
    except Exception:
        pass

from .runtime import _REGISTRY
from .voxel import _rasterize_primitives_gpu, depth_integrate
