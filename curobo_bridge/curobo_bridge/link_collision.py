# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

def set_link_collision(handle: int, link_names, enabled: bool):
    """Enable/disable link collision spheres (PORT: robot_model_manager.set_link_collision)."""
    entry = _REGISTRY[handle]
    kc = entry["kin"].kinematics_config
    applied, unknown = [], []
    for link in list(link_names):
        if link not in kc.link_name_to_idx_map:
            unknown.append(link)
        else:
            if enabled:
                kc.enable_link_spheres(link)
            else:
                kc.disable_link_spheres(link)
            applied.append(link)
    return {"applied": applied, "unknown": unknown}
def _solver_bound_scene(scene, mode="cuboid"):
    """PORT: obstacle_manager.collision_world_scene.

    Primitives converted to SOLVER-SUPPORTED collision types (sphere/cylinder/
    capsule -> OBB cuboid via get_cuboid()), voxel layer retained. cuRobo
    silently DROPS a cylinder it cannot collide, so pushing the raw scene left
    the solvers blind to exactly those obstacles - the robometrics tabletop
    scenes carry 4 cylinders, and the ROS leg then "solved" problems that
    collide in the native leg (7 success mismatches on the full dataset).
    """
    return _collision_supported(scene, mode)

from .runtime import _REGISTRY
from .voxel import _collision_supported
