# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

_LAST_WORLD_PUSH_ERROR_T = 0.0
def _log_world_push_failure(reason: str, interval_s: float = 5.0) -> None:
    """Report a world-push failure, throttled.

    The plan path's freshness gate keeps retrying while the push fails, so an
    unthrottled message would fire on every plan. Always report the FIRST
    failure, then at most once per `interval_s`.
    """
    global _LAST_WORLD_PUSH_ERROR_T
    import time

    now = time.monotonic()
    if now - _LAST_WORLD_PUSH_ERROR_T < interval_s:
        return
    _LAST_WORLD_PUSH_ERROR_T = now
    print(f"[curobo_bridge] world_push FAILED: {reason}", flush=True)
def world_push(handle: int) -> bool:
    """Push the shared scene (incl. the perception voxel layer) into every
    initialized solver's collision model.

    PORT: unified_planner_node.update_all_solvers_world - without it the planner
    would not collide against camera-observed voxels.
    """
    entry = _REGISTRY[handle]
    scene = entry.get("scene")
    if scene is None:
        return False
    # PORT: unified_planner_node.update_all_solvers_world's
    # `collision_cache["voxel"] is None` branch - degrade to analytic primitives
    # (no camera obstacles) rather than raise on every world update.
    if entry.get("voxel_cache_disabled"):
        import copy

        scene = copy.copy(scene)     # shallow: share the primitive buckets
        scene.voxel = None
    pushed = False
    # PORT: collision_world_scene() - the SOLVERS get the converted scene.
    solver_scene = _solver_bound_scene(scene, _obstacle_mode(entry))
    mp = entry.get("motion_planner")
    if mp is not None:
        try:
            mp.update_world(solver_scene)
            pushed = True
        except Exception as exc:
            # NEVER swallow this. A failed update leaves the solver on its
            # PREVIOUS world - possibly with no camera voxels at all - while the
            # perception map keeps publishing the obstacles. Silently continuing
            # is how a plan ends up going straight through the wall. Report it and
            # let the caller refuse to plan (throttled: the caller retries every
            # plan while the failure persists).
            _log_world_push_failure(
                f"motion planner update_world raised {exc!r}; the solver keeps "
                f"its previous world and camera voxels may be missing")
        if pushed:
            # Best-effort verification that the voxel layer actually landed in the
            # solver's collision model (attributes vary by cuRobo build; only act
            # when we can positively see an empty voxel store). Accessors can
            # RAISE (properties on the planner do), so never let a probe turn a
            # successful push into a crash.
            try:
                has_voxel_in = bool(getattr(solver_scene, "voxel", None))
                data = getattr(getattr(mp, "scene_collision_checker", None), "data", None)
                if has_voxel_in and data is not None and \
                        getattr(data, "voxels", None) is None:
                    _log_world_push_failure(
                        "the scene carried a voxel layer but the motion planner "
                        "has no voxel cache - camera obstacles are NOT being "
                        "collided against")
                    pushed = False
            except Exception:
                pass
    ik = entry.get("ik")
    if ik is not None:
        try:
            ik.update_world(solver_scene)
        except Exception:
            pass
    # PORT: fk_services._push_world - the FK collision validator is a THIRD
    # collision model and needs the same world. It has no voxel cache at all, so
    # it gets the primitives-only scene (a voxel-carrying one raises "Voxel cache
    # not initialized"); a failed update disables it rather than aborting the
    # push, exactly as the reference does.
    fk_checker = entry.get("checker")
    if fk_checker not in (None, False):
        try:
            fk_checker.update_world(_collision_supported(
                _primitives_only_scene(entry), _obstacle_mode(entry)))
        except Exception as exc:
            # PORT: fk_services._push_world's error path - say why, then drop the
            # validator (advisory: _validate reports all-True without it) instead
            # of letting a failed refresh abort the push.
            print(
                f"[curobo_bridge] FK collision validator world update failed ({exc}); "
                "disabling it - poses_valid will report all True", flush=True)
            entry["checker"] = False
    entry["scene_collision"] = None
    # Every update_world above clears + re-adds all obstacles with enable=1,
    # which wipes the flag set for an attached obstacle - re-assert the disabled
    # set now or the static copy collides with the attached spheres again.
    # (PORT: unified_planner_node.update_all_solvers_world tail.)
    _reapply_attached_disables(entry)
    if pushed:
        # The captured trajopt/rollout CUDA graphs bake in the collision-scene
        # buffers that load_collision_model just rewrote. Replaying one against
        # the new world faults ("CUDA error: an illegal memory access") - exactly
        # the crash the perception refresh triggered on the plan right after it.
        # Leave no mode recorded so the next plan re-captures its graphs under
        # the exclusive GPU barrier (see prepare_solve_mode).
        entry["_solve_mode"] = None
    return pushed

from .runtime import _REGISTRY
from .ik import ik
from .scene import _reapply_attached_disables
from .link_collision import _solver_bound_scene
from .voxel import _collision_supported, _obstacle_mode, _primitives_only_scene
