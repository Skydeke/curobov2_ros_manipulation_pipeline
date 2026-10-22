# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

def _scene_collision(entry, dcfg):
    """Lazily build the SceneCollision the distance query runs against
    (PORT: _resolve_scene_checker / _sphere_collision_snapshot).

    Built from the current scene on demand; every scene mutation invalidates it
    (``entry["scene_collision"] = None``), so a rebuild picks up the new
    obstacles. Do NOT ``load_collision_model`` on every call - the 2 Hz
    collision-spheres publisher queries this ~240x per servo run and re-syncing
    each time corrupted the host heap in the M5 soak.
    """
    from curobo._src.geom.collision.collision_scene import (
        SceneCollision, SceneCollisionCfg)
    from curobo.scene import Scene

    scene = entry.get("scene")
    if scene is None:
        # NB: `or Scene()` is WRONG here - curobo's Scene(==SceneCfg) defines
        # __len__ and no __bool__, so it is FALSY even when it holds cuboids,
        # and `entry.get("scene") or Scene()` silently swaps in an EMPTY scene.
        scene = Scene()
    checker = entry.get("scene_collision")
    if checker is None:
        checker = SceneCollision.from_config(
            SceneCollisionCfg(device_cfg=dcfg, scene_model=scene))
        entry["scene_collision"] = checker
        # Built lazily on the first distance query, which is usually AFTER an
        # attach, so re-assert the attached-object disables on the fresh buffer.
        _reapply_attached_disables(entry)
    return checker
def _to_scene(obj) -> "object":
    """Build cuRobo scene primitives from a SceneObject-shaped dict.

    PORT: obstacle_manager._append_spec.
    """
    from curobo.scene import Cuboid, Capsule, Cylinder, Sphere, Mesh

    # cuRobo obstacles carry a FLAT [x,y,z,qw,qx,qy,qz] pose, not a batched
    # Pose: Obstacle.get_transform_matrix slices self.pose[:3] / [3:], so a
    # batched Pose makes fit_spheres/trimesh blow up with
    # "setting an array element with a sequence" (PORT: obstacle_manager._append_spec).
    pose = [float(v) for v in obj["pose"][0:7]]
    dims = list(obj["dims"])
    kind = int(obj["type"])
    if kind == 0:      # CUBOID
        return Cuboid(name=obj["name"], pose=pose, dims=dims)
    if kind == 1:      # SPHERE
        return Sphere(name=obj["name"], pose=pose, radius=dims[0])
    if kind == 2:      # CAPSULE
        return Capsule(
            name=obj["name"], pose=pose, base=[0, 0, 0], tip=[0, 0, dims[1]],
            radius=dims[0])
    if kind == 3:      # CYLINDER
        return Cylinder(name=obj["name"], pose=pose, radius=dims[0], height=dims[1])
    if kind == 4:      # MESH
        if obj.get("mesh_file_path"):
            return Mesh(
                name=obj["name"], pose=pose, file_path=obj["mesh_file_path"], scale=dims)
        return Mesh(
            name=obj["name"], pose=pose,
            vertices=[list(v) for v in obj.get("vertices", [])],
            faces=list(obj.get("triangles", [])), scale=dims)
    raise ValueError(f'Object type "{kind}" not recognized')
def scene_set(handle: int, objects_json: str, clear_first: bool):
    """Replace/append scene objects; returns the scene version (PORT: add/set_objects)."""
    import json

    from curobo.scene import Scene

    entry = _REGISTRY[handle]
    specs = json.loads(objects_json)
    previous = entry.get("scene")
    scene = Scene() if clear_first else previous
    if scene is None:
        scene = Scene()
    # The perception voxel layer is NOT an analytic object. Rebuilding the
    # primitive scene for a set_objects/clear must not drop the camera obstacles
    # the last ESDF refresh installed on the previous scene: an empty
    # `set_objects` (e.g. the MoveIt side's syncPlanningScene) would otherwise
    # silently blind the solver to every camera-observed voxel.
    if clear_first and previous is not None:
        carried = getattr(previous, "voxel", None)
        if carried:
            scene.voxel = carried
    for spec in specs:
        primitive = _to_scene(spec)
        attr = {0: "cuboid", 1: "sphere", 2: "capsule", 3: "cylinder", 4: "mesh"}[
            int(spec["type"])]
        bucket = getattr(scene, attr)
        if bucket is None:
            setattr(scene, attr, [primitive])
        else:
            bucket.append(primitive)
    entry["scene"] = scene
    entry["scene_collision"] = None   # rebuild the distance checker on next query
    entry["scene_version"] = entry.get("scene_version", 0) + 1
    entry["scene_names"] = [s["name"] for s in specs] if clear_first else (
        entry.get("scene_names", []) + [s["name"] for s in specs])
    entry["scene_specs"] = list(specs) if clear_first else (
        entry.get("scene_specs", []) + list(specs))
    return entry["scene_version"]
def scene_get(handle: int) -> str:
    """The current scene objects as JSON (name/type/dims), for get_scene_objects."""
    import json

    entry = _REGISTRY[handle]
    return json.dumps({
        "version": entry.get("scene_version", 0),
        "names": entry.get("scene_names", []),
    }, sort_keys=True)
ATTACH_LINK = "attached_object"
def _attachment_manager(entry):
    """PORT: attachment_services._attachment_manager.

    Prefers the SHARED MotionPlanner's AttachmentManager (cuRobo composes it in
    the trajopt solver core): that manager's kinematics ARE the solver's
    kinematics, so the fitted payload spheres land in the model the solver
    collides with, and its `_scene_collision` is the very checker the name-based
    disable acts on. Falls back to a standalone manager over the robot-model
    kinematics when no planner is built yet (nothing to attach into)."""
    planner = entry.get("motion_planner")
    if planner is not None:
        try:
            am = getattr(planner, "attachment_manager", None)  # property may raise
        except Exception:
            am = None
        if am is None:
            try:
                am = planner.trajopt_solver.core.attachment_manager
            except Exception:
                am = None
        if am is not None:
            return am
    am = entry.get("attach_manager")
    if am is None:
        import torch

        from curobo._src.collision.attachment_manager import AttachmentManager
        from curobo._src.types.device_cfg import DeviceCfg

        am = AttachmentManager(
            kinematics=entry["kin"],
            scene_collision=None,
            device_cfg=DeviceCfg(device=torch.device("cuda", torch.cuda.current_device()),
                            dtype=torch.float32),
        )
        entry["attach_manager"] = am
    return am
def _cuboid_world_aabb(spec):
    """World-aligned AABB of a cuboid scene spec.

    PORT: attachment_services._obstacle_world_aabb - transforms the 8 box
    corners by the obstacle's pose (position + wxyz quaternion) and adds the
    same 1.5 cm margin so voxels just outside the faces are cleared too.
    Returns ``(None, None)`` for a non-cuboid spec.
    """
    import numpy as np

    if int(spec.get("type", 0)) != 0:
        return None, None
    dims = np.asarray(spec.get("dims", []), dtype=np.float32)
    pose = np.asarray(spec.get("pose", []), dtype=np.float32)
    if dims.size < 3 or pose.size < 7:
        return None, None
    rot = _rot_from_quat_wxyz(pose[3:7]).astype(np.float32)
    corners = np.array([
        [sx, sy, sz]
        for sx in (-0.5, 0.5)
        for sy in (-0.5, 0.5)
        for sz in (-0.5, 0.5)
    ], dtype=np.float32) * dims[:3]
    world = corners @ rot.T + pose[:3]
    margin = 0.015  # 1.5 cm clearance beyond the faces
    return world.min(axis=0) - margin, world.max(axis=0) + margin
def _solver_scene_checkers(entry):
    """Every initialized solver's scene collision checker.

    PORT: obstacle_manager.reapply_attached_disables' checker gathering
    (motion_planner / mpc / ik / fk / retarget), narrowed to the solvers this
    registry actually builds. Best-effort: unbuilt or checker-less solvers are
    skipped so a missing lane can never break an attach or a world push.
    """
    checkers = []
    seen = set()

    def _add(obj, attr="scene_collision_checker"):
        checker = getattr(obj, attr, None) if obj is not None else None
        if checker is None or checker is False or id(checker) in seen:
            return
        seen.add(id(checker))
        checkers.append(checker)

    _add(entry.get("motion_planner"))
    _add(entry.get("mpc"))
    _add(entry.get("mpc_ik_solver"))
    _add(entry.get("ik"))
    retargeter = entry.get("retargeter")
    if retargeter is not None:
        _add(getattr(retargeter, "solver", None))
        _add(getattr(getattr(retargeter, "solver", None), "_global_ik_solver", None))
    # FK services' advisory checker (Python: node.fk_services._collision_checker)
    # is the checker object itself, not an attribute of one.
    fk_checker = entry.get("checker")
    if fk_checker not in (None, False) and id(fk_checker) not in seen:
        seen.add(id(fk_checker))
        checkers.append(fk_checker)
    # The DIAGNOSTIC scene checker (`_scene_collision` - the one the collision
    # feedback and the collision-spheres publisher read) is a separate
    # SceneCollision with its own `enable` buffer. Leaving it out made the
    # printed "collision feedback" list the attached object's own world copy as
    # a contact (`attached_object -> object_0`): the planner had it disabled but
    # the diagnostics never did, so the report overstated the start collision.
    diag_checker = entry.get("scene_collision")
    if diag_checker not in (None, False) and id(diag_checker) not in seen:
        seen.add(id(diag_checker))
        checkers.append(diag_checker)
    return checkers
def _reapply_attached_disables(entry) -> None:
    """Re-assert the attached obstacles' disabled flag on every solver. PORT: obstacle_manager.reapply_attached_disables. cuRobo's per-obstacle
    ``enable`` flag lives in GPU buffers and a world push resets it to 1
    (``SceneData.load_from_scene_cfg`` clears + re-adds with enable=1), so the
    flag must be re-armed after every push: the grasped obstacle stays
    REGISTERED in the solver scenes while attached, but collision-inert."""
    names = entry.get("attached_disabled") or []
    if not names:
        return
    for name in tuple(names):
        for checker in _solver_scene_checkers(entry):
            try:
                if not checker.check_obstacle_exists(name):
                    continue
                checker.enable_obstacle(name, enable=False)
            except Exception:
                continue
def _clear_attached_voxels(handle: int, spec) -> bool:
    """Clear the depth-derived voxels spanning the attached obstacle's world
    AABB, then refresh the ESDF and push the world to the solvers. PORT: attachment_services._clear_attached_voxels. Disabling the obstacle BY
    NAME only disables its cuboid collision buffer - the perception voxel layer
    still holds the grasped object's voxels at its world position, so the
    freshly attached spheres collide with the object's own leftover voxels."""
    entry = _REGISTRY[handle]
    if entry.get("mapper") is None:
        return False
    bounds_min, bounds_max = _cuboid_world_aabb(spec)
    if bounds_min is None:
        return False
    try:
        clear_voxel_region(handle, bounds_min.tolist(), bounds_max.tolist())
        if esdf_refresh(handle):
            world_push(handle)
            return True
    except Exception:
        return False
    return False
def attach_object(handle: int, name: str, q) -> None:
    """Fit + attach `name` to the attach link at joint config `q`.

    PORT: attachment_services.attach / _fit_and_attach (the local sphere write
    that avoids cuRobo's single-sphere `update` crash).
    """
    import numpy as np
    import torch
    from curobo._src.geom.sphere_fit.types import SphereFitType
    from curobo.types import JointState, Pose

    entry = _REGISTRY[handle]
    spec = next((s for s in entry.get("scene_specs", []) if s["name"] == name), None)
    if spec is None:
        raise ValueError(f"Obstacle '{name}' not found in the scene")

    device = torch.device("cuda", torch.cuda.current_device())
    obstacle = _to_scene(spec)
    am = _attachment_manager(entry)
    # PORT: _fit_and_attach writes into the ATTACHMENT MANAGER's kinematics -
    # once the planner exists that is the SOLVER's own model (AttachmentManager
    # composes the solver's Kinematics), so the link lookup and the fit FK must
    # both use that kin rather than the standalone robot-model one, or the
    # payload spheres land somewhere no solver collides with.
    kin = getattr(am, "_kinematics", None) or entry["kin"]
    kc = kin.kinematics_config
    if ATTACH_LINK not in kc.link_name_to_idx_map:
        raise ValueError(
            f"Robot YAML does not declare the '{ATTACH_LINK}' link — attach is "
            "unavailable for this robot (the robot YAML must declare it).")
    kparams = am.kinematics_params
    link_idx = kparams.get_sphere_index_from_link_name(ATTACH_LINK)
    n_slots = int(link_idx.shape[0])

    # PORT: attachment_services.attach - cap the fit at the link's slot budget,
    # but warn when the payload needs more (a silently truncated model is
    # coarser than the real shape the caller plans against).
    from curobo.sphere_fit import estimate_sphere_count
    try:
        n_needed = int(estimate_sphere_count(obstacle.get_trimesh_mesh(transform_with_pose=True)))
    except Exception:
        n_needed = 0
    if n_needed > n_slots:
        print(
            f"[curobo_bridge] Attached object '{name}' needs about {n_needed} "
            f"collision spheres but link '{ATTACH_LINK}' allocates only {n_slots}: "
            "its collision model is a coarse approximation that may not cover "
            f"corners or protrusions. Raise 'extra_collision_spheres.{ATTACH_LINK}' "
            "in the robot YAML before planning with this payload on real hardware.",
            flush=True)
    n_fit = min(n_needed, n_slots) if n_needed > 0 else n_slots

    sphere_tensor = am.fit_spheres(
        [obstacle], num_spheres=n_fit, surface_radius=0.002,
        sphere_fit_type=SphereFitType.SURFACE)
    centers = sphere_tensor[:, :3].contiguous()
    radii = sphere_tensor[:, 3].unsqueeze(-1)
    # Payload world envelope: the fitted spheres are world-frame (the fit bakes
    # the obstacle's pose in), so this is where the attached collision model
    # actually sits. If the low edge is below the support plane's top the start
    # state is rejected by every planner ("Start or End state in collision"),
    # which is what a too-small object_lift looks like.
    if centers.numel() > 0:
        print(
            f"[curobo_bridge] Attached '{name}': {int(centers.shape[0])} spheres, "
            f"world z [{float(centers[:, 2].min()) - float(radii.min()):.4f}, "
            f"{float(centers[:, 2].max()) + float(radii.max()):.4f}] m", flush=True)

    q_t = torch.tensor([list(np.asarray(q, dtype=np.float32))],
                       dtype=torch.float32, device=device)
    joint_state = JointState.from_position(q_t, joint_names=kin.joint_names)
    fk_result = kin.compute_kinematics(joint_state)
    if fk_result.tool_poses is None:
        raise RuntimeError("FK result has no tool_poses; cannot resolve EE for attachment")
    ee_link = kin.tool_frames[0]
    obj_to_link = fk_result.tool_poses.get_link_pose(ee_link).inverse()

    n_fit = int(centers.shape[0])
    if n_fit > n_slots:
        raise RuntimeError(
            f"Fitted {n_fit} spheres but link '{ATTACH_LINK}' only has {n_slots} sphere slots")
    padding = None
    if n_fit < n_slots:
        padding = torch.zeros(n_slots - n_fit, 4,
                              device=sphere_tensor.device, dtype=sphere_tensor.dtype)
        padding[:, 3] = -100.0

    num_envs = q_t.shape[0]
    for i in range(num_envs):
        env_pose = Pose(
            position=obj_to_link.position[i:i + 1],
            quaternion=obj_to_link.quaternion[i:i + 1])
        link_centers = env_pose.transform_points(centers).reshape(-1, 3)
        env_spheres = torch.cat([link_centers, radii], dim=-1)
        if padding is not None:
            env_spheres = torch.cat([env_spheres, padding], dim=0)
        kparams.link_spheres[i, link_idx, :] = env_spheres
    am._attached_link_name = ATTACH_LINK
    # PORT: _fit_and_attach's cuRobo-native disable - records the name and turns
    # the world obstacle off on the attachment manager's scene checker (the
    # solver's own checker once the manager comes from the planner). The
    # _reapply_attached_disables below sets the same flag; both are in the
    # reference, and only the env_idx form matches cuRobo's own `attach`.
    scene_collision = getattr(am, "_scene_collision", None)
    if scene_collision is not None:
        try:
            if scene_collision.check_obstacle_exists(name):
                for env_idx in range(num_envs):
                    scene_collision.enable_obstacle(
                        name, enable=False, env_idx=env_idx)
                am._disabled_obstacle_names = [name]
                am._disabled_num_envs = num_envs
        except Exception:
            pass
    entry["attached_name"] = name
    # PORT: attachment_services.attach - keep the grasped obstacle REGISTERED in
    # the solver scenes but disabled by name (re-armed after every world push via
    # _reapply_attached_disables), then drop it from the voxel-map rasterization
    # and clear its leftover depth voxels so the fitted spheres do not collide
    # with the object's own copy.
    entry["attached_disabled"] = [name]
    if not _clear_attached_voxels(handle, spec):
        # PORT: no mapper / non-cuboid - nothing to clear or refresh, but the
        # world must still be pushed ONCE so the exclusion reaches the solvers
        # and the voxel map (and a push re-enables obstacles, so the re-assert
        # below matters). The reference pushes unconditionally: inside
        # _clear_attached_voxels on the ESDF path, here on the fallback path.
        world_push(handle)
    _reapply_attached_disables(entry)
def detach_object(handle: int) -> str:
    """Detach the current payload; returns the released name ("" if none).

    PORT: attachment_services.detach.
    """
    entry = _REGISTRY[handle]
    released = entry.get("attached_name") or ""
    # _attachment_manager (not the cached standalone one) so the SAME manager the
    # attach wrote into resets `link_spheres` for the attach link - the solver's
    # kinematics once the planner exists. cuRobo's AttachmentManager.detach also
    # re-enables its recorded disabled names on its scene checker.
    am = _attachment_manager(entry)
    if am is not None:
        am.detach(ATTACH_LINK)
    entry["attached_name"] = ""
    # PORT: attachment_services.detach - restore the released obstacle to
    # collision checking. Dropping it from the disabled set, then pushing the
    # world, re-adds it with enable=1 (the push clears + re-adds every
    # obstacle, and _reapply_attached_disables now no-ops).
    entry["attached_disabled"] = []
    try:
        world_push(handle)
    except Exception:
        pass
    return released

from .runtime import _REGISTRY
from .collision import _rot_from_quat_wxyz
from .mapper import clear_voxel_region, esdf_refresh
from .world_push import world_push
