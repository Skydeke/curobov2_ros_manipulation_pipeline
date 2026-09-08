#!/usr/bin/env python3
"""
Attach/detach services for the unified planner node.

Attaches a scene obstacle to the arm's ``attached_object`` link as collision
spheres (fitted by ``sphere_fit`` via cuRobo's attachment_manager) so it moves
with the arm for subsequent open-loop transport planning, until ``detach``
releases it. The robot YAML must declare the ``attached_object`` link (see
docs/tutorials/03-collision-objects.md).

Standalone feature: independent of any particular planning pipeline. Useful
for pre-positioned objects, simulation, and tests — the caller supplies the
scene obstacle name and the joint state to fit spheres at (usually the
robot's current pose).

Registers its own services (`<node>/attach_object`, `<node>/detach_object`),
following the same self-registering pattern as ``IKServices`` / ``FKServices``.
"""

from std_srvs.srv import Trigger
from curobo.types import JointState, Pose
from curobo.sphere_fit import SphereFitType, estimate_sphere_count

from isaac_ros_cumotion_interfaces.srv import AttachObject

ATTACH_LINK = "attached_object"


class AttachmentServices:
    """Attach/detach a scene obstacle to the robot's ``attached_object`` link."""

    def __init__(self, node, config_wrapper):
        self.node = node
        self.config_wrapper = config_wrapper
        self._attached_name = None  # name of the currently attached obstacle

        name = node.get_name()
        self.attach_object_srv = node.create_service(
            AttachObject, f'{name}/attach_object', self._attach_object_callback)
        self.detach_object_srv = node.create_service(
            Trigger, f'{name}/detach_object', self._detach_object_callback)

    @property
    def attached_name(self):
        """Name of the currently attached obstacle, or None."""
        return self._attached_name

    @property
    def motion_planner(self):
        """The node's shared MotionPlanner (may be None before warmup)."""
        return self.node.motion_planner

    def kinematics(self):
        """The MotionPlanner's kinematics (carries attached-object spheres), or
        None if the planner isn't ready. Used by ros_service_manager to render
        the fitted attached-object spheres in the collision-sphere viz.
        """
        try:
            return self._attachment_manager()._kinematics
        except Exception:
            return None

    def _attachment_manager(self):
        """cuRobo AttachmentManager for the shared MotionPlanner.

        In this build ``MotionPlanner.attachment_manager`` forwards to
        ``trajopt_solver.attachment_manager``, but TrajOptSolver composes
        SolverCore and doesn't re-expose it — reach into ``.core`` instead.
        """
        mp = self.motion_planner
        am = getattr(mp, 'attachment_manager', None)  # property may raise -> None
        if am is None:
            am = mp.trajopt_solver.core.attachment_manager
        return am

    def _has_attach_link(self) -> bool:
        """Whether the loaded robot YAML declares the attach link."""
        mp = self.motion_planner
        if mp is None:
            return False
        kc = mp.kinematics.kinematics_config
        return ATTACH_LINK in kc.link_name_to_idx_map

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self, object_name: str, grasp_end_state: JointState) -> None:
        """Fit + attach the named scene obstacle to the arm at the given config.

        The obstacle is fetched from the ObstacleManager's authoritative Scene
        (so it works even if the solver's loaded scene_model is out of sync) and
        fitted via ``_fit_and_attach`` (not ``attach_from_scene``, which looks
        up by name in the solver's scene_model). cuRobo's native
        ``disable_obstacle_names`` disables the static copy in the MotionPlanner
        checker; ``reapply_attached_disables`` (run after every solver world
        push) re-asserts that flag across ALL solvers, because a world push
        clears and re-adds every obstacle with enable=1. The obstacle stays
        REGISTERED in the solver scenes while attached (so cuRobo keeps its
        collision model and disabling is a flag toggle, not a removal) and is
        dropped only from the voxel-map rasterization via exclude_obstacle, so
        its analytic voxels don't reappear in the published grid. Detach
        re-enables it. Affects the shared MotionPlanner only (open-loop
        transport).

        Sphere-fitting is a GPU operation — held under node.gpu_lock so it can't
        overlap a concurrent CUDA-graph capture or depth-camera integrate.
        """
        if not self._has_attach_link():
            raise ValueError(
                f"Robot YAML does not declare the '{ATTACH_LINK}' link — "
                "attach is unavailable for this robot (see "
                "docs/tutorials/03-collision-objects.md)."
            )
        scene = self.config_wrapper.obstacle_manager.get_scene()
        obstacle = self._find_obstacle(scene, object_name)
        if obstacle is None:
            raise ValueError(f"Obstacle '{object_name}' not found in the scene")
        with self.node.gpu_lock:
            am = self._attachment_manager()
            # cuRobo's automatic fit (num_spheres=None) sizes itself to the
            # obstacle's geometry, not to what the robot YAML allocated for
            # ATTACH_LINK, and aborts the attach when it overruns. Capping the
            # fit at the slot count makes attach always succeed -- but a cap
            # that truncates silently hands back a collision model far coarser
            # than the payload's real shape, and the caller is about to plan
            # motions against exactly that model. So cap, and say so.
            n_slots = am.kinematics_params.get_sphere_index_from_link_name(
                ATTACH_LINK).shape[0]
            n_needed = self._estimate_sphere_need(obstacle)
            if n_needed > n_slots:
                self.node.get_logger().warn(
                    f"Attached object '{object_name}' needs about {n_needed} "
                    f"collision spheres but link '{ATTACH_LINK}' allocates only "
                    f"{n_slots}: its collision model is a coarse approximation "
                    f"that may not cover corners or protrusions. Raise "
                    f"'extra_collision_spheres.{ATTACH_LINK}' in the robot YAML "
                    f"before planning with this payload on real hardware."
                )
            # Only cap when the estimate exceeds the budget; below it, let the
            # estimate stand so a small object isn't padded to the full slot
            # count. n_needed == 0 means the estimate failed -> fall back.
            n_fit = min(n_needed, n_slots) if n_needed > 0 else n_slots
            disable = [object_name]
            am_checker = getattr(self.node, 'motion_planner', None)
            am_checker = (getattr(am_checker, 'scene_collision_checker', None)
                          if am_checker is not None else None)
            if am_checker is not None:
                try:
                    if not am_checker.check_obstacle_exists(object_name):
                        disable = []
                except Exception:
                    disable = []
            self._fit_and_attach(am, obstacle, n_fit, grasp_end_state,
                                 disable or None)
            # The perception voxel layer still holds the voxels of the grasped
            # object at its world position. Disabling the obstacle BY NAME only
            # disables the cuboid collision buffer — the ESDF voxels remain, so
            # the freshly attached spheres collide with the object's own leftover
            # voxels and any motion from here starts "in collision". Clear just
            # that region from the depth-derived voxel channel, refresh the ESDF
            # and push the world so the next plan sees the object removed.
            self._attached_name = object_name
            # Drop the object's STATIC copy from the voxel-map RASTERIZATION
            # only (primitives_only_scene) so its analytic voxels don't reappear
            # in the published grid. Solver scenes KEEP it registered — cuRobo
            # disables it via the enable flag (disable_obstacle_names above),
            # re-asserted by reapply_attached_disables on every world push
            # (including the one _clear_attached_voxels triggers below).
            obs_mgr = self.config_wrapper.obstacle_manager
            obs_mgr.exclude_obstacle(object_name)
            if not self._clear_attached_voxels(obstacle):
                # No mapper / non-cuboid: no ESDF to refresh, but the exclusion
                # must still reach the solvers (and the voxel map) — push the
                # excluded world once.
                self.node.update_all_solvers_world(obs_mgr.get_scene())
        self._attached_name = object_name
        self.node.get_logger().info(
            f"Attached '{object_name}' to link '{ATTACH_LINK}'")

    def _fit_and_attach(self, am, obstacle, n_fit, joint_states, disable):
        """Fit spheres and write them to the attach link (curobo-side).

        Mirrors ``AttachmentManager.attach()`` (fit + ``update`` + disable), but
        replaces the ``update`` step with local code, because cuRobo's ``update``
        CRASHES when the fit yields exactly ONE sphere together with a
        ``world_objects_pose_offset``: ``env_pose.transform_points(centers)``
        returns shape ``(num_spheres, 3)`` (see geom/transform.py) and the
        following ``.squeeze(0)`` collapses ``[1, 3] -> [3]``, so the
        ``torch.cat`` in the body hits "Tensors must have same number of
        dimensions: got 1 and 2". We cannot patch curobo_core, so:

        * ``fit_spheres`` is called the same way (world-frame centers — the fit
          bakes the obstacle world pose into the mesh, ``transform_with_pose``);
        * the obstacle-to-link transform is applied HERE with an identity world
          offset (obj_to_link = ee.inverse()), reproducing curobo's math; the
          FK render then re-applies FK(attached_object) == FK(ee), so the
          world placement is correct (fit is not applied twice) — identical to
          the previous ``world_objects_pose_offset=identity`` path;
        * link-frame spheres are written straight into
          ``kinematics_params.link_spheres`` with the same padding (radius -100
          for unused slots) cuRobo would insert, and the world obstacle is
          disabled by name (recorded for detach) — again matching ``attach``.

        Must be called under ``node.gpu_lock`` (caller holds it). Raises on any
        cuRobo error, which surfaces as "Attach error: ...".
        """
        import torch
        sphere_tensor = am.fit_spheres(
            [obstacle], num_spheres=n_fit, surface_radius=0.002,
            sphere_fit_type=SphereFitType.SURFACE)
        centers = sphere_tensor[:, :3].contiguous()  # warp kernel needs contiguous
        radii = sphere_tensor[:, 3].unsqueeze(-1)

        q = joint_states.position
        if q.dim() == 1:
            q = q.unsqueeze(0)
        num_envs = q.shape[0]

        joint_state = JointState.from_position(
            q, joint_names=am._kinematics.joint_names)
        fk_result = am._kinematics.compute_kinematics(joint_state)
        if fk_result.tool_poses is None:
            raise RuntimeError(
                "FK result has no tool_poses; cannot resolve EE for attachment")
        ee_link = am._kinematics.tool_frames[0]
        obj_to_link = fk_result.tool_poses.get_link_pose(ee_link).inverse()

        kparams = am.kinematics_params
        link_idx = kparams.get_sphere_index_from_link_name(ATTACH_LINK)
        n_slots = link_idx.shape[0]
        n_fit = centers.shape[0]
        if n_fit > n_slots:
            raise RuntimeError(
                f"Fitted {n_fit} spheres but link '{ATTACH_LINK}' only has "
                f"{n_slots} sphere slots")
        padding = None
        if n_fit < n_slots:
            padding = torch.zeros(
                n_slots - n_fit, 4,
                device=sphere_tensor.device, dtype=sphere_tensor.dtype,
            )
            padding[:, 3] = -100.0

        for i in range(num_envs):
            env_pose = Pose(
                position=obj_to_link.position[i: i + 1],
                quaternion=obj_to_link.quaternion[i: i + 1],
            )
            # reshape(-1, 3) absorbs any spurious leading batch dims; cuRobo's
            # own `.squeeze(0)` is what breaks on a single fitted sphere.
            link_centers = env_pose.transform_points(centers).reshape(-1, 3)
            env_spheres = torch.cat([link_centers, radii], dim=-1)
            if padding is not None:
                env_spheres = torch.cat([env_spheres, padding], dim=0)
            kparams.link_spheres[i, link_idx, :] = env_spheres
        am._attached_link_name = ATTACH_LINK

        # cuRobo's native disable: records the name and disables it in the
        # MotionPlanner checker via scene_collision.enable_obstacle. The
        # obstacle STAYS registered in the solver scenes (exclusion is now
        # voxel-map-rasterization only), so the name-based disable/re-enable is
        # safe — and the world push below would re-enable it
        # (load_from_scene_cfg clears + re-adds with enable=1), so
        # reapply_attached_disables() re-asserts the flag after every push.
        if disable and am._scene_collision is not None:
            for name in disable:
                for env_idx in range(num_envs):
                    am._scene_collision.enable_obstacle(
                        name, enable=False, env_idx=env_idx)
            am._disabled_obstacle_names = list(disable)
            am._disabled_num_envs = num_envs

    def _clear_attached_voxels(self, obstacle) -> bool:
        """Remove the depth-derived voxels spanning the attached obstacle's
        world AABB, then refresh the ESDF and push the world to the solvers.

        Returns True if the world was pushed (only when a Mapper exists and a
        fresh ESDF was produced). Best-effort: quietly returns False if there is
        no Mapper or the obstacle isn't a cuboid. Must be called under
        ``node.gpu_lock`` (already held by the caller) because it runs GPU
        Mapper + world-update work.
        """
        try:
            obs_mgr = self.config_wrapper.obstacle_manager
            if getattr(obs_mgr, 'mapper', None) is None:
                return False
            bounds_min, bounds_max = self._obstacle_world_aabb(obstacle)
            if bounds_min is None:
                return False
            obs_mgr.clear_voxel_region(bounds_min, bounds_max, self.node)
            if obs_mgr.refresh_esdf():
                self.node.update_all_solvers_world(obs_mgr.get_scene())
                return True
        except Exception as e:
            self.node.get_logger().warn(
                f"Could not clear voxels under attached object: {e}",
                throttle_duration_sec=5.0,
            )
        return False

    @staticmethod
    def _obstacle_world_aabb(obstacle) -> tuple:
        """World-aligned bounding box of a cuboid obstacle.

        Transforms the 8 box corners by the obstacle's pose (position +
        wxyz quaternion) and returns (bounds_min, bounds_max) length-3 torch
        tensors, or (None, None) for a non-cuboid obstacle. A small margin is
        added so voxels just outside the faces are cleared too.
        """
        dims = getattr(obstacle, 'dims', None)
        pose = getattr(obstacle, 'pose', None)
        if dims is None or pose is None:
            return None, None
        import torch
        import numpy as np
        from scipy.spatial.transform import Rotation
        px, py, pz, qw, qx, qy, qz = pose[:7]
        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        corners = np.array([
            [sx, sy, sz]
            for sx in (-0.5, 0.5)
            for sy in (-0.5, 0.5)
            for sz in (-0.5, 0.5)
        ], dtype=np.float32) * np.array(dims, dtype=np.float32)
        world = corners @ rot.T + np.array([px, py, pz], dtype=np.float32)
        margin = 0.015  # 1.5 cm clearance beyond the faces
        lo = world.min(axis=0) - margin
        hi = world.max(axis=0) + margin
        return (torch.from_numpy(lo), torch.from_numpy(hi))

    @staticmethod
    def _estimate_sphere_need(obstacle) -> int:
        """cuRobo's own estimate of how many spheres this geometry needs.

        Same heuristic the AttachmentManager applies when ``num_spheres=None``
        (bounding-box volume, 1 sphere per 15 cm3, capped at 100). Returns 0 if
        it can't be computed, so a failure here never blocks an attach.
        """
        try:
            return estimate_sphere_count(
                obstacle.get_trimesh_mesh(transform_with_pose=True))
        except Exception:
            return 0

    @staticmethod
    def _find_obstacle(scene, name: str):
        """Find an obstacle by name across the Scene's typed buckets.

        The ObstacleManager stores obstacles in typed lists (cuboid/sphere/…),
        not in Scene.objects, so Scene.get_obstacle (which reads .objects) misses
        runtime-added ones — search the buckets directly.
        """
        for bucket in ('cuboid', 'sphere', 'capsule', 'cylinder', 'mesh'):
            for obj in (getattr(scene, bucket, None) or []):
                if obj.name == name:
                    return obj
        if getattr(scene, 'objects', None):
            return scene.get_obstacle(name)
        return None

    def detach(self) -> str:
        """Release the attached object (reset link spheres, restore the static
        obstacle to collision checking)."""
        released = self._attached_name
        with self.node.gpu_lock:
            self._attachment_manager().detach(ATTACH_LINK)
            if released:
                # Restore the static copy to the voxel-map rasterization, and
                # push the world so every solver re-loads the obstacle ENABLED
                # (a push clears + re-adds with enable=1; with nothing left in
                # excluded_obstacle_names, reapply_attached_disables no-ops).
                # am.detach(ATTACH_LINK) already re-enabled it on the
                # MotionPlanner checker via its recorded disabled names.
                self.config_wrapper.obstacle_manager.include_obstacle(released)
                self.node.update_all_solvers_world(
                    self.config_wrapper.obstacle_manager.get_scene())
        self._attached_name = None
        self.node.get_logger().info(f"Detached '{released}' from '{ATTACH_LINK}'")
        return released or ""

    # ------------------------------------------------------------------
    # Service callbacks
    # ------------------------------------------------------------------

    def _attach_object_callback(self, request, response):
        """Attach a scene obstacle to the arm at its current joint configuration."""
        try:
            if self.motion_planner is None:
                self.node._warmup_classic()
            _, current_state = self.node._resolve_start_state(None)
            self.attach(request.object_name, current_state)
            response.success = True
            response.message = f"Attached '{request.object_name}'"
        except Exception as e:
            response.success = False
            response.message = f"Attach error: {e}"
            self.node.get_logger().error(response.message)
        return response

    def _detach_object_callback(self, request, response):
        """Release a previously attached object from the arm."""
        try:
            released = self.detach()
            response.success = True
            response.message = (
                f"Detached '{released}'" if released else "Nothing attached")
        except Exception as e:
            response.success = False
            response.message = f"Detach error: {e}"
            self.node.get_logger().error(response.message)
        return response
