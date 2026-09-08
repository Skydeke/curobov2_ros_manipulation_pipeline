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
from curobo._src.types.device_cfg import DeviceCfg
from curobo.sphere_fit import estimate_sphere_count

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
        passed directly to ``attach`` (not ``attach_from_scene``, which looks up
        by name in the solver's scene_model). The obstacle is EXCLUDED from
        every derived collision world via the ObstacleManager (so neither the
        solver buffers nor the voxel-map rasterization double-count it with the
        attached spheres) and the world is pushed to the solvers; detach
        restores it. cuRobo's own ``disable_obstacle_names``/re-enable cycle is
        NOT used — it re-enables via ``scene_collision.enable_obstacle`` on
        detach, which fails because the excluded obstacle is no longer loaded.
        Affects the shared MotionPlanner only (open-loop transport).

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
            device_cfg = DeviceCfg(
                device=self.config_wrapper._device,
                dtype=self.config_wrapper._ops_dtype,
            )
            # cuRobo's fit_spheres bakes the obstacle world pose into the mesh
            # (transform_with_pose=True), so `centers` are already in WORLD
            # space. update() computes obj_to_link = ee.inverse() * P_world and
            # transforms the (world) centers by it, then the FK render applies
            # FK(attached_object) — which equals FK(ee) since attached_object is
            # an identity-fixed child of the tool frame. So the obstacle pose
            # would be applied TWICE and the spheres land off the gripper.
            # Passing an IDENTITY world_objects_pose_offset makes the extra
            # factor drop out: obj_to_link = ee.inverse(), rendered = P_world *
            # centers (correct world placement). This is the place that can't
            # change curobo_core, so compensate here.
            identity_pose = Pose(
                position=device_cfg.to_device([[0.0, 0.0, 0.0]]),
                quaternion=device_cfg.to_device([[1.0, 0.0, 0.0, 0.0]]),
            )
            # NB: no disable_obstacle_names here. cuRobo records them and
            # re-enables them on detach via scene_collision.enable_obstacle —
            # but the world push below EXCLUDES the obstacle from the derived
            # solver scene (it is no longer loaded), so that re-enable raises
            # "Obstacle 'object_name' not found in environment 0" and detach
            # fails. Exclusion + restore is handled by the ObstacleManager
            # (exclude_obstacle / include_obstacle + world push), so cuRobo's
            # name-based disable/re-enable would be redundant anyway.
            am.attach(
                joint_states=grasp_end_state,
                obstacles=[obstacle],
                link_name=ATTACH_LINK,
                num_spheres=n_fit,
                world_objects_pose_offset=identity_pose,
            )
            # The perception voxel layer still holds the voxels of the grasped
            # object at its world position. Disabling the obstacle BY NAME only
            # disables the cuboid collision buffer — the ESDF voxels remain, so
            # the freshly attached spheres collide with the object's own leftover
            # voxels and any motion from here starts "in collision". Clear just
            # that region from the depth-derived voxel channel, refresh the ESDF
            # and push the world so the next plan sees the object removed.
            self._attached_name = object_name
            # Drop the object's STATIC copy from every derived collision world
            # (solver buffers + voxel map) so it doesn't collide with its own
            # attached spheres — the world push in _clear_attached_voxels below
            # already ships the excluded scene.
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
        obstacle to the collision world)."""
        released = self._attached_name
        with self.node.gpu_lock:
            self._attachment_manager().detach(ATTACH_LINK)
            if released:
                # Bring the static copy back into every derived collision world
                # now that the arm no longer carries the attached spheres.
                self.config_wrapper.obstacle_manager.include_obstacle(released)
                # The push re-renders the obstacle in the solvers AND the voxel map.
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
