#!/usr/bin/env python3
"""
Single shared collision-distance query harness for every per-sphere consumer.

All call sites (the GetCollisionDistance service, the collision-sphere RViz
colouring and the plan-failure diagnostic) used to re-implement the same
FK + CollisionBuffer + ``scene_collision_checker.get_sphere_distance`` sequence
(plus the sphere-to-link attribution it feeds). This leaf module is imported by
config_wrapper_motion / ros_service_manager / unified_planner_node without
creating an import cycle.

Activation distance vs true contact
-----------------------------------
``get_sphere_distance`` inflates every query sphere by the activation distance
(radius_adjusted = radius + eta) and reports a positive cost as soon as the
sphere surface comes within ``eta`` metres of an obstacle — that is a *safety
margin*, not physical contact. The RViz red spheres and the plan-failure
diagnostic share this margin (the same ``collision_activation_distance`` param
that feeds ``optimizer_collision_activation_distance``), so "red / in
collision" means exactly what the planner's optimizer/feasibility check sees:
a positive cost at the same activation distance. Callers that need the *true*
contact answer (no clearance) pass ``activation_distance=0.0`` — then a
positive value means real overlap and its magnitude is the penetration depth.
"""

import torch

from curobo.types import JointState
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.types.device_cfg import DeviceCfg

from isaac_ros_cumotion_interfaces.srv import GetCollisionDistance


def _resolve_scene_checker(node, solver):
    """Resolve a SceneCollision from an explicit solver or the node's defaults.

    Accepts a planner wrapper that carries ``.motion_planner`` (the
    SinglePlanner subclasses) as well as the raw solvers — the planner-level
    wrappers do not expose ``scene_collision_checker`` themselves.
    """
    if solver is None:
        solver = (getattr(node, 'motion_planner', None)
                  or getattr(node, 'mpc', None)
                  or getattr(node, 'ik_solver', None))
    if solver is None:
        return None
    checker = getattr(solver, 'scene_collision_checker', None)
    if checker is None:
        mp = getattr(solver, 'motion_planner', None)
        checker = getattr(mp, 'scene_collision_checker', None)
    return checker


def _resolve_activation(node, default_activation=0.0):
    """Resolve the effective activation distance (metres).

    None/undeclared param -> ``default_activation``; otherwise the node's
    ``collision_activation_distance`` double param (the same value that is
    passed as ``optimizer_collision_activation_distance`` to the MotionPlanner,
    so display and planner margin stay identical).
    """
    if node is not None and node.has_parameter('collision_activation_distance'):
        return node.get_parameter(
            'collision_activation_distance').get_parameter_value().double_value
    return default_activation


def _query_sphere_collision(
        wrapper, node, kin=None, solver=None,
        default_activation=0.0, activation_distance=None):
    """Per-sphere collision distance at the robot's current joint state.

    v2 notes: solvers (`MotionPlanner`, `ModelPredictiveControl`,
    `InverseKinematics`) expose the world/scene collision checker as
    ``scene_collision_checker`` (a SceneCollision), whose ``get_sphere_distance``
    takes the full ``KinematicsState``, a pre-allocated ``CollisionBuffer`` and
    ``weight``/``activation_distance`` tensors; ``robot_spheres`` replaces the
    legacy ``link_spheres_tensor``, already [B, H, N, 4].

    Args:
        wrapper: ConfigWrapperMotion (robot joint pose, device, dtype).
        node: The planning node (param lookup + default solver resolution).
        kin: Kinematics to FK with; defaults to ``wrapper.kin_model``. Pass the
            MotionPlanner's kinematics (attachment_services.kinematics()) to
            include fitted attached-object spheres — and pair the returned
            distances with ``kin.config.kinematics_config.link_sphere_idx_map``
            for the collision-link attribution.
        solver: Which solver's ``scene_collision_checker`` to query. Defaults to
            motion_planner -> mpc -> ik_solver on the node. Planner wrappers
            (with ``.motion_planner``) are unwrapped.
        default_activation: ``activation_distance`` used when the
            ``collision_activation_distance`` param is not declared.
        activation_distance: Explicit override for the activation distance.
            None = use the ``collision_activation_distance`` node param (safety
            margin); 0.0 = true contact only (positive value = real overlap and
            the magnitude is the penetration depth in metres).

    Returns:
        Flat per-sphere distance list (value > 0 = sphere collides), or None if
        the solver/checker isn't ready — callers fall back to "no collision
        info". Never raises.
    """
    try:
        if kin is None:
            kin = wrapper.kin_model
        q_js = JointState(
            position=torch.tensor(
                wrapper.robot.get_joint_pose(),
                dtype=wrapper._ops_dtype,
                device=wrapper._device,
            ),
            joint_names=kin.joint_names,
        )
        kinematics_state = kin.compute_kinematics(q_js)
        robot_spheres = kinematics_state.robot_spheres

        scene_collision_checker = _resolve_scene_checker(node, solver)
        if scene_collision_checker is None or not hasattr(
                scene_collision_checker, 'get_sphere_distance'):
            return None

        if activation_distance is None:
            activation = _resolve_activation(node, default_activation)
        else:
            activation = activation_distance

        device_cfg = DeviceCfg(device=wrapper._device, dtype=wrapper._ops_dtype)
        collision_buffer = CollisionBuffer.from_shape(robot_spheres.shape, device_cfg)
        weight = device_cfg.to_device([1.0])
        activation_t = device_cfg.to_device([activation])

        sphere_dist = scene_collision_checker.get_sphere_distance(
            kinematics_state,
            collision_buffer,
            weight,
            activation_distance=activation_t,
        )
        return torch.flatten(sphere_dist, start_dim=0).tolist()
    except Exception:
        return None


def _rot_from_quat_wxyz(quat):
    """3x3 rotation matrix (numpy) from a [qw, qx, qy, qz] quaternion."""
    import numpy as np
    qw, qx, qy, qz = map(float, quat)
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ], dtype=np.float64)


def _attributed_collisions(wrapper, node, kin=None, solver=None,
                           activation_distance=None):
    """Collisions (with obstacle attribution) at the robot's current joint state.

    Identical kinematics/shape-source as ``_query_sphere_collision`` (so the
    returned sphere indices line up with the marker-array spheres), but for
    every reported sphere also names the enabled scene cuboids it comes within
    the activation distance of (computed against the same OBB SDF the GPU
    kernel uses), and flags mesh / voxel-layer involvement when present but
    unattributable.

    The activation distance defaults to the ``collision_activation_distance``
    param (None) — the SAME margin the planner's optimizer uses, so the red
    spheres / failure diagnostic and the planner agree on what counts as "in
    collision". Pass ``activation_distance=0.0`` for true contact only (no
    clearance) — then ``depth`` is the exact penetration in metres; under the
    default margin it is the kernel's reported overlap of the inflated sphere
    (``radius + activation - signed_distance``, > 0 ⟺ within margin, and > the
    activation iff the surfaces really touch).

    Returns a list of dicts (one per red sphere):
        index, position [x,y,z], radius, activation (m), depth (m), obstacles
        [names], link (kinematics link name), clobber_note (str, empty unless
        the sphere is within margin of only mesh/voxel obstacles, which we
        cannot name precisely).
    Returns None when the solver/collision checker isn't ready. Never raises.
    """
    import numpy as np
    dists = _query_sphere_collision(
        wrapper, node, kin=kin, solver=solver,
        activation_distance=activation_distance)
    if dists is None:
        return None
    red = [(i, d) for i, d in enumerate(dists) if d > 0.0]
    if not red:
        return []

    activation = _resolve_activation(node, default_activation=0.0) \
        if activation_distance is None else activation_distance
    if activation is None:
        activation = 0.0

    if kin is None:
        kin = wrapper.kin_model
    q_js = JointState(
        position=torch.tensor(
            wrapper.robot.get_joint_pose(),
            dtype=wrapper._ops_dtype,
            device=wrapper._device,
        ),
        joint_names=kin.joint_names,
    )
    spheres = kin.compute_kinematics(q_js).robot_spheres.reshape(-1, 4)

    # Sphere -> link name attribution (same mapping the diagnostic uses).
    kp = kin.config.kinematics_config
    idx_map = (kp.link_sphere_idx_map.reshape(-1).detach().cpu().tolist()
               if kp.link_sphere_idx_map is not None else [])
    idx_to_name = {int(v): k for k, v in kp.link_name_to_idx_map.items()}
    link_of = {}
    for sph_idx in (i for i, _ in red):
        li = idx_map[sph_idx] if sph_idx < len(idx_map) else None
        link_of[sph_idx] = (idx_to_name.get(li, f"link#{li}")
                            if li is not None else "unknown-link")

    checker = _resolve_scene_checker(node, solver)
    if checker is None:
        return None

    # Enabled scene obstacles we could attribute to.
    scene_data = getattr(checker, 'data', None)

    def _active_names(store):
        """[(name, local-full-dims or None, inv_pose-or-None)] for enabled obs."""
        if store is None:
            return []
        out = []
        try:
            count = int(store.count[0].item())
            enable = store.enable[0]
            names = store.names[0][:count]
            dims = store.dims[0].detach().cpu().numpy()
            inv_pose = store.inv_pose[0].detach().cpu().numpy()
        except Exception:
            return []
        for j in range(count):
            if int(enable[j].item() if hasattr(enable[j], 'item') else enable[j]) != 1:
                continue
            out.append((names[j], dims[j], inv_pose[j]))
        return out

    cuboids = _active_names(getattr(scene_data, 'cuboids', None))
    meshes = _active_names(getattr(scene_data, 'meshes', None))
    has_voxels = getattr(scene_data, 'voxels', None) is not None

    reports = []
    for sph_idx, depth_sum in red:
        c = spheres[sph_idx].detach().cpu().numpy()
        center, radius = c[:3].astype(np.float64), float(c[3])

        hits = []
        for name, dims, inv_pose in cuboids:
            half = dims[:3] * 0.5
            t = inv_pose[:3]
            R = _rot_from_quat_wxyz(inv_pose[3:7])
            local = R @ center + t
            q = np.abs(local) - half
            outside = np.sqrt(np.sum(np.maximum(q, 0.0) ** 2))
            sdf = outside + min(np.max(q), 0.0)
            # Same reach test as the GPU kernel's inflated sphere: within
            # `radius + activation` means the agent's cost is non-zero.
            if sdf < radius + activation:
                hits.append(name)
        # Mesh and voxel collisions cannot be named from the accumulated
        # buffer, so flag the *possibility* when nothing else explains red.
        note = ""
        if not hits:
            causes = []
            if meshes:
                causes.append("mesh:" + ",".join(name for name, _, _ in meshes))
            if has_voxels:
                causes.append("depth-voxel-layer")
            note = " or ".join(causes)
        reports.append({
            "index": sph_idx,
            "position": center.tolist(),
            "radius": radius,
            "activation": activation,
            "depth": float(depth_sum),
            "obstacles": hits,
            "link": link_of.get(sph_idx, "unknown-link"),
            "clobber_note": note,
        })
    return reports


def _compute_sphere_distance(wrapper, node, response):
    """Query collision distance at the robot's current configuration and fill a
    GetCollisionDistance response (shared harness above)."""
    sphere_dist_ar = _query_sphere_collision(
        wrapper, node, default_activation=0.025,
    )
    if sphere_dist_ar is None:
        response.nb_sphere = 0
        response.data = []
        return response
    response.nb_sphere = len(sphere_dist_ar)
    response.data = sphere_dist_ar
    return response