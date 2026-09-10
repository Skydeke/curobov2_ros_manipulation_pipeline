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


def _at_joints_pose(wrapper, at_joints):
    """Joint positions (list, in ``kin.joint_names`` order) or the live pose.

    ``at_joints`` lets callers attribute collisions at an arbitrary
    configuration (e.g. a goal state) instead of only the robot's current
    pose — the start/end-state collision check has no way to distinguish the
    two otherwise.
    """
    if at_joints is not None:
        return list(at_joints)
    return wrapper.robot.get_joint_pose()


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


def _sphere_collision_snapshot(
        wrapper, node, kin=None, solver=None,
        default_activation=0.0, activation_distance=None, at_joints=None):
    """Single FK snapshot: per-sphere distances AND the sphere array they came from.

    One ``kin.compute_kinematics`` call feeds both the scene-collision distance
    query and the returned ``spheres`` array, so the distance list and the array
    always index 1:1 (identical N). Callers MUST attribute distances against the
    returned sphere array — never re-run FK (or query spheres) separately and
    index from a different source, and never reshape ``sphere_dist`` in a way
    that changes its length relative to ``spheres``. Either would give distance
    entries that index past the N-row sphere array.

    Returns (``spheres`` [N, 4] float-tensor, flat per-sphere distance list) on
    success, or (None, None) if the solver/checker isn't ready or FK failed.
    Never raises.
    """
    try:
        if kin is None:
            kin = wrapper.kin_model
        q_js = JointState(
            position=torch.tensor(
                _at_joints_pose(wrapper, at_joints),
                dtype=wrapper._ops_dtype,
                device=wrapper._device,
            ),
            joint_names=kin.joint_names,
        )
        kinematics_state = kin.compute_kinematics(q_js)
        robot_spheres = kinematics_state.robot_spheres
        spheres = robot_spheres.reshape(-1, 4)

        scene_collision_checker = _resolve_scene_checker(node, solver)
        if scene_collision_checker is None or not hasattr(
                scene_collision_checker, 'get_sphere_distance'):
            return None, None

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
        # A negative-radius sphere has no physical meaning and its inflated
        # radius (radius + activation) stays positive against any obstacle, so
        # it would always read as "colliding". Ignore such spheres outright.
        #
        # CRITICAL: the mask must be SHAPE-PRESERVING. This buffer is
        # [batch, horizon, num_spheres] (no trailing singleton), so applying
        # the mask as ``negative_radius.unsqueeze(-1)`` (4-D) makes
        # ``masked_fill`` BROADCAST the result to [b, h, N, N] — a sphere×sphere
        # cross product whose flattened indices no longer map back onto the N
        # robot spheres. That was the "index <k> is out of bounds for dimension
        # 0 with size <N>" crash in ``_attributed_collisions`` (any red index
        # >= N from the cross-product list blew past the N-row sphere array).
        negative_radius = robot_spheres[..., 3] < 0
        if negative_radius.any():
            sphere_dist = sphere_dist * (~negative_radius)
        return spheres, torch.flatten(sphere_dist, start_dim=0).tolist()
    except Exception:
        return None, None


def _query_sphere_collision(
        wrapper, node, kin=None, solver=None,
        default_activation=0.0, activation_distance=None, at_joints=None):
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
        at_joints: Joint positions (list, ``kin.joint_names`` order) to evaluate
            at instead of the robot's live pose. None = live pose.

    Returns:
        Flat per-sphere distance list (value > 0 = sphere collides), or None if
        the solver/checker isn't ready — callers fall back to "no collision
        info". Never raises.
    """
    _spheres, dists = _sphere_collision_snapshot(
        wrapper, node, kin=kin, solver=solver,
        default_activation=default_activation,
        activation_distance=activation_distance, at_joints=at_joints)
    return dists


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
                           activation_distance=None, at_joints=None):
    """Collisions (with obstacle attribution) at the robot's current joint state.

    Computes distances and the sphere centers from the SAME
    ``kin.compute_kinematics`` snapshot (``_sphere_collision_snapshot``), so
    every returned sphere index indexes the array the distances were computed
    against (see that function for why cross-indexing a differently-computed
    sphere array breaks). The returned sphere indices line up with the
    marker-array spheres. For every reported sphere it also names the enabled
    scene cuboids it comes within the activation distance of (computed against
    the same OBB SDF the GPU kernel uses), and flags mesh / voxel-layer
    involvement when present but unattributable.

    ``at_joints`` overrides the evaluated configuration (robot's live pose by
    default) — pass the goal joint positions to attribute "start or end state
    in collision" failures that the live-pose check can't see.

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
    spheres, dists = _sphere_collision_snapshot(
        wrapper, node, kin=kin, solver=solver,
        activation_distance=activation_distance, at_joints=at_joints)
    if spheres is None or dists is None:
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
        # Belt-and-braces: only attribute indices that exist in THIS snapshot's
        # sphere array. With correct per-sphere distances the red indices are
        # always < len(spheres); a past shape bug (negative-radius mask that
        # broadcast the distance tensor to a sphere×sphere cross product) put
        # garbage indices here and crashed with "index out of bounds".
        if sph_idx >= len(spheres):
            continue
        c = spheres[sph_idx].detach().cpu().numpy()
        center, radius = c[:3].astype(np.float64), float(c[3])
        if radius < 0:
            continue

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


def _attributed_self_collisions(wrapper, node, kin=None, at_joints=None):
    """Self-collisions (link-vs-link) at a joint state, attributed by link.

    cuRobo's self-collision check is a DIFFERENT mechanism from the scene
    collision checker: a fixed table of sphere pairs (one sphere on each of two
    robot links) queried directly against the FK'd robot spheres, inflated by a
    per-link ``self_collision_link_padding``. Scene obstacles play no part — a
    closed gripper on an empty bench still self-collides. That is why
    "Start or End state in collision" failures caused by self-contact produced
    no ``collision_contacts``: ``_query_sphere_collision`` only queries the
    scene checker, which never sees robot-self sphere pairs.

    Mirrors the kernel (``SelfCollisionDistance`` in curobo) exactly: for every
    sphere pair in the kinematics' ``SelfCollisionKinematicsCfg.collision_pairs``
    it computes the same squared overlap
    ``(r_a + pad_a + r_b + pad_b)**2 - |c_a - c_b|**2`` (positive = the pair is
    in self-collision). The planner applies NO activation margin to
    self-collision (only scene collision uses ``collision_activation_distance``,
    see ``RobotCostManagerCfg.update_collision_activation_distance``), so these
    are hard contacts at the configured per-link padding — reported with a
    linear penetration depth for readability on top of the kernel's squared
    value.

    Returns a list of dicts (one per colliding sphere pair): ``sphere_a`` /
    ``sphere_b`` (indices), ``link`` / ``link_b`` (kinematics link names),
    ``radius_a`` / ``radius_b`` (m), ``padding_a`` / ``padding_b`` (m),
    ``center_distance`` (m), ``overlap2`` (m^2, the kernel's stored value),
    ``depth`` (m, linear penetration). Returns None when self-collision is
    disabled or the kinematics/sphere counts don't line up, [] when nothing
    collides, and [] duplicates can't happen (indices are unique). Never raises.
    """
    import numpy as np
    try:
        if kin is None:
            kin = wrapper.kin_model
        sc = getattr(kin.config, 'self_collision_config', None)
        if sc is None:
            return None
        pairs = getattr(sc, 'collision_pairs', None)
        padding = getattr(sc, 'sphere_padding', None)
        if pairs is None or padding is None or pairs.numel() == 0:
            return None

        q_js = JointState(
            position=torch.tensor(
                _at_joints_pose(wrapper, at_joints),
                dtype=wrapper._ops_dtype,
                device=wrapper._device,
            ),
            joint_names=kin.joint_names,
        )
        spheres = kin.compute_kinematics(q_js).robot_spheres.reshape(-1, 4)
        # The self-collision table indexes into the SAME robot-sphere array
        # (kinematic spheres). Bail if it somehow doesn't apply to this kin.
        if spheres.shape[0] != int(getattr(sc, 'num_spheres', -1)):
            return None

        c = spheres.detach().cpu().numpy()
        pad = padding.detach().cpu().numpy().reshape(-1)
        idx = pairs.detach().cpu().numpy().astype(np.int64)

        ca = c[idx[:, 0], :3]
        cb = c[idx[:, 1], :3]
        ra = c[idx[:, 0], 3] + pad[idx[:, 0]]
        rb = c[idx[:, 1], 3] + pad[idx[:, 1]]
        center_dist = np.linalg.norm(ca - cb, axis=1)
        overlap2 = (ra + rb) ** 2 - center_dist ** 2  # kernel's pair value

        kp = kin.config.kinematics_config
        idx_map = (kp.link_sphere_idx_map.reshape(-1).detach().cpu().tolist()
                   if kp.link_sphere_idx_map is not None else [])
        idx_to_name = {int(v): k for k, v in kp.link_name_to_idx_map.items()}

        def _link_of(sph):
            li = idx_map[sph] if sph < len(idx_map) else None
            return (idx_to_name.get(li, f"link#{li}")
                    if li is not None else "unknown-link")

        reports = []
        radius_valid = c[:, 3] >= 0
        good = ((overlap2 > 0.0)
                & radius_valid[idx[:, 0]] & radius_valid[idx[:, 1]])
        for k in np.flatnonzero(good):
            a, b = int(idx[k, 0]), int(idx[k, 1])
            reports.append({
                "sphere_a": a,
                "sphere_b": b,
                "link": _link_of(a),
                "link_b": _link_of(b),
                "radius_a": float(c[a, 3]),
                "radius_b": float(c[b, 3]),
                "padding_a": float(pad[a]),
                "padding_b": float(pad[b]),
                "center_distance": float(center_dist[k]),
                "overlap2": float(overlap2[k]),
                "depth": float(ra[k] + rb[k] - center_dist[k]),
            })
        return reports
    except Exception:
        return None


def _attributed_joint_limit_violations(wrapper, node, kin=None, at_joints=None):
    """Position-limit (cspace bound) violations at a joint state.

    The graph planner's feasibility check treats the cspace boundary as a hard
    constraint: a start or goal configuration outside a joint's position limits
    makes that state infeasible and prints the SAME ``Start or End state in
    collision`` warning as a sphere contact. So a plan failure can come from a
    joint value out of range even when every collision sphere is clear — and,
    because RViz sphere colouring only queries world/self sphere collisions,
    NO sphere turns red for it.

    Returns a list of dicts (one per violated joint): ``joint`` (name),
    ``value``, ``lower``, ``upper`` (all rad). None when the limits can't be
    reached or the joint ordering can't be aligned with the kinematics; never
    raises.
    """
    try:
        if kin is None:
            kin = wrapper.kin_model
        limits = kin.get_joint_limits()
        pos = limits.position.detach().cpu().numpy()
        lim_names = list(limits.joint_names)
        names = list(kin.joint_names)
        at = _at_joints_pose(wrapper, at_joints)
        if at is None or len(at) != len(names) or len(lim_names) != len(names):
            return None
        lim_idx = {n: i for i, n in enumerate(lim_names)}
        for n in names:
            if n not in lim_idx:
                return None
        reports = []
        for i, name in enumerate(names):
            v = float(at[i])
            lo = float(pos[0, lim_idx[name]])
            hi = float(pos[1, lim_idx[name]])
            if v < lo or v > hi:
                reports.append({"joint": name, "value": v,
                                "lower": lo, "upper": hi})
        return reports
    except Exception:
        return None


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