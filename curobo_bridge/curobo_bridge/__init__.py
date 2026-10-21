# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6).

ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary. This is the M1 skeleton: the lifecycle, lane binding and global flags.
The scene/plan/fk/ik/perception entry points are added per milestone, with the
signatures frozen by the C++ runtime's ``IBridge``.
"""

from __future__ import annotations

from typing import Dict, Optional

# Opaque-handle registry. Kinematics, collision checkers, solvers and the Mapper
# live here (M2+); cuTAMP reads the same registry (C5).
_REGISTRY: Dict[int, dict] = {}
_NEXT_HANDLE = 1

# Lane name -> per-lane stream object, created in bind_lane() on the lane thread.
_LANES: Dict[str, object] = {}

_GLOBAL_FLAGS = {"cuda_graph_reset": False, "debug_cuda_graphs": False}


def init(cfg: Optional[dict] = None) -> int:
    """Build the shared cuRobo objects and return their handle.

    M1 records the config; the cuRobo construction lands at M2.
    """
    global _NEXT_HANDLE
    handle = _NEXT_HANDLE
    _NEXT_HANDLE += 1
    _REGISTRY[handle] = {"cfg": dict(cfg or {}), "built": False}
    return handle


def shutdown(handle: int) -> None:
    """Release the registry entry for ``handle``."""
    _REGISTRY.pop(int(handle), None)


def bind_lane(name: str) -> None:
    """Create the calling lane's CUDA stream and store it thread-locally.

    No-op without torch/CUDA (unit tests, CI without a GPU).
    """
    try:
        import torch  # noqa: F401
        if torch.cuda.is_available():
            _LANES[name] = torch.cuda.Stream(device=torch.device("cuda", torch.cuda.current_device()))
            return
    except Exception:
        pass
    _LANES[name] = None


def set_global_flags(cuda_graph_reset: bool, debug_cuda_graphs: bool) -> None:
    """Process-global cuRobo flags (PORT: config_wrapper_motion import effects)."""
    import curobo._src.runtime as runtime_mod

    runtime_mod.cuda_graph_reset = bool(cuda_graph_reset)
    runtime_mod.debug_cuda_graphs = bool(debug_cuda_graphs)
    # `curobo/runtime.py` re-exports these with `from curobo._src.runtime import`,
    # so it holds its OWN binding. Consumers that do `from curobo import runtime`
    # (is_cuda_graph_reset_available, cuda_graph_util) read THAT copy - setting
    # only `curobo._src.runtime` here left the flag effectively off and made
    # reset_cuda_graph() raise. Update both bindings.
    try:
        import curobo.runtime as runtime_pkg

        runtime_pkg.cuda_graph_reset = bool(cuda_graph_reset)
        runtime_pkg.debug_cuda_graphs = bool(debug_cuda_graphs)
    except Exception:
        pass
    _GLOBAL_FLAGS["cuda_graph_reset"] = bool(cuda_graph_reset)
    _GLOBAL_FLAGS["debug_cuda_graphs"] = bool(debug_cuda_graphs)


def registered_handles() -> list:
    """Introspection for tests: the live handle ids."""
    return sorted(_REGISTRY)


def lane_names() -> list:
    """Introspection for tests: the lanes bound so far."""
    return sorted(_LANES)


# ---------------------------------------------------------------------------
# M2: robot model, FK, IK, validate, scene
# ---------------------------------------------------------------------------

def _resolve_uri(path: str, base_dir: str) -> str:
    """PORT: robot_description._resolve_uri - resolve package:// and relative paths."""
    import os

    if not path:
        return path
    if path.startswith("package://"):
        from ament_index_python.packages import get_package_share_directory

        rest = path[len("package://"):]
        pkg, _, tail = rest.partition("/")
        return os.path.join(get_package_share_directory(pkg), tail)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _detect_asset_root() -> str:
    """PORT: config_manager.ConfigManager._detect_asset_root - the ROS install
    share/ dir (the parent of this package's share). URDF meshes are referenced
    package://<pkg>/... and cuRobo joins the remainder onto this root."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("curobov2_ros")
        return os.path.dirname(os.path.abspath(share))
    except Exception:
        return ""


def _translate_xrdf(xrdf_path: str, urdf_path: str, asset_root: str) -> str:
    """PORT: config_manager.ConfigManager._translate_xrdf.

    An xrdf config (``format: xrdf``) is the canonical robot definition; cuRobo's
    v2 factories load robot files with a raw ``yaml.load`` and expect
    ``robot_cfg.kinematics``, so the xrdf must go through cuRobo's own translator
    (``load_robot_yaml`` / ``convert_xrdf_to_curobo``), which needs the URDF (for
    the actuated/locked joints and base link) plus a mesh asset root. The
    translated ``robot_cfg`` is written to a resolved copy on disk.
    """
    import hashlib
    import os

    import yaml

    from curobo._src.types.content_path import ContentPath
    from curobo._src.robot.loader.util import load_robot_yaml

    if not urdf_path or not os.path.isfile(urdf_path):
        raise ValueError(
            "xrdf robot config "
            f"'{xrdf_path}' requires a valid 'urdf_path' (pointing at the "
            "generated URDF), because cuRobo's xrdf translator derives the "
            "actuated joints, locked joints and base link from it. Got: "
            f"{urdf_path!r}. Set the 'urdf_path' ROS parameter (and, if the "
            "meshes use package:// URIs, 'asset_root_path').")

    if not asset_root:
        asset_root = _detect_asset_root()
    if not asset_root or not os.path.isdir(asset_root):
        # Non-fatal: cuRobo warns and mesh resolution may fail.
        print(
            f"[curobo_bridge] asset_root_path {asset_root!r} does not exist; "
            "meshes may fail to resolve", flush=True)

    content = ContentPath(
        robot_xrdf_absolute_path=os.path.abspath(xrdf_path),
        robot_urdf_absolute_path=os.path.abspath(urdf_path),
        robot_asset_absolute_path=os.path.abspath(asset_root) if asset_root else None,
    )
    translated = load_robot_yaml(content)
    digest = hashlib.sha1(os.path.abspath(xrdf_path).encode()).hexdigest()[:10]
    resolved_dir = "/tmp/curobov2_ros_resolved"
    os.makedirs(resolved_dir, exist_ok=True)
    resolved_path = os.path.join(resolved_dir, f"xrdf-{digest}.yml")
    with open(resolved_path, "w") as fh:
        yaml.safe_dump(translated, fh, default_flow_style=None, sort_keys=False)
    return resolved_path


def _robot_kinematics_block(data: dict) -> dict:
    """The kinematics mapping of a robot YAML, structure-aware.

    cuRobo's loader accepts three shapes: ``robot_cfg.kinematics.*``, a
    root-level ``kinematics`` block (``kate.curobo.yaml``), or a flat config with
    the kinematics keys at the document root. Reading ``data["lock_joints"]``
    directly therefore found nothing for ``kate.curobo.yaml`` - which silently
    left the unpinned lock joints (``lock_joints: {joint: null}``) unreported, so
    nothing ever drove them from the joint state.
    """
    if not isinstance(data, dict):
        return {}
    root = data.get("kinematics")
    if isinstance(root, dict):
        return root
    robot_cfg = data.get("robot_cfg")
    if isinstance(robot_cfg, dict):
        block = robot_cfg.get("kinematics")
        return block if isinstance(block, dict) else robot_cfg
    return data


def robot_model_build(handle: int, cfg_json: str) -> None:
    """Build the shared Kinematics from the robot YAML (PORT: RobotModelManager.__init__)."""
    import hashlib
    import json
    import os

    import yaml

    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo._src.types.robot import RobotCfg
    from curobo._src.types.device_cfg import DeviceCfg

    cfg = json.loads(cfg_json)
    robot_config_file = cfg["robot_config_file"]
    if not robot_config_file or not os.path.isfile(robot_config_file):
        raise ValueError(
            "robot_model_build: 'robot_config_file' is not set to a readable "
            f"file (got {robot_config_file!r}). Pass -p robot_config_file:=<path> "
            "or set it via the launch params.")

    with open(robot_config_file) as fh:
        data = yaml.safe_load(fh) or {}

    # PORT: config_manager._load_robot_config / _translate_xrdf - an xrdf config
    # is translated to cuRobo's robot_cfg.kinematics form using the urdf_path /
    # asset_root_path parameters (which are NOT inside the xrdf file).
    if data.get("format") == "xrdf":
        resolved = _translate_xrdf(
            robot_config_file,
            cfg.get("urdf_path", ""),
            cfg.get("asset_root_path", ""))
        if not os.path.isfile(resolved):
            raise ValueError(
                f"xrdf translation of {robot_config_file} produced no resolved "
                f"config at {resolved}")
        kin_cfg = KinematicsCfg.from_robot_yaml_file(resolved)
    else:
        # PORT: robot_description.resolve_curobo_config - resolve the yml's
        # package:// / yml-relative urdf_path + asset_root_path WITHOUT
        # restructuring the document. cuRobo's loader accepts either
        # ``robot_cfg.kinematics.*`` or the kinematics keys at the document ROOT
        # (see curobo's load_robot_yaml fallback), and injecting an empty
        # ``robot_cfg.kinematics`` would hide the root keys from that fallback.
        base = os.path.dirname(os.path.abspath(robot_config_file))
        kinematics = _robot_kinematics_block(data)

        changed = False
        for key in ("urdf_path", "asset_root_path"):
            original = kinematics.get(key)
            if not original:
                continue
            if original.startswith("package://"):
                kinematics[key] = _resolve_uri(original, base)
                changed = True
            elif not os.path.isabs(original):
                candidate = os.path.normpath(os.path.join(base, original))
                if os.path.exists(candidate):
                    kinematics[key] = candidate
                    changed = True

        if changed:
            digest = hashlib.sha1(
                os.path.abspath(robot_config_file).encode()).hexdigest()[:10]
            out_dir = "/tmp/curobo_bridge_resolved"
            os.makedirs(out_dir, exist_ok=True)
            resolved = os.path.join(
                out_dir, f"{os.path.basename(robot_config_file)}-{digest}.yml")
            with open(resolved, "w") as fh:
                yaml.safe_dump(data, fh)
        else:
            # Nothing to resolve: hand cuRobo the ORIGINAL file (its own loader
            # normalizes robot_cfg/kinematics).
            resolved = robot_config_file

        kin_cfg = KinematicsCfg.from_robot_yaml_file(resolved)

    # PORT: RobotModelManager.set_torque_mode / _wrap_robot_cfg - the pinocchio
    # inverse-dynamics model is attached ONLY when the node runs torque-limited
    # (load_dynamics:=true; the launch default is false). This must be gated:
    # with a dynamics model present the rollout computes joint_torque (RNEA
    # forward + backward) inside the cspace cost on EVERY optimizer iteration
    # (see rollout/cost_manager/cost_manager_robot.py, `joint_torque=`), which
    # roughly doubles the cost of pure-kinematics modes such as the standalone
    # IK solver used by the reachability client. The pre-port Python
    # RobotModelManager built `dynamics=None` unless load_dynamics was set.
    load_dynamics = bool(cfg.get("load_dynamics", False))
    dynamics = None
    if load_dynamics:
        dynamics = RobotCfg._create_dynamics_config(
            kinematics_config=kin_cfg.kinematics_config, device_cfg=DeviceCfg())
    robot_cfg = RobotCfg(kinematics=kin_cfg, dynamics=dynamics)
    kin = Kinematics(robot_cfg.kinematics)

    entry = _REGISTRY[handle]
    entry["kin_cfg"] = kin_cfg
    entry["robot_cfg"] = robot_cfg
    entry["kin"] = kin
    entry["joint_names"] = list(kin.joint_names)
    entry["dof"] = int(kin.get_dof())
    # Robot-masking knobs (PORT: RobotSegmentation.__init__, which reads
    # robot_segmentation_distance_threshold / robot_segmentation_mask_margin).
    # robot_segment() looks both up on THIS registry entry; without them it
    # silently fell back to its own 0.05 / 0.0 defaults, so the launch
    # parameters were accepted but had no effect.
    if "seg_distance_threshold" in cfg:
        entry["seg_distance_threshold"] = float(cfg["seg_distance_threshold"])
    if "seg_mask_margin" in cfg:
        entry["seg_mask_margin"] = float(cfg["seg_mask_margin"])
    # The robot YAML's own lock_joints (PORT: RobotModelManager.config_lock_joints).
    entry["resolved_config"] = resolved
    # PORT: the lock_joints map, structure-aware (robot_cfg.kinematics.* or the
    # kinematics keys at the document root).
    _root_kin = _robot_kinematics_block(data)
    _lock_joints = _root_kin.get("lock_joints") or {}
    # A `null` value marks an *unpinned* lock: the joint stays out of the planning
    # cspace but is not pinned, its value coming from the joint state (see
    # robot_model_update_lock_joints). Keep it out of the pinned map so nothing
    # here has to float(None) it.
    entry["lock_joints"] = {
        str(k): float(v) for k, v in _lock_joints.items() if v is not None
    }
    entry["unpinned_lock_joints"] = [
        str(k) for k, v in _lock_joints.items() if v is None
    ]
    # The URDF link the payload is fixed to, used to publish the attached
    # spheres LINK-RELATIVE so a trajectory preview can ride the gripper.
    _extra_links = _root_kin.get("extra_links") or {}
    _attach = _extra_links.get(ATTACH_LINK) or {}
    entry["attach_parent_link"] = (
        _attach.get("parent_link_name") if isinstance(_attach, dict) else None)
    entry["built"] = True


def joint_limits(handle: int):
    """cspace names + position limits + locked joints (PORT: get_joint_info)."""
    import numpy as np

    entry = _REGISTRY[handle]
    kin = entry["kin"]
    limits = kin.get_joint_limits()
    positions = limits.position.detach().cpu().numpy().astype(np.float32)  # [2, dof]
    locked = entry.get("lock_joints", {})
    return {
        "positions": positions,
        "joint_names": list(limits.joint_names),
        "locked_joint_names": list(locked.keys()),
        "locked_positions": [float(v) for v in locked.values()],
        # lock_joints written {joint: null}: out of the cspace but not pinned.
        "unpinned_lock_joint_names": list(entry.get("unpinned_lock_joints", [])),
    }


def fk(handle: int, q):
    """Tool poses for [N, dof] joint positions (PORT: fk_services._compute_poses)."""
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    kin = entry["kin"]
    device = torch.device("cuda", torch.cuda.current_device())

    q_t = torch.tensor(np.asarray(q, dtype=np.float32), dtype=torch.float32, device=device)
    js = JointState.from_position(q_t, joint_names=kin.joint_names)
    kin_state = kin.compute_kinematics(js)

    # [B, H=1, L, 3/4]; first tool frame. Quaternion is wxyz (cuRobo order).
    positions = kin_state.tool_poses.position[:, 0, 0, :].detach().cpu().numpy()
    quaternions = kin_state.tool_poses.quaternion[:, 0, 0, :].detach().cpu().numpy()

    out = np.zeros((positions.shape[0], 1, 7), dtype=np.float32)
    out[:, 0, 0:3] = positions
    out[:, 0, 3:7] = quaternions
    return out


def _build_ik_solver(entry, batch_size, num_seeds, self_collision_check,
                     use_lm_seed, use_cuda_graph):
    """Create the standalone IK solver (PORT: ConfigWrapperIK.build).

    ``num_seeds`` runs that many parallel optimisation trajectories per pose, so
    GPU memory grows with ``batch_size x num_seeds``; the flags bake into the
    solver buffers / CUDA graph at build time.
    """
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.scene import Scene

    batch_size = max(1, int(batch_size))
    num_seeds = max(1, int(num_seeds))
    # Build against an EMPTY scene, never the live one: after a cache shrink the
    # live world may not fit the new cache, and constructing against it raises
    # "Cannot load N cuboids" (observed live, killing a benchmark rebuild). The
    # real world is pushed right after via world_push (PORT: the build tail's
    # update_world(primitives_only_scene())).
    scene = Scene() if self_collision_check else None
    cfg = InverseKinematicsCfg.create(
        robot=entry["robot_cfg"],
        scene_model=scene,
        num_seeds=num_seeds,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        self_collision_check=bool(self_collision_check),
        collision_cache=entry.get("collision_cache") if self_collision_check else None,
        use_cuda_graph=bool(use_cuda_graph),
        max_batch_size=batch_size,
    )
    # use_lm_seed is a dataclass default (True), not a create() kwarg.
    cfg.use_lm_seed = bool(use_lm_seed)
    solver = InverseKinematics(cfg)
    entry["ik"] = solver
    entry["ik_batch_size"] = batch_size
    entry["ik_num_seeds"] = num_seeds
    entry["ik_self_collision_check"] = bool(self_collision_check)
    entry["ik_use_lm_seed"] = bool(use_lm_seed)
    entry["ik_use_cuda_graph"] = bool(use_cuda_graph)
    return solver


def _ik_solver(entry):
    """The registered IK solver, built on demand if ik_config never ran.

    IKServices always warms up (ik_config) before the first solve, so this
    fallback only covers a direct bridge call; it reuses the cached build
    tunables when present, else the node-param defaults (32 seeds, collisions
    on, CUDA graphs on).
    """
    solver = entry.get("ik")
    if solver is None:
        solver = _build_ik_solver(
            entry,
            batch_size=max(1, int(entry.get("ik_batch_size") or 1)),
            num_seeds=max(1, int(entry.get("ik_num_seeds") or 32)),
            self_collision_check=bool(entry.get("ik_self_collision_check", True)),
            use_lm_seed=bool(entry.get("ik_use_lm_seed", True)),
            use_cuda_graph=bool(entry.get("ik_use_cuda_graph", True)),
        )
    return solver


def ik_config(handle: int, batch_size: int, num_seeds: int,
              self_collision_check: bool = True, use_lm_seed: bool = True,
              use_cuda_graph: bool = True) -> None:
    """(Re)build the IK solver (PORT: ConfigWrapperIK.build)."""
    entry = _REGISTRY[handle]
    _build_ik_solver(
        entry, batch_size, num_seeds, self_collision_check, use_lm_seed,
        use_cuda_graph)
    # PORT: the build tail re-pushes the live world when collision is on, so a
    # rebuild (batch-size or param change) does not leave the solver sceneless.
    if self_collision_check:
        world_push(handle)


def ik(handle: int, poses):
    """Solve [N,7] tool poses (PORT: ik_services._solve / _solve_locked)."""
    import numpy as np
    import torch
    from curobo.types import Pose, GoalToolPose

    entry = _REGISTRY[handle]
    solver = _ik_solver(entry)
    device = torch.device("cuda", torch.cuda.current_device())
    poses = np.asarray(poses, dtype=np.float32)

    pose = Pose(
        position=torch.tensor(poses[:, 0:3], dtype=torch.float32, device=device),
        quaternion=torch.tensor(poses[:, 3:7], dtype=torch.float32, device=device),
    )
    tool_frame = solver.kinematics.tool_frames[0]
    goal = GoalToolPose.from_poses({tool_frame: pose})
    result = solver.solve_pose(goal_tool_poses=goal)

    solution = result.solution.detach().cpu().numpy()
    success = result.success.detach().cpu().numpy()
    return {
        "success": bool(np.any(success)),
        "positions": solution.astype(np.float32),
        "valid": success.reshape(-1).astype(int).tolist(),
        "joint_names": list(solver.kinematics.joint_names),
    }


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
    """Per-contact collision attribution at joint state `q`.

    PORT: collision_distance._attributed_collisions +
    _attributed_self_collisions + _attributed_joint_limit_violations. Feeds the
    node's StateCollisions message and the "collision feedback:" failure log.

    Returns {"world": [{link, obstacle, depth}],
             "self":  [{link, link_b, depth}],
             "cspace":[{joint, value, lower, upper}]}.
    """
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
    return checkers


def _reapply_attached_disables(entry) -> None:
    """Re-assert the attached obstacles' disabled flag on every solver.

    PORT: obstacle_manager.reapply_attached_disables. cuRobo's per-obstacle
    ``enable`` flag lives in GPU buffers and a world push resets it to 1
    (``SceneData.load_from_scene_cfg`` clears + re-adds with enable=1), so the
    flag must be re-armed after every push: the grasped obstacle stays
    REGISTERED in the solver scenes while attached, but collision-inert.

    Without this the freshly fitted payload spheres collide with the object's
    own static copy, and every plan from here on starts "in collision" (the
    deployment's post-attach ``close_2`` failure).
    """
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
    AABB, then refresh the ESDF and push the world to the solvers.

    PORT: attachment_services._clear_attached_voxels. Disabling the obstacle BY
    NAME only disables its cuboid collision buffer - the perception voxel layer
    still holds the grasped object's voxels at its world position, so the
    freshly attached spheres collide with the object's own leftover voxels.
    Best-effort: no mapper or a non-cuboid obstacle quietly returns False.
    Returns True when a world push happened (already carries the re-apply).
    """
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


# ---------------------------------------------------------------------------
# M3: planning
# ---------------------------------------------------------------------------

def _warmup_pose(planner, iterations: int = 1, joint_delta: float = 0.2) -> None:
    """Record/refresh the POSE (``plan_pose``) CUDA graphs.

    The pose counterpart of :func:`_warmup_cspace`; mirrors
    :meth:`MotionPlanner.warmup`'s single-goalset branch.
    """
    current = planner.default_joint_state.clone().unsqueeze(0)
    for _ in range(iterations):
        goal = current.clone()
        goal.position[..., 0] = goal.position[..., 0] + joint_delta
        goal_tool_poses = planner.compute_kinematics(goal).tool_poses.as_goal()
        planner.plan_pose(goal_tool_poses, current, max_attempts=1)
        planner.reset_seed()


def _reset_scene_cuda_graphs(planner, entry) -> None:
    """Reset every CUDA graph that captured the shared scene collision buffers.

    ``world_push`` rewrites those buffers (``scene_collision_checker
    .load_collision_model``), and a CUDA graph baked against the old buffers
    faults with "CUDA error: an illegal memory access" when it is replayed.
    cuRobo's ``MotionPlanner`` keeps THREE graphs over the one shared
    ``scene_collision_checker`` - ``ik_solver`` (pose plans only),
    ``trajopt_solver`` and the PRM ``graph_planner`` - and its
    ``SolverCore.reset_cuda_graph`` walks only the trajopt solver, so resetting
    "the solver" silently leaves the other two stale. The auxiliary solvers the
    node owns (IK services, MPC) hold graphs over the same checker too.

    Reset them all; the warmup that follows re-captures the ones this solve
    mode needs. Must be called under the exclusive GPU barrier (capture is
    process-global).
    """
    targets = [
        ("planner.ik_solver", getattr(planner, "ik_solver", None)),
        ("planner.trajopt_solver", getattr(planner, "trajopt_solver", None)),
        ("planner.graph_planner", getattr(planner, "graph_planner", None)),
        ("ik services solver", entry.get("ik")),
        ("mpc_ik_solver", entry.get("mpc_ik_solver")),
    ]
    for name, obj in targets:
        if obj is None or not callable(getattr(obj, "reset_cuda_graph", None)):
            continue
        try:
            obj.reset_cuda_graph()
            if name == "planner.graph_planner":
                # cuRobo's PRMGraphPlanner.reset_cuda_graph() resets only
                # ``auxiliary_rollout`` - but that rollout is built WITHOUT a CUDA
                # graph; the one that HAS a graph (use_cuda_graph_for_rollout) is
                # ``feasibility_rollout``. Walk every rollout so the graph the
                # PRM seeding actually replays gets reset too.
                try:
                    rollouts = obj.get_all_rollout_instances()
                except Exception:
                    rollouts = []
                for rollout in rollouts or []:
                    reset = getattr(rollout, "reset_cuda_graph", None)
                    if callable(reset):
                        reset()
                # ...and its search tree is over the previous world.
                for extra in ("reset_buffer", "reset_seed"):
                    fn = getattr(obj, extra, None)
                    if callable(fn):
                        fn()
        except Exception as exc:
            print(
                f"[curobo_bridge] reset {name} CUDA graph failed ({exc!r}); its "
                f"stale graph may still be replayed", flush=True)


def prepare_solve_mode(handle: int, mode: str) -> None:
    """Make the trajopt solver's CUDA graphs match `mode` ("pose"/"cspace").

    cuRobo captures ONE set of optimizer/rollout CUDA graphs on the first solve
    and replays them blindly afterwards. A pose<->cspace switch changes the goal
    buffer structure (link poses vs an explicit joint goal) under those graphs'
    feet; replaying the stale graph reads freed/rewritten memory and the CUDA
    context dies with "an illegal memory access" - which is exactly the
    ``plan_cspace`` crash this port hit. cuRobo resets the graphs only from
    ``SolverCore.reset_shape`` (never triggered by the plan path) and that reset
    is gated on the ``cuda_graph_reset`` runtime flag.

    So: on a mode change, reset the graphs and re-capture them. The caller MUST
    hold the exclusive GPU barrier (the C++ plan path does), because capture is
    process-global and a concurrent camera/ESDF kernel invalidates it.
    """
    entry = _REGISTRY[handle]
    if entry.get("_solve_mode") == mode:
        return
    planner = entry.get("motion_planner")
    if planner is None:
        return

    _reset_scene_cuda_graphs(planner, entry)
    # Record the flag only once the graphs below were attempted, so a failure
    # keeps retrying on the next request instead of locking in a bad state.
    try:
        if mode == "cspace":
            _warmup_cspace(planner, iterations=1)
        else:
            _warmup_pose(planner, iterations=1)
        # The PRM graph planner seeds the LATER attempts of both solve modes, so
        # exercise it explicitly here: its graph (``feasibility_rollout``) must be
        # re-captured under this exclusive barrier, not lazily inside the next
        # plan where the capture would race the camera/ESDF lane. The per-mode
        # warmups above only reach it when their first attempt happens to fail.
        graph_planner = getattr(planner, "graph_planner", None)
        if graph_planner is not None and callable(getattr(graph_planner, "warmup", None)):
            try:
                # Batch it exactly like the real seeding call does:
                # `_get_graph_seed_trajectories` runs find_path over the trajopt
                # seeds, and a CUDA graph has FIXED shapes - warming with any
                # other batch would make the plan re-capture (or mis-copy) it.
                solver = getattr(planner, "trajopt_solver", None)
                seeds = int(getattr(getattr(solver, "config", None), "num_seeds", 4) or 4)
                graph_planner.warmup(num_warmup_iterations=1, max_batch_size=seeds)
            except Exception as exc:
                # Advisory: must not fail the whole mode switch.
                print(
                    f"[curobo_bridge] graph-planner re-warmup failed ({exc!r}); its "
                    f"graph is captured on first use instead", flush=True)
        planner.reset_seed()
    except Exception as exc:
        print(
            f"[curobo_bridge] prepare_solve_mode: {mode} graph capture failed "
            f"({exc!r})", flush=True)
        return
    entry["_solve_mode"] = mode


def _warmup_cspace(planner, iterations: int = 3, joint_delta: float = 0.2) -> None:
    """Record the joint-space (``plan_cspace``) CUDA graphs at build time.

    Mirrors :meth:`MotionPlanner.warmup`'s pose loop, but exercises ``plan_cspace``
    (and, via ``enable_graph_attempt``, the graph-planner seed path) so every
    graph the joint-space planner replays is captured while the caller holds the
    exclusive GPU barrier. Without this the first joint-space request captures
    them mid-plan next to the camera/ESDF lane, which can fault the CUDA context.

    The start/goal shapes are the ones real requests use (``max_batch_size`` x
    ``num_trajopt_seeds``), so later requests only replay, never re-capture.
    """
    current = planner.default_joint_state.clone().unsqueeze(0)
    for _ in range(iterations):
        goal = current.clone()
        goal.position[..., 0] = goal.position[..., 0] + joint_delta
        try:
            planner.plan_cspace(goal, current, max_attempts=2, enable_graph_attempt=1)
        except TypeError:
            # Older/alternate signature without the graph-seed argument.
            planner.plan_cspace(goal, current, max_attempts=1)
        planner.reset_seed()


def motion_planner_build(handle: int, cfg_json: str) -> None:
    """Build the shared MotionPlanner (PORT: ConfigWrapperMotion.set_motion_gen_config)."""
    import json

    from curobo.content import get_task_configs_path
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Scene
    from curobo._src.util.config_io import join_path, resolve_config

    entry = _REGISTRY[handle]
    try:
        cfg = json.loads(cfg_json)
    except json.JSONDecodeError as exc:
        # Include the offending payload: a malformed hand-built config JSON is
        # otherwise an opaque one-line decode error.
        raise ValueError(
            f"motion_planner_build: invalid config JSON ({exc}); payload={cfg_json!r}"
        ) from exc

    # Build against an EMPTY scene (PORT comment in the Python source): the live
    # world is pushed right after via scene_set/update_world.
    scene = Scene()
    # ...and KEEP it as the entry's live scene. `scene_set` (the only other place
    # that assigns entry["scene"]) runs on the first analytic object mutation, so
    # with no analytic obstacles - the common case, e.g. MoveIt syncing an empty
    # world while the camera sees a wall - the entry had NO scene at all. Then
    # esdf_refresh had nowhere to attach the voxel layer and world_push hit
    # `if scene is None: return False`: the perception world was silently never
    # pushed and the planner collided against nothing.
    entry["scene"] = scene
    kwargs = {
        "robot": entry["robot_cfg"],
        "scene_model": scene,
        "num_ik_seeds": int(cfg.get("num_ik_seeds", 32)),
        "num_trajopt_seeds": int(cfg.get("num_trajopt_seeds", 12)),
        "position_tolerance": 0.005,
        "orientation_tolerance": 0.05,
        "use_cuda_graph": bool(cfg.get("use_cuda_graph", True)),
        "self_collision_check": True,
        "collision_cache": cfg.get("collision_cache"),
        "optimizer_collision_activation_distance": float(
            cfg.get("collision_activation_distance", 0.025)),
        "max_batch_size": int(cfg.get("max_batch_size", 1)),
        "multi_env": False,
        "max_goalset": int(cfg.get("max_goalset", 1)),
    }
    graph_config = None
    if not cfg.get("disable_graph_planner", False):
        graph_config = resolve_config(
            join_path(get_task_configs_path(), "graph_planner/exact_graph_planner.yml"))
        if cfg.get("graph_planner"):
            graph_config["graph_planner"].update(cfg["graph_planner"])
        kwargs["graph_planner_config"] = graph_config
    kwargs.update({
        "ik_optimizer_configs": ["ik/particle_ik.yml", "ik/lbfgs_ik.yml"],
        "ik_transition_model": "ik/transition_ik.yml",
        "metrics_rollout": "metrics_base.yml",
        "trajopt_optimizer_configs": [
            "trajopt/particle_trajopt.yml", "trajopt/lbfgs_bspline_trajopt.yml"],
        "trajopt_transition_model": "trajopt/transition_bspline_trajopt.yml",
        "store_debug": False,
    })

    planner = MotionPlanner(MotionPlannerCfg.create(**kwargs))
    if "interpolation_dt" in cfg:
        try:
            planner.trajopt_solver.config.interpolation_dt = float(cfg["interpolation_dt"])
        except AttributeError:
            pass
    entry["collision_cache"] = cfg.get("collision_cache")
    # PORT: obstacle_manager reads obstacle_collision_mode at startup.
    entry["obstacle_collision_mode"] = str(
        cfg.get("obstacle_collision_mode") or "cuboid")
    # PORT: obstacle_manager.collision_cache["voxel"] is None - SetCollisionCache
    # blox=0 builds the solvers with NO voxel storage. A scene carrying an ESDF
    # layer then raises inside update_world ("Cannot load 1 voxel grids, max
    # cache size is 0"), so world_push degrades to primitives-only exactly like
    # unified_planner_node.update_all_solvers_world does.
    _voxel = (cfg.get("collision_cache") or {}).get("voxel")
    _disabled = bool(_voxel) and int((_voxel or {}).get("layers", 1) or 0) <= 0
    entry["voxel_cache_disabled"] = _disabled
    if _disabled:
        print(
            "[curobo_bridge] voxel collision cache disabled (SetCollisionCache "
            "blox=0): solvers see analytic primitives ONLY, no camera obstacles",
            flush=True)
    planner.warmup()
    # `MotionPlanner.warmup()` only records the POSE (plan_pose) graphs. The
    # joint-space path (plan_cspace -> solve_cspace) records its OWN CUDA graphs
    # lazily on the first joint-space request - i.e. during planning. Capture is
    # process-global, so a concurrent camera/ESDF kernel invalidates it and the
    # next replay faults ("CUDA error: an illegal memory access") and bricks the
    # context for every later request. The node calls motion_planner_build (and
    # rebuild_planner) under the exclusive GPU barrier, so record the cspace
    # graphs HERE, while nothing else is on the GPU.
    try:
        _warmup_cspace(planner)
    except Exception as exc:  # a warmup must never block the build
        print(
            f"[curobo_bridge] cspace CUDA-graph warmup failed ({exc!r}); the first "
            f"joint-space plan will capture its graphs unchecked", flush=True)
    entry["motion_planner"] = planner
    # PORT: config_wrapper_motion.set_motion_gen_config tail - a freshly built
    # solver arrives with every obstacle enabled, so an already-attached payload
    # must have its static copy re-disabled on the new checker.
    _reapply_attached_disables(entry)


def _as_2d_list(tensor):
    import numpy as np

    if tensor is None:
        return None
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype("float32")


def _flat_list(value):
    """PORT: single_planner.SinglePlanner._flat_values (best-effort flatten)."""
    if value is None:
        return []
    try:
        if hasattr(value, "reshape"):
            value = value.reshape(-1)
        if hasattr(value, "tolist"):
            return list(value.tolist())
        return list(value)
    except Exception:
        return []


def _select_goal_index(result) -> int:
    """PORT: single_planner._select_goal_index - the winner candidate index of a
    plan_pose goalset solve (0 when goalset_index is None, -1 on failure)."""
    if result is None:
        return -1
    succ = getattr(result, "success", False)
    if hasattr(succ, "item") and not bool(succ.item()):
        return -1
    if not succ:
        return -1
    idx = getattr(result, "goalset_index", None)
    if idx is None:
        return 0
    try:
        flat = idx.view(-1) if hasattr(idx, "view") else idx
        return int(flat[0].item())
    except (TypeError, ValueError, IndexError):
        return -1


def _select_seed_index(result) -> int:
    """PORT: single_planner._select_seed_index - the winner restart (seed) index
    (seed_rank row 0 argmin, else seed_cost argmin, else 0; -1 on failure)."""
    if result is None:
        return -1
    succ = getattr(result, "success", None)
    if succ is None:
        return -1
    try:
        ok = bool(succ.item()) if hasattr(succ, "item") else bool(succ)
        if not ok:
            return -1
    except Exception:
        return -1
    rank = getattr(result, "seed_rank", None)
    if rank is not None:
        try:
            row0 = rank[0]
            winner = row0.argmin().item() if hasattr(row0, "argmin") else row0[0]
            return int(winner)
        except Exception:
            pass
    cost = getattr(result, "seed_cost", None)
    if cost is not None:
        try:
            row0 = cost[0]
            winner = row0.argmin().item() if hasattr(row0, "argmin") else row0
            return int(winner)
        except Exception:
            pass
    return 0


def considered_rows(result, segment_i=0, fallback_candidate=None):
    """PORT: single_planner.SinglePlanner._segment_considered_rows.

    One row per seed of a solved segment. The whole-task-only fields
    (waypoint_cost/path_length/clearance) stay 0, as in the Python.
    """
    if result is None:
        return []
    success = _flat_list(getattr(result, "success", None))
    costs = _flat_list(getattr(result, "seed_cost", None))
    perr = _flat_list(getattr(result, "position_error", None))
    gidx = getattr(result, "goalset_index", None)
    gidx_flat = _flat_list(gidx)
    n_links = 1
    if gidx is not None and getattr(gidx, "ndim", 0) >= 3:
        try:
            n_links = int(gidx.shape[-1])
        except Exception:
            n_links = 1
    # PORT: single_planner._segment_considered_rows(row, 0,
    # self._selected_goal_indexes[0]) - the fallback candidate for a failed seed
    # is the WINNER candidate index, not 0.
    if fallback_candidate is None:
        fallback_candidate = _select_goal_index(result)
    n_seeds = len(success) if success else 1
    if n_seeds <= 0:
        n_seeds = 1
    solve_time = getattr(result, "solve_time", 0.0)
    rows = []
    for s in range(n_seeds):
        ok = bool(success[s]) if s < len(success) else False
        cost = float(costs[s]) if costs and s < len(costs) else 0.0
        err = float(perr[s]) if perr and s < len(perr) else 0.0
        cand = 0
        if gidx_flat and s * n_links < len(gidx_flat):
            try:
                cand = int(gidx_flat[s * n_links])
            except (TypeError, ValueError):
                cand = 0
        elif not ok:
            cand = max(0, int(fallback_candidate))
        rows.append({
            "problem": 0, "segment": segment_i, "goalset_candidate": cand, "seed": s,
            "success": ok, "cost": cost, "waypoint_cost": 0.0, "path_length": 0.0,
            "clearance": 0.0, "max_waypoint_error": err, "solve_time": float(solve_time),
        })
    return rows


def _position_error_mm(result):
    """Winner candidate's solver-reported position error in mm (NaN if absent)."""
    import math

    try:
        perr = getattr(result, "position_error", None)
        if perr is None:
            return math.nan
        flat = perr.reshape(-1) if hasattr(perr, "reshape") else None
        if flat is None:
            return math.nan
        return float(flat[0].item() if hasattr(flat[0], "item") else flat[0]) * 1000.0
    except Exception:
        return math.nan


def _knot_motion_time(planner, result):
    """PORT: single_planner.SinglePlanner._knot_motion_time.

    Native measures the KNOT solution (``js_solution.dt * (H_knots - offset)``,
    ``offset = 2 * interpolation_steps`` for bspline, else 1) - not the
    interpolated waypoints. NaN when unavailable.
    """
    import math

    try:
        from curobo._src.types.control_space import ControlSpace

        js = getattr(result, "js_solution", None)
        if js is None:
            return math.nan
        h = int(js.position.shape[-2])
        dt = getattr(js, "dt", None)
        if dt is None:
            return math.nan
        if hasattr(dt, "detach"):
            dt = dt.detach()
        if hasattr(dt, "flatten"):
            flat = dt.flatten()
            dt = flat[0] if len(flat) else None
        if hasattr(dt, "item"):
            dt = dt.item()
        dt = float(dt)
        interp_steps = None
        try:
            interp_steps = getattr(planner.trajopt_solver, "interpolation_steps", None)
        except Exception:
            interp_steps = None
        if getattr(js, "control_space", None) in ControlSpace.bspline_types():
            if interp_steps is None or int(interp_steps) <= 0:
                return math.nan
            offset = 2 * int(interp_steps)
        else:
            offset = 1
        return max(0.0, dt * (h - offset))
    except Exception:
        return math.nan


def _plan_outcome(planner, result):
    """PORT: single_planner result shaping + unified_planner._build_planning_stats."""
    import math

    names = list(planner.kinematics.joint_names)
    base = {
        "success": False, "feasible": False, "status": "no solution",
        "positions": None, "velocity": None, "acceleration": None,
        "joint_names": names, "goalset_index": -1, "seed_index": -1,
        "total_time": math.nan,
        # PORT: the solve was attempted with >= 1 seed, so the candidate tally
        # and the (unsolved) segment row are present even on failure.
        "problems": 1, "candidates_generated": 1, "candidates_solved": 0,
        "candidates_pruned": 0, "waypoints_planned": 0,
        "solver_time_s": math.nan, "motion_time_s": math.nan,
        "position_error_mm": math.nan, "waypoint_status": [0],
    }
    if result is None:
        return base

    trajectory = result.get_interpolated_plan()
    # PORT: unified_planner._result_trajectory uses traj.joint_names verbatim
    # (the plan carries the full joint set, fingers included).
    traj_names = getattr(trajectory, "joint_names", None)
    if traj_names:
        names = list(traj_names)

    success = result.success
    try:
        success = bool(success.reshape(-1)[0].item()) if hasattr(success, "reshape") \
            else bool(success)
    except Exception:
        success = False

    positions = _as_2d_list(trajectory.position)
    velocity = _as_2d_list(getattr(trajectory, "velocity", None))
    acceleration = _as_2d_list(getattr(trajectory, "acceleration", None))

    # PORT: single_planner._tally_candidates
    try:
        generated = max(1, int(result.num_seeds))
    except Exception:
        generated = 1
    solved = 1 if success else 0

    # PORT: unified_planner._solver_time_s
    solver_time = math.nan
    try:
        solver_time = float(result.solve_time)
    except Exception:
        try:
            solver_time = float(result.solve_time.detach().flatten()[0].item())
        except Exception:
            try:
                solver_time = float(result.total_time)
            except Exception:
                solver_time = math.nan

    return {
        "success": success,
        "feasible": bool(getattr(result, "feasible", success)),
        "status": str(getattr(result, "status", None) or ""),
        "positions": positions,
        "velocity": velocity,
        "acceleration": acceleration,
        "joint_names": names,
        # PORT: single_planner._selected_goal_indexes / _selected_seed_index
        # (the WINNER candidate / seed, not a constant 0).
        "goalset_index": max(0, _select_goal_index(result)),
        "seed_index": max(0, _select_seed_index(result)),
        "total_time": solver_time,
        "problems": 1,
        "candidates_generated": generated,
        "candidates_solved": solved,
        "candidates_pruned": 0,
        "waypoints_planned": 1,
        "solver_time_s": solver_time,
        "motion_time_s": _knot_motion_time(planner, result),
        # PORT: single_planner._winner_position_error_mm - the winner candidate's
        # solver-reported position error. The Python reads it from the considered
        # rows' max_waypoint_error (== result.position_error per seed); the
        # native leg's own metric is ``result.position_error * 1000`` and that is
        # what the service always exposes here.
        "position_error_mm": _position_error_mm(result),
        "waypoint_status": [1] if success else [],
        "considered": considered_rows(result),
    }


def collision_cache_bucket_counts(handle: int) -> dict:
    """PORT: obstacle_manager._solver_bucket_counts - (cuboid, mesh) slots the
    current scene occupies under the default 'cuboid' collision mode: cuboids
    stay cuboids, sphere/capsule/cylinder ride the cuboid bucket, meshes stay
    meshes (attached, disabled obstacles still occupy slots).
    """
    entry = _REGISTRY[handle]
    scene = entry.get("scene")
    if scene is None:
        return {"cuboid": 0, "mesh": 0}
    n_cuboid = len(getattr(scene, "cuboid", None) or [])
    n_mesh = len(getattr(scene, "mesh", None) or [])
    converted = sum(
        len(getattr(scene, bucket, None) or [])
        for bucket in ("sphere", "capsule", "cylinder")
    )
    return {"cuboid": int(n_cuboid + converted), "mesh": int(n_mesh)}


def render_plan_image(position, velocity, acceleration, names, dt):
    """PORT: plan_plot.render_plan_plot - multi-panel joint-trajectory RGB image.

    position/velocity/acceleration: [T, D] float32 arrays (velocity/acceleration
    may be empty). Returns an HxWx3 uint8 array, or None when matplotlib/PIL are
    unavailable (the caller then skips publishing, as the Python does).
    """
    import numpy as np

    try:
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage
    except Exception:
        return None

    pos = np.asarray(position, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    n, dof = pos.shape

    vel = np.asarray(velocity, dtype=float) if velocity is not None and len(velocity) else None
    if vel is not None:
        if vel.ndim == 1:
            vel = vel.reshape(1, -1)
        vel = vel[:n, :dof]
    else:
        vel = np.zeros((n, dof))
    d = max(float(dt), 1e-6)
    if not np.any(vel):
        vel = np.diff(pos, axis=0, prepend=pos[:1]) / d
    acc = np.diff(vel, axis=0, prepend=vel[:1]) / d
    jrk = np.diff(acc, axis=0, prepend=acc[:1]) / d

    names = list(names or [f"J{i}" for i in range(dof)])
    plot_data = [(pos, "Position (rad)"), (vel, "Velocity (rad/s)"),
                 (acc, "Accel (rad/s^2)"), (jrk, "Jerk (rad/s^3)")]
    n_plots = len(plot_data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(6, 2 * n_plots), dpi=100, sharex=True)
    if n_plots == 1:
        axes = [axes]
    t = np.arange(n) * d
    for ax, (data, ylabel) in zip(axes, plot_data):
        for j in range(dof):
            label = names[j] if j < len(names) else f"J{j}"
            if len(label) > 10:
                label = label[:8] + ".."
            ax.plot(t, data[:, j], linewidth=1.2, label=label)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return np.array(PILImage.open(buf).convert("RGB"))


def torch_synchronize() -> None:
    """PORT: unified_planner_node.torch_sync_enabled - CPU/GPU ordering bridge.

    Off by default (torch_sync param): a per-frame synchronize() blocks the
    worker on the GPU and starves the viz timers.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_seed(handle: int) -> None:
    """PORT: MotionPlanner.reset_seed (reset_seed_per_plan)."""
    entry = _REGISTRY[handle]
    planner = entry.get("motion_planner")
    if planner is not None:
        planner.reset_seed()


def robot_model_rebuild_locks(handle: int, lock_json: str) -> None:
    """Rebuild the shared Kinematics with a lock_joints override.

    PORT: robot_model_manager.set_lock_joints. The solvers built from the robot
    cfg must be rebuilt afterwards (motion_planner_build).
    """
    import json

    from curobo.kinematics import Kinematics
    from curobo._src.types.robot import RobotCfg

    entry = _REGISTRY[handle]
    locks = json.loads(lock_json)
    from curobo.kinematics import KinematicsCfg
    # NB: entry["robot_config_file"] is NOT recorded (the build stores the
    # RESOLVED path); an eager .get(default) here raised KeyError on every
    # joint-lock change. Resolve lazily and say so if neither is on record.
    resolved = entry.get("resolved_config") or entry.get("robot_config_file")
    if not resolved:
        raise ValueError(
            "robot_model_rebuild_locks: no resolved robot config on record; "
            "build the robot model first")
    # Empty locks -> rebuild from the YAML's OWN lock_joints (restore). A
    # non-empty override REPLACES the YAML's lock_joints, so the YAML's unpinned
    # locks (`lock_joints: {joint: null}`) must be merged back in - they are not
    # part of an override map, but dropping them would leave those joints neither
    # in the cspace nor locked and the model would refuse to build.
    if not locks:
        kin_cfg = KinematicsCfg.from_robot_yaml_file(resolved)
    else:
        _unpinned = {name: None for name in entry.get("unpinned_lock_joints", [])}
        kin_cfg = KinematicsCfg.from_robot_yaml_file(
            resolved, lock_joints={**_unpinned, **locks})
    robot_cfg = RobotCfg(
        kinematics=kin_cfg, dynamics=entry["robot_cfg"].dynamics)
    entry["kin_cfg"] = kin_cfg
    entry["robot_cfg"] = robot_cfg
    entry["kin"] = Kinematics(robot_cfg.kinematics)
    entry["joint_names"] = list(entry["kin"].joint_names)
    entry["dof"] = int(entry["kin"].get_dof())
    entry["unpinned_lock_joints"] = list(
        entry["kin"].config.kinematics_config.unpinned_lock_joints or [])
    if locks:
        entry["lock_joints"] = dict(locks)
    else:
        # Restore: the model now carries the YAML's OWN lock_joints, and
        # joint_limits() reports this map - so report the YAML's values, not
        # an empty override (which would claim nothing is locked while the
        # model has them baked in).
        try:
            import yaml as _yaml
            with open(resolved) as _fh:
                _cfg = _yaml.safe_load(_fh) or {}
            _kin = _robot_kinematics_block(_cfg)
            entry["lock_joints"] = {
                str(k): float(v)
                for k, v in (_kin.get("lock_joints") or {}).items()
                if v is not None
            }
        except Exception:
            entry["lock_joints"] = {}
    entry["motion_planner"] = None
    entry["ik"] = None


def robot_model_update_lock_joints(handle: int, lock_json: str) -> None:
    """Update the values of the robot's unpinned lock joints, in place.

    PORT: robot_model_manager.update_lock_joints. These are the joints written
    ``lock_joints: {joint: null}`` in the robot YAML: locked out of the planning
    cspace (never planned) but NOT pinned, so their value follows the joint state.
    The baked transform of each joint's own link is rewritten in place, so no model
    re-parse and no solver rebuild is needed (unlike robot_model_rebuild_locks).
    """
    import json

    entry = _REGISTRY[handle]
    values = {str(k): float(v) for k, v in json.loads(lock_json).items()}
    if not values:
        return
    # The shared Kinematics drives FK and segmentation. The solvers keep their own
    # kinematics, so push the same rewrite into them where they expose one.
    targets = [entry["kin"]]
    planner = entry.get("motion_planner")
    if planner is not None:
        targets.append(getattr(planner, "kinematics", None))
    ik = entry.get("ik")
    if ik is not None:
        targets.append(getattr(ik, "kinematics", ik))
    updated = set()
    for target in targets:
        config = getattr(getattr(target, "config", None), "kinematics_config", None)
        if config is None or id(config) in updated:
            continue
        updated.add(id(config))
        target.update_lock_joints(values)


def plan_cspace(handle: int, req) -> dict:
    """Joint-space plan (PORT: joint_space_planner._plan_trajectory)."""
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    planner = entry["motion_planner"]
    device = torch.device("cuda", torch.cuda.current_device())

    start = torch.tensor(
        [list(np.asarray(req["start"], dtype=np.float32))], dtype=torch.float32, device=device)
    goal = torch.tensor(
        [list(np.asarray(req["joint_target"], dtype=np.float32))],
        dtype=torch.float32, device=device)
    names = planner.kinematics.joint_names
    start_state = JointState.from_position(start, joint_names=names)
    goal_state = JointState.from_position(goal, joint_names=names)
    result = planner.plan_cspace(
        goal_state, start_state,
        max_attempts=int(req.get("max_attempts", 1)),
        enable_graph_attempt=int(req.get("enable_graph_attempt", 1)))
    return _plan_outcome(planner, result)


def plan_pose(handle: int, req) -> dict:
    """Pose plan (PORT: single_planner._plan_trajectory / classic_planner)."""
    import numpy as np
    import torch
    from curobo.types import JointState, Pose, GoalToolPose

    entry = _REGISTRY[handle]
    planner = entry["motion_planner"]
    device = torch.device("cuda", torch.cuda.current_device())

    start = torch.tensor(
        [list(np.asarray(req["start"], dtype=np.float32))], dtype=torch.float32, device=device)
    start_state = JointState.from_position(start, joint_names=planner.kinematics.joint_names)

    goalset_arg = req.get("goalset")
    if goalset_arg is None or len(goalset_arg) == 0:
        raise ValueError(
            "plan_pose requires a 'goalset' of [x,y,z,qw,qx,qy,qz] poses; got no "
            "poses (a joint-space goal must go through plan_cspace / the "
            "joint-space planner).")
    goalset = np.asarray(goalset_arg, dtype=np.float32)
    pose = Pose(
        position=torch.tensor(goalset[:, 0:3], dtype=torch.float32, device=device),
        quaternion=torch.tensor(goalset[:, 3:7], dtype=torch.float32, device=device))
    tool_frame = planner.tool_frames[0]
    goal = GoalToolPose.from_poses({tool_frame: pose}, num_goalset=len(goalset))

    # PORT: single_planner._apply_pose_constraints - hold the requested Cartesian
    # axes along the whole path (Goalset.trajectory_constraints order
    # [theta_x, theta_y, theta_z, x, y, z] -> ToolPoseCriteria order
    # [x, y, z, roll, pitch, yaw]). This is what makes a CartesianPath solve
    # trace the straight line; without it the solve curves and the achieved
    # fraction check rejects it. The MotionPlanner is SHARED, so the default
    # (unconstrained) criteria are restored right after the solve.
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

    constraints = [int(c) for c in (req.get("trajectory_constraints") or [])]
    applied = len(constraints) == 6 and any(c == 1 for c in constraints)
    if applied:
        tx, ty, tz, cx, cy, cz = (1.0 if c == 1 else 0.0 for c in constraints)
        planner.update_tool_pose_criteria({
            tool_frame: ToolPoseCriteria(
                non_terminal_pose_axes_weight_factor=[cx, cy, cz, tx, ty, tz])})
    try:
        result = planner.plan_pose(
            goal, start_state,
            max_attempts=int(req.get("max_attempts", 1)),
            enable_graph_attempt=int(req.get("enable_graph_attempt", 1)))
    finally:
        if applied:
            planner.update_tool_pose_criteria({tool_frame: ToolPoseCriteria()})
    return _plan_outcome(planner, result)


# ---------------------------------------------------------------------------
# M4: perception (Mapper TSDF/ESDF, depth integration, voxel grid)
# ---------------------------------------------------------------------------

# PORT: obstacle_manager._load_perception_params defaults.
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
    """PORT: obstacle_manager._collision_supported.


    * ``cuboid`` (default, the deployment's): sphere/capsule/cylinder -> OBB via
      ``get_cuboid()`` - the same approximation native cuRobo makes for the
      primitives. Cuboids, USER MESHES (deliberate exact geometry) and the voxel
      layer pass through unchanged.
    * ``mesh`` (legacy): ``SceneCfg.create_collision_support_world`` - primitives
      become trimesh meshes (exact round geometry, slower kernels).
    """
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


def warmup_mapper(handle: int, iterations: int = 2) -> None:
    """Force the Mapper's Warp kernels to compile AND LOAD now.

    Warp JIT-compiles one module per kernel specialization and loads it lazily
    with ``cuModuleLoadData``. A load that happens while the plan lane is
    capturing/replaying a CUDA graph fails with
    ``Warp CUDA error 716: misaligned address (wp_cuda_load_module)`` and poisons
    the CUDA context - the ``recycle_empty_blocks_kernel_bsN_<hash>`` failure
    seen during depth integration. The recycle/decay kernels only load once
    enough frames need block recycling, i.e. long after startup and right in the
    middle of planning.

    Warm them here by integrating a synthetic frame (which runs the decay path),
    with the caller holding the exclusive GPU barrier so nothing else is on the
    GPU, then clear the synthetic data back out.
    """
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


# ---------------------------------------------------------------------------
# M5: reactive control (MPC)
# ---------------------------------------------------------------------------

# PORT: mpc_planner._MPPI_CSPACE_REGULARIZATION
_MPPI_CSPACE_REGULARIZATION = [0.3, 1.0, 0.0, 0.0, 0.0]


def _build_mppi_transition_model(step_dt, horizon, interpolation_steps=4):
    """PORT: mpc_planner._build_mppi_transition_model."""
    return {
        "transition_model_cfg": {
            "control_space": "ACCELERATION",
            "dt_traj_params": {"base_dt": step_dt, "base_ratio": 1.0, "max_dt": step_dt},
            "horizon": horizon,
            "interpolation_steps": interpolation_steps,
            "n_knots": 0,
            "state_filter_cfg": {
                "enable": True,
                "filter_coeff": {
                    "position": 0.1, "velocity": 0.1, "acceleration": 0.0, "jerk": 0.0},
            },
            "teleport_mode": False,
            "vel_scale": 1.0,
            "return_full_act_buffer": True,
        }
    }


def _build_mppi_optimizer_config(num_iters, num_particles=800):
    """PORT: mpc_planner._build_mppi_optimizer_config."""
    return {
        "rollout": {
            "cost_cfg": {
                "cspace_cfg": {
                    "activation_distance": [0.01] * 5,
                    "squared_l2_regularization_weight": [0.0] * 5,
                    "weight": [10.0, 0.0, 0.0, 0.0, 0.0],
                    "cost_type": "STATE",
                    "retime_weights": False,
                    "retime_regularization_weights": True,
                    "cspace_target_weight": 0.0,
                    "cspace_non_terminal_weight_factor": 0.05,
                },
                "tool_pose_cfg": {
                    "use_lie_group": True,
                    "weight": [5000.0, 2000.0],
                    "_terminal_pose_convergence_tolerance": [0.0, 0.0],
                },
            },
            "constraint_cfg": {
                "scene_collision_cfg": {
                    "activation_distance": 0.12,
                    "use_speed_metric": True,
                    "use_sweep": True,
                    "use_sweep_kernel": True,
                    "weight": 10000.0,
                },
                "self_collision_cfg": {"weight": 100000.0},
            },
        },
        "optimizer": {
            "solver_type": "mppi",
            "solver_name": "mppi",
            "num_iters": num_iters,
            "inner_iters": 1,
            "num_particles": num_particles,
            "null_act_frac": 0.05,
            "beta": 0.1,
            "gamma": 0.98,
            "kappa": 0.0001,
            "init_cov": 0.05,
            "update_cov": True,
            "cov_type": "DIAG_A",
            "step_size_mean": 0.9,
            "step_size_cov": 0.01,
            "sample_mode": "BEST",
            "squash_fn": "CLAMP",
            "base_action": "REPEAT",
            "random_mean": False,
            "seed": 0,
            "sample_per_problem": True,
            "sample_params": {
                "fixed_samples": True,
                "n_knots": 5,
                "filter_coeffs": [0.3, 0.3, 0.4],
                "sample_ratio": {
                    "halton": 0.3,
                    "halton-knot": 0.7,
                    "random": 0.0,
                    "random-knot": 0.0,
                    "stomp": 0.0,
                },
                "seed": 0,
            },
            "store_debug": False,
            "sync_cuda_time": True,
            "use_coo_sparse": True,
        },
    }


def _build_metrics_rollout_cfg(cost_cfg_source):
    """PORT: mpc_planner._build_metrics_rollout_cfg."""
    import copy

    from curobo.content import get_task_configs_path
    from curobo._src.util.config_io import join_path, resolve_config

    metrics_cfg = copy.deepcopy(
        resolve_config(join_path(get_task_configs_path(), "metrics_base.yml")))
    metrics_cfg["rollout"]["cost_cfg"] = {
        "tool_pose_cfg": copy.deepcopy(cost_cfg_source["tool_pose_cfg"]),
        "cspace_cfg": copy.deepcopy(cost_cfg_source["cspace_cfg"]),
    }
    return metrics_cfg


def mpc_build(handle: int, cfg_json: str) -> None:
    """Build the ModelPredictiveControl (PORT: mpc_planner.MPCController.build_solver)."""
    import json

    from curobo.content import get_task_configs_path
    from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
    from curobo.scene import Scene
    from curobo._src.util.config_io import join_path, resolve_config

    entry = _REGISTRY[handle]
    cfg = json.loads(cfg_json)

    step_dt = float(cfg["step_dt"])
    horizon = int(cfg["horizon"])
    warm_iters = int(cfg["warm_start_iters"])
    cold_iters = int(cfg["cold_start_iters"])
    solver_type = str(cfg["solver_type"])

    # Built against an EMPTY scene; the real world is pushed via mpc_load_scene
    # (see the Python source comment). No perception voxel layer in the ctor.
    base_kwargs = dict(
        robot=entry["robot_cfg"],
        scene_model=Scene(),
        optimization_dt=step_dt,
        use_cuda_graph=bool(cfg.get("use_cuda_graph", True)),
        self_collision_check=True,
        collision_cache=cfg.get("collision_cache") or entry.get("collision_cache"),
        store_debug=False,
        warm_start_optimization_num_iters=warm_iters,
        cold_start_optimization_num_iters=cold_iters,
    )

    entry["mpc_use_mppi_acceleration"] = (solver_type == "mppi_acceleration")
    entry["mpc_step_dt"] = step_dt
    if entry["mpc_use_mppi_acceleration"]:
        num_particles = int(cfg.get("mppi_num_particles", 400))
        mppi_optimizer_cfg = _build_mppi_optimizer_config(warm_iters, num_particles)
        mpccfg = ModelPredictiveControlCfg.create(
            optimizer_configs=[mppi_optimizer_cfg],
            transition_model=_build_mppi_transition_model(step_dt, horizon),
            squared_l2_regularization_weight=_MPPI_CSPACE_REGULARIZATION,
            metrics_rollout=_build_metrics_rollout_cfg(
                mppi_optimizer_cfg["rollout"]["cost_cfg"]),
            **base_kwargs,
        )
    else:
        lbfgs_cost_cfg = resolve_config(
            join_path(get_task_configs_path(), "mpc/lbfgs_mpc.yml"))["rollout"]["cost_cfg"]
        mpccfg = ModelPredictiveControlCfg.create(
            num_control_points=horizon,
            metrics_rollout=_build_metrics_rollout_cfg(lbfgs_cost_cfg),
            **base_kwargs,
        )
    solver = ModelPredictiveControl(mpccfg)
    entry["mpc"] = solver
    entry["mpc_ik_solver"] = None
    entry["mpc_goal"] = None
    entry["mpc_goal_applied"] = False


def _mpc(entry):
    solver = entry.get("mpc")
    if solver is None:
        raise ValueError("MPC solver not built; call mpc_build first")
    return solver


def _mpc_goal_tool_pose(solver, raw):
    """PORT: reactive_controller._set_target."""
    from curobo.types import GoalToolPose, Pose
    import numpy as np
    import torch

    device = torch.device("cuda", torch.cuda.current_device())
    pose = Pose.from_list(list(raw))
    return GoalToolPose.from_poses(
        {solver.tool_frames[0]: pose},
        ordered_tool_frames=solver.tool_frames,
        num_goalset=1,
    )


def _mpc_ensure_ik_solver(entry):
    """PORT: mpc_planner.MPCController._ensure_ik_solver."""
    ik = entry.get("mpc_ik_solver")
    if ik is not None:
        return ik
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

    ikcfg = InverseKinematicsCfg.create(
        robot=entry["robot_cfg"],
        scene_model=None,
        num_seeds=20,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        self_collision_check=True,
        use_cuda_graph=False,
    )
    ik = InverseKinematics(ikcfg)
    entry["mpc_ik_solver"] = ik
    return ik


def mpc_solve_goal_state(handle: int, goal_raw, seed_q):
    """PORT: mpc_planner.MPCController._solve_goal_state (multi-seed IK anchor)."""
    import numpy as np
    import torch
    from curobo.types import GoalToolPose, JointState, Pose

    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        ik = _mpc_ensure_ik_solver(entry)
        raw = list(goal_raw)
        pose = Pose(
            position=torch.tensor([raw[0:3]], dtype=torch.float32, device=device),
            quaternion=torch.tensor([raw[3:7]], dtype=torch.float32, device=device),
        )
        goal = GoalToolPose.from_poses({ik.kinematics.tool_frames[0]: pose})
        seed_state = None
        seed = np.asarray(seed_q, dtype=np.float32).reshape(1, -1)
        if seed.size:
            seed_state = JointState.from_position(
                torch.tensor(seed[:, :solver.action_dim], dtype=torch.float32, device=device),
                joint_names=solver.joint_names,
            )
        result = ik.solve_pose(goal_tool_poses=goal, current_state=seed_state)
        if not bool(result.success.reshape(-1)[0].item()):
            return {"success": False, "solution": None}
        sol = result.solution.reshape(-1, solver.action_dim)[0:1].clone()
        return {"success": True, "solution": sol.detach().cpu().numpy().astype(np.float32)}
    except Exception:
        return {"success": False, "solution": None}


def _mpc_apply_goal(entry, solver, goal, raw, seed_q):
    """PORT: mpc_planner.MPCController._apply_goal."""
    if solver.update_goal_tool_poses(goal, run_ik=False):
        return True
    solved = mpc_solve_goal_state(entry, raw, seed_q if seed_q is not None else [])
    if solved["success"]:
        import numpy as np
        import torch
        from curobo.types import JointState

        device = torch.device("cuda", torch.cuda.current_device())
        solver.update_goal_tool_poses(goal, run_ik=False)
        sol = torch.tensor(
            np.asarray(solved["solution"], dtype=np.float32), dtype=torch.float32,
            device=device)
        solver.update_goal_state(
            JointState.from_position(sol, joint_names=solver.joint_names))
        solver.enable_joint_position_tracking()
        return True
    solver.update_goal_tool_poses(goal, run_ik=False)
    return False


def mpc_setup(handle: int, start_q):
    """PORT: mpc_planner.MPCController.setup (solver.setup(start_state) + EE).

    The goal is applied separately (mpc_update_goal_pose / _state), mirroring
    bridge_api.md section 7."""
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    device = torch.device("cuda", torch.cuda.current_device())

    start = torch.tensor(
        np.asarray(start_q, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32, device=device)
    start_state = JointState.from_position(start, joint_names=solver.joint_names)
    solver.setup(start_state)

    ee_position = None
    try:
        kin = solver.compute_kinematics(start_state)
        ee_position = [
            float(v) for v in kin.tool_poses.position.reshape(-1, 3)[0].cpu().tolist()]
    except Exception:
        ee_position = None
    return {"ee_position": ee_position}


def mpc_update_goal_pose(handle: int, goal_raw, run_ik: bool) -> bool:
    """PORT: mpc_planner.MPCController._apply_goal -> update_goal_tool_poses."""
    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    raw = list(goal_raw)
    goal = _mpc_goal_tool_pose(solver, raw)
    entry["mpc_goal"] = raw
    applied = bool(solver.update_goal_tool_poses(goal, run_ik=bool(run_ik)))
    entry["mpc_goal_applied"] = applied
    return applied


def mpc_update_goal_state(handle: int, q) -> None:
    """PORT: mpc_planner.MPCController._apply_goal joint branch."""
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    device = torch.device("cuda", torch.cuda.current_device())
    sol = torch.tensor(
        np.asarray(q, dtype=np.float32).reshape(1, -1), dtype=torch.float32, device=device)
    solver.update_goal_state(JointState.from_position(sol, joint_names=solver.joint_names))
    solver.enable_joint_position_tracking()


def mpc_reseed(handle: int, q) -> None:
    """PORT: mpc_planner.MPCController._reseed_toward_goal."""
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    device = torch.device("cuda", torch.cuda.current_device())
    sol = torch.tensor(
        np.asarray(q, dtype=np.float32).reshape(1, -1), dtype=torch.float32, device=device)
    solver.update_seed_trajectory_from_goal_state(
        JointState.from_position(sol, joint_names=solver.joint_names))


def _mpc_state_from_dict(state):
    import numpy as np
    import torch
    from curobo.types import JointState

    device = torch.device("cuda", torch.cuda.current_device())
    pos = torch.tensor(
        np.asarray(state["position"], dtype=np.float32).reshape(1, -1),
        dtype=torch.float32, device=device)
    js = JointState.from_position(pos, joint_names=None)
    for key in ("velocity", "acceleration"):
        if state.get(key) is not None:
            val = np.asarray(state[key], dtype=np.float32)
            if val.size:
                setattr(js, key, torch.tensor(
                    val.reshape(1, -1), dtype=torch.float32, device=device))
    return js


def _mpc_js_to_dict(js):
    """{position, velocity, acceleration} as float32 [n, dof] lists (or None)."""
    import numpy as np

    def _arr(t):
        if t is None:
            return None
        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32)

    if js is None:
        return None
    return {
        "position": _arr(js.position),
        "velocity": _arr(getattr(js, "velocity", None)),
        "acceleration": _arr(getattr(js, "acceleration", None)),
    }


def mpc_cost_breakdown(handle: int) -> dict:
    """PORT: mpc_planner.MPCController._cost_breakdown.

    Per-term COST and CONSTRAINT values (horizon-summed) for the last solve,
    read from the metrics rollout's already-populated _current_metrics (never
    rebatched - see the crash-safety note in the Python source).
    """
    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    try:
        m = solver.trajectory_execution_manager.get_current_metrics()
        if m is None:
            return {}
        cc = m.costs_and_constraints
        out = {}
        for name, val in zip(cc.costs.names, cc.costs.values):
            if name == "tool_pose":
                out["cost_tool_pose_pos"] = float(val[0, :, 0].sum().item())
                out["cost_tool_pose_orient"] = float(val[0, :, 1].sum().item())
            else:
                out[f"cost_{name}"] = float(val.sum().item())
        for name, val in zip(cc.constraints.names, cc.constraints.values):
            if name == "cspace":
                out["con_cspace_bound"] = float(val.sum().item())
            else:
                out[f"con_{name}"] = float(val.sum().item())
        return out
    except Exception:
        return {}


def mpc_step(handle: int, state, paced: bool):
    """PORT: mpc_planner.MPCController.step / step_paced."""
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    current_state = _mpc_state_from_dict(state)
    # joint names were required for the solver to label the state.
    current_state = JointState.from_position(
        current_state.position, joint_names=solver.joint_names)

    if paced:
        result = solver.optimize_action_sequence(current_state)
        action = getattr(result, "action_sequence", None)
    else:
        result = solver.optimize_next_action(current_state)
        action = getattr(result, "next_action", None)

    ee_position = None
    try:
        kin = solver.compute_kinematics(current_state)
        ee_position = [
            float(v) for v in kin.tool_poses.position.reshape(-1, 3)[0].cpu().tolist()]
    except Exception:
        ee_position = None

    if action is None or getattr(action, "position", None) is None:
        action_dict = None
    else:
        action_dict = _mpc_js_to_dict(action.clone())

    horizon_position = None
    try:
        state_seq = getattr(result, "robot_state_sequence", None)
        js = getattr(state_seq, "joint_state", None) if state_seq is not None else None
        if js is not None and getattr(js, "position", None) is not None:
            pos = js.position
            while pos.ndim > 2:
                pos = pos[0]
            if pos.ndim == 1:
                pos = pos.unsqueeze(0)
            names = list(getattr(js, "joint_names", None) or [])
            try:
                from curobo.types import JointState
                probe = JointState.from_position(pos, joint_names=names)
                active = solver.kinematics.get_active_js(probe)
                pos = active.position
            except Exception:
                pass
            horizon_position = pos.detach().cpu().numpy().astype("float32")
    except Exception:
        horizon_position = None

    entry["mpc_last_result"] = result
    # PORT: mpc_planner._publish_costs reads result.position_error /
    # rotation_error for the deprecated pose_* fields of MpcCosts.
    def _first(tensor):
        try:
            flat = tensor.reshape(-1)
            return float(flat[0].item()) if flat.numel() else None
        except Exception:
            return None

    return {
        "action": action_dict,
        "ee_position": ee_position,
        "horizon_position": horizon_position,
        "pose_pos_err_m": _first(getattr(result, "position_error", None)),
        "pose_rot_err_rad": _first(getattr(result, "rotation_error", None)),
    }


def mpc_load_scene(handle: int, scene_json: str) -> None:
    """PORT: mpc_planner.MPCController.update_world."""
    import json

    from curobo.scene import Scene

    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    specs = json.loads(scene_json) if scene_json else []
    scene = Scene()
    for spec in specs:
        primitive = _to_scene(spec)
        attr = {0: "cuboid", 1: "sphere", 2: "capsule", 3: "cylinder", 4: "mesh"}[
            int(spec["type"])]
        bucket = getattr(scene, attr)
        if bucket is None:
            setattr(scene, attr, [primitive])
        else:
            bucket.append(primitive)
    solver.scene_collision_checker.load_collision_model(scene)


def mpc_metrics(handle: int) -> str:
    """PORT: mpc_planner.MPCController._cost_breakdown (as a JSON dict)."""
    import json

    entry = _REGISTRY[handle]
    solver = _mpc(entry)
    try:
        m = solver.trajectory_execution_manager.get_current_metrics()
        if m is None:
            return "{}"
        cc = m.costs_and_constraints
        out = {}
        for name, val in zip(cc.costs.names, cc.costs.values):
            if name == "tool_pose":
                out["cost_tool_pose_pos"] = float(val[0, :, 0].sum().item())
                out["cost_tool_pose_orient"] = float(val[0, :, 1].sum().item())
            else:
                out[f"cost_{name}"] = float(val.sum().item())
        for name, val in zip(cc.constraints.names, cc.constraints.values):
            if name == "cspace":
                out["con_cspace_bound"] = float(val.sum().item())
            else:
                out[f"con_{name}"] = float(val.sum().item())
        return json.dumps(out)
    except Exception:
        return "{}"


# ---------------------------------------------------------------------------
# M5: reactive control (retarget / teleop)
# ---------------------------------------------------------------------------

def retarget_build(handle: int, cfg_json: str) -> None:
    """Build the MotionRetargeter (PORT: retarget_controller.build_solver)."""
    import json

    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
    from curobo.motion_retargeter import MotionRetargeter, MotionRetargeterCfg
    from curobo.scene import Scene

    entry = _REGISTRY[handle]
    cfg = json.loads(cfg_json)
    tool = cfg.get("tool_frame") or entry["kin"].tool_frames[0]
    w_pos = float(cfg.get("position_weight", 1.0))
    w_rot = float(cfg.get("orientation_weight", 1.0))
    solver = MotionRetargeter(MotionRetargeterCfg.create(
        robot=entry["robot_cfg"],
        tool_pose_criteria={
            tool: ToolPoseCriteria.track_position_and_orientation(
                xyz=[w_pos, w_pos, w_pos], rpy=[w_rot, w_rot, w_rot],
            )
        },
        scene_model=Scene(),
        use_mpc=bool(cfg.get("use_mpc", False)),
        num_seeds_global=64,
        self_collision_check=True,
    ))
    entry["retargeter"] = solver
    entry["retarget_tool"] = tool


def _retargeter(entry):
    solver = entry.get("retargeter")
    if solver is None:
        raise ValueError("Retargeter not built; call retarget_build first")
    return solver


def retarget_reset(handle: int) -> None:
    """PORT: retarget_controller.setup -> solver.reset()."""
    _retargeter(_REGISTRY[handle]).reset()


def retarget_step(handle: int, goal_raw):
    """PORT: retarget_controller.step -> solve_frame + FK position error input."""
    import numpy as np

    from curobo.types import GoalToolPose, Pose

    entry = _REGISTRY[handle]
    solver = _retargeter(entry)
    raw = list(goal_raw)
    pose = Pose.from_list(raw)
    tool = entry.get("retarget_tool") or solver.kinematics.tool_frames[0]
    goal = GoalToolPose.from_poses(
        {tool: pose},
        ordered_tool_frames=list(solver.kinematics.tool_frames),
        num_goalset=1,
    )
    result = solver.solve_frame(goal)
    js = result.joint_state
    ee_position = None
    try:
        kin = solver.kinematics.compute_kinematics(js)
        ee_position = [
            float(v) for v in kin.tool_poses.position.reshape(-1, 3)[0].cpu().tolist()]
    except Exception:
        ee_position = None
    return {"action": _mpc_js_to_dict(js), "ee_position": ee_position}


def retarget_load_scene(handle: int, scene_json: str) -> None:
    """PORT: retarget_controller.update_world (IK solvers' update_world)."""
    import json

    from curobo.scene import Scene

    entry = _REGISTRY[handle]
    solver = _retargeter(entry)
    specs = json.loads(scene_json) if scene_json else []
    scene = Scene()
    for spec in specs:
        primitive = _to_scene(spec)
        attr = {0: "cuboid", 1: "sphere", 2: "capsule", 3: "cylinder", 4: "mesh"}[
            int(spec["type"])]
        bucket = getattr(scene, attr)
        if bucket is None:
            setattr(scene, attr, [primitive])
        else:
            bucket.append(primitive)
    global_ik = getattr(solver, "_global_ik_solver", None)
    if global_ik is not None:
        global_ik.update_world(scene)
    local_ik = getattr(solver, "_local_ik_solver", None)
    if local_ik is not None:
        local_ik.update_world(scene)


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


def tool_frames(handle: int):
    """The kinematics model's tool frames (kin.tool_frames)."""
    return list(_REGISTRY[handle]["kin"].tool_frames)


def robot_spheres(handle: int, q):
    """Robot collision spheres in the base frame as [S,4] (PORT:
    robot_model_manager.get_collision_spheres)."""
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    kin = entry["kin"]
    js = JointState.from_position(
        torch.tensor([list(np.asarray(q, dtype=np.float32))], dtype=torch.float32,
                     device=torch.device("cuda", torch.cuda.current_device())),
        joint_names=kin.joint_names)
    res = kin.compute_kinematics(js)
    return res.robot_spheres.reshape(-1, 4).detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Robot description (PORT: curobov2_ros/robot/robot_description.load_robot_description)
# ---------------------------------------------------------------------------

def _resolve_descriptor(robot_arg: str) -> str:
    """Locate robots/<name>.yaml: an absolute path, package://, or a package
    share lookup."""
    import os

    from ament_index_python.packages import get_package_share_directory

    if not robot_arg:
        return ""
    if robot_arg.startswith("package://"):
        return _resolve_uri(robot_arg, "")
    if os.path.isabs(robot_arg) or os.path.isfile(robot_arg):
        return robot_arg
    # Bare name -> the package's robots/ dir, then the legacy config/robots dirs.
    candidates = []
    try:
        share = get_package_share_directory("curobov2_ros")
        candidates.append(os.path.join(share, "robots", f"{robot_arg}.yaml"))
        candidates.append(os.path.join(share, "config", "robots", f"{robot_arg}.yaml"))
    except Exception:
        pass
    candidates.append(os.path.join(os.getcwd(), "robots", f"{robot_arg}.yaml"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def robot_description(robot_arg: str, robot_config_file: str):
    """Load a robot descriptor (PORT: robot_description.load_robot_description).

    Returns the fields the C++ node needs: base_link, the resolved cuRobo config
    (`curobo_config`, a package:// URI resolved to a path), the control strategy
    key + its params, and the `home` named joint config.
    """
    import os

    import yaml

    path = _resolve_descriptor(robot_arg)
    if not path or not os.path.isfile(path):
        return {"found": False, "path": path}
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    curobo_config = data.get("curobo_config") or ""
    if curobo_config:
        curobo_config = _resolve_uri(curobo_config, os.path.dirname(path))
    params = data.get("strategy_params") or {}
    home = (data.get("named_joint_configs") or {}).get("home") or {}
    return {
        "found": True,
        "path": os.path.abspath(path),
        "name": data.get("name", ""),
        "base_link": data.get("base_link", ""),
        "curobo_config": curobo_config or robot_config_file,
        "strategy": data.get("strategy") or data.get("control_strategy") or "",
        "strategy_params": {str(k): str(v) for k, v in params.items()},
        "home_names": [str(k) for k in home.keys()],
        "home_positions": [float(v) for v in home.values()],
    }


def collision_spheres_with_attached(handle: int, q):
    """World collision spheres + per-sphere attached flag.

    PORT: robot_model_manager.get_collision_spheres_with_attached - sourced from
    the MotionPlanner's kinematics (which carries the fitted attached-object
    spheres); our own kin model never receives grasps.
    """
    import numpy as np
    import torch
    from curobo.types import JointState

    entry = _REGISTRY[handle]
    mp = entry.get("motion_planner")
    kin = getattr(mp, "kinematics", None) or entry["kin"]
    js = JointState.from_position(
        torch.tensor([list(np.asarray(q, dtype=np.float32))], dtype=torch.float32,
                     device=torch.device("cuda", torch.cuda.current_device())),
        joint_names=kin.joint_names)
    spheres = kin.compute_kinematics(js).robot_spheres.reshape(-1, 4)
    kc = kin.kinematics_config
    attached_idx = kc.link_name_to_idx_map.get("attached_object")
    mask = [False] * int(spheres.shape[0])
    link_map = getattr(kc, "link_sphere_idx_map", None)
    if attached_idx is not None and link_map is not None:
        link_ids = link_map.reshape(-1).cpu().tolist()
        if len(link_ids) == len(mask):
            mask = [li == attached_idx for li in link_ids]
    # Link-relative copy of the attached spheres. The consumer (trajectory
    # preview) re-applies FK of this URDF link at every waypoint, so the
    # payload rides the ghost instead of staying at the live pose. We derive
    # the link frame from the solver's own FK rather than the YAML transform
    # so it is exact for whatever `fixed_transform` the config uses.
    world = spheres.detach().cpu().numpy().astype(np.float32)
    attached_frame = ""
    attached_local = np.zeros_like(world)
    parent = entry.get("attach_parent_link")
    if parent and any(mask):
        try:
            q_t = torch.tensor(
                [list(np.asarray(q, dtype=np.float32))], dtype=torch.float32,
                device=torch.device("cuda", torch.cuda.current_device()))
            link_pose = kin.get_link_poses(q_t, [parent])
            p = link_pose.position[0, 0].detach().cpu().numpy()
            # Pose.get_rotation() only accepts a flat [b, 4] quaternion, and FK
            # returns [b, n_links, 4]; convert the quaternion here instead of
            # relying on that shape contract.
            w, x, y, z = link_pose.quaternion[0, 0].detach().cpu().numpy()
            rot = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x),
                 1 - 2 * (x * x + y * y)],
            ], dtype=np.float64)
            keep = np.asarray(mask, dtype=bool)
            # world -> link-local: R^T (p_world - p_link); v @ R == R^T v.
            attached_local[keep, :3] = (world[keep, :3] - p) @ rot
            attached_local[keep, 3] = world[keep, 3]
            attached_frame = parent
        except Exception:
            attached_frame = ""
            attached_local = np.zeros_like(world)
    return {
        "spheres": world,
        "attached": mask,
        "attached_local": attached_local,
        "attach_frame": attached_frame,
    }


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
