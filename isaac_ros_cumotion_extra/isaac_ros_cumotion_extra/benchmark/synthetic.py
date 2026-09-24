# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seeded synthetic benchmark inputs shared by the IK and kinematics/collision legs.

The upstream curobo benchmark page lists, besides motion planning
(``benchmark/motion_plan_benchmark.py``, already wired up), two more families:
inverse kinematics (``benchmark/ik_benchmark.py``) and kinematics & collision
checking (``benchmark/cost_gradient_benchmark.py``). Both generate their own
random inputs instead of consuming a dataset, so the parity legs here need an
input source that is *leg-shared and deterministic*: ``load_ik_goals`` /
``load_cost_configs`` regenerate identical inputs on both legs from a fixed
seed, guaranteeing both legs solve / FK the exact same goals and joint states.

The robot used is the SAME robot the ROS server is launched with
(``--robot-config``, defaulting to the benchmark envelope's
``config/franka.curobo.reference.yml`` — ``panda_hand`` tool frame, joint
limits expanded by ±0.2 rad like the upstream reference recipe). Building the
kinematics/scene machinery here mirrors the server's own construction
(``RobotModelManager`` / ``ConfigWrapperMotion``), so per-config outcomes are
comparable leg-to-leg.

This module stays importable in pure-Python environments: every cuRobo/ROS
dependency is imported lazily inside the functions below (the legs are
box-only — they need a GPU and a running planner server).
"""

# Standard Library
from typing import Any, Dict, List, Optional, Tuple

# cuRobo world the IK benchmark solves against (upstream ik_benchmark's
# collision_table.yml: a 4x4x0.2 m table, top at z=0). Pose list order is
# curobo's [x, y, z, qw, qx, qy, qz] (wxyz quaternion).
WORLD_IK: Dict[str, Any] = {
    "cuboid": {
        "table": {
            "dims": [4.0, 4.0, 0.2],
            "pose": [0.0, 0.0, -0.2, 1.0, 0.0, 0.0, 0.0],  # x, y, z, qw, qx, qy, qz
        },
    },
}

# Kinematics & collision world (upstream cost_gradient_benchmark's scene:
# table + a tall obstacle on top of it, runs conservatively sampled configs
# through the ball of the workspace).
WORLD_COST: Dict[str, Any] = {
    "cuboid": {
        "table": dict(WORLD_IK["cuboid"]["table"]),
        "cube6": {
            "dims": [0.1, 0.1, 1.5],
            "pose": [0.45, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
        },
    },
}

# Where the benchmark envelope (docker/compose_benchmark.yaml) launches the
# server's robot from — the same file must be used to build the native legs'
# kinematics so both sides share joint limits / tool frames / collision
# spheres. Pass another path via --robot-config outside the envelope layout.
DEFAULT_ROBOT_CONFIG = (
    "/root/ros2_ws/src/isaac_ros_cumotion/config/franka.curobo.reference.yml"
)

# Number of cuboid obstacles per world (for collision-cache sizing and native
# kernel-grid parity). WORLD_COST has table + cube6.
IK_WORLD_CUBOIDS = 1
COST_WORLD_CUBOIDS = 2

# Mirror of the server's FkBatch collision validator settings (fk_services.py).
COLLISION_ACTIVATION_DISTANCE = 0.001


def _require_curobo() -> None:
    """Fail with a clear message when cuRobo (or torch) is unavailable.

    The synthetic legs are box-only (GPU + curobo + a running planner
    server); this keeps accidental runs in pure-Python environments explicit
    instead of surfacing a cryptic ``import curobo`` traceback.
    """
    try:
        import curobo  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - box-only
        raise ImportError(
            "The IK / kinematics-and-collision benchmark legs need cuRobo "
            "and torch (the benchmark container has them; run `core` / `ros` "
            "planning legs for pure-machinery tests elsewhere)"
        ) from exc


def _normalize_device_string(device: Any) -> Any:
    """Map an index-less ``"cuda"`` to ``"cuda:0"`` (pure string helper).

    Warp's ``wp.device_from_torch`` indexes the runtime's CUDA devices with
    ``torch.device(...).index`` — an index-less ``torch.device("cuda")`` (the
    value ``torch.device("cuda")`` produces) raises
    ``TypeError: list indices must be integers or slices, not NoneType`` when
    cuRobo builds any ``MeshData`` (robot collision checkers, scene costs).
    The upstream reference benchmarks always pass an indexed device
    (``ik_benchmark.py`` relies on ``DeviceCfg``'s ``torch.device("cuda", 0)``
    default; ``cost_gradient_benchmark.py`` builds ``torch.device("cuda:0")``
    explicitly) — the parity legs canonicalize the same way.
    """
    if not isinstance(device, str):
        return device
    return "cuda:0" if device.strip().lower() == "cuda" else device


def canonical_device(device: Any) -> Any:
    """Return ``device`` with an explicit CUDA index (warp-safe).

    Accepts a ``str`` or ``torch.device``; index-less CUDA devices are mapped
    to ``cuda:0`` (the slot the ``DeviceCfg`` default resolves to anyway).
    Everything downstream accepts both forms (``torch`` tensor ops and
    ``DeviceCfg``), so callers can thread the result through unchanged.
    """
    import torch

    if isinstance(device, torch.device):
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", 0)
        return device
    return _normalize_device_string(device)


def resolved_robot_path(robot_config: str) -> str:
    """Mirror the server's robot-config resolution (absolute urdf/asset paths).

    ``ConfigManager.__init__`` resolves the launch ``robot_config_file``
    through ``resolve_curobo_config`` before anything parses it, so a
    yml-relative ``urdf_path``/``asset_root_path`` becomes absolute. The native
    legs must do the same, or cuRobo's own relative resolution falls back to
    its bundled assets dir and loads the wrong URDF.
    """
    if not robot_config:
        raise ValueError("robot_config must be a path to the robot YAML")
    try:
        # Exact mirror of the server when the package is importable (it is on
        # the box, where the benchmark container sources the workspace).
        from isaac_ros_cumotion.robot.robot_description import (  # type: ignore
            resolve_curobo_config,
        )
    except ImportError:  # pragma: no cover - box-only fallback
        return _minimal_resolve(robot_config)
    return resolve_curobo_config(robot_config)


def _minimal_resolve(robot_config: str) -> str:
    """Local stand-in for ``resolve_curobo_config`` (used if unavailable)."""
    import os
    import tempfile

    try:
        import yaml  # PyYAML ships with curobo on the box
    except ImportError:  # pragma: no cover - box-only
        return robot_config
    with open(robot_config) as f:
        data = yaml.safe_load(f) or {}
    robot_cfg = data.get("robot_cfg", data)
    kin = robot_cfg.get("kinematics", robot_cfg)
    if not kin:
        return robot_config
    yml_dir = os.path.dirname(os.path.abspath(robot_config))
    changed = False
    for key in ("urdf_path", "asset_root_path"):
        original = kin.get(key)
        if original and not os.path.isabs(original):
            candidate = os.path.normpath(os.path.join(yml_dir, original))
            if os.path.exists(candidate):
                kin[key] = candidate
                changed = True
    if not changed:
        return robot_config
    resolved = os.path.join(
        tempfile.gettempdir(), f"curobo_bench_{os.path.basename(robot_config)}"
    )
    with open(resolved, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return resolved


def load_kinematics_cfg(
    robot_config: str = DEFAULT_ROBOT_CONFIG,
    device: str = "cuda",
    dtype: Any = None,
):
    """Build the robot's ``KinematicsCfg`` the same way the server does.

    ``RobotModelManager`` uses ``KinematicsCfg.from_robot_yaml_file`` on the
    resolved config path; mirroring that call (same file, same constructor)
    guarantees identical tool frames / joint limits / FK math on both legs.
    """
    _require_curobo()
    import torch

    from curobo.kinematics import KinematicsCfg
    from curobo.types import DeviceCfg

    dtype = torch.float32 if dtype is None else dtype
    device = canonical_device(device)
    device_cfg = DeviceCfg(device=device, dtype=dtype)
    resolved = resolved_robot_path(robot_config)
    return KinematicsCfg.from_robot_yaml_file(resolved, device_cfg=device_cfg)


def load_robot_cfg_dict(robot_config: str = DEFAULT_ROBOT_CONFIG) -> Dict[str, Any]:
    """Return the server's ``get_robot_config_dict()`` equivalent.

    ``ConfigManager`` loads the resolved YAML, unwraps ``robot_cfg`` and pops
    a stray top-level ``cspace`` key; the checker (``RobotCollisionCheckerCfg.
    load_from_config``) consumes this dict, so the native legs build the
    validator from exactly the same input the server uses.
    """
    _require_curobo()
    import yaml

    with open(resolved_robot_path(robot_config)) as f:
        data = yaml.safe_load(f) or {}
    robot_cfg = data.get("robot_cfg", data)
    robot_cfg = dict(robot_cfg)
    robot_cfg.pop("cspace", None)
    return robot_cfg


def build_checker(
    robot_cfg_dict: Dict[str, Any],
    world: Dict[str, Any],
    device: str = "cuda",
    dtype: Any = None,
    activation_distance: float = COLLISION_ACTIVATION_DISTANCE,
):
    """Native ``RobotCollisionChecker`` mirroring the server's FkBatch validator.

    Same class, same constructor, same world (the primitives-only scene) and
    same activation distance as ``FKServices._init``, so ``validate()`` flags
    agree leg-to-leg on identical joint states.
    """
    _require_curobo()
    import torch

    from curobo.collision_checking import (
        RobotCollisionChecker,
        RobotCollisionCheckerCfg,
    )
    from curobo._src.geom.types import SceneCfg
    from curobo.types import DeviceCfg

    dtype = torch.float32 if dtype is None else dtype
    device = canonical_device(device)
    device_cfg = DeviceCfg(device=device, dtype=dtype)
    scene = SceneCfg.create(world)
    cfg = RobotCollisionCheckerCfg.load_from_config(
        robot_config=robot_cfg_dict,
        scene_model=scene,
        device_cfg=device_cfg,
        collision_activation_distance=activation_distance,
    )
    return RobotCollisionChecker(cfg)


def fk_tool_pose(kin, q: Any) -> Tuple[Any, Any]:
    """FK a ``[B, dof]`` joint tensor to tool poses, like the server's FK service.

    Returns ``(positions [B, 3], quaternions [B, 4] wxyz)`` — the first tool
    frame (``panda_hand``), extracted with the same ``[:, 0, 0, :]`` indexing
    as ``FKServices._compute_poses``. The input is moved onto the kinematics
    model's device/dtype defensively (the ROS leg hands over a host tensor).
    """
    from curobo.types import JointState as CuRoboJS

    device_cfg = getattr(kin, "device_cfg", None)
    if device_cfg is not None:
        q = q.to(device_cfg.device, dtype=kin.config.device_cfg.dtype)
    js = CuRoboJS.from_position(q, joint_names=kin.joint_names)
    kin_state = kin.compute_kinematics(js)
    return (
        kin_state.tool_poses.position[:, 0, 0, :],
        kin_state.tool_poses.quaternion[:, 0, 0, :],
    )


def _sample_configs(
    kin,
    batch: int,
    device: str,
    dtype: Any,
    filter_valid: Optional[Any] = None,
    max_attempts: int = 20000,
) -> Any:
    """Sample ``batch`` joint configs uniform in joint limits, optionally valid.

    ``filter_valid`` is a callable taking a ``[K, 1, dof]`` tensor and
    returning a ``[K, 1]`` boolean mask (a ``RobotCollisionChecker.validate``
    bound method); when given, configs failing it are rejected (rejection
    sampling) so the goals below are reachable by a collision-free config —
    the same idea as the upstream benchmark's ``sample_configs(rejection_ratio)``.
    Deterministic for a fixed outer seed (reseeded by the callers).
    """
    import torch

    limits = kin.get_joint_limits().position  # [2, dof] min/max rows
    lo, hi = limits[0].to(device), limits[1].to(device)
    accepted = []
    attempts = 0
    while len(accepted) < batch:
        attempts += 1
        if attempts > max_attempts:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Rejection sampling failed to find {batch} valid configs in "
                f"{max_attempts} draws"
            )
        need = batch - len(accepted)
        q = lo + (hi - lo) * torch.rand((need, lo.shape[0]), device=device, dtype=dtype)
        if filter_valid is None:
            accepted.append(q)
        else:
            mask = filter_valid(q.unsqueeze(1)).squeeze(1)  # [need]
            if bool(mask.any()):
                accepted.append(q[mask])
    q = torch.cat(accepted, dim=0)
    return q[:batch]


def load_ik_goals(
    variant: str = "cfree",
    batch: int = 100,
    n_batches: int = 5,
    robot_config: str = DEFAULT_ROBOT_CONFIG,
    seed: int = 2,
    device: str = "cuda",
    dtype: Any = None,
) -> List[List[Dict[str, Any]]]:
    """Deterministic IK goal poses shared by the native and ROS legs.

    Goals are the tool pose (FK) of random near-homed joint configs — the same
    family the upstream ``ik_benchmark.py`` solves, but generated once here so
    BOTH legs solve identical goals (``torch.manual_seed(seed)`` +
    ``np.random.seed(seed)`` are applied at entry, so calling this from either
    leg reproduces the same batches).

    ``variant``:
    - ``"cfree"`` — collision-aware goals: configs rejected unless they are
      self- and scene-collision-free against ``WORLD_IK`` (parity variant —
      this is the ROS server's only IK mode).
    - ``"plain"`` — kinematics-only goals, no filtering (native-reference row
      mirroring the upstream ``collision_free=False`` run; not compared).

    Returns ``n_batches`` lists of ``batch`` goal dicts with
    ``position_xyz`` / ``quaternion_wxyz`` (curobo wxyz order).
    """
    _require_curobo()
    import numpy as np
    import torch

    from curobo.kinematics import Kinematics

    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.float32 if dtype is None else dtype
    device = canonical_device(device)

    kin_cfg = load_kinematics_cfg(robot_config, device=device, dtype=dtype)
    kin = Kinematics(kin_cfg)

    filter_valid = None
    if variant == "cfree":
        checker = build_checker(
            load_robot_cfg_dict(robot_config), WORLD_IK, device=device, dtype=dtype
        )
        filter_valid = checker.validate

    batches: List[List[Dict[str, Any]]] = []
    for _ in range(n_batches):
        q = _sample_configs(kin, batch, device, dtype, filter_valid=filter_valid)
        pos, quat = fk_tool_pose(kin, q)
        goals = [
            {
                "position_xyz": [float(v) for v in pos[i].cpu().tolist()],
                "quaternion_wxyz": [float(v) for v in quat[i].cpu().tolist()],
            }
            for i in range(batch)
        ]
        batches.append(goals)
    return batches


def load_cost_configs(
    batch: int = 100,
    n_batches: int = 5,
    robot_config: str = DEFAULT_ROBOT_CONFIG,
    seed: int = 2,
    device: str = "cuda",
    dtype: Any = None,
) -> List[List[List[float]]]:
    """Deterministic joint configurations the FK/collision legs evaluate.

    Uniformly sampled within the robot's joint limits (reseeded at entry);
    identical on both legs. Returns ``n_batches`` lists of ``batch``
    joint-vector lists (solver joint order, 7 DOF for franka).
    """
    _require_curobo()
    import numpy as np
    import torch

    from curobo.kinematics import Kinematics

    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.float32 if dtype is None else dtype
    device = canonical_device(device)

    kin_cfg = load_kinematics_cfg(robot_config, device=device, dtype=dtype)
    kin = Kinematics(kin_cfg)

    batches: List[List[List[float]]] = []
    for _ in range(n_batches):
        q = _sample_configs(kin, batch, device, dtype)
        batches.append(
            [[float(v) for v in row.cpu().tolist()] for row in q]
        )
    return batches


def order_ros_parity_setup(
    node,
    *,
    warmup,
    size_collision_cache: bool = True,
    timeout: float = 120.0,
) -> None:
    """Register the parity world on the server in cache-safe order.

    Works against any object exposing ``clear_world`` /
    ``size_collision_cache`` / ``add_world`` (each with a ``timeout`` kwarg),
    ``get_logger().warn``, and a ``warmup(timeout=...)`` callable — the
    service-clients API of ``ros_ik.RosIkRunner`` / ``ros_cost.RosCostRunner``.

    The ordering is load-bearing: the registered scene must never hold more
    cuboids than the server's active collision-cache capacity. cuRobo enforces
    it at every world update — (a) a cache change rebuilds every solver from
    the then-current scene and raises if that scene exceeds the NEW cap, and
    (b) an add raises if the scene would exceed the CURRENT cap. Clearing
    first empties the scene, so sizing *down* over planning-suite leftovers
    (up to 8 cuboids) is safe; sizing *before* the adds then guarantees the
    parity world can never overflow the active cache, which a previous leg may
    have shrunk (the IK leg sizes to cuboid=1 — with the old clear->add->size
    order the cost leg's 2-cuboid world crashed the box mid-add with "Cannot
    add cuboid, cache is full"). The freshly sized solver finally captures the
    registered world at construction (the ``warmup`` call), i.e. the same
    construction order as the native leg. Passing ``size_collision_cache=False``
    skips the sizing and leaves the (larger) server defaults in place for A/B.
    """
    node.clear_world(timeout=timeout)

    if size_collision_cache:
        node.size_collision_cache(timeout=timeout)
    else:
        node.get_logger().warn(
            "size_collision_cache=False: leaving the server's collision "
            "cache in place (padded kernel grids — diagnostic)"
        )

    node.add_world(timeout=timeout)
    warmup(timeout=timeout)