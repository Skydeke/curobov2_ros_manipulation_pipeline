# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

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
    """Reset every CUDA graph that captured the shared scene collision buffers. ``world_push`` rewrites those buffers (``scene_collision_checker
    .load_collision_model``), and a CUDA graph baked against the old buffers
    faults with "CUDA error: an illegal memory access" when it is replayed. cuRobo's ``MotionPlanner`` keeps THREE graphs over the one shared
    ``scene_collision_checker`` - ``ik_solver`` (pose plans only),
    ``trajopt_solver`` and the PRM ``graph_planner`` - and its
    ``SolverCore.reset_cuda_graph`` walks only the trajopt solver, so resetting
    "the solver" silently leaves the other two stale."""
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
    """Make the trajopt solver's CUDA graphs match `mode` ("pose"/"cspace"). cuRobo captures ONE set of optimizer/rollout CUDA graphs on the first solve
    and replays them blindly afterwards. A pose<->cspace switch changes the goal
    buffer structure (link poses vs an explicit joint goal) under those graphs'
    feet; replaying the stale graph reads freed/rewritten memory and the CUDA
    context dies with "an illegal memory access" - which is exactly the
    ``plan_cspace`` crash this port hit."""
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
        attempts = attempt_config(entry)
        # Warm the trajopt graphs with a DIRECT solve (enable_graph_attempt ==
        # max_attempts means no attempt can reach the PRM seed path), and warm the
        # PRM graph separately below when the runtime can reach it. Warming with
        # the runtime's own enable_graph_attempt would make this warmup plan run a
        # full PRM search *on top of* that separate warmup.
        if mode == "cspace":
            _warmup_cspace(
                planner, iterations=1,
                max_attempts=max(1, attempts[0]), enable_graph_attempt=max(1, attempts[0]))
        else:
            _warmup_pose(planner, iterations=1)
        # The PRM graph planner seeds the LATER attempts of both solve modes, so
        # exercise it explicitly here: its graph (``feasibility_rollout``) must be
        # re-captured under this exclusive barrier, not lazily inside the next
        # plan where the capture would race the camera/ESDF lane. The per-mode
        # warmups above only reach it when their first attempt happens to fail.
        #
        # Skip it entirely when the configured attempt budget can never reach the
        # graph path (`enable_graph_attempt >= max_attempts`, e.g. the node's
        # max_attempts=1 / enable_graph_attempt=1): warming a graph that no plan
        # will ever replay costs this re-warmup's full GPU time on EVERY
        # perception world push, for nothing.
        graph_planner = getattr(planner, "graph_planner", None)
        if graph_attempt_used(attempts) and graph_planner is not None and callable(
                getattr(graph_planner, "warmup", None)):
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
def attempt_config(entry, cfg: Optional[dict] = None):
    """The ``(max_attempts, enable_graph_attempt)`` a plan request will use.

    PORT: config max_attempts / enable_graph_attempt. Recorded by
    :func:`motion_planner_build` from the node's parameters so the warmup in
    :func:`prepare_solve_mode` can mirror what the runtime actually replays.
    Defaults reproduce the reference/older behaviour (2 attempts, graph seeding
    from attempt 1)."""
    if cfg is None:
        cfg = entry.get("attempt_cfg") or {}
    return (int(cfg.get("max_attempts", 2)), int(cfg.get("enable_graph_attempt", 1)))
def graph_attempt_used(cfg) -> bool:
    """Whether ``plan_cspace``/``plan_pose`` can ever reach the PRM seed path.

    cuRobo consults the graph planner on every attempt where
    ``current_attempt >= enable_graph_attempt`` (see
    ``MotionPlanner._plan_cspace_attempts``), so a graph attempt is reachable iff
    some attempt index is >= the threshold - i.e. iff
    ``enable_graph_attempt < max_attempts``."""
    max_attempts, enable_graph_attempt = cfg
    return enable_graph_attempt < max_attempts
def _warmup_cspace(
    planner,
    iterations: int = 3,
    joint_delta: float = 0.2,
    max_attempts: int = 2,
    enable_graph_attempt: int = 1,
) -> None:
    """Record the joint-space (``plan_cspace``) CUDA graphs at build time. Mirrors :meth:`MotionPlanner.warmup`'s pose loop, but exercises ``plan_cspace``
    (and, via ``enable_graph_attempt``, the graph-planner seed path) so every
    graph the joint-space planner replays is captured while the caller holds the
    exclusive GPU barrier. Without this the first joint-space request captures
    them mid-plan next to the camera/ESDF lane, which can fault the CUDA context.

    ``max_attempts``/``enable_graph_attempt`` MUST mirror the values the plan
    requests use (see :func:`motion_planner_build`): a wider warmup walks code
    paths - an extra attempt, and the PRM seed search - that the runtime never
    replays, and each re-capture happens on the perception lane's time, where it
    is the single most expensive thing this bridge does."""
    current = planner.default_joint_state.clone().unsqueeze(0)
    for _ in range(iterations):
        goal = current.clone()
        goal.position[..., 0] = goal.position[..., 0] + joint_delta
        try:
            planner.plan_cspace(
                goal, current,
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt)
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
    # The plan-request attempt budget, recorded so prepare_solve_mode's re-capture
    # warms exactly the code paths a plan will replay (and skips the PRM warmup
    # when the graph path is unreachable). Passed by the node from its
    # max_attempts / enable_graph_attempt parameters.
    entry["attempt_cfg"] = {
        "max_attempts": int(cfg.get("max_attempts", 2)),
        "enable_graph_attempt": int(cfg.get("enable_graph_attempt", 1)),
    }
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
        _warmup_cspace(
            planner,
            max_attempts=entry["attempt_cfg"]["max_attempts"],
            enable_graph_attempt=entry["attempt_cfg"]["enable_graph_attempt"])
    except Exception as exc:  # a warmup must never block the build
        print(
            f"[curobo_bridge] cspace CUDA-graph warmup failed ({exc!r}); the first "
            f"joint-space plan will capture its graphs unchecked", flush=True)
    entry["motion_planner"] = planner
    # PORT: config_wrapper_motion.set_motion_gen_config tail - a freshly built
    # solver arrives with every obstacle enabled, so an already-attached payload
    # must have its static copy re-disabled on the new checker.
    _reapply_attached_disables(entry)

from .runtime import _REGISTRY
from .scene import _reapply_attached_disables
