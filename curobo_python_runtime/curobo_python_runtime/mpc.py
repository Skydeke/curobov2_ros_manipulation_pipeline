# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_python_runtime - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

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
def _mpc_held_tool_poses(entry, solver, hold_q=None):
    """FK for every solver tool frame at ``hold_q`` (or the setup start).

    A Cartesian MPC goal must name every configured tool frame. For a single
    arm on a bimanual robot the unnamed tool is anchored at its CURRENT pose,
    not at a copy of another tool's target.
    """
    import numpy as np
    import torch
    from curobo.types import JointState

    if hold_q is None or len(hold_q) == 0:
        hold_q = entry.get("mpc_start")
    if hold_q is None or len(hold_q) == 0:
        return None
    device = torch.device("cuda", torch.cuda.current_device())
    q = torch.tensor(
        np.asarray(hold_q, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32, device=device)
    state = JointState.from_position(q[:, :solver.action_dim], joint_names=solver.joint_names)
    kin = solver.compute_kinematics(state)
    return kin.tool_poses.to_dict()


def _mpc_goal_tool_pose(solver, raw, held_poses=None):
    """PORT: reactive_controller._set_target.

    ``held_poses`` (a {frame: Pose} map from ``_mpc_held_tool_poses``) fills
    every configured tool frame; without it only the first frame is addressed,
    which is all a single-tool robot needs.
    """
    from curobo.types import GoalToolPose, Pose

    pose = Pose.from_list(list(raw))
    pose_dict = dict(held_poses) if held_poses else {}
    pose_dict[solver.tool_frames[0]] = pose
    ordered = list(solver.tool_frames) if held_poses else None
    return GoalToolPose.from_poses(
        pose_dict,
        ordered_tool_frames=ordered,
        num_goalset=1,
    )


def _mpc_goal_tool_poses(solver, tool_frames, poses, held_poses=None):
    """Build one GoalToolPose from parallel per-tool (frame, 7-vector) goals."""
    import numpy as np
    import torch
    from curobo.types import GoalToolPose, Pose

    rows = np.asarray(poses, dtype=np.float32).reshape(-1, 7)
    frames = [str(frame) for frame in (tool_frames or [])]
    if not frames:
        if rows.shape[0] != 1:
            raise ValueError("a multi-pose MPC goal needs one tool frame per pose")
        frames = [solver.tool_frames[0]]
    if len(frames) != rows.shape[0]:
        raise ValueError(
            f"tool_frames/poses length mismatch: {len(frames)} frame(s), "
            f"{rows.shape[0]} pose(s)")
    pose_dict = dict(held_poses) if held_poses else {}
    device = torch.device("cuda", torch.cuda.current_device())
    for frame, row in zip(frames, rows):
        pose_dict[frame] = Pose(
            position=torch.tensor(row[0:3], dtype=torch.float32,
                                  device=device).reshape(1, 3),
            quaternion=torch.tensor(row[3:7], dtype=torch.float32,
                                    device=device).reshape(1, 4),
        )
    return GoalToolPose.from_poses(
        pose_dict,
        ordered_tool_frames=list(solver.tool_frames),
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
    # Kept so single-tool goals can anchor the other tools at the phase start.
    entry["mpc_start"] = list(start_q)

    ee_position = None
    try:
        kin = solver.compute_kinematics(start_state)
        ee_position = [
            float(v) for v in kin.tool_poses.position.reshape(-1, 3)[0].cpu().tolist()]
    except Exception:
        ee_position = None
    return {"ee_position": ee_position}
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
    entry["mpc_start"] = list(q)


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

from .runtime import _REGISTRY
from .ik import ik
from .scene import _to_scene
