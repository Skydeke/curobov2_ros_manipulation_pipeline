# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native leg: kinematics & collision forces benchmark (the ``cost`` capability).

The upstream ``curobo/benchmark/cost_gradient_benchmark.py`` moves a robot
through random joint configurations and reports the forward-kinematics tool
pose plus the cost gradient / collision status per sample. Its parity-relevant
outs are *per-config correlation of the ROS server's FK and collision
validity* with the native machinery:

- the FK tool pose of each config (curobo ``Kinematics`` — the same model the
  server's ``FKServices`` builds from the shared ``--robot-config``), and
- ``valid`` — ``RobotCollisionChecker.validate()`` on the same world
  (``WORLD_COST``: table + tall cuboid) with the server's
  ``collision_activation_distance=0.001``; the service calls this its
  ``poses_valid`` output and folds joint limits + self-collision + scene
  collision into it.

The native leg evaluates the SAME configs the ROS leg sends (shared,
deterministically reseeded ``synthetic.load_cost_configs``), so ``valid``
agreement is exact-by-construction and FK poses should agree to float
reproducibility. Timing is informational (client wall around each
FK+validate batch, ms).
"""

# Standard Library
import time
from typing import Any, Dict, List

from .synthetic import (
    COST_WORLD_CUBOIDS,
    DEFAULT_ROBOT_CONFIG,
    WORLD_COST,
    build_checker,
    canonical_device,
    load_cost_configs,
    load_kinematics_cfg,
    load_robot_cfg_dict,
)

# Server-matching validator activation distance (FKServices._init).
ACTIVATION_DISTANCE = 0.001


def run_cost_core(
    batch: int = 100,
    n_batches: int = 5,
    robot_config: str = DEFAULT_ROBOT_CONFIG,
    seed: int = 2,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Run the native kinematics & collision leg over the shared configs.

    Returns one entry per evaluated joint state (schema identical to the ROS
    leg, so ``compare_cost`` can match rows 1:1 by ``problem_name``).
    """
    from curobo.kinematics import Kinematics
    from curobo.types import JointState as CuRoboJS

    import torch

    dtype = torch.float32
    device = canonical_device(device)
    kin = Kinematics(load_kinematics_cfg(robot_config, device=device, dtype=dtype))
    checker = build_checker(
        load_robot_cfg_dict(robot_config),
        WORLD_COST,
        device=device,
        dtype=dtype,
        activation_distance=ACTIVATION_DISTANCE,
    )

    config_batches = load_cost_configs(
        batch=batch,
        n_batches=n_batches,
        robot_config=robot_config,
        seed=seed,
        device=device,
        dtype=dtype,
    )

    # Warmup mirroring the server's WarmupFK handler: one sample-configs FK
    # pass to prime the kinematics kernels before the timed batches.
    q_sample = torch.rand((batch, kin.get_dof()), dtype=dtype, device=device)
    js_sample = CuRoboJS.from_position(q_sample, joint_names=kin.joint_names)
    kin.compute_kinematics(js_sample)

    results: List[Dict[str, Any]] = []
    for b_idx, configs in enumerate(config_batches, start=1):
        q = torch.tensor(configs, dtype=dtype, device=device)
        js = CuRoboJS.from_position(q, joint_names=kin.joint_names)

        t0 = time.perf_counter()
        kin_state = kin.compute_kinematics(js)
        mask = checker.validate(q.unsqueeze(1)).squeeze(1)  # [B]
        time_ms = (time.perf_counter() - t0) * 1000.0

        positions = kin_state.tool_poses.position[:, 0, 0, :].cpu().tolist()
        quaternions = kin_state.tool_poses.quaternion[:, 0, 0, :].cpu().tolist()
        valid = mask.cpu().tolist()

        for i, _ in enumerate(configs, start=1):
            results.append(
                {
                    "problem_name": f"cost_b{b_idx:02d}_g{i:03d}",
                    "scene_key": "cost",
                    "capability": "cost",
                    "batch": b_idx,
                    "index": i,
                    "n_configs": batch,
                    "valid": bool(valid[i - 1]),
                    "position_xyz": [float(v) for v in positions[i - 1]],
                    "quaternion_wxyz": [float(v) for v in quaternions[i - 1]],
                    "time_ms": time_ms,
                }
            )
    return results