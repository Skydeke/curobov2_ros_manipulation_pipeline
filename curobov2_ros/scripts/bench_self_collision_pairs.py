#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the cost of the self-collision pair-distance diagnostics.

The RViz collision-sphere publisher uses cuRobo's ``store_pair_distance=True``
self-collision kernel so the colours come from the exact pair list and sphere
padding the planner uses. This script measures that kernel against the normal
max-reduction kernel and against an equivalent CPU pass.

Run inside the cuRobo image, with the workspace sourced:

    source /root/ros2_ws/install/setup.bash
    python3 curobov2_ros/scripts/bench_self_collision_pairs.py \
        /path/to/<robot>.curobo.yaml
"""
import argparse
import json
import time

import numpy as np
import torch

import curobo_python_runtime as cpr
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo.types import JointState


def _timeit(fn, iterations, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_config", help="resolved cuRobo robot YAML")
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    handle = cpr.init({})
    cpr.robot_model_build(
        handle, json.dumps({"robot_config_file": args.robot_config}))
    kin = cpr._REGISTRY[handle]["kin"]
    sc = kin.config.self_collision_config
    print(f"robot:           {args.robot_config}")
    print(f"joints:          {len(kin.joint_names)}")
    print(f"spheres:         {sc.num_spheres}")
    print(f"collision pairs: {sc.collision_pairs.shape[0]}")
    print(f"link AABB path:  {sc.use_link_aabb}")
    print(f"device:          {torch.cuda.get_device_name(0)}")

    torch.manual_seed(0)
    lo, hi = kin.get_joint_limits().position
    q = lo + (hi - lo) * torch.rand((1, kin.get_dof()), device=lo.device)
    joint_state = JointState.from_position(q, joint_names=kin.joint_names)
    spheres = kin.compute_kinematics(joint_state).robot_spheres

    weight = torch.tensor([1.0], device=spheres.device)
    max_cost = SelfCollisionCost(
        SelfCollisionCostCfg(
            weight=weight,
            self_collision_kin_config=sc,
            store_pair_distance=False,
        )
    )
    pair_cost = SelfCollisionCost(
        SelfCollisionCostCfg(
            weight=weight,
            self_collision_kin_config=sc,
            store_pair_distance=True,
        )
    )
    for cost in (max_cost, pair_cost):
        cost.setup_batch_tensors(1, 1)

    def fk():
        kin.compute_kinematics(joint_state)

    def max_forward():
        max_cost.forward(spheres)

    def pair_forward():
        pair_cost.forward(spheres)

    pairs = sc.collision_pairs.detach().cpu().numpy().astype(np.int64)
    padding = sc.sphere_padding.detach().cpu().numpy().reshape(-1)
    sphere_data = spheres.reshape(-1, 4).detach().cpu().numpy()

    def cpu_pairs():
        radius_a = sphere_data[pairs[:, 0], 3] + padding[pairs[:, 0]]
        radius_b = sphere_data[pairs[:, 1], 3] + padding[pairs[:, 1]]
        centers = np.linalg.norm(
            sphere_data[pairs[:, 0], :3] - sphere_data[pairs[:, 1], :3], axis=1)
        return centers - (radius_a + radius_b)

    fk_time = _timeit(fk, args.iterations)
    max_time = _timeit(max_forward, args.iterations)
    pair_time = _timeit(pair_forward, args.iterations)
    cpu_time = _timeit(cpu_pairs, args.iterations * 5)

    print()
    print(f"FK (spheres)               : {fk_time * 1e3:8.3f} ms")
    print(f"self-collision max kernel  : {max_time * 1e3:8.3f} ms")
    print(f"store_pair_distance kernel : {pair_time * 1e3:8.3f} ms"
          f"  ({pair_time / max_time:.2f}x max kernel)")
    print(f"CPU pair math (numpy)      : {cpu_time * 1e3:8.3f} ms")
    print()
    print("The marginal cost of the exact RViz colours is the difference between")
    print("the two kernels; FK and the world-distance query are shared with the")
    print("normal publisher path.")


if __name__ == "__main__":
    main()
