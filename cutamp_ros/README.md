# cutamp_ros

C++ `rclcpp_components` component that runs the Python cuTAMP planner inside
the cuRobo process's existing embedded CPython interpreter (the one owned by
cuRobo's `PyBridge`). It never creates an interpreter, a CUDA context or a
second cuRobo model.

```
cutamp_ros::CuTampComponent      node "cutamp", service "~/plan"
        │
        └── cutamp_ros/bridge.py  (installed into site-packages)
                 │
                 └── cutamp.algorithm.solve_cutamp  →  CuTampSolution
```

## Service

`~/plan` is `cutamp_ros_interfaces/srv/Plan`:

| request | meaning |
|---|---|
| `curobov2_ros_interfaces/SceneObject[] objects` | complete TAMP scene (cuboid/sphere/cylinder/capsule; MESH rejected) |
| `string[] movable_names` | names in `objects` that cuTAMP may move; all others are static surfaces |
| `string[] goal_atoms` | PDDL atoms such as `On(block0,table)`, `HandEmpty()` |
| `float64[] q_init` | initial joints in cuRobo active-DOF order (required for custom robots) |
| `int32 num_particles` | 0 = cuTAMP default |
| `int32 num_opt_steps` | 0 = cuTAMP default |
| `int32 num_initial_plans` | 0 = cuTAMP default |
| `float64 max_loop_dur` | <= 0 = no limit |

> `On(obj,surface)` requires `surface` to be a static object, so do not list a
> support object in `movable_names`. Every non-movable object is typed as a
> cuTAMP `Surface` (the pick/place domain has no inferred buttons/sticks).

| response | meaning |
|---|---|
| `bool success`, `string message` | result / exception text |
| `curobov2_ros_interfaces/Goalset[] goalsets` | ordered segments; exactly one `Pose` per segment |
| `int32[] gripper_close` | parallel: 1 close, 0 open |
| `int32[] action_index` | parallel: zero-based pick/place action |
| `int32 num_satisfying` | satisfying particles in the winning plan |

The caller hands `goalsets` to `/curobo_server/generate_trajectory` in order.

## Goalset conversion

cuTAMP's winning plan is the tuple
`(plan_skeleton, best_particle, plan_info["rollout_fn"])`. The bridge replays
the best particle once:

```python
rollout = plan_info["rollout_fn"]({k: v[None] for k, v in best_particle.items()})
tool_poses = rollout["world_from_tool_desired"][0]   # [T, 4, 4], base frame
```

Walking `plan_skeleton` (skipping `MoveFree`/`MoveHolding`), each `Pick` and
`Place` consumes one timestep and produces three goalsets:

| operator | segment 1 | segment 2 | segment 3 |
|---|---|---|---|
| `Pick` | approach (open) | grasp pose (close) | lift (close) |
| `Place` | place pose (close) | same pose (open) | retreat (open) |

Approach/lift/retreat are 5 cm along the legacy EE-frame approach axis,
converted to the tool frame:
`tool_from_ee @ Trans(0, 0, -0.05) @ inv(tool_from_ee)`, applied on the right
of the tool pose. This is the cuRobo-v2 equivalent of the old
`approach_offset[2, 3] = -0.05` in `motion_solver.py`.

`Push`/`PushStick` plans are rejected with a clear error.

Poses are in the robot base frame; `Goalset.frame_id` is the node's
`frame_id` parameter (default empty).

## Parameters (flat keys)

The deployment launch file loads `cutamp_config` with
`yaml.safe_load` and passes the mapping straight to `parameters=[...]`, so the
file must contain **flat** keys (no `cutamp: ros__parameters:` nesting). See
`config/cutamp_params.example.yaml` for the complete list:

`robot_config_path` (or `robot_config_file`), `urdf_path`, `tool_frame`,
`tool_from_ee` (16 row-major), `gripper_spheres`, `ik_batch_size`,
`ik_num_seeds`, `self_collision_check`, `use_cuda_graph`, `coll_n_spheres`,
`coll_sphere_radius`, `device`, `warmup`, `frame_id`.

## Robot-side data still required

The generic package contains no robot data. For each robot the deployment must
provide:

1. a cuRobo v2 robot YAML (`robot_config_path`) and its URDF,
2. the `tool_frame` declared in that YAML,
3. `tool_from_ee`: the 4x4 transform from the YAML's EE frame to the tool frame
   (the same frame the gripper spheres are expressed in),
4. `gripper_spheres`: an `(n, 4)` `.npy` of `[x, y, z, radius]` collision
   spheres of the closed gripper in that tool frame,
5. `q_init` per request (cuTAMP only ships a built-in home pose for
   panda/ur5).

## Python call sequence

```python
# once, on the first plan (or at construction with warmup:=true)
cfg    = KinematicsCfg.from_robot_yaml_file(robot_config_path,
                                            tool_frames=[tool_frame],
                                            urdf_path=urdf_path,
                                            device_cfg=DeviceCfg(...))
kin    = Kinematics(cfg)
model  = RobotKinematics(kin, tool_frame)
ik     = build_ik(cfg, tool_frame, batch_size=..., num_seeds=...,
                  self_collision_check=..., use_cuda_graph=...)
robot  = RobotContainer(name=..., kin_model=model, joint_limits=model.joint_limits,
                        gripper_spheres=np.load(path), tool_from_ee=...,
                        ik_solver=ik)

# every request
env      = TAMPEnvironment(name, movables, statics,
                           {"Movable": movables, "Surface": statics}, goal_state)
config   = TAMPConfiguration(robot_container=robot, curobo_plan=False, ...)
solution = solve_cutamp(env, config,
                        CostReducer(default_constraint_to_mult.copy()),
                        ConstraintChecker(default_constraint_to_tol.copy()),
                        q_init=q_init)
```

All calls are serialized by a mutex in the component; the GIL is only ever
acquired with `py::gil_scoped_acquire`.
