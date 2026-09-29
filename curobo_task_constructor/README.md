# curobo_task_constructor

A MoveIt Task Constructor–shaped stage/container framework over the
`isaac_ros_cumotion` cuRobo server. A task is a tree of stages; the executor
plans it, ranks the solutions, and executes one. Nothing here calls IK on the
client's behalf — **the planner resolves the goals**, which is what lets one
task offer many candidates per leg and let cuRobo pick the winner.

## The wire format

A task travels as a **flat, pre-order** list of `curobo_task_constructor_interfaces/StageSpec`
messages: element 0 is the root (`parent_id == id`), every other element's
`parent_id` points at an earlier one. Each stage's parameters ride in the
opaque `params_yaml` field, which that stage class parses itself. The format
is unchanged by anything described below — new capabilities are new params
inside `params_yaml`, and new stages, never new messages.

The in-memory tree is plain dicts (`pick_tree.leaf` / `container`) so it can be
built and unit-tested without ROS.

## Stages

| `stage_type` | Class | Role |
|---|---|---|
| `current_state` | `CurrentState` | Write the robot's live joint state. `require_attached` / `require_not_attached` list object names the state must (not) be holding. |
| `fixed_state` | `FixedState` | Write a named joint configuration: `goal: <name>`, resolved from the robot descriptor's `named_joint_configs` (optionally via `robot_config_path`). |
| `generate_grasp_pose` | `GenerateGraspPose` | Emit N candidate tool poses around a sampled object. `object`, `pre_grasp` (named config defining the approach), `angle_delta` (default 15°), `angle_end` / `num_steps`, `approach_offset`, `mode: grasp \| place`. |
| `compute_ik` | `ComputeIK` | Resolve poses to joint solutions. `max_solutions`. |
| `move_to` | `MoveTo` | Plan to a goal: `goal: {name}` \| `{joints}` \| `{joints: {j: v}}` \| `{pose}` \| `{poses}`. |
| `move_relative` | `MoveRelative` | Plan a translation with the tool orientation frozen. `axis` (named config) or `distance`, or sample `min_distance`..`max_distance` in `num_samples` steps. `hold`, `pos_tol`, `rot_tol`, `link`, `direction`. |
| `cartesian_path` | `CartesianPath` | Plan a **straight line**. See below. |
| `connect` | `Connect` | Plan a motion between two already-known states. |
| `modify_scene` | `ModifyScene` | `add` (an object spec: `name`, `shape`, `pose` as a flat `{x,y,z,qx,qy,qz,qw}` dict, `dimensions`) / `remove` (by name) / `remove_all` (clear the server world) / `attach` / `detach` an object, `detach_all` (release whatever is attached — clears the attach state that `remove_all` does not), or `allow_collisions: {object, links, enabled}`. |

`goal: {joints: {j: v}}` is a **sparse** goal: the listed joints are merged onto
the start state and everything else holds. That is the "close the fingers, park
the arm" goal, and it needs no IK result for the arm — so a client can command
a gripper without ever learning the arm's solved joints.

All motion stages also accept `planner`, `cost`, `link`, `direction`, and the
`PlanningOptions` keys below.

Containers: `serial`, `alternatives` (plan all, rank, take the best),
`fallbacks` (plan in priority order, take the first that solves),
`independent_components`.

### Planning options

`planner` selects the cuRobo planner: `classic` (pose goals, `ToolPoseCriteria`
honoured) or `joint_space` (open-loop joint interpolation, ignores
`trajectory_constraints`).

`exact_joints`, `waypoint_tolerance`, `log_considered`, `num_seeds` become the
request's `PlanningOptions`. **The classic planner rejects any non-default
`PlanningOptions`** ("classic/reactive planners carry zero options"), so send
options only on `joint_space` stages.

### `cost` — how solutions are ranked

`cost:` on a motion stage picks the ranking term; the default is `auto`
(`PlanResult.cost` when an adapter populated it, else `path_length`).

| mode | term |
|---|---|
| `auto` | server cost if available, else `path_length` |
| `path_length` | MTC `cost::PathLength` over the returned waypoints |
| `waypoints` | waypoint count (the older proxy) |
| `solver_cost` | cheapest `stats.considered[*].cost`; needs `log_considered: true` |
| `inf` | never rank, never pick (first-solution-wins) |

`path_length` is the default because it is exactly computable from the wire and
the only term comparable across planners (cuRobo's `seed_cost` is a per-problem
trajopt objective). It also *rewards* straightness on a Cartesian move: a
straight line is a **longer** joint path than a shortcut around an obstacle, so
minimising length is part of what pushes the solver onto the line.

Cumulative clearance is **not** obtainable — `GetCollisionDistance` reports
sphere clearances at the current configuration only, not along a trajectory.

## `cartesian_path` — straight lines on a planner that has none

cuRobo has no Cartesian solve mode. A line is a **cost**: a goalset carries
`trajectory_constraints` (`int8[6]` = `[theta_x, theta_y, theta_z, x, y, z]`,
`1` = pin that axis for the whole path), which the server turns into
`ToolPoseCriteria` and scores on every non-terminal waypoint. That replaces
MoveIt's `computeCartesianPath(step=0.01, fraction>0.95, jump<5mm)`.

Three goal forms:

- `goal.pose` — absolute tool pose
- `goal.poses` — N candidates in **one** goalset; the server resolves the set
  inside a single `plan_pose()` and reports the winner via `selected_goal_index`
- `goal.relative` — `{x, y, z, frame: world|hand}` from `FK(start)`, keeping
  the start orientation. Anchored to the pose the arm *actually reached*, which
  is what a retreat/lift needs.

`hold` pins the vector explicitly. Omit it and the stage derives it: hold every
axis on which start and goal agree, free the rest. Two consequences are
deliberate — orientation is **all-or-nothing** (a partial orientation hold is
not expressible in six slots), and with several candidates the hold is the
**intersection** over all of them, so it stays correct whichever one wins.

The hold is a soft cost, so `check_straightness` (default on) reproduces the
reference's partial-path guarantee client-side: one `Fk.srv` batch over the
whole trajectory, fail the stage when the tool strays more than
`straightness_tol` (default 0.01 m) from the segment. A failure is not fatal —
it propagates to the enclosing `Fallbacks`, which is how the reference
pipeline reached the next candidate after rejecting a Cartesian solve.

This stage is **forward-only**: the hold comes from the start pose, so a
backward (end-seeded) move would need a two-pass plan.

## Server requirements

- `max_goalset > 1` for `goal.poses` / candidate fan-out. The server rejects an
  oversized goalset outright rather than truncating it, so the per-stage
  candidate count must stay `<= max_goalset`. The buffer costs memory
  proportionally; the launch file defaults it to 8.
- `trajectory_constraints` is honoured **only** by `ClassicPlanner`.
  `JointSpacePlanner` ignores it.
- `MULTIPOINT` (4) is reserved-but-unimplemented in `SetPlanner.srv`, as is
  `BATCH` (2). `max_goalset` caps **candidates within one goalset**, not
  waypoints — a multipoint path is one goalset with one pose per waypoint, and
  no shipped planner consumes that shape.

## Tests

```
pytest tests/ -q
```

`tests/mock_curobo.py` is an analytical `RobotInterface` double — a 3-joint
planar arm with a closed-form FK/IK pair — so the whole framework runs with no
GPU and no ROS. It honours `trajectory_constraints` (interpolating the tool
through Cartesian space and IK'ing each waypoint back), and takes `bow=` (metres
of lateral deviation injected at mid-path) and `winner_index=` so the
straightness gate and candidate ranking have failing branches to exercise.
