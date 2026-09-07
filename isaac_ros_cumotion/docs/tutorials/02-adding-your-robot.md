# Tutorial 2: Adding Your Robot

**Difficulty**: Intermediate · **Time**: ~45 min · **Prerequisites**: [Tutorial 1](01-first-trajectory.md)

`curobo_ros` is **robot-abstract**: it ships no robot model or hardware driver wiring. Each robot — its cuRobo kinematic configuration and its ROS driver topics — lives in its own repository (a "deploy repo", e.g. a Kortex package) and is selected at launch with the `robot` argument.

Integrating your robot therefore takes two files, both owned by your deploy repo:

1. A **cuRobo robot configuration YAML** — kinematics (URDF), joint limits, and collision spheres. This is what the GPU solvers consume.
2. A **robot descriptor** — a small YAML that tells curobo_ros where the cuRobo config is and how to *command* the robot (topics, control strategy).

## 1. The robot descriptor (`robots/<name>.yaml`)

The `robot` launch argument selects a descriptor either **by name** (resolved against `robots/<name>.yaml` inside the installed package that ships it) or by an **absolute path**. `load_robot_description` accepts both.

A deploy repo ships its descriptor alongside its cuRobo config, e.g. `iki_kortex_moveit_config/robots/kortex.yaml`:

```yaml
name: kortex
display_name: "Kinova Kortex Gen3"

# cuRobo robot config; package:// and relative paths are resolved automatically
curobo_config: package://iki_kortex_moveit_config/config/kortex.curobo.yml

base_link: "base_link"

# How to command the robot (see Tutorial 4): emulator | joint_speed | joint_pose
strategy: joint_speed
strategy_params:
  command_topic: /joint_trajectory_controller/joint_trajectory
  joint_states_topic: /joint_states
```

`base_link`, joint names, and DOF are **not** duplicated here — they come from the cuRobo config. `strategy_params` describes only the robot driver interface. The shipped `robots/emulator.yaml` shows the minimal hardware-free variant (`strategy: emulator`, one published topic) — it intentionally ships **without** `curobo_config`, so running the emulator against a concrete robot requires supplying the model via the `robot_config_file` parameter.

To add *your* robot to a deploy repo: drop `robots/my_robot.yaml` there, point `curobo_config` at your cuRobo YAML, pick a strategy, and pass the descriptor at launch — `robot:=<absolute path to robots/my_robot.yaml>` or `robot:=my_robot` if the descriptor is installed where the name lookup finds it.

## 2. The cuRobo robot configuration

The cuRobo config in your deploy repo is the reference model. The essential structure:

```yaml
robot_cfg:
  kinematics:
    urdf_path: <path to your .urdf>
    base_link: "base_0"          # root of the kinematic chain
    ee_link: "link6"             # planning frame (tool)
    collision_spheres: { ... }   # per-link sphere approximation (see step 3)
    collision_link_names: [...]
    cspace:
      joint_names: [joint1, ..., joint6]
      # position/velocity/acceleration/jerk limits — this is also where
      # robot speed is scaled (see Tutorial 4)
```

Requirements for the URDF: a fixed, connected chain from `base_link` to `ee_link` with correct joint limits. Meshes are only needed for visualization — collision uses the spheres.

For the full schema, see the [cuRobo configuration documentation](https://curobo.org/tutorials/1_robot_configuration.html).

## 3. Collision spheres

cuRobo approximates the robot volume with spheres attached to links — this is what makes GPU collision checking fast. Guidelines:

- 2–5 spheres per link, more for long links; 10–20 % overlap between neighbours so no gap opens at joint bends.
- Cover the real volume conservatively: a missed wrist cover becomes a real collision.
- Include the tool/gripper if it is part of the URDF.

The [`curobo_robot_setup`](https://github.com/Lab-CORO/curobo_robot_setup) companion package provides an RViz-based editor for this step: load your URDF, place or auto-generate spheres interactively (backed by cuRobo's mesh sphere-fitting), and export the YAML. See its README for usage. You can also write the spheres by hand for a simple arm.

To inspect the result at runtime: enable the sphere markers and look at them in RViz —

```bash
ros2 service call /unified_planner/set_collision_spheres_enabled std_srvs/srv/SetBool "{data: true}"
```

## 4. Test the integration

Start with the emulator strategy so nothing physical moves — either temporarily set `strategy: emulator` in your descriptor, or switch at runtime:

```bash
ros2 launch curobo_ros gen_traj.launch.py robot:=<path to your descriptor> robot_config_file:=<path to your curobo.yml>
ros2 service call /unified_planner/set_robot_strategy curobo_msgs/srv/SetRobotStrategy "{robot_strategy: 'emulator'}"
```

Checklist:

1. The node reaches `node_is_available: True` without errors (watch the launch log for YAML/URDF problems).
2. The robot model looks right in RViz and the collision spheres cover it.
3. A `generate_trajectory` call to a pose in the workspace succeeds ([Tutorial 1](01-first-trajectory.md)).
4. IK sanity check ([Tutorial 6](06-ik-fk-services.md)): FK of a known joint state returns the expected pose.

Then connect the real driver via the descriptor's `strategy_params` — see [Tutorial 4](04-robot-execution.md).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Node crashes at startup parsing the config | Wrong indentation or missing `kinematics` keys in the cuRobo YAML |
| `No robot model configured` | The descriptor ships without `curobo_config` and no `robot_config_file` was provided at launch — supply the model |
| Plans collide with the real robot body | Sphere coverage too sparse — add/enlarge spheres |
| IK always fails | `ee_link` name doesn't match the URDF, or joint limits are wrong |

## Next steps

- [Tutorial 4: Robot Execution](04-robot-execution.md) — wire up the real driver
- [Parameters Guide](../concepts/parameters.md) — solver parameters that interact with the robot config

[← Tutorial 1](01-first-trajectory.md) | [Tutorial 3 →](03-collision-objects.md)