# isaac_ros_cumotion_extra

Extra tools for isaac_ros_cumotion: ESDF/Viser visualizer and cuRobo config generator.

## CLI: `build_curobo_config`

Build a cuRobo v2 YAML config (collision spheres + self-collision matrix) from a URDF.

### Kortex (Kinova Gen3 + Robotiq 2F-140)

```bash
# 1. Generate URDF from xacro
xacro $(ros2 pkg prefix iki_kortex_description)/share/iki_kortex_description/urdf_xacro/kortex_standalone.urdf.xacro \
  name:=gen3 arm:=gen3 dof:=7 gripper:=robotiq_2f_140 sim_gazebo:=true \
  > /tmp/kortex.urdf

# 2. Build cuRobo config
ros2 run isaac_ros_cumotion_extra build_curobo_config \
  --urdf /tmp/kortex.urdf \
  --asset-path ~/ros2_ws \
  --output /tmp/kortex_curobo.yml \
  --tool-frame grasping_frame
```

### Any other robot

```bash
ros2 run isaac_ros_cumotion_extra build_curobo_config \
  --urdf /path/to/robot.urdf \
  --asset-path /path/to/mesh/parent \
  --output robot.yml \
  --tool-frame tool0
```

## Arguments

| Argument | Description |
|---|---|
| `--urdf` | Path to URDF file |
| `--asset-path` | Parent directory containing mesh dirs |
| `--output` | Output YAML path (default: `curobo_config.yml`) |
| `--tool-frame` | Tool frame name (default: `grasping_frame`) |
| `--base-link` | Base link (auto-detected if omitted) |
| `--sphere-density` | Sphere density multiplier (default: 1.0) |
| `--num-collision-samples` | Collision matrix samples (default: 1000) |
| `--visualize` | Start Viser viewer |

## Viser nodes

The package provides viser visualization nodes with interactive GUIs. They keep
the ROS2 service architecture — motion is computed on `curobo_server` while the
viser view adds interactive controls:

| Node | Interactive UI |
|---|---|
| `fk_viser_node` | per-joint sliders + a "Sweep" toggle to animate the arm |
| `ik_viser_node` | a draggable 6-DOF goal gizmo; IK re-solves live as you drag it |
| `mp_viser_node` | draggable goal gizmo + upstream "Move"/"Grasp" buttons (Classic vs MultiPoint plan) and a joint-trajectory plot |
| `mpc_viser_node` | draggable target gizmo driving a live MPC closed loop via the execution action + live-goal topic |

```bash
ros2 launch isaac_ros_cumotion_extra getting_started_viser.launch.py \
  content_path:=/path/to/robot.curobo.yml \
  urdf_path:=/tmp/robot.urdf \
  server_node:=curobo_server \
  nodes:=mp_viser_node
```

Leave `nodes` empty to start all eight demo nodes (each on its own viser port).
