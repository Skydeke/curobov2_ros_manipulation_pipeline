# isaac_ros_cumotion_robot_description

Robot description files (URDF, XRDF) and a CLI builder for creating cuRobo robot configurations.

## CLI: `builder`

Build a cuRobo robot configuration (collision spheres + self-collision matrix) from a URDF.

### Kortex (Kinova Gen3 + Robotiq 2F-140)

```bash
# 1. Generate URDF from xacro
xacro $(ros2 pkg prefix iki_kortex_description)/share/iki_kortex_description/urdf_xacro/kortex_standalone.urdf.xacro \
  name:=gen3 arm:=gen3 dof:=7 gripper:=robotiq_2f_140 sim_gazebo:=true \
  > /tmp/kortex.urdf

# 2. Build cuRobo config (package:// URIs resolved automatically)
ros2 run isaac_ros_cumotion_robot_description builder \
  --urdf /tmp/kortex.urdf \
  --output kortex_curobo.yml \
  --tool-frames grasping_frame \
  --compute-metrics \
  --visualize
```

### Any other robot

```bash
ros2 run isaac_ros_cumotion_robot_description builder \
  --urdf /path/to/robot.urdf \
  --output robot.yml \
  --tool-frames tool0 \
  --compute-metrics
```

### Edit existing config

```bash
ros2 run isaac_ros_cumotion_robot_description builder \
  --edit-config robot.yml \
  --output robot_refined.yml \
  --refit-link gripper_link \
  --recompute-collisions \
  --sphere-density 2.0
```

## Arguments

| Argument | Description |
|---|---|
| `--urdf` | Path to URDF file |
| `--edit-config` | Path to existing .yml config (mutually exclusive with --urdf) |
| `--output` | Output file (.yml or .xrdf) |
| `--export-xrdf` | Also export XRDF alongside YAML |
| `--tool-frames` | End-effector link names (e.g. `grasping_frame`) |
| `--sphere-density` | Sphere density multiplier (default: 1.0) |
| `--coverage-weight` | Sphere fitting coverage weight |
| `--protrusion-weight` | Sphere fitting protrusion weight |
| `--compute-metrics` | Print per-link sphere fit metrics |
| `--clip-link LINK AXIS OFFSET` | Clip spheres on a link (repeatable) |
| `--num-collision-samples` | Samples for collision matrix (default: 1000) |
| `--no-prune` | Skip collision pruning |
| `--visualize` | Start Viser viewer at `http://localhost:8080` |
| `--viz-port` | Viser port (default: 8080) |
| `--seed` | Random seed for reproducibility |

## How it works

`package://` URIs in the URDF are resolved to absolute filesystem paths via
`ament_index_python`
