# cuRobo + cuTAMP + GraspGenX images

`docker/Dockerfile` is the standalone cuRobo image. It does not know about
cuTAMP or GraspGenX and should stay unchanged.

`docker/Dockerfile.cutamp` layers the additional Python libraries and ROS
components on top of an existing cuRobo image:

- cuTAMP (`cutamp/cutamp`) and GraspGenX (`graspgenx/GraspGenX`), installed
  editable from the submodules;
- `cutamp_ros_interfaces` (the `Plan` service) and `cutamp_ros` (the
  `cutamp_ros::CuTampComponent`), built with colcon into the existing
  `${ROS_WS}`; `cutamp_ros` installs its Python helper (`cutamp_ros/bridge.py`)
  into the colcon site-packages directory so the cuRobo process's embedded
  interpreter can import it;
- `graspgenx_ros`, built with colcon into the existing `${ROS_WS}`;
- the Python dependencies those packages import (`diffusers`, `omegaconf`,
  `jaxtyping`, `roma`, `msgpack`, `msgpack-numpy`, `pyzmq`).

Rerun is intentionally not installed here: the base image's numpy 1.26 conflicts
with current `rerun-sdk`/`pyarrow`. cuTAMP treats rerun as optional; use the
separate introspection image for replay/debugging.

The component needs the robot-side cuTAMP data (cuRobo v2 robot YAML, URDF,
tool frame, `tool_from_ee`, closed-gripper spheres). Point the launch's
`cutamp_config` at a flat parameter YAML; see
`cutamp_ros/config/cutamp_params.example.yaml`.

## Build

```bash
# 1. standalone cuRobo image
docker build -f docker/Dockerfile -t curobo:latest .

# 2. cuRobo + cuTAMP / GraspGenX
docker build -f docker/Dockerfile.cutamp \
  --build-arg BASE_IMAGE=curobo:latest \
  -t curobo-cutamp:latest .
```

For KATE, use the existing robot image as the base:

```bash
docker build -f docker/Dockerfile.cutamp \
  --build-arg BASE_IMAGE=curobo:latest \
  -t curobo-cutamp:latest .
```

That image already contains the KATE packages and
`grasp_pose_detection_interfaces`; the standalone generic image needs the
interface package if `graspgenx_ros` is built.

## Runtime

- Checkpoints and gripper descriptions are not downloaded at build time.
  Mount them and set `GRASPGENX_CHECKPOINT_DIR` /
  `GRASPGENX_GRIPPER_CFG_DIR` (or pass the component parameters).
- All nodes share one 8 GB GPU; start only the selected grasp/planner
  components.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the image.
