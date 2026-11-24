# cuRobo, cuTAMP and GraspGenX images

Three separate layers:

- `docker/Dockerfile` — standalone cuRobo image.
- `docker/Dockerfile.cutamp` — cuTAMP (`cutamp/cutamp`, `cutamp_ros_interfaces`, `cutamp_ros`).
- `docker/Dockerfile.graspgenx` — GraspGenX (`graspgenx/GraspGenX`, `graspgenx_ros`).

Each image layers one component on top of an existing cuRobo image and can be
combined by passing the previous image as `BASE_IMAGE`.

## Build

```bash
# 1. cuRobo
docker build -f docker/Dockerfile -t curobo:latest .

# 2. cuTAMP
docker build -f docker/Dockerfile.cutamp \
  --build-arg BASE_IMAGE=curobo:latest \
  -t curobo-cutamp:latest .

# 3. GraspGenX (on cuTAMP, or directly on curobo:latest)
docker build -f docker/Dockerfile.graspgenx \
  --build-arg BASE_IMAGE=curobo-cutamp:latest \
  -t curobo-cutamp-graspgenx:latest .
```

`graspgenx_ros` requires `grasp_pose_detection_interfaces` in the base image.

## Runtime

- Checkpoints and gripper descriptions are not part of `cutamp`; GraspGenX
  downloads them at build time into `/opt/graspgenx/ext`. Override
  `GRASPGENX_CHECKPOINT_DIR` / `GRASPGENX_GRIPPER_CFG_DIR` or pass the
  component parameters.
- Start only the selected grasp/planner components so all nodes fit on one GPU.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by the GraspGenX
  base requirements; set it in the deployment image if needed.
