# Tutorial 7: Camera-Based Obstacle Detection

**Difficulty**: Intermediate · **Time**: ~40 min · **Prerequisites**: [Tutorial 3](03-collision-objects.md), a depth camera (or a rosbag with depth topics)

curobo_ros integrates depth cameras directly into the collision world. Each depth frame is pushed into cuRobo v2's `Mapper` (a GPU TSDF/ESDF), and every solver — planning, MPC, IK — sees the result. No point-cloud processing on your side, no service calls: anything the camera sees becomes an obstacle.

## How it works

```
 raw depth (camera_topic) ──► RobotSegmentationCameraStrategy ──► masked depth (derived)
                                 (per segmented camera, §4)         │
                                                                   ▼
depth image topic ──► DepthMapCameraStrategy ──► mapper.integrate(CameraObservation)
                                                         │ (GPU TSDF, decays over time)
                                                         ▼
                                               ESDF shared by all solvers
```

Key properties:

- **Push-based**: integration happens in the camera callback, under a non-blocking GPU lock. Frames arriving while the GPU is busy (e.g. during CUDA-graph capture) are dropped — the next one catches up.
- **Voxels decay**: stale voxels fade with half-life `decay_half_life_s` (default 0.7 s), so a person walking through the scene doesn't leave a permanent ghost. The decay rate is derived from the cameras' declared `frame_rate_hz`.
- Only `type: depth_camera` is supported in v2 (the old pull-based point-cloud path was removed).

## 1. Configure the cameras (`camera_*` node params)

The cameras are configured with node parameters — **no `cameras.yaml` file**. One
`camera_*` param holds an *array*, one entry per camera (declared by
`PerceptionCameraCfg`):
```yaml
# Params of the unified_planner node (set in your launch file or a params YAML).
# Every entry is indexed by position; camera_index pairs a camera with the
# same-position entry of every other array.
camera_topic: [/depth_to_rgb/image_raw]        # sensor_msgs/Image, 16UC1 (mm) or 32FC1 (m)
camera_info_topic: [/depth_to_rgb/camera_info] # used when camera_intrinsics is empty
camera_frame: [rgb_camera_link]                # map/mask frame: the frame the mapper
                                               # integrates into (and the segmenter masks in);
                                               # '' -> the depth msg's own frame_id
camera_intrinsics: ['']                        # '' = read once from camera_info (5 s wait)
                                               # or 'fx,0,cx,0,fy,cy,0,0,1' row-major K
camera_extrinsics: ['']                        # '' = TF base frame -> camera_frame per frame,
                                               # or 'x,y,z,qw,qx,qy,qz' camera pose in base frame
camera_frame_rate_hz: [30.0]                   # declared publication rate (drives decay)
camera_purpose: [all]                          # what this camera feeds: all (default) |
                                               # esdf | segmentation
```

`camera_purpose` decides per camera which pipelines it feeds:

| Purpose | Mapper (ESDF) | Robot segmentation |
|---|---|---|
| `all` (default) | yes | yes |
| `esdf` | yes | no |
| `segmentation` | no | yes |

So a fixed scene camera can map the world (`all` or `esdf`) while a wrist camera
is used only to keep the arm out of its own way (`segmentation`). The master
switch `enable_robot_segmentation` (default `true` — "use cameras for
everything") additionally gates all segmentation streams.

The raw depth topics and camera-info topics are **shared with
`RobotSegmentation`** — both consumers read the same arrays, so they can't
drift apart.

Two decisions per camera:

- **Intrinsics** — leave `''` to read them once from the `camera_info` topic at startup, or hardcode a comma-separated K: `'fx,0,cx,0,fy,cy,0,0,1'`.
- **Extrinsics** — hardcode `'x,y,z,qw,qx,qy,qz'` (recommended once calibrated), or leave `''` to resolve TF `base frame → camera_frame` per frame. With TF, frames are **dropped** when the transform is unavailable — there is no identity-pose fallback.

The depth streams are pushed into the Mapper from within the planner node; the
old multi-camera YAML file is gone (multiple cameras are configured by adding
array entries).

## 2. Launch with the cameras

```bash
ros2 launch curobo_ros gen_traj.launch.py robot:=emulator robot_config_file:=<path-to-your-robot-curobo.yml> \
  camera_topic:='["/depth_to_rgb/image_raw"]' \
  camera_info_topic:='["/depth_to_rgb/camera_info"]'
```

Startup log lines to look for:

```
Camera 'depth_camera_0' (index 0): raw=/depth_to_rgb/image_raw, purpose=all [esdf+segmentation], ...
Added camera strategy 'depth_camera_0' of type 'depth_camera' ...
RobotSegmentation stream for camera 'depth_camera_0' (index 0): /depth_to_rgb/image_raw -> /depth_to_rgb/masked_depth
Added camera strategy 'depth_camera_0' of type 'robot_segmentation' ...
Camera intrinsics from topic: fx=...        # or: Using static extrinsics from config file
DepthMap camera initialized with depth topic: /depth_to_rgb/masked_depth
```

Relevant perception parameters (see [Parameters](../concepts/parameters.md)): `mapper_extent_xyz` (perception volume), `voxel_size`, `mapper_image_width`/`mapper_image_height` (frames are resized to this before integration), `decay_half_life_s`.

## 3. Verify obstacles are seen

Wave a hand (or a box) in front of the camera, inside the mapper volume, then:

```bash
# Occupied voxels streamed as a sparse grid (default 7 Hz)
ros2 topic echo /unified_planner/voxel_grid_sparse --once

# Distances of the robot spheres to the nearest obstacle
ros2 service call /unified_planner/get_collision_distance curobo_msgs/srv/GetCollisionDistance

# Full voxel snapshot of a region
ros2 service call /unified_planner/get_voxel_grid curobo_msgs/srv/GetVoxelGrid \
  "{bbox_min_x: -1.0, bbox_min_y: -1.0, bbox_min_z: 0.0, bbox_max_x: 1.0, bbox_max_y: 1.0, bbox_max_z: 1.5}"
```

Then plan through the space the obstacle occupies ([Tutorial 1](01-first-trajectory.md)): the trajectory deflects around it. Remove the obstacle, wait a second (decay), and the same plan goes straight again. With MPC active ([Tutorial 5](05-mpc-planner.md)) the avoidance happens *during* motion.

## 4. Remove the robot from its own view (in-server `RobotSegmentation`)

If the camera sees the robot arm, the arm becomes an "obstacle" for itself. The planner
node can subtract the robot from the depth image *before* integration — no separate
executable. On by default (`enable_robot_segmentation: true`); with it on, every camera
whose `camera_purpose` is `all` or `segmentation` gets its own masking stream, and that
camera's mapper input switches to its masked output automatically:

```yaml
enable_robot_segmentation: true        # master switch (default true)
camera_purpose: [all]                  # 'segmentation' -> mask, don't map
```

Each stream computes the robot's collision spheres at the depth frame's capture-time
joint state, masks every pixel within `robot_segmentation_distance_threshold` (default
0.05 m) of a sphere (camera frame from the shared `camera_*` array
entry), and republishes the cleaned image. Segmentation streams are first-class
`CameraStrategy`s — the component registers one `RobotSegmentationCameraStrategy` per
segmented camera through the same `CameraContext` the mapper uses — and each masked
output topic is **derived from that camera's own `camera_topic`**: the leaf segment is
stripped and published as `masked_depth` (e.g. raw `/kortex_vision/depth/image` →
`/kortex_vision/depth/masked_depth`), so every segmented camera gets a unique stream
with no extra configuration:

| Interface | Name | Notes |
|---|---|---|
| Subscribes | raw depth + camera info from `camera_topic` / `camera_info_topic` | one strategy per segmented camera |
| Publishes | `<derived>/masked_depth` | The depth image with the robot removed (derived from the raw topic, e.g. `/depth_to_rgb/image_raw` → `/depth_to_rgb/masked_depth`) |
| Publishes | `<derived>/robot_pointcloud_debug` | The masked-out points (base frame) |
| Services | `<node>/set_mask`, `<node>/remove_mask` | Extra user-defined masks, shared by every stream, optionally riding TF frames |

With segmentation on, each segmented camera's mapper strategy *derives* its input to
that camera's masked output topic automatically (`camera_topic` then only feeds the
segmenter). So the loop per segmented camera is: raw depth → segmenter masks the robot →
mapper integrates the cleaned image → solvers avoid what *remains*. A camera marked
`esdf` (or with the master switch off) keeps feeding the mapper its raw stream.
Key parameters: `robot_segmentation_distance_threshold`, `robot_segmentation_mask_margin`.

## Tuning

| Goal | Knob |
|---|---|
| Finer obstacles | `voxel_size` 0.02–0.03 (more VRAM; rebuild required) |
| Obstacles linger too long / flicker | `decay_half_life_s` up / down |
| Camera too far / too close clipped | `mapper_depth_min` / `mapper_depth_max` |
| Bigger workspace covered | `mapper_extent_xyz` (volume is centred on `mapper_grid_center`) |
| More safety margin | `collision_activation_distance` up |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Failed to receive camera info from <topic>` at startup | `camera_info` topic wrong or not published within 5 s — check `ros2 topic list`, or hardcode `intrinsics` |
| `Could not transform <base> to <frame_id>` warnings | TF extrinsics chosen but the transform isn't published — publish a static TF or hardcode `extrinsics` |
| Obstacles appear shifted | Wrong extrinsics — re-check the calibration (position *and* quaternion order `[qw, qx, qy, qz]`) |
| Robot avoids itself / plans fail near the arm | Camera sees the arm — insert `robot_segmentation` (section 4) |
| Nothing appears in the voxel grid | Object outside `mapper_extent_xyz`, or depth encoding not `16UC1`/`32FC1`, or `blox` cache disabled |

## Next steps

- [Tutorial 5: MPC and Reactive Control](05-mpc-planner.md) — avoid moving obstacles while moving
- [Manager Architecture](../concepts/manager-architecture.md) — how perception integrates internally

[← Tutorial 6](06-ik-fk-services.md) | [Tutorials index](index.md)
