# graspgenx_ros

Drop-in `grasp_pose_detection_interfaces/srv/GenerateGrasps` backed by
GraspGenX. The node is an `rclcpp` component and uses the CPython interpreter
owned by the cuRobo `PyBridge`; it never creates an interpreter or a CUDA
context.

## GPU budget

All AI/ROS nodes share one 8 GB GPU. Consequences:

- Load this component only when the GraspGenX backend is selected.
- GPD may stay resident as the object-segmentation publisher, but call only one
  `generate_grasps` service (GPD's or this one), never both in one task.
- Point `objects_topic` at the publisher (e.g.
  `/perception/grasp_pose_detection/objects`).
- Keep `num_grasps` and `max_candidates` configurable; measure against the
  shared budget before raising them.
- Prefer `warmup:=true` in the launch so the model and kernels are loaded once
  at startup, not inside the first task.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the container.
- A Python/CUDA error is returned as `success=false`; it must not kill the
  container.

## Data flow

`ObjectArray` (`~/objects`, transient-local) -> `object_index` selection ->
GraspGenX -> `Grasp[]`. The per-object cloud must already be in `frame_id`
(e.g. `base_footprint`); the component rejects a mismatched frame instead
of relabelling it.

GraspGenX poses are canonical gripper-base poses (`+Z` approach, `+X` close).
`grasp_to_tool` maps them to the arm tool frame:

```text
grasp     = T_grasp @ grasp_to_tool
pre_grasp = grasp @ Trans(-pre_grasp_dist, 0, 0)
retreat   = grasp @ Trans(retreat_axis * retreat_dist)
```

`pre_grasp_dist`, `retreat_axis` and `retreat_dist` are in the arm tool frame
(`+X` = approach, `-X` = retrace the approach).

The robot repo supplies `grasp_to_tool`, the gripper name/assets (or the 12
sweep-volume values), the checkpoint directory, and the optional collision
mesh.

## Main parameters

| Parameter | Meaning |
|---|---|
| `checkpoint_dir` | Root containing `gen/` and `dis/` (or `gen/` with sibling `dis/`) |
| `gripper_name`, `gripper_assets_dir` | Named gripper description; empty name switches to sweep-volume params |
| `sweep_extents_open/offset_open/extents_mid/offset_mid` | 3+3+3+3 values in the GripperGen base frame |
| `gripper_type`, `fingertip_depth` | `0/1/2` = parallel/revolute 2f/3f; optional base-to-tip |
| `grasp_to_tool` | 16 row-major values, grasp frame -> arm tool frame |
| `retreat_axis` | 3 values in the tool frame, default `[-1,0,0]` (retrace the approach) |
| `planner`, `score_threshold`, `num_grasps` | GraspGenX sampling |
| `max_candidates` | 0 = keep the request's `num_grasps_selected`; >0 = hard cap |
| `objects_topic` | Default `objects`; relative names resolve under the node |
| `frame_id` | Cloud and output frame, default `base_footprint` |
| `filter_collisions`, `collision_threshold`, `gripper_colmesh` | Optional scene collision filter |
| `warmup` | Load the model and run a synthetic inference at startup |

The service request's `pre_grasp_dist`, `retreat_dist`, and
`num_grasps_selected` take precedence when positive. `sample_cloud_from_obb`,
`disable_top_grasp`, `min_distance_to_plane`, `approach_direction` and
`thresh_deg` have no GraspGenX equivalent and are ignored.
