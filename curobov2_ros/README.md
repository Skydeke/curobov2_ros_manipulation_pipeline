# curobov2_ros

The GPU cuRobo server (C++). The `curobo_trajectory_planner` node exposes
planning (trajectory generation, batch, IK/FK, collision distance), the
scene/ESDF world and robot segmentation, and the `SendTrajectory` action used
to execute a validated trajectory. `curobo_planner_component` wraps the same
server as a composable node.
