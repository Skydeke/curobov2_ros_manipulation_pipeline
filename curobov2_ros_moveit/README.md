# curobov2_ros_moveit

MoveIt 2 planning plugin that exposes the cuRobo server as a MoveIt planning
pipeline (`curobov2_ros/CumotionPlanner`). It translates a `MotionPlanRequest`
into a cuRobo solve and offers the `cuMotion/JointSpace` planner id.
