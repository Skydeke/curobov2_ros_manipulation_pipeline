# Live reactive (MPC / retarget) goal update.
#
# `header.frame_id` is the frame of `pose` (empty = planner base frame).
# `tool_frame` names the tool the pose retargets; empty means the solver's
# first tool (legacy single-arm publishers). A bimanual publisher must set it
# so the server updates the arm that actually moved.
std_msgs/Header header
string tool_frame
geometry_msgs/Pose pose
