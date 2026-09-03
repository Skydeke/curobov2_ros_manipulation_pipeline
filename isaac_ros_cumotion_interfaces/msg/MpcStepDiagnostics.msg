std_msgs/Header header

# Timing for this control cycle's solve, and the real-time budget it was
# checked against (LBFGSController: the published segment's duration,
# (n-1)*interpolation_dt; MPPIController: mpc_command_interval). budget_ms
# is 0.0 when the caller has no fixed budget to compare against (e.g.
# MPPIController running unpaced) -- budget_exceeded is then always false.
float64 solve_ms
float64 budget_ms
bool budget_exceeded

# Trajectory actually handed to the robot for this cycle -- position,
# velocity and acceleration per point, in the same units and points as sent
# onward to the Doosan driver (radians / rad/s / rad/s^2). Left EMPTY when
# budget_exceeded: an already-late cycle should not pay for building and
# publishing more data on top of the overrun, so only solve_ms/budget_ms/
# budget_exceeded above are meaningful in that case.
trajectory_msgs/JointTrajectory trajectory

# FK error of the trajectory's LAST point above -- what the optimizer
# expects to have reached once this segment finishes playing, not a
# real-time measured error (see fk_err_m in MpcCosts for that). NaN when
# budget_exceeded.
float64 arrival_fk_err_m
float64 arrival_fk_rot_err_deg

# Cost terms (horizon-summed), from the optimization rollout's cost_cfg --
# see MPCDiagnostics.cost_breakdown() for how these are recovered safely
# (without an extra GPU rebatch call) and cost_cspace_vel/acc/jerk's caveat
# (a Python-side proxy, not a true decomposition of cost_cspace). NaN when
# budget_exceeded, or when unavailable for the active solver/config.
float64 cost_tool_pose_pos
float64 cost_tool_pose_orient
float64 cost_cspace
float64 cost_cspace_vel
float64 cost_cspace_acc
float64 cost_cspace_jerk

# Constraint terms (horizon-summed); 0.0 = no violation. NaN when
# budget_exceeded, or when unavailable for the active solver/config.
float64 con_self_collision
float64 con_scene_collision
float64 con_cspace_bound
