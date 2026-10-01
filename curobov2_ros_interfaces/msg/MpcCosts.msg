std_msgs/Header header

# End-effector errors used for convergence/on_target, both measured
# independently via FK (see _fk_position_error / _fk_orientation_error in
# reactive_controller.py). These are the authoritative ones.
float64 fk_err_m
float64 fk_rot_err_deg

# DEPRECATED — do not use for anything but relative shape. Pose tracking error
# as reported by the solver itself. Measured 2026-08-07 against fk_err_m over
# three decades: pose_pos_err_m is the true error scaled by ~1e-3 (median ratio
# 1001.8), and pose_rot_err_rad plateaus around 1.6e-4, so at any sane print
# precision it carries ~16 distinct values across a whole run. Kept only so
# existing bag files stay parseable; use fk_err_m / fk_rot_err_deg instead.
float64 pose_pos_err_m
float64 pose_rot_err_rad

# Cost terms (horizon-summed), from the optimization rollout's cost_cfg.
float64 cost_tool_pose_pos
float64 cost_tool_pose_orient
float64 cost_cspace

# Constraint terms (horizon-summed); 0 = no violation.
float64 con_self_collision
float64 con_scene_collision
float64 con_cspace_bound
