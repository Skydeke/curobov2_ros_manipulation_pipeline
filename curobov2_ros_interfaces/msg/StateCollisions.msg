# One robot state's collision diagnostics (populated on planning failure).
# Carried in TrajectoryGeneration.Response and SendTrajectory.Result as
# `start_state_collisions` (the plan's start configuration) and
# `end_state_collisions` (the goal / final waypoint configuration; joint-space
# plans only — Cartesian goals have no joint target to evaluate).
WorldCollisionContact[] world_contacts
SelfCollisionContact[] self_contacts
JointLimitViolation[] cspace_violations