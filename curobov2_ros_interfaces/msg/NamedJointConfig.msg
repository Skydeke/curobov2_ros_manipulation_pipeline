# One named joint configuration from the robot descriptor's
# `named_joint_configs:` map.
#
# The configuration is a NAME -> position map, not a cspace vector: a pose may
# name only a subset of the cspace joints (and the descriptor deliberately never
# names the jaws, which are not planning DOFs). Consumers therefore index by
# joint name, exactly like the task constructor's FixedState/MoveTo stages.
string name
string[] joint_names
float64[] positions
