# One considered (attempted) solution of a stage, MTC SolutionInfo shape.
# The executor publishes one per stage attempt as compute() runs, matching
# MTC's choice to keep failed attempts visible for debugging (MTC marks a
# failure with cost=inf; the explicit success flag below says the same).
uint32 id                    # unique record id within the stage's id space
float32 cost                 # associated cost, inf = failed attempt
string comment               # failure hint / human note
uint32 stage_id              # id of stage that created this solution
string planner_id            # planner backing this stage (e.g. "classic")
visualization_msgs/Marker[] markers   # debug markers, e.g. goal candidates
# --- curobo extensions (no MTC counterpart; needed without moveit_msgs) ---
string task_id               # task this attempt belongs to
string stage_name            # display name of the creating stage
bool success                 # false for failed attempts (cost is inf then)
bool ranked true             # ranked (stored, lifted, executable) vs a
                             # best-of-N seed that succeeded but lost: shown
                             # and counted, never ranked or executed
sensor_msgs/JointState[] trajectory   # waypoints (empty on failure)
geometry_msgs/Pose[] tool_poses       # FK tool pose per waypoint
