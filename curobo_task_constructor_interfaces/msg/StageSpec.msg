# One node in the task graph, MTC StageDescription shape.
#
# stage_type/container_type/params_yaml are the curobo authoring extension
# (MTC describes stages via flags + Property[]; the open stage registry
# resolves stage_type, and params_yaml carries what MTC would put in
# properties — new capabilities never touch this file).
#
# The task graph is carried FLAT, exactly like MTC's
# moveit_task_constructor_msgs/StageDescription: "id" is unique within the
# task and "parent_id" points at the parent node's id; the root marks itself
# with parent_id == id. Messages are emitted in pre-order (a stage's parent
# always precedes it, the root is element 0) so a receiver can re-link the
# tree with one pass. (Recursive message types are not buildable in ROS 2 —
# the C++ codegen cannot size a struct containing a vector of itself — which
# is why the tree is flattened instead of nested.)

uint32 id                    # unique within the task
uint32 parent_id             # == id marks the root
string name                  # stage name
uint32 flags                 # interface flags (MTC bits: 0x01 reads start,
                             # 0x02 reads end, 0x04 writes next start,
                             # 0x08 writes prev end); 0 when unknown
string stage_type            # e.g. "current_state", "fixed_state", "move_to", ...
string container_type        # "" | "serial" | "alternatives" | "fallbacks" | "independent"
string params_yaml           # stage-specific YAML config
