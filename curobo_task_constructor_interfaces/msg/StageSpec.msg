# One node in the task graph, generic across all registered stage types.
#
# stage_type is resolved through the executor's STAGE_REGISTRY (an open set,
# keyed by string) — never a closed enum. Adding a capability means writing a
# Stage subclass + register_stage("name"), no message changes.
#
# containers are themselves stages: a node with a non-empty container_type is
# built as the matching ContainerStage subclass. "params_yaml" is an opaque
# string parsed by the stage class itself — this is what keeps the wire format
# open: new stage types invent their own param schema without touching
# StageSpec.msg.
#
# The task graph is carried FLAT (StageSpec[]), exactly like MTC's
# moveit_task_constructor_msgs/StageDescription: "id" is unique within the
# task and "parent_id" points at the parent node's id; the root marks itself
# with parent_id == id. Messages are emitted in pre-order (a stage's parent
# always precedes it, the root is element 0) so a receiver can re-link the
# tree with one pass. (Recursive message types are not buildable in ROS 2 —
# the C++ codegen cannot size a struct containing a vector of itself — which
# is why the tree is flattened instead of nested.)

uint32 id                    # unique within the task
uint32 parent_id             # == id marks the root
string stage_type            # e.g. "current_state", "fixed_state", "move_to", ...
string name                  # optional display name; stage may default it
string container_type        # "" | "serial" | "alternatives" | "fallbacks" | "independent"
string  params_yaml          # stage-specific YAML config