#pragma once

#include <rviz_common/config.hpp>
#include <rviz_common/panel.hpp>

#include <QString>

class QTabWidget;

namespace isaac_ros_cumotion_rviz
{

class PlanningTab;
class ContextTab;
class JointsTab;
class ManipulationTab;
class SceneObjectsTab;

}  // namespace isaac_ros_cumotion_rviz

namespace isaac_ros_cumotion_rviz
{

/// The unified curobo panel: ONE dock whose tabs mirror MoveIt's Motion Planning
/// panel, instead of several separate docks plus a joint-state monitor nobody
/// had. Nothing here is a dock of its own -- the five tabs are pages of a single
/// QTabWidget, so there is exactly one panel to add, move and resize.
///
/// Tabs, in the order they appear, and where each one comes from. The order is
/// MoveIt's own, minus the tabs it has between them: MoveIt inserts Joints
/// programmatically at index 2 (see `MotionPlanningFrame`'s
/// `ui_->tabWidget->insertTab(2, joints_tab_, "Joints")`), so its real order is
/// Context, Planning, Joints, Scene Objects, Stored Scenes, Stored States, Status,
/// Manipulation. The three kept here out of that list are genuine MoveIt tab
/// names, not borrowed ones.
///
///  * **Context** — ContextTab: MoveIt's "Planning Library" group, whose first
///    row carries MoveIt's own "Planning Group:" label over the planner-node
///    combo, plus the trajectory type and the server's control strategy, and one
///    ungrouped collision-sphere checkbox below it. It is the authoritative home
///    of the planner node and the trajectory type; see "who owns what" below.
///  * **Planning** — PlanningTab, the EXISTING planner-args panel (formerly the
///    standalone "RvizArgsPanel"), re-laid out to MoveIt's Planning tab: a
///    "Query" group holding the goal pose as a position triple and an
///    orientation quaternion (x, y, z, w) — a quaternion because that is what
///    `geometry_msgs/msg/Pose` carries and what curobo's IK takes, so the whole
///    path is one identity instead of an RPY round trip. An "Options" group with
///    time dilation and a read-only echo of the
///    planner node, and a "Commands" group whose strip is Clear / Compute /
///    Execute / Compute + Execute / Cancel followed by the status line.
///  * **Joints** — JointsTab: MoveIt's Joints widget, reproduced rather than
///    approximated — a bare "Group joints:" label over a
///    `Joint | Current | Target` table with the value column painted as a
///    DRAGGABLE PROGRESS BAR (ProgressBarDelegate, a port of MoveIt's), with no
///    group box, because MoveIt's Joints tab has none either. Below it, a
///    "Commands" group. It also carries the pre-flight collision/limit check.
///  * **Manipulation** — ManipulationTab: MoveIt's own Manipulation tab, which is
///    an object-DETECTION tab, laid out as MoveIt lays it out (a two-column
///    QGridLayout: "Detected Objects" spanning both rows on the left, "Support
///    Surfaces" and "ROI" stacked on the right). curobo has no object recognition,
///    no support surfaces and no ROI — no service for any of the three — so the
///    tab is MoveIt's frame without its content: `&Detect`, `&Pick` and `P&lace`
///    and the six ROI spinboxes are ABSENT and each omission names the service
///    that would be needed. It used to be the Joints widget scoped to the gripper
///    joints; that was the wrong tab, and it duplicated the Joints tab, which
///    already lists `finger_joint` because the cspace does.
///  * **Scene Objects** — SceneObjectsTab, MoveIt's two columns: "Current Scene
///    Objects" over a live flat QListWidget fed by `get_scene_objects`, then
///    "Add/Remove scene object(s)" with MoveIt's own three size spinboxes, shape
///    combo and [Add] [Del] [Clr] toolbuttons, then an "Object status" group in
///    the right column. The add form is part of this widget now rather than a
///    separate panel stacked above the list, which is not a shape MoveIt has.
///
/// Every group title in the panel is MoveIt's own wording, transcribed from
/// `motion_planning_rviz_plugin_frame.ui`. Every MoveIt group curobo has no
/// honest control for is ABSENT rather than stubbed — a permanently dead control
/// is worse than a missing one — and each omission is named in the source next to
/// the code it would have gone in. What is absent, tab by tab: Context's
/// "Warehouse" (curobo has no planning database) and "Workspace" (no service
/// reports a scene centre and size); Planning's "Path Constraints" (TrajectoryGoal
/// has no such field) and four of Options' five numbers (planning time and
/// attempts have no equivalent, and `waypoint_tolerance` is rejected by the
/// server's own `_validate_planning_options` on both planner types this panel
/// offers); Joints'
/// "Nullspace exploration:" sliders (`generate_trajectory` takes a goal
/// configuration, not a nullspace goal); Scene Objects' "Change object pose/scale"
/// (no object edit service) and "Scene Geometry" (no `PlanningSceneMonitor` to
/// Publish / Export / Import); and the whole of Manipulation's CONTENT — its
/// `&Detect` button, its `&Pick` / `P&lace` buttons, its two lists and the six
/// ROI spinboxes, since curobo has no object recognition, no support surfaces
/// and no ROI. Manipulation keeps MoveIt's frame; MoveIt's Perception tab is
/// Stored Scenes, Stored States and Status, and MoveIt's Trajectory Execution
/// (playback of a plan the server already reports back through the action
/// result), are likewise absent.
///
/// The Planning tab is the EXISTING panel class, constructed directly and
/// parented into the tab widget rather than re-implemented. That is the whole
/// point of the composite: its readiness polling, MPC loop and marker sync are
/// ~1500 lines of subtle behaviour, and a copy would be a fork that silently
/// diverges the first time one of them is fixed. It is no longer registered as a
/// standalone rviz panel, so the class was renamed PlanningTab and the five tabs
/// are one family; it still derives from rviz_common::Panel because that is
/// where its DisplayContext and Config-based load/save come from.
///
/// Who owns what, and why it is centralised here:
///  * The **planner node** and the **trajectory type** are chosen on the Context
///    tab, because they describe the server rather than one request -- which is
///    where MoveIt keeps them too. ContextTab emits, this panel relays: one
///    signal fans the node out to every tab that has clients, so a panel that
///    plans against `curobo_server` while its object list reads
///    `unified_planner` cannot happen. The Planning tab keeps a read-only echo of
///    the node. The Manipulation tab is not in the fan-out because it has no
///    clients to repoint.
///  * The **readiness probe** runs in the Planning tab (it owns the
///    node_is_available parameter client) and is forwarded back to the Context
///    tab, which uses it to gate the trajectory-type dropdown.
///
/// The task constructor is deliberately NOT one of these tabs. Its panel is a
/// standalone dock in curobo_task_constructor_rviz, because a task run is its
/// own workflow with its own lifecycle: it is about what to execute, not about
/// this planner, and folding it in would also have made this package depend on
/// the task-constructor one.
///
/// Two things embedding does not do for free, both handled in onInitialize():
///
///  1. It does not forward rviz's DisplayContext. rviz installs one on a
///     standalone panel before calling onInitialize(), and a panel constructed by
///     another panel never receives it — so the Planning tab exposes
///     initializeEmbedded() to take the container's context instead, and prefers
///     it over the base class's. Without it the Planning tab's TargetDisplay
///     lookup, which is where it reads the goal pose from, stays empty.
///  2. It does not hand out a ROS node. The four plain-QWidget tabs take rviz's
///     own node (from the display context) rather than creating one, because rviz
///     already spins it in RosNodeAbstraction and rclcpp refuses to add a node to
///     a second executor. This is also why those tabs spin nothing, unlike the
///     Planning tab, which owns a private node + NodeSpinner because its clients
///     were built to be self-sufficient.
class CuroboPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit CuroboPanel(QWidget * parent = nullptr);
  ~CuroboPanel() override;

  void onInitialize() override;

  void load(const rviz_common::Config & config) override;
  void save(rviz_common::Config config) const override;

private:
  /// Repoint every other tab at the planner node the Context tab is currently
  /// using, so all five tabs always talk to ONE server. Re-run after load() too,
  /// so a saved planner_node_name propagates to every tab -- and re-run from the
  /// Context tab's own signal, so a live switch propagates just the same.
  void syncPlannerNode();

  QTabWidget * tabs_ = nullptr;

  // Each is a raw pointer to a heap widget owned by tabs_ (a tab's widget is
  // destroyed with the tab), which keeps the declaration order readable.
  ContextTab * context_tab_ = nullptr;
  PlanningTab * planning_tab_ = nullptr;
  JointsTab * joints_tab_ = nullptr;
  // A frame, not a controller: MoveIt's Manipulation tab is an object-perception
  // grid, and curobo has no perception service to put behind any of it. It holds
  // no node, no clients and no state, so it takes no part in the planner-node
  // fan-out below. See manipulation_tab.hpp.
  ManipulationTab * manipulation_tab_ = nullptr;
  // The whole Scene Objects tab, list and add/remove form together, which is how
  // MoveIt has it: one widget, two columns. There is no separate AddObjectsPanel
  // any more — it was stacked above the list in a splitter, which is not a shape
  // MoveIt has.
  SceneObjectsTab * scene_objects_tab_ = nullptr;
};

}  // namespace isaac_ros_cumotion_rviz
