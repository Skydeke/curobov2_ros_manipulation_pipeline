#include <curobov2_ros_rviz/curobo_panel.hpp>

#include <curobov2_ros_rviz/context_tab.hpp>
#include <curobov2_ros_rviz/joints_tab.hpp>
#include <curobov2_ros_rviz/manipulation_tab.hpp>
#include <curobov2_ros_rviz/planning_tab.hpp>
#include <curobov2_ros_rviz/scene_objects_tab.hpp>

#include <rviz_common/display_context.hpp>
// Complete RosNodeAbstraction, needed for ->get_raw_node() (a weak_ptr from
// getRosNodeAbstraction() alone would only need the declaration). Path is
// rviz_common/ros_integration/..., matching target_display.cpp,
// curobo_trajectory_display.cpp and reachability_map_display.cpp.
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

#include <QTabWidget>
#include <QVBoxLayout>
#include <QWidget>

namespace curobov2_ros_rviz
{

CuroboPanel::CuroboPanel(QWidget * parent) : rviz_common::Panel(parent)
{
  setObjectName("CuroboPanel");

  auto * layout = new QVBoxLayout(this);
  layout->setContentsMargins(2, 2, 2, 2);

  tabs_ = new QTabWidget(this);
  layout->addWidget(tabs_);

  // Tab order and names are the five asked for: Context, Planning, Joints,
  // Manipulation, Scene Objects. No tab is in a scroll area and none is split:
  // every tab's layout comes from a .ui that reflows, so all five fill the dock
  // directly, and each is shaped like the MoveIt tab it stands for.

  // --- Context: which server, which trajectory type, how it is driven ---------
  context_tab_ = new ContextTab();
  tabs_->addTab(context_tab_, tr("Context"));

  // --- Planning: the existing planner-args panel, reused as a tab -------------
  planning_tab_ = new PlanningTab();
  tabs_->addTab(planning_tab_, tr("Planning"));

  // --- Joints: live joint monitor + joint-space planning ----------------------
  joints_tab_ = new JointsTab();
  tabs_->addTab(joints_tab_, tr("Joints"));

  // --- Manipulation: MoveIt's own object-perception grid, controls absent -----
  // Its controls are absent because curobo has no perception services to wire
  // them to. See manipulation_tab.hpp: this tab is a frame, not a gripper, and
  // the gripper it briefly duplicated is reachable from the Joints tab.
  manipulation_tab_ = new ManipulationTab();
  tabs_->addTab(manipulation_tab_, tr("Manipulation"));

  // --- Scene Objects: MoveIt's two columns, list + add/remove form -------------
  // The tab itself, not a splitter over an add form and a list: MoveIt's Scene
  // Objects tab is ONE widget with the list and the form in its left column, and
  // the add form used to be a separate AddObjectsPanel stacked above the list,
  // which is the main reason this tab did not look like MoveIt's. The form now
  // lives in MoveIt's "Add/Remove scene object(s)" group, and the one object
  // list comes from the server's get_scene_objects. See scene_objects_tab.hpp.
  scene_objects_tab_ = new SceneObjectsTab();
  tabs_->addTab(scene_objects_tab_, tr("Scene Objects"));

  // The Context tab owns the planner node and the trajectory type — the two
  // settings that describe the server rather than one request — and the others
  // follow it. Both connections are made here, in the constructor, on purpose:
  // load() runs before onInitialize(), and it is load() that pushes a saved
  // planner node into ContextTab's combo, which is what fires the first one.
  connect(context_tab_, &ContextTab::plannerNodeChosen, this,
          [this](const QString & planner_node) {
            planning_tab_->setPlannerNode(planner_node);
            syncPlannerNode();
          });
  connect(context_tab_, &ContextTab::trajectoryTypeChanged, planning_tab_,
          &PlanningTab::setTrajectoryType);

  // Readiness is probed by the Planning tab, which owns the parameter client, but
  // the control it gates now lives on the Context tab — so the signal goes back
  // across.
  connect(planning_tab_, &PlanningTab::plannerReadyChanged, context_tab_,
          &ContextTab::setPlannerReady);
}

CuroboPanel::~CuroboPanel()
{
  // Every tab is a QObject child of `this` (directly, or through the tab widget),
  // so they are destroyed with the panel.
}

void CuroboPanel::onInitialize()
{
  rviz_common::Panel::onInitialize();

  // The Planning tab is an rviz panel too, but rviz hands a DisplayContext to a
  // standalone panel before calling onInitialize(), and a panel constructed by
  // another panel is never given one. So it is passed down explicitly instead --
  // and AddObjectsPanel needs no call at all, since it never reads the context.
  // This panel's own getDisplayContext() is valid by this point: rviz calls
  // onInitialize() only after it has set the panel up, which is exactly why the
  // base call above succeeds.
  rviz_common::DisplayContext * display_context = getDisplayContext();
  planning_tab_->initializeEmbedded(display_context);

  // rviz's ROS node. RosNodeAbstraction already spins it on its own thread, so
  // the plain-QWidget tabs use it as-is and must NOT spin it again: rclcpp
  // throws "Node has already been added to an executor" if they did.
  //
  // The Manipulation tab is not in this list because it has no ROS state left:
  // it is MoveIt's perception grid with no controls, so there is nothing for it
  // to call. It is still a tab; it just does not listen to anything.
  if (display_context != nullptr) {
    if (auto ros_node_abstraction = display_context->getRosNodeAbstraction().lock()) {
      auto rviz_node = ros_node_abstraction->get_raw_node();
      if (rviz_node != nullptr) {
        // Context first: it is the source of the planner node, and this is the
        // first point at which it has a node to build its clients against.
        context_tab_->initialize(rviz_node);
        joints_tab_->initialize(rviz_node);
        scene_objects_tab_->initialize(rviz_node);
      }
    }
  }
  // If there is no display context, the Planning tab still works where it can, and
  // only the other two stay inert. Nothing here can be logged: this override
  // runs before the tabs have any node to log through.

  // The Context tab now has a node, so seed every other tab with the planner node
  // it settled on (from the loaded config, or its default).
  syncPlannerNode();
}

void CuroboPanel::syncPlannerNode()
{
  // Read from the Context tab, which is where the choice is made, and push it to
  // the Planning tab and the two tabs that only consume it. The Planning tab is
  // included even though it already got the value from the same signal: it is
  // idempotent, and this way load() reaches it even if the combo never emitted.
  const QString planner_node = context_tab_->plannerNode();
  planning_tab_->setPlannerNode(planner_node);
  joints_tab_->setPlannerNode(planner_node);
  scene_objects_tab_->setPlannerNode(planner_node);
}

void CuroboPanel::load(const rviz_common::Config & config)
{
  // The base class writes "Class" and "Name" — the two keys
  // VisualizationFrame::loadPanels() requires before it will even try to resolve
  // this panel. Omitting this call is what made the previous task-constructor
  // panel un-saveable: rviz wrote a nameless orphan entry and then silently
  // dropped the panel on the next load.
  rviz_common::Panel::load(config);

  // Context FIRST: it owns planner_node_name, and pushing it into its combo is
  // what emits plannerNodeChosen — which is how the other four tabs learn the
  // stored value. Loading them first would leave them on their defaults until
  // something else changed the node.
  //
  // Each tab gets its OWN child key — mapGetChild on a key that does not exist
  // yields a DETACHED config, so a child cannot write through it (that is what
  // mapMakeChild is for, in save()).
  context_tab_->load(config.mapGetChild("Context"));
  planning_tab_->load(config.mapGetChild("Planning"));
  scene_objects_tab_->load(config.mapGetChild("Scene Objects"));
  joints_tab_->load(config.mapGetChild("Joints"));

  // The Manipulation tab is deliberately not loaded, and neither is anything from
  // the old AddObjectsPanel. Both used to be, but neither has any state left to
  // persist: the planner node belongs to the Context tab, the set of gripper
  // joints is no longer re-derived from anywhere, and the shape, size, name and
  // colour in the Scene Objects add form are properties of the PLANNER's scene
  // rather than of this panel. A "Manipulation" child in an old config is
  // therefore simply ignored -- there is nothing in it this panel would honour,
  // and keeping the read would mean keeping a knob that looks configurable and is
  // not.
  syncPlannerNode();
}

void CuroboPanel::save(rviz_common::Config config) const
{
  rviz_common::Panel::save(config);

  // mapMakeChild ATTACHES the child to this config, so the writes the children
  // make through the returned handle actually reach the saved document.
  context_tab_->save(config.mapMakeChild("Context"));
  planning_tab_->save(config.mapMakeChild("Planning"));
  scene_objects_tab_->save(config.mapMakeChild("Scene Objects"));
  joints_tab_->save(config.mapMakeChild("Joints"));
  // No "Manipulation" child is written, for the reason given in load() above.
}

}  // namespace curobov2_ros_rviz

#include <pluginlib/class_list_macros.hpp>

// Pluginlib registration lives in the SAME translation unit as the class. The
// vtable is NOT anchored here: Q_OBJECT's `virtual metaObject()` is the key
// function, so the vtable is emitted in the moc TU built from this header by
// qt5_wrap_cpp in CMakeLists.txt. A missing moc TU surfaces exactly like a
// missing registration — "undefined symbol: vtable for
// curobov2_ros_rviz::CuroboPanel".
PLUGINLIB_EXPORT_CLASS(curobov2_ros_rviz::CuroboPanel, rviz_common::Panel)