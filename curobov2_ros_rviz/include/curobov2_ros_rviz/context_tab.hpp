#pragma once

#include <rclcpp/rclcpp.hpp>

#include <curobov2_ros_interfaces/srv/get_robot_strategies.hpp>
#include <curobov2_ros_interfaces/srv/get_workspace.hpp>
#include <curobov2_ros_interfaces/srv/set_planner.hpp>
#include <curobov2_ros_interfaces/srv/set_robot_strategy.hpp>
#include <curobov2_ros_interfaces/srv/set_workspace.hpp>
#include <rviz_common/config.hpp>

#include <QString>
#include <QVector>
#include <QWidget>

#include <functional>
#include <memory>
#include <set>
#include <string>

class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QPushButton;

namespace curobov2_ros_rviz
{

/// The Context tab: WHICH robot this panel is talking to, HOW trajectories are
/// produced for it, HOW it is driven, and how it is visualised — the curobo
/// analogue of MoveIt's "Context" tab.
///
/// MoveIt puts its robot description, planning-scene monitor and fixed table
/// here: the things that are true for the whole session and are not part of any
/// one request. The curobo equivalent is the same idea, expressed in terms this
/// server actually has, and it is the authoritative home of the two settings
/// that used to sit on the Planning tab:
///
///  * **Planning Group / planner node** — which unified_planner instance every
///    tab talks to. This is the single editable control for it in the whole
///    panel; CuroboPanel relays `plannerNodeChosen` to the other four tabs, and
///    the Planning tab only mirrors the value read-only. The list comes from the
///    live node graph (what `ros2 node list` reports), re-polled every 2 s so a
///    planner that starts later appears on its own.
///  * **Trajectory type** — Classic (MotionGen) or MPC (Real-time), i.e. how a
///    trajectory is produced at all. The dropdown *and* the SetPlanner call are
///    here; the Planning tab is told the outcome through `trajectoryTypeChanged`
///    and uses it only to decide whether "Compute + Execute" should open the
///    live-tracking MPC loop. The combo is reverted if the server refuses, so
///    what is displayed is always what the server is actually doing.
///  * **Control strategy** — emulator / joint_speed / joint_pose, i.e. HOW the
///    generated trajectory is handed to the robot. Genuinely per-server state
///    that no other tab could set. Switchable at runtime, which is what the
///    server's `set_robot_strategy` exists for.
///  * **Workspace** — the centre and size of the collision workspace, MoveIt's
///    third Context group. In curobo that is exactly the Mapper's TSDF/ESDF
///    grid extent: the box the cameras can observe, the box whose voxels become
///    collision geometry, and the box the `mapper_workspace` wireframe draws.
///    Backed by the server's `get_workspace` / `set_workspace`, so the drawn box
///    and the collision world move together. Unlike MoveIt — which applies the
///    bounds the instant a spin box changes — an edit here is committed by
///    "Apply workspace", because a resize rebuilds the Mapper and every solver
///    (blocking, ~20 s) and must not fire on every keystroke.
///
/// MoveIt's "Warehouse" group (planning-database host/port) has no curobo
/// equivalent, so it is absent rather than rendered permanently disabled.
///
/// Collision-sphere visualisation is deliberately NOT here: it is display
/// state, and it lives on the trajectory display (Show Collision / Show
/// Attached over the spheres topic), where a visualisation toggle belongs.
/// A per-server enable switch for it on this tab would change server state
/// from a panel whose other controls are all about which server to talk to
/// and how it plans.
///
/// The planner node and the trajectory type live in MoveIt's "Planning Library"
/// group, with MoveIt's own "Planning Group:" label on the first row — in MoveIt
/// that label is the first row of the *Planning* tab, but it is the same
/// question ("which planner am I talking to") and curobo has one answer to it.
///
/// Deliberately NOT here, despite looking like it belongs:
///
///  * `set_collision_cache`. The service still exists, but the v2 rewrite
///    deleted the triple OBB/mesh/blox cache it drove ("v2 uses a single
///    `collision_cache` parameter on MotionPlannerCfg"), so the knob is
///    vestigial. The one part of it that still matters — the voxel cache's
///    extent — is derived from the Workspace group above, which is the control
///    an operator actually reasons about. A dead triple in a panel is worse
///    than no control.
///  * `set_link_collision`. It needs link names, and no service enumerates
///    them — the panel would have to make the user type them blind.
///  * MoveIt's "Start State" / "Goal State" combos. curobo has no selectable
///    start state: the server always resolves it from the robot's live joint
///    state, so the selector would be a fiction. The goal lives in the Planning
///    and Joints tabs, where it is actually editable.
class ContextTab : public QWidget
{
  Q_OBJECT

public:
  explicit ContextTab(QWidget * parent = nullptr);
  ~ContextTab() override;

  /// The planner node every other tab has been pointed at. Authoritative: this
  /// is the one place it is chosen.
  QString plannerNode() const { return planner_node_; }

  /// The confirmed trajectory type (0 = Classic, 1 = MPC), i.e. what the server
  /// last acknowledged, not what the dropdown currently shows.
  int trajectoryType() const { return trajectory_type_; }

  /// Forwarded from the Planning tab's readiness probe. Gates the trajectory
  /// type combo and the Workspace group, which are both calls against that
  /// server; the planner-node combo stays live regardless, since repointing
  /// merely re-runs the probe.
  void setPlannerReady(bool ready);

  /// Called once rviz has handed us a display context, so the tab can take the
  /// shared rviz node instead of creating (and separately spinning) its own.
  void initialize(rclcpp::Node::SharedPtr node);

  void save(rviz_common::Config config) const;
  void load(const rviz_common::Config & config);

Q_SIGNALS:
  /// Emitted when the user picks a planner node. CuroboPanel relays it to the
  /// other four tabs so all five can never point at different servers.
  void plannerNodeChosen(const QString & planner_node);

  /// Emitted ONLY once the server has confirmed the switch, so a subscriber can
  /// treat it as fact rather than as a request. Never emitted on failure — the
  /// combo is put back first.
  void trajectoryTypeChanged(int combo_index);

private Q_SLOTS:
  void refreshPlannerNodeList();
  void onPlannerNodeChosen(const QString & text);
  void onTrajectoryTypeChosen(int index);
  void refreshStrategies();
  void applyStrategy();
  void applyTrajectoryType();
  void refreshWorkspace();
  void applyWorkspace();
  void markWorkspaceDirty();

private:
  void runOnGuiThread(std::function<void()> fn);
  void setStatus(const QString & text);
  void rebuildClients();
  /// Writes the six workspace spin boxes from a server reply, with signals
  /// blocked so populating them never marks the group dirty or sends anything.
  void showWorkspace(const QVector<double> & center, const QVector<double> & extent,
                     double voxel_size, bool perception_active);

  rclcpp::Node::SharedPtr node_;
  QString planner_node_ = QStringLiteral("curobo_server");
  bool initialized_ = false;
  /// Set by load() when the config carried a planner node, which makes that value
  /// authoritative over a launch-time parameter override. See initialize().
  bool planner_node_from_config_ = false;
  /// Forwarded from the Planning tab; gates the trajectory-type combo only.
  bool planner_ready_ = false;
  /// Set by load(): a saved trajectory type cannot be sent at load
  /// time (rviz loads before onInitialize, so there is no client yet), so it
  /// waits here until refreshStrategies() pushes it once the service exists.
  bool trajectory_type_pending_ = false;
  /// What the server last acknowledged, and what trajectoryType() reports.
  int trajectory_type_ = 0;

  rclcpp::Client<curobov2_ros_interfaces::srv::GetRobotStrategies>::SharedPtr
    get_strategies_client_;
  rclcpp::Client<curobov2_ros_interfaces::srv::SetRobotStrategy>::SharedPtr
    set_strategy_client_;
  rclcpp::Client<curobov2_ros_interfaces::srv::SetPlanner>::SharedPtr
    set_planner_client_;
  rclcpp::Client<curobov2_ros_interfaces::srv::GetWorkspace>::SharedPtr
    get_workspace_client_;
  rclcpp::Client<curobov2_ros_interfaces::srv::SetWorkspace>::SharedPtr
    set_workspace_client_;

  /// Last node set shown in the planner-node combo, so refreshPlannerNodeList()
  /// only rebuilds items when the graph actually changed.
  std::set<std::string> last_planner_nodes_;

  /// Last control-strategy list and selection shown, so refreshStrategies()
  /// (a 2 s poll) does not clear and repopulate the combo when nothing changed —
  /// a rebuild under the user's cursor is both wasteful and disruptive.
  QStringList last_strategies_;
  QString last_current_strategy_;

  /// True when the user has edited a workspace spin box but not yet pressed
  /// Apply. While set, refreshWorkspace() leaves the fields alone so a poll can
  /// never wipe out an edit in progress.
  bool workspace_dirty_ = false;
  /// Whether the last get_workspace reply said a Mapper exists. Reported in the
  /// group's caption; also what makes an Apply worth a ~20 s rebuild.
  bool workspace_perception_active_ = false;

  QComboBox * planner_node_combo_ = nullptr;
  QComboBox * trajectory_type_combo_ = nullptr;
  QLabel * current_strategy_label_ = nullptr;
  QComboBox * strategy_combo_ = nullptr;
  QPushButton * apply_button_ = nullptr;
  /// MoveIt's Workspace group: centre and size of the collision workspace, one
  /// spin box per axis, in x/y/z order.
  QDoubleSpinBox * workspace_center_[3] = {nullptr, nullptr, nullptr};
  QDoubleSpinBox * workspace_size_[3] = {nullptr, nullptr, nullptr};
  QPushButton * workspace_apply_button_ = nullptr;
  /// Read-only caption under the workspace fields: voxel size and whether the
  /// box has any collision geometry behind it.
  QLabel * workspace_info_label_ = nullptr;
  QLabel * status_ = nullptr;
};

}  // namespace curobov2_ros_rviz
