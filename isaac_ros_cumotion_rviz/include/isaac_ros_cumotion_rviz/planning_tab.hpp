#pragma once
#include <functional>
#include <iostream>

// ROS2
#include "rclcpp_action/rclcpp_action.hpp"
#include <geometry_msgs/msg/pose.hpp>
#include <rclcpp/rclcpp.hpp>

// Projet
#include "isaac_ros_cumotion_interfaces/srv/set_planner.hpp"
#include "isaac_ros_cumotion_interfaces/srv/trajectory_generation.hpp"
#include "isaac_ros_cumotion_rviz/node_spinner.hpp"
#include "isaac_ros_cumotion_rviz/target_display.hpp"
// Return type of buildGoalsets() below, so it belongs here, not just in the
// .cpp. It happens to be reachable transitively today (send_trajectory.hpp ->
// TrajectoryGoal has a Goalset[] field), which is exactly the kind of accident
// that breaks the next time the goal schema changes.
#include "isaac_ros_cumotion_interfaces/action/send_trajectory.hpp"
#include "isaac_ros_cumotion_interfaces/msg/goalset.hpp"

// RVIZ2
#include <rviz_common/display.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/display_group.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/visualization_manager.hpp>
// Qt
#include <QtWidgets>
// STL
#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>
/**
 *  Include header generated from ui file
 *  Note that you will need to use add_library function first
 *  in order to generate the header file from ui.
 */
#include <ui_planning_tab.h>

// Only ever needed as a pointer type, for PlanningTab's embedding API.
namespace rviz_common {
class DisplayContext;
}

namespace isaac_ros_cumotion_rviz {
/**
 * \brief The Planning tab of CuroboPanel: goal state, planning request, and
 *        the Compute / Execute / Cancel row.
 *
 * This is the panel that used to be registered as the standalone rviz panel
 * "isaac_ros_cumotion_rviz/RvizArgsPanel" (class RvizArgsPanel). It is no
 * longer registered: CuroboPanel is the only way to reach it, and the class
 * was renamed to PlanningTab so the five tabs of that panel are one family
 * (ContextTab, PlanningTab, JointsTab, ManipulationTab, SceneObjectsTab).
 * It still derives from rviz_common::Panel, because that is where the
 * DisplayContext and the Config-based load/save it needs come from, and
 * because CuroboPanel still has to hand it one explicitly -- see
 * initializeEmbedded().
 *
 * The layout of resource/planning_tab.ui deliberately mirrors MoveIt's
 * "Planning" tab: a "Goal State" group holding the query goal, a "Planning
 * Request" group holding the request parameters, and a single row of
 * Clear / Compute / Execute / Cancel at the bottom of the tab. What the
 * groups contain is curobo's, not MoveIt's -- this server has no octree, no
 * planning-time or scaling parameters, and no separate start state.
 *
 * Two settings that MoveIt shows in Planning live on the Context tab instead,
 * because they describe the server rather than one request:
 *
 *  * **Planner node** -- the single editable control for it is ContextTab's
 *    dropdown. This class only ever receives it, through setPlannerNode().
 *  * **Trajectory type** (Classic / MPC) -- likewise owned by ContextTab,
 *    which also makes the SetPlanner call. This class only needs to know
 *    which one is active, to decide whether "Compute + Execute" should start
 *    the live-tracking MPC stream, and receives that through
 *    setTrajectoryType().
 */
class PlanningTab : public rviz_common::Panel {
  Q_OBJECT
public:
  explicit PlanningTab(QWidget *parent = nullptr);
  ~PlanningTab();

  /// Load and save configuration data
  virtual void load(const rviz_common::Config &config) override;
  virtual void save(rviz_common::Config config) const override;

  /// Event filter to detect when user starts editing pose spinboxes
  bool eventFilter(QObject *obj, QEvent *event) override;

  /// The planner node this tab's clients are currently bound to.
  QString plannerNode() const { return QString::fromStdString(planner_node_); }

  /// Bind the tab to `planner_node`: cancel any in-flight goal, stop MPC,
  /// rebuild every client, re-probe readiness, and update the planner-node
  /// echo shown in the Planning Request group. Called by CuroboPanel when
  /// the Context tab's dropdown changes, so all five tabs can never end up
  /// pointed at different servers.
  void setPlannerNode(const QString &planner_node);

  /// Which trajectory type the Context tab has selected. 0 = Classic
  /// (MotionGen), 1 = MPC (real-time). ContextTab owns the dropdown and the
  /// SetPlanner call; this is only the read-back the execute path needs.
  void setTrajectoryType(int combo_index);
  int trajectoryType() const { return static_cast<int>(trajectory_type_); }

  /// Bootstrap for EMBEDDING, used by CuroboPanel to reuse this tab. rviz
  /// supplies a standalone panel with its DisplayContext before calling
  /// onInitialize(), through a context-setting entry point that a panel
  /// constructed by another panel is never given -- so the embedded case is
  /// wired from outside instead: `context` is the container panel's own
  /// display context, and refreshTargetDisplays() prefers it over the base
  /// class's.
  void initializeEmbedded(rviz_common::DisplayContext *context);

  /// The DisplayContext to use: the embedded one when CuroboPanel supplied it,
  /// otherwise the base class' own. Never dereferenced unchecked.
  rviz_common::DisplayContext *displayContext() const {
    return embedded_display_context_ != nullptr ? embedded_display_context_
                                                : getDisplayContext();
  }

Q_SIGNALS:
  /// Emitted from setPlannerReady(), i.e. only when the readiness probe
  /// actually flips. CuroboPanel forwards it to the Context tab, which owns
  /// the trajectory-type dropdown that this used to gate itself.
  void plannerReadyChanged(bool ready);

private Q_SLOTS:
  void updateTimeDilationFactor(double value);
  void on_clearGoal_clicked();
  void on_sendTrajectory_clicked();
  void on_generateTrajectory_clicked();
  void on_generateAndSend_clicked();
  void on_stopRobot_clicked();
  void result_callback(
      const rclcpp_action::ClientGoalHandle<
          isaac_ros_cumotion_interfaces::action::SendTrajectory>::WrappedResult
          &result);
  void goal_response_callback(
      std::shared_ptr<rclcpp_action::ClientGoalHandle<
          isaac_ros_cumotion_interfaces::action::SendTrajectory>>
          goal_handle);
  // void
  // goal_response_callback(std::shared_future<rclcpp_action::ClientGoalHandle<actionfaces::action::Fibonacci>::SharedPtr>
  // future)

  // Marker control slots
  void updateMarkerPoseDisplay();
  void refreshTargetDisplays();
  void applyPoseFromSpinboxes();

  // Helper methods for quaternion <-> Euler conversion
  void quaternionToEuler(const geometry_msgs::msg::Quaternion &q, double &roll,
                         double &pitch, double &yaw);
  void eulerToQuaternion(double roll, double pitch, double yaw,
                         geometry_msgs::msg::Quaternion &q);

private:
  // Set by initializeEmbedded() when this tab is hosted by CuroboPanel; null
  // in the (no longer reachable) standalone case, where the base class' own
  // display context would be used.
  rviz_common::DisplayContext *embedded_display_context_ = nullptr;

  // Runs fn on the Qt GUI thread. ROS callbacks fire on the background spin
  // thread (see spinner_ below) and must never touch QWidgets directly. Uses
  // the context-object overload of invokeMethod so Qt drops the call if `this`
  // is destroyed before the event loop gets to it.
  void runOnGuiThread(std::function<void()> fn);

  // Single source of truth for "is the planner actually responding" (not just
  // discoverable -- see node_is_available poll in the
  // constructor/pollPlannerReady). Gates every planner-facing widget so a click
  // can never queue a request against a not-yet-responsive planner (e.g. during
  // its ~90s GPU warmup, or after it respawns following a robot reboot).
  void setPlannerReady(bool ready);
  void pollPlannerReady();

  // Combines planner_ready_ with "is a goal currently executing" to drive
  // clearGoal/generateTrajectory/sendTrajectory/generateAndSend/stopRobot
  // enabled state.
  void updateActionButtons();

  // Writes the bottom status line. MUST be called on the GUI thread -- the
  // result callback that feeds it fires on the background spin thread.
  //
  // `detail` is appended after the outcome when non-empty; it is where the
  // server's PlanningStats scalars land (see result_callback).
  void setStatus(const QString &text, const QString &detail = QString());

  // Shared by on_generateTrajectory_clicked and the classic-mode branch of
  // on_generateAndSend_clicked (which used to chain onto a blocking generate
  // call via a fragile QTimer::singleShot(500, ...) guess -- now chains on the
  // real async completion instead).
  void generateTrajectoryAsync(std::function<void(bool)> on_done);

  // Build the goalset list from the TargetDisplays currently in RViz. A
  // single Goalset (single pose) is emitted from the PRIMARY (first)
  // display only, matching what the server's Classic/JointSpace path
  // consumes. Returns an empty vector when no TargetDisplay is available.
  std::vector<isaac_ros_cumotion_interfaces::msg::Goalset> buildGoalsets();

  // (Re)creates every planner-facing client/publisher against the given planner
  // node name. Called from the constructor and from setPlannerNode().
  void createPlannerClients(const std::string &planner_node);

  // --- MPC live-tracking (owned by this tab; the TargetDisplay is generic) ---
  // Switch the planner to MPC, send an execute_trajectory goal toward the
  // current target and stream the target pose to /<planner>/mpc_goal at 10 Hz
  // while the gizmo is dragged.
  void startMpc();
  void stopMpc();
  void streamMpcGoal();

  void sendMpcGoal();

  std::unique_ptr<Ui::PlanningTabForm> ui_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::AsyncParametersClient::SharedPtr param_client_;
  bool planner_ready_;
  bool planner_poll_in_flight_;
  // Timestamp the current readiness probe was sent at + a sequence counter,
  // for the watchdog in pollPlannerReady(): an unanswered async
  // get_parameters must not freeze planner_ready_ (and the Execute buttons
  // with it) forever.
  std::chrono::steady_clock::time_point planner_poll_sent_at_;
  uint64_t planner_poll_seq_;
  bool goal_active_;
  rclcpp_action::Client<isaac_ros_cumotion_interfaces::action::SendTrajectory>::
      SharedPtr action_ptr_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration>::
      SharedPtr trajectory_generation_client_;
  rclcpp_action::Client<isaac_ros_cumotion_interfaces::action::SendTrajectory>::
      GoalHandle::SharedPtr goal_handle_;
  float time_dilation_factor_;
  // The TargetDisplay owning the draggable 6-DOF target (self-contained;
  // polling timer refreshes it lazily). Not owned by this tab. For multiple
  // TargetDisplays, `target_display_` is the FIRST one (the one Classic/MPC
  // and the pose spin boxes act on); `target_displays_` carries ALL of them
  // in display-tree order.
  TargetDisplay *target_display_;
  std::vector<TargetDisplay *> target_displays_;

  // Collects every TargetDisplay under `group` (in display-tree order) into
  // `out`. Used by refreshTargetDisplays(); the shipped rviz configs place the
  // display inside a Group (e.g. "Curobo Planning"), so the scan must descend
  // into subgroups.
  void collectTargetDisplaysInGroup(rviz_common::DisplayGroup *group,
                                    std::vector<TargetDisplay *> &out);
  bool user_editing_pose_; // Flag to prevent auto-update while user is editing

  // Last displayed pose to avoid unnecessary updates
  double last_displayed_x_;
  double last_displayed_y_;
  double last_displayed_z_;
  double last_displayed_roll_;
  double last_displayed_pitch_;
  double last_displayed_yaw_;

  // Planner node this tab's clients currently point at. Owned by the Context
  // tab; see setPlannerNode().
  std::string planner_node_;

  // MPC live-tracking members.
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr mpc_goal_pub_;
  bool mpc_active_;
  bool mpc_starting_;
  geometry_msgs::msg::Pose last_published_goal_;
  QTimer *mpc_goal_timer_;

  // Trajectory type, pushed in by the Context tab (0 = Classic, 1 = MPC).
  // It decides whether on_generateAndSend_clicked() starts the live-tracking
  // MPC stream; the dropdown and the user's choice both live on Context.
  uint8_t trajectory_type_;

  // SetPlanner client, used by startMpc() ONLY, to re-assert MPC on the server
  // before opening the live-tracking loop. It is not a user-facing control --
  // the Context tab owns that -- but the reactive path genuinely needs the
  // call, and borrowing Context's client would mean reaching across tabs.
  // NB: this type needs set_planner.hpp in THIS header, not just the .cpp --
  // the member is declared here, so moving the include down to the .cpp breaks
  // the build (and moc, which parses this header on its own).
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::SetPlanner>::SharedPtr
      set_planner_client_;

  // Declared LAST so it is destroyed FIRST (members are torn down in reverse
  // declaration order): stops and joins the background spin thread before any
  // client/publisher/node it might still be delivering callbacks against is
  // destroyed.
  std::unique_ptr<NodeSpinner> spinner_;
};
} // namespace isaac_ros_cumotion_rviz
