#pragma once
#include <functional>
#include <iostream>

// ROS2
#include "rclcpp_action/rclcpp_action.hpp"
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

// Projet
#include "curobov2_ros_interfaces/srv/get_joint_info.hpp"
#include "curobov2_ros_interfaces/srv/set_planner.hpp"
#include "curobov2_ros_interfaces/srv/trajectory_generation.hpp"
#include "curobov2_ros_rviz/node_spinner.hpp"
#include "curobov2_ros_rviz/curobo_planning_display.hpp"
#include "curobov2_ros_interfaces/action/send_trajectory.hpp"
#include "curobov2_ros_interfaces/msg/goalset.hpp"

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

namespace curobov2_ros_rviz {
/**
 * \brief The Planning tab of CuroboPanel: goal state, planning request, and
 *        the Compute / Execute / Cancel row.
 *
 * This is the panel that used to be registered as the standalone rviz panel
 * "curobov2_ros_rviz/RvizArgsPanel" (class RvizArgsPanel). It is no
 * longer registered: CuroboPanel is the only way to reach it, and the class
 * was renamed to PlanningTab so the five tabs of that panel are one family
 * (ContextTab, PlanningTab, JointsTab, SceneObjectsTab).
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

  /// Rebind every planner client to `planner_node`.
  void setPlannerNode(const QString &planner_node);

  /// Which trajectory type the Context tab has selected. 0 = Classic
  /// (MotionGen), 1 = MPC (real-time). ContextTab owns the dropdown and the
  /// SetPlanner call; this is only the read-back the execute path needs.
  void setTrajectoryType(int combo_index);
  int trajectoryType() const { return static_cast<int>(trajectory_type_); }

  /// Supply the container panel's display context when embedded.
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
          curobov2_ros_interfaces::action::SendTrajectory>::WrappedResult
          &result);
  void goal_response_callback(
      std::shared_ptr<rclcpp_action::ClientGoalHandle<
          curobov2_ros_interfaces::action::SendTrajectory>>
          goal_handle);
  // void
  // goal_response_callback(std::shared_future<rclcpp_action::ClientGoalHandle<actionfaces::action::Fibonacci>::SharedPtr>
  // future)

  // Marker control slots
  void updateMarkerPoseDisplay();
  void refreshPlanningDisplays();
  void applyPoseFromSpinboxes();
  void onTargetToolChanged(int index);

  // The goal orientation is a QUATERNION, not RPY -- see the block comment above
  static void canonicalize(geometry_msgs::msg::Quaternion & q);
  /// Scale to unit length, returning false for the zero quaternion. Unlike the
  /// RPY path it replaced, four free spin boxes CAN hold a non-rotation, so this
  /// is checked rather than assumed.
  static bool normalize(geometry_msgs::msg::Quaternion & q);
  /// Read the four orientation boxes into `q`, unnormalised.
  void readQuaternionFromSpinboxes(geometry_msgs::msg::Quaternion & q);
  /// Write `q` into the four orientation boxes, with their signals blocked.
  void writeQuaternionToSpinboxes(const geometry_msgs::msg::Quaternion & q);

private:
  // Set by initializeEmbedded() when this tab is hosted by CuroboPanel; null
  // in the (no longer reachable) standalone case, where the base class' own
  // display context would be used.
  rviz_common::DisplayContext *embedded_display_context_ = nullptr;

  // Runs fn on the Qt GUI thread. ROS callbacks fire on the background spin
  void runOnGuiThread(std::function<void()> fn);

  // Single source of truth for "is the planner actually responding" (not just
  void setPlannerReady(bool ready);
  void pollPlannerReady();

  // Combines planner_ready_ with "is a goal currently executing" to drive
  // clearGoal/generateTrajectory/sendTrajectory/generateAndSend/stopRobot
  // enabled state.
  void updateActionButtons();

  // Writes the bottom status line. MUST be called on the GUI thread -- the
  void setStatus(const QString &text, const QString &detail = QString());

  // Shared by on_generateTrajectory_clicked and the classic-mode branch of
  void generateTrajectoryAsync(std::function<void(bool)> on_done);

  // Build the goalset list from the CuroboPlanningDisplays currently in RViz. A
  std::vector<curobov2_ros_interfaces::msg::Goalset> buildGoalsets();

  // (Re)creates every planner-facing client/publisher against the given planner
  // node name. Called from the constructor and from setPlannerNode().
  void createPlannerClients(const std::string &planner_node);

  // --- MPC live-tracking (owned by this tab; the CuroboPlanningDisplay is generic) ---
  void startMpc();
  void stopMpc();
  void streamMpcGoal();

  void sendMpcGoal();

  std::unique_ptr<Ui::PlanningTabForm> ui_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::AsyncParametersClient::SharedPtr param_client_;
  bool planner_ready_;
  bool planner_poll_in_flight_;
  // Readiness-probe timestamp and sequence (watchdog).
  std::chrono::steady_clock::time_point planner_poll_sent_at_;
  uint64_t planner_poll_seq_;
  bool goal_active_;
  rclcpp_action::Client<curobov2_ros_interfaces::action::SendTrajectory>::
      SharedPtr action_ptr_;
  rclcpp::Client<curobov2_ros_interfaces::srv::TrajectoryGeneration>::
      SharedPtr trajectory_generation_client_;
  rclcpp_action::Client<curobov2_ros_interfaces::action::SendTrajectory>::
      GoalHandle::SharedPtr goal_handle_;
  float time_dilation_factor_;
  // The CuroboPlanningDisplay owning the draggable 6-DOF target (self-contained;
  CuroboPlanningDisplay *target_display_;
  std::vector<CuroboPlanningDisplay *> target_displays_;

  // Tool frames reported by the planner (get_joint_info). Pushed into the
  // primary CuroboPlanningDisplay so one marker exists per arm.
  std::vector<std::string> tool_frames_;
  // Last list pushed into the display, so the 500 ms poll never overwrites a
  // user-edited Tool Frames property.
  std::vector<std::string> pushed_tool_frames_;
  rclcpp::Client<curobov2_ros_interfaces::srv::GetJointInfo>::SharedPtr
      joint_info_client_;
  bool tool_frames_in_flight_;

  // Active tool selector in the Query group.
  QComboBox *target_tool_combo_;
  void refreshToolFrames();
  void applyToolFramesToDisplay();
  void refreshTargetToolCombo();

  // Collects every CuroboPlanningDisplay under `group` (in display-tree order) into
  void collectPlanningDisplaysInGroup(rviz_common::DisplayGroup *group,
                                    std::vector<CuroboPlanningDisplay *> &out);
  bool user_editing_pose_; // Flag to prevent auto-update while user is editing

  // Last displayed pose to avoid unnecessary updates. The orientation is kept as a
  double last_displayed_x_;
  double last_displayed_y_;
  double last_displayed_z_;
  geometry_msgs::msg::Quaternion last_displayed_q_;

  // Planner node this tab's clients currently point at. Owned by the Context
  // tab; see setPlannerNode().
  std::string planner_node_;

  // MPC live-tracking members.
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mpc_goal_pub_;
  bool mpc_active_;
  bool mpc_starting_;
  geometry_msgs::msg::Pose last_published_goal_;
  QTimer *mpc_goal_timer_;
  // Readiness poll. Runs at 250 ms until the planner first answers, then backs
  QTimer *readiness_timer_;

  // Live tool pose (base frame) used to one-shot seed a pristine CuroboPlanningDisplay
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      tool_pose_sub_;
  geometry_msgs::msg::PoseStamped::SharedPtr tool_pose_;
  mutable std::mutex tool_pose_mutex_;
  // Display already seeded (latch): reseeds only ever happen for a display
  // that has not been seeded yet AND still shows the untouched default pose.
  CuroboPlanningDisplay *seeded_display_ = nullptr;
  // Own TF buffer for the one-time base->fixed seed lookup (frame names come
  // from the tool message + the RViz fixed frame: zero frame configuration).
  std::shared_ptr<tf2_ros::Buffer> seed_tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> seed_tf_listener_;
  /// Pose every outgoing goal is expressed in (RViz fixed frame); also the
  /// seed target frame.
  std::string goalFrame() const;
  /// One-shot seed: move a pristine primary display to the live tool pose.
  void maybeSeedTargetFromTool();

  // Trajectory type, pushed in by the Context tab (0 = Classic, 1 = MPC).
  // It decides whether on_generateAndSend_clicked() starts the live-tracking
  // MPC stream; the dropdown and the user's choice both live on Context.
  uint8_t trajectory_type_;

  // The plan from the last successful generate_trajectory call, in the
  trajectory_msgs::msg::JointTrajectory last_trajectory_;

  // SetPlanner client, used by startMpc() ONLY, to re-assert MPC on the server
  rclcpp::Client<curobov2_ros_interfaces::srv::SetPlanner>::SharedPtr
      set_planner_client_;

  // Declared LAST so it is destroyed FIRST (members are torn down in reverse
  std::unique_ptr<NodeSpinner> spinner_;
};
} // namespace curobov2_ros_rviz
