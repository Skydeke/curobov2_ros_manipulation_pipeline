#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <curobov2_ros_interfaces/srv/fk_batch.hpp>
#include <curobov2_ros_interfaces/srv/get_joint_info.hpp>
#include <curobov2_ros_interfaces/srv/trajectory_generation.hpp>
#include <curobov2_ros_interfaces/action/send_trajectory.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <control_msgs/action/gripper_command.hpp>
#include <control_msgs/action/parallel_gripper_command.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <rviz_common/config.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "curobov2_ros_rviz/node_spinner.hpp"

#include <QHash>
#include <QPair>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QWidget>

#include <functional>
#include <memory>
#include <mutex>

class QComboBox;
class QCheckBox;
class QLabel;
class QPushButton;
class QShowEvent;
class QTreeWidget;
class QTreeWidgetItem;

namespace curobov2_ros_rviz
{

/// Live joint-state monitor with joint-space planning — the curobo analogue of
/// the Joints tab of MoveIt's Motion Planning panel.
///
/// MoveIt's version shows the robot's joints as draggable progress bars (a
/// ProgressBarDelegate over a two-column table headed "Joint Name" / "Value")
/// that write straight into a goal state, and plans to them. That bar is
/// reproduced here rather than approximated with a spinbox; see
/// progress_bar_delegate.hpp for why a QSlider could not do it. curobo's server
/// has no "set joint state" service (it plans and executes, it does not write),
/// so this tab mirrors the useful half of that panel:
///
///  * one row per cspace joint, showing the live /joint_states position
///    (subscribed, name-matched — never by index, since the kortex sim
///    publishes the finger joint FIRST) plus a draggable target bar bounded by
///    the model's real position limits -- MoveIt's ProgressBarDelegate;
///  * "Set from current" / "Zero" to seed a target configuration;
///  * "Plan" and "Plan + send" driving a JOINT-SPACE goal
///    (`Goalset.target_joint_positions`), which is what makes this a planner
///    tab and not just a monitor;
///  * "Stop" while a goal executes;
///  * a "Check" button that runs the server's `fk_batch` on the TARGET
///    configuration, so a bad target is flagged (self/scene collision, or out
///    of joint limits) before you spend a plan on it. curobo's plan failures
///    are famously opaque — "Start or End state in collision" with no joint
///    attributed — so this is the difference between a usable joint tab and a
///    guessing one;
///  * `kinematics.lock_joints`, fetched with the cspace from the read-only
///    `get_joint_info` service, reported in the status line, plus a "Clear locks"
///    button that drops the override and rebuilds the model.
///  * a "Named pose:" combo of the robot descriptor's named joint
///    configurations (`named_joint_configs`), fetched from the planner's
///    `get_joint_info` service. This deliberately does NOT read MoveIt's SRDF:
///    the descriptor is the planner's own source of truth, and the task
///    constructor resolves the same names from it. Selecting an arm pose writes
///    its joints into the target column BY NAME (a pose names its joints, and
///    the cspace may not contain all of them); a joint the pose does NOT name
///    falls back to the live /joint_states value, so a subset state leaves the
///    rest of the arm where it is rather than holding a stale target. This keeps
///    the combo compatible with the bar/slider and with Plan / Plan + send on
///    the same request.
///  * one "open"/"close" entry per gripper controller from the descriptor's
///    `gripper_controllers` map. A gripper is NOT a planning DOF, and its
///    finger joints are usually locked out of the cspace, so selecting one of
///    these sends the descriptor's open/close position to the controller action
///    directly (MoveIt's simple-controller-manager semantics) instead of
///    editing the target. That is what keeps open/close working even when the
///    gripper is not in the planning space. A descriptor also carries
///    `open`/`close` in `named_joint_configs` (so the task constructor can
///    resolve them by name); those named configs are skipped here because the
///    synthesized entry is the one that can drive the controller, and showing
///    both would put two identical "open" rows in the combo.
///
/// A locked joint is pinned OUT of the cspace, so it has no row here and cannot
/// be planned — the panel says so explicitly rather than silently dropping it
/// from the target, which is the failure mode that produces a plan the operator
/// did not ask for.
class JointsTab : public QWidget
{
  Q_OBJECT

public:
  /// A named pose from the robot descriptor: the pose name, the group it
  /// belongs to (empty for `named_joint_configs`, the gripper name for a
  /// synthesized open/close entry), and the joint values it pins, each by joint
  /// NAME (the cspace index is not known until get_joint_info answers, and the
  /// pose's joints may be a superset of the cspace's, so names are the only
  /// stable key).
  struct NamedPose
  {
    QString name;
    QString group;
    QVector<QPair<QString, double>> joints;

    /// True for a synthesized gripper open/close entry: selecting it sends a
    /// controller action instead of editing the planning target.
    bool gripper = false;
    QString gripper_type;
    QString gripper_action;
    QString gripper_command_joint;
    QStringList gripper_joints;
    double gripper_max_effort = 0.0;
    double gripper_time_from_start = 2.0;

    bool operator==(const NamedPose & other) const
    {
      return name == other.name && group == other.group &&
             joints == other.joints && gripper == other.gripper &&
             gripper_type == other.gripper_type &&
             gripper_action == other.gripper_action &&
             gripper_command_joint == other.gripper_command_joint &&
             gripper_joints == other.gripper_joints &&
             gripper_max_effort == other.gripper_max_effort &&
             gripper_time_from_start == other.gripper_time_from_start;
    }
  };

public:
  explicit JointsTab(QWidget * parent = nullptr);
  ~JointsTab() override;

  /// Point every client at `planner_node` (e.g. "curobo_server"). Re-creates
  /// the clients and re-queries the cspace, so switching planners in the Plan
  /// tab also re-targets this one.
  void setPlannerNode(const QString & planner_node);
  QString plannerNode() const { return planner_node_; }

  /// Called once rviz has handed us a display context, so the tab can take the
  /// shared rviz node instead of creating (and separately spinning) its own.
  void initialize(rclcpp::Node::SharedPtr node);

  /// The cspace order the server resolves position lists in. Exposed so the
  /// panel can persist it and warn when a plan is attempted without it.
  QStringList cspaceOrder() const;

  void save(rviz_common::Config config) const;
  void load(const rviz_common::Config & config);

  /// The joints the server reports as locked out of the cspace, as read by the
  /// last `get_joint_info` response. The panel's Objects tab shows these next
  /// to the scene objects, because "the gripper is pinned" and "the gripper is
  /// closed" look identical in the joint tree and are not the same problem.
  QStringList lockedJoints() const { return locked_; }

protected:
  /// Populate the tab as soon as it is actually shown. The 2 s cspace poll is
  /// deliberately visibility-gated, and the one-shot fetch in initialize() may
  /// race the planner's get_joint_info service coming up. Without this, a panel
  /// whose Joints tab was hidden at startup stays with an empty (disabled)
  /// Named pose combo until the next poll tick.
  void showEvent(QShowEvent * event) override;

private Q_SLOTS:
  void onJointState(const sensor_msgs::msg::JointState::ConstSharedPtr msg);
  void refreshJointInfo();
  void setTargetsFromCurrent();
  void zeroTargets();
  void onNamedPoseChosen(int index);
  /// Publish the current target column as a one-waypoint JointTrajectory on
  /// <planner_node>/trajectory (the ghost topic the CuroboPlanningDisplay
  /// shows), so moving a joint bar / selecting a named pose displays the
  /// configuration without planning. No-op unless "Preview" is checked and the
  /// target actually changed.
  void publishPreview();
  void checkTarget();
  void planTarget();
  void sendTarget();
  void stopGoal();
  // By value, not const&: rclcpp's SendGoalOptions::goal_response_callback is
  // std::function<void(ClientGoalHandle<ActionT>::SharedPtr)>, so the handle is
  // handed over as a shared_ptr. A top-level const here would be ignored for
  // signature matching anyway, which is exactly how a const& definition in the
  // .cpp slipped past review and failed to match this declaration.
  void onGoalResponse(
    rclcpp_action::ClientGoalHandle<curobov2_ros_interfaces::action::SendTrajectory>::
      SharedPtr goal_handle);
  void onGoalResult(
    const rclcpp_action::ClientGoalHandle<curobov2_ros_interfaces::action::SendTrajectory>::
      WrappedResult & result);
  void setGoalActive(bool active);

private:
  /// Runs fn on the Qt GUI thread. ROS callbacks fire on the spinner thread
  /// and must never touch a widget directly.
  void runOnGuiThread(std::function<void()> fn);
  void setStatus(const QString & text);
  void setButtonsEnabled();
  QTreeWidgetItem * rowFor(const QString & joint) const;
  void rebuildRows(const QStringList & names, const QVector<double> & lower,
                   const QVector<double> & upper, const QVector<bool> & continuous);
  /// Replace the combo's poses, preserving the selected item when the new list
  /// still contains it. get_joint_info is polled, so this runs repeatedly with
  /// the same descriptor data and must not reset the user's selection.
  void setNamedPoses(const QVector<NamedPose> & poses);
  /// Write the joints of named_poses_[index] into the target column, by name, or
  /// send its gripper action when it is a synthesized open/close entry.
  void applyNamedPose(int index);
  /// Send an open/close command to the descriptor's gripper controller. The
  /// jaws are not planning DOFs, so this is the only way a locked gripper can
  /// still be opened or closed from this panel.
  void sendGripperCommand(const NamedPose & pose);

  rclcpp::Node::SharedPtr node_;
  QString planner_node_ = QStringLiteral("curobo_server");
  bool initialized_ = false;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

  rclcpp::Client<curobov2_ros_interfaces::srv::GetJointInfo>::SharedPtr joint_info_client_;
  rclcpp::Client<curobov2_ros_interfaces::srv::FkBatch>::SharedPtr fk_client_;
  rclcpp::Client<curobov2_ros_interfaces::srv::TrajectoryGeneration>::SharedPtr
    trajectory_client_;
  rclcpp_action::Client<curobov2_ros_interfaces::action::SendTrajectory>::SharedPtr
    action_client_;
  // The plan from the last successful generate_trajectory call, in the execute
  // action's replay form. Execution is MTC-literal: the server never re-plans
  // nor looks a plan up, so sendTarget() plans first and replays this verbatim.
  // Written and read on the GUI thread only (see sendTarget()).
  trajectory_msgs::msg::JointTrajectory last_trajectory_;
  // ClientGoalHandle has no nested GoalHandle typedef: it IS the goal handle
  // type, and its pointer alias is SharedPtr. (Client<ActionT>::GoalHandle does
  // exist, because ClientBase declares one; ClientGoalHandle<ActionT> does not.)
  rclcpp_action::ClientGoalHandle<curobov2_ros_interfaces::action::SendTrajectory>::SharedPtr
    goal_handle_;

  /// cspace order, exactly as the server resolves position lists in.
  QStringList cspace_;
  /// Joints pinned out of the cspace (`kinematics.lock_joints`) — reported,
  /// not commandable.
  QStringList locked_;
  /// Position limits from the last get_joint_info response, parallel to cspace_.
  /// Cached so the rows can be rebuilt (e.g. to restore a saved target) without
  /// losing the model's real bounds.
  QVector<double> limits_lower_;
  QVector<double> limits_upper_;
  /// Continuous-joint flags from get_joint_info, parallel to cspace_. The
  /// reported limits are a display window for these, never a physical stop.
  QVector<bool> continuous_;
  /// Target positions read from the config, waiting for the rows to exist. rviz
  /// calls load() before onInitialize(), and the rows only appear once
  /// get_joint_info has answered; rebuildRows() consumes and clears this.
  QVector<double> pending_targets_;
  /// Live positions by joint NAME (not index: the kortex sim publishes the
  /// finger joint first, so /joint_states order is NOT the cspace order).
  /// Written on the GUI thread only (the drain in onJointState).
  QHash<QString, double> current_;
  /// Coalescing buffer for /joint_states. The subscription fires on rviz's
  /// spinner thread far faster than the tree can be repainted, so readings are
  /// merged here (under joint_state_mutex_) and at most one GUI-thread drain is
  /// queued at a time; that drain swaps this into current_ and repaints once.
  /// Without it, a 100 Hz publisher queues a full-tree rewrite per message, all
  /// of which the GUI thread then has to run to keep up with.
  QHash<QString, double> pending_current_;
  bool joint_state_posted_ = false;
  std::mutex joint_state_mutex_;

  QTreeWidget * tree_ = nullptr;
  QLabel * status_ = nullptr;
  QPushButton * plan_button_ = nullptr;
  QPushButton * send_button_ = nullptr;
  QPushButton * stop_button_ = nullptr;
  QPushButton * check_button_ = nullptr;

  /// Named poses from the robot descriptor (empty until get_joint_info
  /// answers).
  QVector<NamedPose> named_poses_;
  QComboBox * named_pose_combo_ = nullptr;

  /// Gripper action clients, keyed by action name. Created lazily on the first
  /// open/close selection and kept for the planner's lifetime so the goal can
  /// outlive the click that sent it. The three maps match the three action
  /// shapes the descriptor schema allows.
  QHash<QString, rclcpp_action::Client<control_msgs::action::GripperCommand>::SharedPtr>
  gripper_clients_;
  QHash<QString,
    rclcpp_action::Client<control_msgs::action::ParallelGripperCommand>::SharedPtr>
  parallel_gripper_clients_;
  QHash<QString,
    rclcpp_action::Client<control_msgs::action::FollowJointTrajectory>::SharedPtr>
  gripper_trajectory_clients_;

  /// "Preview" gate + its topic. Checked => the target column is mirrored onto
  /// <planner_node>/trajectory (a one-waypoint JointTrajectory) whenever it
  /// changes, so the CuroboPlanningDisplay shows the configuration live while
  /// the bars are dragged. Unchecked => the tab only edits the request.
  QCheckBox * preview_check_ = nullptr;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr preview_pub_;
  QVector<double> last_preview_target_;

  bool goal_active_ = false;
  bool cspace_known_ = false;
};

}  // namespace curobov2_ros_rviz
