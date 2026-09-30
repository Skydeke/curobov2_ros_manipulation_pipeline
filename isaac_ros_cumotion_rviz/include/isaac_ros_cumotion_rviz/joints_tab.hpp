#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <isaac_ros_cumotion_interfaces/srv/fk_batch.hpp>
#include <isaac_ros_cumotion_interfaces/srv/get_joint_info.hpp>
#include <isaac_ros_cumotion_interfaces/srv/set_joint_locks.hpp>
#include <isaac_ros_cumotion_interfaces/srv/trajectory_generation.hpp>
#include <isaac_ros_cumotion_interfaces/action/send_trajectory.hpp>
#include <rviz_common/config.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "isaac_ros_cumotion_rviz/node_spinner.hpp"

#include <QHash>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QWidget>

#include <functional>
#include <memory>

class QLabel;
class QPushButton;
class QTreeWidget;
class QTreeWidgetItem;

namespace isaac_ros_cumotion_rviz
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
///
/// A locked joint is pinned OUT of the cspace, so it has no row here and cannot
/// be commanded — the panel says so explicitly rather than silently dropping
/// it from the target, which is the failure mode that produces a plan the
/// operator did not ask for.
class JointsTab : public QWidget
{
  Q_OBJECT

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

private Q_SLOTS:
  void onJointState(const sensor_msgs::msg::JointState::ConstSharedPtr msg);
  void refreshJointInfo();
  void setTargetsFromCurrent();
  void zeroTargets();
  void checkTarget();
  void clearLocks();
  void planTarget();
  void sendTarget();
  void stopGoal();
  // By value, not const&: rclcpp's SendGoalOptions::goal_response_callback is
  // std::function<void(ClientGoalHandle<ActionT>::SharedPtr)>, so the handle is
  // handed over as a shared_ptr. A top-level const here would be ignored for
  // signature matching anyway, which is exactly how a const& definition in the
  // .cpp slipped past review and failed to match this declaration.
  void onGoalResponse(
    rclcpp_action::ClientGoalHandle<isaac_ros_cumotion_interfaces::action::SendTrajectory>::
      SharedPtr goal_handle);
  void onGoalResult(
    const rclcpp_action::ClientGoalHandle<isaac_ros_cumotion_interfaces::action::SendTrajectory>::
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
                   const QVector<double> & upper);

  rclcpp::Node::SharedPtr node_;
  QString planner_node_ = QStringLiteral("curobo_server");
  bool initialized_ = false;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::GetJointInfo>::SharedPtr joint_info_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::FkBatch>::SharedPtr fk_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::SetJointLocks>::SharedPtr joint_locks_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration>::SharedPtr
    trajectory_client_;
  rclcpp_action::Client<isaac_ros_cumotion_interfaces::action::SendTrajectory>::SharedPtr
    action_client_;
  // ClientGoalHandle has no nested GoalHandle typedef: it IS the goal handle
  // type, and its pointer alias is SharedPtr. (Client<ActionT>::GoalHandle does
  // exist, because ClientBase declares one; ClientGoalHandle<ActionT> does not.)
  rclcpp_action::ClientGoalHandle<isaac_ros_cumotion_interfaces::action::SendTrajectory>::SharedPtr
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
  /// Target positions read from the config, waiting for the rows to exist. rviz
  /// calls load() before onInitialize(), and the rows only appear once
  /// get_joint_info has answered; rebuildRows() consumes and clears this.
  QVector<double> pending_targets_;
  /// Live positions by joint NAME (not index: the kortex sim publishes the
  /// finger joint first, so /joint_states order is NOT the cspace order).
  QHash<QString, double> current_;

  QTreeWidget * tree_ = nullptr;
  QLabel * status_ = nullptr;
  QPushButton * plan_button_ = nullptr;
  QPushButton * send_button_ = nullptr;
  QPushButton * stop_button_ = nullptr;
  QPushButton * check_button_ = nullptr;
  QPushButton * clear_locks_button_ = nullptr;

  bool goal_active_ = false;
  bool cspace_known_ = false;
};

}  // namespace isaac_ros_cumotion_rviz
