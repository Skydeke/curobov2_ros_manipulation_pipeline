#include <curobov2_ros_rviz/joints_tab.hpp>

#include <curobov2_ros_rviz/progress_bar_delegate.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPointF>
#include <QPushButton>
#include <QStringList>
#include <QTimer>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>
#include <QXmlStreamReader>

namespace curobov2_ros_rviz
{

namespace
{

enum Column
{
  COL_JOINT,
  COL_CURRENT,
  COL_TARGET,
  COL_COUNT
};

/// A planned TrajectoryResult -> the execute action's replay form.
///
/// Execution is MTC-literal: the server drives exactly the plan it is handed
/// and never re-plans nor looks one up, so this tab re-packs the result it just
/// generated into the action's replay trajectory. The wire form of a result is
/// sensor_msgs/JointState[] + a scalar dt; the replay field wants
/// JointTrajectoryPoints with time_from_start.
trajectory_msgs::msg::JointTrajectory toReplayTrajectory(
  const curobov2_ros_interfaces::msg::TrajectoryResult & result)
{
  trajectory_msgs::msg::JointTrajectory traj;
  if (result.trajectory.empty()) {
    return traj;
  }
  traj.joint_names = result.trajectory.front().name;
  const double dt = result.dt;
  for (std::size_t i = 0; i < result.trajectory.size(); ++i) {
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions = result.trajectory[i].position;
    pt.velocities = result.trajectory[i].velocity;
    const double sec = dt * static_cast<double>(i);
    pt.time_from_start.sec = static_cast<int32_t>(sec);
    pt.time_from_start.nanosec = static_cast<uint32_t>((sec - std::floor(sec)) * 1e9);
    traj.points.push_back(pt);
  }
  return traj;
}

/// Bounds of a row, or a null QPointF for a joint the model reports no limit for.
/// Absent bounds are what make such a row plain text and not draggable, rather
/// than a bar spanning a made-up range.
QPointF boundsOf(const QTreeWidgetItem * item)
{
  return item->data(COL_TARGET, ProgressBarDelegate::VariableBoundsRole).toPointF();
}

/// Whether this row has bounds at all, and so is drawn and dragged as a bar.
///
/// The test is on the QVariant, NOT on the QPointF: QPointF has no isValid()
/// (only QPoint, QSize, QSizeF, QRect and QRectF do), so a y() > x() test would
/// be the only geometric option -- and the wrong one to use, because this must
/// agree EXACTLY with ProgressBarDelegate::paint's own test. The delegate decides
/// "bar or plain text" from `index.data(role).isValid()`, so if this tab decided
/// it from the geometry instead, the two could disagree: the tab would clamp a
/// cell the delegate leaves as plain text, and a QPointF(0, 0) written for an
/// unbounded joint is indistinguishable from a real zero-width bound.
bool hasBounds(const QTreeWidgetItem * item)
{
  return item->data(COL_TARGET, ProgressBarDelegate::VariableBoundsRole).isValid();
}

/// The target value of a row. This lives in the ITEM rather than in a child
/// spinbox because MoveIt's joint table has no spinbox: the cell is painted as a
/// progress bar by ProgressBarDelegate and edited by dragging it. EditRole, not
/// DisplayRole, because that is the role a committed drag writes to -- see the
/// comment in ProgressBarDelegate::paint.
double targetOf(const QTreeWidgetItem * item)
{
  return item->data(COL_TARGET, Qt::EditRole).toDouble();
}

/// Writes a row's target and repaints it. EVERY write in this file goes through
/// here, so the clamp to the joint's own limits cannot be bypassed by one path.
///
/// A CONTINUOUS joint (reported range is a full turn or wider) is periodic, so
/// a value written as the "other" turn — e.g. the SRDF's
/// laundry_check_gripper j5 = 6.3641, just past the 6.28 top — is not out of
/// range, it is the SAME pose as 6.3641 - 2*pi = 0.0809. Clamping it to 6.28
/// (what a plain std::clamp does) silently plans a different configuration;
/// wrapping it to the in-range full-turn equivalent keeps the pose exact.
void setTarget(QTreeWidgetItem * item, double value)
{
  if (hasBounds(item)) {
    const QPointF limits = boundsOf(item);
    const double lo = limits.x();
    const double hi = limits.y();
    constexpr double kTwoPi = 6.283185307179586;
    if (hi - lo >= kTwoPi - 1e-6) {
      // Continuous: pick the in-range 2*pi-equivalent nearest the raw value.
      double best = value;
      double best_dist = std::numeric_limits<double>::infinity();
      for (int k = -2; k <= 2; ++k) {
        const double cand = value + static_cast<double>(k) * kTwoPi;
        if (cand < lo - 1e-9 || cand > hi + 1e-9) {
          continue;
        }
        const double dist = std::fabs(cand - value);
        if (dist < best_dist) {
          best_dist = dist;
          best = cand;
        }
      }
      value = best;
    } else {
      value = std::clamp(value, lo, hi);
    }
  }
  // ARGUMENT ORDER: QTreeWidgetItem::data is (column, role) but
  // QTreeWidgetItem::setData is (column, role, value) -- the value goes LAST,
  // not in the middle. Qt 5 declares only that one form, so the natural-looking
  // setData(column, value, role) does not merely fail to compile: `double` is
  // implicitly convertible to `int`, so it binds as setData(column, role=(int)
  // value, value=QVariant(role)) and writes the role number into the cell as its
  // data, under a nonsense role. Nothing complains and the bar silently never
  // moves. data() and setData() taking their arguments in different orders is the
  // whole trap.
  item->setData(COL_TARGET, Qt::EditRole, value);
  // DisplayRole carries the same number as text, so a row with no bar (an
  // unbounded joint) still shows its value instead of going blank.
  item->setText(COL_TARGET, QString::number(value, 'f', 4));
}

}  // namespace

namespace
{

/// Reads the target column into a vector in TREE (== cspace) order.
QVector<double> targetVector(QTreeWidget * tree)
{
  QVector<double> out;
  for (int i = 0; i < tree->topLevelItemCount(); ++i) {
    out << targetOf(tree->topLevelItem(i));
  }
  return out;
}

}  // namespace


void JointsTab::publishPreview()
{
  if (preview_pub_ == nullptr || preview_check_ == nullptr ||
    !preview_check_->isChecked() || !cspace_known_ ||
    tree_->topLevelItemCount() == 0)
  {
    return;
  }
  const QVector<double> target = targetVector(tree_);
  if (target == last_preview_target_) {
    return;
  }
  trajectory_msgs::msg::JointTrajectory msg;
  for (int i = 0; i < tree_->topLevelItemCount(); ++i) {
    msg.joint_names.push_back(tree_->topLevelItem(i)->text(COL_JOINT).toStdString());
  }
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions.assign(target.begin(), target.end());
  msg.points.push_back(point);
  preview_pub_->publish(msg);
  last_preview_target_ = target;
}

void JointsTab::checkTarget()
{
  if (fk_client_ == nullptr || !fk_client_->service_is_ready()) {
    setStatus(tr("fk_batch service not available"));
    return;
  }
  auto request = std::make_shared<curobov2_ros_interfaces::srv::FkBatch::Request>();
  auto & config = request->joint_states.emplace_back();
  // FK resolves its input BY NAME, so the request is correct regardless of the
  // cspace order — which is exactly what makes this a usable pre-check.
  for (int i = 0; i < tree_->topLevelItemCount(); ++i) {
    auto * item = tree_->topLevelItem(i);
    config.name.push_back(item->text(COL_JOINT).toStdString());
    config.position.push_back(targetOf(item));
  }

  setStatus(tr("checking the target configuration..."));
  fk_client_->async_send_request(
    request,
    [this](rclcpp::Client<curobov2_ros_interfaces::srv::FkBatch>::SharedFuture future)
    {
      bool valid = false;
      QString error;
      try {
        auto result = future.get();
        valid = !result->poses_valid.empty() && result->poses_valid[0].data;
        error = QString::fromStdString(result->error_msg.data);
      } catch (const std::exception & e) {
        error = QString::fromStdString(e.what());
      }
      runOnGuiThread(
        [this, valid, error]()
        {
          if (error.isEmpty()) {
            setStatus(
              valid ? tr("target OK — collision-free and within joint limits")
                    : tr("target REJECTED — self-collision, scene collision, or a "
                         "joint outside its limits (the server prints the detail "
                         "to its console)"));
          } else {
            setStatus(tr("fk_batch failed: %1").arg(error));
          }
        });
    });
}

void JointsTab::clearLocks()
{
  if (joint_locks_client_ == nullptr || !joint_locks_client_->service_is_ready()) {
    setStatus(tr("set_joint_locks service not available"));
    return;
  }
  setStatus(
    tr("clearing joint locks — the server rebuilds its model and solvers "
       "(blocking, ~20 s). Watch its console; this tab stays responsive."));
  auto request = std::make_shared<curobov2_ros_interfaces::srv::SetJointLocks::Request>();
  // restore: true drops the whole override and re-parses the robot YAML's own
  // lock_joints; every other request field is ignored. An empty (non-restore)
  // request would be a read-only query instead, which is what refreshJointInfo
  // already gets for free.
  request->restore = true;
  joint_locks_client_->async_send_request(
    request,
    [this](
      rclcpp::Client<curobov2_ros_interfaces::srv::SetJointLocks>::SharedFuture future)
    {
      bool success = false;
      QString message;
      try {
        auto result = future.get();
        success = result->success;
        message = QString::fromStdString(result->message);
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread(
        [this, success, message]()
        {
          setStatus(success ? tr("joint locks updated: %1").arg(message)
                            : tr("could not clear joint locks: %1").arg(message));
          // The cspace just changed shape (a locked joint may have come back
          // into it), so re-read it rather than trusting the old row set.
          refreshJointInfo();
        });
    });
}

void JointsTab::planTarget()
{
  if (!cspace_known_) {
    setStatus(tr("cspace unknown — cannot build a joint-space goal yet"));
    return;
  }
  if (trajectory_client_ == nullptr || !trajectory_client_->service_is_ready()) {
    setStatus(tr("generate_trajectory service not available"));
    return;
  }

  auto request = std::make_shared<curobov2_ros_interfaces::srv::TrajectoryGeneration::Request>();
  request->request.goalsets.resize(1);
  auto & goalset = request->request.goalsets[0];
  // Joint-space segment: a non-empty target_joint_positions switches the
  // planner to plan_cspace. Values go in cspace order, which is the tree's row
  // order (get_joint_info reported it), so no re-sorting is needed.
  for (double v : targetVector(tree_)) {
    goalset.target_joint_positions.push_back(v);
  }

  setStatus(tr("planning to the target configuration..."));
  trajectory_client_->async_send_request(
    request,
    [this](
      rclcpp::Client<curobov2_ros_interfaces::srv::TrajectoryGeneration>::SharedFuture
        future)
    {
      bool success = false;
      QString message;
      int waypoints = 0;
      try {
        auto result = future.get();
        success = result->response.success;
        message = QString::fromStdString(result->response.message);
        waypoints = static_cast<int>(result->response.trajectory.size());
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread(
        [this, success, message, waypoints]()
        {
          setStatus(
            success ? tr("plan OK — %1 trajectory segment(s); the robot did NOT move")
                          .arg(waypoints)
                    : tr("plan FAILED: %1").arg(message));
        });
    });
}

void JointsTab::sendTarget()
{
  if (!cspace_known_) {
    setStatus(tr("cspace unknown — cannot build a joint-space goal yet"));
    return;
  }
  if (action_client_ == nullptr || !action_client_->action_server_is_ready()) {
    setStatus(tr("execute_trajectory action not available"));
    return;
  }
  if (trajectory_client_ == nullptr || !trajectory_client_->service_is_ready()) {
    setStatus(tr("generate_trajectory service not available"));
    return;
  }

  // Execution is MTC-literal: the server neither re-plans nor reuses a cached
  // plan, so plan FIRST and replay the result verbatim. (This is the open-loop
  // path; a reactive planner has no waypoints to stream and the server will
  // refuse the replay.)
  auto request =
    std::make_shared<curobov2_ros_interfaces::srv::TrajectoryGeneration::Request>();
  request->request.goalsets.resize(1);
  for (double v : targetVector(tree_)) {
    request->request.goalsets[0].target_joint_positions.push_back(v);
  }

  setStatus(tr("planning + executing to the target configuration..."));
  trajectory_client_->async_send_request(
    request,
    [this](
      rclcpp::Client<curobov2_ros_interfaces::srv::TrajectoryGeneration>::
        SharedFuture future)
    {
      bool success = false;
      QString message;
      trajectory_msgs::msg::JointTrajectory replay;
      try {
        auto result = future.get();
        success = result->response.success;
        message = QString::fromStdString(result->response.message);
        if (success) {
          replay = toReplayTrajectory(result->response);
        }
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      if (!success) {
        runOnGuiThread([this, message]() {
          setStatus(tr("execute plan FAILED: %1").arg(message));
        });
        return;
      }
      runOnGuiThread([this, replay]() {
        // The plan landed; keep it and drive exactly it.
        last_trajectory_ = replay;
        curobov2_ros_interfaces::action::SendTrajectory::Goal goal;
        goal.goal.goalsets.resize(1);
        for (double v : targetVector(tree_)) {
          goal.goal.goalsets[0].target_joint_positions.push_back(v);
        }
        goal.replay = true;
        goal.replay_trajectory = last_trajectory_;

        rclcpp_action::Client<
          curobov2_ros_interfaces::action::SendTrajectory>::SendGoalOptions options;
        options.goal_response_callback = [this](
          rclcpp_action::ClientGoalHandle<
            curobov2_ros_interfaces::action::SendTrajectory>::SharedPtr handle)
        { onGoalResponse(handle); };
        action_client_->async_send_goal(goal, options);
      });
    });
}

void JointsTab::onGoalResponse(
  rclcpp_action::ClientGoalHandle<
    curobov2_ros_interfaces::action::SendTrajectory>::SharedPtr handle)
{
  if (handle == nullptr) {
    runOnGuiThread([this]() { setStatus(tr("goal rejected (server busy?)")); });
    return;
  }
  goal_handle_ = handle;
  runOnGuiThread([this]() { setGoalActive(true); });

  auto result_cb = [this](
                     const rclcpp_action::ClientGoalHandle<
                       curobov2_ros_interfaces::action::SendTrajectory>::WrappedResult & r)
  { onGoalResult(r); };
  action_client_->async_get_result(handle, result_cb);
}

void JointsTab::onGoalResult(
  const rclcpp_action::ClientGoalHandle<
    curobov2_ros_interfaces::action::SendTrajectory>::WrappedResult & result)
{
  // Two hops, not one. WrappedResult::result is the action's Result message
  // (SendTrajectory_Result), which wraps the payload as a single field
  // `TrajectoryResult result` -- success/message live on THAT, not on the
  // wrapper. The service callers in this file read `result->success` directly
  // because those services return TrajectoryResult itself.
  bool success = false;
  QString message;
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      if (result.result != nullptr) {
        success = result.result->result.success;
        message = QString::fromStdString(result.result->result.message);
      } else {
        message = tr("succeeded but returned no result");
      }
      break;
    case rclcpp_action::ResultCode::CANCELED:
      message = tr("goal canceled");
      break;
    case rclcpp_action::ResultCode::ABORTED:
      if (result.result != nullptr) {
        message = tr("aborted: %1").arg(QString::fromStdString(result.result->result.message));
      } else {
        message = tr("aborted");
      }
      break;
    default:
      message = tr("goal failed");
      break;
  }
  runOnGuiThread(
    [this, success, message]()
    {
      setGoalActive(false);
      setStatus(success ? tr("target reached: %1").arg(message)
                        : tr("execution failed: %1").arg(message));
    });
}

void JointsTab::setGoalActive(bool active)
{
  goal_active_ = active;
  if (!active) {
    goal_handle_.reset();
  }
  setButtonsEnabled();
}

void JointsTab::stopGoal()
{
  if (goal_handle_ == nullptr || action_client_ == nullptr) {
    return;
  }
  setStatus(tr("canceling the active goal..."));
  action_client_->async_cancel_goal(goal_handle_);
}


}  // namespace curobov2_ros_rviz
