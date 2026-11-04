#include "curobov2_ros_rviz/planning_tab.hpp"
#include "curobov2_ros_interfaces/msg/goalset.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// Only startMpc() still talks to SetPlanner: it re-asserts MPC on the server
#include "curobov2_ros_interfaces/srv/set_planner.hpp"
// Reached transitively via send_trajectory.hpp -> TrajectoryResult, but named
// explicitly because result_callback reads its fields directly.
#include "curobov2_ros_interfaces/msg/planning_stats.hpp"
// Named explicitly for toReplayTrajectory() below (same reasoning: reached
// transitively via trajectory_generation.hpp, but read field-by-field here).
#include "curobov2_ros_interfaces/msg/trajectory_result.hpp"
#include <QApplication>
#include <QListWidget>
#include <QSignalBlocker>
#include <QThread>
#include <QTimer>
#include <cmath>

namespace {

// A planned TrajectoryResult -> the execute action's replay form.
trajectory_msgs::msg::JointTrajectory toReplayTrajectory(
    const curobov2_ros_interfaces::msg::TrajectoryResult &result) {
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
    pt.time_from_start.nanosec =
        static_cast<uint32_t>((sec - std::floor(sec)) * 1e9);
    traj.points.push_back(pt);
  }
  return traj;
}

}  // namespace

namespace curobov2_ros_rviz {
void PlanningTab::refreshTargetToolList() {
  if (!target_tool_list_) {
    return;
  }
  const auto tools =
      target_display_ ? target_display_->getToolFrames() : std::vector<std::string>();
  const auto enabled =
      target_display_ ? target_display_->getEnabledTools() : std::vector<std::string>();
  const QString active = target_display_
                             ? QString::fromStdString(target_display_->getActiveTool())
                             : QString();

  const QSignalBlocker blocker(target_tool_list_);
  bool same = target_tool_list_->count() == static_cast<int>(tools.size());
  for (int i = 0; same && i < target_tool_list_->count(); ++i) {
    same = target_tool_list_->item(i)->data(Qt::UserRole).toString() ==
      QString::fromStdString(tools[static_cast<size_t>(i)]);
  }
  if (!same) {
    target_tool_list_->clear();
    for (const auto & tool : tools) {
      const QString label = tool.empty() ? tr("target") : QString::fromStdString(tool);
      auto * item = new QListWidgetItem(label, target_tool_list_);
      item->setData(Qt::UserRole, QString::fromStdString(tool));
      item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
    }
  }

  for (int i = 0; i < target_tool_list_->count(); ++i) {
    auto * item = target_tool_list_->item(i);
    const std::string tool = item->data(Qt::UserRole).toString().toStdString();
    const bool is_enabled =
      std::find(enabled.begin(), enabled.end(), tool) != enabled.end();
    item->setCheckState(is_enabled ? Qt::Checked : Qt::Unchecked);
  }

  int active_row = -1;
  for (int i = 0; i < target_tool_list_->count(); ++i) {
    if (target_tool_list_->item(i)->data(Qt::UserRole).toString() == active) {
      active_row = i;
      break;
    }
  }
  if (active_row < 0 && target_tool_list_->count() > 0) {
    active_row = 0;
  }
  if (active_row >= 0) {
    target_tool_list_->setCurrentRow(active_row);
  }
  target_tool_list_->setEnabled(!tools.empty());
}

void PlanningTab::onTargetToolChecked(QListWidgetItem * item) {
  if (!target_display_ || item == nullptr) {
    return;
  }
  const QString tool = item->data(Qt::UserRole).toString();
  const bool checked = item->checkState() == Qt::Checked;
  if (!checked) {
    int checked_count = 0;
    for (int i = 0; i < target_tool_list_->count(); ++i) {
      if (target_tool_list_->item(i)->checkState() == Qt::Checked) {
        ++checked_count;
      }
    }
    if (checked_count == 0) {
      // Empty selection would mean "all tools" to the display. Keep the last
      // goal checked; a plan with no tool target is never what the user wants.
      const QSignalBlocker blocker(target_tool_list_);
      item->setCheckState(Qt::Checked);
      return;
    }
  }
  target_display_->setToolEnabled(tool.toStdString(), checked);
  if (checked) {
    target_display_->setActiveTool(tool.toStdString());
  }
  updateMarkerPoseDisplay();
}

void PlanningTab::onTargetToolSelected(
    QListWidgetItem * current, QListWidgetItem * previous) {
  (void)previous;
  if (!target_display_ || current == nullptr) {
    return;
  }
  target_display_->setActiveTool(
      current->data(Qt::UserRole).toString().toStdString());
  updateMarkerPoseDisplay();
}

void PlanningTab::canonicalize(geometry_msgs::msg::Quaternion & q) {
  const double c[4] = {q.x, q.y, q.z, q.w};
  int sign_ref = 0;
  for (int i = 1; i < 4; ++i) {
    if (std::fabs(c[i]) > std::fabs(c[sign_ref])) {
      sign_ref = i;
    }
  }

  if (c[sign_ref] < 0.0) {
    q.x = -q.x;
    q.y = -q.y;
    q.z = -q.z;
    q.w = -q.w;
  }
}

bool PlanningTab::normalize(geometry_msgs::msg::Quaternion & q) {
  // A quaternion is only a rotation if it is unit length. Four independent spin
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (norm < 1e-9) {
    return false;
  }
  q.x /= norm;
  q.y /= norm;
  q.z /= norm;
  q.w /= norm;
  return true;
}

void PlanningTab::readQuaternionFromSpinboxes(geometry_msgs::msg::Quaternion & q) {
  q.x = ui_->spinBoxQx->value();
  q.y = ui_->spinBoxQy->value();
  q.z = ui_->spinBoxQz->value();
  q.w = ui_->spinBoxQw->value();
}

void PlanningTab::writeQuaternionToSpinboxes(const geometry_msgs::msg::Quaternion & q) {
  // Every one of these boxes has valueChanged connected to
  const QSignalBlocker bq(ui_->spinBoxQx);
  const QSignalBlocker by(ui_->spinBoxQy);
  const QSignalBlocker bz(ui_->spinBoxQz);
  const QSignalBlocker bw(ui_->spinBoxQw);
  ui_->spinBoxQx->setValue(q.x);
  ui_->spinBoxQy->setValue(q.y);
  ui_->spinBoxQz->setValue(q.z);
  ui_->spinBoxQw->setValue(q.w);
}

void PlanningTab::applyPoseFromSpinboxes() {
  if (!target_display_) {
    RCLCPP_WARN(node_->get_logger(), "Target display not available yet");
    return;
  }

  // Get values from spinboxes
  geometry_msgs::msg::Pose pose;
  pose.position.x = ui_->spinBoxPosX->value();
  pose.position.y = ui_->spinBoxPosY->value();
  pose.position.z = ui_->spinBoxPosZ->value();

  // The orientation is read, normalised and canonicalised, then WRITTEN BACK to
  geometry_msgs::msg::Quaternion q;
  readQuaternionFromSpinboxes(q);
  const bool usable = normalize(q);
  if (!usable) {
    RCLCPP_WARN(
        node_->get_logger(),
        "Goal orientation is a zero quaternion (0,0,0,0), which is not a "
        "rotation; leaving the goal orientation alone. w=1 is the identity.");
    return;
  }
  canonicalize(q);
  writeQuaternionToSpinboxes(q);
  pose.orientation = q;

  // Apply the new pose to the marker
  target_display_->setPose(pose);

  // Update last displayed values to match what we just set
  last_displayed_x_ = pose.position.x;
  last_displayed_y_ = pose.position.y;
  last_displayed_z_ = pose.position.z;
  last_displayed_q_ = q;

  RCLCPP_INFO(
      node_->get_logger(),
      "Applied pose: X=%.3f, Y=%.3f, Z=%.3f, Q=(%.4f, %.4f, %.4f, %.4f)",
      pose.position.x, pose.position.y, pose.position.z, q.x, q.y, q.z, q.w);
}

bool PlanningTab::eventFilter(QObject *obj, QEvent *event) {
  // Check if the event is a FocusIn event on one of the pose spinboxes
  if (event->type() == QEvent::FocusIn) {
    if (obj == ui_->spinBoxPosX || obj == ui_->spinBoxPosY ||
        obj == ui_->spinBoxPosZ || obj == ui_->spinBoxQx ||
        obj == ui_->spinBoxQy || obj == ui_->spinBoxQz ||
        obj == ui_->spinBoxQw) {
      user_editing_pose_ = true;
    }
  }
  // Pass the event to the base class
  return QObject::eventFilter(obj, event);
}

void PlanningTab::maybeSeedTargetFromTool() {
  // One-shot seed: a freshly added CuroboPlanningDisplay parks at its hardcoded
  if (!target_display_ || target_display_ == seeded_display_ ||
      user_editing_pose_) {
    return;
  }
  if (!target_display_->isDefaultPose()) {
    seeded_display_ = target_display_;
    return;
  }
  geometry_msgs::msg::PoseStamped::SharedPtr tool;
  {
    std::lock_guard<std::mutex> lock(tool_pose_mutex_);
    tool = tool_pose_;
  }
  if (!tool || tool->header.frame_id.empty()) {
    return;
  }
  const std::string fixed = goalFrame();
  if (fixed.empty()) {
    return;  // No fixed frame yet: leave the default, retry next poll.
  }
  if (fixed == tool->header.frame_id) {
    target_display_->setPose(tool->pose);
    seeded_display_ = target_display_;
    return;
  }
  try {
    const auto tf = seed_tf_buffer_->lookupTransform(
        fixed, tool->header.frame_id, tf2::TimePointZero,
        tf2::durationFromSec(0.5));
    geometry_msgs::msg::PoseStamped out;
    tf2::doTransform(*tool, out, tf);
    target_display_->setPose(out.pose);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 5000,
        "Target seed: cannot transform tool pose '%s' -> '%s': %s",
        tool->header.frame_id.c_str(), fixed.c_str(), ex.what());
    return;
  }
  seeded_display_ = target_display_;
}

void PlanningTab::updateMarkerPoseDisplay() {
  // Don't update if marker not found yet or if user is editing
  if (!target_display_ || user_editing_pose_) {
    return;
  }

  maybeSeedTargetFromTool();

  auto pose = target_display_->getPose();

  geometry_msgs::msg::Quaternion q = pose.orientation;
  canonicalize(q);

  // Compare with last displayed values - only update if changed
  constexpr double epsilon_pos = 1e-6;  // metres
  // Quaternion components live in [-1, 1], so the threshold is an absolute one
  // on the components rather than an angle in degrees.
  constexpr double epsilon_rot = 1e-4;

  bool x_changed = std::isnan(last_displayed_x_) ||
                   std::fabs(pose.position.x - last_displayed_x_) > epsilon_pos;
  bool y_changed = std::isnan(last_displayed_y_) ||
                   std::fabs(pose.position.y - last_displayed_y_) > epsilon_pos;
  bool z_changed = std::isnan(last_displayed_z_) ||
                   std::fabs(pose.position.z - last_displayed_z_) > epsilon_pos;
  // A quaternion has no meaningful per-component "change" on its own -- the
  const bool q_changed = std::isnan(last_displayed_q_.w) ||
                         std::fabs(q.x - last_displayed_q_.x) > epsilon_rot ||
                         std::fabs(q.y - last_displayed_q_.y) > epsilon_rot ||
                         std::fabs(q.z - last_displayed_q_.z) > epsilon_rot ||
                         std::fabs(q.w - last_displayed_q_.w) > epsilon_rot;

  // Only update if at least one value has changed
  if (!x_changed && !y_changed && !z_changed && !q_changed) {
    return;
  }

  // Update the spinboxes with current pose
  // Block signals to avoid triggering updates while we're setting values
  ui_->spinBoxPosX->blockSignals(true);
  ui_->spinBoxPosY->blockSignals(true);
  ui_->spinBoxPosZ->blockSignals(true);
  // The four quaternion boxes are muted inside writeQuaternionToSpinboxes();
  // doing it here too would be the nested-blockSignals hazard described there.

  if (x_changed) {
    ui_->spinBoxPosX->setValue(pose.position.x);
    last_displayed_x_ = pose.position.x;
  }
  if (y_changed) {
    ui_->spinBoxPosY->setValue(pose.position.y);
    last_displayed_y_ = pose.position.y;
  }
  if (z_changed) {
    ui_->spinBoxPosZ->setValue(pose.position.z);
    last_displayed_z_ = pose.position.z;
  }
  if (q_changed) {
    writeQuaternionToSpinboxes(q);
    last_displayed_q_ = q;
  }

  ui_->spinBoxPosX->blockSignals(false);
  ui_->spinBoxPosY->blockSignals(false);
  ui_->spinBoxPosZ->blockSignals(false);
}

// Clear re-seeds the goal from the robot's live state.
void PlanningTab::on_clearGoal_clicked() {
  if (target_display_ == nullptr) {
    RCLCPP_WARN(
        node_->get_logger(),
        "Clear needs a CuroboPlanningDisplay in RViz; there is no pose to read back");
    return;
  }

  user_editing_pose_ = false;

  last_displayed_x_ = std::numeric_limits<double>::quiet_NaN();
  last_displayed_y_ = std::numeric_limits<double>::quiet_NaN();
  last_displayed_z_ = std::numeric_limits<double>::quiet_NaN();
  // NaN w = "never displayed".
  last_displayed_q_.x = 0.0;
  last_displayed_q_.y = 0.0;
  last_displayed_q_.z = 0.0;
  last_displayed_q_.w = std::numeric_limits<double>::quiet_NaN();

  updateMarkerPoseDisplay();
  RCLCPP_INFO(node_->get_logger(),
              "Goal state reset to the target marker pose");
}


} // namespace curobov2_ros_rviz
