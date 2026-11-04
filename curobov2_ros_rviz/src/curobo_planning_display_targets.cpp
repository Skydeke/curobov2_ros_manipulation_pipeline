#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <OgreQuaternion.h>
#include <OgreSceneNode.h>
#include <OgreVector3.h>

#include <rclcpp/exceptions.hpp>
#include <rclcpp/parameter_client.hpp>
#include <rclcpp/time.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/frame_manager_iface.hpp>
#include <rviz_common/logging.hpp>
#include <rviz_common/panel_dock_widget.hpp>
#include <rviz_common/properties/parse_color.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/validate_floats.hpp>
#include <rviz_common/window_manager_interface.hpp>
#include <rviz_default_plugins/robot/robot_link.hpp>
#include <rviz_rendering/objects/shape.hpp>

#include <curobov2_ros_rviz/curobo_link_updater.hpp>
#include <curobov2_ros_rviz/curobo_planning_display.hpp>

namespace
{

bool validateFloats(const visualization_msgs::msg::InteractiveMarker & msg)
{
  bool valid = rviz_common::validateFloats(msg.pose);
  valid = valid && rviz_common::validateFloats(msg.scale);
  for (const auto & control : msg.controls) {
    valid = valid && rviz_common::validateFloats(control.orientation);
    for (const auto & marker : control.markers) {
      valid = valid && rviz_common::validateFloats(marker.pose);
      valid = valid && rviz_common::validateFloats(marker.scale);
      valid = valid && rviz_common::validateFloats(marker.color);
      valid = valid && rviz_common::validateFloats(marker.points);
    }
  }
  return valid;
}

std::vector<std::string> splitToolFrames(const std::string & text)
{
  std::vector<std::string> frames;
  std::string current;
  for (char c : text) {
    if (c == ',' || c == ';' || std::isspace(static_cast<unsigned char>(c))) {
      if (!current.empty()) {
        frames.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) {
    frames.push_back(current);
  }
  return frames;
}

}  // namespace

namespace curobov2_ros_rviz
{

void CuroboPlanningDisplay::ensureTarget(const std::string & frame)
{
  if (tools_.find(frame) != tools_.end()) {
    return;
  }
  ToolTarget target;
  target.frame = frame;
  target.pose = defaultPose();
  if (seed_from_tf_property_ && seed_from_tf_property_->getBool() && !frame.empty()) {
    target.tf_seeded = seedToolFromTf(target);
  }
  tools_.emplace(frame, target);
  tool_order_.push_back(frame);
}

bool CuroboPlanningDisplay::seedToolFromTf(ToolTarget & target)
{
  if (target.frame.empty() || context_ == nullptr) {
    return false;
  }
  auto * frame_manager = context_->getFrameManager();
  if (frame_manager == nullptr) {
    return false;
  }
  Ogre::Vector3 position;
  Ogre::Quaternion orientation;
  if (!frame_manager->getTransform(target.frame, position, orientation)) {
    return false;
  }
  target.pose.position.x = position.x;
  target.pose.position.y = position.y;
  target.pose.position.z = position.z;
  target.pose.orientation.x = orientation.x;
  target.pose.orientation.y = orientation.y;
  target.pose.orientation.z = orientation.z;
  target.pose.orientation.w = orientation.w;
  return true;
}

void CuroboPlanningDisplay::seedPendingTargets()
{
  if (!seed_from_tf_property_ || !seed_from_tf_property_->getBool()) {
    return;
  }
  for (auto & entry : tools_) {
    ToolTarget & target = entry.second;
    if (target.tf_seeded || entry.first.empty()) {
      continue;
    }
    if (!seedToolFromTf(target)) {
      continue;
    }
    target.tf_seeded = true;
    if (entry.first == active_tool_) {
      updatePoseProperties(target.pose);
    } else {
      sync_pending_.insert(entry.first);
    }
  }
}

geometry_msgs::msg::Pose CuroboPlanningDisplay::activePose() const
{
  auto it = tools_.find(active_tool_);
  return it == tools_.end() ? defaultPose() : it->second.pose;
}

geometry_msgs::msg::Pose CuroboPlanningDisplay::getPose() const
{
  return activePose();
}

geometry_msgs::msg::Pose CuroboPlanningDisplay::getPose(const std::string & tool) const
{
  auto it = tools_.find(tool);
  return it == tools_.end() ? defaultPose() : it->second.pose;
}

void CuroboPlanningDisplay::setPose(const geometry_msgs::msg::Pose & pose)
{
  setPose(active_tool_, pose);
}

void CuroboPlanningDisplay::setPose(
  const std::string & tool, const geometry_msgs::msg::Pose & pose)
{
  ensureTarget(tool);
  tools_[tool].pose = pose;
  tools_[tool].tf_seeded = true;
  goal_refresh_all_ = true;
  if (tool == active_tool_) {
    updatePoseProperties(pose);
  } else {
    sync_pending_.insert(tool);
  }
}

std::string CuroboPlanningDisplay::getGizmoFrame() const
{
  std::lock_guard<std::mutex> lock(gizmo_mutex_);
  return gizmo_frame_;
}

bool CuroboPlanningDisplay::isDefaultPose() const
{
  const auto & pose = activePose();
  const auto expected = defaultPose();
  return std::abs(pose.position.x - expected.position.x) < 1e-9 &&
         std::abs(pose.position.y - expected.position.y) < 1e-9 &&
         std::abs(pose.position.z - expected.position.z) < 1e-9 &&
         pose.orientation.x == expected.orientation.x &&
         pose.orientation.y == expected.orientation.y &&
         pose.orientation.z == expected.orientation.z &&
         pose.orientation.w == expected.orientation.w;
}

std::string CuroboPlanningDisplay::getActiveTool() const
{
  return active_tool_;
}

void CuroboPlanningDisplay::setActiveTool(const std::string & tool)
{
  if (tool == active_tool_ || tools_.find(tool) == tools_.end()) {
    return;
  }
  active_tool_ = tool;
  updatePoseProperties(activePose());
  // Every tool keeps its own IK ghost now, so switching the active tool must
  // NOT invalidate the others.
}

std::vector<std::string> CuroboPlanningDisplay::getToolFrames() const
{
  return tool_order_;
}

bool CuroboPlanningDisplay::isToolEnabled(const std::string & tool) const
{
  return enabled_tools_.empty() ||
         enabled_tools_.find(tool) != enabled_tools_.end();
}

std::vector<std::string> CuroboPlanningDisplay::getEnabledTools() const
{
  std::vector<std::string> enabled;
  for (const auto & tool : tool_order_) {
    if (isToolEnabled(tool)) {
      enabled.push_back(tool);
    }
  }
  return enabled;
}

void CuroboPlanningDisplay::setEnabledTools(const std::vector<std::string> & tools)
{
  std::vector<std::string> wanted;
  for (const auto & tool : tools) {
    if (std::find(wanted.begin(), wanted.end(), tool) == wanted.end()) {
      wanted.push_back(tool);
    }
  }

  enabled_tools_.clear();
  for (const auto & tool : wanted) {
    enabled_tools_.insert(tool);
  }

  applying_enabled_tools_ = true;
  std::string joined;
  for (const auto & tool : tool_order_) {
    if (!isToolEnabled(tool) || tool.empty()) {
      continue;
    }
    if (!joined.empty()) {
      joined += ", ";
    }
    joined += tool;
  }
  active_tools_property_->setStdString(joined);
  applying_enabled_tools_ = false;

  // The goal set changed: refresh every tool so the next plan carries exactly
  // the selected targets.
  goal_refresh_all_ = true;
  updateGoalRobotLinks();
}

void CuroboPlanningDisplay::setToolEnabled(const std::string & tool, bool enabled)
{
  if (enabled) {
    if (enabled_tools_.empty()) {
      // "Empty == all" -> materialize the current full set first.
      for (const auto & frame : tool_order_) {
        enabled_tools_.insert(frame);
      }
    }
    enabled_tools_.insert(tool);
  } else {
    if (enabled_tools_.empty()) {
      for (const auto & frame : tool_order_) {
        enabled_tools_.insert(frame);
      }
    }
    enabled_tools_.erase(tool);
  }
  std::vector<std::string> enabled_list;
  for (const auto & frame : tool_order_) {
    if (isToolEnabled(frame)) {
      enabled_list.push_back(frame);
    }
  }
  setEnabledTools(enabled_list);
}

void CuroboPlanningDisplay::setToolFrames(const std::vector<std::string> & frames)
{
  std::vector<std::string> wanted;
  for (const auto & frame : frames) {
    if (!frame.empty() &&
      std::find(wanted.begin(), wanted.end(), frame) == wanted.end())
    {
      wanted.push_back(frame);
    }
  }
  if (wanted.empty()) {
    wanted.push_back("");
  }

  std::map<std::string, ToolTarget> updated;
  std::vector<std::string> order;
  for (const auto & frame : wanted) {
    auto it = tools_.find(frame);
    if (it != tools_.end()) {
      updated.emplace(frame, it->second);
    } else {
      ToolTarget target;
      target.frame = frame;
      target.pose = defaultPose();
      if (seed_from_tf_property_ && seed_from_tf_property_->getBool()) {
        target.tf_seeded = seedToolFromTf(target);
      }
      updated.emplace(frame, target);
    }
    order.push_back(frame);
  }

  const bool same = order == tool_order_;
  const std::vector<std::string> previous_order = tool_order_;
  tools_ = std::move(updated);
  tool_order_ = std::move(order);
  if (std::find(tool_order_.begin(), tool_order_.end(), active_tool_) ==
    tool_order_.end())
  {
    active_tool_ = tool_order_.front();
  }
  // Forget selections for tools that actually disappeared. Names that were
  // never in the previous list are kept: they may be a config value loaded
  // before get_joint_info supplied the server's tool list.
  for (auto it = enabled_tools_.begin(); it != enabled_tools_.end();) {
    const bool was_known =
      std::find(previous_order.begin(), previous_order.end(), *it) !=
      previous_order.end();
    if (was_known && tools_.find(*it) == tools_.end()) {
      it = enabled_tools_.erase(it);
    } else {
      ++it;
    }
  }
  // Active Tools is deliberately NOT rewritten here: an empty value means
  // "all configured tools", so materialising the current server list would
  // turn a later server-side tool addition into an exclusion.

  goal_refresh_all_ = true;
  updateGoalRobotLinks();
  updatePoseProperties(activePose());
  if (!same && gizmo_active_) {
    rebuildGizmoMarkers();
  }
}

std::vector<std::pair<std::string, geometry_msgs::msg::Pose>>
CuroboPlanningDisplay::getToolPoses() const
{
  std::vector<std::pair<std::string, geometry_msgs::msg::Pose>> poses;
  poses.reserve(tool_order_.size());
  for (const auto & frame : tool_order_) {
    // Only selected targets become goals; an unselected arm is left
    // unconstrained by the planner instead of being pinned to a stale pose.
    if (isToolEnabled(frame)) {
      poses.emplace_back(frame, tools_.at(frame).pose);
    }
  }
  return poses;
}

bool CuroboPlanningDisplay::poseInBase(
  const geometry_msgs::msg::Pose & pose,
  geometry_msgs::msg::Pose & base_pose) const
{
  return goalPoseInBase(pose, base_pose);
}

std::vector<std::pair<std::string, geometry_msgs::msg::Pose>>
CuroboPlanningDisplay::getToolPosesInBase() const
{
  std::vector<std::pair<std::string, geometry_msgs::msg::Pose>> poses;
  for (const auto & target : getToolPoses()) {
    geometry_msgs::msg::Pose base_pose;
    if (goalPoseInBase(target.second, base_pose)) {
      poses.emplace_back(target.first, base_pose);
    }
  }
  return poses;
}

geometry_msgs::msg::Pose CuroboPlanningDisplay::targetPoseFromProperties() const
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = target_position_x_->getFloat();
  pose.position.y = target_position_y_->getFloat();
  pose.position.z = target_position_z_->getFloat();
  pose.orientation.x = target_orientation_x_->getFloat();
  pose.orientation.y = target_orientation_y_->getFloat();
  pose.orientation.z = target_orientation_z_->getFloat();
  pose.orientation.w = target_orientation_w_->getFloat();
  return pose;
}

void CuroboPlanningDisplay::updateTargetPose()
{
  if (applying_gizmo_) {
    return;
  }
  ensureTarget(active_tool_);
  tools_[active_tool_].pose = targetPoseFromProperties();
  tools_[active_tool_].tf_seeded = true;
  goal_refresh_all_ = true;
  sync_pending_.insert(active_tool_);
}

void CuroboPlanningDisplay::updatePoseProperties(const geometry_msgs::msg::Pose & pose)
{
  applying_gizmo_ = true;
  target_position_x_->setFloat(pose.position.x);
  target_position_y_->setFloat(pose.position.y);
  target_position_z_->setFloat(pose.position.z);
  target_orientation_x_->setFloat(pose.orientation.x);
  target_orientation_y_->setFloat(pose.orientation.y);
  target_orientation_z_->setFloat(pose.orientation.z);
  target_orientation_w_->setFloat(pose.orientation.w);
  applying_gizmo_ = false;
  sync_pending_.insert(active_tool_);
}

void CuroboPlanningDisplay::updateActiveTools()
{
  if (applying_enabled_tools_) {
    return;
  }
  const auto frames = splitToolFrames(active_tools_property_->getStdString());
  // Empty string is the "all configured tools" sentinel; do not materialize
  // it here or a config round-trip would freeze the current server list.
  enabled_tools_.clear();
  for (const auto & frame : frames) {
    enabled_tools_.insert(frame);
  }
  goal_refresh_all_ = true;
}

// ============================================================================
// Interactive markers
// ============================================================================


}  // namespace curobov2_ros_rviz
