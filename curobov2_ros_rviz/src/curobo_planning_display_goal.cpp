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

void CuroboPlanningDisplay::updateGoalStateVisibility()
{
  // Per-link visibility is applied by updateGoalRobotLinks(); calling
  // setVisible() alone is not enough because that path can re-show every link.
  updateGoalRobotLinks();
  goal_attempted_ = false;
}

void CuroboPlanningDisplay::updateGoalStateAlpha()
{
  if (goal_robot_) {
    goal_robot_->setAlpha(goal_state_alpha_property_->getFloat());
  }
}

bool CuroboPlanningDisplay::posesEqual(
  const geometry_msgs::msg::Pose & a, const geometry_msgs::msg::Pose & b)
{
  constexpr double eps = 1e-5;
  return std::abs(a.position.x - b.position.x) < eps &&
         std::abs(a.position.y - b.position.y) < eps &&
         std::abs(a.position.z - b.position.z) < eps &&
         std::abs(a.orientation.x - b.orientation.x) < eps &&
         std::abs(a.orientation.y - b.orientation.y) < eps &&
         std::abs(a.orientation.z - b.orientation.z) < eps &&
         std::abs(a.orientation.w - b.orientation.w) < eps;
}

bool CuroboPlanningDisplay::goalPoseInBase(
  const geometry_msgs::msg::Pose & pose, geometry_msgs::msg::Pose & base_pose) const
{
  base_pose = pose;
  // The target lives in the RViz fixed frame. IK expects the planner's base
  // link, NOT the URDF root: a robot whose URDF root is "world" (OpenArm) has
  // its configured base_link below that root, so using the root silently
  // solves for a point offset by the root->base transform.
  const std::string base_frame =
    fk_base_link_.empty() ? robot_root_frame_ : fk_base_link_;
  if (base_frame.empty() || context_ == nullptr ||
    context_->getFrameManager() == nullptr)
  {
    return true;
  }
  auto * frame_manager = context_->getFrameManager();
  if (base_frame == frame_manager->getFixedFrame()) {
    return true;
  }
  Ogre::Vector3 base_position;
  Ogre::Quaternion base_rotation;
  if (!frame_manager->getTransform(
      base_frame, rclcpp::Time(0, 0, RCL_ROS_TIME), base_position,
      base_rotation))
  {
    return false;
  }
  const Ogre::Quaternion inverse = base_rotation.Inverse();
  const Ogre::Vector3 position(
    static_cast<float>(pose.position.x), static_cast<float>(pose.position.y),
    static_cast<float>(pose.position.z));
  const Ogre::Quaternion orientation(
    static_cast<float>(pose.orientation.w), static_cast<float>(pose.orientation.x),
    static_cast<float>(pose.orientation.y), static_cast<float>(pose.orientation.z));
  const Ogre::Vector3 local = inverse * (position - base_position);
  Ogre::Quaternion local_rotation = inverse * orientation;
  local_rotation.normalise();
  base_pose.position.x = local.x;
  base_pose.position.y = local.y;
  base_pose.position.z = local.z;
  base_pose.orientation.w = local_rotation.w;
  base_pose.orientation.x = local_rotation.x;
  base_pose.orientation.y = local_rotation.y;
  base_pose.orientation.z = local_rotation.z;
  return true;
}

void CuroboPlanningDisplay::resetGoalState()
{
  goal_stage_ = GoalStage::Idle;
  goal_valid_ = false;
  goal_attempted_ = false;
  {
    std::lock_guard<std::mutex> lock(goal_mutex_);
    goal_ik_result_.reset();
    goal_fk_result_.reset();
  }
  if (goal_robot_) {
    goal_robot_->setVisible(false);
  }
}

void CuroboPlanningDisplay::onJointState(
  const sensor_msgs::msg::JointState::ConstSharedPtr & msg)
{
  if (!msg) {
    return;
  }
  std::map<std::string, double> joints;
  const std::size_t n = std::min(msg->name.size(), msg->position.size());
  for (std::size_t i = 0; i < n; ++i) {
    joints[msg->name[i]] = msg->position[i];
  }
  std::lock_guard<std::mutex> lock(joint_state_mutex_);
  current_joints_ = std::move(joints);
}

std::map<std::string, double>
CuroboPlanningDisplay::currentJointPositions() const
{
  std::lock_guard<std::mutex> lock(joint_state_mutex_);
  return current_joints_;
}

std::set<std::string> CuroboPlanningDisplay::activeChainJoints() const
{
  std::set<std::string> joints;
  if (!urdf_model_) {
    return joints;
  }
  for (const auto & tool : tool_order_) {
    if (!isToolEnabled(tool)) {
      continue;
    }
    auto link_it = urdf_model_->links_.find(tool);
    if (link_it == urdf_model_->links_.end()) {
      continue;
    }
    urdf::LinkConstSharedPtr link = link_it->second;
    while (link) {
      if (link->parent_joint) {
        joints.insert(link->parent_joint->name);
      }
      if (!link->parent_joint) {
        break;
      }
      link = urdf_model_->getLink(link->parent_joint->parent_link_name);
    }
  }
  return joints;
}

void CuroboPlanningDisplay::overlayPassiveJoints(
  sensor_msgs::msg::JointState & joint_state) const
{
  if (joint_state.name.empty()) {
    std::lock_guard<std::mutex> lock(tool_frames_mutex_);
    if (joint_state.position.size() >= cspace_joints_.size()) {
      joint_state.name = cspace_joints_;
    }
  }

  const auto active = activeChainJoints();
  const auto current = currentJointPositions();
  const std::size_t n =
    std::min(joint_state.name.size(), joint_state.position.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (active.find(joint_state.name[i]) != active.end()) {
      continue;  // active chain: keep the IK solution
    }
    auto it = current.find(joint_state.name[i]);
    if (it != current.end()) {
      joint_state.position[i] = it->second;
    }
  }

  // Carry the remaining live joints (e.g. gripper fingers) so the client-side
  // FK pass can place them too. The server's FK ignores non-cspace names.
  std::set<std::string> present(joint_state.name.begin(), joint_state.name.end());
  for (const auto & entry : current) {
    if (present.find(entry.first) == present.end()) {
      joint_state.name.push_back(entry.first);
      joint_state.position.push_back(entry.second);
    }
  }
}

void CuroboPlanningDisplay::requestGoalState()
{
  if (goal_stage_ != GoalStage::Idle || !goal_robot_ || !robot_loaded_ ||
    robot_root_frame_.empty())
  {
    return;
  }
  if (!show_goal_state_property_ || !show_goal_state_property_->getBool()) {
    return;
  }
  if (!ik_client_ || !ik_client_->service_is_ready()) {
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "Goal State",
      QString("Waiting for /%1/ik_batch.").arg(
        QString::fromStdString(planner_node_)));
    return;
  }

  // Solve ONLY the active tools. Passive tool frames must not become IK
  // goals: sending their FK pose as an end-effector target let the solver move
  // the shared linear axis to satisfy them, corrupting the active solve.
  // Passive joints are restored from the live joint state after the solve.
  std::vector<geometry_msgs::msg::Pose> goal_poses;
  std::vector<std::string> goal_tools;
  for (const auto & tool : tool_order_) {
    if (!isToolEnabled(tool)) {
      continue;
    }
    auto target = tools_.find(tool);
    if (target == tools_.end()) {
      continue;
    }

    geometry_msgs::msg::Pose base_pose;
    if (!goalPoseInBase(target->second.pose, base_pose)) {
      setStatus(
        rviz_common::properties::StatusProperty::Warn, "Goal State",
        "Waiting for the robot base transform.");
      return;
    }
    goal_poses.push_back(base_pose);
    goal_tools.push_back(tool);
  }
  if (goal_poses.empty()) {
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  bool same_poses = goal_attempted_ &&
    goal_poses.size() == last_goal_poses_.size() &&
    goal_tools == last_goal_tools_;
  for (std::size_t i = 0; same_poses && i < goal_poses.size(); ++i) {
    same_poses = posesEqual(goal_poses[i], last_goal_poses_[i]);
  }
  const bool forced = goal_refresh_all_;
  goal_refresh_all_ = false;
  if (!forced && same_poses) {
    // A solved goal is stable; a failed one is retried slowly (the IK solver
    // may still be warming up, or the scene may have changed).
    if (goal_valid_ || now - last_goal_request_ < std::chrono::seconds(1)) {
      return;
    }
  } else if (now - last_goal_request_ < std::chrono::milliseconds(200)) {
    return;
  }

  auto request = std::make_shared<curobov2_ros_interfaces::srv::IkBatch::Request>();
  request->frame_id = "";
  if (goal_poses.size() == 1 && goal_tools.front().empty()) {
    // Legacy single unnamed target: the server treats this as its first tool.
    request->poses = goal_poses;
    request->tool_frame = "";
  } else {
    request->poses = goal_poses;
    request->tool_frames = goal_tools;
    request->tool_frame = "";
  }

  last_goal_poses_ = goal_poses;
  last_goal_tools_ = goal_tools;
  last_goal_pose_ = activePose();
  goal_attempted_ = true;
  last_goal_request_ = now;
  goal_stage_ = GoalStage::WaitingIk;
  goal_stage_start_ = now;
  goal_inflight_seq_ = ++goal_seq_;
  const uint64_t seq = goal_inflight_seq_;

  ik_client_->async_send_request(
    request,
    [this, seq](rclcpp::Client<curobov2_ros_interfaces::srv::IkBatch>::
               SharedFuture future) {
      auto result = std::make_shared<GoalIkResult>();
      result->seq = seq;
      try {
        auto response = future.get();
        result->ok = response->success && !response->joint_states.empty() &&
          !response->joint_states_valid.empty() &&
          response->joint_states_valid.front().data;
        result->error = response->error_msg.data;
        if (!response->joint_states.empty()) {
          result->joint_state = response->joint_states.front();
        }
      } catch (const std::exception & exc) {
        result->error = exc.what();
      }
      std::lock_guard<std::mutex> lock(goal_mutex_);
      if (seq == goal_inflight_seq_) {
        goal_ik_result_ = result;
      }
    });
}

void CuroboPlanningDisplay::updateGoalRobotLinks()
{
  if (!goal_robot_ || !urdf_model_) {
    return;
  }

  const auto chain_links = [this](const std::string & tool) {
      std::set<std::string> links;
      auto link_it = urdf_model_->links_.find(tool);
      if (link_it == urdf_model_->links_.end()) {
        return links;
      }
      urdf::LinkConstSharedPtr link = link_it->second;
      while (link) {
        links.insert(link->name);
        if (!link->parent_joint) {
          break;
        }
        link = urdf_model_->getLink(link->parent_joint->parent_link_name);
      }
      return links;
    };

  // The ghost is a complete robot: keep every configured tool chain,
  // including passive ones, so no arm disappears from the IK preview.
  std::set<std::string> visible;
  for (const auto & tool : tool_order_) {
    auto links = chain_links(tool);
    visible.insert(links.begin(), links.end());
  }

  const bool overall = goal_valid_ && show_goal_state_property_ &&
    show_goal_state_property_->getBool();
  goal_robot_->setVisible(overall);
  for (const auto & entry : goal_robot_->getLinks()) {
    auto * link = goal_robot_->getLink(entry.first);
    if (!link) {
      continue;
    }
    // An empty set means the tool names did not resolve in this URDF; fall
    // back to showing the whole robot rather than an empty ghost.
    const bool show = overall && (visible.empty() || visible.count(entry.first) > 0);
    if (link->getVisualNode()) {
      link->getVisualNode()->setVisible(show);
    }
    if (link->getCollisionNode()) {
      link->getCollisionNode()->setVisible(show);
    }
  }
}

void CuroboPlanningDisplay::consumeGoalIk()
{
  if (goal_stage_ != GoalStage::WaitingIk) {
    return;
  }
  std::shared_ptr<GoalIkResult> result;
  {
    std::lock_guard<std::mutex> lock(goal_mutex_);
    if (!goal_ik_result_ || goal_ik_result_->seq != goal_inflight_seq_) {
      return;
    }
    result = goal_ik_result_;
    goal_ik_result_.reset();
  }
  if (!result->ok) {
    goal_stage_ = GoalStage::Idle;
    goal_valid_ = false;
    if (goal_robot_) {
      goal_robot_->setVisible(false);
    }
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "Goal State",
      QString("No IK solution yet (%1).").arg(
        QString::fromStdString(result->error)));
    return;
  }

  // Keep the active tools at their IK solution and restore every passive
  // joint to the live robot state, exactly like passive motion planning.
  auto joint_state = result->joint_state;
  overlayPassiveJoints(joint_state);

  auto request = std::make_shared<curobov2_ros_interfaces::srv::FkBatch::Request>();
  request->joint_states.push_back(joint_state);
  if (urdf_model_) {
    for (const auto & link : urdf_model_->links_) {
      request->link_names.push_back(link.first);
    }
  }

  goal_stage_ = GoalStage::WaitingFk;
  goal_stage_start_ = std::chrono::steady_clock::now();
  goal_fk_client_->async_send_request(
    request,
    [this, seq = goal_inflight_seq_, joint_state](
      rclcpp::Client<curobov2_ros_interfaces::srv::FkBatch>::SharedFuture future) {
      auto result = std::make_shared<GoalFkResult>();
      result->seq = seq;
      result->joint_state = joint_state;
      try {
        auto response = future.get();
        result->ok = response->success;
        result->error = response->error_msg.data;
        result->link_names = response->link_names;
        result->base_link = response->base_link;
        result->poses.reserve(response->link_poses.size());
        for (const auto & pose : response->link_poses) {
          result->poses.push_back({
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.w, pose.orientation.x, pose.orientation.y,
            pose.orientation.z});
        }
      } catch (const std::exception & exc) {
        result->error = exc.what();
      }
      std::lock_guard<std::mutex> lock(goal_mutex_);
      if (seq == goal_inflight_seq_) {
        goal_fk_result_ = result;
      }
    });
}

void CuroboPlanningDisplay::consumeGoalFk()
{
  if (goal_stage_ != GoalStage::WaitingFk) {
    return;
  }
  std::shared_ptr<GoalFkResult> result;
  {
    std::lock_guard<std::mutex> lock(goal_mutex_);
    if (!goal_fk_result_ || goal_fk_result_->seq != goal_inflight_seq_) {
      return;
    }
    result = goal_fk_result_;
    goal_fk_result_.reset();
  }
  goal_stage_ = GoalStage::Idle;
  if (!result->ok || result->link_names.empty()) {
    goal_valid_ = false;
    if (goal_robot_) {
      goal_robot_->setVisible(false);
    }
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "Goal State",
      QString("fk_batch failed: %1").arg(QString::fromStdString(result->error)));
    return;
  }

  if (!result->base_link.empty()) {
    fk_base_link_ = result->base_link;
  }
  const size_t links = result->link_names.size();
  std::map<std::string, Eigen::Isometry3d> transforms;
  for (size_t j = 0; j < links && j < result->poses.size(); ++j) {
    const auto & pose = result->poses[j];
    Eigen::Quaterniond q(pose[3], pose[4], pose[5], pose[6]);
    if (q.squaredNorm() < 1e-12) {
      q = Eigen::Quaterniond::Identity();
    }
    q.normalize();
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = Eigen::Vector3d(pose[0], pose[1], pose[2]);
    transform.linear() = q.toRotationMatrix();
    transforms[result->link_names[j]] = transform;
  }
  std::map<std::string, double> joint_positions;
  const size_t n = std::min(
    result->joint_state.name.size(), result->joint_state.position.size());
  for (size_t j = 0; j < n; ++j) {
    joint_positions[result->joint_state.name[j]] = result->joint_state.position[j];
  }
  mergeClientFk(joint_positions, transforms);
  if (goal_robot_) {
    CuroboLinkUpdater updater(transforms);
    goal_robot_->update(updater);
    goal_valid_ = true;
    updateGoalRobotLinks();  // respects Show IK result and link filtering
    deleteStatus("Goal State");
  }
  updateTrajectoryRoot();
}

// ============================================================================
// Collision spheres
// ============================================================================


}  // namespace curobov2_ros_rviz
