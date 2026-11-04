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

void CuroboPlanningDisplay::updateTrajectoryRoot()
{
  Ogre::Vector3 offset = Ogre::Vector3::ZERO;
  Ogre::Quaternion rotation = Ogre::Quaternion::IDENTITY;
  auto * frame_manager = context_ ? context_->getFrameManager() : nullptr;
  const std::string root_frame =
    fk_base_link_.empty() ? robot_root_frame_ : fk_base_link_;
  if (!root_frame.empty() && frame_manager != nullptr &&
    root_frame != frame_manager->getFixedFrame())
  {
    rclcpp::Time stamp(0, 0, RCL_ROS_TIME);
    if (frame_manager->getTransform(root_frame, stamp, offset, rotation)) {
      setTransformOk();
    } else {
      setMissingTransformToFixedFrame(root_frame);
      return;
    }
  }
  if (trajectory_root_) {
    trajectory_root_->setPosition(offset);
    trajectory_root_->setOrientation(rotation);
  }
  if (goal_root_) {
    goal_root_->setPosition(offset);
    goal_root_->setOrientation(rotation);
  }
  robot_root_offset_ = offset;
  robot_root_rotation_ = rotation;
  for (auto & trail : trail_robots_) {
    if (trail.node) {
      trail.node->setPosition(offset);
      trail.node->setOrientation(rotation);
    }
  }
}

// ============================================================================
// Trajectory topics and FK
// ============================================================================

void CuroboPlanningDisplay::updateTopic()
{
  subscription_.reset();
  trajectory_msg_new_.reset();
  animating_ = false;
  current_state_ = 0;
  current_frac_ = 0.0;
  anim_time_ = 0.0;
  current_transforms_.clear();
  // Not fk_base_link_: switching the trajectory topic does not change the
  // planner's base link (only setPlannerNode() does).
  waypoints_.clear();
  clearTrailRobots();
  resetFkState();
  if (trajectory_panel_) {
    trajectory_panel_->setDuration(0.0);
    trajectory_panel_->setPaused(false);
  }

  if (!topic_property_->getStdString().empty()) {
    auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
    if (!ros_node_abstraction) {
      return;
    }
    subscription_ = ros_node_abstraction->get_raw_node()
      ->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      topic_property_->getStdString(), 10,
      std::bind(&CuroboPlanningDisplay::onTrajectoryMessage, this,
        std::placeholders::_1));
  }
}

void CuroboPlanningDisplay::onTrajectoryMessage(
  trajectory_msgs::msg::JointTrajectory::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(trajectory_mutex_);
  trajectory_msg_new_ = msg;
}

void CuroboPlanningDisplay::createFkClient()
{
  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (!ros_node_abstraction) {
    return;
  }
  std::string node = planner_node_;
  while (!node.empty() && node.front() == '/') {
    node.erase(0, 1);
  }
  while (!node.empty() && node.back() == '/') {
    node.pop_back();
  }
  auto raw_node = ros_node_abstraction->get_raw_node();
  fk_client_ = raw_node->create_client<curobov2_ros_interfaces::srv::FkBatch>(
    "/" + node + "/fk_batch");
  goal_fk_client_ = raw_node->create_client<curobov2_ros_interfaces::srv::FkBatch>(
    "/" + node + "/fk_batch");
  ik_client_ = raw_node->create_client<curobov2_ros_interfaces::srv::IkBatch>(
    "/" + node + "/ik_batch");
  tool_info_client_ =
    raw_node->create_client<curobov2_ros_interfaces::srv::GetJointInfo>(
    "/" + node + "/get_joint_info");
  tool_info_in_flight_ = false;
  last_tool_info_request_ = std::chrono::steady_clock::time_point{};
  {
    std::lock_guard<std::mutex> lock(tool_frames_mutex_);
    pending_tool_frames_.clear();
    pending_tool_frames_ready_ = false;
  }

  // fk_batch reports the base link only after the first solve. The goal-state
  // IK needs that same frame *before* any trajectory exists, so ask the planner
  // for its configured base_link and use it until an FK response confirms it.
  param_client_ = std::make_shared<rclcpp::AsyncParametersClient>(raw_node, node);
  ++param_generation_;
  base_link_query_in_flight_ = false;
  base_link_query_sent_ = false;
  {
    std::lock_guard<std::mutex> lock(base_link_mutex_);
    pending_base_link_.clear();
  }
}

void CuroboPlanningDisplay::refreshToolFrames()
{
  if (!tool_info_client_ || tool_info_in_flight_) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  if (now - last_tool_info_request_ < std::chrono::seconds(2)) {
    return;
  }
  if (!tool_info_client_->service_is_ready()) {
    return;
  }

  tool_info_in_flight_ = true;
  last_tool_info_request_ = now;
  auto request =
    std::make_shared<curobov2_ros_interfaces::srv::GetJointInfo::Request>();
  tool_info_client_->async_send_request(
    request,
    [this](rclcpp::Client<curobov2_ros_interfaces::srv::GetJointInfo>::
          SharedFuture future) {
      std::vector<std::string> frames;
      std::vector<std::string> joints;
      bool ok = false;
      try {
        auto response = future.get();
        ok = response->success;
        frames = response->tool_frames;
        joints = response->joint_names;
      } catch (const std::exception &) {
      }
      tool_info_in_flight_ = false;
      if (!ok) {
        return;
      }
      std::lock_guard<std::mutex> lock(tool_frames_mutex_);
      pending_tool_frames_ = std::move(frames);
      cspace_joints_ = std::move(joints);
      pending_tool_frames_ready_ = true;
    });
}

void CuroboPlanningDisplay::setPlannerNode(const std::string & node)
{
  std::string wanted = node;
  while (!wanted.empty() && wanted.front() == '/') {
    wanted.erase(0, 1);
  }
  while (!wanted.empty() && wanted.back() == '/') {
    wanted.pop_back();
  }
  if (wanted.empty() || wanted == planner_node_) {
    return;
  }

  // The planner owns the base link and the service names. Drop everything
  // derived from the previous server before rebinding: a stale base link moves
  // both the goal ghost and the trajectory ghost to the wrong place.
  std::string previous = planner_node_;
  while (!previous.empty() && previous.front() == '/') {
    previous.erase(0, 1);
  }
  planner_node_ = wanted;
  applying_planner_node_ = true;
  if (planner_node_property_) {
    planner_node_property_->setStdString(wanted);
  }
  applying_planner_node_ = false;

  // Default topics follow the planner node. A topic the user deliberately
  // pointed elsewhere (the task-constructor ghost is the common case) is left
  // alone.
  if (topic_property_ && !previous.empty()) {
    const std::string current = topic_property_->getStdString();
    if (current.empty() || current == "/" + previous + "/trajectory") {
      topic_property_->setStdString("/" + wanted + "/trajectory");
    }
  }
  if (collision_topic_property_ && !previous.empty()) {
    const std::string current = collision_topic_property_->getStdString();
    if (current.empty() || current == "/" + previous + "/collision_spheres") {
      collision_topic_property_->setStdString("/" + wanted + "/collision_spheres");
    }
  }

  fk_base_link_.clear();
  createFkClient();
  resetFkState();
  resetGoalState();
  updateTrajectoryRoot();
}

void CuroboPlanningDisplay::queryBaseLink()
{
  if (param_client_ == nullptr || !fk_base_link_.empty() ||
    base_link_query_sent_.load() || base_link_query_in_flight_.load())
  {
    return;
  }
  if (!param_client_->service_is_ready()) {
    return;
  }

  base_link_query_sent_ = true;
  base_link_query_in_flight_ = true;
  const uint64_t generation = param_generation_.load();
  param_client_->get_parameters(
    {"base_link"},
    [this, generation](std::shared_future<std::vector<rclcpp::Parameter>> future) {
      std::string base;
      try {
        const auto params = future.get();
        if (!params.empty() &&
          params.front().get_type() == rclcpp::ParameterType::PARAMETER_STRING)
        {
          base = params.front().as_string();
        }
      } catch (const std::exception &) {
        // A planner without the parameter keeps using the URDF-root fallback.
      }
      // A reply for a planner we already switched away from must not relabel
      // the current planner's base frame.
      if (param_generation_.load() != generation) {
        return;
      }
      {
        std::lock_guard<std::mutex> lock(base_link_mutex_);
        pending_base_link_ = base;
      }
      base_link_query_in_flight_ = false;
    });
}

void CuroboPlanningDisplay::updatePlannerNode()
{
  if (applying_planner_node_) {
    return;
  }
  setPlannerNode(planner_node_property_->getStdString());
}

void CuroboPlanningDisplay::resetFkState()
{
  fk_in_flight_ = false;
  fk_pending_msg_.reset();
  {
    std::lock_guard<std::mutex> lock(fk_mutex_);
    fk_response_.reset();
  }
}

bool CuroboPlanningDisplay::requestFk(
  const trajectory_msgs::msg::JointTrajectory::ConstSharedPtr & msg)
{
  if (!msg || msg->points.empty()) {
    return false;
  }
  if (!fk_client_ || !fk_client_->service_is_ready()) {
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "FK",
      QString("Waiting for /%1/fk_batch.").arg(
        QString::fromStdString(planner_node_)));
    return false;
  }

  auto request = std::make_shared<curobov2_ros_interfaces::srv::FkBatch::Request>();
  request->joint_states.reserve(msg->points.size());
  for (const auto & point : msg->points) {
    sensor_msgs::msg::JointState state;
    state.name = msg->joint_names;
    state.position = point.positions;
    request->joint_states.push_back(std::move(state));
  }
  if (urdf_model_) {
    for (const auto & link : urdf_model_->links_) {
      request->link_names.push_back(link.first);
    }
  }

  const uint64_t seq = ++fk_seq_;
  fk_inflight_seq_ = seq;
  fk_in_flight_ = true;
  fk_sent_at_ = std::chrono::steady_clock::now();
  fk_pending_msg_ = msg;

  fk_client_->async_send_request(
    request,
    [this, seq](rclcpp::Client<curobov2_ros_interfaces::srv::FkBatch>::
               SharedFuture future) {
      auto response = std::make_shared<FkResponse>();
      response->seq = seq;
      try {
        auto result = future.get();
        response->ok = result->success;
        response->error = result->error_msg.data;
        response->link_names = result->link_names;
        response->base_link = result->base_link;
        response->poses.reserve(result->link_poses.size());
        for (const auto & pose : result->link_poses) {
          response->poses.push_back({
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.w, pose.orientation.x, pose.orientation.y,
            pose.orientation.z});
        }
      } catch (const std::exception & exc) {
        response->error = exc.what();
      }
      std::lock_guard<std::mutex> lock(fk_mutex_);
      if (seq == fk_inflight_seq_) {
        fk_response_ = response;
      }
    });
  return true;
}

bool CuroboPlanningDisplay::consumeFk(uint64_t seq)
{
  std::shared_ptr<FkResponse> response;
  {
    std::lock_guard<std::mutex> lock(fk_mutex_);
    if (!fk_response_ || fk_response_->seq != seq) {
      return false;
    }
    response = fk_response_;
    fk_response_.reset();
  }

  fk_in_flight_ = false;
  if (!response->ok) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "FK",
      QString("fk_batch failed: %1").arg(QString::fromStdString(response->error)));
    fk_pending_msg_.reset();
    return false;
  }
  if (!fk_pending_msg_) {
    return false;
  }
  if (!response->base_link.empty()) {
    fk_base_link_ = response->base_link;
  }

  const size_t points = fk_pending_msg_->points.size();
  const size_t links = response->link_names.size();
  if (links == 0 || response->poses.size() < points * links) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "FK",
      "fk_batch returned no link poses.");
    fk_pending_msg_.reset();
    return false;
  }

  // Trajectories may omit non-cspace joints (the gripper fingers). Seed the
  // client-side FK with the live state so an omitted jaw keeps its real
  // position instead of falling back to zero (closed).
  const auto live_joints = currentJointPositions();
  waypoints_.clear();
  waypoints_.reserve(points);
  for (size_t i = 0; i < points; ++i) {
    WaypointData waypoint;
    for (size_t j = 0; j < links; ++j) {
      const auto & pose = response->poses[i * links + j];
      Eigen::Quaterniond q(pose[3], pose[4], pose[5], pose[6]);
      if (q.squaredNorm() < 1e-12) {
        q = Eigen::Quaterniond::Identity();
      }
      q.normalize();
      Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
      transform.translation() = Eigen::Vector3d(pose[0], pose[1], pose[2]);
      transform.linear() = q.toRotationMatrix();
      waypoint.link_transforms[response->link_names[j]] = transform;
    }
    std::map<std::string, double> joint_positions = live_joints;
    const auto & msg = *fk_pending_msg_;
    const auto & point = msg.points[i];
    const size_t n = std::min(msg.joint_names.size(), point.positions.size());
    for (size_t j = 0; j < n; ++j) {
      joint_positions[msg.joint_names[j]] = point.positions[j];
    }
    // The message is authoritative for every joint it names. `live_joints`
    // only supplies joints the trajectory omits (e.g. a finger a producer
    // forgot); it must never override an explicit planned jaw value.
    mergeClientFk(joint_positions, waypoint.link_transforms);
    const auto & time = fk_pending_msg_->points[i].time_from_start;
    waypoint.time_from_start = time.sec + time.nanosec * 1e-9;
    waypoints_.push_back(std::move(waypoint));
  }

  double previous = -1.0;
  for (auto & waypoint : waypoints_) {
    if (waypoint.time_from_start <= previous) {
      waypoint.time_from_start = previous + 0.025;
    }
    previous = waypoint.time_from_start;
  }

  anim_time_ = 0.0;
  current_state_ = 0;
  current_frac_ = 0.0;
  animating_ = true;
  fk_pending_msg_.reset();
  updateTrajectoryRoot();
  deleteStatus("FK");
  return true;
}

void CuroboPlanningDisplay::mergeClientFk(
  const std::map<std::string, double> & joint_positions,
  std::map<std::string, Eigen::Isometry3d> & transforms) const
{
  std::map<std::string, Eigen::Isometry3d> all;
  if (!fk_engine_.computeFK(joint_positions, all) || all.empty()) {
    return;
  }

  // CuroboFK works in the URDF root frame; fk_batch poses are in the planner's
  // base_link. Convert the client result into base before merging.
  const std::string base = fk_base_link_.empty() ? robot_root_frame_ : fk_base_link_;
  const auto base_it = all.find(base);
  if (base_it == all.end()) {
    return;
  }
  const Eigen::Isometry3d world_to_base = base_it->second.inverse();
  for (auto & entry : all) {
    entry.second = world_to_base * entry.second;
  }
  // The server only knows the planner cspace. Any other joint in the request
  // (the gripper fingers are the important case) is invisible to it, so its
  // fk_batch poses for those links are the URDF default (closed) and must NOT
  // overwrite the client FK that used the full joint state. When extra joints
  // are present the client URDF result is authoritative for the whole model;
  // server-only links are still added below.
  bool has_non_cspace_joint = false;
  for (const auto & entry : joint_positions) {
    if (std::find(cspace_joints_.begin(), cspace_joints_.end(), entry.first) ==
        cspace_joints_.end()) {
      has_non_cspace_joint = true;
      break;
    }
  }
  if (has_non_cspace_joint) {
    for (const auto & entry : transforms) {
      if (all.find(entry.first) == all.end())
        all[entry.first] = entry.second;
    }
  } else {
    // Server poses are authoritative where present.
    for (const auto & entry : transforms) {
      all[entry.first] = entry.second;
    }
  }
  transforms.swap(all);
}

// ============================================================================
// Goal-state ghost
// ============================================================================

void CuroboPlanningDisplay::createTrajectoryPanel()
{
  auto * window_manager = context_ ? context_->getWindowManager() : nullptr;
  if (window_manager == nullptr || trajectory_panel_ != nullptr) {
    return;
  }
  trajectory_panel_ = new TrajectoryPanel(window_manager->getParentWindow());
  trajectory_dock_panel_ =
    window_manager->addPane(getName() + " - Trajectory", trajectory_panel_);
  trajectory_panel_->onScrub = [this](double seconds) {
    anim_time_ = seconds;
    animating_ = !waypoints_.empty();
  };
  trajectory_panel_->onPauseToggled = [this](bool paused) {
    if (!paused && !waypoints_.empty() &&
      anim_time_ >= waypoints_.back().time_from_start)
    {
      anim_time_ = 0.0;
    }
  };
}

void CuroboPlanningDisplay::destroyTrajectoryPanel()
{
  if (trajectory_dock_panel_ != nullptr) {
    delete trajectory_dock_panel_;
    trajectory_dock_panel_ = nullptr;
    trajectory_panel_ = nullptr;
  } else {
    delete trajectory_panel_;
    trajectory_panel_ = nullptr;
  }
}

// ============================================================================
// Property slots
// ============================================================================

void CuroboPlanningDisplay::updateAlpha()
{
  if (trajectory_robot_) {
    trajectory_robot_->setAlpha(alpha_property_->getFloat());
  }
  for (auto & trail : trail_robots_) {
    if (trail.robot) {
      trail.robot->setAlpha(alpha_property_->getFloat() * 0.4f);
    }
  }
}

void CuroboPlanningDisplay::updateShowTrail()
{
  updateTrailVisibility();
}

void CuroboPlanningDisplay::updateTrailStepSize()
{
  if (!waypoints_.empty()) {
    rebuildTrail();
  }
}

void CuroboPlanningDisplay::updateLoop() {}
void CuroboPlanningDisplay::updateSpeed() {}

// ============================================================================
// Per-frame update
// ============================================================================

void CuroboPlanningDisplay::update(float wall_dt, float ros_dt)
{
  Display::update(wall_dt, ros_dt);

  syncSpheres();
  if (!spheres_got_message_ && !collision_topic_property_->getStdString().empty()) {
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "Collision Spheres",
      QString("No messages on '%1' yet.")
      .arg(QString::fromStdString(collision_topic_property_->getStdString())));
  }

  // Targets work without a URDF.
  if (gizmo_client_) {
    gizmo_client_->update();
  }
  for (auto & entry : interactive_markers_map_) {
    entry.second->update();
  }
  if (gizmo_active_) {
    syncGizmoToProperties();
    syncPropertiesToGizmo();
  }
  const bool gizmo_want = gizmo_visible_property_ && gizmo_visible_property_->getBool();
  if (gizmo_want && !gizmo_active_) {
    createGizmo();
    gizmo_active_ = true;
  } else if (!gizmo_want && gizmo_active_) {
    destroyGizmo();
    gizmo_active_ = false;
  }
  if (gizmo_active_ && gizmo_server_) {
    const std::string fixed = context_->getFrameManager()->getFixedFrame();
    if (!fixed.empty() && fixed != gizmo_frame_) {
      gizmo_frame_.clear();
      gizmo_server_->clear();
      gizmo_server_->applyChanges();
      gizmo_server_.reset();
    }
  }
  if (gizmo_active_ && !gizmo_server_) {
    createGizmo();
  }

  // The server owns the tool-frame list; no RViz property can override it.
  refreshToolFrames();
  {
    std::vector<std::string> server_tools;
    bool have_tools = false;
    {
      std::lock_guard<std::mutex> lock(tool_frames_mutex_);
      if (pending_tool_frames_ready_) {
        server_tools = std::move(pending_tool_frames_);
        pending_tool_frames_.clear();
        pending_tool_frames_ready_ = false;
        have_tools = true;
      }
    }
    if (have_tools) {
      const bool unnamed =
        tool_order_.size() == 1 && tool_order_.front().empty();
      if (server_tools.empty()) {
        if (!unnamed) {
          setToolFrames({});
        }
      } else if (server_tools != tool_order_) {
        setToolFrames(server_tools);
      }
    }
  }

  if (!robot_loaded_) {
    loadRobotModel();
  }
  if (!robot_loaded_) {
    return;
  }

  // Learn the planner base link from its parameter before the first FK reply.
  queryBaseLink();
  {
    std::string learned;
    {
      std::lock_guard<std::mutex> lock(base_link_mutex_);
      if (!pending_base_link_.empty()) {
        learned.swap(pending_base_link_);
      }
    }
    if (!learned.empty() && learned != fk_base_link_) {
      fk_base_link_ = learned;
      // The previous goal was solved/placed relative to the fallback frame.
      // Drop it so it is re-solved and re-placed in the right one.
      resetGoalState();
      updateTrajectoryRoot();
    }
  }

  updateTrajectoryRoot();
  seedPendingTargets();

  // Goal-state ghost: one combined multi-tool IK -> FK for all targets.
  if (goal_stage_ == GoalStage::WaitingIk) {
    consumeGoalIk();
  } else if (goal_stage_ == GoalStage::WaitingFk) {
    consumeGoalFk();
  } else {
    requestGoalState();
  }
  if (goal_stage_ != GoalStage::Idle &&
    std::chrono::steady_clock::now() - goal_stage_start_ > std::chrono::seconds(15))
  {
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "Goal State",
      "Timed out waiting for the IK/FK services.");
    resetGoalState();
  }

  // Fetch all trajectory link poses from the server.
  bool promoted = false;
  if (!fk_in_flight_) {
    trajectory_msgs::msg::JointTrajectory::ConstSharedPtr pending;
    {
      std::lock_guard<std::mutex> lock(trajectory_mutex_);
      if (trajectory_msg_new_ && !trajectory_msg_new_->points.empty()) {
        pending = trajectory_msg_new_;
      }
    }
    if (pending && requestFk(pending)) {
      std::lock_guard<std::mutex> lock(trajectory_mutex_);
      if (trajectory_msg_new_ == pending) {
        trajectory_msg_new_.reset();
      }
    }
  }
  if (fk_in_flight_) {
    if (consumeFk(fk_inflight_seq_)) {
      promoted = true;
    } else if (std::chrono::steady_clock::now() - fk_sent_at_ >
      std::chrono::seconds(30))
    {
      setStatus(
        rviz_common::properties::StatusProperty::Warn, "FK",
        QString("Timed out waiting for /%1/fk_batch.").arg(
          QString::fromStdString(planner_node_)));
      resetFkState();
    }
  }

  if (waypoints_.empty()) {
    return;
  }

  const double total = waypoints_.back().time_from_start;
  if (promoted) {
    trajectory_robot_->setVisible(true);
    playback_spheres_ = live_spheres_;
    applySpheres(playback_spheres_);
    poseSpheres();
    rebuildTrail();
    if (trajectory_panel_) {
      trajectory_panel_->setDuration(total);
      trajectory_panel_->setPaused(false);
      trajectory_panel_->setPosition(0.0);
    }
  }

  if (trajectory_panel_ && trajectory_panel_->isPaused()) {
    anim_time_ = std::clamp(trajectory_panel_->position(), 0.0, total);
    animating_ = true;
  } else {
    const double speed =
      std::max(0.0, static_cast<double>(speed_property_->getFloat()));
    anim_time_ += std::min(static_cast<double>(wall_dt), 0.1) * speed;
    if (total > 0.0) {
      if (anim_time_ >= total) {
        if (loop_property_->getBool()) {
          anim_time_ = std::fmod(anim_time_, total);
        } else {
          anim_time_ = total;
          animating_ = false;
          if (trajectory_panel_) {
            trajectory_panel_->setPosition(anim_time_);
            trajectory_panel_->setPaused(true);
          }
        }
      }
    } else {
      animating_ = false;
    }
    if (trajectory_panel_) {
      trajectory_panel_->setPosition(anim_time_);
    }
  }

  size_t segment = waypoints_.size() - 1;
  double fraction = 1.0;
  for (size_t i = 0; i + 1 < waypoints_.size(); ++i) {
    if (anim_time_ < waypoints_[i + 1].time_from_start) {
      segment = i;
      const double dt =
        waypoints_[i + 1].time_from_start - waypoints_[i].time_from_start;
      fraction = (dt > 0.0)
        ? (anim_time_ - waypoints_[i].time_from_start) / dt
        : 0.0;
      break;
    }
  }
  current_state_ = static_cast<int>(segment);
  current_frac_ = fraction;

  current_transforms_ =
    (segment + 1 < waypoints_.size() && fraction > 0.0001)
      ? interpolateWaypoints(segment, segment + 1, fraction)
      : waypoints_[segment].link_transforms;

  CuroboLinkUpdater updater(current_transforms_);
  trajectory_robot_->update(updater);
  poseSpheres();
  updateTrailVisibility();
}

std::map<std::string, Eigen::Isometry3d> CuroboPlanningDisplay::interpolateWaypoints(
  size_t waypoint_a, size_t waypoint_b, double t) const
{
  const auto & a = waypoints_[waypoint_a].link_transforms;
  const auto & b = waypoints_[waypoint_b].link_transforms;
  std::map<std::string, Eigen::Isometry3d> out;
  for (const auto & entry : a) {
    auto it = b.find(entry.first);
    if (it == b.end()) {
      out[entry.first] = entry.second;
      continue;
    }
    Eigen::Quaterniond qa(entry.second.linear());
    Eigen::Quaterniond qb(it->second.linear());
    qa.normalize();
    qb.normalize();
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() =
      entry.second.translation() * (1.0 - t) + it->second.translation() * t;
    transform.linear() = qa.slerp(t, qb).toRotationMatrix();
    out[entry.first] = transform;
  }
  for (const auto & entry : b) {
    if (out.find(entry.first) == out.end()) {
      out[entry.first] = entry.second;
    }
  }
  return out;
}


}  // namespace curobov2_ros_rviz
