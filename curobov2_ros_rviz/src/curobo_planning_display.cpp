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
#include <rclcpp/time.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/frame_manager_iface.hpp>
#include <rviz_common/logging.hpp>
#include <rviz_common/panel_dock_widget.hpp>
#include <rviz_common/properties/parse_color.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/validate_floats.hpp>
#include <rviz_common/window_manager_interface.hpp>
#include <rviz_default_plugins/robot/link_updater.hpp>
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

class TfLinkUpdater : public rviz_default_plugins::robot::LinkUpdater
{
public:
  explicit TfLinkUpdater(rviz_common::FrameManagerIface * frame_manager)
  : frame_manager_(frame_manager) {}

  bool getLinkTransforms(
    const std::string & link_name, Ogre::Vector3 & visual_position,
    Ogre::Quaternion & visual_orientation, Ogre::Vector3 & collision_position,
    Ogre::Quaternion & collision_orientation) const override
  {
    if (frame_manager_ == nullptr ||
      !frame_manager_->getTransform(link_name, visual_position, visual_orientation))
    {
      return false;
    }
    collision_position = visual_position;
    collision_orientation = visual_orientation;
    return true;
  }

private:
  rviz_common::FrameManagerIface * frame_manager_;
};

}  // namespace

namespace curobov2_ros_rviz
{

std::atomic<int> CuroboPlanningDisplay::gizmo_instance_counter_{0};

CuroboPlanningDisplay::CuroboPlanningDisplay()
{
  const int instance = gizmo_instance_counter_++;
  gizmo_name_ = (instance == 0) ? "target" : "target_" + std::to_string(instance);
}

CuroboPlanningDisplay::~CuroboPlanningDisplay()
{
  destroyTrajectoryPanel();
  destroyGizmo();
}

geometry_msgs::msg::Pose CuroboPlanningDisplay::defaultPose()
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = 0.4;
  pose.position.z = 0.4;
  pose.orientation.w = 1.0;
  return pose;
}

// ============================================================================
// Lifecycle
// ============================================================================

void CuroboPlanningDisplay::onInitialize()
{
  Display::onInitialize();

  // --- Planning request / targets ---
  tool_frames_property_ = new rviz_common::properties::StringProperty(
    "Tool Frames", "",
    "Comma- or space-separated cuRobo tool frames to show targets for. Empty "
    "shows a single unnamed target. The Curobo panel seeds this from "
    "get_joint_info; editing it overrides the server list.",
    this, SLOT(updateToolFrames()));

  active_tool_property_ = new rviz_common::properties::EditableEnumProperty(
    "Active Tool", "",
    "Tool frame edited by the Position/Orientation properties and the panel.",
    this, SLOT(updateActiveTool()));

  seed_from_tf_property_ = new rviz_common::properties::BoolProperty(
    "Seed From TF", true,
    "Place a newly added tool target at the tool's current TF pose.", this);

  gizmo_visible_property_ = new rviz_common::properties::BoolProperty(
    "Gizmo Visible", true,
    "Interactive 6-DOF markers for each tool target.", this,
    SLOT(updateGizmoVisible()));

  gizmo_scale_property_ = new rviz_common::properties::FloatProperty(
    "Marker Scale", 0.2, "Size of the 6-DOF marker handles.", this,
    SLOT(updateGizmoScale()));
  gizmo_scale_property_->setMin(0.01);
  gizmo_scale_property_->setMax(2.0);

  target_position_x_ = new rviz_common::properties::FloatProperty(
    "Target Position X", 0.4, "Active tool target X (m).", this,
    SLOT(updateTargetPose()));
  target_position_y_ = new rviz_common::properties::FloatProperty(
    "Target Position Y", 0.0, "Active tool target Y (m).", this,
    SLOT(updateTargetPose()));
  target_position_z_ = new rviz_common::properties::FloatProperty(
    "Target Position Z", 0.4, "Active tool target Z (m).", this,
    SLOT(updateTargetPose()));
  target_orientation_x_ = new rviz_common::properties::FloatProperty(
    "Target Orientation X", 0.0, "Active tool quaternion X.", this,
    SLOT(updateTargetPose()));
  target_orientation_y_ = new rviz_common::properties::FloatProperty(
    "Target Orientation Y", 0.0, "Active tool quaternion Y.", this,
    SLOT(updateTargetPose()));
  target_orientation_z_ = new rviz_common::properties::FloatProperty(
    "Target Orientation Z", 0.0, "Active tool quaternion Z.", this,
    SLOT(updateTargetPose()));
  target_orientation_w_ = new rviz_common::properties::FloatProperty(
    "Target Orientation W", 1.0, "Active tool quaternion W.", this,
    SLOT(updateTargetPose()));

  // --- Scene robot ---
  show_scene_robot_property_ = new rviz_common::properties::BoolProperty(
    "Show Scene Robot", true,
    "Render the live robot from TF.", this, SLOT(updateSceneRobotVisibility()));
  scene_robot_alpha_property_ = new rviz_common::properties::FloatProperty(
    "Scene Robot Alpha", 1.0, "Live robot opacity.", this,
    SLOT(updateSceneRobotAlpha()));
  scene_robot_alpha_property_->setMin(0.0);
  scene_robot_alpha_property_->setMax(1.0);

  show_goal_state_property_ = new rviz_common::properties::BoolProperty(
    "Show Goal State", true,
    "Render a ghost robot at the IK solution for the active target.", this,
    SLOT(updateGoalStateVisibility()));
  goal_state_alpha_property_ = new rviz_common::properties::FloatProperty(
    "Goal State Alpha", 0.5, "Goal-state ghost opacity.", this,
    SLOT(updateGoalStateAlpha()));
  goal_state_alpha_property_->setMin(0.0);
  goal_state_alpha_property_->setMax(1.0);

  // --- Planned path ---
  topic_property_ = new rviz_common::properties::RosTopicProperty(
    "Trajectory Topic", "/unified_planner/trajectory",
    "trajectory_msgs/msg/JointTrajectory",
    "trajectory_msgs/JointTrajectory topic to display.", this,
    SLOT(updateTopic()));
  topic_property_->initialize(context_->getRosNodeAbstraction());

  planner_node_property_ = new rviz_common::properties::StringProperty(
    "Planner Node", "unified_planner",
    "Node whose fk_batch service provides the waypoint link poses.", this,
    SLOT(updatePlannerNode()));

  alpha_property_ = new rviz_common::properties::FloatProperty(
    "Alpha", 1.0, "Planned robot opacity.", this, SLOT(updateAlpha()));
  alpha_property_->setMin(0.0);
  alpha_property_->setMax(1.0);

  show_trail_property_ = new rviz_common::properties::BoolProperty(
    "Show Trail", false, "Show ghost copies at past waypoints.", this,
    SLOT(updateShowTrail()));

  trail_step_size_property_ = new rviz_common::properties::IntProperty(
    "Trail Step Size", 10, "Show every Nth waypoint as a ghost.", this,
    SLOT(updateTrailStepSize()));
  trail_step_size_property_->setMin(1);

  loop_property_ = new rviz_common::properties::BoolProperty(
    "Loop Animation", false, "Restart when the trajectory finishes.", this,
    SLOT(updateLoop()));

  speed_property_ = new rviz_common::properties::FloatProperty(
    "Speed", 1.0, "Playback speed multiplier.", this, SLOT(updateSpeed()));
  speed_property_->setMin(0.01);
  speed_property_->setMax(100.0);

  collision_topic_property_ = new rviz_common::properties::RosTopicProperty(
    "Collision Spheres Topic", "/curobo_server/collision_spheres",
    "visualization_msgs/msg/MarkerArray",
    "Collision-sphere MarkerArray topic to display.", this,
    SLOT(updateCollisionTopic()));
  collision_topic_property_->initialize(context_->getRosNodeAbstraction());

  show_attached_property_ = new rviz_common::properties::BoolProperty(
    "Show Attached", true,
    "Show the grasped object's spheres (the 'attached_spheres' namespace).", this,
    SLOT(updateShowAttached()));
  show_collision_spheres_property_ = new rviz_common::properties::BoolProperty(
    "Show Robot Collision Spheres", false,
    "Show the robot's own collision spheres ('collision_spheres' namespace).",
    this, SLOT(updateShowCollisionSpheres()));

  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (ros_node_abstraction) {
    auto node = ros_node_abstraction->get_raw_node();
    auto transformer = context_->getFrameManager()->getTransformer();
    gizmo_client_ = std::make_unique<interactive_markers::InteractiveMarkerClient>(
      node, transformer, fixed_frame_.toStdString());
    gizmo_client_->setInitializeCallback(
      std::bind(&CuroboPlanningDisplay::gizmoInitializeCallback, this,
        std::placeholders::_1));
    gizmo_client_->setUpdateCallback(
      std::bind(&CuroboPlanningDisplay::gizmoUpdateCallback, this,
        std::placeholders::_1));
    gizmo_client_->setResetCallback(
      std::bind(&CuroboPlanningDisplay::gizmoResetCallback, this));
    gizmo_client_->setStatusCallback(
      std::bind(&CuroboPlanningDisplay::gizmoStatusCallback, this,
        std::placeholders::_1, std::placeholders::_2));
  }

  {
    std::lock_guard<std::mutex> lock(gizmo_mutex_);
    ensureTarget(active_tool_);
  }
  updatePoseProperties(activePose());

  try {
    rclcpp::get_logger("interactive_markers").set_level(rclcpp::Logger::Level::Warn);
    rclcpp::get_logger("interactive_markers::InteractiveMarkerClient")
    .set_level(rclcpp::Logger::Level::Warn);
  } catch (...) {
  }

  planner_node_ = planner_node_property_->getStdString();
  createFkClient();
  createTrajectoryPanel();
  updateTopic();
  updateCollisionTopic();

  if (!isEnabled()) {
    scene_node_->setVisible(false);
  }
}

void CuroboPlanningDisplay::onEnable()
{
  Display::onEnable();
  if (trajectory_dock_panel_) {
    trajectory_dock_panel_->show();
  }
  if (gizmo_visible_property_ && gizmo_visible_property_->getBool() && !gizmo_active_) {
    createGizmo();
    gizmo_active_ = true;
  }
}

void CuroboPlanningDisplay::onDisable()
{
  if (gizmo_active_) {
    destroyGizmo();
    gizmo_active_ = false;
  }
  if (trajectory_dock_panel_) {
    trajectory_dock_panel_->hide();
  }
  Display::onDisable();
}

void CuroboPlanningDisplay::fixedFrameChanged()
{
  if (gizmo_client_) {
    gizmo_client_->setTargetFrame(fixed_frame_.toStdString());
  }
  Display::fixedFrameChanged();
}

void CuroboPlanningDisplay::reset()
{
  Display::reset();
  resetFkState();
  resetGoalState();
  std::lock_guard<std::mutex> lock(trajectory_mutex_);
  trajectory_msg_new_.reset();
  waypoints_.clear();
  animating_ = false;
  current_state_ = 0;
  current_frac_ = 0.0;
  anim_time_ = 0.0;
  current_transforms_.clear();
  clearTrailRobots();
  if (trajectory_robot_) {
    trajectory_robot_->setVisible(false);
  }
  {
    std::lock_guard<std::mutex> spheres_lock(spheres_mutex_);
    spheres_msg_new_.reset();
  }
  spheres_got_message_ = false;
  sphere_active_count_ = 0;
  live_spheres_.clear();
  playback_spheres_.clear();
  for (auto & sv : sphere_pool_) {
    if (sv.node) {
      sv.node->setVisible(false);
    }
  }
  eraseAllGizmoMarkers();
  if (trajectory_panel_) {
    trajectory_panel_->setDuration(0.0);
    trajectory_panel_->setPaused(false);
  }
}

// ============================================================================
// URDF and robots
// ============================================================================

bool CuroboPlanningDisplay::loadURDF()
{
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    if (!urdf_xml_.empty()) {
      return true;
    }
  }

  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (!ros_node_abstraction) {
    return false;
  }
  auto node = ros_node_abstraction->get_raw_node();

  if (!robot_description_sub_) {
    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.durability(rclcpp::DurabilityPolicy::TransientLocal);
    robot_description_sub_ = node->create_subscription<std_msgs::msg::String>(
      "/robot_description", qos,
      [this](std_msgs::msg::String::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        if (urdf_xml_.empty()) {
          urdf_xml_ = msg->data;
        }
      });
  }

  std::lock_guard<std::mutex> lock(trajectory_mutex_);
  if (!urdf_xml_.empty()) {
    return true;
  }

  try {
    if (node->has_parameter("robot_description")) {
      auto parameter = node->get_parameter("robot_description");
      if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
        urdf_xml_ = parameter.as_string();
      }
    }
  } catch (const rclcpp::exceptions::ParameterUninitializedException &) {
  } catch (const std::exception &) {
  }
  if (urdf_xml_.empty()) {
    std::ifstream ifs("/tmp/kortex.urdf");
    if (ifs) {
      std::stringstream buffer;
      buffer << ifs.rdbuf();
      urdf_xml_ = buffer.str();
    }
  }
  return !urdf_xml_.empty();
}

void CuroboPlanningDisplay::loadRobotModel()
{
  if (robot_loaded_) {
    return;
  }
  if (!loadURDF()) {
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "Robot Model",
      "No URDF yet on /robot_description.");
    return;
  }

  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    urdf_model_ = urdf::parseURDF(urdf_xml_);
  }
  if (!urdf_model_) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "Robot Model",
      "Failed to parse URDF.");
    urdf_xml_.clear();
    return;
  }

  for (auto & link_entry : urdf_model_->links_) {
    auto & link = link_entry.second;
    auto materialise = [](const urdf::VisualSharedPtr & visual) {
        if (!visual) {
          return;
        }
        if (!visual->material) {
          visual->material = std::make_shared<urdf::Material>();
        }
        urdf::Material & material = *visual->material;
        const bool empty = material.texture_filename.empty() &&
          (material.color.a <= 0.001f ||
          (material.color.r == 0.0f && material.color.g == 0.0f &&
          material.color.b == 0.0f));
        if (empty) {
          material.color.r = 0.60f;
          material.color.g = 0.60f;
          material.color.b = 0.65f;
          material.color.a = 1.0f;
        }
      };
    for (auto & visual : link->visual_array) {
      materialise(visual);
    }
    materialise(link->visual);
  }

  const auto root_link = urdf_model_->getRoot();
  robot_root_frame_ = root_link ? root_link->name : "";

  scene_robot_root_ = scene_node_->createChildSceneNode();
  scene_robot_ = std::make_unique<rviz_default_plugins::robot::Robot>(
    scene_robot_root_, context_, "Scene Robot", nullptr);
  scene_robot_->load(*urdf_model_, true, false, false, false);
  scene_robot_->setVisualVisible(true);
  scene_robot_->setCollisionVisible(false);
  scene_robot_->setAlpha(scene_robot_alpha_property_->getFloat());
  scene_robot_->setVisible(false);

  goal_root_ = scene_node_->createChildSceneNode();
  goal_robot_ = std::make_unique<rviz_default_plugins::robot::Robot>(
    goal_root_, context_, "Goal State", nullptr);
  goal_robot_->load(*urdf_model_, true, false, false, false);
  goal_robot_->setVisualVisible(true);
  goal_robot_->setCollisionVisible(false);
  for (auto & link : goal_robot_->getLinks()) {
    goal_robot_->getLink(link.first)->setColor(0.2f, 0.8f, 0.2f);
  }
  goal_robot_->setAlpha(goal_state_alpha_property_->getFloat());
  goal_robot_->setVisible(false);

  trajectory_root_ = scene_node_->createChildSceneNode();
  trajectory_robot_ = std::make_unique<rviz_default_plugins::robot::Robot>(
    trajectory_root_, context_, "cuRobo Robot", nullptr);
  trajectory_robot_->load(*urdf_model_, true, false, false, false);
  trajectory_robot_->setVisualVisible(true);
  trajectory_robot_->setCollisionVisible(false);
  trajectory_robot_->setAlpha(alpha_property_->getFloat());
  trajectory_robot_->setVisible(false);

  robot_loaded_ = true;
  setStatus(
    rviz_common::properties::StatusProperty::Ok, "Robot Model",
    QString("Loaded URDF with %1 links, %2 joints.")
    .arg(urdf_model_->links_.size())
    .arg(urdf_model_->joints_.size()));
}

void CuroboPlanningDisplay::updateSceneRobot()
{
  if (!scene_robot_) {
    return;
  }
  const bool show = show_scene_robot_property_ &&
    show_scene_robot_property_->getBool();
  scene_robot_->setVisible(show);
  if (!show) {
    return;
  }
  scene_robot_->update(TfLinkUpdater(context_->getFrameManager()));
}

void CuroboPlanningDisplay::updateSceneRobotVisibility()
{
  if (scene_robot_) {
    scene_robot_->setVisible(
      show_scene_robot_property_ && show_scene_robot_property_->getBool());
  }
}

void CuroboPlanningDisplay::updateSceneRobotAlpha()
{
  if (scene_robot_) {
    scene_robot_->setAlpha(scene_robot_alpha_property_->getFloat());
  }
}

void CuroboPlanningDisplay::updateGoalStateVisibility()
{
  if (goal_robot_) {
    goal_robot_->setVisible(
      goal_valid_ && show_goal_state_property_ &&
      show_goal_state_property_->getBool());
  }
  goal_attempted_ = false;
}

void CuroboPlanningDisplay::updateGoalStateAlpha()
{
  if (goal_robot_) {
    goal_robot_->setAlpha(goal_state_alpha_property_->getFloat());
  }
}

void CuroboPlanningDisplay::updateTrajectoryRoot()
{
  Ogre::Vector3 offset = Ogre::Vector3::ZERO;
  Ogre::Quaternion rotation = Ogre::Quaternion::IDENTITY;
  auto * frame_manager = context_ ? context_->getFrameManager() : nullptr;
  if (!robot_root_frame_.empty() && frame_manager != nullptr &&
    robot_root_frame_ != frame_manager->getFixedFrame())
  {
    rclcpp::Time stamp(0, 0, RCL_ROS_TIME);
    if (frame_manager->getTransform(robot_root_frame_, stamp, offset, rotation)) {
      setTransformOk();
    } else {
      setMissingTransformToFixedFrame(robot_root_frame_);
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
// Scene robot helpers
// ============================================================================

// (scene robot is driven by TfLinkUpdater in updateSceneRobot)

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
}

void CuroboPlanningDisplay::updatePlannerNode()
{
  planner_node_ = planner_node_property_->getStdString();
  createFkClient();
  resetFkState();
  resetGoalState();
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

  const size_t points = fk_pending_msg_->points.size();
  const size_t links = response->link_names.size();
  if (links == 0 || response->poses.size() < points * links) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "FK",
      "fk_batch returned no link poses.");
    fk_pending_msg_.reset();
    return false;
  }

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
  deleteStatus("FK");
  return true;
}

// ============================================================================
// Goal-state ghost
// ============================================================================

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
  if (robot_root_frame_.empty() || context_ == nullptr ||
    context_->getFrameManager() == nullptr)
  {
    return true;
  }
  auto * frame_manager = context_->getFrameManager();
  if (robot_root_frame_ == frame_manager->getFixedFrame()) {
    return true;
  }
  Ogre::Vector3 base_position;
  Ogre::Quaternion base_rotation;
  if (!frame_manager->getTransform(
      robot_root_frame_, rclcpp::Time(0, 0, RCL_ROS_TIME), base_position,
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

  const auto pose = activePose();
  const auto now = std::chrono::steady_clock::now();
  const bool same_pose = goal_attempted_ && posesEqual(pose, last_goal_pose_);
  if (same_pose) {
    // A solved goal is stable; a failed one is retried slowly (the IK solver
    // may still be warming up, or the scene may have changed).
    if (goal_valid_ || now - last_goal_request_ < std::chrono::seconds(1)) {
      return;
    }
  } else if (now - last_goal_request_ < std::chrono::milliseconds(200)) {
    return;
  }
  if (!ik_client_ || !ik_client_->service_is_ready()) {
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "Goal State",
      QString("Waiting for /%1/ik_batch.").arg(
        QString::fromStdString(planner_node_)));
    return;
  }

  geometry_msgs::msg::Pose base_pose;
  if (!goalPoseInBase(pose, base_pose)) {
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "Goal State",
      "Waiting for the robot base transform.");
    return;
  }

  auto request = std::make_shared<curobov2_ros_interfaces::srv::IkBatch::Request>();
  request->poses.push_back(base_pose);
  request->frame_id = "";
  request->tool_frame = active_tool_;

  last_goal_pose_ = pose;
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

  auto request = std::make_shared<curobov2_ros_interfaces::srv::FkBatch::Request>();
  request->joint_states.push_back(result->joint_state);
  if (urdf_model_) {
    for (const auto & link : urdf_model_->links_) {
      request->link_names.push_back(link.first);
    }
  }

  goal_stage_ = GoalStage::WaitingFk;
  goal_stage_start_ = std::chrono::steady_clock::now();
  goal_fk_client_->async_send_request(
    request,
    [this, seq = goal_inflight_seq_](
      rclcpp::Client<curobov2_ros_interfaces::srv::FkBatch>::SharedFuture future) {
      auto result = std::make_shared<GoalFkResult>();
      result->seq = seq;
      try {
        auto response = future.get();
        result->ok = response->success;
        result->error = response->error_msg.data;
        result->link_names = response->link_names;
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
  if (goal_robot_) {
    CuroboLinkUpdater updater(transforms);
    goal_robot_->update(updater);
    goal_valid_ = true;
    goal_robot_->setVisible(
      show_goal_state_property_ && show_goal_state_property_->getBool());
    deleteStatus("Goal State");
  }
}

// ============================================================================
// Collision spheres
// ============================================================================

void CuroboPlanningDisplay::updateCollisionTopic()
{
  spheres_subscription_.reset();
  {
    std::lock_guard<std::mutex> lock(spheres_mutex_);
    spheres_msg_new_.reset();
  }
  spheres_got_message_ = false;
  sphere_active_count_ = 0;
  live_spheres_.clear();
  playback_spheres_.clear();
  for (auto & sv : sphere_pool_) {
    if (sv.node) {
      sv.node->setVisible(false);
    }
  }

  if (!collision_topic_property_->getStdString().empty()) {
    auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
    if (!ros_node_abstraction) {
      return;
    }
    spheres_subscription_ = ros_node_abstraction->get_raw_node()
      ->create_subscription<visualization_msgs::msg::MarkerArray>(
      collision_topic_property_->getStdString(), 10,
      std::bind(&CuroboPlanningDisplay::onCollisionSpheresMessage, this,
        std::placeholders::_1));
  }
}

void CuroboPlanningDisplay::onCollisionSpheresMessage(
  visualization_msgs::msg::MarkerArray::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(spheres_mutex_);
  spheres_msg_new_ = msg;
}

void CuroboPlanningDisplay::updateShowAttached()
{
  updateSpheresVisibility();
}

void CuroboPlanningDisplay::updateShowCollisionSpheres()
{
  updateSpheresVisibility();
}

void CuroboPlanningDisplay::updateSpheresVisibility()
{
  const bool show_attached = show_attached_property_->getBool();
  const bool show_collision = show_collision_spheres_property_->getBool();
  const size_t active = std::min(sphere_active_count_, sphere_pool_.size());
  for (size_t i = 0; i < active; ++i) {
    if (sphere_pool_[i].node) {
      sphere_pool_[i].node->setVisible(
        sphere_pool_[i].attached ? show_attached : show_collision);
    }
  }
  for (size_t i = active; i < sphere_pool_.size(); ++i) {
    if (sphere_pool_[i].node) {
      sphere_pool_[i].node->setVisible(false);
    }
  }
}

void CuroboPlanningDisplay::applySpheres(const std::vector<SphereDatum> & data)
{
  while (sphere_pool_.size() < data.size()) {
    SphereVisual sv;
    sv.node = scene_node_->createChildSceneNode();
    sv.shape = std::make_shared<rviz_rendering::Shape>(
      rviz_rendering::Shape::Sphere, scene_manager_, sv.node);
    sv.node->setVisible(false);
    sphere_pool_.push_back(std::move(sv));
  }
  for (size_t i = 0; i < data.size(); ++i) {
    SphereVisual & sv = sphere_pool_[i];
    const SphereDatum & datum = data[i];
    sv.shape->setScale(Ogre::Vector3(datum.sx, datum.sy, datum.sz));
    sv.shape->setColor(datum.r, datum.g, datum.b, datum.a);
    sv.attached = datum.attached;
  }
  for (size_t i = data.size(); i < sphere_pool_.size(); ++i) {
    if (sphere_pool_[i].node) {
      sphere_pool_[i].node->setVisible(false);
    }
  }
  sphere_active_count_ = data.size();
  updateSpheresVisibility();
}

void CuroboPlanningDisplay::syncSpheres()
{
  visualization_msgs::msg::MarkerArray::ConstSharedPtr msg;
  {
    std::lock_guard<std::mutex> lock(spheres_mutex_);
    if (!spheres_msg_new_) {
      return;
    }
    msg = spheres_msg_new_;
    spheres_msg_new_.reset();
  }
  spheres_got_message_ = true;
  deleteStatus("Collision Spheres");

  size_t want = 0;
  for (const auto & marker : msg->markers) {
    if (marker.action == visualization_msgs::msg::Marker::ADD &&
      marker.type == visualization_msgs::msg::Marker::SPHERE)
    {
      ++want;
    }
  }
  while (sphere_pool_.size() < want) {
    SphereVisual sv;
    sv.node = scene_node_->createChildSceneNode();
    sv.shape = std::make_shared<rviz_rendering::Shape>(
      rviz_rendering::Shape::Sphere, scene_manager_, sv.node);
    sv.node->setVisible(false);
    sphere_pool_.push_back(std::move(sv));
  }

  std::vector<SphereDatum> live;
  for (const auto & marker : msg->markers) {
    const bool attached = marker.ns == "attached_spheres";
    const bool collision = marker.ns == "collision_spheres";
    if (marker.action != visualization_msgs::msg::Marker::ADD ||
      marker.type != visualization_msgs::msg::Marker::SPHERE ||
      (!attached && !collision))
    {
      continue;
    }
    SphereDatum datum;
    datum.local = Ogre::Vector3(
      static_cast<float>(marker.pose.position.x),
      static_cast<float>(marker.pose.position.y),
      static_cast<float>(marker.pose.position.z));
    datum.frame = marker.header.frame_id;
    datum.sx = static_cast<float>(marker.scale.x);
    datum.sy = static_cast<float>(marker.scale.y);
    datum.sz = static_cast<float>(marker.scale.z);
    datum.r = marker.color.r;
    datum.g = marker.color.g;
    datum.b = marker.color.b;
    datum.a = marker.color.a;
    datum.attached = attached;
    live.push_back(datum);
  }

  live_spheres_ = live;
  const bool has_attached = std::any_of(
    live_spheres_.begin(), live_spheres_.end(),
    [](const SphereDatum & datum) {return datum.attached;});
  if (!animating_ || has_attached) {
    playback_spheres_ = live_spheres_;
  }
  applySpheres(playback_spheres_);
  poseSpheres();
}

void CuroboPlanningDisplay::poseSpheres()
{
  const size_t active = std::min(sphere_active_count_, sphere_pool_.size());
  Ogre::Vector3 offset = Ogre::Vector3::ZERO;
  Ogre::Quaternion rotation = Ogre::Quaternion::IDENTITY;
  std::string resolved_frame;
  for (size_t i = 0; i < active; ++i) {
    SphereVisual & sv = sphere_pool_[i];
    if (!sv.node || i >= playback_spheres_.size()) {
      continue;
    }
    const SphereDatum & datum = playback_spheres_[i];
    // Only attached spheres are link-relative and ride the trajectory ghost.
    // Robot collision spheres are already expressed in the live base frame, so
    // they must follow TF even while a plan is displayed.
    const auto fk = datum.attached ? current_transforms_.find(datum.frame) :
      current_transforms_.end();
    if (fk != current_transforms_.end()) {
      const Eigen::Vector3d point = fk->second * Eigen::Vector3d(
        datum.local.x, datum.local.y, datum.local.z);
      sv.node->setPosition(robot_root_offset_ + robot_root_rotation_ *
        Ogre::Vector3(
          static_cast<float>(point.x()), static_cast<float>(point.y()),
          static_cast<float>(point.z())));
      continue;
    }
    auto * frame_manager = context_ ? context_->getFrameManager() : nullptr;
    if (datum.frame.empty() || frame_manager == nullptr ||
      datum.frame == frame_manager->getFixedFrame())
    {
      sv.node->setPosition(datum.local);
      continue;
    }
    if (datum.frame != resolved_frame) {
      rclcpp::Time stamp(0, 0, RCL_ROS_TIME);
      if (!frame_manager->getTransform(datum.frame, stamp, offset, rotation)) {
        setMissingTransformToFixedFrame(datum.frame);
        sv.node->setVisible(false);
        continue;
      }
      setTransformOk();
      resolved_frame = datum.frame;
    }
    sv.node->setPosition(rotation * datum.local + offset);
  }
}

// ============================================================================
// Trail
// ============================================================================

void CuroboPlanningDisplay::rebuildTrail()
{
  std::vector<int> wanted;
  if (show_trail_property_->getBool() && trajectory_robot_ && !waypoints_.empty()) {
    const int step = std::max(1, trail_step_size_property_->getInt());
    const int count = static_cast<int>(waypoints_.size());
    for (int i = 0; i < count; i += step) {
      wanted.push_back(i);
    }
  }

  const float alpha = alpha_property_->getFloat() * 0.4f;
  while (trail_robots_.size() < wanted.size()) {
    TrailRobot trail;
    trail.node = scene_node_->createChildSceneNode();
    trail.robot = std::make_unique<rviz_default_plugins::robot::Robot>(
      trail.node, context_, "Trail", nullptr);
    trail.robot->load(*urdf_model_, true, false, false, false);
    trail.robot->setVisualVisible(true);
    trail.robot->setCollisionVisible(false);
    trail.robot->setAlpha(alpha);
    trail.robot->setVisible(false);
    trail_robots_.push_back(std::move(trail));
  }

  for (size_t i = 0; i < trail_robots_.size(); ++i) {
    TrailRobot & trail = trail_robots_[i];
    if (i < wanted.size()) {
      trail.waypoint_index = wanted[i];
      CuroboLinkUpdater updater(waypoints_[wanted[i]].link_transforms);
      trail.robot->update(updater);
      trail.robot->setAlpha(alpha);
    } else {
      trail.waypoint_index = -1;
      trail.robot->setVisible(false);
    }
  }
  updateTrailVisibility();
}

void CuroboPlanningDisplay::clearTrailRobots()
{
  for (auto & trail : trail_robots_) {
    trail.robot.reset();
    if (trail.node) {
      scene_node_->removeAndDestroyChild(trail.node);
      trail.node = nullptr;
    }
  }
  trail_robots_.clear();
}

void CuroboPlanningDisplay::updateTrailVisibility()
{
  const bool show = show_trail_property_->getBool() && animating_;
  for (auto & trail : trail_robots_) {
    trail.robot->setVisible(show && trail.waypoint_index <= current_state_);
  }
}

// ============================================================================
// Targets
// ============================================================================

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
  applying_active_tool_ = true;
  active_tool_property_->setStdString(tool);
  applying_active_tool_ = false;
  updatePoseProperties(activePose());
}

std::vector<std::string> CuroboPlanningDisplay::getToolFrames() const
{
  return tool_order_;
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
  tools_ = std::move(updated);
  tool_order_ = std::move(order);
  if (std::find(tool_order_.begin(), tool_order_.end(), active_tool_) ==
    tool_order_.end())
  {
    active_tool_ = tool_order_.front();
  }

  applying_tool_frames_ = true;
  std::string joined;
  for (const auto & frame : tool_order_) {
    if (frame.empty()) {
      continue;
    }
    if (!joined.empty()) {
      joined += ", ";
    }
    joined += frame;
  }
  tool_frames_property_->setStdString(joined);
  applying_tool_frames_ = false;

  applying_active_tool_ = true;
  active_tool_property_->clearOptions();
  for (const auto & frame : tool_order_) {
    if (!frame.empty()) {
      active_tool_property_->addOptionStd(frame);
    }
  }
  active_tool_property_->setStdString(active_tool_);
  applying_active_tool_ = false;

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
    poses.emplace_back(frame, tools_.at(frame).pose);
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

void CuroboPlanningDisplay::updateToolFrames()
{
  if (applying_tool_frames_) {
    return;
  }
  tool_frames_user_set_ = true;
  setToolFrames(splitToolFrames(tool_frames_property_->getStdString()));
}

void CuroboPlanningDisplay::updateActiveTool()
{
  if (applying_active_tool_) {
    return;
  }
  setActiveTool(active_tool_property_->getStdString());
}

// ============================================================================
// Interactive markers
// ============================================================================

std::string CuroboPlanningDisplay::markerName(const std::string & tool) const
{
  return "EE:" + gizmo_name_ + "_" + (tool.empty() ? "target" : tool);
}

void CuroboPlanningDisplay::createGizmo()
{
  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (!ros_node_abstraction) {
    return;
  }
  auto node = ros_node_abstraction->get_raw_node();

  if (gizmo_frame_.empty()) {
    gizmo_frame_ = context_->getFrameManager()->getFixedFrame();
    if (gizmo_frame_.empty()) {
      gizmo_frame_ = "base_link";
    }
  }

  if (!gizmo_server_) {
    gizmo_server_ = std::make_shared<interactive_markers::InteractiveMarkerServer>(
      gizmo_name_, node.get());
    RCLCPP_INFO(
      node->get_logger(), "Target markers on /%s/* (%zu tool frame(s)).",
      gizmo_name_.c_str(), tool_order_.size());
  }
  rebuildGizmoMarkers();
  gizmo_server_->applyChanges();
  connectGizmoClient();
}

void CuroboPlanningDisplay::destroyGizmo()
{
  disconnectGizmoClient();
  if (gizmo_server_) {
    gizmo_server_->clear();
    gizmo_server_->applyChanges();
  }
  gizmo_server_.reset();
}

void CuroboPlanningDisplay::rebuildGizmoMarkers()
{
  if (!gizmo_server_) {
    return;
  }
  gizmo_server_->clear();
  for (const auto & frame : tool_order_) {
    insertGizmoMarker(frame);
  }
  gizmo_server_->applyChanges();
}

void CuroboPlanningDisplay::insertGizmoMarker(const std::string & tool)
{
  if (!gizmo_server_) {
    return;
  }
  auto & target = tools_[tool];
  visualization_msgs::msg::InteractiveMarker marker;
  marker.header.frame_id = gizmo_frame_;
  marker.pose = target.pose;
  marker.scale = gizmo_scale_property_ ? gizmo_scale_property_->getFloat() : 0.2;
  marker.name = markerName(tool);
  marker.description = tool.empty() ? "Target" : tool;

  static const float palette[][3] = {
    {0.10f, 0.75f, 0.75f},
    {0.90f, 0.55f, 0.10f},
    {0.55f, 0.35f, 0.90f},
    {0.25f, 0.75f, 0.25f},
    {0.90f, 0.25f, 0.45f},
    {0.70f, 0.70f, 0.20f},
  };
  size_t index = 0;
  for (size_t i = 0; i < tool_order_.size(); ++i) {
    if (tool_order_[i] == tool) {
      index = i;
      break;
    }
  }
  const auto & colour = palette[index % (sizeof(palette) / sizeof(palette[0]))];

  visualization_msgs::msg::Marker sphere;
  sphere.type = visualization_msgs::msg::Marker::SPHERE;
  sphere.pose.orientation.w = 1.0;
  sphere.scale.x = 0.025;
  sphere.scale.y = 0.025;
  sphere.scale.z = 0.025;
  sphere.color.r = colour[0];
  sphere.color.g = colour[1];
  sphere.color.b = colour[2];
  sphere.color.a = 0.85f;

  visualization_msgs::msg::Marker axis;
  axis.type = visualization_msgs::msg::Marker::ARROW;
  axis.pose.orientation.w = 1.0;
  axis.scale.x = 0.08;
  axis.scale.y = 0.015;
  axis.scale.z = 0.015;
  axis.color.r = colour[0];
  axis.color.g = colour[1];
  axis.color.b = colour[2];
  axis.color.a = 1.0f;

  visualization_msgs::msg::InteractiveMarkerControl body;
  body.always_visible = true;
  body.markers.push_back(sphere);
  body.markers.push_back(axis);
  marker.controls.push_back(body);

  static constexpr double kSqrt2Inv = 0.7071067811865476;
  auto add_axis_control = [&marker](double x, double y, double z, uint8_t mode) {
      visualization_msgs::msg::InteractiveMarkerControl control;
      control.orientation.w = kSqrt2Inv;
      control.orientation.x = x * kSqrt2Inv;
      control.orientation.y = y * kSqrt2Inv;
      control.orientation.z = z * kSqrt2Inv;
      control.interaction_mode = mode;
      marker.controls.push_back(control);
    };
  using Control = visualization_msgs::msg::InteractiveMarkerControl;
  add_axis_control(1, 0, 0, Control::ROTATE_AXIS);
  add_axis_control(0, 1, 0, Control::ROTATE_AXIS);
  add_axis_control(0, 0, 1, Control::ROTATE_AXIS);
  add_axis_control(1, 0, 0, Control::MOVE_AXIS);
  add_axis_control(0, 1, 0, Control::MOVE_AXIS);
  add_axis_control(0, 0, 1, Control::MOVE_AXIS);

  gizmo_server_->insert(
    marker,
    std::bind(&CuroboPlanningDisplay::gizmoFeedback, this, tool,
      std::placeholders::_1));
}

void CuroboPlanningDisplay::gizmoFeedback(
  const std::string & tool,
  const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr & feedback)
{
  if (!feedback ||
    feedback->event_type != visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE)
  {
    return;
  }
  std::lock_guard<std::mutex> lock(gizmo_mutex_);
  feedback_poses_[tool] = feedback->pose;
  feedback_dirty_.insert(tool);
}

void CuroboPlanningDisplay::syncGizmoToProperties()
{
  std::map<std::string, geometry_msgs::msg::Pose> poses;
  {
    std::lock_guard<std::mutex> lock(gizmo_mutex_);
    if (feedback_dirty_.empty()) {
      return;
    }
    for (const auto & tool : feedback_dirty_) {
      auto it = feedback_poses_.find(tool);
      if (it != feedback_poses_.end()) {
        poses.emplace(tool, it->second);
      }
    }
    feedback_dirty_.clear();
  }

  std::string last_dragged;
  for (const auto & entry : poses) {
    ensureTarget(entry.first);
    tools_[entry.first].pose = entry.second;
    tools_[entry.first].tf_seeded = true;
    last_dragged = entry.first;
  }
  if (!last_dragged.empty() && last_dragged != active_tool_) {
    active_tool_ = last_dragged;
    applying_active_tool_ = true;
    active_tool_property_->setStdString(active_tool_);
    applying_active_tool_ = false;
  }
  updatePoseProperties(activePose());
}

void CuroboPlanningDisplay::syncPropertiesToGizmo()
{
  if (!gizmo_server_ || sync_pending_.empty()) {
    sync_pending_.clear();
    return;
  }
  for (const auto & tool : sync_pending_) {
    const std::string name = markerName(tool);
    gizmo_server_->setPose(name, tools_[tool].pose);
  }
  sync_pending_.clear();
  gizmo_server_->applyChanges();
}

void CuroboPlanningDisplay::connectGizmoClient()
{
  if (!gizmo_client_ || gizmo_client_connected_) {
    return;
  }
  gizmo_client_->connect(gizmo_name_);
  gizmo_client_connected_ = true;
}

void CuroboPlanningDisplay::disconnectGizmoClient()
{
  if (gizmo_client_) {
    gizmo_client_->disconnect();
  }
  gizmo_client_connected_ = false;
  eraseAllGizmoMarkers();
}

void CuroboPlanningDisplay::gizmoInitializeCallback(
  visualization_msgs::srv::GetInteractiveMarkers::Response::SharedPtr msg)
{
  eraseAllGizmoMarkers();
  if (msg) {
    updateGizmoMarkers(msg->markers);
  }
}

void CuroboPlanningDisplay::gizmoUpdateCallback(
  visualization_msgs::msg::InteractiveMarkerUpdate::ConstSharedPtr msg)
{
  if (!msg) {
    return;
  }
  updateGizmoMarkers(msg->markers);
  updateGizmoPoses(msg->poses);
  for (const std::string & name : msg->erases) {
    interactive_markers_map_.erase(name);
  }
}

void CuroboPlanningDisplay::gizmoResetCallback()
{
  eraseAllGizmoMarkers();
}

void CuroboPlanningDisplay::gizmoStatusCallback(
  interactive_markers::InteractiveMarkerClient::Status status,
  const std::string & message)
{
  rviz_common::properties::StatusProperty::Level level =
    rviz_common::properties::StatusProperty::Ok;
  switch (status) {
    case interactive_markers::InteractiveMarkerClient::STATUS_WARN:
      level = rviz_common::properties::StatusProperty::Warn;
      break;
    case interactive_markers::InteractiveMarkerClient::STATUS_ERROR:
      level = rviz_common::properties::StatusProperty::Error;
      break;
    default:
      break;
  }
  setStatusStd(level, "Gizmo", message);
}

void CuroboPlanningDisplay::updateGizmoMarkers(
  const std::vector<visualization_msgs::msg::InteractiveMarker> & markers)
{
  for (const auto & marker : markers) {
    if (!validateFloats(marker)) {
      setStatusStd(
        rviz_common::properties::StatusProperty::Error, marker.name,
        "Marker contains invalid floats!");
      continue;
    }

    auto entry = interactive_markers_map_.find(marker.name);
    if (entry == interactive_markers_map_.end()) {
      auto scene_marker =
        std::make_shared<rviz_default_plugins::displays::InteractiveMarker>(
          getSceneNode(), context_);
      connect(
        scene_marker.get(),
        SIGNAL(userFeedback(visualization_msgs::msg::InteractiveMarkerFeedback&)),
        this,
        SLOT(publishGizmoFeedback(visualization_msgs::msg::InteractiveMarkerFeedback&)));
      connect(
        scene_marker.get(),
        SIGNAL(statusUpdate(
          rviz_common::properties::StatusProperty::Level,
          const std::string&,
          const std::string&)),
        this,
        SLOT(gizmoStatusUpdate(
          rviz_common::properties::StatusProperty::Level,
          const std::string&,
          const std::string&)));
      entry = interactive_markers_map_.emplace(marker.name, scene_marker).first;
    }

    if (entry->second->processMessage(marker)) {
      entry->second->setShowDescription(false);
      entry->second->setShowAxes(false);
      entry->second->setShowVisualAids(false);
    } else {
      setStatusStd(
        rviz_common::properties::StatusProperty::Error, marker.name,
        "Failed to process interactive marker.");
    }
  }
}

void CuroboPlanningDisplay::updateGizmoPoses(
  const std::vector<visualization_msgs::msg::InteractiveMarkerPose> & poses)
{
  for (const auto & pose : poses) {
    auto entry = interactive_markers_map_.find(pose.name);
    if (entry != interactive_markers_map_.end()) {
      entry->second->processMessage(pose);
    }
  }
}

void CuroboPlanningDisplay::eraseAllGizmoMarkers()
{
  interactive_markers_map_.clear();
}

void CuroboPlanningDisplay::publishGizmoFeedback(
  visualization_msgs::msg::InteractiveMarkerFeedback & feedback)
{
  if (gizmo_client_) {
    gizmo_client_->publishFeedback(feedback);
  }
}

void CuroboPlanningDisplay::gizmoStatusUpdate(
  rviz_common::properties::StatusProperty::Level level,
  const std::string & name,
  const std::string & text)
{
  setStatusStd(level, name, text);
}

void CuroboPlanningDisplay::updateGizmoVisible()
{
  const bool want = gizmo_visible_property_ && gizmo_visible_property_->getBool();
  if (want && !gizmo_active_) {
    createGizmo();
    gizmo_active_ = true;
  } else if (!want && gizmo_active_) {
    destroyGizmo();
    gizmo_active_ = false;
  }
}

void CuroboPlanningDisplay::updateGizmoScale()
{
  if (gizmo_active_) {
    rebuildGizmoMarkers();
  }
}

// ============================================================================
// Playback panel
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

  if (!robot_loaded_) {
    loadRobotModel();
  }
  if (!robot_loaded_) {
    return;
  }

  updateSceneRobot();
  updateTrajectoryRoot();
  seedPendingTargets();

  // Goal-state ghost: IK the active target, then FK the solution.
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

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  curobov2_ros_rviz::CuroboPlanningDisplay, rviz_common::Display)
