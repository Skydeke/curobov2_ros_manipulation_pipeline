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
  active_tools_property_ = new rviz_common::properties::StringProperty(
    "Active Tools", "",
    "Comma- or space-separated tool frames whose targets are planned together. "
    "Empty means every configured tool frame. The panel exposes the same set "
    "as a multi-select list; the single tool edited by the pose fields is "
    "chosen there or by dragging its marker.",
    this, SLOT(updateActiveTools()));

  seed_from_tf_property_ = new rviz_common::properties::BoolProperty(
    "Place target at tool", true,
    "Initialise a newly added tool target at that tool frame's current TF "
    "pose. This is a one-shot seed; the target then stays world-anchored.", this);

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

  show_goal_state_property_ = new rviz_common::properties::BoolProperty(
    "Show IK result", true,
    "Render a ghost robot at the IK solution for every tool target.", this,
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

  scene_topic_property_ = new rviz_common::properties::RosTopicProperty(
    "Scene Objects Topic", "/unified_planner/scene_obstacles",
    "visualization_msgs/msg/MarkerArray",
    "Analytic world obstacles (cuboids/spheres/cylinders) to display as exact "
    "primitives. These are collision-checked analytically by the planner and "
    "are not rasterized into the voxel grid.", this,
    SLOT(updateSceneTopic()));
  scene_topic_property_->initialize(context_->getRosNodeAbstraction());

  show_attached_property_ = new rviz_common::properties::BoolProperty(
    "Show Attached", true,
    "Show the grasped object's spheres (the 'attached_spheres' namespace).", this,
    SLOT(updateShowAttached()));
  show_collision_spheres_property_ = new rviz_common::properties::BoolProperty(
    "Show Robot Collision Spheres", false,
    "Show the robot's own collision spheres ('collision_spheres' namespace).",
    this, SLOT(updateShowCollisionSpheres()));
  show_scene_objects_property_ = new rviz_common::properties::BoolProperty(
    "Show Scene Objects", true,
    "Show the analytic world obstacles from the scene_obstacles MarkerArray.",
    this, SLOT(updateShowSceneObjects()));

  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (ros_node_abstraction) {
    auto node = ros_node_abstraction->get_raw_node();
    joint_state_sub_ = node->create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
        onJointState(msg);
      });
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
  // The Scene Objects topic was not present in older saved configs, so derive
  // its default from the configured planner node instead of leaving it pinned
  // to the generic /unified_planner name.
  if (scene_topic_property_ && !planner_node_.empty()) {
    const std::string current = scene_topic_property_->getStdString();
    if (current.empty() || current == "/unified_planner/scene_obstacles") {
      scene_topic_property_->setStdString("/" + planner_node_ + "/scene_obstacles");
    }
  }
  createFkClient();
  createTrajectoryPanel();
  updateTopic();
  updateCollisionTopic();
  updateSceneTopic();

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
  updateTrajectoryRoot();
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
  // fk_base_link_ is a property of the PLANNER, not of the current trajectory
  // or of this display's reset lifetime. Keeping it means the next goal/trajectory
  // is placed correctly immediately instead of falling back to the URDF root
  // until an FK response happens to arrive.
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
  attached_spheres_.clear();
  for (auto & sv : sphere_pool_) {
    if (sv.node) {
      sv.node->setVisible(false);
    }
  }
  if (robot_sphere_mesh_) {
    robot_sphere_mesh_->clear();
  }
  if (robot_sphere_node_) {
    robot_sphere_node_->setVisible(false);
  }
  {
    std::lock_guard<std::mutex> scene_lock(scene_mutex_);
    scene_msg_new_.reset();
  }
  scene_got_message_ = false;
  scene_active_count_ = 0;
  live_scene_objects_.clear();
  for (auto & sv : scene_pool_) {
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
  fk_engine_.initFromString(urdf_xml_);

  for (auto & link_entry : urdf_model_->links_) {
    auto & link = link_entry.second;
    auto materialise = [](const urdf::VisualSharedPtr & visual) {
        if (!visual) {
          return;
        }
        // Only supply a material where the URDF has NONE. Overriding an
        // existing colour (even black) turned legitimate dark parts into the
        // grey/white boxes this display used to show.
        if (!visual->material) {
          visual->material = std::make_shared<urdf::Material>();
          visual->material->color.r = 0.60f;
          visual->material->color.g = 0.60f;
          visual->material->color.b = 0.65f;
          visual->material->color.a = 1.0f;
        }
      };
    for (auto & visual : link->visual_array) {
      materialise(visual);
    }
    materialise(link->visual);
  }

  const auto root_link = urdf_model_->getRoot();
  robot_root_frame_ = root_link ? root_link->name : "";

  // The live robot is intentionally NOT rendered here: RViz's own RobotModel
  // display already does that from TF, and duplicating it only produced two
  // overlapping robots. This display owns the ghost/target overlays.

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
  updateGoalRobotLinks();

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

}  // namespace curobov2_ros_rviz

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  curobov2_ros_rviz::CuroboPlanningDisplay, rviz_common::Display)
