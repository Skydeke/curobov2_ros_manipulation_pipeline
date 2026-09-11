#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>

#include <OgreVector3.h>
#include <OgreQuaternion.h>
#include <OgreSceneNode.h>

#include <rclcpp/time.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/logging.hpp>
#include <rviz_common/properties/parse_color.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

#include <isaac_ros_cumotion_rviz/curobo_trajectory_display.hpp>
#include <isaac_ros_cumotion_rviz/curobo_link_updater.hpp>

namespace isaac_ros_cumotion_rviz
{

// ============================================================================
// Construction / destruction
// ============================================================================

CuroboTrajectoryDisplay::CuroboTrajectoryDisplay()
  : robot_(nullptr)
  , robot_loaded_(false)
  , topic_property_(nullptr)
  , alpha_property_(nullptr)
  , show_trail_property_(nullptr)
  , trail_step_size_property_(nullptr)
  , loop_property_(nullptr)
  , speed_property_(nullptr)
{
}

CuroboTrajectoryDisplay::~CuroboTrajectoryDisplay() = default;

// ============================================================================
// Display lifecycle
// ============================================================================

void CuroboTrajectoryDisplay::onInitialize()
{
  Display::onInitialize();

  // --- Display properties ---
  topic_property_ = new rviz_common::properties::RosTopicProperty(
    "Trajectory Topic", "",
    "trajectory_msgs/msg/JointTrajectory",
    "trajectory_msgs/JointTrajectory topic to display.", this,
    SLOT(updateTopic()));

  // Required for the topic dropdown: without this the weak_ptr inside
  // RosTopicProperty stays empty and opening the dropdown segfaults
  // (fillTopicList() dereferences a null lock()).
  topic_property_->initialize(context_->getRosNodeAbstraction());

  alpha_property_ = new rviz_common::properties::FloatProperty(
    "Alpha", 1.0, "Robot opacity (0 = invisible, 1 = opaque).", this,
    SLOT(updateAlpha()));
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
    "Loop Animation", true, "Restart from beginning when finished.", this,
    SLOT(updateLoop()));

  speed_property_ = new rviz_common::properties::FloatProperty(
    "Speed", 1.0, "Playback speed multiplier (0.5 = half speed, 2.0 = double).", this,
    SLOT(updateSpeed()));
  speed_property_->setMin(0.01);
  speed_property_->setMax(100.0);

  updateTopic();
}

void CuroboTrajectoryDisplay::reset()
{
  Display::reset();
  std::lock_guard<std::mutex> lock(trajectory_mutex_);
  trajectory_msg_new_.reset();
  trajectory_msg_active_.reset();
  waypoints_.clear();
  animating_ = false;
  current_state_ = 0;
  current_frac_ = 0.0;
  anim_time_ = 0.0;
  current_transforms_.clear();
  trail_robots_.clear();
  if (robot_) {
    robot_->setVisible(false);
  }
}

// ============================================================================
// URDF loading (always from `/robot_description`)
// ============================================================================

bool CuroboTrajectoryDisplay::loadURDF()
{
  // Fast path: already loaded (check under mutex — subscription callback
  // writes urdf_xml_ from a different thread).
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    if (!urdf_xml_.empty()) {return true;}
  }

  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (!ros_node_abstraction) {return false;}
  auto node = ros_node_abstraction->get_raw_node();

  // Subscribe to the latched `/robot_description` topic (the convention used
  // by robot_state_publisher and kortex.rviz).  The URDF arrives
  // asynchronously in the callback; loadURDF returns false this frame and the
  // next update() frame retries.
  if (!robot_description_sub_) {
    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.transient_local();
    robot_description_sub_ = node->create_subscription<std_msgs::msg::String>(
      "/robot_description",
      qos,
      [this](std_msgs::msg::String::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        if (urdf_xml_.empty()) {
          urdf_xml_ = msg->data;
        }
      });
  }

  std::lock_guard<std::mutex> lock(trajectory_mutex_);
  if (!urdf_xml_.empty()) {return true;}

  // Silent fallbacks so the display also works when RViz runs standalone
  // without robot_state_publisher:
  //   1) the rviz node's own `robot_description` parameter (rviz.launch.py),
  //   2) the URDF file that moveit_cumotion.launch.py writes to /tmp.
  if (node->has_parameter("robot_description")) {
    urdf_xml_ = node->get_parameter("robot_description").as_string();
  }
  if (urdf_xml_.empty()) {
    std::ifstream ifs("/tmp/kortex.urdf");
    if (ifs) {
      std::stringstream buf;
      buf << ifs.rdbuf();
      urdf_xml_ = buf.str();
    }
  }
  return !urdf_xml_.empty();
}

// ============================================================================
// Robot model
// ============================================================================

void CuroboTrajectoryDisplay::loadRobotModel()
{
  if (robot_loaded_) {return;}

  if (!loadURDF()) {
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "Robot Model",
      QString("No URDF yet on /robot_description.\n"
        "Reading the latched URDF from the /robot_description topic\n"
        "(robot_state_publisher) or the `robot_description` parameter."));
    return;
  }

  // Parse URDF
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

  // Give every link visual a usable material.  rviz falls back to the red
  // "RVIZ/ShadedRed" material for a visual with no material tag, and renders
  // visuals with an empty (all-zero) colour fully transparent — both make the
  // planned-robot appear as red/black boxes.  Injecting a default grey only
  // affects visuals that lack one; materialised/textured visuals are kept.
  for (auto & link_entry : urdf_model_->links_) {
    auto & link = link_entry.second;
    auto materialise = [](const urdf::VisualSharedPtr & visual) {
      if (!visual) {return;}
      if (!visual->material) {
        visual->material = std::make_shared<urdf::Material>();
      }
      urdf::Material & mat = *visual->material;
      const bool empty_colour =
        mat.texture_filename.empty() &&
        (mat.color.a <= 0.001f ||
         (mat.color.r == 0.0f && mat.color.g == 0.0f && mat.color.b == 0.0f));
      if (empty_colour) {
        mat.color.r = 0.60f;
        mat.color.g = 0.60f;
        mat.color.b = 0.65f;
        mat.color.a = 1.0f;
      }
    };
    for (auto & visual : link->visual_array) {
      materialise(visual);
    }
    materialise(link->visual);
  }

  // Initialise FK engine
  if (!fk_engine_.initFromString(urdf_xml_)) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "Robot Model",
      "Failed to build FK tree from URDF.");
    urdf_xml_.clear();
    return;
  }

  // Create rviz Robot (4th arg: parent Property*, nullptr — we don't use per-link
  // toggle properties in the RViz tree).
  robot_ = std::make_unique<rviz_default_plugins::robot::Robot>(
    scene_node_, context_, "cuRobo Robot", nullptr);
  robot_->load(*urdf_model_, true, false);  // visual=true, collision=false
  robot_->setVisualVisible(true);
  robot_->setCollisionVisible(false);
  robot_->setAlpha(alpha_property_->getFloat());
  robot_->setVisible(false);

  robot_loaded_ = true;
  setStatus(
    rviz_common::properties::StatusProperty::Ok, "Robot Model",
    QString("Loaded URDF with %1 links, %2 joints.")
      .arg(fk_engine_.getLinkNames().size())
      .arg(fk_engine_.getJointNames().size()));
}

// ============================================================================
// Subscriber
// ============================================================================

void CuroboTrajectoryDisplay::updateTopic()
{
  subscription_.reset();
  trajectory_msg_new_.reset();
  trajectory_msg_active_.reset();
  animating_ = false;
  current_state_ = 0;
  current_frac_ = 0.0;
  anim_time_ = 0.0;
  current_transforms_.clear();
  waypoints_.clear();
  trail_robots_.clear();

  if (!topic_property_->getStdString().empty()) {
    auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
    if (!ros_node_abstraction) {return;}
    auto node = ros_node_abstraction->get_raw_node();
    subscription_ = node->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      topic_property_->getStdString(), 10,
      std::bind(&CuroboTrajectoryDisplay::onTrajectoryMessage, this,
        std::placeholders::_1));
  }
}

void CuroboTrajectoryDisplay::onTrajectoryMessage(
  trajectory_msgs::msg::JointTrajectory::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(trajectory_mutex_);
  trajectory_msg_new_ = msg;
}

// ============================================================================
// Property update slots
// ============================================================================

void CuroboTrajectoryDisplay::updateAlpha()
{
  if (robot_) {
    robot_->setAlpha(alpha_property_->getFloat());
  }
  for (auto & tr : trail_robots_) {
    if (tr.robot) {
      tr.robot->setAlpha(alpha_property_->getFloat() * 0.4f);
    }
  }
}

void CuroboTrajectoryDisplay::updateShowTrail()
{
  updateTrailVisibility();
}

void CuroboTrajectoryDisplay::updateTrailStepSize()
{
  // Rebuild trail when step size changes
  if (!waypoints_.empty()) {
    rebuildTrail();
  }
}

void CuroboTrajectoryDisplay::updateLoop() {}
void CuroboTrajectoryDisplay::updateSpeed() {}

// ============================================================================
// FK: build waypoints from JointTrajectory
// ============================================================================

static std::map<std::string, double> buildJointMap(
  const trajectory_msgs::msg::JointTrajectory & traj,
  size_t point_index)
{
  std::map<std::string, double> result;
  if (point_index >= traj.points.size()) {return result;}
  const auto & pt = traj.points[point_index];
  const size_t n = std::min(traj.joint_names.size(), pt.positions.size());
  for (size_t i = 0; i < n; ++i) {
    result[traj.joint_names[i]] = pt.positions[i];
  }
  return result;
}

// ============================================================================
// Trail
// ============================================================================

void CuroboTrajectoryDisplay::rebuildTrail()
{
  trail_robots_.clear();
  if (!show_trail_property_->getBool() || !robot_ || waypoints_.empty()) {return;}

  const int step = trail_step_size_property_->getInt();
  const int count = static_cast<int>(waypoints_.size());
  const float alpha = alpha_property_->getFloat() * 0.4f;

  for (int i = 0; i < count; i += step) {
    TrailRobot tr;
    tr.robot = std::make_unique<rviz_default_plugins::robot::Robot>(
      scene_node_, context_, "Trail", nullptr);
    tr.robot->load(*urdf_model_, true, false);
    tr.robot->setVisualVisible(true);
    tr.robot->setCollisionVisible(false);
    tr.robot->setAlpha(alpha);
    tr.waypoint_index = i;

    CuroboLinkUpdater updater(waypoints_[i].link_transforms);
    tr.robot->update(updater);
    tr.robot->setVisible(false);

    trail_robots_.push_back(std::move(tr));
  }
}

void CuroboTrajectoryDisplay::updateTrailVisibility()
{
  const bool show = show_trail_property_->getBool() && animating_;
  for (auto & tr : trail_robots_) {
    bool visible = show && (tr.waypoint_index <= current_state_);
    tr.robot->setVisible(visible);
  }
}

// ============================================================================
// Animation update (called every RViz frame)
// ============================================================================

void CuroboTrajectoryDisplay::update(float wall_dt, float ros_dt)
{
  Display::update(wall_dt, ros_dt);

  // Lazy-load robot model
  if (!robot_loaded_) {
    loadRobotModel();
    if (!robot_loaded_) {return;}
  }

  // Promote a pending trajectory whenever one is available — even mid-animation.
  // cuRobo publishes each plan exactly once on /planned_path, so the next plan
  // can arrive while the previous one is still playing.  Without this the
  // display would only ever show the very first plan.
  bool promoted = false;
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    if (trajectory_msg_new_ && !trajectory_msg_new_->points.empty()) {
      trajectory_msg_active_ = trajectory_msg_new_;
      trajectory_msg_new_.reset();

      // Precompute FK for all waypoints
      waypoints_.clear();
      waypoints_.reserve(trajectory_msg_active_->points.size());
      const size_t num_points = trajectory_msg_active_->points.size();
      for (size_t i = 0; i < num_points; ++i) {
        WaypointData wp;
        auto joint_map = buildJointMap(*trajectory_msg_active_, i);
        fk_engine_.computeFK(joint_map, wp.link_transforms);
        const auto & tfs = trajectory_msg_active_->points[i].time_from_start;
        wp.time_from_start = tfs.sec + tfs.nanosec * 1e-9;
        waypoints_.push_back(std::move(wp));
      }

      // Normalise to a strictly increasing timeline in case the publisher
      // emitted all-zero or duplicate time_from_start values; otherwise the
      // segment interpolation below would be ill-defined.
      double prev_t = -1.0;
      for (auto & wp : waypoints_) {
        if (wp.time_from_start <= prev_t) {
          wp.time_from_start = prev_t + 0.025;  // default 40 Hz
        }
        prev_t = wp.time_from_start;
      }

      anim_time_ = 0.0;
      current_state_ = 0;
      current_frac_ = 0.0;
      animating_ = true;
      promoted = true;
    }
  }

  if (waypoints_.empty()) {return;}

  if (promoted) {
    robot_->setVisible(true);
    // Rebuild ghost trail for the new trajectory
    rebuildTrail();
  }

  // Advance playback time, but clamp the per-frame step so a single stalled or
  // slow frame can never fast-forward through the whole trajectory (which
  // would make the robot appear frozen at the final waypoint).  Max 0.1 s of
  // playback per render frame — irrelevant at normal frame rates.
  const double speed = std::max(0.0, static_cast<double>(speed_property_->getFloat()));
  const double total = waypoints_.back().time_from_start;
  anim_time_ += std::min(static_cast<double>(wall_dt), 0.1) * speed;

  if (total > 0.0) {
    if (anim_time_ >= total) {
      if (loop_property_->getBool()) {
        anim_time_ = std::fmod(anim_time_, total);
      } else {
        anim_time_ = total;
        animating_ = false;
      }
    }
  } else {
    // Single-waypoint / degenerate trajectory: show the one pose.
    animating_ = false;
  }

  // Locate the segment containing anim_time_.
  size_t seg = waypoints_.size() - 1;
  double frac = 1.0;
  for (size_t i = 0; i + 1 < waypoints_.size(); ++i) {
    if (anim_time_ < waypoints_[i + 1].time_from_start) {
      seg = i;
      const double seg_dt =
        waypoints_[i + 1].time_from_start - waypoints_[i].time_from_start;
      frac = (seg_dt > 0.0)
        ? (anim_time_ - waypoints_[i].time_from_start) / seg_dt
        : 0.0;
      break;
    }
  }
  current_state_ = static_cast<int>(seg);
  current_frac_ = frac;

  // Interpolate between the segment's waypoints so the motion is smooth even
  // between published waypoints.
  current_transforms_ =
    (seg + 1 < waypoints_.size() && frac > 0.0001)
      ? interpolateWaypoints(seg, seg + 1, frac)
      : waypoints_[seg].link_transforms;

  // Render every frame (not only on waypoint changes): the robot always
  // shows the correct pose for the current playback time, and every link is
  // refreshed each frame so no link is ever left in rviz's error state.
  CuroboLinkUpdater updater(current_transforms_);
  robot_->update(updater);
  updateTrailVisibility();
}

std::map<std::string, Eigen::Isometry3d>
CuroboTrajectoryDisplay::interpolateWaypoints(
  size_t waypoint_a, size_t waypoint_b, double t) const
{
  const auto & A = waypoints_[waypoint_a].link_transforms;
  const auto & B = waypoints_[waypoint_b].link_transforms;
  std::map<std::string, Eigen::Isometry3d> out;
  for (const auto & kv : A) {
    auto it = B.find(kv.first);
    if (it == B.end()) {
      out[kv.first] = kv.second;
      continue;
    }
    const Eigen::Isometry3d & a = kv.second;
    const Eigen::Isometry3d & b = it->second;
    Eigen::Quaterniond qa(a.linear());
    Eigen::Quaterniond qb(b.linear());
    qa.normalize();
    qb.normalize();
    Eigen::Isometry3d m = Eigen::Isometry3d::Identity();
    m.translation() = a.translation() * (1.0 - t) + b.translation() * t;
    m.linear() = qa.slerp(t, qb).toRotationMatrix();
    out[kv.first] = m;
  }
  // Links only present in B (shouldn't normally happen).
  for (const auto & kv : B) {
    if (out.find(kv.first) == out.end()) {
      out[kv.first] = kv.second;
    }
  }
  return out;
}

}  // namespace isaac_ros_cumotion_rviz

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  isaac_ros_cumotion_rviz::CuroboTrajectoryDisplay, rviz_common::Display)
