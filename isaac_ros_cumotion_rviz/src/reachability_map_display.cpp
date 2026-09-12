#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>

#include <OgreColourValue.h>
#include <OgreQuaternion.h>
#include <OgreSceneNode.h>
#include <OgreVector3.h>

#include <rclcpp/time.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/logging.hpp>
#include <rviz_common/properties/parse_color.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/validate_floats.hpp>

#include <isaac_ros_cumotion_rviz/curobo_link_updater.hpp>
#include <isaac_ros_cumotion_rviz/reachability_map_display.hpp>

namespace
{

/// All floats finite? Same checks as rviz's InteractiveMarkerDisplay.
bool validateFloats(const visualization_msgs::msg::InteractiveMarker & msg)
{
  bool valid = true;
  valid = valid && rviz_common::validateFloats(msg.pose);
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

}  // namespace

namespace isaac_ros_cumotion_rviz
{

// ============================================================================
// Construction / destruction
// ============================================================================

ReachabilityMapDisplay::ReachabilityMapDisplay()
  : plane_shape_(nullptr)
  , cloud_(nullptr)
  , service_name_property_(nullptr)
  , plane_position_x_(nullptr)
  , plane_position_y_(nullptr)
  , plane_position_z_(nullptr)
  , plane_orientation_x_(nullptr)
  , plane_orientation_y_(nullptr)
  , plane_orientation_z_(nullptr)
  , plane_orientation_w_(nullptr)
  , plane_size_x_(nullptr)
  , plane_size_y_(nullptr)
  , plane_color_property_(nullptr)
  , plane_alpha_property_(nullptr)
  , gizmo_visible_property_(nullptr)
  , grid_cells_x_(nullptr)
  , grid_cells_y_(nullptr)
  , auto_refresh_property_(nullptr)
  , refresh_property_(nullptr)
  , cell_style_property_(nullptr)
  , solved_color_property_(nullptr)
  , failed_color_property_(nullptr)
  , cell_alpha_property_(nullptr)
  , show_solutions_property_(nullptr)
  , selected_cell_property_(nullptr)
  , solution_alpha_property_(nullptr)
{
  const int instance = gizmo_instance_counter_++;
  gizmo_name_ = (instance == 0) ? "reachability_plane" :
    "reachability_plane_" + std::to_string(instance);
}

std::atomic<int> ReachabilityMapDisplay::gizmo_instance_counter_{0};

ReachabilityMapDisplay::~ReachabilityMapDisplay()
{
  destroyGizmo();
}

// ============================================================================
// Display lifecycle
// ============================================================================

void ReachabilityMapDisplay::onInitialize()
{
  Display::onInitialize();

  // --- Solve service ---
  service_name_property_ = new rviz_common::properties::StringProperty(
    "Service Name", "/curobo_server/generate_rm",
    "GenerateRM service that solves the reachability grid.",
    this, SLOT(updateServiceName()));

  // --- Plane geometry ---
  plane_position_x_ = new rviz_common::properties::FloatProperty(
    "Plane Position X", 0.4, "Plane centre X (m).", this, SLOT(updatePlane()));
  plane_position_y_ = new rviz_common::properties::FloatProperty(
    "Plane Position Y", 0.0, "Plane centre Y (m).", this, SLOT(updatePlane()));
  plane_position_z_ = new rviz_common::properties::FloatProperty(
    "Plane Position Z", 0.4, "Plane centre Z (m).", this, SLOT(updatePlane()));

  // MoveIt-style pose entry: position + quaternion (X, Y, Z, W), the same
  // convention as the Goal pose tab in MotionPlanning.
  plane_orientation_x_ = new rviz_common::properties::FloatProperty(
    "Plane Orientation X", 0.0, "Plane quaternion X.", this, SLOT(updatePlane()));
  plane_orientation_y_ = new rviz_common::properties::FloatProperty(
    "Plane Orientation Y", 0.0, "Plane quaternion Y.", this, SLOT(updatePlane()));
  plane_orientation_z_ = new rviz_common::properties::FloatProperty(
    "Plane Orientation Z", 0.0, "Plane quaternion Z.", this, SLOT(updatePlane()));
  plane_orientation_w_ = new rviz_common::properties::FloatProperty(
    "Plane Orientation W", 1.0, "Plane quaternion W.", this, SLOT(updatePlane()));

  plane_size_x_ = new rviz_common::properties::FloatProperty(
    "Plane Size X", 1.0, "Plane extent along its local X axis (m).", this,
    SLOT(updatePlane()));
  plane_size_x_->setMin(0.05);
  plane_size_y_ = new rviz_common::properties::FloatProperty(
    "Plane Size Y", 1.0, "Plane extent along its local Y axis (m).", this,
    SLOT(updatePlane()));
  plane_size_y_->setMin(0.05);

  plane_color_property_ = new rviz_common::properties::ColorProperty(
    "Plane Color", QColor(190, 190, 190), "Grid plane colour.", this,
    SLOT(updatePlane()));
  plane_alpha_property_ = new rviz_common::properties::FloatProperty(
    "Plane Alpha", 0.4, "Grid plane opacity.", this, SLOT(updatePlane()));
  plane_alpha_property_->setMin(0.0);
  plane_alpha_property_->setMax(1.0);

  gizmo_visible_property_ = new rviz_common::properties::BoolProperty(
    "Gizmo Visible", true,
    "Interactive 6-DOF marker for dragging the plane. Rendered in-place by "
    "this display (no separate 'Interactive Markers' display needed).", this,
    SLOT(updateGizmoVisible()));

  // --- Grid ---
  grid_cells_x_ = new rviz_common::properties::IntProperty(
    "Grid Cells X", 10, "Number of cells along the plane's X axis.", this,
    SLOT(updatePlane()));
  grid_cells_x_->setMin(1);
  grid_cells_y_ = new rviz_common::properties::IntProperty(
    "Grid Cells Y", 10, "Number of cells along the plane's Y axis.", this,
    SLOT(updatePlane()));
  grid_cells_y_->setMin(1);

  // --- Solving ---
  auto_refresh_property_ = new rviz_common::properties::BoolProperty(
    "Auto Refresh", true,
    "Re-solve automatically after a plane/grid change settles (0.5 s). "
    "Disable to solve only when 'Refresh' is clicked.", this,
    SLOT(updateAutoRefresh()));
  refresh_property_ = new rviz_common::properties::BoolProperty(
    "Refresh", false, "Solve the map now.", this, SLOT(updateRefresh()));

  // --- Cell rendering ---
  cell_style_property_ = new rviz_common::properties::EnumProperty(
    "Cell Style", "Arrows",
    "Rendering style for each grid cell. 'Arrows' is the default: every cell "
    "is a small arrow showing the 6-DOF tool pose (position + orientation), "
    "coloured green/red by IK success.", this,
    SLOT(updateCellStyle()));
  cell_style_property_->addOption("Arrows", kCellStyleArrows);
  cell_style_property_->addOption("Points", rviz_rendering::PointCloud::RM_POINTS);
  cell_style_property_->addOption("Spheres", rviz_rendering::PointCloud::RM_SPHERES);
  cell_style_property_->addOption("Squares", rviz_rendering::PointCloud::RM_SQUARES);
  cell_style_property_->addOption("Boxes", rviz_rendering::PointCloud::RM_BOXES);

  solved_color_property_ = new rviz_common::properties::ColorProperty(
    "Solved Color", QColor(0, 200, 0), "Color of IK-solvable cells.", this,
    SLOT(updateCellColors()));
  failed_color_property_ = new rviz_common::properties::ColorProperty(
    "Failed Color", QColor(200, 0, 0), "Color of unsolvable cells.", this,
    SLOT(updateCellColors()));
  cell_alpha_property_ = new rviz_common::properties::FloatProperty(
    "Cell Alpha", 1.0, "Cell opacity.", this, SLOT(updateCellColors()));
  cell_alpha_property_->setMin(0.0);
  cell_alpha_property_->setMax(1.0);

  // --- Solved-configuration robot ghosts ---
  show_solutions_property_ = new rviz_common::properties::EnumProperty(
    "Show Solutions", "Off",
    "Render the full robot at solved grid cells.", this,
    SLOT(updateShowSolutions()));
  show_solutions_property_->addOption("Off", 0);
  show_solutions_property_->addOption("All Solved Cells", 1);
  show_solutions_property_->addOption("Selected Cell", 2);

  selected_cell_property_ = new rviz_common::properties::IntProperty(
    "Selected Cell", 0,
    "Row-major cell index (iy * grid_size_x + ix) to show in 'Selected Cell' "
    "mode.", this, SLOT(updateSelectedCell()));
  selected_cell_property_->setMin(0);

  solution_alpha_property_ = new rviz_common::properties::FloatProperty(
    "Solution Alpha", 0.6, "Opacity of the solved-configuration ghost robots.",
    this, SLOT(updateSolutionAlpha()));
  solution_alpha_property_->setMin(0.0);
  solution_alpha_property_->setMax(1.0);

  // --- Plane visual ---
  plane_shape_ = std::make_unique<rviz_rendering::Shape>(
    rviz_rendering::Shape::Cube, context_->getSceneManager(), scene_node_);

  // --- Grid cell cloud ---
  cloud_ = std::make_unique<rviz_rendering::PointCloud>();
  cloud_->setRenderMode(rviz_rendering::PointCloud::RM_SPHERES);
  scene_node_->attachObject(cloud_.get());

  // --- Service client ---
  updateServiceName();

  updatePlane();

  // --- Embedded interactive-marker client (self-contained gizmo rendering) ---
  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (ros_node_abstraction) {
    auto node = ros_node_abstraction->get_raw_node();
    auto transformer = context_->getFrameManager()->getTransformer();
    gizmo_client_ = std::make_unique<interactive_markers::InteractiveMarkerClient>(
      node, transformer, fixed_frame_.toStdString());
    gizmo_client_->setInitializeCallback(
      std::bind(
        &ReachabilityMapDisplay::gizmoInitializeCallback, this,
        std::placeholders::_1));
    gizmo_client_->setUpdateCallback(
      std::bind(
        &ReachabilityMapDisplay::gizmoUpdateCallback, this,
        std::placeholders::_1));
    gizmo_client_->setResetCallback(
      std::bind(&ReachabilityMapDisplay::gizmoResetCallback, this));
    gizmo_client_->setStatusCallback(
      std::bind(
        &ReachabilityMapDisplay::gizmoStatusCallback, this,
        std::placeholders::_1, std::placeholders::_2));
  }
}

void ReachabilityMapDisplay::reset()
{
  Display::reset();
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  metrics_msg_new_.reset();
  clearSolutionRobots();
  cell_arrows_.clear();
  plane_dirty_ = false;
  settle_timer_ = 0.0f;
  metrics_active_.reset();
  if (cloud_) {
    cloud_->clear();
  }
}

void ReachabilityMapDisplay::onEnable()
{
  Display::onEnable();
  if (gizmo_visible_property_ && gizmo_visible_property_->getBool() && !gizmo_active_) {
    createGizmo();
    gizmo_active_ = true;
  }
}

void ReachabilityMapDisplay::onDisable()
{
  if (gizmo_active_) {
    destroyGizmo();
    gizmo_active_ = false;
  }
  Display::onDisable();
}

void ReachabilityMapDisplay::fixedFrameChanged()
{
  if (gizmo_client_) {
    gizmo_client_->setTargetFrame(fixed_frame_.toStdString());
  }
  Display::fixedFrameChanged();
}

// ============================================================================
// URDF loading (always from `/robot_description`)
// ============================================================================

void ReachabilityMapDisplay::loadURDF()
{
  // Fast path: already loaded (callback writes urdf_xml_ from another thread).
  {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (!urdf_xml_.empty()) {
      urdf_loaded_ = true;
      return;
    }
  }

  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (!ros_node_abstraction) {return;}
  auto node = ros_node_abstraction->get_raw_node();

  if (!robot_description_sub_) {
    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.transient_local();
    robot_description_sub_ = node->create_subscription<std_msgs::msg::String>(
      "/robot_description",
      qos,
      [this](std_msgs::msg::String::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        if (urdf_xml_.empty()) {
          urdf_xml_ = msg->data;
        }
      });
  }

  {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (!urdf_xml_.empty()) {
      urdf_loaded_ = true;
      return;
    }
  }

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
  urdf_loaded_ = !urdf_xml_.empty();
}

void ReachabilityMapDisplay::loadRobotModel()
{
  if (robot_loaded_) {return;}

  loadURDF();
  if (!urdf_loaded_) {
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "Robot Model",
      QString("No URDF yet on /robot_description.\n"
        "Needed only for the solution-robot ghosts; the map itself works "
        "without it."));
    return;
  }

  urdf_model_ = urdf::parseURDF(urdf_xml_);
  if (!urdf_model_) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "Robot Model",
      "Failed to parse URDF.");
    urdf_xml_.clear();
    urdf_loaded_ = false;
    return;
  }

  // Give every link visual a usable material (same fix as the trajectory
  // display): rviz falls back to the red "RVIZ/ShadedRed" material for a
  // visual with no material tag and renders all-zero colours transparent.
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

  if (!fk_engine_.initFromString(urdf_xml_)) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "Robot Model",
      "Failed to build FK tree from URDF.");
    urdf_xml_.clear();
    urdf_loaded_ = false;
    return;
  }

  robot_loaded_ = true;
  setStatus(
    rviz_common::properties::StatusProperty::Ok, "Robot Model",
    QString("Loaded URDF with %1 links, %2 joints.")
      .arg(fk_engine_.getLinkNames().size())
      .arg(fk_engine_.getJointNames().size()));

  updateSolutionRobots();
}

// ============================================================================
// Property update slots
// ============================================================================

void ReachabilityMapDisplay::updateServiceName()
{
  client_.reset();
  solve_pending_.store(false);
  solve_inflight_.store(false);
  {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_msg_new_.reset();
  }

  const std::string service = service_name_property_->getStdString();
  if (service.empty()) {
    return;
  }

  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (!ros_node_abstraction) {return;}
  auto node = ros_node_abstraction->get_raw_node();
  client_ = node->create_client<GenerateRM>(service);

  // Solve immediately with the new service.
  queueSolve();
}

void ReachabilityMapDisplay::updatePlane()
{
  updatePlaneShape();
  if (!applying_gizmo_) {
    // Property edit (not a gizmo drag) -> keep the marker in sync.
    gizmo_sync_pending_ = true;
  }
  if (auto_refresh_property_->getBool()) {
    plane_dirty_ = true;
    settle_timer_ = 0.0f;
  }
}

void ReachabilityMapDisplay::updateAutoRefresh()
{
  if (auto_refresh_property_->getBool()) {
    // Turning auto-refresh on flags the current plane for a solve.
    plane_dirty_ = true;
    settle_timer_ = 0.0f;
  }
}

void ReachabilityMapDisplay::updateRefresh()
{
  if (!refresh_property_->getBool()) {
    return;
  }
  // Uncheck so the button works as a trigger, then solve immediately
  // regardless of auto-refresh.
  refresh_property_->setBool(false);
  queueSolve();
}

void ReachabilityMapDisplay::updateCellStyle()
{
  if (cloud_ == nullptr) {
    return;
  }
  // Rebuild the cells (arrows vs point-cloud styles).
  updateMapPoints();
}

void ReachabilityMapDisplay::updateCellColors()
{
  updateMapPoints();
}

void ReachabilityMapDisplay::updateShowSolutions()
{
  if (show_solutions_property_->getOptionInt() != 0 && !robot_loaded_) {
    loadRobotModel();
  }
  updateSolutionRobots();
}

void ReachabilityMapDisplay::updateSelectedCell()
{
  updateSolutionRobots();
}

void ReachabilityMapDisplay::updateSolutionAlpha()
{
  updateSolutionAlphaInternal();
}

// ============================================================================
// Solve pumping
// ============================================================================

void ReachabilityMapDisplay::queueSolve()
{
  solve_pending_.store(true);
}

void ReachabilityMapDisplay::drainSolve()
{
  if (!client_ || solve_inflight_.load() || !solve_pending_.load()) {
    return;
  }
  if (!client_->service_is_ready()) {
    return;
  }

  auto req = std::make_shared<GenerateRM::Request>();
  req->plane_position_x = plane_position_x_->getFloat();
  req->plane_position_y = plane_position_y_->getFloat();
  req->plane_position_z = plane_position_z_->getFloat();
  const Ogre::Quaternion q = currentPlaneOrientation();
  req->plane_orientation_x = q.x;
  req->plane_orientation_y = q.y;
  req->plane_orientation_z = q.z;
  req->plane_orientation_w = q.w;
  req->plane_size_x = plane_size_x_->getFloat();
  req->plane_size_y = plane_size_y_->getFloat();
  req->grid_size_x = static_cast<uint32_t>(grid_cells_x_->getInt());
  req->grid_size_y = static_cast<uint32_t>(grid_cells_y_->getInt());

  solve_inflight_.store(true);
  solve_pending_.store(false);
  client_->async_send_request(
    req,
    [this](rclcpp::Client<GenerateRM>::SharedFuture future) {
      onSolveResponse(future);
    });
}

void ReachabilityMapDisplay::onSolveResponse(
  rclcpp::Client<GenerateRM>::SharedFuture future)
{
  solve_inflight_.store(false);
  try {
    auto resp = future.get();
    if (!resp) {
      return;
    }
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_msg_new_ = std::make_shared<ReachabilityMetrics>(resp->metrics);
    last_solve_message_ = resp->message;
    last_solve_ok_ = resp->success;
    // If the plane/grid changed again while this solve was in flight, the
    // flags have been re-marked since; update()'s drainSolve() will re-fire.
  } catch (const std::exception &) {
    // A failed service call surfaces as an exception from future.get().
  }
}

void ReachabilityMapDisplay::promoteMetrics()
{
  std::shared_ptr<ReachabilityMetrics> next;
  {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (!metrics_msg_new_) {
      return;
    }
    next = std::move(metrics_msg_new_);
    metrics_msg_new_.reset();
  }
  metrics_active_ = std::move(next);
  rebuildVisualization();
}

// ============================================================================
// Visualization
// ============================================================================

void ReachabilityMapDisplay::updatePlaneShape()
{
  if (plane_shape_ == nullptr) {
    return;
  }
  plane_shape_->setPosition(Ogre::Vector3(
    plane_position_x_->getFloat(),
    plane_position_y_->getFloat(),
    plane_position_z_->getFloat()));

  plane_shape_->setOrientation(currentPlaneOrientation());

  // Thin slab: the unit cube is 1x1x1, so depth is just 4 mm.
  plane_shape_->setScale(Ogre::Vector3(
    plane_size_x_->getFloat(), plane_size_y_->getFloat(), 0.004f));

  Ogre::ColourValue c =
    rviz_common::properties::qtToOgre(plane_color_property_->getColor());
  c.a = plane_alpha_property_->getFloat();
  plane_shape_->setColor(c);
}

Ogre::Quaternion ReachabilityMapDisplay::currentPlaneOrientation() const
{
  // MoveIt-style quaternion entry: the properties hold (X, Y, Z, W).
  // Normalize so a malformed/mostly-zero entry falls back to identity instead
  // of producing NaN in the plane and the goals.
  Ogre::Quaternion q(
    plane_orientation_w_->getFloat(),
    plane_orientation_x_->getFloat(),
    plane_orientation_y_->getFloat(),
    plane_orientation_z_->getFloat());
  if (std::abs(q.w) < 1e-6f && std::abs(q.x) < 1e-6f &&
    std::abs(q.y) < 1e-6f && std::abs(q.z) < 1e-6f)
  {
    return Ogre::Quaternion::IDENTITY;
  }
  q.normalise();
  return q;
}

void ReachabilityMapDisplay::updateMapPoints()
{
  if (cloud_ == nullptr) {
    return;
  }
  cloud_->clear();
  cell_arrows_.clear();
  if (!metrics_active_) {
    return;
  }
  const size_t n = metrics_active_->goals.size();
  if (n == 0) {
    return;
  }

  const double sx = metrics_active_->plane_size_x;
  const double sy = metrics_active_->plane_size_y;
  const double gx = std::max(1.0, static_cast<double>(metrics_active_->grid_size_x));
  const double gy = std::max(1.0, static_cast<double>(metrics_active_->grid_size_y));
  const double cell = 0.9 * std::min(sx / gx, sy / gy);

  const Ogre::ColourValue solved =
    rviz_common::properties::qtToOgre(solved_color_property_->getColor());
  const Ogre::ColourValue failed =
    rviz_common::properties::qtToOgre(failed_color_property_->getColor());
  const float alpha = cell_alpha_property_->getFloat();

  if (cell_style_property_->getOptionInt() == kCellStyleArrows) {
    // One small arrow per cell, based on the *solved* end-effector pose rather
    // than the requested goal: for solvable cells the cell's joint config is
    // FK'd and the link pose landing on the goal is used, so the arrows line
    // up with the "solution robot" ghosts (same FK, same URDF). The shaft
    // points along the tool frame's +X — the axis the offset grasping_frame's
    // end effector visibly extends along. The server composes every goal with
    // +90deg about the plane's local Y (tool +X INTO the plane), so arrows and
    // solved IKs agree and both look toward the plane.
    // Unsolvable cells keep the requested goal pose and are coloured red.
    cell_arrows_.reserve(n);
    const float shaft_len = static_cast<float>(cell * 0.55);
    const float head_len = static_cast<float>(cell * 0.25);
    for (size_t i = 0; i < n; ++i) {
      const auto & goal = metrics_active_->goals[i];
      const bool ok =
        i < metrics_active_->joint_states_valid.size() &&
        metrics_active_->joint_states_valid[i].data;
      const Ogre::ColourValue & c = ok ? solved : failed;

      Ogre::Quaternion q;
      Ogre::Vector3 pos(
        static_cast<float>(goal.position.x),
        static_cast<float>(goal.position.y),
        static_cast<float>(goal.position.z));
      Eigen::Isometry3d solved_pose;
      if (ok && solvedLinkPose(i, goal, solved_pose)) {
        pos = Ogre::Vector3(
          static_cast<float>(solved_pose.translation().x()),
          static_cast<float>(solved_pose.translation().y()),
          static_cast<float>(solved_pose.translation().z()));
        const Eigen::Quaterniond q_eig(solved_pose.linear());
        q = Ogre::Quaternion(
          q_eig.w(), q_eig.x(), q_eig.y(), q_eig.z());
      } else {
        q = Ogre::Quaternion(
          goal.orientation.w, goal.orientation.x,
          goal.orientation.y, goal.orientation.z);
        if (std::abs(q.w) < 1e-6f && std::abs(q.x) < 1e-6f &&
          std::abs(q.y) < 1e-6f && std::abs(q.z) < 1e-6f)
        {
          q = Ogre::Quaternion::IDENTITY;
        } else {
          q.normalise();
        }
      }

      Ogre::Vector3 x_dir = q * Ogre::Vector3::UNIT_X;
      if (x_dir.length() < 1e-6f) {
        x_dir = Ogre::Vector3::UNIT_X;
      }

      auto arrow = std::make_unique<rviz_rendering::Arrow>(
        context_->getSceneManager(), scene_node_, shaft_len,
        static_cast<float>(cell * 0.08), head_len,
        static_cast<float>(cell * 0.22));
      arrow->setColor(c.r, c.g, c.b, alpha);
      arrow->setPosition(pos);
      arrow->setDirection(x_dir);
      cell_arrows_.push_back(std::move(arrow));
    }
    return;
  }

  // Point-cloud styles (Points / Spheres / Squares / Boxes).
  cloud_->setRenderMode(
    static_cast<rviz_rendering::PointCloud::RenderMode>(
      cell_style_property_->getOptionInt()));
  cloud_->setDimensions(
    static_cast<float>(cell), static_cast<float>(cell), static_cast<float>(cell));

  std::vector<rviz_rendering::PointCloud::Point> points;
  points.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const auto & goal = metrics_active_->goals[i];
    const bool ok =
      i < metrics_active_->joint_states_valid.size() &&
      metrics_active_->joint_states_valid[i].data;
    const Ogre::ColourValue & c = ok ? solved : failed;
    rviz_rendering::PointCloud::Point pt;
    pt.position = Ogre::Vector3(
      static_cast<float>(goal.position.x),
      static_cast<float>(goal.position.y),
      static_cast<float>(goal.position.z));
    pt.setColor(c.r, c.g, c.b, alpha);
    points.push_back(pt);
  }
  cloud_->addPoints(points.begin(), points.end());
}

void ReachabilityMapDisplay::destroySolutionRobot(size_t index)
{
  SolutionRobot & entry = solution_robots_[index];
  if (entry.robot) {
    // Hide first so the ghost disappears a frame early; nothing lingers.
    entry.robot->setVisible(false);
  }
  // Robot's own destructor already destroys its three scene nodes, but the
  // holder node itself is ours: drop the whole subtree to be certain every
  // entity is gone even if a link outlives the Robot object.
  entry.robot.reset();
  if (entry.node) {
    entry.node->removeAndDestroyAllChildren();
    scene_node_->removeAndDestroyChild(entry.node);
  }
}

void ReachabilityMapDisplay::clearSolutionRobots()
{
  while (!solution_robots_.empty()) {
    destroySolutionRobot(solution_robots_.size() - 1);
    solution_robots_.pop_back();
  }
}

void ReachabilityMapDisplay::updateSolutionRobots()
{
  if (!metrics_active_ || !robot_loaded_) {
    clearSolutionRobots();
    return;
  }
  const int mode = show_solutions_property_->getOptionInt();
  if (mode == 0) {
    clearSolutionRobots();
    return;
  }
  const size_t n = metrics_active_->goals.size();
  const size_t n_valid = metrics_active_->joint_states_valid.size();
  if (n == 0 || n_valid == 0) {
    clearSolutionRobots();
    return;
  }

  std::vector<size_t> cells;
  if (mode == 2) {
    // Selected cell — wrap into [0, n) so any property value is safe.
    long idx = static_cast<long>(selected_cell_property_->getInt()) %
      static_cast<long>(n);
    if (idx < 0) {
      idx += static_cast<long>(n);
    }
    const size_t u = static_cast<size_t>(idx);
    if (u < n_valid && metrics_active_->joint_states_valid[u].data) {
      cells.push_back(u);
    }
  } else {
    size_t n_solved = 0;
    for (const auto & v : metrics_active_->joint_states_valid) {
      if (v.data) {
        ++n_solved;
      }
    }
    // Auto-derived only (no user knob): keep the ghost count within
    // kMaxSolutionRobots so a fully-solved map stays renderable.
    size_t step = 1;
    if (n_solved > kMaxSolutionRobots) {
      step = static_cast<size_t>(std::ceil(
        static_cast<double>(n_solved) / static_cast<double>(kMaxSolutionRobots)));
    }
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
      if (i >= n_valid || !metrics_active_->joint_states_valid[i].data) {
        continue;
      }
      if (k % step == 0) {
        cells.push_back(i);
        if (cells.size() >= kMaxSolutionRobots) {
          break;
        }
      }
      ++k;
    }
  }
  if (cells.empty()) {
    clearSolutionRobots();
    return;
  }

  const float alpha = solution_alpha_property_->getFloat();

  // Reuse the ghost pool between solved maps instead of tearing every Robot
  // down and re-loading the whole URDF each time: keep as many Robot objects
  // as the new cell list needs, create only the missing ones and retire the
  // surplus. The old per-solve rebuild storm (destroy + re-create + re-load up
  // to 120 full robot models per map) caused the hitches and the Ogre
  // resource growth when dragging the plane with solutions enabled.
  while (solution_robots_.size() < cells.size()) {
    Ogre::SceneNode * holder = scene_node_->createChildSceneNode();
    auto robot = std::make_unique<rviz_default_plugins::robot::Robot>(
      holder, context_, "Reachability Solution", nullptr);
    robot->load(*urdf_model_, true, false);
    robot->setVisualVisible(true);
    robot->setCollisionVisible(false);
    solution_robots_.push_back(SolutionRobot{holder, std::move(robot)});
  }
  while (solution_robots_.size() > cells.size()) {
    destroySolutionRobot(solution_robots_.size() - 1);
    solution_robots_.pop_back();
  }

  for (size_t i = 0; i < cells.size(); ++i) {
    const std::map<std::string, double> joint_map = jointMapForCell(cells[i]);
    std::map<std::string, Eigen::Isometry3d> transforms;
    fk_engine_.computeFK(joint_map, transforms);

    CuroboLinkUpdater updater(transforms);
    solution_robots_[i].robot->update(updater);
    solution_robots_[i].robot->setAlpha(alpha);
    solution_robots_[i].robot->setVisible(true);
  }
}

std::map<std::string, double> ReachabilityMapDisplay::jointMapForCell(
  size_t cell_index) const
{
  std::map<std::string, double> out;
  if (metrics_active_ && cell_index < metrics_active_->joint_states.size()) {
    const auto & js = metrics_active_->joint_states[cell_index];
    const size_t n = std::min(js.name.size(), js.position.size());
    for (size_t i = 0; i < n; ++i) {
      out[js.name[i]] = js.position[i];
    }
  }
  return out;
}

void ReachabilityMapDisplay::updateSolutionAlphaInternal()
{
  const float alpha = solution_alpha_property_->getFloat();
  for (auto & entry : solution_robots_) {
    if (entry.robot) {
      entry.robot->setAlpha(alpha);
    }
  }
}

bool ReachabilityMapDisplay::solvedLinkPose(
  size_t cell_index,
  const geometry_msgs::msg::Pose & goal,
  Eigen::Isometry3d & pose_out) const
{
  if (!robot_loaded_) {
    return false;
  }
  std::map<std::string, Eigen::Isometry3d> transforms;
  if (!fk_engine_.computeFK(jointMapForCell(cell_index), transforms) ||
      transforms.empty())
  {
    return false;
  }

  const Eigen::Vector3d target(
    goal.position.x, goal.position.y, goal.position.z);
  const std::string * best_link = nullptr;
  double best_dist = std::numeric_limits<double>::max();
  for (const auto & [name, T] : transforms) {
    const double dist = (T.translation() - target).norm();
    if (dist < best_dist) {
      best_dist = dist;
      best_link = &name;
    }
  }

  // The server solves each goal to within its position tolerance (0.05 m). A
  // link landing well beyond that means the metrics' joint names don't line
  // up with this URDF (or the config doesn't reach the goal), so the FK is
  // not trustworthy for this cell.
  constexpr double kMaxToolToGoalDistance = 0.25;
  if (!best_link || best_dist > kMaxToolToGoalDistance) {
    return false;
  }
  pose_out = transforms.at(*best_link);
  return true;
}

void ReachabilityMapDisplay::rebuildVisualization()
{
  updatePlaneShape();
  updateMapPoints();
  updateSolutionRobots();

  if (!metrics_active_) {
    return;
  }
  const QString status =
    QString("Solved %1/%2 in %3 ms")
      .arg(metrics_active_->n_solved)
      .arg(metrics_active_->n_total)
      .arg(metrics_active_->solve_time_ms, 0, 'f', 1);
  if (last_solve_ok_) {
    setStatus(rviz_common::properties::StatusProperty::Ok,
      "Reachability Map", status);
  } else {
    const QString msg = last_solve_message_.empty()
      ? status : QString::fromStdString(last_solve_message_);
    setStatus(rviz_common::properties::StatusProperty::Warn,
      "Reachability Map", msg);
  }
}

// ============================================================================
// Interactive plane control (MoveIt-style 6-DOF marker)
// ============================================================================

geometry_msgs::msg::Pose ReachabilityMapDisplay::planePoseFromProperties() const
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = plane_position_x_->getFloat();
  pose.position.y = plane_position_y_->getFloat();
  pose.position.z = plane_position_z_->getFloat();
  const Ogre::Quaternion q = currentPlaneOrientation();
  pose.orientation.w = q.w;
  pose.orientation.x = q.x;
  pose.orientation.y = q.y;
  pose.orientation.z = q.z;
  return pose;
}

void ReachabilityMapDisplay::createGizmo()
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
      node->get_logger(),
      "Reachability plane gizmo on /%s/* (rendered via embedded "
      "interactive-marker client).", gizmo_name_.c_str());
  }
  makeGizmoMarker(planePoseFromProperties());
  gizmo_server_->applyChanges();
  connectGizmoClient();
}

void ReachabilityMapDisplay::destroyGizmo()
{
  disconnectGizmoClient();
  if (gizmo_server_) {
    gizmo_server_->clear();
    gizmo_server_->applyChanges();
  }
  gizmo_server_.reset();
}

void ReachabilityMapDisplay::connectGizmoClient()
{
  if (!gizmo_client_ || gizmo_client_connected_) {
    return;
  }
  gizmo_client_->connect(gizmo_name_);
  gizmo_client_connected_ = true;
}

void ReachabilityMapDisplay::disconnectGizmoClient()
{
  if (gizmo_client_) {
    gizmo_client_->disconnect();
  }
  gizmo_client_connected_ = false;
  eraseAllGizmoMarkers();
}

void ReachabilityMapDisplay::makeGizmoMarker(const geometry_msgs::msg::Pose & pose)
{
  if (!gizmo_server_) {
    return;
  }

  gizmo_body_size_x_ = plane_size_x_->getFloat();
  gizmo_body_size_y_ = plane_size_y_->getFloat();

  visualization_msgs::msg::InteractiveMarker int_marker;
  int_marker.header.frame_id = gizmo_frame_;
  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (ros_node_abstraction) {
    int_marker.header.stamp =
      ros_node_abstraction->get_raw_node()->get_clock()->now();
  }
  int_marker.pose = pose;
  int_marker.scale = 1.0;
  int_marker.name = gizmo_name_;
  int_marker.description = "Reachability plane (6-DOF)";

  // Semi-transparent body that tracks the plane geometry.
  visualization_msgs::msg::Marker body;
  body.type = visualization_msgs::msg::Marker::CUBE;
  body.pose.orientation.w = 1.0;
  body.scale.x = static_cast<float>(gizmo_body_size_x_);
  body.scale.y = static_cast<float>(gizmo_body_size_y_);
  body.scale.z = 0.004f;
  body.color.r = 0.75f;
  body.color.g = 0.75f;
  body.color.b = 0.75f;
  body.color.a = 0.35f;
  visualization_msgs::msg::InteractiveMarkerControl body_control;
  body_control.always_visible = true;
  body_control.markers.push_back(body);
  int_marker.controls.push_back(body_control);

  // 6-DOF controls (rotate/move per axis), same scheme as ArrowInteraction.
  const struct { float wx, wy, wz; const char * rotate; const char * move; } axes[] = {
    {1.0f, 0.0f, 0.0f, "rotate_x", "move_x"},
    {0.0f, 1.0f, 0.0f, "rotate_y", "move_y"},
    {0.0f, 0.0f, 1.0f, "rotate_z", "move_z"},
  };
  for (const auto & a : axes) {
    visualization_msgs::msg::InteractiveMarkerControl rotate;
    rotate.orientation.w = 1.0;
    rotate.orientation.x = a.wx;
    rotate.orientation.y = a.wy;
    rotate.orientation.z = a.wz;
    rotate.name = a.rotate;
    rotate.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS;
    int_marker.controls.push_back(rotate);

    visualization_msgs::msg::InteractiveMarkerControl move;
    move.orientation.w = 1.0;
    move.orientation.x = a.wx;
    move.orientation.y = a.wy;
    move.orientation.z = a.wz;
    move.name = a.move;
    move.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
    int_marker.controls.push_back(move);
  }

  gizmo_server_->insert(
    int_marker,
    std::bind(
      &ReachabilityMapDisplay::gizmoFeedback, this, std::placeholders::_1));
}

void ReachabilityMapDisplay::gizmoFeedback(
  const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr & feedback)
{
  if (!feedback) {
    return;
  }
  if (feedback->event_type !=
      visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE) {
    return;
  }
  std::lock_guard<std::mutex> lock(gizmo_mutex_);
  gizmo_feedback_pose_ = feedback->pose;
  gizmo_feedback_dirty_ = true;
}

void ReachabilityMapDisplay::syncGizmoToProperties()
{
  geometry_msgs::msg::Pose pose;
  {
    std::lock_guard<std::mutex> lock(gizmo_mutex_);
    if (!gizmo_feedback_dirty_) {
      return;
    }
    pose = gizmo_feedback_pose_;
    gizmo_feedback_dirty_ = false;
  }

  // Write the dragged pose back, quaternion straight through (no RPY round
  // trip). Suppress the property->gizmo echo in updatePlane().
  applying_gizmo_ = true;
  plane_position_x_->setFloat(pose.position.x);
  plane_position_y_->setFloat(pose.position.y);
  plane_position_z_->setFloat(pose.position.z);
  plane_orientation_w_->setFloat(pose.orientation.w);
  plane_orientation_x_->setFloat(pose.orientation.x);
  plane_orientation_y_->setFloat(pose.orientation.y);
  plane_orientation_z_->setFloat(pose.orientation.z);
  applying_gizmo_ = false;
}

void ReachabilityMapDisplay::syncPropertiesToGizmo()
{
  gizmo_sync_pending_ = false;
  if (!gizmo_server_) {
    return;
  }

  // A size change reshapes the body cube, which needs a re-insert.
  const double sx = plane_size_x_->getFloat();
  const double sy = plane_size_y_->getFloat();
  if (sx != gizmo_body_size_x_ || sy != gizmo_body_size_y_) {
    gizmo_server_->clear();
    makeGizmoMarker(planePoseFromProperties());
    gizmo_server_->applyChanges();
    return;
  }

  gizmo_server_->setPose(gizmo_name_, planePoseFromProperties());
  gizmo_server_->applyChanges();
}

// ============================================================================
// Embedded interactive-marker client (renders the plane gizmo in this display)
// ============================================================================

void ReachabilityMapDisplay::gizmoInitializeCallback(
  visualization_msgs::srv::GetInteractiveMarkers::Response::SharedPtr msg)
{
  eraseAllGizmoMarkers();
  if (msg) {
    updateGizmoMarkers(msg->markers);
  }
}

void ReachabilityMapDisplay::gizmoUpdateCallback(
  visualization_msgs::msg::InteractiveMarkerUpdate::ConstSharedPtr msg)
{
  if (!msg) {
    return;
  }
  updateGizmoMarkers(msg->markers);
  updateGizmoPoses(msg->poses);
  for (const std::string & marker_name : msg->erases) {
    interactive_markers_map_.erase(marker_name);
  }
}

void ReachabilityMapDisplay::gizmoResetCallback()
{
  eraseAllGizmoMarkers();
}

void ReachabilityMapDisplay::gizmoStatusCallback(
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
  setStatusStd(level, "Gizmo Visible", message);
}

void ReachabilityMapDisplay::updateGizmoMarkers(
  const std::vector<visualization_msgs::msg::InteractiveMarker> & markers)
{
  for (const visualization_msgs::msg::InteractiveMarker & marker : markers) {
    if (!validateFloats(marker)) {
      setStatusStd(
        rviz_common::properties::StatusProperty::Error, marker.name,
        "Marker contains invalid floats!");
      continue;
    }

    auto int_marker_entry = interactive_markers_map_.find(marker.name);
    if (int_marker_entry == interactive_markers_map_.end()) {
      auto scene_marker =
        std::make_shared<rviz_default_plugins::displays::InteractiveMarker>(
          getSceneNode(), context_);
      // Drag feedback -> server (which writes the plane properties).
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
      int_marker_entry =
        interactive_markers_map_.emplace(marker.name, scene_marker).first;
    }

    if (int_marker_entry->second->processMessage(marker)) {
      int_marker_entry->second->setShowDescription(false);
      // The plane gizmo is rotated by its ROTATE_AXIS rings; the white XYZ
      // triad helper just adds clutter, so keep axis + visual-aid overlays off.
      int_marker_entry->second->setShowAxes(false);
      int_marker_entry->second->setShowVisualAids(false);
    } else {
      setStatusStd(
        rviz_common::properties::StatusProperty::Error, marker.name,
        "Failed to process interactive marker.");
    }
  }
}

void ReachabilityMapDisplay::updateGizmoPoses(
  const std::vector<visualization_msgs::msg::InteractiveMarkerPose> & poses)
{
  for (const visualization_msgs::msg::InteractiveMarkerPose & pose : poses) {
    auto int_marker_entry = interactive_markers_map_.find(pose.name);
    if (int_marker_entry != interactive_markers_map_.end()) {
      int_marker_entry->second->processMessage(pose);
    }
  }
}

void ReachabilityMapDisplay::eraseAllGizmoMarkers()
{
  interactive_markers_map_.clear();
}

void ReachabilityMapDisplay::publishGizmoFeedback(
  visualization_msgs::msg::InteractiveMarkerFeedback & feedback)
{
  if (gizmo_client_) {
    gizmo_client_->publishFeedback(feedback);
  }
}

void ReachabilityMapDisplay::gizmoStatusUpdate(
  rviz_common::properties::StatusProperty::Level level,
  const std::string & name,
  const std::string & text)
{
  setStatusStd(level, name, text);
}

void ReachabilityMapDisplay::updateGizmoVisible()
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

// ============================================================================
// Per-frame update
// ============================================================================

void ReachabilityMapDisplay::update(float wall_dt, float ros_dt)
{
  Display::update(wall_dt, ros_dt);

  // Drive the embedded interactive-marker client (service call, pending
  // updates) and its rendered scene objects.
  if (gizmo_client_) {
    gizmo_client_->update();
  }
  for (const auto & name_marker_pair : interactive_markers_map_) {
    name_marker_pair.second->update();
  }

  // Interactive gizmo: drag feedback -> properties, property edits -> marker.
  if (gizmo_active_) {
    if (gizmo_feedback_dirty_) {
      syncGizmoToProperties();
    }
    if (gizmo_sync_pending_) {
      syncPropertiesToGizmo();
    }
  }

  // "Gizmo Visible" is a pure visibility toggle for the gizmo: only
  // create/destroy on the actual transition, never every frame.
  const bool gizmo_want = gizmo_visible_property_ && gizmo_visible_property_->getBool();
  if (gizmo_want && !gizmo_active_) {
    createGizmo();
    gizmo_active_ = true;
  } else if (!gizmo_want && gizmo_active_) {
    destroyGizmo();
    gizmo_active_ = false;
  }

  // The fixed frame may only be set after the display loaded; rebind the
  // gizmo so it lives in the real frame rather than a startup fallback.
  if (gizmo_active_ && gizmo_server_) {
    const std::string fixed = context_->getFrameManager()->getFixedFrame();
    if (!fixed.empty() && fixed != gizmo_frame_) {
      gizmo_server_->clear();
      gizmo_server_->applyChanges();
      gizmo_server_.reset();
      // Force createGizmo() to re-fetch the frame: gizmo_frame_ is only
      // filled on first use, otherwise the new server would publish its
      // marker in the stale frame.
      gizmo_frame_.clear();
    }
  }
  if (gizmo_active_ && !gizmo_server_) {
    createGizmo();
  }

  // Debounced re-solve: wait for the plane/grid to settle before firing.
  if (plane_dirty_) {
    settle_timer_ += wall_dt;
    if (settle_timer_ >= kReachabilitySettleTime) {
      plane_dirty_ = false;
      settle_timer_ = 0.0f;
      if (auto_refresh_property_->getBool()) {
        queueSolve();
      }
    }
  }

  drainSolve();
  promoteMetrics();

  // Lazily load the URDF once the user actually wants solution ghosts.
  if (!robot_loaded_ && show_solutions_property_->getOptionInt() != 0) {
    loadRobotModel();
  }
}

}  // namespace isaac_ros_cumotion_rviz

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  isaac_ros_cumotion_rviz::ReachabilityMapDisplay, rviz_common::Display)