#include <OgreVector3.h>
#include <OgreColourValue.h>
#include <OgreQuaternion.h>
#include <OgreSceneNode.h>

#include <rcl/time.h>
#include <rclcpp/time.hpp>

#include <rviz_common/logging.hpp>
#include <rviz_common/properties/parse_color.hpp>

#include <isaac_ros_cumotion_rviz/sparse_voxel_grid_display.hpp>

namespace isaac_ros_cumotion_rviz
{

SparseVoxelGridDisplay::SparseVoxelGridDisplay()
  : color_property_(nullptr)
  , alpha_property_(nullptr)
  , point_style_property_(nullptr)
  , cloud_(nullptr)
{
}

SparseVoxelGridDisplay::~SparseVoxelGridDisplay() = default;

void SparseVoxelGridDisplay::onInitialize()
{
  RosTopicDisplay<isaac_ros_cumotion_interfaces::msg::SparseVoxelGrid>::onInitialize();

  cloud_ = std::make_unique<rviz_rendering::PointCloud>();
  cloud_->setRenderMode(rviz_rendering::PointCloud::RM_BOXES);
  scene_node_->attachObject(cloud_.get());

  color_property_ = new rviz_common::properties::ColorProperty(
    "Color", QColor(255, 50, 0), "Voxel color.", this,
    SLOT(updateColorAndAlpha()));
  alpha_property_ = new rviz_common::properties::FloatProperty(
    "Alpha", 1.0, "Voxel opacity.", this, SLOT(updateColorAndAlpha()));
  alpha_property_->setMin(0.0);
  alpha_property_->setMax(1.0);

  point_style_property_ = new rviz_common::properties::EnumProperty(
    "Point Style", "Boxes", "Rendering style for each voxel.", this,
    SLOT(updatePointStyle()));
  point_style_property_->addOption("Points", rviz_rendering::PointCloud::RM_POINTS);
  point_style_property_->addOption("Squares", rviz_rendering::PointCloud::RM_SQUARES);
  point_style_property_->addOption("Flat Squares", rviz_rendering::PointCloud::RM_FLAT_SQUARES);
  point_style_property_->addOption("Spheres", rviz_rendering::PointCloud::RM_SPHERES);
  point_style_property_->addOption("Boxes", rviz_rendering::PointCloud::RM_BOXES);

  updateColorAndAlpha();
  updatePointStyle();
}

void SparseVoxelGridDisplay::reset()
{
  RosTopicDisplay<isaac_ros_cumotion_interfaces::msg::SparseVoxelGrid>::reset();
  if (cloud_) {
    cloud_->clear();
  }
}

void SparseVoxelGridDisplay::updateColorAndAlpha()
{
  if (cloud_ == nullptr) {
    return;
  }
  Ogre::ColourValue c = rviz_common::properties::qtToOgre(color_property_->getColor());
  c.a = alpha_property_->getFloat();
  cloud_->setColor(c);
  cloud_->setAlpha(c.a);
}

void SparseVoxelGridDisplay::updatePointStyle()
{
  if (cloud_ == nullptr) {
    return;
  }
  cloud_->setRenderMode(
    static_cast<rviz_rendering::PointCloud::RenderMode>(
      point_style_property_->getOptionInt()));
}

void SparseVoxelGridDisplay::processMessage(
  isaac_ros_cumotion_interfaces::msg::SparseVoxelGrid::ConstSharedPtr msg)
{
  if (cloud_ == nullptr) {
    return;
  }

  cloud_->clear();

  const uint32_t sy = msg->size_y;
  const uint32_t sz = msg->size_z;
  if (sy == 0 || sz == 0) {
    return;
  }
  const uint32_t syz = sy * sz;
  const float vs = msg->resolution;
  const float half = vs / 2.0f;

  // Voxel dimensions come from the message, not from a UI property.
  cloud_->setDimensions(vs, vs, vs);

  const std::string frame = msg->header.frame_id;

  Ogre::Vector3 origin_offset = Ogre::Vector3::ZERO;
  Ogre::Quaternion rotation = Ogre::Quaternion::IDENTITY;
  bool need_transform = false;
  if (context_ != nullptr && context_->getFrameManager() != nullptr &&
    !frame.empty() && frame != context_->getFrameManager()->getFixedFrame())
  {
    need_transform = true;
    Ogre::Quaternion orient;
    if (context_->getFrameManager()->getTransform(
        frame, rclcpp::Time(msg->header.stamp, RCL_ROS_TIME),
        origin_offset, orient)) {
      rotation = orient;
      setTransformOk();
    } else {
      setMissingTransformToFixedFrame(frame);
      return;
    }
  }

  Ogre::ColourValue c = rviz_common::properties::qtToOgre(color_property_->getColor());
  c.a = alpha_property_->getFloat();

  std::vector<rviz_rendering::PointCloud::Point> points;
  points.reserve(msg->occupied_indices.size());

  for (const int32_t linear : msg->occupied_indices) {
    const uint32_t u = static_cast<uint32_t>(linear);
    const uint32_t gx = u / syz;
    const uint32_t rem = u % syz;
    const uint32_t gy = rem / sz;
    const uint32_t gz = rem % sz;

    Ogre::Vector3 pos;
    pos.x = msg->origin.x + static_cast<float>(gx) * vs + half;
    pos.y = msg->origin.y + static_cast<float>(gy) * vs + half;
    pos.z = msg->origin.z + static_cast<float>(gz) * vs + half;

    if (need_transform) {
      pos = rotation * pos + origin_offset;
    }

    rviz_rendering::PointCloud::Point pt;
    pt.position = pos;
    pt.setColor(c.r, c.g, c.b, c.a);
    points.push_back(pt);
  }

  if (!points.empty()) {
    cloud_->addPoints(points.begin(), points.end());
  }
  setTransformOk();
}

}  // namespace isaac_ros_cumotion_rviz

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  isaac_ros_cumotion_rviz::SparseVoxelGridDisplay, rviz_common::Display)
