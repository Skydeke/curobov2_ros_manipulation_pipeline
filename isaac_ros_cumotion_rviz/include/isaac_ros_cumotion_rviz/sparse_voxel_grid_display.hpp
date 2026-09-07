#pragma once

#ifndef ISAAC_ROS_CUMOTION_RVIZ__SPARSE_VOXEL_GRID_DISPLAY_HPP_
#define ISAAC_ROS_CUMOTION_RVIZ__SPARSE_VOXEL_GRID_DISPLAY_HPP_

#include <memory>

#include <OgreVector3.h>

#include <rviz_common/message_filter_display.hpp>
#include <rviz_common/properties/color_property.hpp>
#include <rviz_common/properties/enum_property.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_rendering/objects/point_cloud.hpp>

#include <isaac_ros_cumotion_interfaces/msg/sparse_voxel_grid.hpp>

namespace isaac_ros_cumotion_rviz
{
/**
 * RViz Display that renders the cuRobo server's occupied voxels directly from
 * the SparseVoxelGrid message (curobo_server/voxel_grid_sparse), with no
 * intermediate MarkerArray republish and no dependency on any panel being
 * loaded.
 *
 * Each occupied cell is drawn as a 3D box in a rviz_rendering::PointCloud
 * (default style: Boxes — axis-aligned cubes like an octomap).  Voxel
 * dimensions are taken directly from the message's resolution field; there
 * is no user-configurable size property.  Empty grids clear the cloud, so
 * voxels disappear the moment the mapper stops observing anything.
 */
class SparseVoxelGridDisplay
  : public rviz_common::RosTopicDisplay<
      isaac_ros_cumotion_interfaces::msg::SparseVoxelGrid>
{
  Q_OBJECT
public:
  SparseVoxelGridDisplay();
  ~SparseVoxelGridDisplay() override;

protected:
  void onInitialize() override;
  void reset() override;
  void processMessage(
    isaac_ros_cumotion_interfaces::msg::SparseVoxelGrid::ConstSharedPtr msg) override;

private Q_SLOTS:
  void updateColorAndAlpha();
  void updatePointStyle();

private:
  rviz_common::properties::ColorProperty * color_property_;
  rviz_common::properties::FloatProperty * alpha_property_;
  rviz_common::properties::EnumProperty * point_style_property_;
  std::unique_ptr<rviz_rendering::PointCloud> cloud_;
};

}  // namespace isaac_ros_cumotion_rviz

#endif  // ISAAC_ROS_CUMOTION_RVIZ__SPARSE_VOXEL_GRID_DISPLAY_HPP_
