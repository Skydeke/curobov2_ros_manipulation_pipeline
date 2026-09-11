#pragma once

#ifndef ISAAC_ROS_CUMOTION_RVIZ__CUROBO_LINK_UPDATER_HPP_
#define ISAAC_ROS_CUMOTION_RVIZ__CUROBO_LINK_UPDATER_HPP_

#include <map>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <rviz_default_plugins/robot/link_updater.hpp>

namespace isaac_ros_cumotion_rviz
{

/**
 * Feeds precomputed Eigen link transforms to rviz_default_plugins::robot::Robot.
 *
 * Used by CuroboTrajectoryDisplay to render the robot at each trajectory
 * waypoint without any kinematic computation — FK is done once when the
 * message arrives, this class just bridges Eigen → Ogre for Robot::update().
 */
class CuroboLinkUpdater : public rviz_default_plugins::robot::LinkUpdater
{
public:
  explicit CuroboLinkUpdater(
    const std::map<std::string, Eigen::Isometry3d> & transforms);

  bool getLinkTransforms(
    const std::string & link_name,
    Ogre::Vector3 & visual_position,
    Ogre::Quaternion & visual_orientation,
    Ogre::Vector3 & collision_position,
    Ogre::Quaternion & collision_orientation) const override;

private:
  const std::map<std::string, Eigen::Isometry3d> & transforms_;
};

}  // namespace isaac_ros_cumotion_rviz

#endif  // ISAAC_ROS_CUMOTION_RVIZ__CUROBO_LINK_UPDATER_HPP_
