#pragma once

#ifndef ISAAC_ROS_CUMOTION_RVIZ__CUROBO_FK_HPP_
#define ISAAC_ROS_CUMOTION_RVIZ__CUROBO_FK_HPP_

#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>

namespace isaac_ros_cumotion_rviz
{

/**
 * URDF-based forward kinematics engine.
 *
 * Parses a URDF and computes global link transforms for any joint configuration.
 * No MoveIt, no KDL, no cuRobo dependency — pure urdf::Model + Eigen.
 */
class CuroboFK
{
public:
  CuroboFK() = default;

  /** Parse URDF and build the kinematic tree. Returns false on failure. */
  bool initFromString(const std::string & urdf_xml);

  /** Compute global link transforms for the given joint positions.
   *  joint_positions: map of joint_name -> angle (radians) for revolute joints.
   *  link_transforms: output map of link_name -> Isometry3d (global pose).
   *  Returns false if the model is not initialised. */
  bool computeFK(
    const std::map<std::string, double> & joint_positions,
    std::map<std::string, Eigen::Isometry3d> & link_transforms) const;

  const std::vector<std::string> & getLinkNames() const { return link_names_; }
  const std::vector<std::string> & getJointNames() const { return joint_names_; }

private:
  struct JointInfo
  {
    std::string name;
    std::string parent_link;
    std::string child_link;
    int type;  // urdf::Joint::REVOLUTE / CONTINUOUS / PRISMATIC / FIXED, as int
    Eigen::Vector3d axis{0, 0, 0};  // local joint axis (normalised for revolute)
    Eigen::Isometry3d origin{Eigen::Isometry3d::Identity()};  // parent -> joint frame
  };

  struct LinkInfo
  {
    std::string name;
    std::string parent_joint;       // empty for root
    std::vector<std::string> child_joints;
  };

  bool initialised_ = false;
  std::string root_link_;
  std::map<std::string, LinkInfo> links_;
  std::map<std::string, JointInfo> joints_;
  std::vector<std::string> link_names_;
  std::vector<std::string> joint_names_;
};

}  // namespace isaac_ros_cumotion_rviz

#endif  // ISAAC_ROS_CUMOTION_RVIZ__CUROBO_FK_HPP_
