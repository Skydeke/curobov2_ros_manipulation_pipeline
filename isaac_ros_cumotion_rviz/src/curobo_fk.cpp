#include <algorithm>
#include <queue>

#include <isaac_ros_cumotion_rviz/curobo_fk.hpp>

namespace isaac_ros_cumotion_rviz
{

bool CuroboFK::initFromString(const std::string & urdf_xml)
{
  links_.clear();
  joints_.clear();
  link_names_.clear();
  joint_names_.clear();
  initialised_ = false;

  urdf::ModelInterfaceSharedPtr model = urdf::parseURDF(urdf_xml);
  if (!model) {
    return false;
  }

  // Build link map
  for (const auto & [name, link] : model->links_) {
    LinkInfo li;
    li.name = name;
    if (link->parent_joint) {
      li.parent_joint = link->parent_joint->name;
    }
    links_[name] = li;
    link_names_.push_back(name);
  }

  // Build joint map and populate child_joints on parents
  for (const auto & [name, joint] : model->joints_) {
    JointInfo ji;
    ji.name = name;
    ji.parent_link = joint->parent_link_name;
    ji.child_link = joint->child_link_name;
    ji.type = static_cast<int>(joint->type);

    // Origin: parent_link_frame -> joint_frame
    ji.origin = Eigen::Isometry3d::Identity();
    ji.origin.translation() = Eigen::Vector3d(
      joint->parent_to_joint_origin_transform.position.x,
      joint->parent_to_joint_origin_transform.position.y,
      joint->parent_to_joint_origin_transform.position.z);
    const auto & q = joint->parent_to_joint_origin_transform.rotation;
    Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
    ji.origin.linear() = quat.toRotationMatrix();

    // Axis (only meaningful for revolute/prismatic)
    ji.axis = Eigen::Vector3d(
      joint->axis.x, joint->axis.y, joint->axis.z);
    if (ji.axis.squaredNorm() > 1e-12) {
      ji.axis.normalize();
    }

    joints_[name] = ji;
    joint_names_.push_back(name);

    auto parent_it = links_.find(ji.parent_link);
    if (parent_it != links_.end()) {
      parent_it->second.child_joints.push_back(name);
    }
  }

  // Determine root link (the one with no parent_joint)
  root_link_.clear();
  for (const auto & [name, link] : links_) {
    if (link.parent_joint.empty()) {
      root_link_ = name;
      break;
    }
  }
  if (root_link_.empty()) {
    return false;
  }

  initialised_ = true;
  return true;
}

bool CuroboFK::computeFK(
  const std::map<std::string, double> & joint_positions,
  std::map<std::string, Eigen::Isometry3d> & link_transforms) const
{
  if (!initialised_) {
    return false;
  }

  link_transforms.clear();

  // BFS from root
  std::queue<std::string> queue;
  link_transforms[root_link_] = Eigen::Isometry3d::Identity();
  queue.push(root_link_);

  while (!queue.empty()) {
    const std::string current_link = queue.front();
    queue.pop();

    const auto & link_info = links_.at(current_link);
    const Eigen::Isometry3d & T_parent = link_transforms.at(current_link);

    for (const std::string & joint_name : link_info.child_joints) {
      const auto & ji = joints_.at(joint_name);

      // T_child = T_parent * origin * joint_motion
      Eigen::Isometry3d T_child = T_parent * ji.origin;

      switch (ji.type) {
        case static_cast<int>(urdf::Joint::REVOLUTE):
        case static_cast<int>(urdf::Joint::CONTINUOUS): {
          auto it = joint_positions.find(joint_name);
          double angle = (it != joint_positions.end()) ? it->second : 0.0;
          T_child.rotate(Eigen::AngleAxisd(angle, ji.axis));
          break;
        }
        case static_cast<int>(urdf::Joint::PRISMATIC): {
          auto it = joint_positions.find(joint_name);
          double displacement = (it != joint_positions.end()) ? it->second : 0.0;
          // Axis is in the joint frame (after the origin transform); applying it
          // with pretranslate() is equivalent to right-multiplying by a
          // translation, i.e. displacement along the joint frame's axis.
          T_child.pretranslate(ji.axis * displacement);
          break;
        }
        case static_cast<int>(urdf::Joint::FIXED):
        default:
          break;
      }

      link_transforms[ji.child_link] = T_child;
      queue.push(ji.child_link);
    }
  }

  return true;
}

}  // namespace isaac_ros_cumotion_rviz
