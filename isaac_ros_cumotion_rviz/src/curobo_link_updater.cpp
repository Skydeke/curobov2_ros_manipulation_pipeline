#include <isaac_ros_cumotion_rviz/curobo_link_updater.hpp>

namespace isaac_ros_cumotion_rviz
{

CuroboLinkUpdater::CuroboLinkUpdater(
  const std::map<std::string, Eigen::Isometry3d> & transforms)
  : transforms_(transforms)
{
}

bool CuroboLinkUpdater::getLinkTransforms(
  const std::string & link_name,
  Ogre::Vector3 & visual_position,
  Ogre::Quaternion & visual_orientation,
  Ogre::Vector3 & collision_position,
  Ogre::Quaternion & collision_orientation) const
{
  auto it = transforms_.find(link_name);
  if (it == transforms_.end()) {
    return false;
  }

  const Eigen::Isometry3d & t = it->second;
  const Eigen::Vector3d & p = t.translation();
  Eigen::Quaterniond q(t.rotation());

  visual_position = Ogre::Vector3(
    static_cast<Ogre::Real>(p.x()),
    static_cast<Ogre::Real>(p.y()),
    static_cast<Ogre::Real>(p.z()));
  visual_orientation = Ogre::Quaternion(
    static_cast<Ogre::Real>(q.w()),
    static_cast<Ogre::Real>(q.x()),
    static_cast<Ogre::Real>(q.y()),
    static_cast<Ogre::Real>(q.z()));

  collision_position = visual_position;
  collision_orientation = visual_orientation;
  return true;
}

}  // namespace isaac_ros_cumotion_rviz
