// SPDX-License-Identifier: Apache-2.0
// Runtime-agnostic planning request adapters.
//
// This is the cuRobo counterpart of MoveIt's PlanningRequestAdapter: adapters
// run on the already-parsed request before the planner runtime is chosen, so
// cpp and python see exactly the same adapted request. No wire message is
// changed and no client knows adapters exist.
#ifndef CUROBO_CORE__ADAPTERS__PLANNING_REQUEST_ADAPTER_HPP_
#define CUROBO_CORE__ADAPTERS__PLANNING_REQUEST_ADAPTER_HPP_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <rclcpp/rclcpp.hpp>

#include "curobov2_ros_interfaces/msg/trajectory_goal.hpp"

namespace curobo_core
{
namespace adapters
{

/// Immutable model view plus the mutable request. Built by the server for one
/// planning request; never serialized.
struct RequestContext
{
  curobov2_ros_interfaces::msg::TrajectoryGoal & goal;
  const std::vector<std::string> & joint_names;
  const std::vector<double> & lower;
  const std::vector<double> & upper;
  const std::vector<std::uint8_t> & continuous;
  /// Start resolved into cspace order (may be empty when the request has no
  /// start and the server has no current pose yet).
  const std::vector<float> & start;
  /// Optional resolver matching the server's own TF resolution. Adapters should
  /// use it instead of doing their own TF lookup.
  std::function<std::vector<geometry_msgs::msg::Pose>(
      const std::vector<geometry_msgs::msg::Pose> &, const std::string &)> pose_resolver;

  std::string error;

  std::size_t index_of(const std::string & name) const
  {
    for (std::size_t i = 0; i < joint_names.size(); ++i) {
      if (joint_names[i] == name) {
        return i;
      }
    }
    return joint_names.size();
  }

  bool reject(const std::string & reason)
  {
    error = reason;
    return false;
  }
};

/// Adapter base class. Plugins are loaded through pluginlib and run in order.
class PlanningRequestAdapter
{
public:
  virtual ~PlanningRequestAdapter() = default;

  /// `ns` is the plugin instance namespace; adapters read `<ns>.*` parameters.
  virtual void initialize(rclcpp::Node * node, const std::string & ns) = 0;

  virtual std::string getDescription() const = 0;

  /// Mutate `context.goal` (and/or the context) and return true to continue.
  /// Return false with `context.error` set to reject the request.
  virtual bool adapt(RequestContext & context) = 0;
};

}  // namespace adapters
}  // namespace curobo_core

#endif  // CUROBO_CORE__ADAPTERS__PLANNING_REQUEST_ADAPTER_HPP_
