// SPDX-License-Identifier: Apache-2.0
#include "curobo_core/adapters/adapter_chain.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace curobo_core
{
namespace adapters
{
namespace
{
std::string parameter_namespace(const std::string & plugin_name)
{
  std::string out = plugin_name;
  for (char & c : out) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.')) {
      c = '_';
    }
  }
  return out;
}
}  // namespace

AdapterChain::AdapterChain(
  rclcpp::Node * node, curobo_runtime::IBridge * bridge,
  curobo_runtime::RobotHandle handle)
: node_(node)
{
  if (node_ == nullptr) {
    throw std::invalid_argument("AdapterChain requires a node");
  }

  const auto limits = bridge->joint_limits(handle);
  joint_names_ = limits.joint_names;
  const std::size_t dof = joint_names_.size();
  if (limits.positions.data.size() >= 2 * dof) {
    lower_.assign(limits.positions.data.begin(), limits.positions.data.begin() + dof);
    upper_.assign(limits.positions.data.begin() + dof, limits.positions.data.end());
  } else {
    lower_.assign(dof, 0.0);
    upper_.assign(dof, 0.0);
  }
  continuous_ = limits.continuous;

  std::vector<std::string> names;
  if (node_->has_parameter("request_adapters")) {
    names = node_->get_parameter("request_adapters").as_string_array();
  } else {
    names = node_->declare_parameter<std::vector<std::string>>(
      "request_adapters", std::vector<std::string>{});
  }
  if (names.empty()) {
    return;
  }

  loader_ = std::make_unique<pluginlib::ClassLoader<PlanningRequestAdapter>>(
    "curobov2_ros", "curobo_core::adapters::PlanningRequestAdapter");
  for (const auto & name : names) {
    try {
      auto adapter = loader_->createUniqueInstance(name);
      adapter->initialize(node_, parameter_namespace(name));
      RCLCPP_INFO(
        node_->get_logger(), "Loaded planning request adapter '%s' (%s)",
        name.c_str(), adapter->getDescription().c_str());
      adapters_.push_back(std::move(adapter));
    } catch (const pluginlib::PluginlibException & e) {
      throw std::runtime_error(
              "Failed to load planning request adapter '" + name + "': " + e.what());
    }
  }
}

AdapterChain::~AdapterChain() = default;

bool AdapterChain::adapt(
  curobov2_ros_interfaces::msg::TrajectoryGoal & goal,
  const std::vector<float> & start,
  const std::function<std::vector<geometry_msgs::msg::Pose>(
    const std::vector<geometry_msgs::msg::Pose> &, const std::string &)> & pose_resolver,
  std::string & error)
{
  RequestContext context{
    goal, joint_names_, lower_, upper_, continuous_, start, pose_resolver, ""};
  for (auto & adapter : adapters_) {
    context.error.clear();
    if (!adapter->adapt(context)) {
      error = context.error.empty() ?
        ("planning request adapter '" + adapter->getDescription() + "' rejected the request") :
        context.error;
      return false;
    }
  }
  error.clear();
  return true;
}

}  // namespace adapters
}  // namespace curobo_core
