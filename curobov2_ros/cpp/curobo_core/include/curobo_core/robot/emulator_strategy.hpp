// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/robot/emulator_strategy.py
#ifndef CUROBO_CORE__ROBOT__EMULATOR_STRATEGY_HPP_
#define CUROBO_CORE__ROBOT__EMULATOR_STRATEGY_HPP_

#include <atomic>

#include <sensor_msgs/msg/joint_state.hpp>
#include <thread>

#include "curobo_core/robot/joint_control_strategy.hpp"

namespace curobo_core
{

class EmulatorStrategy : public JointCommandStrategy
{
public:
  EmulatorStrategy(
    rclcpp::Node * node, double dt, std::vector<std::string> joint_names,
    std::map<std::string, std::string> params = {},
    std::vector<double> default_home = {});
  // PORT: EmulatorStrategy.__del__ (stop + join the playback thread).
  ~EmulatorStrategy() override;

  // PORT: EmulatorStrategy._publish_state
  void publish_state(
    const std::vector<std::string> & names, const std::vector<double> & positions,
    const std::vector<double> & velocities);

  // PORT: EmulatorStrategy.send_trajectrory
  void send_trajectrory() override;
  // PORT: EmulatorStrategy.get_joint_pose
  std::vector<double> get_joint_pose() override;
  // PORT: EmulatorStrategy.stop_robot
  void stop_robot() override;
  // PORT: EmulatorStrategy.get_progression
  double get_progression() override;

  // PORT: EmulatorStrategy.wait_for_execution_complete
  bool wait_for_execution_complete(double timeout_s = 5.0);

private:
  void apply_immediate(long index);
  void execute_trajectory(
    std::vector<std::vector<double>> positions, std::vector<std::vector<double>> velocities,
    std::vector<std::string> names, int epoch);

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_joint_states_;
  std::string joint_states_topic_;
  std::vector<double> current_joint_positions_;
  std::thread execution_thread_;
  std::atomic<bool> stop_execution_{false};
};

}  // namespace curobo_core

#endif  // CUROBO_CORE__ROBOT__EMULATOR_STRATEGY_HPP_
