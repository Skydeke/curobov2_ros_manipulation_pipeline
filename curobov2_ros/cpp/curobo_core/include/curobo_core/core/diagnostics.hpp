// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/diagnostics.py
//
// Diagnostic CSV writer shared by the MPC controller and the joint-speed
// strategy. Gated by their own ROS parameters (off by default); the file goes to
// the `diagnostic_csv_dir` parameter or the ROS logging directory. A diagnostic
// sink must never break the control loop it observes.
#ifndef CUROBO_CORE__CORE__DIAGNOSTICS_HPP_
#define CUROBO_CORE__CORE__DIAGNOSTICS_HPP_

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace curobo_core
{

// PORT: diagnostics.resolve_diag_dir
std::string resolve_diag_dir(rclcpp::Node * node);

// PORT: diagnostics.DiagCsv
class DiagCsv
{
public:
  explicit DiagCsv(const std::string & path);

  // PORT: DiagCsv.writerow (never raises; flushes every row)
  void writerow(const std::vector<std::string> & row);
  // PORT: DiagCsv.write_header_once
  bool write_header_once(const std::vector<std::string> & header);
  void close();

private:
  std::ofstream file_;
  bool header_written_ = false;
};

// PORT: diagnostics.open_diag_csv
std::shared_ptr<DiagCsv> open_diag_csv(rclcpp::Node * node, const std::string & prefix);
}  // namespace curobo_core

#endif  // CUROBO_CORE__CORE__DIAGNOSTICS_HPP_
