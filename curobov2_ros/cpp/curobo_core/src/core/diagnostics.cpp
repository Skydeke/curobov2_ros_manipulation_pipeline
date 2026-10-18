// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/core/diagnostics.py.
#include "curobo_core/core/diagnostics.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

namespace curobo_core
{

namespace
{
std::string timestamp()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&t, &tm);
  std::ostringstream out;
  out << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return out.str();
}
}  // namespace

// PORT: diagnostics.resolve_diag_dir
std::string resolve_diag_dir(rclcpp::Node * node)
{
  if (!node->has_parameter("diagnostic_csv_dir")) {
    node->declare_parameter("diagnostic_csv_dir", "");
  }
  const std::string override = node->get_parameter("diagnostic_csv_dir").as_string();
  if (!override.empty()) {
    return override;
  }
  return rclcpp::get_logging_directory().string();
}

// PORT: diagnostics.DiagCsv.__init__
DiagCsv::DiagCsv(const std::string & path)
{
  file_.open(path);
}

// PORT: diagnostics.DiagCsv.writerow
void DiagCsv::writerow(const std::vector<std::string> & row)
{
  try {
    for (std::size_t i = 0; i < row.size(); ++i) {
      if (i != 0) {
        file_ << ',';
      }
      // CSV-escape exactly like Python's csv.writer (quote when needed).
      const std::string & cell = row[i];
      const bool needs_quote = cell.find_first_of(",\"\n\r") != std::string::npos;
      if (needs_quote) {
        file_ << '"';
        for (char c : cell) {
          if (c == '"') {
            file_ << '"';
          }
          file_ << c;
        }
        file_ << '"';
      } else {
        file_ << cell;
      }
    }
    file_ << '\n';
    file_.flush();
  } catch (const std::exception &) {
    // PORT: a diagnostic sink must not break the control loop it observes.
  }
}

// PORT: diagnostics.DiagCsv.write_header_once
bool DiagCsv::write_header_once(const std::vector<std::string> & header)
{
  if (header_written_) {
    return false;
  }
  writerow(header);
  header_written_ = true;
  return true;
}

void DiagCsv::close()
{
  try {
    if (file_.is_open()) {
      file_.close();
    }
  } catch (const std::exception &) {
  }
}

// PORT: diagnostics.open_diag_csv
std::shared_ptr<DiagCsv> open_diag_csv(rclcpp::Node * node, const std::string & prefix)
{
  try {
    const std::string dir = resolve_diag_dir(node);
    ::mkdir(dir.c_str(), 0755);
    const std::string path = dir + "/" + prefix + "_" + timestamp() + ".csv";
    auto diag = std::make_shared<DiagCsv>(path);
    RCLCPP_INFO(node->get_logger(), "Diagnostic CSV -> %s", path.c_str());
    return diag;
  } catch (const std::exception & e) {
    RCLCPP_WARN(node->get_logger(), "Diagnostic CSV (%s) init failed: %s",
      prefix.c_str(), e.what());
    return nullptr;
  }
}

}  // namespace curobo_core
