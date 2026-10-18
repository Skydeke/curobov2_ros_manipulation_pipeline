// SPDX-License-Identifier: Apache-2.0
// PORT: unified_planner_node._format_state_collisions / _format_collision_feedback
#include "curobo_core/core/collision_diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

namespace curobo_core
{
namespace
{
struct Row
{
  double depth;
  std::string a;
  std::string op;
  std::string b;
  std::string suffix;
};

std::string rstrip(std::string text)
{
  while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) {
    text.pop_back();
  }
  return text;
}

// PORT: f"{depth * 1000:.1f} mm {noun}".
std::string millimetres(double depth, const char * noun)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(1) << depth * 1000.0 << " mm " << noun;
  return out.str();
}
}  // namespace

double round6(double value)
{
  return std::round(value * 1e6) / 1e6;
}

std::string format_state_collisions(
  const std::string & label, const curobo_runtime::CollisionAttribution & sc)
{
  std::vector<Row> rows;
  for (const auto & c : sc.world_contacts) {
    rows.push_back({c.depth, c.link, "->", c.obstacle, millimetres(c.depth, "past surface")});
  }
  for (const auto & c : sc.self_contacts) {
    rows.push_back({c.depth, c.link_a, "<->", c.link_b, millimetres(c.depth, "penetration")});
  }
  for (const auto & c : sc.cspace_violations) {
    std::ostringstream detail;
    detail << "outside [" << std::fixed << std::setprecision(3) << c.lower << ", "
           << c.upper << "] (value " << c.value << ")";
    rows.push_back(
      {-std::numeric_limits<double>::infinity(), c.joint, "limit", detail.str(), ""});
  }
  if (rows.empty()) {
    return "  " + label + ": clear (no world/self contacts, within joint limits)";
  }
  // Deepest first; -inf joint-limit rows sort last. Stable so equal depths keep
  // the world -> self -> cspace order.
  std::stable_sort(
    rows.begin(), rows.end(),
    [](const Row & a, const Row & b) {return a.depth > b.depth;});
  std::size_t width = 0;
  for (const auto & row : rows) {
    width = std::max(width, row.a.size());
  }
  std::ostringstream out;
  out << "  " << label << ": " << rows.size() << " "
      << (rows.size() == 1 ? "contact" : "contacts");
  for (const auto & row : rows) {
    std::ostringstream line;
    line << "    " << std::left << std::setw(static_cast<int>(width)) << row.a
         << "  " << row.op << "  " << row.b << "  " << row.suffix;
    out << "\n" << rstrip(line.str());
  }
  return out.str();
}

std::string format_collision_feedback(
  const curobo_runtime::CollisionAttribution & start,
  const curobo_runtime::CollisionAttribution & goal)
{
  return "collision feedback:\n" + format_state_collisions("start", start) + "\n" +
         format_state_collisions("goal", goal);
}

}  // namespace curobo_core
