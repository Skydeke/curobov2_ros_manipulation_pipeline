/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2026
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Desc: rviz panel for curobo_task_constructor tasks.
 *
 * Structurally mirrors moveit_task_constructor_visualization's TaskPanel
 * (pluginlib-registered rviz_common::Panel) but only knows the
 * curobo_task_constructor_interfaces wire format — it never touches
 * isaac_ros_cumotion_interfaces or the existing marker publishers.
 */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rviz_common/config.hpp>
#include <rviz_common/panel.hpp>

#include <curobo_task_constructor_interfaces/action/task.hpp>
#include <curobo_task_constructor_interfaces/msg/solution_info.hpp>
#include <curobo_task_constructor_interfaces/msg/stage_spec.hpp>
#include <curobo_task_constructor_interfaces/msg/stage_statistics.hpp>
#include <curobo_task_constructor_interfaces/msg/task_description.hpp>

#include <visualization_msgs/msg/marker_array.hpp>

#include <QMap>
#include <QString>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

class QCheckBox;
class QLabel;
class QPushButton;
class QTreeWidget;
class QTreeWidgetItem;

namespace curobo_task_constructor_rviz
{

using TaskAction = curobo_task_constructor_interfaces::action::Task;

/// Browse curobo_task_constructor tasks: render the StageSpec tree, color
/// stages by their last attempted solution (success/failure), show the
/// selected stage's debug markers and re-request task execution.
class TaskConstructorPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  TaskConstructorPanel(QWidget * parent = nullptr);
  ~TaskConstructorPanel() override;

  void onInitialize() override;
  void load(const rviz_common::Config & config) override;
  void save(rviz_common::Config config) const override;

private Q_SLOTS:
  /// GUI-thread refresh points (ROS callbacks marshal data here via
  /// QMetaObject::invokeMethod with a QueuedConnection; rviz's ROS callbacks
  /// may run on a different thread than the widgets).
  void refreshTaskDescription();
  void refreshSolutionInfo();
  void refreshStageStatistics();
  void setStatus(const QString & text);

  void onSelectedItemChanged();
  void onReexecuteClicked();

private:
  /// Build the widget tree (labels, tree widget, re-execute controls).
  void setupUi();

  /// Re-link one node of the flat StageSpec[] from TaskDescription and
  /// recursively append its rows, keyed by the wire stage ids (the executor
  /// assigns those ids in pre-order, so SolutionInfo/StageStatistics line up).
  QTreeWidgetItem * buildSpecItem(
      const std::map<uint32_t, const curobo_task_constructor_interfaces::msg::StageSpec *> & by_id,
      const std::map<uint32_t, std::vector<uint32_t>> & children_of,
      uint32_t id, QTreeWidgetItem * parent);

  void applyStoredData();
  void applySolutionToItem(QTreeWidgetItem * item,
                           const curobo_task_constructor_interfaces::msg::SolutionInfo & sol);
  void publishSelectedMarkers();
  void reexecute(bool execute);

  rclcpp::Node::SharedPtr node_;
  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::Subscription<curobo_task_constructor_interfaces::msg::TaskDescription>::SharedPtr sub_task_description_;
  rclcpp::Subscription<curobo_task_constructor_interfaces::msg::SolutionInfo>::SharedPtr sub_solution_info_;
  rclcpp::Subscription<curobo_task_constructor_interfaces::msg::StageStatistics>::SharedPtr sub_stage_statistics_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_selected_markers_;
  rclcpp_action::Client<TaskAction>::SharedPtr action_client_;

  QTreeWidget * tree_;
  QLabel * status_label_;
  QPushButton * reexecute_button_;
  QCheckBox * execute_checkbox_;

  /// Latest TaskDescription: the built graph as a flat StageSpec[] list
  /// (root = the stage with parent_id == id) used for both rendering and
  /// re-execution goals.
  curobo_task_constructor_interfaces::msg::TaskDescription::ConstSharedPtr last_task_;

  /// stage_id -> QTreeWidgetItem (stage ids mirror executor.init() ordering).
  QMap<uint32_t, QTreeWidgetItem *> stage_item_;
  /// stage_id -> last attempted solution (success AND failure), for coloring.
  QMap<uint32_t, curobo_task_constructor_interfaces::msg::SolutionInfo::ConstSharedPtr> last_solution_;
  /// stage_id -> rollup statistics, for the numeric columns.
  QMap<uint32_t, curobo_task_constructor_interfaces::msg::StageStatistics::ConstSharedPtr> stage_stats_;
  /// stage_id -> debug markers from the last successful attempt.
  QMap<uint32_t, visualization_msgs::msg::MarkerArray::ConstSharedPtr> stage_markers_;
};

}  // namespace curobo_task_constructor_rviz