/* Desc: rviz panel for curobo_task_constructor tasks.
 *
 * MTC TaskPanel/TaskView equivalent on the
 * curobo_task_constructor_interfaces wire format: a toolbar with an Exec
 * tool button, a Task Tree beside a (multi-select, sortable) solutions
 * tree, and a Properties pane. The selected solution is published as
 * trajectory_msgs/JointTrajectory for the CuroboTrajectoryDisplay, which
 * animates the full robot through it — the same topic Task.publish uses.
 * Exec drives the selected solution (complete chain or single attempt)
 * with no replanning, like MTC's Execute solution button.
 */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rviz_common/config.hpp>
#include <rviz_common/panel.hpp>

#include <curobo_task_constructor_interfaces/action/execute_task_solution.hpp>
#include <curobo_task_constructor_interfaces/msg/solution_info.hpp>
#include <curobo_task_constructor_interfaces/msg/stage_spec.hpp>
#include <curobo_task_constructor_interfaces/msg/stage_statistics.hpp>
#include <curobo_task_constructor_interfaces/msg/task_description.hpp>
#include <curobo_task_constructor_interfaces/msg/task_solution.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <QList>
#include <QMap>
#include <QString>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

class QAction;
class QLabel;
class QSplitter;
class QToolButton;
class QTreeWidget;
class QTreeWidgetItem;

namespace curobo_task_constructor_rviz
{

using ExecuteAction = curobo_task_constructor_interfaces::action::ExecuteTaskSolution;
using SolutionInfoMsg = curobo_task_constructor_interfaces::msg::SolutionInfo;

/// Task tree + solutions tree + properties, like MTC's TaskView.
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
  void refreshTaskDescription();
  void refreshSolutionInfo();
  void refreshStageStatistics();
  void refreshChains();
  void setStatus(const QString & text);

  void onSelectedItemChanged();
  void onSolutionSelectionChanged();
  void onExecSolution();
  void onShowTimeChanged();

private:
  using SolutionInfoShared =
      curobo_task_constructor_interfaces::msg::SolutionInfo::ConstSharedPtr;

  void setupUi();
  QTreeWidgetItem * buildSpecItem(
      const std::map<uint32_t, const curobo_task_constructor_interfaces::msg::StageSpec *> & by_id,
      const std::map<uint32_t, std::vector<uint32_t>> & children_of,
      uint32_t id, QTreeWidgetItem * parent);
  void applyStoredData();
  void updateStageCounts(uint32_t id);
  int stageOkCount(uint32_t id) const;
  int stageFailCount(uint32_t id) const;
  void applySolutionToItem(QTreeWidgetItem * item, const SolutionInfoMsg & sol);
  void highlightStage(const QString & name);
  void clearHighlight();
  bool showingChains() const;
  void rebuildSolutionList();
  void refreshProperties();
  /// Rewrite the properties pane in place (same rows, fresh values) so new
  /// introspection data ticks the displayed values live without collapsing
  /// the user's expanded rows or needing a reselection.
  void syncProperties(const std::vector<std::pair<QString, QString>> & rows);
  void showSelectedSolution();
  void showStageProperties(QTreeWidgetItem * item);
  void showSolutionProperties(const QString & rank, const QString & cost,
                              const QString & comment, const QString & extra);
  void showSolutionPropertiesFor(QTreeWidgetItem * row);

  /// Waypoints behind the current solutions-view selection.
  SolutionInfoShared currentAttempt(QTreeWidgetItem * row) const;
  SolutionInfoShared findAttempt(uint32_t stage_id, uint32_t solution_id) const;
  std::vector<SolutionInfoShared> selectedAttempts(
      const QList<QTreeWidgetItem *> & selected) const;
  std::vector<sensor_msgs::msg::JointState> currentWaypoints() const;

  void publishSolutionTrajectory(const QList<QTreeWidgetItem *> & selected);
  void publishToolPath(const QList<QTreeWidgetItem *> & selected);

  rclcpp::Node::SharedPtr node_;
  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::Subscription<curobo_task_constructor_interfaces::msg::TaskDescription>::SharedPtr sub_task_description_;
  rclcpp::Subscription<curobo_task_constructor_interfaces::msg::SolutionInfo>::SharedPtr sub_solution_info_;
  rclcpp::Subscription<curobo_task_constructor_interfaces::msg::StageStatistics>::SharedPtr sub_stage_statistics_;
  rclcpp::Subscription<curobo_task_constructor_interfaces::msg::TaskSolution>::SharedPtr sub_task_solutions_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_selected_markers_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_solution_trajectory_;
  rclcpp_action::Client<ExecuteAction>::SharedPtr exec_client_;

  // The one task on display (a new description replaces it).
  curobo_task_constructor_interfaces::msg::TaskDescription::ConstSharedPtr task_;
  QMap<uint32_t, std::vector<SolutionInfoShared>> attempts_;
  QMap<uint32_t, curobo_task_constructor_interfaces::msg::StageStatistics::ConstSharedPtr> stats_;
  // Complete ranked solutions (MTC root-container solutions).
  std::vector<curobo_task_constructor_interfaces::msg::TaskSolution::ConstSharedPtr> chains_;

  QToolButton * exec_button_;
  QTreeWidget * tree_;
  QTreeWidget * solutions_;
  QTreeWidget * properties_;
  QAction * show_time_action_;
  QLabel * status_label_;

  QMap<uint32_t, QTreeWidgetItem *> stage_item_;
  QTreeWidgetItem * task_root_item_ = nullptr;
};

}  // namespace curobo_task_constructor_rviz
