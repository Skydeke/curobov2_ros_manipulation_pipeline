/* Desc: rviz panel for curobo_task_constructor tasks.
 *
 * MTC TaskPanel/TaskView equivalent on the
 * curobo_task_constructor_interfaces wire format (mirroring
 * moveit_task_constructor_msgs): TaskPanel toolbar with Exec + "..." tool
 * buttons (task_panel.ui, verbatim) plus Tasks View / Global Settings
 * sub-panel toggles; TaskView with Task Tree beside a (multi-select,
 * sortable) solutions tree and a Properties pane; GlobalSettingsWidget with
 * the Task View Settings. Planning is local (in-process, like MoveIt); the
 * view renders TaskDescription/TaskStatistics/Solution introspection,
 * fetches full solutions via per-task GetSolution, and drives the selected
 * solution through ExecuteTaskSolution with no replanning, like MTC's
 * Execute solution button. The selected solution is additionally published
 * as trajectory_msgs/JointTrajectory for the CuroboTrajectoryDisplay.
 *
 * Class map to moveit_task_constructor (same names, same roles, same slot
 * names): SubPanel / TaskConstructorPanel (the TaskPanel; keeps its
 * established plugin name for saved-layout compat) / TaskView (with the
 * TaskExpand/OldTaskHandling enums and the initial_task_expand /
 * old_task_handling / show_time_column settings) / GlobalSettingsWidget.
 * Deviations, all documented at the use site: no TaskDisplay exists in this
 * package, so TaskView owns the ROS node, the introspection subscriptions
 * and the trajectory/marker publishers; the execute action name is absolute
 * (deployment rendezvous); sub-panel toggle buttons carry text (this
 * package ships no icon resources); usernames for tree views are kept as
 * QTreeWidgets with an MTC-order comparator instead of the model/view
 * classes.
 */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rviz_common/config.hpp>
#include <rviz_common/panel.hpp>

#include <curobo_task_constructor_interfaces/action/execute_task_solution.hpp>
#include <curobo_task_constructor_interfaces/msg/solution.hpp>
#include <curobo_task_constructor_interfaces/msg/stage_description.hpp>
#include <curobo_task_constructor_interfaces/msg/stage_statistics.hpp>
#include <curobo_task_constructor_interfaces/msg/task_description.hpp>
#include <curobo_task_constructor_interfaces/msg/task_statistics.hpp>
#include <curobo_task_constructor_interfaces/srv/get_solution.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <QList>
#include <QMap>
#include <QPair>
#include <QSet>
#include <QString>
#include <QButtonGroup>
#include <QStackedWidget>
#include <QWidget>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <thread>
#include <vector>

namespace rviz_common {
class WindowManagerInterface;
namespace properties {
class Property;
class BoolProperty;
class EnumProperty;
class PropertyTreeModel;
class PropertyTreeWidget;
}  // namespace properties
}  // namespace rviz_common

class QAction;
class QHBoxLayout;
class QLabel;
class QSplitter;
class QTimer;
class QToolButton;
class QTreeView;
class QTreeWidget;

#include <QTreeWidgetItem>

namespace curobo_task_constructor_rviz
{

using ExecuteAction = curobo_task_constructor_interfaces::action::ExecuteTaskSolution;
using SolutionMsg = curobo_task_constructor_interfaces::msg::Solution;

/// Base class for all sub panels within the Task Panel (MTC SubPanel).
class SubPanel : public QWidget
{
  Q_OBJECT
public:
  SubPanel(QWidget* parent = nullptr) : QWidget(parent) {}

  virtual void save(rviz_common::Config /*config*/) {}
  virtual void load(const rviz_common::Config& /*config*/) {}
Q_SIGNALS:
  void configChanged();
};

class TaskConstructorPanelPrivate;

/// The TaskPanel is the central panel of this plugin, collecting various
/// sub panels (MTC TaskPanel; keeps the established plugin class name).
class TaskConstructorPanel : public rviz_common::Panel
{
  Q_OBJECT
  Q_DECLARE_PRIVATE(TaskConstructorPanel)
  TaskConstructorPanelPrivate* d_ptr;

public:
  TaskConstructorPanel(QWidget* parent = nullptr);
  ~TaskConstructorPanel() override;

  /// add a new sub panel widget (MTC TaskPanel::addSubPanel)
  void addSubPanel(SubPanel* w, const QString& title, const QIcon& icon);

  void onInitialize() override;
  void load(const rviz_common::Config& config) override;
  void save(rviz_common::Config config) const override;

protected Q_SLOTS:
  void showStageDockWidget();
};

/// TaskView displays all known tasks (MTC TaskView).
class TaskView : public SubPanel
{
  Q_OBJECT

public:
  using SolutionShared =
      curobo_task_constructor_interfaces::msg::Solution::ConstSharedPtr;

  enum TaskExpand : uint8_t
  {
    EXPAND_TOP = 1,
    EXPAND_ALL,
    EXPAND_NONE
  };

  enum OldTaskHandling : uint8_t
  {
    OLD_TASK_KEEP = 1,
    OLD_TASK_REPLACE,
    OLD_TASK_REMOVE
  };

  TaskView(TaskConstructorPanel* parent, rviz_common::properties::Property* root);
  ~TaskView() override;

  void save(rviz_common::Config config) override;
  void load(const rviz_common::Config& config) override;

public Q_SLOTS:
  void addTask();

protected Q_SLOTS:
  void removeSelectedStages();
  void onCurrentStageChanged();
  void onCurrentSolutionChanged();
  void onSolutionSelectionChanged();
  void onExecCurrentSolution();
  void onShowTimeChanged();

private:
  /// One solutions-view row (MTC RemoteSolutionModel::Data equivalent).
  struct SolutionRow
  {
    uint32_t id = 0;
    uint32_t stage_id = 0;
    double cost = std::numeric_limits<double>::quiet_NaN();
    QString comment;
    bool failed = false;
    int rank = 0;
  };

  /// Per-task live data (MTC TaskListModel/RemoteTaskModel equivalent).
  struct TaskData
  {
    curobo_task_constructor_interfaces::msg::TaskDescription::ConstSharedPtr desc;
    QMap<uint32_t, curobo_task_constructor_interfaces::msg::StageStatistics> stats;
    QMap<uint32_t, SolutionRow> rows;
    QMap<uint32_t, SolutionShared> solutions;
    QSet<uint32_t> pending;
    QMap<uint32_t, QTreeWidgetItem*> items;
    QMap<uint32_t, std::vector<curobo_task_constructor_interfaces::msg::Property>> props;
    uint32_t root_id = 0;
    QTreeWidgetItem* top_item = nullptr;
  };

  /// Solutions-view item with MTC sort order (rank column and cost column
  /// sort by solution id — MTC sorts by cost_rank, i.e. arrival order, with
  /// id tiebreak; comment column sorts by comment text, id tiebreak).
  class SolutionRowItem : public QTreeWidgetItem
  {
  public:
    using QTreeWidgetItem::QTreeWidgetItem;
    bool operator<(const QTreeWidgetItem& other) const override;
  };

  void setupUi();
  QTreeWidgetItem* buildDescriptionItem(
      TaskData& data,
      const std::map<uint32_t, const curobo_task_constructor_interfaces::msg::StageDescription*>& by_id,
      const std::map<uint32_t, std::vector<uint32_t>>& children_of, uint32_t id,
      QTreeWidgetItem* parent);
  void updateStageCounts(TaskData& data, uint32_t id);
  int stageOkCount(const TaskData& data, uint32_t id) const;
  int stageFailCount(const TaskData& data, uint32_t id) const;
  void refreshTaskStatistics(const QString& task_id);
  void highlightStageById(const QString& task_id, uint32_t id);
  void clearHighlight();
  void rebuildSolutionList();
  void refreshProperties();
  void syncProperties(const std::vector<std::pair<QString, QString>>& rows);
  void showSelectedSolution();
  void showStageProperties(QTreeWidgetItem* item);
  void showSolutionProperties(const QString& rank, const QString& cost,
                              const QString& comment, const QString& extra);
  void showSolutionPropertiesFor(QTreeWidgetItem* row);

  SolutionShared cachedSolution(const QString& task_id, uint32_t id) const;
  void requestSolution(const QString& task_id, uint32_t id);
  void requestMissingSolutions(const QString& task_id, uint32_t stage_id);
  void onSolutionMessage(SolutionShared msg, bool live = true);
  void onTaskStatistics(
      curobo_task_constructor_interfaces::msg::TaskStatistics::ConstSharedPtr msg);
  void onSolutionReceived(const QString& task_id, uint32_t id, SolutionShared msg);
  void clearPending(const QString& task_id, uint32_t id);

  std::vector<sensor_msgs::msg::JointState> currentWaypoints() const;
  static std::vector<sensor_msgs::msg::JointState> waypointsOf(SolutionShared sol);
  std::vector<visualization_msgs::msg::Marker> selectedMarkers() const;

  void publishSolutionTrajectory(const std::vector<sensor_msgs::msg::JointState>& waypoints);
  void publishMarkers(const std::vector<visualization_msgs::msg::Marker>& markers);
  void setStatus(const QString& text);
  void ensureGetSolutionClient(const std::string& task_id);
  void ensureDataSubs();
  void addOrReplaceTask(
      curobo_task_constructor_interfaces::msg::TaskDescription::ConstSharedPtr msg);
  bool selectedStage(QString& task_id, uint32_t& stage_id) const;
  void applyTaskExpansion(QTreeWidgetItem* top);
  void requestDisplay(const QString& task_id, SolutionShared msg, bool lock);
  void showDisplayed(const QString& task_id, SolutionShared msg);
  void maybePromotePending();
  void onPlaybackTick();

  // configuration settings (MTC TaskView members, verbatim)
  rviz_common::properties::EnumProperty* initial_task_expand;
  rviz_common::properties::EnumProperty* old_task_handling;
  rviz_common::properties::BoolProperty* show_time_column;

  TaskConstructorPanel* panel_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  std::thread spin_thread_;
  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::Subscription<curobo_task_constructor_interfaces::msg::TaskDescription>::SharedPtr sub_task_description_;
  rclcpp::Subscription<curobo_task_constructor_interfaces::msg::TaskStatistics>::SharedPtr sub_task_statistics_;
  rclcpp::Subscription<curobo_task_constructor_interfaces::msg::Solution>::SharedPtr sub_solution_;
  rclcpp::Client<curobo_task_constructor_interfaces::srv::GetSolution>::SharedPtr get_solution_client_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_selected_markers_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_solution_trajectory_;
  rclcpp_action::Client<ExecuteAction>::SharedPtr exec_client_;

  QMap<QString, TaskData> tasks_;
  std::string get_solution_task_id_;
  bool data_subscribed_ = false;
  int pending_exec_id_ = -1;
  QString pending_exec_task_;

  // Ghost display state (MTC TaskSolutionVisualization dynamics).
  QString displayed_task_;
  SolutionShared displayed_msg_ = nullptr;
  QString pending_task_;
  SolutionShared pending_msg_ = nullptr;
  bool display_locked_ = false;
  // Currently highlighted stage (MTC highlighted_row_index_): repaint
  // nothing while it does not move.
  QString highlighted_task_;
  uint32_t highlighted_id_ = 0;
  QSet<QPair<QString, int>> displayed_selected_;
  QTimer* playback_timer_ = nullptr;
  SolutionShared playback_sol_ = nullptr;
  QString playback_task_;
  size_t playback_index_ = 0;

  QSplitter* tasks_property_splitter_ = nullptr;
  QSplitter* tasks_solutions_splitter_ = nullptr;
  QLabel* tasks_view_label_ = nullptr;
  QTreeWidget* tasks_view_ = nullptr;
  QTreeWidget* solutions_view_ = nullptr;
  QLabel* property_view_label_ = nullptr;
  QTreeWidget* property_view_ = nullptr;
  QAction* actionRemoveTaskTreeRows_ = nullptr;
  QAction* actionAddLocalTask_ = nullptr;
  QAction* actionShowTimeColumn_ = nullptr;
  // Status readout (no MTC counterpart in the panel; the Display owns
  // status there). Kept: execution/fetch failures would otherwise be
  // silent in a display-less panel.
  QLabel* status_label_ = nullptr;
};

/// Global settings widget (MTC GlobalSettingsWidget).
class GlobalSettingsWidget : public SubPanel
{
  Q_OBJECT

public:
  GlobalSettingsWidget(TaskConstructorPanel* parent, rviz_common::properties::Property* root);
  ~GlobalSettingsWidget() override;

  void save(rviz_common::Config config) override;
  void load(const rviz_common::Config& config) override;

private:
  rviz_common::properties::Property* root_;
  rviz_common::properties::PropertyTreeModel* properties_ = nullptr;
  rviz_common::properties::PropertyTreeWidget* view_ = nullptr;
};

}  // namespace curobo_task_constructor_rviz
