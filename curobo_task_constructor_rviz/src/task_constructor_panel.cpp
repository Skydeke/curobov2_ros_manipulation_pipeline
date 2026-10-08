#include "curobo_task_constructor_rviz/task_constructor_panel.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <rmw/qos_profiles.h>

#include <rviz_common/display_context.hpp>

#include <QAction>
#include <QBrush>
#include <QColor>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMetaObject>
#include <QSet>
#include <QSplitter>
#include <QToolButton>
#include <QTreeWidget>
#include <QVariant>
#include <QVBoxLayout>
#include <visualization_msgs/msg/marker.hpp>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace curobo_task_constructor_rviz
{

namespace
{
enum Column
{
  COL_STAGE = 0,
  COL_TYPE,
  COL_OK,
  COL_FAIL,
  COL_COST,
  COL_TIME,
  COL_COUNT
};

enum SolutionColumn
{
  SOL_RANK = 0,
  SOL_COST,
  SOL_COMMENT,
  SOL_COUNT
};

const char * kTopicTaskDescription = "/curobo_task_constructor/task_description";
const char * kTopicSolutionInfo = "/curobo_task_constructor/solution_info";
const char * kTopicStageStatistics = "/curobo_task_constructor/stage_statistics";
const char * kTopicTaskSolutions = "/curobo_task_constructor/task_solutions";
const char * kTopicSelectedMarkers = "/curobo_task_constructor/selected_solution_markers";
const char * kTopicSolutionTrajectory = "/curobo_task_constructor/solution_trajectory";
const char * kActionExecute = "/curobo_task_constructor/execute_task_solution";

rclcpp::QoS introspectionQoS()
{
  // Depth MUST match the node's INTROSPECTION_QOS_DEPTH: per-try progress
  // snapshots plus per-attempt SolutionInfo multiply the burst size well
  // past the stage count, and a shallow history silently drops the early
  // attempts a panel joining mid-plan needs to render the full picture.
  rclcpp::QoS qos(rclcpp::KeepLast(500));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  return qos;
}

QString nameOf(const curobo_task_constructor_interfaces::msg::StageSpec & spec)
{
  return QString::fromStdString(spec.name.empty() ? spec.stage_type : spec.name);
}

QString typeOf(const curobo_task_constructor_interfaces::msg::StageSpec & spec)
{
  if (!spec.container_type.empty()) {
    return QObject::tr("%1 container").arg(QString::fromStdString(spec.container_type));
  }
  return QString::fromStdString(spec.stage_type);
}

using GoalHandleExecute = rclcpp_action::ClientGoalHandle<ExecuteAction>;

/// stage_id value addressing a whole chain rather than one segment.
constexpr uint32_t NO_STAGE = 0xFFFFFFFF;
}  // namespace

TaskConstructorPanel::TaskConstructorPanel(QWidget * parent) : rviz_common::Panel(parent)
{
  setObjectName("CuroboTaskConstructorPanel");
  setupUi();
}

TaskConstructorPanel::~TaskConstructorPanel() {}

void TaskConstructorPanel::setupUi()
{
  auto * layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);

  // Toolbar (MTC task_panel.ui): spacer + Exec tool button.
  auto * tools = new QHBoxLayout();
  tools->setContentsMargins(0, 2, 0, 0);
  tools->addStretch(1);
  exec_button_ = new QToolButton(this);
  exec_button_->setText(tr("Exec"));
  exec_button_->setToolTip(tr("Execute solution"));
  exec_button_->setEnabled(false);
  layout->addLayout(tools);
  tools->addWidget(exec_button_);

  layout->addWidget(new QLabel(tr("Task Tree"), this));

  // Horizontal splitter (MTC task_view.ui): task tree | solutions, 2:1.
  auto * splitter = new QSplitter(Qt::Horizontal, this);
  tree_ = new QTreeWidget(this);
  tree_->setColumnCount(COL_COUNT);
  tree_->setHeaderLabels({tr("Stage"), tr("Type"), QString::fromUtf8("✓"),
                          QString::fromUtf8("✗"), tr("Cost"), tr("Compute time")});
  tree_->setRootIsDecorated(true);
  tree_->setIndentation(15);
  tree_->setUniformRowHeights(true);
  tree_->setAllColumnsShowFocus(true);
  tree_->setAlternatingRowColors(true);
  tree_->setContextMenuPolicy(Qt::ActionsContextMenu);
  tree_->header()->setStretchLastSection(true);
  tree_->headerItem()->setForeground(COL_OK, QColor(Qt::darkGreen));
  tree_->headerItem()->setToolTip(COL_OK, tr("successful solutions"));
  tree_->headerItem()->setForeground(COL_FAIL, QColor(Qt::red));
  tree_->headerItem()->setToolTip(COL_FAIL, tr("failed solution attempts"));
  show_time_action_ = new QAction(tr("ShowTimeColumn"), this);
  show_time_action_->setCheckable(true);
  show_time_action_->setChecked(true);
  show_time_action_->setToolTip(tr("show time column"));
  tree_->addAction(show_time_action_);
  splitter->addWidget(tree_);
  splitter->setStretchFactor(0, 2);

  solutions_ = new QTreeWidget(this);
  solutions_->setColumnCount(SOL_COUNT);
  solutions_->setHeaderLabels({tr("#"), tr("cost"), tr("comment")});
  solutions_->setRootIsDecorated(false);
  solutions_->setUniformRowHeights(true);
  solutions_->setAllColumnsShowFocus(true);
  solutions_->setSelectionMode(QAbstractItemView::ExtendedSelection);
  solutions_->setSelectionBehavior(QAbstractItemView::SelectRows);
  solutions_->setSortingEnabled(true);
  splitter->addWidget(solutions_);
  splitter->setStretchFactor(1, 1);
  layout->addWidget(splitter, /*stretch=*/3);

  layout->addWidget(new QLabel(tr("Properties"), this));
  properties_ = new QTreeWidget(this);
  properties_->setColumnCount(2);
  properties_->setHeaderLabels({tr("property"), tr("value")});
  properties_->setRootIsDecorated(true);
  layout->addWidget(properties_, /*stretch=*/1);

  status_label_ = new QLabel(tr("waiting for task_description..."), this);
  status_label_->setWordWrap(true);
  layout->addWidget(status_label_);

  connect(tree_, &QTreeWidget::currentItemChanged, this,
          [this](QTreeWidgetItem *, QTreeWidgetItem *) { onSelectedItemChanged(); });
  connect(solutions_, &QTreeWidget::itemSelectionChanged, this,
          &TaskConstructorPanel::onSolutionSelectionChanged);
  connect(exec_button_, &QToolButton::clicked, this, &TaskConstructorPanel::onExecSolution);
  connect(show_time_action_, &QAction::toggled, this,
          [this](bool) { onShowTimeChanged(); });
}

void TaskConstructorPanel::onInitialize()
{
  rviz_common::Panel::onInitialize();
  rviz_common::DisplayContext * display_context = getDisplayContext();
  if (display_context == nullptr) {
    setStatus(QStringLiteral("internal error: no display context from rviz"));
    return;
  }
  auto ros_node_abstraction = display_context->getRosNodeAbstraction().lock();
  if (!ros_node_abstraction) {
    setStatus(QStringLiteral("internal error: no ROS node from rviz"));
    return;
  }
  node_ = ros_node_abstraction->get_raw_node();
  cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  const auto qos = introspectionQoS();
  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = cb_group_;

  sub_task_description_ = node_->create_subscription<
      curobo_task_constructor_interfaces::msg::TaskDescription>(
      kTopicTaskDescription, qos,
      [this](curobo_task_constructor_interfaces::msg::TaskDescription::ConstSharedPtr msg) {
        QMetaObject::invokeMethod(
            this, [this, msg]() {
              // An empty description is the "task finished" signal: keep
              // showing the last task, like MTC keeps old tasks visible.
              if (msg->stages.empty() || msg->stage_count == 0) {
                setStatus(msg->task_id.empty()
                              ? tr("no task — waiting for task_description...")
                              : tr("task '%1' finished")
                                    .arg(QString::fromStdString(msg->task_id)));
                return;
              }
              // One task on display; a new description replaces it.
              task_ = msg;
              attempts_.clear();
              stats_.clear();
              chains_.clear();
              refreshTaskDescription();
            },
            Qt::QueuedConnection);
      },
      sub_options);
  sub_solution_info_ = node_->create_subscription<
      curobo_task_constructor_interfaces::msg::SolutionInfo>(
      kTopicSolutionInfo, qos,
      [this](curobo_task_constructor_interfaces::msg::SolutionInfo::ConstSharedPtr msg) {
        if (!task_ || task_->task_id != msg->task_id) return;
        const uint32_t id = msg->stage_id;
        attempts_[id].push_back(msg);
        QMetaObject::invokeMethod(
            this, [this, id]() {
              // Follow the plan: with nothing selected, select the stage
              // that just reported, so the solutions list tracks the live
              // plan instead of sitting empty. A manual selection sticks.
              if (!tree_->currentItem()) {
                auto item_it = stage_item_.find(id);
                if (item_it != stage_item_.end()) tree_->setCurrentItem(item_it.value());
              }
              refreshSolutionInfo();
            },
            Qt::QueuedConnection);
      },
      sub_options);
  sub_stage_statistics_ = node_->create_subscription<
      curobo_task_constructor_interfaces::msg::StageStatistics>(
      kTopicStageStatistics, qos,
      [this](curobo_task_constructor_interfaces::msg::StageStatistics::ConstSharedPtr msg) {
        if (!task_ || task_->task_id != msg->task_id) return;
        stats_[msg->id] = msg;
        QMetaObject::invokeMethod(this, "refreshStageStatistics", Qt::QueuedConnection);
      },
      sub_options);
  sub_task_solutions_ = node_->create_subscription<
      curobo_task_constructor_interfaces::msg::TaskSolution>(
      kTopicTaskSolutions, qos,
      [this](curobo_task_constructor_interfaces::msg::TaskSolution::ConstSharedPtr msg) {
        if (!task_ || task_->task_id != msg->task_id) return;
        while (chains_.size() <= msg->solution_index) chains_.push_back(nullptr);
        chains_[msg->solution_index] = msg;
        QMetaObject::invokeMethod(this, "refreshChains", Qt::QueuedConnection);
      },
      sub_options);

  pub_selected_markers_ =
      node_->create_publisher<visualization_msgs::msg::MarkerArray>(kTopicSelectedMarkers, qos);
  // Transient-local like Task.publish: a trajectory display that subscribes
  // later still picks up the selected solution.
  rclcpp::QoS traj_qos(rclcpp::KeepLast(1));
  traj_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  traj_qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  pub_solution_trajectory_ = node_->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      kTopicSolutionTrajectory, traj_qos);
  exec_client_ = rclcpp_action::create_client<ExecuteAction>(node_, kActionExecute, cb_group_);
  setStatus(tr("listening on %1").arg(kTopicTaskDescription));
}

void TaskConstructorPanel::load(const rviz_common::Config & config)
{
  rviz_common::Panel::load(config);
  QVariant show_time;
  if (config.mapGetValue("ShowTime", &show_time) && show_time.canConvert<bool>())
    show_time_action_->setChecked(show_time.toBool());
  onShowTimeChanged();
}

void TaskConstructorPanel::save(rviz_common::Config config) const
{
  rviz_common::Panel::save(config);
  config.mapSetValue("ShowTime", show_time_action_->isChecked());
}

QTreeWidgetItem * TaskConstructorPanel::buildSpecItem(
    const std::map<uint32_t, const curobo_task_constructor_interfaces::msg::StageSpec *> & by_id,
    const std::map<uint32_t, std::vector<uint32_t>> & children_of, uint32_t id,
    QTreeWidgetItem * parent)
{
  const auto spec_it = by_id.find(id);
  if (spec_it == by_id.end()) return nullptr;
  const auto & spec = *spec_it->second;
  auto * item = new QTreeWidgetItem();
  item->setText(COL_STAGE, nameOf(spec));
  item->setText(COL_TYPE, typeOf(spec));
  item->setText(COL_OK, "-");
  item->setText(COL_FAIL, "-");
  item->setText(COL_COST, "");
  item->setText(COL_TIME, "");
  item->setData(0, Qt::UserRole, id);
  if (parent != nullptr)
    parent->addChild(item);
  else
    tree_->addTopLevelItem(item);
  stage_item_[id] = item;
  const auto children_it = children_of.find(id);
  if (children_it != children_of.end())
    for (uint32_t child_id : children_it->second) buildSpecItem(by_id, children_of, child_id, item);
  return item;
}

void TaskConstructorPanel::refreshTaskDescription()
{
  tree_->clear();
  stage_item_.clear();
  task_root_item_ = nullptr;
  solutions_->clear();
  syncProperties({});
  exec_button_->setEnabled(false);
  if (!task_ || task_->stages.empty()) {
    setStatus(tr("no task — waiting for task_description..."));
    return;
  }
  std::map<uint32_t, const curobo_task_constructor_interfaces::msg::StageSpec *> by_id;
  std::map<uint32_t, std::vector<uint32_t>> children_of;
  uint32_t root_id = 0;
  bool have_root = false;
  for (const auto & stage : task_->stages) {
    by_id[stage.id] = &stage;
    if (stage.parent_id != stage.id)
      children_of[stage.parent_id].push_back(stage.id);
    else {
      root_id = stage.id;
      have_root = true;
    }
  }
  if (!have_root) {
    setStatus(tr("task_description has no root stage (parent_id == id)"));
    return;
  }
  // Task row on top (MTC lists tasks above their stages); the spec root
  // hangs beneath it.
  task_root_item_ = new QTreeWidgetItem();
  task_root_item_->setText(COL_STAGE, QString::fromStdString(task_->task_id));
  task_root_item_->setText(COL_TYPE, tr("task"));
  task_root_item_->setText(COL_OK, "-");
  task_root_item_->setText(COL_FAIL, "-");
  tree_->addTopLevelItem(task_root_item_);
  buildSpecItem(by_id, children_of, root_id, task_root_item_);
  applyStoredData();
  tree_->expandAll();
  QString status = tr("task '%1' (%2 stages)").arg(
      QString::fromStdString(task_->task_id)).arg(task_->stage_count);
  if (!task_->valid) status += tr(" — INVALID: %1").arg(QString::fromStdString(task_->comment));
  setStatus(status);
  exec_button_->setEnabled(task_->valid);
}

void TaskConstructorPanel::applyStoredData()
{
  if (!task_) return;
  int ok_total = 0, fail_total = 0;
  for (auto it = stage_item_.begin(); it != stage_item_.end(); ++it) {
    updateStageCounts(it.key());
    auto stat = stats_.find(it.key());
    if (stat != stats_.end() && stat.value()) {
      it.value()->setText(COL_TIME, QString::number(stat.value()->total_compute_time, 'g', 4));
      it.value()->setToolTip(COL_TYPE,
                             QStringLiteral("successful attempts: %1")
                                 .arg(stat.value()->success_count));
    }
    ok_total += stageOkCount(it.key());
    fail_total += stageFailCount(it.key());
  }
  if (task_root_item_) {
    task_root_item_->setText(COL_OK, QString::number(ok_total));
    task_root_item_->setText(COL_FAIL, QString::number(fail_total));
  }
}

int TaskConstructorPanel::stageOkCount(uint32_t id) const
{
  int ok = 0;
  auto att = attempts_.find(id);
  if (att != attempts_.end()) {
    for (const auto &sol : att.value()) {
      if (sol && sol->success) ++ok;
    }
  }
  return ok;
}

int TaskConstructorPanel::stageFailCount(uint32_t id) const
{
  int fail = 0;
  auto att = attempts_.find(id);
  if (att != attempts_.end()) {
    for (const auto &sol : att.value()) {
      if (sol && !sol->success) ++fail;
    }
  }
  return fail;
}

void TaskConstructorPanel::updateStageCounts(uint32_t id)
{
  // MTC ✓/✗ columns: successful vs failed attempts streamed so far. Kept
  // purely stream-driven (never derived from the statistics rollup, whose
  // attempt_count also folds in multi-attempt extra solves that never
  // produced a row).
  auto item_it = stage_item_.find(id);
  if (item_it == stage_item_.end()) return;
  item_it.value()->setText(COL_OK, QString::number(stageOkCount(id)));
  item_it.value()->setText(COL_FAIL, QString::number(stageFailCount(id)));
}

void TaskConstructorPanel::refreshSolutionInfo()
{
  if (!task_) return;
  int ok_total = 0, fail_total = 0;
  for (auto it = attempts_.begin(); it != attempts_.end(); ++it) {
    auto item_it = stage_item_.find(it.key());
    if (item_it == stage_item_.end() || it.value().empty() || !it.value().back()) continue;
    applySolutionToItem(item_it.value(), *it.value().back());
    updateStageCounts(it.key());
    ok_total += stageOkCount(it.key());
    fail_total += stageFailCount(it.key());
  }
  if (task_root_item_) {
    task_root_item_->setText(COL_OK, QString::number(ok_total));
    task_root_item_->setText(COL_FAIL, QString::number(fail_total));
  }
  rebuildSolutionList();
}

void TaskConstructorPanel::applySolutionToItem(
    QTreeWidgetItem * item, const curobo_task_constructor_interfaces::msg::SolutionInfo & sol)
{
  // No success/failure background (MTC rows stay palette-default).
  if (sol.success) {
    item->setText(COL_COST, QString::number(sol.cost, 'g', 6));
  } else {
    item->setText(COL_COST, QStringLiteral("—"));
  }
  item->setToolTip(COL_STAGE, QString::fromStdString(sol.comment));
}

void TaskConstructorPanel::highlightStage(const QString & name)
{
  // MTC highlightStage: yellow background on the row whose (sub-)trajectory
  // is currently playing or executing.
  clearHighlight();
  for (auto it = stage_item_.begin(); it != stage_item_.end(); ++it) {
    if (it.value()->text(COL_STAGE) == name) {
      for (int col = 0; col < COL_COUNT; ++col)
        it.value()->setBackground(col, QBrush(QColor(Qt::yellow)));
      return;
    }
  }
  if (task_root_item_ && task_root_item_->text(COL_STAGE) == name) {
    for (int col = 0; col < COL_COUNT; ++col)
      task_root_item_->setBackground(col, QBrush(QColor(Qt::yellow)));
  }
}

void TaskConstructorPanel::clearHighlight()
{
  const QBrush none;
  if (task_root_item_) {
    for (int col = 0; col < COL_COUNT; ++col) task_root_item_->setBackground(col, none);
  }
  for (auto it = stage_item_.begin(); it != stage_item_.end(); ++it) {
    for (int col = 0; col < COL_COUNT; ++col) it.value()->setBackground(col, none);
  }
}

void TaskConstructorPanel::refreshChains()
{
  // A newly ranked chain lands while its task is on display: show it when
  // the task root (chains mode) is selected, like incoming attempts do.
  if (showingChains()) rebuildSolutionList();
}

void TaskConstructorPanel::refreshStageStatistics()
{
  if (!task_) return;
  for (auto it = stats_.begin(); it != stats_.end(); ++it) {
    auto item_it = stage_item_.find(it.key());
    if (item_it == stage_item_.end() || !it.value()) continue;
    item_it.value()->setText(COL_TIME, QString::number(it.value()->total_compute_time, 'g', 4));
    item_it.value()->setToolTip(COL_TYPE, QStringLiteral("successful attempts: %1")
                                                 .arg(it.value()->success_count));
  }
  refreshProperties();
}

void TaskConstructorPanel::refreshProperties()
{
  // Properties follow data ticks (compute time, attempts) without waiting
  // for a selection change. No trajectory/marker republish here — that
  // stays on selection changes via showSelectedSolution.
  auto selected = solutions_->selectedItems();
  if (selected.isEmpty()) {
    auto * current = solutions_->currentItem();
    if (current) selected.push_back(current);
  }
  if (!selected.isEmpty()) {
    showSolutionPropertiesFor(selected.front());
  } else if (tree_->currentItem()) {
    showStageProperties(tree_->currentItem());
  } else {
    syncProperties({});
  }
}

bool TaskConstructorPanel::showingChains() const
{
  return tree_->currentItem() == task_root_item_;
}

void TaskConstructorPanel::onSelectedItemChanged()
{
  clearHighlight();
  rebuildSolutionList();
  auto * item = tree_->currentItem();
  if (!item) {
    syncProperties({});
    return;
  }
  showStageProperties(item);
  setStatus(tr("selected '%1'").arg(item->text(COL_STAGE)));
}

void TaskConstructorPanel::showStageProperties(QTreeWidgetItem * item)
{
  // MTC property view: the selected stage's properties. Rows are synced in
  // place (syncProperties) so every data tick — attempts, statistics, chains
  // — refreshes the displayed values live while the selection stays put.
  if (!item) {
    syncProperties({});
    return;
  }
  std::vector<std::pair<QString, QString>> rows;
  auto add = [&rows](const QString & key, const QString & value) {
    rows.emplace_back(key, value);
  };
  if (item == task_root_item_ && task_) {
    add(tr("task"), QString::fromStdString(task_->task_id));
    add(tr("stages"), QString::number(task_->stage_count));
    add(tr("valid"), task_->valid ? tr("true") : tr("false"));
    if (!task_->comment.empty()) add(tr("comment"), QString::fromStdString(task_->comment));
    syncProperties(rows);
    return;
  }
  const uint32_t id = item->data(0, Qt::UserRole).toUInt();
  add(tr("stage"), item->text(COL_STAGE));
  add(tr("type"), item->text(COL_TYPE));
  add(tr("successful"), item->text(COL_OK));
  add(tr("failed"), item->text(COL_FAIL));
  add(tr("cost"), item->text(COL_COST));
  add(tr("compute time"), item->text(COL_TIME));
  auto stat = stats_.find(id);
  if (stat != stats_.end() && stat.value())
    add(tr("planner attempts"), QString::number(stat.value()->attempt_count));
  syncProperties(rows);
}

void TaskConstructorPanel::syncProperties(
    const std::vector<std::pair<QString, QString>> & rows)
{
  // In-place sync, like MTC's RemoteTaskModel::setProperties reusing the
  // existing rviz properties: values tick live on every introspection burst
  // without collapsing the user's expanded rows. A clear() + re-add here
  // would reset expansion several times a second mid-plan.
  const int want = static_cast<int>(rows.size());
  while (properties_->topLevelItemCount() > want)
    delete properties_->takeTopLevelItem(properties_->topLevelItemCount() - 1);
  for (int i = 0; i < want; ++i) {
    QTreeWidgetItem * row = properties_->topLevelItem(i);
    if (!row) row = new QTreeWidgetItem(properties_);
    // Row order is stable per view, so index i keeps its expansion state
    // across ticks; only the texts are rewritten.
    row->setText(0, rows[i].first);
    row->setText(1, rows[i].second);
  }
}

void TaskConstructorPanel::showSolutionProperties(
    const QString & rank, const QString & cost, const QString & comment,
    const QString & extra)
{
  std::vector<std::pair<QString, QString>> rows;
  rows.emplace_back(tr("solution"), rank);
  rows.emplace_back(tr("cost"), cost);
  if (!comment.isEmpty()) rows.emplace_back(tr("comment"), comment);
  if (!extra.isEmpty()) rows.emplace_back(tr("detail"), extra);
  syncProperties(rows);
}

void TaskConstructorPanel::rebuildSolutionList()
{
  // MTC solutions view: selecting the task lists its complete solutions,
  // selecting a stage lists that stage's attempts. The rebuild runs on
  // every incoming attempt, so the selection is snapshotted by record id
  // and restored afterwards — otherwise the properties pane and the
  // trajectory display would go stale mid-plan.
  QSet<int> selected_ids;
  for (auto * row : solutions_->selectedItems()) selected_ids.insert(row->data(0, Qt::UserRole).toInt());
  int current_id = -1;
  if (solutions_->currentItem()) current_id = solutions_->currentItem()->data(0, Qt::UserRole).toInt();
  solutions_->blockSignals(true);
  solutions_->clear();
  if (!task_) {
    solutions_->blockSignals(false);
    return;
  }
  if (showingChains()) {
    for (const auto &sol : chains_) {
      if (!sol) continue;
      QString chain;
      for (const auto &name : sol->stage_names) {
        if (!chain.isEmpty()) chain += QStringLiteral(" → ");
        chain += QString::fromStdString(name);
      }
      auto * row = new QTreeWidgetItem(solutions_);
      row->setText(SOL_RANK, QStringLiteral("#%1").arg(sol->solution_index));
      row->setText(SOL_COST, QString::number(sol->cost, 'g', 4));
      row->setText(SOL_COMMENT, chain);
      row->setData(0, Qt::UserRole, static_cast<int>(sol->solution_index));
    }
  } else {
    auto * item = tree_->currentItem();
    if (item) {
      const uint32_t id = item->data(0, Qt::UserRole).toUInt();
      auto it = attempts_.find(id);
      if (it != attempts_.end()) {
        int idx = 0;
        for (const auto &sol : it.value()) {
          if (!sol) continue;
          // MTC solution rows: rank, cost (∞ for failures), comment —
          // failures in red foreground, no background wash.
          auto * row = new QTreeWidgetItem(solutions_);
          row->setText(SOL_RANK, QStringLiteral("#%1").arg(idx + 1));
          row->setText(SOL_COST, sol->success ? QString::number(sol->cost, 'g', 4)
                                              : QString::fromUtf8("∞"));
          row->setText(SOL_COMMENT, QString::fromStdString(sol->comment));
          row->setData(0, Qt::UserRole, idx);
          if (!sol->success) {
            for (int col = 0; col < SOL_COUNT; ++col)
              row->setForeground(col, QColor(Qt::red));
          }
          ++idx;
        }
      }
    }
  }
  solutions_->blockSignals(false);
  for (int i = 0; i < solutions_->topLevelItemCount(); ++i) {
    auto * row = solutions_->topLevelItem(i);
    const int id = row->data(0, Qt::UserRole).toInt();
    if (selected_ids.contains(id)) row->setSelected(true);
    if (id == current_id) solutions_->setCurrentItem(row);
  }
  showSelectedSolution();
}

void TaskConstructorPanel::onSolutionSelectionChanged()
{
  // MTC: the current solution displays its trajectory; every selected row
  // contributes its markers.
  showSelectedSolution();
}

void TaskConstructorPanel::showSelectedSolution()
{
  auto selected = solutions_->selectedItems();
  if (selected.isEmpty()) {
    auto * current = solutions_->currentItem();
    if (current) selected.push_back(current);
  }
  publishSolutionTrajectory(selected);
  publishToolPath(selected);
  refreshProperties();
}

void TaskConstructorPanel::showSolutionPropertiesFor(QTreeWidgetItem * row)
{
  if (!row) return;
  if (showingChains()) {
    const int index = row->data(0, Qt::UserRole).toInt();
    for (const auto &sol : chains_) {
      if (sol && static_cast<int>(sol->solution_index) == index) {
        QString chain;
        for (const auto &name : sol->stage_names) {
          if (!chain.isEmpty()) chain += QStringLiteral(" → ");
          chain += QString::fromStdString(name);
        }
        showSolutionProperties(QStringLiteral("#%1").arg(sol->solution_index),
                               QString::number(sol->cost, 'g', 4), chain,
                               tr("%1 segments").arg(sol->stage_names.size()));
        return;
      }
    }
  } else {
    auto * item = tree_->currentItem();
    if (!item) return;
    const uint32_t id = item->data(0, Qt::UserRole).toUInt();
    auto it = attempts_.find(id);
    const int idx = row->data(0, Qt::UserRole).toInt();
    if (it != attempts_.end() && idx >= 0 && idx < static_cast<int>(it.value().size())) {
      const auto & sol = it.value()[idx];
      if (!sol) return;
      showSolutionProperties(
          QStringLiteral("#%1").arg(idx + 1),
          sol->success ? QString::number(sol->cost, 'g', 4) : QString::fromUtf8("∞"),
          QString::fromStdString(sol->comment),
          tr("%1 waypoints").arg(static_cast<int>(sol->trajectory.size())));
    }
  }
}

void TaskConstructorPanel::publishSolutionTrajectory(const QList<QTreeWidgetItem *> & selected)
{
  // The current solution as JointTrajectory for the CuroboTrajectoryDisplay
  // (full-robot animation + trail) — the same topic Task.publish uses. For a
  // complete chain the waypoints concatenate its leaves' winning attempts.
  (void)selected;
  if (!pub_solution_trajectory_) return;
  trajectory_msgs::msg::JointTrajectory msg;
  const auto waypoints = currentWaypoints();
  if (!waypoints.empty()) {
    msg.joint_names = waypoints.front().name;
    double t = 0.0;
    for (const auto &wp : waypoints) {
      trajectory_msgs::msg::JointTrajectoryPoint pt;
      pt.positions = wp.position;
      pt.time_from_start.sec = static_cast<int32_t>(t);
      pt.time_from_start.nanosec =
          static_cast<uint32_t>((t - static_cast<int32_t>(t)) * 1e9);
      t += 0.05;
      msg.points.push_back(pt);
    }
  }
  pub_solution_trajectory_->publish(msg);
}

std::vector<sensor_msgs::msg::JointState> TaskConstructorPanel::currentWaypoints() const
{
  // Current row's waypoints: the attempt itself, or the concatenated leaf
  // trajectories of the selected chain (boundary waypoint deduplicated like
  // the executor's chain lifting).
  std::vector<sensor_msgs::msg::JointState> out;
  auto * current = solutions_->currentItem();
  if (!current || !task_) return out;
  if (showingChains()) {
    const int index = current->data(0, Qt::UserRole).toInt();
    for (const auto &chain : chains_) {
      if (!chain || static_cast<int>(chain->solution_index) != index) continue;
      for (size_t i = 0; i < chain->stage_ids.size(); ++i) {
        const auto sol = findAttempt(chain->stage_ids[i], chain->solution_ids[i]);
        if (!sol) continue;
        for (const auto &wp : sol->trajectory) {
          if (!out.empty() && !wp.position.empty() && !out.back().position.empty() &&
              out.back().position.size() == wp.position.size()) {
            bool same = true;
            for (size_t j = 0; j < wp.position.size(); ++j) {
              if (out.back().position[j] != wp.position[j]) {
                same = false;
                break;
              }
            }
            if (same) continue;
          }
          out.push_back(wp);
        }
      }
    }
    return out;
  }
  const SolutionInfoShared sol = currentAttempt(current);
  if (sol) {
    for (const auto &wp : sol->trajectory) out.push_back(wp);
  }
  return out;
}

TaskConstructorPanel::SolutionInfoShared
TaskConstructorPanel::findAttempt(uint32_t stage_id, uint32_t solution_id) const
{
  auto it = attempts_.find(stage_id);
  if (it == attempts_.end()) return nullptr;
  for (const auto &sol : it.value()) {
    if (sol && sol->id == solution_id) return sol;
  }
  return nullptr;
}

TaskConstructorPanel::SolutionInfoShared TaskConstructorPanel::currentAttempt(
    QTreeWidgetItem * row) const
{
  if (!row || showingChains()) return nullptr;
  auto * item = tree_->currentItem();
  if (!item) return nullptr;
  const uint32_t id = item->data(0, Qt::UserRole).toUInt();
  auto it = attempts_.find(id);
  const int idx = row->data(0, Qt::UserRole).toInt();
  if (it == attempts_.end() || idx < 0 || idx >= static_cast<int>(it.value().size()))
    return nullptr;
  return it.value()[idx];
}

std::vector<TaskConstructorPanel::SolutionInfoShared>
TaskConstructorPanel::selectedAttempts(const QList<QTreeWidgetItem *> & selected) const
{
  // Every selected row's records (MTC shows all selected rows' markers):
  // attempts directly, or the leaves of selected chains.
  std::vector<SolutionInfoShared> out;
  for (auto * row : selected) {
    if (!row) continue;
    if (showingChains()) {
      const int index = row->data(0, Qt::UserRole).toInt();
      for (const auto &chain : chains_) {
        if (!chain || static_cast<int>(chain->solution_index) != index) continue;
        for (size_t i = 0; i < chain->stage_ids.size(); ++i) {
          auto sol = findAttempt(chain->stage_ids[i], chain->solution_ids[i]);
          if (sol) out.push_back(sol);
        }
      }
    } else if (auto sol = currentAttempt(row)) {
      out.push_back(sol);
    }
  }
  return out;
}

void TaskConstructorPanel::publishToolPath(const QList<QTreeWidgetItem *> & selected)
{
  if (!pub_selected_markers_) return;
  // Every selected row's tool trail (MTC shows all selected rows' markers);
  // with nothing selected, every attempt of the current stage stays visible
  // faintly so all alternatives read at a glance. The current row draws on
  // top with the scrub... (no scrub: current row draws strongest).
  visualization_msgs::msg::MarkerArray out;
  std::vector<SolutionInfoShared> records = selectedAttempts(selected);
  if (records.empty() && !showingChains()) {
    auto * item = tree_->currentItem();
    if (item && task_) {
      const uint32_t id = item->data(0, Qt::UserRole).toUInt();
      auto it = attempts_.find(id);
      if (it != attempts_.end()) {
        for (const auto &sol : it.value()) {
          if (sol) records.push_back(sol);
        }
      }
    }
  }
  auto current = solutions_->currentItem();
  const SolutionInfoShared current_sol = current ? selectedAttempts({current}).front() : nullptr;
  int marker_id = 0;
  for (const auto &sol : records) {
    if (!sol || sol->tool_poses.empty()) continue;
    const bool strong = (sol == current_sol);
    visualization_msgs::msg::Marker line;
    line.header.frame_id = "world";
    line.ns = "tool_path";
    line.id = marker_id++;
    line.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line.action = visualization_msgs::msg::Marker::ADD;
    line.scale.x = strong ? 0.008 : 0.003;
    if (sol->success) {
      line.color.g = strong ? 0.9 : 0.45;
    } else {
      line.color.r = strong ? 1.0 : 0.5;
    }
    line.color.a = strong ? 1.0 : 0.45;
    for (const auto &p : sol->tool_poses) {
      geometry_msgs::msg::Point pt;
      pt.x = p.position.x;
      pt.y = p.position.y;
      pt.z = p.position.z;
      line.points.push_back(pt);
    }
    out.markers.push_back(line);
  }
  // Each drawn attempt's own markers (candidate spheres, start/goal frames).
  for (const auto &sol : records) {
    if (sol && !sol->markers.empty())
      out.markers.insert(out.markers.end(), sol->markers.begin(),
                         sol->markers.end());
  }
  pub_selected_markers_->publish(out);
}

void TaskConstructorPanel::onExecSolution()
{
  // MTC onExecCurrentSolution: drive the selected solution with no
  // replanning. A chain row executes the whole stored chain; an attempt row
  // executes that single stored segment.
  if (!task_) {
    setStatus(tr("no task to execute yet"));
    return;
  }
  ExecuteAction::Goal goal;
  goal.task_id = task_->task_id;
  auto * current = solutions_->currentItem();
  if (!current) {
    setStatus(tr("select a solution to execute"));
    return;
  }
  if (showingChains()) {
    goal.stage_id = NO_STAGE;
    goal.solution_index = static_cast<uint32_t>(current->data(0, Qt::UserRole).toInt());
    goal.attempt_id = 0;
  } else {
    auto * item = tree_->currentItem();
    if (!item) {
      setStatus(tr("select a solution to execute"));
      return;
    }
    goal.stage_id = item->data(0, Qt::UserRole).toUInt();
    goal.solution_index = 0;
    auto sol = currentAttempt(current);
    if (!sol || !sol->success || !sol->ranked) {
      setStatus(tr("only ranked solutions execute"));
      return;
    }
    goal.attempt_id = sol->id;
  }
  setStatus(tr("executing selected solution (no replanning)..."));

  rclcpp_action::Client<ExecuteAction>::SendGoalOptions options;
  options.feedback_callback = [this](const GoalHandleExecute::SharedPtr &,
                                     ExecuteAction::Feedback::ConstSharedPtr feedback) {
    const QString stage = QString::fromStdString(feedback->current_stage_name);
    QMetaObject::invokeMethod(
        this,
        [this, feedback, stage]() {
          if (!stage.isEmpty()) highlightStage(stage);
          setStatus(tr("executing '%1' (%2/%3)...")
                        .arg(stage)
                        .arg(feedback->segments_done)
                        .arg(feedback->segments_total));
        },
        Qt::QueuedConnection);
  };
  options.goal_response_callback = [this](const GoalHandleExecute::SharedPtr & goal_handle) {
    if (!goal_handle) {
      QMetaObject::invokeMethod(
          this, [this]() { setStatus(tr("execution rejected (another one in progress?)")); },
          Qt::QueuedConnection);
      return;
    }
    auto result_cb = [this](const GoalHandleExecute::WrappedResult & result) {
      QString text;
      if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
        text = result.result->success
                   ? tr("execution succeeded")
                   : tr("execution failed: %1").arg(
                         QString::fromStdString(result.result->error));
      } else if (result.code == rclcpp_action::ResultCode::CANCELED) {
        text = tr("execution canceled");
      } else {
        text = tr("execution aborted: %1").arg(
            QString::fromStdString(result.result->error));
      }
      QMetaObject::invokeMethod(
          this,
          [this, text]() {
            clearHighlight();
            setStatus(text);
          },
          Qt::QueuedConnection);
    };
    exec_client_->async_get_result(goal_handle, result_cb);
  };
  exec_client_->async_send_goal(goal, options);
}

void TaskConstructorPanel::onShowTimeChanged()
{
  tree_->setColumnHidden(COL_TIME, !show_time_action_->isChecked());
}

void TaskConstructorPanel::setStatus(const QString & text)
{
  status_label_->setText(text);
}

}  // namespace curobo_task_constructor_rviz

PLUGINLIB_EXPORT_CLASS(curobo_task_constructor_rviz::TaskConstructorPanel, rviz_common::Panel)
