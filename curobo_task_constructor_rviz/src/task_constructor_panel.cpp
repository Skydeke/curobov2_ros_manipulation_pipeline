#include "curobo_task_constructor_rviz/task_constructor_panel.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <rmw/qos_profiles.h>

#include <rviz_common/display_context.hpp>

#include <QBrush>
#include <QCheckBox>
#include <QColor>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMetaObject>
#include <QPushButton>
#include <QTreeWidget>
#include <QVariant>
#include <QVBoxLayout>

#include <functional>
#include <string>

namespace curobo_task_constructor_rviz
{

namespace
{
// column layout of the stage tree
enum Column
{
  COL_STAGE = 0,
  COL_TYPE,
  COL_ATTEMPTS,
  COL_LAST,
  COL_COST,
  COL_TIME,
  COL_COUNT
};

// soft success/failure tints (scheme-agnostic; message text carries the detail)
const QColor kSuccessBg(0xE8, 0xF5, 0xE9);
const QColor kFailureBg(0xFD, 0xEB, 0xEC);

// introspection topics (must match node.py's _INTROSPECT_QOS_TOPICS and the
// action server name)
const char * kTopicTaskDescription = "/curobo_task_constructor/task_description";
const char * kTopicSolutionInfo = "/curobo_task_constructor/solution_info";
const char * kTopicStageStatistics = "/curobo_task_constructor/stage_statistics";
const char * kTopicSelectedMarkers = "/curobo_task_constructor/selected_solution_markers";
const char * kActionTask = "/curobo_task_constructor/task";

//: same QoS as the node's introspection publishers: RELIABLE + TRANSIENT_LOCAL
// so a panel that joins mid-task still sees structure and last attempts.
rclcpp::QoS introspectionQoS()
{
  rclcpp::QoS qos(rclcpp::KeepLast(10));
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

using GoalHandleTask = rclcpp_action::ClientGoalHandle<TaskAction>;
}  // namespace

TaskConstructorPanel::TaskConstructorPanel(QWidget * parent) : rviz_common::Panel(parent)
{
  setObjectName("CuroboTaskConstructorPanel");
  setupUi();
}

TaskConstructorPanel::~TaskConstructorPanel() {}
// Out-of-line destructor: the class's key function (first non-inline virtual
// per the Itanium ABI) anchors the vtable to THIS translation unit. A plain
// body mirrors the sibling add_objects_panel layout exactly ({}= default here
// previously). If the dtor ever moved inline, the vtable would move with it
// and a partial build could dlopen-bomb with "undefined symbol: vtable".
void TaskConstructorPanel::setupUi()
{
  auto * layout = new QVBoxLayout(this);

  tree_ = new QTreeWidget(this);
  tree_->setColumnCount(COL_COUNT);
  tree_->setHeaderLabels({ tr("Stage"), tr("Type"), tr("Attempts"), tr("Last"),
                           tr("Cost"), tr("Compute time") });
  tree_->setRootIsDecorated(true);
  tree_->setAlternatingRowColors(true);
  tree_->header()->setStretchLastSection(true);
  layout->addWidget(tree_, /*stretch=*/1);

  auto * controls = new QHBoxLayout();
  reexecute_button_ = new QPushButton(tr("Re-run task"), this);
  reexecute_button_->setEnabled(false);
  reexecute_button_->setToolTip(
      tr("Send the last received task back to the executor "
         "(via /curobo_task_constructor/task Task.action)"));
  execute_checkbox_ = new QCheckBox(tr("execute on server"), this);
  execute_checkbox_->setChecked(true);
  execute_checkbox_->setToolTip(
      tr("When set, the server also executes the winning solution; "
         "otherwise it only plans."));
  controls->addWidget(reexecute_button_);
  controls->addWidget(execute_checkbox_);
  controls->addStretch(1);
  layout->addLayout(controls);

  status_label_ = new QLabel(tr("waiting for task_description..."), this);
  status_label_->setWordWrap(true);
  layout->addWidget(status_label_);

  connect(tree_, &QTreeWidget::currentItemChanged, this,
          [this](QTreeWidgetItem * /*current*/, QTreeWidgetItem * /*previous*/)
          { onSelectedItemChanged(); });
  connect(reexecute_button_, &QPushButton::clicked, this,
          &TaskConstructorPanel::onReexecuteClicked);
}

void TaskConstructorPanel::onInitialize()
{
  // The base class must see an initialized display context first.
  rviz_common::Panel::onInitialize();

  auto ros_node_abstraction = getDisplayContext()->getRosNodeAbstraction().lock();
  if (!ros_node_abstraction) {
    setStatus(QStringLiteral("internal error: no ROS node from rviz"));
    return;
  }
  node_ = ros_node_abstraction->get_raw_node();

  // Reentrant group: rviz spins a single-threaded executor, so subscriptions,
  // the action client and Qt refresh points must be allowed to interleave.
  cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  const auto qos = introspectionQoS();

  // Jazzy's rclcpp only accepts the callback group inside SubscriptionOptions.
  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = cb_group_;

  sub_task_description_ = node_->create_subscription<
      curobo_task_constructor_interfaces::msg::TaskDescription>(
      kTopicTaskDescription, qos,
      [this](curobo_task_constructor_interfaces::msg::TaskDescription::ConstSharedPtr msg)
      {
        last_task_ = msg;
        QMetaObject::invokeMethod(this, "refreshTaskDescription", Qt::QueuedConnection);
      },
      sub_options);
  sub_solution_info_ = node_->create_subscription<
      curobo_task_constructor_interfaces::msg::SolutionInfo>(
      kTopicSolutionInfo, qos,
      [this](curobo_task_constructor_interfaces::msg::SolutionInfo::ConstSharedPtr msg)
      {
        last_solution_[msg->stage_id] = msg;
        if (msg->success && !msg->markers.markers.empty()) {
          stage_markers_[msg->stage_id] =
              std::make_shared<visualization_msgs::msg::MarkerArray>(msg->markers);
        }
        QMetaObject::invokeMethod(this, "refreshSolutionInfo", Qt::QueuedConnection);
      },
      sub_options);
  sub_stage_statistics_ = node_->create_subscription<
      curobo_task_constructor_interfaces::msg::StageStatistics>(
      kTopicStageStatistics, qos,
      [this](curobo_task_constructor_interfaces::msg::StageStatistics::ConstSharedPtr msg)
      {
        stage_stats_[msg->stage_id] = msg;
        QMetaObject::invokeMethod(this, "refreshStageStatistics", Qt::QueuedConnection);
      },
      sub_options);

  pub_selected_markers_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
      kTopicSelectedMarkers, qos);
  action_client_ = rclcpp_action::create_client<TaskAction>(node_, kActionTask, cb_group_);

  setStatus(tr("listening on %1").arg(kTopicTaskDescription));
}

void TaskConstructorPanel::load(const rviz_common::Config & config)
{
  QVariant execute;
  if (config.mapGetValue("ExecuteOnServer", &execute) && execute.canConvert<bool>()) {
    execute_checkbox_->setChecked(execute.toBool());
  }
}

void TaskConstructorPanel::save(rviz_common::Config config) const
{
  config.mapSetValue("ExecuteOnServer", execute_checkbox_->isChecked());
}

// ----------------------------------------------------------------------------
// tree building
// ----------------------------------------------------------------------------

QTreeWidgetItem * TaskConstructorPanel::buildSpecItem(
    const std::map<uint32_t, const curobo_task_constructor_interfaces::msg::StageSpec *> & by_id,
    const std::map<uint32_t, std::vector<uint32_t>> & children_of,
    uint32_t id, QTreeWidgetItem * parent)
{
  // Rows are keyed by the wire stage id — the executor assigns those ids in
  // pre-order (executor.init(): for idx, stage in enumerate(subtree_stages())),
  // so SolutionInfo/StageStatistics find their node directly.
  const auto spec_it = by_id.find(id);
  if (spec_it == by_id.end()) {
    return nullptr;
  }
  const auto & spec = *spec_it->second;
  auto * item = new QTreeWidgetItem();
  item->setText(COL_STAGE, nameOf(spec));
  item->setText(COL_TYPE, typeOf(spec));
  item->setText(COL_ATTEMPTS, "-");
  item->setText(COL_LAST, "");
  item->setText(COL_COST, "");
  item->setText(COL_TIME, "");
  item->setData(0, Qt::UserRole, id);
  if (parent != nullptr) {
    parent->addChild(item);
  } else {
    tree_->addTopLevelItem(item);
  }
  stage_item_[id] = item;
  const auto children_it = children_of.find(id);
  if (children_it != children_of.end()) {
    for (uint32_t child_id : children_it->second) {
      buildSpecItem(by_id, children_of, child_id, item);
    }
  }
  return item;
}

void TaskConstructorPanel::refreshTaskDescription()
{
  tree_->clear();
  stage_item_.clear();
  reexecute_button_->setEnabled(false);

  if (!last_task_) {
    setStatus(tr("waiting for task_description..."));
    return;
  }

  // Relink the flat StageSpec[] list into a tree (MTC-style wire format:
  // parent_id == id marks the root; every other stage points at its parent).
  // Task graphs are small, so two assoc maps are plenty.
  std::map<uint32_t, const curobo_task_constructor_interfaces::msg::StageSpec *> by_id;
  std::map<uint32_t, std::vector<uint32_t>> children_of;
  uint32_t root_id = 0;
  bool have_root = false;
  for (const auto & stage : last_task_->stages) {
    by_id[stage.id] = &stage;
    if (stage.parent_id != stage.id) {
      // The root's self-marker (parent_id == id) is only a marker — it must
      // not become a child edge or the tree walk would loop into the root.
      children_of[stage.parent_id].push_back(stage.id);
    } else {
      root_id = stage.id;
      have_root = true;
    }
  }
  if (!have_root) {
    setStatus(tr("task_description has no root stage (parent_id == id)"));
    return;
  }
  buildSpecItem(by_id, children_of, root_id, nullptr);
  applyStoredData();
  tree_->expandAll();

  QString status = tr("task '%1' (%2 stages)").arg(
      QString::fromStdString(last_task_->task_id)).arg(last_task_->stage_count);
  if (!last_task_->valid) {
    status += tr(" — INVALID: %1").arg(QString::fromStdString(last_task_->comment));
  }
  setStatus(status);

  // a spec that came back valid can be re-sent; invalid specs would just be
  // re-rejected by the server, so keep the button disabled for those
  reexecute_button_->setEnabled(last_task_->valid);
}

void TaskConstructorPanel::applyStoredData()
{
  for (auto it = stage_item_.begin(); it != stage_item_.end(); ++it) {
    auto sol_it = last_solution_.find(it.key());
    if (sol_it != last_solution_.end() && sol_it.value()) {
      applySolutionToItem(it.value(), *sol_it.value());
    }
    auto stat_it = stage_stats_.find(it.key());
    if (stat_it != stage_stats_.end() && stat_it.value()) {
      it.value()->setText(COL_ATTEMPTS,
                          QString::number(stat_it.value()->attempt_count));
      it.value()->setText(COL_TIME,
                          QString::number(stat_it.value()->total_compute_time, 'g', 4));
      it.value()->setToolTip(
          COL_TYPE, QStringLiteral("successful attempts: %1").arg(stat_it.value()->success_count));
    }
  }
}

void TaskConstructorPanel::refreshSolutionInfo()
{
  for (auto it = last_solution_.begin(); it != last_solution_.end(); ++it) {
    auto item_it = stage_item_.find(it.key());
    if (item_it == stage_item_.end() || !it.value()) {
      continue;
    }
    applySolutionToItem(item_it.value(), *it.value());
  }
}

void TaskConstructorPanel::applySolutionToItem(
    QTreeWidgetItem * item, const curobo_task_constructor_interfaces::msg::SolutionInfo & sol)
{
  item->setText(COL_LAST, sol.success ? QStringLiteral("OK") : QStringLiteral("FAIL"));
  if (sol.success) {
    item->setText(COL_COST, QString::number(sol.cost, 'g', 6));
  } else {
    item->setText(COL_COST, QStringLiteral("—"));
  }
  const QBrush bg(sol.success ? kSuccessBg : kFailureBg);
  for (int col = 0; col < COL_COUNT; ++col) {
    item->setBackground(col, bg);
  }
  if (!sol.comment.empty()) {
    item->setToolTip(COL_STAGE, QString::fromStdString(sol.comment));
  } else {
    item->setToolTip(COL_STAGE, QString());
  }
}

void TaskConstructorPanel::refreshStageStatistics()
{
  for (auto it = stage_stats_.begin(); it != stage_stats_.end(); ++it) {
    auto item_it = stage_item_.find(it.key());
    if (item_it == stage_item_.end() || !it.value()) {
      continue;
    }
    item_it.value()->setText(COL_ATTEMPTS,
                             QString::number(it.value()->attempt_count));
    item_it.value()->setText(COL_TIME,
                             QString::number(it.value()->total_compute_time, 'g', 4));
    item_it.value()->setToolTip(
        COL_TYPE, QStringLiteral("successful attempts: %1").arg(it.value()->success_count));
  }
}

// ----------------------------------------------------------------------------
// selection + markers
// ----------------------------------------------------------------------------

void TaskConstructorPanel::onSelectedItemChanged()
{
  publishSelectedMarkers();

  auto * item = tree_->currentItem();
  if (!item) {
    return;
  }
  const uint32_t id = item->data(0, Qt::UserRole).toUInt();
  QString status = tr("selected '%1'").arg(item->text(COL_STAGE));
  auto sol_it = last_solution_.find(id);
  if (sol_it != last_solution_.end() && sol_it.value() && !sol_it.value()->comment.empty()) {
    status += tr(" — %1").arg(QString::fromStdString(sol_it.value()->comment));
  }
  setStatus(status);
}

void TaskConstructorPanel::publishSelectedMarkers()
{
  if (!pub_selected_markers_) {
    return;
  }
  visualization_msgs::msg::MarkerArray msg;
  auto * item = tree_->currentItem();
  if (item) {
    const uint32_t id = item->data(0, Qt::UserRole).toUInt();
    auto marker_it = stage_markers_.find(id);
    if (marker_it != stage_markers_.end() && marker_it.value()) {
      msg = *marker_it.value();
    }
  }
  pub_selected_markers_->publish(msg);
}

// ----------------------------------------------------------------------------
// re-execution
// ----------------------------------------------------------------------------

void TaskConstructorPanel::onReexecuteClicked()
{
  reexecute(execute_checkbox_->isChecked());
}

void TaskConstructorPanel::reexecute(bool execute)
{
  if (!last_task_) {
    setStatus(tr("no task to re-run yet"));
    return;
  }
  setStatus(tr("sending task '%1' (execute=%2)...")
                .arg(QString::fromStdString(last_task_->task_id))
                .arg(execute ? "true" : "false"));

  TaskAction::Goal goal;
  goal.task_name = last_task_->task_id;
  goal.stages = last_task_->stages;
  goal.execute = execute;

  // Feedback is published live via the action's feedback channel; the result
  // is collected explicitly with async_get_result once the goal is accepted.
  rclcpp_action::Client<TaskAction>::SendGoalOptions options;
  // Jazzy's FeedbackCallback takes the goal handle plus the feedback message.
  options.feedback_callback = [this](const GoalHandleTask::SharedPtr & /* goal_handle */,
                                     TaskAction::Feedback::ConstSharedPtr feedback)
  {
    QMetaObject::invokeMethod(
        this, "setStatus", Qt::QueuedConnection,
        Q_ARG(QString, QString::fromStdString(feedback->feedback)));
  };
  options.goal_response_callback = [this](const GoalHandleTask::SharedPtr & goal_handle)
  {
    if (!goal_handle) {
      QMetaObject::invokeMethod(
          this, "setStatus", Qt::QueuedConnection,
          Q_ARG(QString, tr("task rejected (another solve in progress?)")));
      return;
    }
    auto result_cb = [this](const GoalHandleTask::WrappedResult & result)
    {
      QString text;
      if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
        if (result.result->success) {
          text = result.result->failed_stage_name.empty()
                     ? tr("task succeeded")
                     : tr("task succeeded (last stage '%1')")
                           .arg(QString::fromStdString(result.result->failed_stage_name));
        } else {
          text = tr("task failed: %1").arg(QString::fromStdString(result.result->error));
        }
      } else if (result.code == rclcpp_action::ResultCode::ABORTED) {
        text = tr("task aborted: %1").arg(QString::fromStdString(result.result->error));
      } else {
        text = tr("task canceled");
      }
      QMetaObject::invokeMethod(this, "setStatus", Qt::QueuedConnection,
                                Q_ARG(QString, text));
    };
    action_client_->async_get_result(goal_handle, result_cb);
  };

  action_client_->async_send_goal(goal, options);
}

void TaskConstructorPanel::setStatus(const QString & text)
{
  status_label_->setText(text);
}

}  // namespace curobo_task_constructor_rviz

// Pluginlib registration lives in the SAME translation unit as the class
// (the layout every sibling plugin in this workspace uses). Note: the class
// VTABLE itself is NOT anchored here — Q_OBJECT's `virtual metaObject()`
// is the key function, so the vtable is emitted in the moc TU
// (moc_task_constructor_panel.cpp, compiled into the library via
// qt5_wrap_cpp in CMakeLists.txt). A missing moc TU shows up exactly like a
// missing registration:
//   "undefined symbol: vtable for ...::TaskConstructorPanel"
PLUGINLIB_EXPORT_CLASS(curobo_task_constructor_rviz::TaskConstructorPanel, rviz_common::Panel)