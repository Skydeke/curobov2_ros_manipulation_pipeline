#include "curobo_task_constructor_rviz/task_constructor_panel.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <rmw/qos_profiles.h>

#include <rviz_common/display_context.hpp>
#include <rviz_common/properties/bool_property.hpp>
#include <rviz_common/properties/enum_property.hpp>
#include <rviz_common/properties/property.hpp>
#include <rviz_common/properties/property_tree_model.hpp>
#include <rviz_common/properties/property_tree_widget.hpp>

#include <QAction>
#include <QBrush>
#include <QButtonGroup>
#include <QColor>
#include <QComboBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMetaObject>
#include <QPainter>
#include <QPointer>
#include <QSet>
#include <QSplitter>
#include <QStackedWidget>
#include <QTimer>
#include <QToolButton>
#include <QTreeView>
#include <QTreeWidget>
#include <QVariant>
#include <QVBoxLayout>
#include <visualization_msgs/msg/marker.hpp>

#include <cmath>
#include <chrono>
#include <functional>
#include <future>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace curobo_task_constructor_rviz
{

namespace
{
// MTC TaskListModel::horizontalHeader (task_list_model.cpp:57-99):
// 0 name | 1 ✓ | 2 ✗ | 3 time, with ✓ darkGreen and ✗ red, tooltips
// "successful solutions" / "failed solution attempts" / "total computation
// time [s]".
enum Column
{
  COL_STAGE = 0,
  COL_OK,
  COL_FAIL,
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

const char* kTopicTaskDescription = "/curobo_task_constructor/task_description";
const char* kTopicTaskStatistics = "/curobo_task_constructor/task_statistics";
const char* kTopicSolution = "/curobo_task_constructor/solution";
const char* kServiceGetSolution = "/curobo_task_constructor/get_solution";
const char* kTopicSelectedMarkers = "/curobo_task_constructor/selected_solution_markers";
const char* kTopicSolutionTrajectory = "/curobo_task_constructor/solution_trajectory";
const char* kActionExecute = "/curobo_task_constructor/execute_task_solution";

// MoveIt error code for success (moveit_msgs/MoveItErrorCodes::SUCCESS).
constexpr int kErrorSuccess = 1;

rclcpp::QoS descriptionQoS()
{
  // MTC TaskDisplay subscriber depths, verbatim: description 10,
  // statistics/solution 2, all transient_local.
  rclcpp::QoS qos(rclcpp::KeepLast(10));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  return qos;
}

rclcpp::QoS introspectionQoS()
{
  rclcpp::QoS qos(rclcpp::KeepLast(2));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  return qos;
}

QString propertyValue(
    const std::vector<curobo_task_constructor_interfaces::msg::Property>& props,
    const std::string& key, const QString& fallback = QString())
{
  for (const auto& prop : props) {
    if (prop.name == key)
      return QString::fromStdString(prop.value);
  }
  return fallback;
}

QString nameOf(
    const std::vector<curobo_task_constructor_interfaces::msg::Property>& props,
    const std::string& name)
{
  if (!name.empty())
    return QString::fromStdString(name);
  return propertyValue(props, "stage_type", QStringLiteral("?"));
}

using GoalHandleExecute = rclcpp_action::ClientGoalHandle<ExecuteAction>;

/// MTC InterfaceFlags -> the flow icon shown in column 0 of every
/// non-root stage. MTC loads PNGs from its resource file; there is no
/// resource plumbing in this package, so the same arrow glyphs are drawn
/// as vector pixmaps (identical meaning, identical position). Glyph bytes
/// go through fromUtf8: QStringLiteral on a narrow UTF-8 literal would map
/// each byte to one QChar and render mojibake.
QIcon flowIcon(uint8_t flags)
{
  const bool reads_start = flags & 0x01;
  const bool reads_end = flags & 0x02;
  const bool writes_next = flags & 0x04;
  const bool writes_prev = flags & 0x08;
  const bool generator = !reads_start && !reads_end && writes_next;

  QString glyph;
  QString color;
  if (!generator && reads_start && reads_end) {
    glyph = QString::fromUtf8("\xe2\x86\x94");  // ↔ CONNECT
    color = QStringLiteral("#5555ff");
  } else if (writes_next && reads_start) {
    glyph = QString::fromUtf8("\xe2\x86\x93");  // ↓ PROPAGATE_FORWARDS
    color = QStringLiteral("#55aa55");
  } else if (writes_prev && reads_end) {
    glyph = QString::fromUtf8("\xe2\x86\x91");  // ↑ PROPAGATE_BACKWARDS
    color = QStringLiteral("#aa55aa");
  } else if (generator) {
    glyph = QString::fromUtf8("\xe2\x9c\x94");  // ✓ GENERATE
    color = QStringLiteral("#aa9955");
  } else {
    return {};
  }

  QPixmap pm(16, 16);
  pm.fill(Qt::transparent);
  QPainter painter(&pm);
  QFont font = painter.font();
  font.setPixelSize(12);
  font.setBold(true);
  painter.setFont(font);
  painter.setPen(QColor(color));
  painter.drawText(QRectF(0, 0, 16, 16), Qt::AlignCenter, glyph);
  return QIcon(pm);
}
}  // namespace

// ---------------------------------------------------------------------------
// TaskConstructorPanel (MTC TaskPanel)
// ---------------------------------------------------------------------------

struct TaskConstructorPanelPrivate
{
  TaskConstructorPanelPrivate(TaskConstructorPanel* panel) : q_ptr(panel)
  {
    tools_layout = new QHBoxLayout();
    tools_layout->setContentsMargins(0, 2, 0, 0);
    tool_buttons_group = new QButtonGroup(panel);
    tool_buttons_group->setExclusive(true);
    stacked_widget = new QStackedWidget(panel);
    button_exec_solution = new QToolButton(panel);
    button_exec_solution->setText(TaskConstructorPanel::tr("Exec"));
    button_exec_solution->setToolTip(TaskConstructorPanel::tr("Execute solution"));
    button_exec_solution->setIcon(QIcon::fromTheme(QStringLiteral("system-run")));
    button_exec_solution->setEnabled(false);
    tools_layout->addStretch(1);
    tools_layout->addWidget(button_exec_solution);
    button_show_stage_dock_widget = new QToolButton(panel);
    button_show_stage_dock_widget->setText(QStringLiteral("..."));
    button_show_stage_dock_widget->setToolTip(
        TaskConstructorPanel::tr("Show available stages"));
    button_show_stage_dock_widget->setEnabled(false);  // no stage factory in this package
    button_show_stage_dock_widget->setVisible(false);  // MTC hides it for now
    tools_layout->addWidget(button_show_stage_dock_widget);
  }

  TaskConstructorPanel* q_ptr;
  QHBoxLayout* tools_layout;
  QButtonGroup* tool_buttons_group;
  QStackedWidget* stacked_widget;
  QToolButton* button_exec_solution;
  QToolButton* button_show_stage_dock_widget;
  rviz_common::properties::Property* property_root = nullptr;
  rviz_common::WindowManagerInterface* window_manager_ = nullptr;
};

TaskConstructorPanel::TaskConstructorPanel(QWidget* parent)
  : rviz_common::Panel(parent), d_ptr(new TaskConstructorPanelPrivate(this))
{
  Q_D(TaskConstructorPanel);

  setObjectName("CuroboTaskConstructorPanel");

  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->addLayout(d->tools_layout);
  layout->addWidget(d->stacked_widget, /*stretch=*/1);

  // sync checked tool button with displayed widget
  connect(d->tool_buttons_group, &QButtonGroup::idClicked, d->stacked_widget,
          [d](int index) { d->stacked_widget->setCurrentIndex(index); });
  connect(d->stacked_widget, &QStackedWidget::currentChanged, d->tool_buttons_group,
          [d](int index) { d->tool_buttons_group->button(index)->setChecked(true); });

  d->property_root = new rviz_common::properties::Property("Global Settings");

  auto* task_view = new TaskView(this, d->property_root);
  connect(d->button_exec_solution, SIGNAL(clicked()), task_view, SLOT(onExecCurrentSolution()));

  // create sub widgets with corresponding tool buttons. The buttons carry
  // text: this package ships no icon resources for MTC's tasks/settings
  // PNGs, and textless icon-less buttons would be invisible.
  addSubPanel(task_view, "Tasks View", QIcon());
  d->stacked_widget->setCurrentIndex(0);  // Tasks View is shown by default

  // settings widget should come last
  addSubPanel(new GlobalSettingsWidget(this, d->property_root), "Global Settings", QIcon());

  connect(d->button_show_stage_dock_widget, SIGNAL(clicked()), this, SLOT(showStageDockWidget()));
}

TaskConstructorPanel::~TaskConstructorPanel()
{
  delete d_ptr;
}

void TaskConstructorPanel::addSubPanel(SubPanel* w, const QString& title, const QIcon& icon)
{
  Q_D(TaskConstructorPanel);

  auto button = new QToolButton(w);
  button->setToolTip(title);
  button->setIcon(icon);
  button->setText(title.split(QStringLiteral(" ")).front());
  button->setCheckable(true);

  int index = d->stacked_widget->count();
  d->tools_layout->insertWidget(index, button);
  d->tool_buttons_group->addButton(button, index);
  d->stacked_widget->addWidget(w);

  w->setWindowTitle(title);
  connect(w, SIGNAL(configChanged()), this, SIGNAL(configChanged()));
}

void TaskConstructorPanel::onInitialize()
{
  Q_D(TaskConstructorPanel);
  d->window_manager_ = getDisplayContext()->getWindowManager();
}

void TaskConstructorPanel::save(rviz_common::Config config) const
{
  Q_D(const TaskConstructorPanel);
  rviz_common::Panel::save(config);
  for (int i = 0; i < d->stacked_widget->count(); ++i) {
    SubPanel* w = static_cast<SubPanel*>(d->stacked_widget->widget(i));
    w->save(config.mapMakeChild(w->windowTitle()));
  }
}

void TaskConstructorPanel::load(const rviz_common::Config& config)
{
  Q_D(TaskConstructorPanel);
  rviz_common::Panel::load(config);
  for (int i = 0; i < d->stacked_widget->count(); ++i) {
    SubPanel* w = static_cast<SubPanel*>(d->stacked_widget->widget(i));
    w->load(config.mapGetChild(w->windowTitle()));
  }
}

void TaskConstructorPanel::showStageDockWidget()
{
  // MTC opens the "Motion Planning Stages" pane backed by the stage
  // factory here. This package ships no stage factory (and the "..." button
  // stays disabled for the same reason MTC disables it without one).
  Q_D(TaskConstructorPanel);
  (void)d;
}

// ---------------------------------------------------------------------------
// TaskView (MTC TaskView)
// ---------------------------------------------------------------------------

TaskView::TaskView(TaskConstructorPanel* parent, rviz_common::properties::Property* root)
  : SubPanel(parent), panel_(parent)
{
  setupUi();

  // Own ROS node, like MTC TaskViewPrivate::node_ ("task_view_private"
  // there; suffixed here so both panels can coexist in one RViz without
  // duplicate node names). It also hosts the introspection subscriptions
  // and the trajectory/marker publishers: MTC puts those on its TaskDisplay,
  // which does not exist in this package.
  node_ = rclcpp::Node::make_shared("curobo_task_view", "");
  cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);
  spin_thread_ = std::thread([this]() { executor_->spin(); });

  exec_client_ = rclcpp_action::create_client<ExecuteAction>(
      node_, kActionExecute, cb_group_);
  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = cb_group_;
  sub_task_description_ = node_->create_subscription<
      curobo_task_constructor_interfaces::msg::TaskDescription>(
      kTopicTaskDescription, descriptionQoS(),
      [this](curobo_task_constructor_interfaces::msg::TaskDescription::ConstSharedPtr msg) {
        QMetaObject::invokeMethod(
            this,
            [this, msg]() {
              addOrReplaceTask(msg);
              if (!msg->stages.empty())
                ensureDataSubs();
            },
            Qt::QueuedConnection);
      },
      sub_options);
  get_solution_client_ = nullptr;  // per-task, created in ensureGetSolutionClient
  get_solution_task_id_.clear();
  pub_selected_markers_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
      kTopicSelectedMarkers, introspectionQoS());
  // Transient-local like Task.publish: a trajectory display that subscribes
  // later still picks up the selected solution.
  rclcpp::QoS traj_qos(rclcpp::KeepLast(1));
  traj_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  traj_qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  pub_solution_trajectory_ =
      node_->create_publisher<trajectory_msgs::msg::JointTrajectory>(kTopicSolutionTrajectory,
                                                                      traj_qos);

  tasks_view_->setSelectionMode(QAbstractItemView::ExtendedSelection);

  actionShowTimeColumn_->setChecked(true);

  // init actions. actionAddLocalTask exists (task_view.ui) but is
  // deliberately NOT added: MTC's TODO "add actionAddLocalTask once there
  // is something meaningful to add" — there are no local task models here.
  tasks_view_->addActions({actionRemoveTaskTreeRows_, actionShowTimeColumn_});

  // configuration settings (MTC TaskView ctor, verbatim strings)
  auto configs =
      new rviz_common::properties::Property("Task View Settings", QVariant(), QString(), root);
  initial_task_expand = new rviz_common::properties::EnumProperty(
      "Task Expansion", "All Expanded", "Configure how to initially expand new tasks", configs);
  initial_task_expand->addOption("Top-level Expanded", EXPAND_TOP);
  initial_task_expand->addOption("All Expanded", EXPAND_ALL);
  initial_task_expand->addOption("All Closed", EXPAND_NONE);

  old_task_handling = new rviz_common::properties::EnumProperty(
      "Old task handling", "Keep",
      "Configure what to do with old tasks whose solutions cannot be queried anymore", configs);
  old_task_handling->addOption("Keep", OLD_TASK_KEEP);
  old_task_handling->addOption("Replace", OLD_TASK_REPLACE);
  old_task_handling->addOption("Remove", OLD_TASK_REMOVE);

  show_time_column = new rviz_common::properties::BoolProperty("Show Computation Times", true,
                                                                "Show the 'time' column", configs);

  // connect signals
  connect(actionRemoveTaskTreeRows_, SIGNAL(triggered()), this, SLOT(removeSelectedStages()));
  connect(actionAddLocalTask_, SIGNAL(triggered()), this, SLOT(addTask()));
  connect(actionShowTimeColumn_, &QAction::triggered,
          [this](bool checked) { show_time_column->setValue(checked); });

  connect(tasks_view_->selectionModel(), SIGNAL(currentChanged(QModelIndex, QModelIndex)), this,
          SLOT(onCurrentStageChanged(QModelIndex, QModelIndex)));

  onCurrentStageChanged();

  // propagate infos about config changes
  connect(tasks_property_splitter_, SIGNAL(splitterMoved(int, int)), this, SIGNAL(configChanged()));
  connect(tasks_solutions_splitter_, SIGNAL(splitterMoved(int, int)), this, SIGNAL(configChanged()));
  connect(tasks_view_->header(), SIGNAL(sectionResized(int, int, int)), this, SIGNAL(configChanged()));
  connect(solutions_view_->header(), SIGNAL(sectionResized(int, int, int)), this,
          SIGNAL(configChanged()));
  connect(solutions_view_->header(), SIGNAL(sortIndicatorChanged(int, Qt::SortOrder)), this,
          SIGNAL(configChanged()));

  // Playback clock for the displayed solution's highlight (MTC
  // TaskSolutionVisualization::update at state-display rate, 0.05 s
  // default): steps the ghost's waypoints, highlighting each
  // sub-trajectory's stage as it plays.
  playback_timer_ = new QTimer(this);
  playback_timer_->setInterval(50);
  connect(playback_timer_, &QTimer::timeout, this, &TaskView::onPlaybackTick);

  setStatus(tr("listening"));
}

TaskView::~TaskView()
{
  if (executor_)
    executor_->cancel();
  if (spin_thread_.joinable())
    spin_thread_.join();
}

void TaskView::setupUi()
{
  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);

  // Splitter layout, MTC task_view.ui verbatim: vertical tasks/property
  // splitter (3:1) holding the task-tree/solutions horizontal splitter.
  tasks_property_splitter_ = new QSplitter(Qt::Vertical, this);
  auto* top = new QWidget(tasks_property_splitter_);
  auto* top_layout = new QVBoxLayout(top);
  top_layout->setContentsMargins(0, 0, 0, 0);
  top_layout->setSpacing(0);
  tasks_view_label_ = new QLabel(tr("Task Tree"), top);
  top_layout->addWidget(tasks_view_label_);
  tasks_solutions_splitter_ = new QSplitter(Qt::Horizontal, top);
  tasks_view_ = new QTreeWidget(tasks_solutions_splitter_);
  tasks_view_->setColumnCount(COL_COUNT);
  // MTC TaskListModel::horizontalHeader headers, verbatim.
  tasks_view_->setHeaderLabels(
      {tr("name"), QString::fromUtf8("✓"), QString::fromUtf8("✗"), tr("time")});
  tasks_view_->setRootIsDecorated(true);
  tasks_view_->setIndentation(15);
  tasks_view_->setUniformRowHeights(true);
  tasks_view_->setAllColumnsShowFocus(true);
  tasks_view_->setContextMenuPolicy(Qt::ActionsContextMenu);
  tasks_view_->header()->setCascadingSectionResizes(true);
  tasks_view_->header()->setStretchLastSection(false);
  // MTC task_list_model.cpp:365-384: column 0 stretches, the rest resize to
  // their contents (no stretch on the count columns).
  tasks_view_->header()->setSectionResizeMode(COL_STAGE, QHeaderView::Stretch);
  for (int col = COL_OK; col < COL_COUNT; ++col)
    tasks_view_->header()->setSectionResizeMode(col, QHeaderView::ResizeToContents);
  tasks_view_->headerItem()->setForeground(COL_OK, QColor(Qt::darkGreen));
  tasks_view_->headerItem()->setToolTip(COL_OK, tr("successful solutions"));
  tasks_view_->headerItem()->setForeground(COL_FAIL, QColor(Qt::red));
  tasks_view_->headerItem()->setToolTip(COL_FAIL, tr("failed solution attempts"));
  tasks_view_->headerItem()->setToolTip(COL_TIME, tr("total computation time [s]"));
  tasks_solutions_splitter_->addWidget(tasks_view_);
  tasks_solutions_splitter_->setStretchFactor(0, 2);

  solutions_view_ = new QTreeWidget(tasks_solutions_splitter_);
  solutions_view_->setColumnCount(SOL_COUNT);
  // MTC RemoteSolutionModel::headerData: "#" | "cost" | "comment".
  solutions_view_->setHeaderLabels({tr("#"), tr("cost"), tr("comment")});
  solutions_view_->setRootIsDecorated(false);
  solutions_view_->setUniformRowHeights(true);
  solutions_view_->setAllColumnsShowFocus(true);
  solutions_view_->setSelectionMode(QAbstractItemView::ExtendedSelection);
  solutions_view_->setSelectionBehavior(QAbstractItemView::SelectRows);
  solutions_view_->setTextElideMode(Qt::ElideNone);
  solutions_view_->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
  solutions_view_->setSortingEnabled(true);
  tasks_solutions_splitter_->addWidget(solutions_view_);
  tasks_solutions_splitter_->setStretchFactor(1, 1);
  top_layout->addWidget(tasks_solutions_splitter_, /*stretch=*/1);
  tasks_property_splitter_->addWidget(top);

  auto* bottom = new QWidget(tasks_property_splitter_);
  auto* bottom_layout = new QVBoxLayout(bottom);
  bottom_layout->setContentsMargins(0, 0, 0, 0);
  bottom_layout->setSpacing(0);
  property_view_label_ = new QLabel(tr("Properties"), bottom);
  bottom_layout->addWidget(property_view_label_);
  property_view_ = new QTreeWidget(bottom);
  property_view_->setColumnCount(2);
  property_view_->setHeaderLabels({tr("property"), tr("value")});
  property_view_->setRootIsDecorated(true);
  bottom_layout->addWidget(property_view_, /*stretch=*/1);
  tasks_property_splitter_->addWidget(bottom);
  tasks_property_splitter_->setStretchFactor(0, 3);
  tasks_property_splitter_->setStretchFactor(1, 1);
  layout->addWidget(tasks_property_splitter_, /*stretch=*/1);

  // MTC task_view.ui actions, verbatim (text, tooltip, Del shortcut).
  actionRemoveTaskTreeRows_ = new QAction(tr("Remove"), this);
  actionRemoveTaskTreeRows_->setToolTip(tr("Remove selected rows"));
  actionRemoveTaskTreeRows_->setShortcut(QKeySequence(Qt::Key_Delete));
  actionRemoveTaskTreeRows_->setShortcutContext(Qt::WidgetShortcut);
  actionRemoveTaskTreeRows_->setEnabled(false);
  actionAddLocalTask_ = new QAction(tr("Add task"), this);
  actionShowTimeColumn_ = new QAction(tr("ShowTimeColumn"), this);
  actionShowTimeColumn_->setCheckable(true);
  actionShowTimeColumn_->setChecked(true);
  actionShowTimeColumn_->setToolTip(tr("show time column"));

  status_label_ = new QLabel(tr("waiting for task_description..."), this);
  status_label_->setWordWrap(true);
  layout->addWidget(status_label_);

  connect(tasks_view_, &QTreeWidget::currentItemChanged, this,
          [this](QTreeWidgetItem*, QTreeWidgetItem*) { onCurrentStageChanged(); });
  connect(solutions_view_, &QTreeWidget::currentItemChanged, this,
          [this](QTreeWidgetItem*, QTreeWidgetItem*) { onCurrentSolutionChanged(); });
  connect(solutions_view_, &QTreeWidget::itemSelectionChanged, this,
          &TaskView::onSolutionSelectionChanged);
  connect(actionShowTimeColumn_, &QAction::toggled, this, [this](bool) { onShowTimeChanged(); });
}

void TaskView::save(rviz_common::Config config)
{
  auto write_splitter_sizes = [&config](QSplitter* splitter, const QString& key) {
    rviz_common::Config group = config.mapMakeChild(key);
    for (int s : splitter->sizes()) {
      rviz_common::Config item = group.listAppendNew();
      item.setValue(s);
    }
  };
  write_splitter_sizes(tasks_property_splitter_, "property_splitter");
  write_splitter_sizes(tasks_solutions_splitter_, "solutions_splitter");

  auto write_column_sizes = [&config](QHeaderView* view, const QString& key) {
    rviz_common::Config group = config.mapMakeChild(key);
    for (int c = 0, end = view->count(); c != end; ++c) {
      rviz_common::Config item = group.listAppendNew();
      item.setValue(view->sectionSize(c));
    }
  };
  write_column_sizes(tasks_view_->header(), "tasks_view_columns");
  write_column_sizes(solutions_view_->header(), "solutions_view_columns");

  const QHeaderView* view = solutions_view_->header();
  rviz_common::Config group = config.mapMakeChild("solution_sorting");
  group.mapSetValue("column", view->sortIndicatorSection());
  group.mapSetValue("order", view->sortIndicatorOrder());
}

void TaskView::load(const rviz_common::Config& config)
{
  if (!config.isValid())
    return;

  auto read_sizes = [&config](const QString& key) {
    rviz_common::Config group = config.mapGetChild(key);
    QList<int> sizes, empty;
    for (int i = 0; i < group.listLength(); ++i) {
      rviz_common::Config item = group.listChildAt(i);
      if (item.getType() != rviz_common::Config::Value)
        return empty;
      QVariant value = item.getValue();
      bool ok = false;
      int int_value = value.toInt(&ok);
      if (!ok)
        return empty;
      sizes << int_value;
    }
    return sizes;
  };
  tasks_property_splitter_->setSizes(read_sizes("property_splitter"));
  tasks_solutions_splitter_->setSizes(read_sizes("solutions_splitter"));

  int column = 0;
  for (int w : read_sizes("tasks_view_columns"))
    tasks_view_->setColumnWidth(++column, w);
  column = 0;
  for (int w : read_sizes("solutions_view_columns"))
    solutions_view_->setColumnWidth(++column, w);

  QTreeView* view = solutions_view_;
  rviz_common::Config group = config.mapGetChild("solution_sorting");
  int order = 0;
  if (group.mapGetInt("column", &column) && group.mapGetInt("order", &order))
    view->sortByColumn(column, static_cast<Qt::SortOrder>(order));
}

void TaskView::addTask()
{
  // No local task models exist in this package (MTC builds them from the
  // stage factory, and its Add task action stays unadded for the same
  // reason): adding is only reachable programmatically.
  setStatus(tr("local tasks are not supported in this package"));
}

void TaskView::removeSelectedStages()
{
  // MTC TaskView::removeSelectedStages: removeRows on the model. Remote
  // task content is not editable, so this drops retained TASKS whose
  // top-level items are selected (multi-task list management).
  auto selected = tasks_view_->selectedItems();
  bool removed = false;
  for (auto* item : selected) {
    if (item->parent() != nullptr)
      continue;
    QVariant task_var = item->data(0, Qt::UserRole + 2);
    if (!task_var.isValid())
      continue;
    const QString qid = task_var.toString();
    auto it = tasks_.find(qid);
    if (it != tasks_.end()) {
      delete it.value().top_item;
      tasks_.erase(it);
      removed = true;
    }
  }
  if (removed) {
    solutions_view_->clear();
    syncProperties({});
    // Explicit clear (MTC trajectory_visual_->reset() on task removal):
    // per-tick publishing never emits empty, so wipe the ghost here.
    displayed_task_.clear();
    playback_timer_->stop();
    playback_sol_ = nullptr;
    displayed_msg_ = nullptr;
    displayed_selected_.clear();
    pending_task_.clear();
    pending_msg_ = nullptr;
    publishSolutionTrajectory(std::vector<sensor_msgs::msg::JointState>());
    publishMarkers(std::vector<visualization_msgs::msg::Marker>());
    setStatus(tr("removed selected task(s)"));
  }
}

void TaskView::onCurrentStageChanged()
{
  auto* current = tasks_view_->currentItem();
  // MTC enables adding on top-level/sub-top-level items and removing on any
  // valid non-top-level selection. Remote content is read-only here, so
  // Remove drops retained tasks (task items) instead.
  int depth = 0;
  for (auto* item = current; item != nullptr; item = item->parent())
    ++depth;
  actionAddLocalTask_->setEnabled(current != nullptr && depth <= 2);
  actionRemoveTaskTreeRows_->setEnabled(current != nullptr && current->parent() == nullptr);

  // Unlock the ghost like MTC's lock(nullptr): a stage switch releases a
  // locked selection, and a queued live solution takes over.
  display_locked_ = false;
  maybePromotePending();

  rebuildSolutionList();
  if (!current) {
    syncProperties({});
    return;
  }
  showStageProperties(current);
  setStatus(tr("selected '%1'").arg(current->text(COL_STAGE)));
  // Fetch costs/comments for the newly shown rows.
  QString task_id;
  uint32_t stage_id = 0;
  if (selectedStage(task_id, stage_id))
    requestMissingSolutions(task_id, stage_id);
}

void TaskView::onCurrentSolutionChanged()
{
  // MTC locks the display onto the current solution and shows it.
  showSelectedSolution();
}

void TaskView::onSolutionSelectionChanged()
{
  // MTC: the current solution displays its trajectory; every selected row
  // contributes its markers.
  showSelectedSolution();
}

void TaskView::onExecCurrentSolution()
{
  // MTC TaskView::onExecCurrentSolution: drive the selected solution with
  // no replanning — the full Solution message goes to ExecuteTaskSolution.
  // Fire-and-forget after a 100ms server wait, exactly like MTC. (Blocking
  // spin_until_future_complete like MTC would hang RViz's GUI thread, so
  // the send stays async; observable flow is identical.)
  auto* current = solutions_view_->currentItem();
  if (!current) {
    setStatus(tr("select a solution to execute"));
    return;
  }
  const QString tid = current->data(1, Qt::UserRole).toString();
  const uint32_t id = static_cast<uint32_t>(current->data(0, Qt::UserRole).toInt());
  auto it = tasks_.find(tid);
  if (it == tasks_.end()) {
    setStatus(tr("no task to execute yet"));
    return;
  }
  auto row_it = it.value().rows.find(id);
  if (row_it != it.value().rows.end() && row_it.value().failed) {
    setStatus(tr("only successful solutions execute"));
    return;
  }
  auto sol = cachedSolution(tid, id);
  if (!sol) {
    // Fetch first, execute on arrival (never block the GUI thread).
    pending_exec_id_ = static_cast<int>(id);
    pending_exec_task_ = tid;
    setStatus(tr("fetching solution #%1...").arg(id));
    requestSolution(tid, id);
    return;
  }
  if (!exec_client_->wait_for_action_server(std::chrono::milliseconds(100))) {
    setStatus(tr("Failed to connect to the 'execute_task_solution' action server"));
    return;
  }
  ExecuteAction::Goal goal;
  goal.solution = *sol;
  setStatus(tr("executing selected solution (no replanning)..."));

  rclcpp_action::Client<ExecuteAction>::SendGoalOptions options;
  options.goal_response_callback = [this](const GoalHandleExecute::SharedPtr& goal_handle) {
    if (!goal_handle) {
      QMetaObject::invokeMethod(
          this, [this]() { setStatus(tr("Goal was rejected by server")); },
          Qt::QueuedConnection);
      return;
    }
    // MTC does not track the goal further (fire-and-forget).
  };
  exec_client_->async_send_goal(goal, options);
}

void TaskView::onShowTimeChanged()
{
  auto* header = tasks_view_->header();
  bool show = show_time_column->getBool();
  if (header->count() > 3)
    tasks_view_->header()->setSectionHidden(3, !show);
  actionShowTimeColumn_->setChecked(show);
}

void TaskView::ensureDataSubs()
{
  // MTC TaskDisplay::taskDescriptionCB: statistics/solution subscriptions
  // start on the first non-empty description, exactly once.
  if (data_subscribed_ || node_ == nullptr)
    return;
  data_subscribed_ = true;
  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = cb_group_;
  sub_task_statistics_ = node_->create_subscription<
      curobo_task_constructor_interfaces::msg::TaskStatistics>(
      kTopicTaskStatistics, introspectionQoS(),
      [this](curobo_task_constructor_interfaces::msg::TaskStatistics::ConstSharedPtr msg) {
        QMetaObject::invokeMethod(
            this, [this, msg]() { onTaskStatistics(msg); }, Qt::QueuedConnection);
      },
      sub_options);
  sub_solution_ = node_->create_subscription<curobo_task_constructor_interfaces::msg::Solution>(
      kTopicSolution, introspectionQoS(),
      [this](curobo_task_constructor_interfaces::msg::Solution::ConstSharedPtr msg) {
        QMetaObject::invokeMethod(
            this, [this, msg]() { onSolutionMessage(msg); }, Qt::QueuedConnection);
      },
      sub_options);
}

void TaskView::ensureGetSolutionClient(const std::string& task_id)
{
  if (node_ == nullptr || task_id.empty())
    return;
  if (get_solution_client_ != nullptr && get_solution_task_id_ == task_id)
    return;
  // MTC per-task naming: <base>/get_solution_<task_id> (introspection.cpp).
  const std::string name = std::string(kServiceGetSolution) + "_" + task_id;
  get_solution_client_ = node_->create_client<
      curobo_task_constructor_interfaces::srv::GetSolution>(name, rclcpp::ServicesQoS(),
                                                             cb_group_);
  get_solution_task_id_ = task_id;
}

void TaskView::applyTaskExpansion(QTreeWidgetItem* top)
{
  if (top == nullptr)
    return;
  // MTC TaskViewPrivate::configureInsertedModels: EXPAND_TOP expands the
  // task item and its first level only, EXPAND_ALL everything, EXPAND_NONE
  // collapses (parent group item stays expanded).
  const int expand = initial_task_expand->getOptionInt();
  if (expand == EXPAND_TOP) {
    tasks_view_->collapseAll();
    top->setExpanded(true);
    for (int i = 0; i < top->childCount(); ++i)
      top->child(i)->setExpanded(true);
  } else if (expand == EXPAND_NONE) {
    tasks_view_->collapseAll();
    top->setExpanded(true);
  } else {
    top->setExpanded(true);
    for (int i = 0; i < top->childCount(); ++i) {
      QTreeWidgetItem* child = top->child(i);
      child->setExpanded(true);
    }
    tasks_view_->expandAll();
  }
}

void TaskView::addOrReplaceTask(
    curobo_task_constructor_interfaces::msg::TaskDescription::ConstSharedPtr msg)
{
  const QString qid = QString::fromStdString(msg->task_id);
  // Empty description = "task destroyed" signal (MTC indicateReset +
  // RemoteTaskModel IS_DESTROYED): Remove mode drops the task, otherwise it
  // stays visible with a red name, like MTC's destroyed-task foreground.
  if (msg->stages.empty()) {
    auto it = tasks_.find(qid);
    if (old_task_handling->getOptionInt() == OLD_TASK_REMOVE && it != tasks_.end()) {
      delete it.value().top_item;
      tasks_.erase(it);
      solutions_view_->clear();
      syncProperties({});
      displayed_task_.clear();
      playback_timer_->stop();
      playback_sol_ = nullptr;
      displayed_msg_ = nullptr;
      displayed_selected_.clear();
      pending_task_.clear();
      pending_msg_ = nullptr;
      publishSolutionTrajectory(std::vector<sensor_msgs::msg::JointState>());
      publishMarkers(std::vector<visualization_msgs::msg::Marker>());
      setStatus(tr("task '%1' finished and removed").arg(qid));
      return;
    }
    if (it != tasks_.end() && it.value().top_item != nullptr) {
      it.value().top_item->setForeground(COL_STAGE, QColor(Qt::red));
      setStatus(tr("task '%1' finished").arg(qid));
    } else {
      setStatus(msg->task_id.empty() ? tr("no task — waiting for task_description...")
                                     : tr("task '%1' finished").arg(qid));
    }
    return;
  }
  // MTC Old task handling (TaskListModel::setOldTaskHandling): Keep retains
  // every task, Replace drops all others, Remove drops old ones too (only
  // the newest stays).
  const int handling = old_task_handling->getOptionInt();
  if (handling == OLD_TASK_REPLACE || handling == OLD_TASK_REMOVE) {
    for (auto it = tasks_.begin(); it != tasks_.end();) {
      if (it.key() != qid) {
        delete it.value().top_item;
        it = tasks_.erase(it);
      } else {
        ++it;
      }
    }
  }
  TaskData& data = tasks_[qid];
  if (data.top_item != nullptr)
    delete data.top_item;
  data = TaskData();
  data.desc = msg;
  // A different task takes focus: its rows, costs and highlight start
  // fresh, so a dead task's yellow row can't survive under the new tree
  // and read as stuck.
  if (qid != displayed_task_) {
    clearHighlight();
    displayed_task_.clear();
    displayed_msg_ = nullptr;
    displayed_selected_.clear();
    pending_task_.clear();
    pending_msg_ = nullptr;
    playback_timer_->stop();
    playback_sol_ = nullptr;
  }

  std::map<uint32_t, const curobo_task_constructor_interfaces::msg::StageDescription*> by_id;
  std::map<uint32_t, std::vector<uint32_t>> children_of;
  for (const auto& stage : msg->stages) {
    by_id[stage.id] = &stage;
    std::vector<curobo_task_constructor_interfaces::msg::Property> props(stage.properties.begin(),
                                                                         stage.properties.end());
    data.props[stage.id] = std::move(props);
  }
  // Root = the stage with no known parent. MTC never self-parents: the
  // root's parent is the unpublished task wrapper (id 0), so its parent
  // id simply isn't among the stages. Legacy self-parented roots
  // (parent_id == id) are accepted the same way.
  uint32_t root_id = 0;
  bool have_root = false;
  for (const auto& stage : msg->stages) {
    const bool orphan = by_id.find(stage.parent_id) == by_id.end();
    if (orphan) {
      if (!have_root) {
        root_id = stage.id;
        have_root = true;
      }
      continue;
    }
    children_of[stage.parent_id].push_back(stage.id);
  }
  if (!have_root) {
    setStatus(tr("task_description has no root stage"));
    tasks_.remove(qid);
    return;
  }
  data.root_id = root_id;
  // Top-level tree item = the task itself (MTC MetaTaskListModel groups).
  auto* top = new QTreeWidgetItem();
  top->setText(COL_STAGE, qid);
  top->setText(COL_OK, "0");
  top->setText(COL_FAIL, "0");
  top->setText(COL_TIME, "0");
  top->setTextAlignment(COL_OK, Qt::AlignRight);
  top->setTextAlignment(COL_FAIL, Qt::AlignRight);
  top->setTextAlignment(COL_TIME, Qt::AlignRight);
  top->setData(0, Qt::UserRole + 2, qid);  // task marker
  top->setData(0, Qt::UserRole, root_id);
  tasks_view_->addTopLevelItem(top);
  data.top_item = top;
  buildDescriptionItem(data, by_id, children_of, root_id, top);
  applyTaskExpansion(top);
  ensureGetSolutionClient(msg->task_id);
  // Follow the plan: select the new task root so the solutions list shows
  // the task's top-level (complete) solutions, like MTC's task rows.
  tasks_view_->setCurrentItem(top);
  setStatus(tr("task '%1' (%2 stages)").arg(qid).arg(msg->stages.size()));
  rebuildSolutionList();
}

QTreeWidgetItem* TaskView::buildDescriptionItem(
    TaskData& data,
    const std::map<uint32_t, const curobo_task_constructor_interfaces::msg::StageDescription*>& by_id,
    const std::map<uint32_t, std::vector<uint32_t>>& children_of, uint32_t id,
    QTreeWidgetItem* parent)
{
  const auto spec_it = by_id.find(id);
  if (spec_it == by_id.end())
    return nullptr;
  const auto& spec = *spec_it->second;
  auto props_it = data.props.find(id);
  const auto props = (props_it != data.props.end())
                         ? props_it.value()
                         : std::vector<curobo_task_constructor_interfaces::msg::Property>();
  auto* item = new QTreeWidgetItem();
  item->setText(COL_STAGE, nameOf(props, spec.name));
  item->setText(COL_OK, "0");
  item->setText(COL_FAIL, "0");
  item->setText(COL_TIME, "0");
  item->setTextAlignment(COL_OK, Qt::AlignRight);
  item->setTextAlignment(COL_FAIL, Qt::AlignRight);
  item->setTextAlignment(COL_TIME, Qt::AlignRight);
  // MTC shows the flow icon on every non-root stage's name cell.
  item->setData(COL_STAGE, Qt::DecorationRole, flowIcon(spec.flags));
  item->setData(COL_STAGE, Qt::UserRole + 1, static_cast<uint>(spec.flags));
  item->setData(0, Qt::UserRole, id);
  if (parent != nullptr)
    parent->addChild(item);
  else
    tasks_view_->addTopLevelItem(item);
  data.items[id] = item;
  const auto children_it = children_of.find(id);
  if (children_it != children_of.end())
    for (uint32_t child_id : children_it->second)
      buildDescriptionItem(data, by_id, children_of, child_id, item);
  return item;
}

bool TaskView::selectedStage(QString& task_id, uint32_t& stage_id) const
{
  auto* item = tasks_view_->currentItem();
  if (item == nullptr)
    return false;
  QVariant task_var = item->data(0, Qt::UserRole + 2);
  if (task_var.isValid() && !task_var.toString().isEmpty()) {
    // Task top-level item: addresses the task's root (top-level solutions).
    task_id = task_var.toString();
    auto it = tasks_.find(task_id);
    if (it == tasks_.end())
      return false;
    stage_id = it.value().root_id;
    return true;
  }
  stage_id = item->data(0, Qt::UserRole).toUInt();
  // Find the owning task via its top-level ancestor.
  QTreeWidgetItem* top = item;
  while (top->parent() != nullptr)
    top = top->parent();
  QVariant top_task = top->data(0, Qt::UserRole + 2);
  if (!top_task.isValid())
    return false;
  task_id = top_task.toString();
  return tasks_.contains(task_id);
}

void TaskView::refreshTaskStatistics(const QString& task_id)
{
  auto it = tasks_.find(task_id);
  if (it == tasks_.end())
    return;
  TaskData& data = it.value();
  for (auto jt = data.items.begin(); jt != data.items.end(); ++jt) {
    updateStageCounts(data, jt.key());
    auto stat = data.stats.find(jt.key());
    if (stat != data.stats.end()) {
      // MTC task_list_model.cpp:255 formats the compute time with the
      // default locale, 4 decimals — same here.
      jt.value()->setText(COL_TIME, QLocale().toString(stat.value().total_compute_time, 'f', 4));
    }
  }
  // The task top item mirrors its ROOT stage's row (MTC shows the root
  // container's own statistics entry, not a roll-up sum over stages).
  if (data.top_item != nullptr) {
    auto root_item = data.items.find(data.root_id);
    if (root_item != data.items.end()) {
      data.top_item->setText(COL_OK, (*root_item)->text(COL_OK));
      data.top_item->setText(COL_FAIL, (*root_item)->text(COL_FAIL));
      data.top_item->setText(COL_TIME, (*root_item)->text(COL_TIME));
    }
  }
  refreshProperties();
}

void TaskView::onTaskStatistics(
    curobo_task_constructor_interfaces::msg::TaskStatistics::ConstSharedPtr msg)
{
  const QString qid = QString::fromStdString(msg->task_id);
  auto it = tasks_.find(qid);
  if (it == tasks_.end())
    return;  // unknown task: description always comes first
  TaskData& data = it.value();
  // Register every reported id (MTC processSolutionIDs): solved ids are
  // successes, failed ids are failures; costs/comments arrive with the
  // Solution messages and fill the rows in later.
  for (const auto& stat : msg->stages) {
    data.stats[stat.id] = stat;
    for (const auto id : stat.solved) {
      auto row = data.rows.find(id);
      if (row == data.rows.end()) {
        SolutionRow fresh;
        fresh.id = id;
        fresh.stage_id = stat.id;
        fresh.failed = false;
        data.rows[id] = fresh;
      }
    }
    for (const auto id : stat.failed) {
      auto row = data.rows.find(id);
      if (row == data.rows.end()) {
        SolutionRow fresh;
        fresh.id = id;
        fresh.stage_id = stat.id;
        fresh.cost = std::numeric_limits<double>::infinity();
        fresh.failed = true;
        data.rows[id] = fresh;
      }
    }
  }
  rebuildSolutionList();
  refreshTaskStatistics(qid);
  // Fetch costs/comments for the selected stage's unknown rows.
  QString task_id;
  uint32_t stage_id = 0;
  if (selectedStage(task_id, stage_id) && task_id == qid)
    requestMissingSolutions(task_id, stage_id);
}

void TaskView::onSolutionMessage(SolutionShared msg, bool live)
{
  const QString qid = QString::fromStdString(msg->task_id);
  auto it = tasks_.find(qid);
  if (it == tasks_.end())
    return;
  TaskData& data = it.value();
  // Cache every sub-solution AND sub-trajectory info (MTC
  // processSolutionMessage reads both): successes and failures alike, so
  // rows fill in as data arrives. The ids are the GLOBAL ids the publisher
  // assigns (MTC Introspection::solutionId), which is what makes them
  // joinable to StageStatistics.solved/failed.
  bool changed = false;
  auto note_info = [&](const auto& info) {
    const uint32_t id = info.id;
    if (id == 0)
      return;  // 0 = "no solution" (MTC's sentinel), never a row
    auto row = data.rows.find(id);
    if (row == data.rows.end()) {
      SolutionRow fresh;
      fresh.id = id;
      fresh.stage_id = info.stage_id;
      fresh.failed = std::isinf(info.cost);
      data.rows[id] = fresh;
      row = data.rows.find(id);
    }
    row.value().cost = info.cost;
    row.value().comment = QString::fromStdString(info.comment);
    row.value().failed = std::isinf(info.cost);
    row.value().stage_id = info.stage_id;
    changed = true;
  };
  for (const auto& sub : msg->sub_solution)
    note_info(sub.info);
  for (const auto& sub : msg->sub_trajectory)
    note_info(sub.info);
  // A published full solution is cached by each of its sub-solution ids
  // (the execute button and the trajectory display look solutions up by
  // the row's id, whatever level it addresses).
  for (const auto& sub : msg->sub_solution) {
    if (sub.info.id == 0)
      continue;
    if (!data.solutions.contains(sub.info.id))
      data.solutions[sub.info.id] = msg;
    data.pending.remove(sub.info.id);
  }
  if (!changed)
    return;
  rebuildSolutionList();
  refreshTaskStatistics(qid);
  // Live broadcast (not a fetch completion): offer the ghost, like MTC's
  // taskSolutionCB shows every incoming solution. Fetched payloads reach
  // the ghost through the selection path instead.
  if (live)
    requestDisplay(qid, msg, /*lock=*/false);
  showSelectedSolution();
  // A pending Exec fires once its solution arrives.
  if (pending_exec_id_ >= 0 && pending_exec_task_ == qid) {
    const int wanted = pending_exec_id_;
    pending_exec_id_ = -1;
    pending_exec_task_.clear();
    auto* current = solutions_view_->currentItem();
    if (current && current->data(0, Qt::UserRole).toInt() == wanted)
      onExecCurrentSolution();
  }
}

int TaskView::stageOkCount(const TaskData& data, uint32_t id) const
{
  auto stat = data.stats.find(id);
  if (stat == data.stats.end())
    return 0;
  return static_cast<int>(stat.value().solved.size());
}

int TaskView::stageFailCount(const TaskData& data, uint32_t id) const
{
  auto stat = data.stats.find(id);
  if (stat == data.stats.end())
    return 0;
  const auto& value = stat.value();
  return std::max(static_cast<int>(value.failed.size()), static_cast<int>(value.num_failed));
}

void TaskView::updateStageCounts(TaskData& data, uint32_t id)
{
  // MTC ✓/✗ columns: successful vs failed solutions from the statistics
  // stream for this stage.
  auto item_it = data.items.find(id);
  if (item_it == data.items.end())
    return;
  item_it.value()->setText(COL_OK, QString::number(stageOkCount(data, id)));
  item_it.value()->setText(COL_FAIL, QString::number(stageFailCount(data, id)));
}

void TaskView::highlightStageById(const QString& task_id, uint32_t id)
{
  // MTC TaskListModel::highlightStage: paint the exact stage row yellow
  // (BackgroundRole), clearing the previous one. Matched by id, never by
  // name — names repeat across strategies and tasks. Like MTC (which
  // early-returns when old and new index coincide), repaint nothing when
  // the highlight does not move: unconditionally wiping the whole tree at
  // display rate stalls RViz frame pacing.
  if (task_id == highlighted_task_ && id == highlighted_id_)
    return;
  auto it = tasks_.find(task_id);
  if (it == tasks_.end())
    return;
  auto item_it = it.value().items.find(id);
  if (item_it == it.value().items.end())
    return;
  clearHighlight();
  highlighted_task_ = task_id;
  highlighted_id_ = id;
  for (int col = 0; col < COL_COUNT; ++col)
    item_it.value()->setBackground(col, QBrush(QColor(Qt::yellow)));
}

void TaskView::clearHighlight()
{
  highlighted_task_.clear();
  highlighted_id_ = 0;
  const QBrush none;
  for (auto& data : tasks_) {
    if (data.top_item != nullptr) {
      for (int col = 0; col < COL_COUNT; ++col)
        data.top_item->setBackground(col, none);
    }
    for (auto it = data.items.begin(); it != data.items.end(); ++it) {
      for (int col = 0; col < COL_COUNT; ++col)
        it.value()->setBackground(col, none);
    }
  }
}

void TaskView::refreshProperties()
{
  // Properties follow data ticks without waiting for a selection change.
  // No trajectory/marker republish here — that stays on selection changes
  // via showSelectedSolution.
  auto selected = solutions_view_->selectedItems();
  if (selected.isEmpty()) {
    auto* current = solutions_view_->currentItem();
    if (current)
      selected.push_back(current);
  }
  if (!selected.isEmpty()) {
    showSolutionPropertiesFor(selected.front());
  } else if (tasks_view_->currentItem()) {
    showStageProperties(tasks_view_->currentItem());
  } else {
    syncProperties({});
  }
}

void TaskView::showStageProperties(QTreeWidgetItem* item)
{
  // MTC property view: the selected stage's properties.
  if (!item) {
    syncProperties({});
    return;
  }
  std::vector<std::pair<QString, QString>> rows;
  auto add = [&rows](const QString& key, const QString& value) { rows.emplace_back(key, value); };
  QString task_id;
  uint32_t stage_id = 0;
  if (item->parent() == nullptr) {
    // Task top-level item: show task-level summary (MTC task rows).
    QVariant task_var = item->data(0, Qt::UserRole + 2);
    task_id = task_var.toString();
    add(tr("task"), task_id);
    add(tr("successful"), item->text(COL_OK));
    add(tr("failed"), item->text(COL_FAIL));
    add(tr("compute time"), item->text(COL_TIME));
    syncProperties(rows);
    return;
  }
  if (!selectedStage(task_id, stage_id)) {
    syncProperties({});
    return;
  }
  auto it = tasks_.find(task_id);
  if (it == tasks_.end()) {
    syncProperties({});
    return;
  }
  auto props_it = it.value().props.find(stage_id);
  if (props_it != it.value().props.end()) {
    for (const auto& prop : props_it.value()) {
      QString value = QString::fromStdString(prop.value);
      if (!prop.description.empty())
        value += QStringLiteral(" (%1)").arg(QString::fromStdString(prop.description));
      rows.emplace_back(QString::fromStdString(prop.name), value);
    }
  }
  add(tr("successful"), item->text(COL_OK));
  add(tr("failed"), item->text(COL_FAIL));
  add(tr("compute time"), item->text(COL_TIME));
  syncProperties(rows);
}

void TaskView::syncProperties(const std::vector<std::pair<QString, QString>>& rows)
{
  // In-place sync, like MTC's RemoteTaskModel::setProperties reusing the
  // existing rviz properties: values tick live on every introspection burst
  // without collapsing the user's expanded rows. A clear() + re-add here
  // would reset expansion several times a second mid-plan.
  const int want = static_cast<int>(rows.size());
  while (property_view_->topLevelItemCount() > want)
    delete property_view_->takeTopLevelItem(property_view_->topLevelItemCount() - 1);
  for (int i = 0; i < want; ++i) {
    QTreeWidgetItem* row = property_view_->topLevelItem(i);
    if (!row)
      row = new QTreeWidgetItem(property_view_);
    // Row order is stable per view, so index i keeps its expansion state
    // across ticks; only the texts are rewritten.
    row->setText(0, rows[i].first);
    row->setText(1, rows[i].second);
  }
}

void TaskView::showSolutionProperties(const QString& rank, const QString& cost,
                                      const QString& comment, const QString& extra)
{
  std::vector<std::pair<QString, QString>> rows;
  rows.emplace_back(tr("solution"), rank);
  rows.emplace_back(tr("cost"), cost);
  if (!comment.isEmpty())
    rows.emplace_back(tr("comment"), comment);
  if (!extra.isEmpty())
    rows.emplace_back(tr("detail"), extra);
  syncProperties(rows);
}

bool TaskView::SolutionRowItem::operator<(const QTreeWidgetItem& other) const
{
  // MTC RemoteSolutionModel sorting: cost column sorts by cost_rank
  // (arrival order, id tiebreak), comment column by comment text, id
  // tiebreak; rank column falls through to id order.
  const int col = treeWidget() ? treeWidget()->sortColumn() : 0;
  const int lhs = data(0, Qt::UserRole).toInt();
  const int rhs = other.data(0, Qt::UserRole).toInt();
  if (col == SOL_COMMENT) {
    const int cmp = text(SOL_COMMENT).compare(other.text(SOL_COMMENT));
    if (cmp != 0)
      return cmp < 0;
    return lhs < rhs;
  }
  return lhs < rhs;
}

void TaskView::rebuildSolutionList()
{
  // MTC solutions view: selecting the task lists its complete (root)
  // solutions, selecting a stage lists that stage's solutions. The rebuild
  // runs on every introspection tick, so the selection is snapshotted by
  // (task, solution id) and restored afterwards — otherwise the properties
  // pane and the trajectory display would go stale mid-plan.
  QSet<QPair<QString, int>> selected_ids;
  for (auto* row : solutions_view_->selectedItems())
    selected_ids.insert(
        qMakePair(row->data(1, Qt::UserRole).toString(), row->data(0, Qt::UserRole).toInt()));
  QPair<QString, int> current_id;
  if (solutions_view_->currentItem()) {
    current_id = qMakePair(solutions_view_->currentItem()->data(1, Qt::UserRole).toString(),
                           solutions_view_->currentItem()->data(0, Qt::UserRole).toInt());
  }
  solutions_view_->blockSignals(true);
  solutions_view_->clear();
  QString task_id;
  uint32_t stage_id = 0;
  if (selectedStage(task_id, stage_id)) {
    auto it = tasks_.find(task_id);
    if (it != tasks_.end()) {
      // MTC RemoteSolutionModel::processSolutionIDs assigns consecutive
      // creation ranks over the stage's rows in id order. QMap iterates
      // id-sorted, so the position here IS the MTC rank.
      int rank = 0;
      for (auto jt = it.value().rows.begin(); jt != it.value().rows.end(); ++jt) {
        SolutionRow& row_data = jt.value();
        if (row_data.stage_id != stage_id)
          continue;
        row_data.rank = ++rank;
        // MTC RemoteSolutionModel solution rows: rank, cost, comment.
        // NaN cost renders as an EMPTY cell, inf as "∞" in red; costs use
        // the default locale with 4 decimals. Text alignment: rank/cost
        // right, comment left. Tooltip shows the comment on every column.
        auto* row = new SolutionRowItem(solutions_view_);
        row->setText(SOL_RANK, QString::number(row_data.rank));
        if (row_data.failed) {
          row->setText(SOL_COST, QString::fromUtf8("∞"));
        } else if (std::isnan(row_data.cost)) {
          row->setText(SOL_COST, QString());
        } else {
          row->setText(SOL_COST, QLocale().toString(row_data.cost, 'f', 4));
        }
        row->setText(SOL_COMMENT, row_data.comment);
        row->setTextAlignment(SOL_RANK, Qt::AlignRight);
        row->setTextAlignment(SOL_COST, Qt::AlignRight);
        row->setTextAlignment(SOL_COMMENT, Qt::AlignLeft);
        for (int col = 0; col < SOL_COUNT; ++col)
          row->setToolTip(col, row_data.comment);
        row->setData(0, Qt::UserRole, static_cast<int>(row_data.id));
        row->setData(1, Qt::UserRole, task_id);
        if (row_data.failed) {
          for (int col = 0; col < SOL_COUNT; ++col)
            row->setForeground(col, QColor(Qt::red));
        }
      }
    }
  }
  solutions_view_->blockSignals(false);
  for (int i = 0; i < solutions_view_->topLevelItemCount(); ++i) {
    auto* row = solutions_view_->topLevelItem(i);
    const auto key = qMakePair(row->data(1, Qt::UserRole).toString(),
                               row->data(0, Qt::UserRole).toInt());
    if (selected_ids.contains(key))
      row->setSelected(true);
    if (key == current_id)
      solutions_view_->setCurrentItem(row);
  }
  showSelectedSolution();
}

void TaskView::showSelectedSolution()
{
  auto selected = solutions_view_->selectedItems();
  if (selected.isEmpty()) {
    auto* current = solutions_view_->currentItem();
    if (current)
      selected.push_back(current);
  }
  // Fetch full solutions for rows that only have ids so far (MTC fetches
  // displayed rows via GetSolution; the payload then flows through the
  // selection path below).
  for (auto* row : selected) {
    if (row) {
      const QString tid = row->data(1, Qt::UserRole).toString();
      const uint32_t id = static_cast<uint32_t>(row->data(0, Qt::UserRole).toUInt());
      if (!cachedSolution(tid, id))
        requestSolution(tid, id);
    }
  }
  // Ghost follows the selected row (MTC onCurrentSolutionChanged shows it
  // locked); markers follow the whole selection (MTC adds every selected
  // row's markers).
  QString tid;
  SolutionShared sol;
  if (auto* current = solutions_view_->currentItem()) {
    tid = current->data(1, Qt::UserRole).toString();
    const uint32_t id = static_cast<uint32_t>(current->data(0, Qt::UserRole).toUInt());
    sol = cachedSolution(tid, id);
  }
  if (sol)
    requestDisplay(tid, sol, /*lock=*/true);
  QSet<QPair<QString, int>> sel;
  for (auto* row : selected) {
    if (row) {
      sel.insert(qMakePair(row->data(1, Qt::UserRole).toString(),
                           row->data(0, Qt::UserRole).toInt()));
    }
  }
  if (sel != displayed_selected_) {
    displayed_selected_ = sel;
    publishMarkers(selectedMarkers());
  }
  refreshProperties();
}

void TaskView::requestDisplay(const QString& task_id, SolutionShared msg, bool lock)
{
  // MTC showTrajectory/next_solution_to_display_ dynamics: a new solution
  // request overwrites any queued one (latest wins); selection locks the
  // ghost onto its solution, dropping the queue; playback picks the queued
  // solution up when the current animation finishes (or immediately when
  // idle/unlocked-and-empty).
  if (!msg)
    return;
  if (lock) {
    // Already showing exactly this payload locked: skip (per-tick refreshes
    // must neither republish the trajectory nor restart the highlight
    // playback — that storm dropped RViz to 1 fps).
    if (display_locked_ && msg == displayed_msg_ && task_id == displayed_task_)
      return;
    pending_task_.clear();
    pending_msg_ = nullptr;
    display_locked_ = true;
    showDisplayed(task_id, msg);
    return;
  }
  // Unlocked (live broadcast): queue latest-wins. Skip payloads already
  // showing or queued, so repeat deliveries never restart playback.
  if ((msg == displayed_msg_ && task_id == displayed_task_) ||
      (msg == pending_msg_ && task_id == pending_task_))
    return;
  pending_task_ = task_id;
  pending_msg_ = msg;
  maybePromotePending();
}

void TaskView::showDisplayed(const QString& task_id, SolutionShared msg)
{
  displayed_task_ = task_id;
  displayed_msg_ = msg;
  publishSolutionTrajectory(waypointsOf(msg));
  playback_sol_ = msg;
  playback_task_ = task_id;
  playback_index_ = 0;
  if (!msg->sub_trajectory.empty())
    playback_timer_->start();
  else
    playback_timer_->stop();
}

void TaskView::maybePromotePending()
{
  // MTC update(): the queued solution plays when unlocked, or when nothing
  // is showing.
  if (!pending_msg_)
    return;
  if (display_locked_ && displayed_msg_)
    return;
  QString tid = pending_task_;
  SolutionShared msg = pending_msg_;
  pending_task_.clear();
  pending_msg_ = nullptr;
  showDisplayed(tid, msg);
}

void TaskView::onPlaybackTick()
{
  // One display step (MTC: one waypoint per state-display time): find the
  // sub-trajectory owning the current waypoint and highlight its stage
  // (MTC renderWayPoint emits activeStageChanged on segment switch).
  if (!playback_sol_ || playback_sol_->sub_trajectory.empty()) {
    playback_timer_->stop();
    return;
  }
  size_t cursor = 0;
  bool advanced = false;
  for (const auto& sub : playback_sol_->sub_trajectory) {
    const size_t n = sub.trajectory.joint_trajectory.points.size();
    if (playback_index_ >= cursor && playback_index_ < cursor + n) {
      highlightStageById(playback_task_, sub.info.stage_id);
      advanced = true;
      break;
    }
    cursor += n;
  }
  ++playback_index_;
  if (!advanced) {
    // Animation finished: stop like MTC with Loop Animation off, leaving
    // the last stage highlighted. A queued solution takes over.
    playback_timer_->stop();
    maybePromotePending();
  }
}

void TaskView::showSolutionPropertiesFor(QTreeWidgetItem* row)
{
  if (!row)
    return;
  const QString tid = row->data(1, Qt::UserRole).toString();
  const uint32_t id = static_cast<uint32_t>(row->data(0, Qt::UserRole).toInt());
  auto it = tasks_.find(tid);
  if (it == tasks_.end())
    return;
  auto row_it = it.value().rows.find(id);
  if (row_it == it.value().rows.end())
    return;
  const SolutionRow& data = row_it.value();
  QString cost;
  if (data.failed)
    cost = QString::fromUtf8("∞");
  else if (std::isnan(data.cost))
    cost = QString();
  else
    cost = QString::number(data.cost, 'f', 4);
  QString extra;
  auto sol = cachedSolution(tid, id);
  if (sol) {
    QString planner;
    int waypoints = 0;
    for (const auto& sub : sol->sub_trajectory) {
      if (planner.isEmpty())
        planner = QString::fromStdString(sub.info.planner_id);
      waypoints += static_cast<int>(sub.trajectory.joint_trajectory.points.size());
    }
    if (!planner.isEmpty())
      extra = tr("planner: %1").arg(planner);
    extra += (extra.isEmpty() ? QString() : QStringLiteral("; ")) +
             tr("%1 waypoints").arg(waypoints);
  }
  showSolutionProperties(QStringLiteral("#%1").arg(data.rank), cost, data.comment, extra);
}

TaskView::SolutionShared TaskView::cachedSolution(const QString& task_id, uint32_t id) const
{
  auto it = tasks_.find(task_id);
  if (it == tasks_.end())
    return nullptr;
  auto jt = it.value().solutions.find(id);
  if (jt != it.value().solutions.end())
    return jt.value();
  return nullptr;
}

void TaskView::requestSolution(const QString& task_id, uint32_t id)
{
  auto it = tasks_.find(task_id);
  if (it == tasks_.end())
    return;
  TaskData& data = tasks_[task_id];
  if (data.solutions.contains(id) || data.pending.contains(id))
    return;
  ensureGetSolutionClient(task_id.toStdString());
  if (!get_solution_client_ || !get_solution_client_->service_is_ready())
    return;
  data.pending.insert(id);
  auto request =
      std::make_shared<curobo_task_constructor_interfaces::srv::GetSolution::Request>();
  request->solution_id = id;
  // Never block the GUI thread (MTC blocks here; the shared rviz node must
  // keep spinning): wait in a worker and marshal the reply back. The
  // QPointer guard drops the reply if the panel closed meanwhile.
  auto client = get_solution_client_;
  QPointer<TaskView> guard(this);
  std::thread([guard, client, request, task_id, id]() {
    bool delivered = false;
    try {
      auto future = client->async_send_request(request);
      if (future.wait_for(std::chrono::seconds(5)) == std::future_status::ready) {
        auto response = future.get();
        if (response) {
          delivered = true;
          SolutionShared msg = std::make_shared<
              curobo_task_constructor_interfaces::msg::Solution>(response->solution);
          QMetaObject::invokeMethod(
              guard.data(),
              [guard, task_id, id, msg]() {
                if (!guard)
                  return;
                guard->onSolutionReceived(task_id, id, msg);
              },
              Qt::QueuedConnection);
        }
      }
    } catch (...) {
    }
    if (!delivered) {
      // Timeout or error: release the in-flight mark so a later selection
      // revisit retries instead of leaving the row blank forever.
      QMetaObject::invokeMethod(
          guard.data(),
          [guard, task_id, id]() {
            if (!guard)
              return;
            guard->clearPending(task_id, id);
          },
          Qt::QueuedConnection);
    }
  }).detach();
}

void TaskView::clearPending(const QString& task_id, uint32_t id)
{
  auto it = tasks_.find(task_id);
  if (it != tasks_.end())
    it.value().pending.remove(id);
}

void TaskView::requestMissingSolutions(const QString& task_id, uint32_t stage_id)
{
  // MTC fetches only displayed rows (getSolution on current/selected
  // indexes); non-displayed rows fill via broadcast Solution messages.
  // This fans out to the selected stage's rows so its costs resolve without
  // clicking each row — same end state, one round trip per row at most.
  auto it = tasks_.find(task_id);
  if (it == tasks_.end())
    return;
  for (auto jt = it.value().rows.begin(); jt != it.value().rows.end(); ++jt) {
    if (jt.value().stage_id == stage_id && !it.value().solutions.contains(jt.key()))
      requestSolution(task_id, jt.key());
  }
}

void TaskView::onSolutionReceived(const QString& task_id, uint32_t id, SolutionShared msg)
{
  auto it = tasks_.find(task_id);
  if (it == tasks_.end() || !msg || msg->task_id != task_id.toStdString())
    return;
  it.value().pending.remove(id);
  // Fetched payload (not a live broadcast): fill rows silently. If it is
  // the currently selected row, the selection path below displays it.
  onSolutionMessage(msg, /*live=*/false);
  if (pending_exec_id_ == static_cast<int>(id) && pending_exec_task_ == task_id) {
    pending_exec_id_ = -1;
    pending_exec_task_.clear();
    onExecCurrentSolution();
  }
}

std::vector<sensor_msgs::msg::JointState> TaskView::currentWaypoints() const
{
  // The current row's full trajectory: every sub-trajectory's waypoints in
  // order (MTC shows the selected solution's whole motion).
  auto* current = solutions_view_->currentItem();
  if (!current)
    return {};
  const QString tid = current->data(1, Qt::UserRole).toString();
  const uint32_t id = static_cast<uint32_t>(current->data(0, Qt::UserRole).toUInt());
  return waypointsOf(cachedSolution(tid, id));
}

std::vector<sensor_msgs::msg::JointState> TaskView::waypointsOf(SolutionShared sol)
{
  std::vector<sensor_msgs::msg::JointState> out;
  if (!sol)
    return out;
  for (const auto& sub : sol->sub_trajectory) {
    const auto& jt = sub.trajectory.joint_trajectory;
    for (const auto& pt : jt.points) {
      sensor_msgs::msg::JointState wp;
      wp.name = jt.joint_names;
      wp.position = pt.positions;
      if (!out.empty() && !wp.position.empty() && !out.back().position.empty() &&
          out.back().position.size() == wp.position.size() && out.back().name == wp.name) {
        bool same = true;
        for (size_t j = 0; j < wp.position.size(); ++j) {
          if (out.back().position[j] != wp.position[j]) {
            same = false;
            break;
          }
        }
        if (same)
          continue;
      }
      out.push_back(wp);
    }
  }
  return out;
}

std::vector<visualization_msgs::msg::Marker> TaskView::selectedMarkers() const
{
  // Every selected row's markers (MTC shows all selected rows' markers).
  std::vector<visualization_msgs::msg::Marker> out;
  auto selected = solutions_view_->selectedItems();
  if (selected.isEmpty()) {
    auto* current = solutions_view_->currentItem();
    if (current)
      selected.push_back(current);
  }
  for (auto* row : selected) {
    if (!row)
      continue;
    const QString tid = row->data(1, Qt::UserRole).toString();
    const uint32_t id = static_cast<uint32_t>(row->data(0, Qt::UserRole).toInt());
    auto sol = cachedSolution(tid, id);
    if (!sol)
      continue;
    for (const auto& sub : sol->sub_solution) {
      for (const auto& marker : sub.info.markers)
        out.push_back(marker);
    }
  }
  return out;
}

void TaskView::publishSolutionTrajectory(
    const std::vector<sensor_msgs::msg::JointState>& waypoints)
{
  // The current solution as JointTrajectory for the CuroboTrajectoryDisplay
  // (full-robot animation + trail) — the same topic Task.publish uses.
  if (!pub_solution_trajectory_)
    return;
  trajectory_msgs::msg::JointTrajectory msg;
  if (!waypoints.empty()) {
    msg.joint_names = waypoints.front().name;
    double t = 0.0;
    for (const auto& wp : waypoints) {
      trajectory_msgs::msg::JointTrajectoryPoint pt;
      pt.positions = wp.position;
      pt.time_from_start.sec = static_cast<int32_t>(t);
      pt.time_from_start.nanosec = static_cast<uint32_t>((t - static_cast<int32_t>(t)) * 1e9);
      t += 0.05;
      msg.points.push_back(pt);
    }
  }
  pub_solution_trajectory_->publish(msg);
}

void TaskView::publishMarkers(const std::vector<visualization_msgs::msg::Marker>& markers)
{
  if (!pub_selected_markers_)
    return;
  visualization_msgs::msg::MarkerArray out;
  out.markers = markers;
  pub_selected_markers_->publish(out);
}

void TaskView::setStatus(const QString& text)
{
  status_label_->setText(text);
}

// ---------------------------------------------------------------------------
// GlobalSettingsWidget (MTC GlobalSettingsWidget)
// ---------------------------------------------------------------------------

GlobalSettingsWidget::GlobalSettingsWidget(TaskConstructorPanel* parent,
                                           rviz_common::properties::Property* root)
  : SubPanel(parent), root_(root)
{
  // MTC global_settings.ui verbatim: "Global Settings" label over a
  // PropertyTreeWidget. The tree widget (not a plain QTreeView) provides
  // the item delegates that render EnumProperty as dropdowns and
  // BoolProperty as checkboxes — a plain view shows values without them.
  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(0);
  layout->addWidget(new QLabel(tr("Global Settings"), this));
  view_ = new rviz_common::properties::PropertyTreeWidget(this);
  properties_ = new rviz_common::properties::PropertyTreeModel(root_, this);
  view_->setModel(properties_);
  layout->addWidget(view_);
  view_->expandAll();
  connect(properties_, &rviz_common::properties::PropertyTreeModel::configChanged, this,
          &GlobalSettingsWidget::configChanged);
}

GlobalSettingsWidget::~GlobalSettingsWidget() = default;

void GlobalSettingsWidget::save(rviz_common::Config config)
{
  root_->save(config);
}

void GlobalSettingsWidget::load(const rviz_common::Config& config)
{
  root_->load(config);
}

}  // namespace curobo_task_constructor_rviz

PLUGINLIB_EXPORT_CLASS(curobo_task_constructor_rviz::TaskConstructorPanel, rviz_common::Panel)
