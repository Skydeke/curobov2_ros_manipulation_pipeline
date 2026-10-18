#include <curobov2_ros_rviz/context_tab.hpp>

// QComboBox only forward-declares its view, and refreshPlannerNodeList() asks the
// popup whether it is on screen.
#include <QAbstractItemView>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QStringList>
#include <QTimer>
#include <QVBoxLayout>
#include <QVariant>

#include <algorithm>
#include <cstdint>

namespace curobov2_ros_rviz
{

ContextTab::ContextTab(QWidget * parent) : QWidget(parent)
{
  auto * layout = new QVBoxLayout(this);
  layout->setContentsMargins(4, 4, 4, 4);

  // --- Planning Library: which server, and how it plans and is driven ---------
  // MoveIt's Context tab opens with a group called "Planning Library": a combo
  // naming the planning library, a planner-parameter widget, and nothing else.
  // The first row of MoveIt's PLANNING tab is "Planning Group:", a combo of
  // planning groups. Both are the same question -- which planner am I talking to
  // -- asked of two different layers, and curobo collapses them into one answer,
  // the planner node. So this group is "Planning Library" and its first row keeps
  // MoveIt's "Planning Group:" label, which is what that label means.
  //
  // MoveIt also has a "Warehouse" group here (the planning-database host and
  // port). curobo has no planning database, so that group is absent rather than
  // rendered permanently disabled. MoveIt's third group, "Workspace", IS ported
  // — it is built below.
  auto * server_group = new QGroupBox(tr("Planning Library"), this);
  auto * server_form = new QFormLayout(server_group);
  // Same field-growth policy the other four tabs use, so this form and theirs
  // line up.
  server_form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

  planner_node_combo_ = new QComboBox(server_group);
  planner_node_combo_->setEditable(true);
  planner_node_combo_->setToolTip(
    tr("Which planner node (unified_planner instance) every tab in this panel "
       "talks to — the curobo equivalent of MoveIt's planning group. The list "
       "is the live ROS node graph, refreshed every 2 s, and it is editable so a "
       "planner that is not up yet can still be named. This is the only "
       "editable copy in the panel: the other tabs follow this one."));
  server_form->addRow(tr("Planning Group:"), planner_node_combo_);

  trajectory_type_combo_ = new QComboBox(server_group);
  trajectory_type_combo_->addItem(tr("Classic (MotionGen)"));
  trajectory_type_combo_->addItem(tr("MPC (Real-time)"));
  trajectory_type_combo_->setToolTip(
    tr("How the server should produce trajectories. Classic solves once per "
       "request; MPC keeps running and follows the target marker live, which is "
       "also what makes the Planning tab's \"Compute + Execute\" start the "
       "live-tracking loop. Sent to the server as set_planner, and put back if "
       "the server refuses."));
  server_form->addRow(tr("Trajectory type:"), trajectory_type_combo_);

  current_strategy_label_ = new QLabel(tr("unknown"), server_group);
  current_strategy_label_->setToolTip(
    tr("How the generated trajectory is handed to the robot. Read back from the "
       "server's get_robot_strategies."));
  server_form->addRow(tr("Active strategy:"), current_strategy_label_);

  strategy_combo_ = new QComboBox(server_group);
  strategy_combo_->setToolTip(
    tr("Available control strategies, as reported by the server. Nothing is "
       "sent until Apply strategy is pressed, so refreshing this list — which "
       "happens on a timer — can never re-send set_robot_strategy by itself."));
  server_form->addRow(tr("Control strategy:"), strategy_combo_);

  // MoveIt's "Planning Attempt" row, which is where a lone commit button goes in
  // a MoveIt group. An unlabelled addRow(QString(), button) left the button
  // hanging under the field column with nothing to say what it commits; the strip
  // form matches the other tabs and gives it the MoveIt row title.
  auto * attempt_buttons = new QHBoxLayout();
  apply_button_ = new QPushButton(tr("Apply strategy"), server_group);
  apply_button_->setToolTip(
    tr("Call set_robot_strategy with the selected value. Refused while a "
       "trajectory is executing."));
  connect(apply_button_, &QPushButton::clicked, this, &ContextTab::applyStrategy);
  attempt_buttons->addWidget(apply_button_);
  attempt_buttons->addStretch(1);
  server_form->addRow(tr("Planning Attempt:"), attempt_buttons);

  layout->addWidget(server_group);

  // --- Workspace: the collision workspace (Mapper TSDF/ESDF extent) ----------
  // MoveIt's third Context group, "Workspace" — the scene's centre and size,
  // which in curobo is exactly the Mapper's TSDF/ESDF grid extent: the box the
  // cameras can observe, the box whose voxels become collision geometry, and
  // the box the `mapper_workspace` wireframe draws. Same grid as MoveIt: a
  // "Center (XYZ):" row of three spin boxes, then a "Size (XYZ):" row.
  //
  // The one deliberate deviation: MoveIt commits on every spin-box change
  // because its scene update is instant, whereas a curobo resize rebuilds the
  // Mapper and every solver (blocking, ~20 s) — so a commit button is added and
  // nothing reaches the server until it is pressed.
  auto * workspace_group = new QGroupBox(tr("Workspace"), this);
  auto * workspace_grid = new QGridLayout(workspace_group);
  workspace_grid->setContentsMargins(6, 4, 6, 4);

  auto make_workspace_spin = [&](bool is_size, const QString & tip)
  {
    auto * spin = new QDoubleSpinBox(workspace_group);
    spin->setDecimals(3);
    spin->setSingleStep(0.05);
    // Centre may be anywhere; a size must be positive, which the server enforces
    // too — the widget range only stops obviously-invalid input at the source.
    spin->setRange(is_size ? 0.0 : -10000.0, 10000.0);
    spin->setToolTip(tip);
    // Editing never sends anything (that is Apply's job); it only marks the
    // group dirty so the 2 s poll stops overwriting what the user is typing.
    connect(
      spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
      &ContextTab::markWorkspaceDirty);
    return spin;
  };

  workspace_grid->addWidget(new QLabel(tr("Center (XYZ):"), workspace_group), 0, 0);
  for (int i = 0; i < 3; ++i) {
    workspace_center_[i] = make_workspace_spin(
      false,
      tr("Centre of the collision workspace in the robot base frame (x/y/z, "
         "metres). This is the centre of the Mapper's TSDF/ESDF grid — the same "
         "box the cyan mapper_workspace wireframe draws."));
    workspace_grid->addWidget(workspace_center_[i], 0, i + 1);
  }
  workspace_grid->addWidget(new QLabel(tr("Size (XYZ):"), workspace_group), 1, 0);
  for (int i = 0; i < 3; ++i) {
    workspace_size_[i] = make_workspace_spin(
      true,
      tr("Size of the collision workspace along x/y/z (metres, all > 0). Only "
         "this box is sensed and only the voxels inside it collide; making it "
         "smaller also shrinks the solvers' voxel cache, which makes planning "
         "cheaper but hides anything outside the box."));
    workspace_grid->addWidget(workspace_size_[i], 1, i + 1);
  }

  workspace_apply_button_ = new QPushButton(tr("Apply workspace"), workspace_group);
  workspace_apply_button_->setToolTip(
    tr("Send the centre and size above to the server (set_workspace). The server "
       "rebuilds its Mapper and every solver against the new box — blocking, "
       "~20 s — and is refused while a trajectory is executing. The fields are "
       "always re-read from the server, so what is shown is what is in force."));
  connect(
    workspace_apply_button_, &QPushButton::clicked, this, &ContextTab::applyWorkspace);
  workspace_grid->addWidget(workspace_apply_button_, 2, 0, 1, 4);

  workspace_info_label_ = new QLabel(tr("workspace unknown"), workspace_group);
  workspace_info_label_->setWordWrap(true);
  workspace_grid->addWidget(workspace_info_label_, 3, 0, 1, 4);

  layout->addWidget(workspace_group);

  // Collision-sphere visualisation lives on the trajectory display now (Show
  // Collision / Show Attached over the spheres topic), not here: it is
  // display state, and a per-server enable switch for it does not belong on
  // this tab.
  status_ = new QLabel(tr("waiting for the planner..."), this);
  status_->setWordWrap(true);
  layout->addWidget(status_);
  layout->addStretch(1);

  // Both combos change state on the server, so both start inert: the planner is
  // not known to be up yet, and the node is still the placeholder below. The
  // node list poll is what first proves anything exists. The workspace group is
  // gated the same way (see setPlannerReady): a cheap get_workspace could run
  // early, but an Apply implies a blocking server-side rebuild, so the controls
  // wait until the planner is confirmed ready.
  trajectory_type_combo_->setEnabled(false);
  setPlannerReady(false);

  connect(
    planner_node_combo_, &QComboBox::currentTextChanged, this,
    &ContextTab::onPlannerNodeChosen);
  connect(
    trajectory_type_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
    &ContextTab::onTrajectoryTypeChosen);
}

ContextTab::~ContextTab()
{
  // No spinner of our own: this tab runs on rviz's shared node, which rviz
  // already spins. See the sibling tabs for the full reasoning.
}

void ContextTab::initialize(rclcpp::Node::SharedPtr node)
{
  if (initialized_ || node == nullptr) {
    return;
  }
  node_ = node;
  initialized_ = true;

  // Seed the editable combo with whatever load() (or the default) settled on.
  // blockSignals: the value is already in planner_node_, and the handler would
  // only re-derive what is set here.
  planner_node_combo_->blockSignals(true);
  planner_node_combo_->setCurrentText(planner_node_);
  planner_node_combo_->blockSignals(false);

  // A launch-time override is honoured, but only when the rviz config did not
  // already supply a value: the config is what the user last chose in this GUI,
  // and silently replacing it with a launch default would be the more surprising
  // behaviour. has_parameter() is the guard because rviz is not obliged to
  // declare this parameter -- get_parameter() on an undeclared name throws.
  if (!planner_node_from_config_ && node_->has_parameter("planner_node_name")) {
    try {
      const QString from_launch =
        QString::fromStdString(node_->get_parameter("planner_node_name").as_string()).trimmed();
      if (!from_launch.isEmpty()) {
        planner_node_ = from_launch;
        planner_node_combo_->setCurrentText(planner_node_);
      }
    } catch (const std::exception &) {
      // Wrong type, or declared and then removed. The default stands.
    }
  }

  rebuildClients();
  refreshStrategies();
  refreshWorkspace();
  refreshPlannerNodeList();

  // The same graph poll that used to live in the Planning tab, so a planner
  // (re)started later drops into the dropdown without a restart. The visibility
  // check is the performance guard the audit called out: this tab's three polls
  // are pure UI refreshes, so while the tab is hidden the timer still ticks but
  // the work (a node-graph query / service call / combo rebuild) is skipped.
  auto * node_timer = new QTimer(this);
  connect(node_timer, &QTimer::timeout, this, [this]()
  {
    if (isVisible()) {
      refreshPlannerNodeList();
    }
  });
  node_timer->start(2000);

  // Periodically refresh the control-strategy list. The service may not be
  // ready at initialize() time (the curobo server can take a while to come
  // up), and without a timer the dropdown would stay empty forever.
  auto * strategy_timer = new QTimer(this);
  connect(strategy_timer, &QTimer::timeout, this, [this]()
  {
    if (isVisible()) {
      refreshStrategies();
    }
  });
  strategy_timer->start(2000);

  // Same reasoning for the workspace fields: they are a get_workspace away, so
  // the poll is what first fills them and what reflects a `ros2 param set` made
  // from outside this panel. It only writes the fields when the user has not
  // edited them (see refreshWorkspace / markWorkspaceDirty).
  auto * workspace_timer = new QTimer(this);
  connect(workspace_timer, &QTimer::timeout, this, [this]()
  {
    if (isVisible()) {
      refreshWorkspace();
    }
  });
  workspace_timer->start(2000);
}

void ContextTab::setPlannerReady(bool ready)
{
  planner_ready_ = ready;
  // The planner-node combo is deliberately NOT gated: repointing the panel at
  // another server is always allowed, it only re-runs the readiness probe.
  trajectory_type_combo_->setEnabled(ready);
  // Same gate for the workspace controls: an edit is a blocking (~20 s) rebuild
  // on that server, so they stay inert until the planner is confirmed up.
  for (QDoubleSpinBox * spin : workspace_center_) {
    spin->setEnabled(ready);
  }
  for (QDoubleSpinBox * spin : workspace_size_) {
    spin->setEnabled(ready);
  }
  workspace_apply_button_->setEnabled(ready);
}

void ContextTab::onPlannerNodeChosen(const QString & text)
{
  const QString chosen = text.trimmed();
  if (chosen.isEmpty()) {
    // Mid-typing in the editable combo, or the field was cleared. Put the last
    // good value back rather than binding the panel to "".
    planner_node_combo_->setCurrentText(planner_node_);
    return;
  }
  if (chosen == planner_node_) {
    return;
  }

  planner_node_ = chosen;
  if (node_ != nullptr) {
    rebuildClients();
    refreshStrategies();
    // A different planner is a different workspace; drop any pending edit and
    // read the new server's box rather than leaving the old one on screen.
    workspace_dirty_ = false;
    refreshWorkspace();
  }
  setStatus(tr("planner node: %1").arg(planner_node_));

  // Emitted even before initialize(): CuroboPanel's wiring exists from
  // construction, and load() runs before onInitialize(), so this is how a saved
  // planner node reaches the other four tabs at all.
  Q_EMIT plannerNodeChosen(planner_node_);
}

void ContextTab::refreshPlannerNodeList()
{
  if (node_ == nullptr) {
    return;  // initialize() will do the first pass once it has a node
  }

  // Same data `ros2 node list` reports, queried via the node graph API (no
  // hardcoded dropdown values). Raw names come back "/"-prefixed.
  std::vector<std::string> names;
  try {
    for (auto n : node_->get_node_names()) {
      if (!n.empty() && n.front() == '/') {
        n.erase(n.begin());
      }
      names.push_back(n);
    }
  } catch (const std::exception & e) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000, "Failed to list ROS nodes: %s",
      e.what());
    return;
  }

  const std::set<std::string> seen(names.begin(), names.end());
  if (seen == last_planner_nodes_) {
    return;  // graph unchanged; don't churn the dropdown mid-interaction
  }
  last_planner_nodes_ = seen;

  // Never rebuild while the popup is open or the field has focus: clearing the
  // combo under the user's cursor would drop the selection they are making.
  if (planner_node_combo_->hasFocus() || planner_node_combo_->view()->isVisible()) {
    return;
  }

  // Rebuild the items, preserving whatever the user currently has in the
  // (editable) combo — even if it is not (yet) a live node.
  //
  // Signals are blocked for the whole rebuild, and that is load-bearing rather
  // than tidy: clear() fires currentTextChanged("") and the restoring
  // setCurrentText() fires it again. Left connected, a plain list refresh would
  // run onPlannerNodeChosen() and — if the user happened to have half a name
  // typed that is not yet the bound node — rebind the entire panel to it. This
  // function only ever populates a dropdown; it must never change server state.
  const QString previous = planner_node_combo_->currentText();
  planner_node_combo_->blockSignals(true);
  planner_node_combo_->clear();
  std::sort(names.begin(), names.end());
  for (const auto & n : names) {
    planner_node_combo_->addItem(QString::fromStdString(n));
  }
  // addItem() does not change an editable combo's text, so the text is restored
  // explicitly here.
  planner_node_combo_->setCurrentText(previous);
  planner_node_combo_->blockSignals(false);
}

void ContextTab::rebuildClients()
{
  const std::string ns = "/" + planner_node_.toStdString() + "/";
  get_strategies_client_ = node_->create_client<curobov2_ros_interfaces::srv::GetRobotStrategies>(
    ns + "get_robot_strategies");
  set_strategy_client_ = node_->create_client<curobov2_ros_interfaces::srv::SetRobotStrategy>(
    ns + "set_robot_strategy");
  set_planner_client_ = node_->create_client<curobov2_ros_interfaces::srv::SetPlanner>(
    ns + "set_planner");
  get_workspace_client_ = node_->create_client<curobov2_ros_interfaces::srv::GetWorkspace>(
    ns + "get_workspace");
  set_workspace_client_ = node_->create_client<curobov2_ros_interfaces::srv::SetWorkspace>(
    ns + "set_workspace");
}

void ContextTab::runOnGuiThread(std::function<void()> fn)
{
  QMetaObject::invokeMethod(this, std::move(fn), Qt::QueuedConnection);
}

void ContextTab::setStatus(const QString & text)
{
  status_->setText(text);
}

void ContextTab::refreshStrategies()
{
  // Runs on the GUI thread (QTimer), so it is also where restored settings are
  // finally pushed. They cannot be sent from load(): rviz calls load() before
  // onInitialize(), so at that point there is no node and therefore no client,
  // and the widget would be left showing a state the server never adopted.
  // Retried here until the service is up rather than sent once into the void,
  // because service_is_ready() is false for a while after the planner starts.
  // NOTE: trajectory_type_pending_ is intentionally NOT applied here. Sending
  // set_planner on startup (before the user has interacted with the panel)
  // changes server state without being asked, and can override the planner
  // that another client (e.g. the task constructor) has already selected.
  // The combo is restored from the config for display only; the user must
  // explicitly change it to send anything to the server.

  if (get_strategies_client_ == nullptr || !get_strategies_client_->service_is_ready()) {
    return;
  }
  auto request = std::make_shared<curobov2_ros_interfaces::srv::GetRobotStrategies::Request>();
  get_strategies_client_->async_send_request(
    request,
    [this](
      rclcpp::Client<curobov2_ros_interfaces::srv::GetRobotStrategies>::SharedFuture future) {
      QStringList names;
      QString current;
      bool success = false;
      try {
        auto result = future.get();
        success = result->success;
        current = QString::fromStdString(result->current_strategy_name);
        for (const auto & n : result->strategy_names) {
          names << QString::fromStdString(n);
        }
      } catch (const std::exception & e) {
        // This catch runs on rviz's ROS thread, not the GUI thread, so it must
        // not touch a widget directly.
        const QString error = tr("get_robot_strategies failed: %1").arg(QString::fromStdString(e.what()));
        runOnGuiThread([this, error]() { setStatus(error); });
        return;
      }
      runOnGuiThread([this, names, current, success]() {
        if (!success) {
          return;  // keep the last known list; the error is already on screen
        }
        // Skip a full combo rebuild when nothing changed: clear()+addItems()
        // under the user's cursor is the one thing this 2 s poll must not do
        // gratuitously (it also drops any selection the user was making).
        if (names == last_strategies_ && current == last_current_strategy_) {
          return;
        }
        last_strategies_ = names;
        last_current_strategy_ = current;

        current_strategy_label_->setText(current.isEmpty() ? tr("unknown") : current);
        apply_button_->setEnabled(!names.isEmpty());

        // Re-populating the combo only CHANGES the selection; no signal is
        // connected to currentIndexChanged, so this cannot send anything.
        strategy_combo_->clear();
        strategy_combo_->addItems(names);
        const int idx = names.indexOf(current);
        if (idx >= 0) {
          strategy_combo_->setCurrentIndex(idx);
        }
      });
    });
}

void ContextTab::onTrajectoryTypeChosen(int index)
{
  // Also the re-entrancy guard for the two places below that put the combo back:
  // setCurrentIndex() re-emits currentIndexChanged, and without this a refused
  // switch would bounce between the slot and itself.
  if (index < 0 || index == trajectory_type_) {
    return;
  }
  if (!planner_ready_ || set_planner_client_ == nullptr ||
      !set_planner_client_->service_is_ready()) {
    // Put the combo back: the server is not in a state to be told anything, and
    // leaving the user's pick on screen would be a claim it never adopted.
    trajectory_type_combo_->setCurrentIndex(trajectory_type_);
    setStatus(tr("planner is not ready; trajectory type unchanged"));
    return;
  }
  applyTrajectoryType();
}

void ContextTab::applyTrajectoryType()
{
  if (set_planner_client_ == nullptr || !set_planner_client_->service_is_ready()) {
    setStatus(tr("set_planner service not available"));
    return;
  }
  const int index = trajectory_type_combo_->currentIndex();
  if (index < 0) {
    return;
  }
  // Combo order is Classic, MPC, and SetPlanner's CLASSIC/MPC are 0/1, so the
  // index forwards directly. Kept explicit rather than assumed.
  const uint8_t planner_type = static_cast<uint8_t>(index);

  setStatus(
    tr("switching trajectory type to '%1'...").arg(trajectory_type_combo_->itemText(index)));
  auto request = std::make_shared<curobov2_ros_interfaces::srv::SetPlanner::Request>();
  request->planner_type = planner_type;

  set_planner_client_->async_send_request(
    request,
    [this, planner_type](rclcpp::Client<curobov2_ros_interfaces::srv::SetPlanner>::SharedFuture future) {
      bool success = false;
      QString message;
      try {
        auto result = future.get();
        success = result->success;
        message = QString::fromStdString(result->message);
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread([this, planner_type, success, message]() {
        if (!success) {
          // The server kept whatever it had. Show that, not the request.
          trajectory_type_combo_->setCurrentIndex(trajectory_type_);
          setStatus(tr("could not switch trajectory type: %1").arg(message));
          return;
        }
        trajectory_type_ = static_cast<int>(planner_type);
        setStatus(
          tr("trajectory type is now '%1': %2")
            .arg(trajectory_type_combo_->itemText(trajectory_type_), message));
        // Only now is the other tab's MPC decision safe to act on.
        Q_EMIT trajectoryTypeChanged(trajectory_type_);
      });
    });
}

void ContextTab::applyStrategy()
{
  if (set_strategy_client_ == nullptr || !set_strategy_client_->service_is_ready()) {
    setStatus(tr("set_robot_strategy service not available"));
    return;
  }
  const QString chosen = strategy_combo_->currentText();
  if (chosen.isEmpty()) {
    return;
  }
  setStatus(tr("switching control strategy to '%1'...").arg(chosen));
  auto request = std::make_shared<curobov2_ros_interfaces::srv::SetRobotStrategy::Request>();
  request->robot_strategy = chosen.toStdString();
  set_strategy_client_->async_send_request(
    request,
    [this, chosen](
      rclcpp::Client<curobov2_ros_interfaces::srv::SetRobotStrategy>::SharedFuture future) {
      bool success = false;
      QString message;
      try {
        auto result = future.get();
        success = result->success;
        message = QString::fromStdString(result->message);
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread([this, chosen, success, message]() {
        // Re-read rather than trusting the local label: the server reports which
        // strategy actually won, which need not be the one requested.
        refreshStrategies();
        setStatus(
          success ? tr("control strategy is now '%1': %2").arg(chosen, message)
                  : tr("could not switch strategy: %1").arg(message));
      });
    });
}

void ContextTab::markWorkspaceDirty()
{
  // An edit has not reached the server yet (Apply does that), so from now until
  // Apply succeeds the fields belong to the user and the 2 s poll must not
  // overwrite them.
  workspace_dirty_ = true;
}

void ContextTab::showWorkspace(
  const QVector<double> & center, const QVector<double> & extent, double voxel_size,
  bool perception_active)
{
  // blockSignals on every write: setValue() emits valueChanged, which is wired
  // to markWorkspaceDirty() — a programmatic refresh must not make the group
  // look dirtied by the user.
  if (center.size() == 3) {
    for (int i = 0; i < 3; ++i) {
      workspace_center_[i]->blockSignals(true);
      workspace_center_[i]->setValue(center[i]);
      workspace_center_[i]->blockSignals(false);
    }
  }
  if (extent.size() == 3) {
    for (int i = 0; i < 3; ++i) {
      workspace_size_[i]->blockSignals(true);
      workspace_size_[i]->setValue(extent[i]);
      workspace_size_[i]->blockSignals(false);
    }
  }
  workspace_perception_active_ = perception_active;

  const QString voxel = tr("voxel %1 m").arg(QString::number(voxel_size, 'f', 3));
  workspace_info_label_->setText(
    perception_active
      ? tr("%1 — this box is sensed and collides.").arg(voxel)
      : tr("%1 — no perception source: the box and its wireframe move, but no "
           "collision geometry is built for it.").arg(voxel));
  workspace_info_label_->setToolTip(
    tr("The collision workspace is the Mapper's TSDF/ESDF grid: everything "
       "inside it can become collision geometry, nothing outside it can. The "
       "solvers' voxel cache is sized from the same box, so a smaller box also "
       "makes planning cheaper."));
}

void ContextTab::refreshWorkspace()
{
  if (get_workspace_client_ == nullptr || !get_workspace_client_->service_is_ready()) {
    return;
  }
  auto request = std::make_shared<curobov2_ros_interfaces::srv::GetWorkspace::Request>();
  get_workspace_client_->async_send_request(
    request,
    [this](
      rclcpp::Client<curobov2_ros_interfaces::srv::GetWorkspace>::SharedFuture future) {
      QVector<double> center;
      QVector<double> extent;
      double voxel = 0.0;
      bool perception_active = false;
      bool success = false;
      try {
        auto result = future.get();
        success = result->success;
        for (double v : result->center) {
          center << v;
        }
        for (double v : result->extent) {
          extent << v;
        }
        voxel = result->voxel_size;
        perception_active = result->perception_active;
      } catch (const std::exception &) {
        // Runs on rviz's ROS thread; nothing here touches a widget.
        return;
      }
      runOnGuiThread(
        [this, center, extent, voxel, perception_active, success]()
        {
          if (!success) {
            return;  // keep the last known values on screen
          }
          // A poll must never wipe an edit in progress: while the group is dirty
          // the fields are the user's and only Apply may replace them. The
          // perception state is still refreshed, since it is not editable.
          if (workspace_dirty_) {
            workspace_perception_active_ = perception_active;
            return;
          }
          showWorkspace(center, extent, voxel, perception_active);
        });
    });
}

void ContextTab::applyWorkspace()
{
  if (set_workspace_client_ == nullptr || !set_workspace_client_->service_is_ready()) {
    setStatus(tr("set_workspace service not available"));
    return;
  }
  if (!planner_ready_) {
    setStatus(tr("planner is not ready; workspace unchanged"));
    return;
  }
  auto request = std::make_shared<curobov2_ros_interfaces::srv::SetWorkspace::Request>();
  for (int i = 0; i < 3; ++i) {
    request->center.push_back(workspace_center_[i]->value());
    request->extent.push_back(workspace_size_[i]->value());
  }

  setStatus(
    tr("setting the workspace — the server rebuilds its Mapper and every solver "
       "(blocking, ~20 s). Watch its console; this panel stays responsive."));
  set_workspace_client_->async_send_request(
    request,
    [this](
      rclcpp::Client<curobov2_ros_interfaces::srv::SetWorkspace>::SharedFuture future) {
      QVector<double> center;
      QVector<double> extent;
      double voxel = 0.0;
      bool perception_active = false;
      bool success = false;
      QString message;
      try {
        auto result = future.get();
        success = result->success;
        message = QString::fromStdString(result->message);
        for (double v : result->center) {
          center << v;
        }
        for (double v : result->extent) {
          extent << v;
        }
        voxel = result->voxel_size;
        perception_active = result->perception_active;
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread(
        [this, center, extent, voxel, perception_active, success, message]()
        {
          if (success) {
            // Adopt the SERVER's answer — the echoed values in force, never the
            // request — and clear the dirty flag: the fields now show truth.
            workspace_dirty_ = false;
            showWorkspace(center, extent, voxel, perception_active);
          }
          // On failure the user's values are left in place (and the group stays
          // dirty) so a rejected edit — e.g. a box larger than the cap — can be
          // corrected instead of being silently reverted.
          setStatus(success ? tr("workspace updated: %1").arg(message)
                            : tr("could not set the workspace: %1").arg(message));
        });
    });
}

void ContextTab::save(rviz_common::Config config) const
{
  config.mapSetValue("planner_node_name", planner_node_);
  config.mapSetValue("trajectory_type", trajectory_type_);
}

void ContextTab::load(const rviz_common::Config & config)
{
  QString planner_node;
  if (config.mapGetString("planner_node_name", &planner_node) && !planner_node.isEmpty()) {
    // Routed through the combo rather than assigned to planner_node_ directly,
    // so the emitted plannerNodeChosen reaches the other four tabs -- which is
    // the whole point of this tab owning the setting. That happens with no node
    // yet (rviz loads before onInitialize); the handler only skips client
    // rebuilds in that state, not the notification.
    planner_node_from_config_ = true;
    planner_node_combo_->setCurrentText(planner_node.trimmed());
  }

  // Read through a QVariant, like the sibling panels' keys: mapGetString
  // on a numeric value comes back empty, and mapGetInt is not part of the Config
  // API this panel relies on.
  QVariant type_value;
  if (config.mapGetValue("trajectory_type", &type_value) && type_value.canConvert<int>()) {
    const int wanted = type_value.toInt();
    if (wanted >= 0 && wanted <= 1) {
      // Same reasoning as the sibling panels' pending flags, plus one more:
      // setting the index emits currentIndexChanged, which would fire
      // set_planner straight out of load(), where there is no client and the
      // call would be dropped. The flag hands it to refreshStrategies(), which
      // retries until the service exists.
      trajectory_type_pending_ = true;
      trajectory_type_combo_->setCurrentIndex(wanted);
    }
  }
}

}  // namespace curobov2_ros_rviz
