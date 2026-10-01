#include <curobov2_ros_rviz/context_tab.hpp>

// QComboBox only forward-declares its view, and refreshPlannerNodeList() asks the
// popup whether it is on screen.
#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QFormLayout>
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
  // MoveIt also has "Warehouse" (the planning-database host and port) and
  // "Workspace" (the scene's centre and size) on this tab. curobo has neither: it
  // has no planning database, and the workspace centre/size live in the robot
  // config and in the world representation rather than in a service. Absent,
  // rather than two groups whose fields would be permanently disabled.
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

  // --- the one diagnostic switch, deliberately NOT in a group -----------------
  // This used to be a group box called "World Representation", which was the
  // worst of both: not a MoveIt title (MoveIt's Context tab has no such group),
  // and a titled box around a single checkbox, which is not a shape MoveIt ever
  // draws either. MoveIt puts lone checkboxes in the "Options" group on its
  // Planning tab, next to five other checkboxes; with no other checkbox to sit
  // with, the honest rendering is no box at all. It stays directly on the tab,
  // below the "Planning Library" group, where a single option belongs.
  collision_spheres_check_ = new QCheckBox(tr("Publish collision spheres"), this);
  collision_spheres_check_->setToolTip(
    tr("Ask the server to publish its collision-sphere markers. Diagnostic only "
       "— it changes nothing about what the planner will do."));
  connect(
    collision_spheres_check_, &QCheckBox::toggled, this,
    &ContextTab::applyCollisionSpheres);
  layout->addWidget(collision_spheres_check_);

  status_ = new QLabel(tr("waiting for the planner..."), this);
  status_->setWordWrap(true);
  layout->addWidget(status_);
  layout->addStretch(1);

  // Both combos change state on the server, so both start inert: the planner is
  // not known to be up yet, and the node is still the placeholder below. The
  // node list poll is what first proves anything exists.
  trajectory_type_combo_->setEnabled(false);

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
  refreshPlannerNodeList();

  // Read-only and cheap (parsed config, no solver touched), so polling is
  // insurance against missing the first response while the planner warms up.
  auto * timer = new QTimer(this);
  connect(timer, &QTimer::timeout, this, &ContextTab::refreshStrategies);
  timer->start(2000);

  // The same graph poll that used to live in the Planning tab, so a planner
  // (re)started later drops into the dropdown without a restart.
  auto * node_timer = new QTimer(this);
  connect(node_timer, &QTimer::timeout, this, &ContextTab::refreshPlannerNodeList);
  node_timer->start(2000);
}

void ContextTab::setPlannerReady(bool ready)
{
  planner_ready_ = ready;
  // The planner-node combo is deliberately NOT gated: repointing the panel at
  // another server is always allowed, it only re-runs the readiness probe.
  trajectory_type_combo_->setEnabled(ready);
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
  collision_spheres_client_ = node_->create_client<std_srvs::srv::SetBool>(
    ns + "set_collisions_enabled");
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
  if (collision_spheres_pending_ && collision_spheres_client_ != nullptr &&
      collision_spheres_client_->service_is_ready()) {
    collision_spheres_pending_ = false;
    applyCollisionSpheres(collision_spheres_check_->isChecked());
  }
  if (trajectory_type_pending_ && set_planner_client_ != nullptr &&
      set_planner_client_->service_is_ready()) {
    trajectory_type_pending_ = false;
    applyTrajectoryType();
  }

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

void ContextTab::applyCollisionSpheres(bool enabled)
{
  if (collision_spheres_client_ == nullptr || !collision_spheres_client_->service_is_ready()) {
    setStatus(tr("set_collisions_enabled service not available"));
    return;
  }
  auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
  request->data = enabled;
  collision_spheres_client_->async_send_request(
    request, [this, enabled](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future) {
      bool success = false;
      QString message;
      try {
        auto result = future.get();
        success = result->success;
        message = QString::fromStdString(result->message);
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread([this, enabled, success, message]() {
        // Put the checkbox back where the truth is: a failed call must not
        // leave the box showing a state the server never adopted.
        collision_spheres_check_->setChecked(success ? enabled : !enabled);
        setStatus(
          success ? tr("collision sphere publishing %1: %2")
                        .arg(enabled ? tr("enabled") : tr("disabled"), message)
                  : tr("could not change collision sphere publishing: %1").arg(message));
      });
    });
}

void ContextTab::save(rviz_common::Config config) const
{
  config.mapSetValue("planner_node_name", planner_node_);
  config.mapSetValue("trajectory_type", trajectory_type_);
  config.mapSetValue("collision_spheres", collision_spheres_check_->isChecked());
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

  // Read through a QVariant, like the collision-sphere flag below: mapGetString
  // on a numeric value comes back empty, and mapGetInt is not part of the Config
  // API this panel relies on.
  QVariant type_value;
  if (config.mapGetValue("trajectory_type", &type_value) && type_value.canConvert<int>()) {
    const int wanted = type_value.toInt();
    if (wanted >= 0 && wanted <= 1) {
      // Same reasoning as the collision-sphere flag below, plus one more:
      // setting the index emits currentIndexChanged, which would fire
      // set_planner straight out of load(), where there is no client and the
      // call would be dropped. The flag hands it to refreshStrategies(), which
      // retries until the service exists.
      trajectory_type_pending_ = true;
      trajectory_type_combo_->setCurrentIndex(wanted);
    }
  }

  // A bool is read through a QVariant: mapGetString on a bool value comes back
  // empty, and this is the accessor the sibling panels' "ExecuteOnServer" key is
  // read with.
  QVariant spheres;
  if (config.mapGetValue("collision_spheres", &spheres) && spheres.canConvert<bool>()) {
    // setChecked() emits toggled() only when the value CHANGES, which would fire
    // a service call straight out of load(). The flag hands it to
    // refreshStrategies() instead.
    const bool wanted = spheres.toBool();
    collision_spheres_pending_ = true;
    if (collision_spheres_check_->isChecked() != wanted) {
      collision_spheres_check_->setChecked(wanted);
    }
  }
}

}  // namespace curobov2_ros_rviz
