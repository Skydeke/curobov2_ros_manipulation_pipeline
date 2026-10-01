#include <curobov2_ros_rviz/joints_tab.hpp>

#include <curobov2_ros_rviz/progress_bar_delegate.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <QAbstractItemView>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPointF>
#include <QPushButton>
#include <QTimer>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>

namespace curobov2_ros_rviz
{

namespace
{

enum Column
{
  COL_JOINT,
  COL_CURRENT,
  COL_TARGET,
  COL_COUNT
};

/// Bounds of a row, or a null QPointF for a joint the model reports no limit for.
/// Absent bounds are what make such a row plain text and not draggable, rather
/// than a bar spanning a made-up range.
QPointF boundsOf(const QTreeWidgetItem * item)
{
  return item->data(COL_TARGET, ProgressBarDelegate::VariableBoundsRole).toPointF();
}

/// Whether this row has bounds at all, and so is drawn and dragged as a bar.
///
/// The test is on the QVariant, NOT on the QPointF: QPointF has no isValid()
/// (only QPoint, QSize, QSizeF, QRect and QRectF do), so a y() > x() test would
/// be the only geometric option -- and the wrong one to use, because this must
/// agree EXACTLY with ProgressBarDelegate::paint's own test. The delegate decides
/// "bar or plain text" from `index.data(role).isValid()`, so if this tab decided
/// it from the geometry instead, the two could disagree: the tab would clamp a
/// cell the delegate leaves as plain text, and a QPointF(0, 0) written for an
/// unbounded joint is indistinguishable from a real zero-width bound.
bool hasBounds(const QTreeWidgetItem * item)
{
  return item->data(COL_TARGET, ProgressBarDelegate::VariableBoundsRole).isValid();
}

/// The target value of a row. This lives in the ITEM rather than in a child
/// spinbox because MoveIt's joint table has no spinbox: the cell is painted as a
/// progress bar by ProgressBarDelegate and edited by dragging it. EditRole, not
/// DisplayRole, because that is the role a committed drag writes to -- see the
/// comment in ProgressBarDelegate::paint.
double targetOf(const QTreeWidgetItem * item)
{
  return item->data(COL_TARGET, Qt::EditRole).toDouble();
}

/// Writes a row's target and repaints it. EVERY write in this file goes through
/// here, so the clamp to the joint's own limits cannot be bypassed by one path.
void setTarget(QTreeWidgetItem * item, double value)
{
  if (hasBounds(item)) {
    const QPointF limits = boundsOf(item);
    value = std::clamp(value, limits.x(), limits.y());
  }
  // ARGUMENT ORDER: QTreeWidgetItem::data is (column, role) but
  // QTreeWidgetItem::setData is (column, role, value) -- the value goes LAST,
  // not in the middle. Qt 5 declares only that one form, so the natural-looking
  // setData(column, value, role) does not merely fail to compile: `double` is
  // implicitly convertible to `int`, so it binds as setData(column, role=(int)
  // value, value=QVariant(role)) and writes the role number into the cell as its
  // data, under a nonsense role. Nothing complains and the bar silently never
  // moves. data() and setData() taking their arguments in different orders is the
  // whole trap.
  item->setData(COL_TARGET, Qt::EditRole, value);
  // DisplayRole carries the same number as text, so a row with no bar (an
  // unbounded joint) still shows its value instead of going blank.
  item->setText(COL_TARGET, QString::number(value, 'f', 4));
}

}  // namespace

JointsTab::JointsTab(QWidget * parent) : QWidget(parent)
{
  auto * layout = new QVBoxLayout(this);
  layout->setContentsMargins(4, 4, 4, 4);

  // MoveIt's Joints tab, reproduced from the ground up rather than approximated.
  // Its whole layout is:
  //
  //     [QLabel] "Group joints:"
  //     [QTreeView joints_view_]      <- value column painted as a progress bar
  //     [QLabel] "Nullspace exploration:"
  //     [QSlider] x N                 <- one per nullspace basis vector
  //
  // That is: NO group boxes. A bare label, then the table. The nullspace
  // sliders are the one part with no curobo equivalent -- curobo's
  // generate_trajectory takes a goal configuration, not a nullspace goal, and
  // the nullspace goal-set slot is not in TrajectoryGoal -- so they are absent
  // rather than stubbed. Everything else is here, including the bar, which is
  // the part that made this tab unrecognisable as MoveIt's before.
  auto * joints_label = new QLabel(tr("Group joints:"), this);
  layout->addWidget(joints_label);

  tree_ = new QTreeWidget(this);
  tree_->setColumnCount(COL_COUNT);
  tree_->setHeaderLabels({tr("Joint"), tr("Current"), tr("Target")});
  tree_->setRootIsDecorated(false);
  tree_->setAlternatingRowColors(true);
  tree_->header()->setStretchLastSection(true);
  tree_->setToolTip(
    tr("Target is the configuration Plan / Plan + send will ask for, in the "
       "server's cspace order. Use Check to validate it before planning."));

  // MoveIt sets EditKeyPressed and NoSelection on this view. NoSelection is
  // dropped here: the rows are also what the Planning Attempt strip acts on, so
  // a row has to be selectable. Decoration and double-click expansion are off,
  // as in MoveIt, so the list stays a flat list of cspace joints.
  tree_->setEditTriggers(QAbstractItemView::EditKeyPressed);
  tree_->setItemDelegateForColumn(COL_TARGET, new ProgressBarDelegate(COL_TARGET, this));
  // The event filter is what makes a plain click on a bar open the editor, the
  // way it does in MoveIt. EditKeyPressed on its own would not: it needs a
  // keypress, and it would fire anywhere in the table, including the joint-name
  // column. The filter checks the column first.
  tree_->viewport()->installEventFilter(new ProgressBarEventFilter(tree_, COL_TARGET));
  // A QTreeWidget has no useful minimumSizeHint, so in a layout it will happily
  // shrink to a sliver and leave no way to reach the joints in a short dock.
  // This is not a scroll-area tab, so there is no other way to get at them.
  // 160 px is roughly seven rows, i.e. a Gen3 arm's worth.
  tree_->setMinimumHeight(160);
  layout->addWidget(tree_, 1);

  // MoveIt's "Commands" group, verbatim title and verbatim purpose: a group
  // whose only content is the plan/execute/stop strip ([&Plan] [&Execute]
  // [Plan && Execute] [&Stop] in MoveIt). Plan only computes here while
  // "Plan + send" also executes, so the two are not collapsed into one
  // "Execute" -- that would hide a real difference.
  auto * commands_group = new QGroupBox(tr("Commands"), this);
  auto * commands_form = new QFormLayout(commands_group);
  commands_form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
  auto * commands_buttons = new QHBoxLayout();
  auto add_command_button = [&](const QString & text, const QString & tip,
                                void (JointsTab::* slot)())
  {
    auto * b = new QPushButton(text, commands_group);
    b->setToolTip(tip);
    connect(b, &QPushButton::clicked, this, slot);
    commands_buttons->addWidget(b);
    return b;
  };

  // Editing the request rather than running it -- MoveIt's own split between
  // the [Update]/[Reset] buttons on its scene tabs and the [Plan]/[Execute]
  // pair in Commands.
  auto * attempt_buttons = new QHBoxLayout();
  auto add_attempt_button = [&](const QString & text, const QString & tip,
                                void (JointsTab::* slot)())
  {
    auto * b = new QPushButton(text, commands_group);
    b->setToolTip(tip);
    connect(b, &QPushButton::clicked, this, slot);
    attempt_buttons->addWidget(b);
    return b;
  };

  add_attempt_button(
    tr("Set from current"), tr("Copy every live /joint_states position into the target column"),
    &JointsTab::setTargetsFromCurrent);
  add_attempt_button(tr("Zero"), tr("Zero the targets (a valid, if unhelpful, configuration)"),
                     &JointsTab::zeroTargets);
  check_button_ = add_attempt_button(
    tr("Check"), tr("Run the server's FK/collision validation on the TARGET "
                    "configuration -- reports self-collision, scene collision "
                    "or a joint-limit violation before you plan"),
    &JointsTab::checkTarget);
  clear_locks_button_ = add_attempt_button(
    tr("Clear locks"),
    tr("Drop the joint-lock override and reload the robot config's own "
       "lock_joints. Rebuilds the model and every solver -- blocking, ~20 s, "
       "and refused while a goal executes."),
    &JointsTab::clearLocks);
  attempt_buttons->addStretch(1);
  commands_form->addRow(tr("Planning Attempt:"), attempt_buttons);

  plan_button_ = add_command_button(
    tr("Plan"), tr("Plan a joint-space trajectory to the target (no execution)"),
    &JointsTab::planTarget);
  send_button_ = add_command_button(
    tr("Plan + send"), tr("Plan and execute the joint-space trajectory"),
    &JointsTab::sendTarget);
  stop_button_ = add_command_button(tr("Stop"), tr("Cancel the executing goal"),
                                    &JointsTab::stopGoal);
  commands_buttons->addStretch(1);
  commands_form->addRow(tr("Execute:"), commands_buttons);
  layout->addWidget(commands_group);

  status_ = new QLabel(tr("waiting for the planner..."), this);
  status_->setWordWrap(true);
  layout->addWidget(status_);

  setButtonsEnabled();
}

JointsTab::~JointsTab()
{
  // No spinner of our own: this tab runs on rviz's shared node, which rviz
  // already spins in its own thread (RosNodeAbstraction). Adding that node to a
  // second executor would make rclcpp throw "Node has already been added to an
  // executor" — the sibling panels each create their OWN node for exactly that
  // reason.
}

void JointsTab::initialize(rclcpp::Node::SharedPtr node)
{
  if (initialized_ || node == nullptr) {
    return;
  }
  node_ = node;
  initialized_ = true;

  // Reentrant: rviz spins a single-threaded executor, so the /joint_states
  // subscription, the service responses and the Qt refreshes must be allowed
  // to interleave. Jazzy's rclcpp only takes the group inside
  // SubscriptionOptions.
  auto cb_group = node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = cb_group;

  joint_state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    "joint_states", rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) { onJointState(msg); },
    sub_options);

  setPlannerNode(planner_node_);

  // Poll the cspace: it is read-only and cheap (it reads the parsed kinematics
  // model, touches no solver), so a 2 s poll is cheap insurance against
  // missing the very first response while the planner is still warming up.
  auto * info_timer = new QTimer(this);
  connect(info_timer, &QTimer::timeout, this, &JointsTab::refreshJointInfo);
  info_timer->start(2000);
}

void JointsTab::setPlannerNode(const QString & planner_node)
{
  planner_node_ = planner_node.trimmed().isEmpty() ? QStringLiteral("curobo_server")
                                                   : planner_node.trimmed();
  if (node_ == nullptr) {
    return;  // initialize() will build the clients once it has a node
  }
  const std::string ns = "/" + planner_node_.toStdString() + "/";
  joint_info_client_ = node_->create_client<curobov2_ros_interfaces::srv::GetJointInfo>(
    ns + "get_joint_info");
  fk_client_ = node_->create_client<curobov2_ros_interfaces::srv::FkBatch>(
    ns + "fk_batch");
  joint_locks_client_ = node_->create_client<curobov2_ros_interfaces::srv::SetJointLocks>(
    ns + "set_joint_locks");
  trajectory_client_ =
    node_->create_client<curobov2_ros_interfaces::srv::TrajectoryGeneration>(
      ns + "generate_trajectory");
  action_client_ = rclcpp_action::create_client<curobov2_ros_interfaces::action::SendTrajectory>(
    node_, ns + "execute_trajectory");

  // A different planner is a different robot: the old cspace (and the target
  // positions indexed by it) mean nothing here.
  cspace_.clear();
  current_.clear();
  locked_.clear();
  cspace_known_ = false;
  tree_->clear();
  refreshJointInfo();
  setStatus(tr("planner node: %1").arg(planner_node_));
}

void JointsTab::runOnGuiThread(std::function<void()> fn)
{
  QMetaObject::invokeMethod(this, std::move(fn), Qt::QueuedConnection);
}

void JointsTab::setStatus(const QString & text)
{
  status_->setText(text);
}

void JointsTab::setButtonsEnabled()
{
  const bool ready = initialized_ && cspace_known_ && !goal_active_;
  plan_button_->setEnabled(ready);
  send_button_->setEnabled(ready);
  check_button_->setEnabled(ready);
  clear_locks_button_->setEnabled(initialized_ && !goal_active_);
  stop_button_->setEnabled(goal_active_);
}

void JointsTab::refreshJointInfo()
{
  if (joint_info_client_ == nullptr || !joint_info_client_->service_is_ready()) {
    return;
  }
  auto request = std::make_shared<curobov2_ros_interfaces::srv::GetJointInfo::Request>();
  joint_info_client_->async_send_request(
    request,
    [this](rclcpp::Client<curobov2_ros_interfaces::srv::GetJointInfo>::SharedFuture future)
    {
      QStringList names;
      QVector<double> lower;
      QVector<double> upper;
      QStringList locked;
      bool success = false;
      QString message;
      try {
        auto result = future.get();
        success = result->success;
        message = QString::fromStdString(result->message);
        for (const auto & n : result->joint_names) {
          names << QString::fromStdString(n);
        }
        for (double v : result->lower_limits) {
          lower << v;
        }
        for (double v : result->upper_limits) {
          upper << v;
        }
        for (const auto & n : result->locked_joint_names) {
          locked << QString::fromStdString(n);
        }
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread(
        [this, names, lower, upper, locked, success, message]()
        {
          if (!success) {
            setStatus(tr("get_joint_info failed: %1").arg(message));
            return;
          }
          const bool changed = (names != cspace_);
          cspace_ = names;
          locked_ = locked;
          limits_lower_ = lower;
          limits_upper_ = upper;
          cspace_known_ = !cspace_.isEmpty();
          if (changed || tree_->topLevelItemCount() == 0) {
            rebuildRows(cspace_, limits_lower_, limits_upper_);
          }
          setButtonsEnabled();
          if (cspace_known_) {
            setStatus(
              tr("%1 joint(s) in cspace%2 — locked: %3")
                .arg(cspace_.size())
                .arg(
                  locked_.isEmpty() ? QString()
                                   : tr(", %1 pinned out").arg(locked_.size()))
                .arg(locked_.isEmpty() ? tr("none") : locked_.join(", ")));
          }
        });
    });
}

void JointsTab::rebuildRows(
  const QStringList & names, const QVector<double> & lower, const QVector<double> & upper)
{
  tree_->clear();
  for (int i = 0; i < names.size(); ++i) {
    auto * item = new QTreeWidgetItem(tree_);
    item->setText(COL_JOINT, names[i]);
    // ItemIsEditable is what lets the delegate's editor open. It is an ITEM-level
    // flag, so it cannot say "target column only" -- ProgressBarDelegate::createEditor
    // returning nullptr for the other columns is what actually confines editing to
    // the bar, and ProgressBarEventFilter checks the column before opening it.
    item->setFlags(item->flags() | Qt::ItemIsEditable);

    // A joint the model reports no bound for arrives as +/-inf. MoveIt handles that
    // by simply not providing bounds, and ProgressBarDelegate then leaves the cell
    // as plain, non-draggable text -- so an unbounded joint cannot be dragged into
    // infinity, and nothing invents a range for it.
    const double lo = (i < lower.size()) ? lower[i] : 0.0;
    const double hi = (i < upper.size()) ? upper[i] : 0.0;
    // setData is (column, role, value) -- see the note in setTarget().
    if (std::isfinite(lo) && std::isfinite(hi) && hi > lo) {
      item->setData(
        COL_TARGET, ProgressBarDelegate::VariableBoundsRole, QPointF(lo, hi));
    } else {
      // Explicitly clear it rather than relying on the row being new: an
      // invalidate()d QVariant is the ONLY thing the delegate and hasBounds()
      // treat as "no bar", and a default-constructed QPointF(0, 0) is
      // indistinguishable from a real bound.
      item->setData(COL_TARGET, ProgressBarDelegate::VariableBoundsRole, QVariant());
    }
    item->setToolTip(
      COL_JOINT,
      hasBounds(item)
        ? tr("%1 — drag the bar to set the target, or click once to place it. "
             "Limits: %2 .. %3 rad.")
            .arg(names[i]).arg(lo).arg(hi)
        : tr("%1 — the model reports no position limit for this joint, so its "
             "target has to be typed rather than dragged.").arg(names[i]));

    // Restore a target loaded from the config (rviz calls load() BEFORE the
    // rows exist -- they only appear once get_joint_info answers, which needs a
    // node -- so the value has to be parked until here) and default to zero for
    // anything new. setTarget clamps to the bounds set just above.
    setTarget(item, (i < pending_targets_.size()) ? pending_targets_.at(i) : 0.0);

    item->setText(COL_CURRENT, "-");
  }
  tree_->resizeColumnToContents(COL_JOINT);

  // Consumed: applying a saved target list to a cspace whose shape has since
  // changed would map joint values onto the wrong joints, so it is used exactly
  // once — for the first real row set.
  pending_targets_.clear();
}

QTreeWidgetItem * JointsTab::rowFor(const QString & joint) const
{
  for (int i = 0; i < tree_->topLevelItemCount(); ++i) {
    auto * item = tree_->topLevelItem(i);
    if (item->text(COL_JOINT) == joint) {
      return item;
    }
  }
  return nullptr;
}

void JointsTab::onJointState(const sensor_msgs::msg::JointState::ConstSharedPtr msg)
{
  // Match BY NAME. The kortex sim publishes the finger joint FIRST, so
  // /joint_states order is not the cspace order and indexing into the tree by
  // position would silently show the wrong value on the wrong joint.
  QHash<QString, double> reading;
  for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
    reading.insert(QString::fromStdString(msg->name[i]), msg->position[i]);
  }

  runOnGuiThread(
    [this, reading]()
    {
      current_ = reading;
      for (int i = 0; i < tree_->topLevelItemCount(); ++i) {
        auto * item = tree_->topLevelItem(i);
        auto it = current_.constFind(item->text(COL_JOINT));
        if (it == current_.constEnd()) {
          continue;  // joint absent from this reading (e.g. arm-only publisher)
        }
        item->setText(
          COL_CURRENT, QString::number(it.value(), 'f', 4));
        item->setToolTip(
          COL_CURRENT, tr("live position from /joint_states: %1 rad")
                         .arg(QString::number(it.value(), 'f', 6)));
      }
    });
}

void JointsTab::setTargetsFromCurrent()
{
  int missing = 0;
  for (int i = 0; i < tree_->topLevelItemCount(); ++i) {
    auto * item = tree_->topLevelItem(i);
    auto it = current_.constFind(item->text(COL_JOINT));
    if (it == current_.constEnd()) {
      ++missing;
      continue;
    }
    setTarget(item, it.value());
  }
  setStatus(
    missing == 0 ? tr("targets set from the current /joint_states reading")
                 : tr("targets set from current (%1 joint(s) had no reading)").arg(missing));
}

void JointsTab::zeroTargets()
{
  for (int i = 0; i < tree_->topLevelItemCount(); ++i) {
    setTarget(tree_->topLevelItem(i), 0.0);
  }
  setStatus(tr("targets zeroed"));
}

namespace
{

/// Reads the target column into a vector in TREE (== cspace) order.
QVector<double> targetVector(QTreeWidget * tree)
{
  QVector<double> out;
  for (int i = 0; i < tree->topLevelItemCount(); ++i) {
    out << targetOf(tree->topLevelItem(i));
  }
  return out;
}

}  // namespace

void JointsTab::checkTarget()
{
  if (fk_client_ == nullptr || !fk_client_->service_is_ready()) {
    setStatus(tr("fk_batch service not available"));
    return;
  }
  auto request = std::make_shared<curobov2_ros_interfaces::srv::FkBatch::Request>();
  auto & config = request->joint_states.emplace_back();
  // FK resolves its input BY NAME, so the request is correct regardless of the
  // cspace order — which is exactly what makes this a usable pre-check.
  for (int i = 0; i < tree_->topLevelItemCount(); ++i) {
    auto * item = tree_->topLevelItem(i);
    config.name.push_back(item->text(COL_JOINT).toStdString());
    config.position.push_back(targetOf(item));
  }

  setStatus(tr("checking the target configuration..."));
  fk_client_->async_send_request(
    request,
    [this](rclcpp::Client<curobov2_ros_interfaces::srv::FkBatch>::SharedFuture future)
    {
      bool valid = false;
      QString error;
      try {
        auto result = future.get();
        valid = !result->poses_valid.empty() && result->poses_valid[0].data;
        error = QString::fromStdString(result->error_msg.data);
      } catch (const std::exception & e) {
        error = QString::fromStdString(e.what());
      }
      runOnGuiThread(
        [this, valid, error]()
        {
          if (error.isEmpty()) {
            setStatus(
              valid ? tr("target OK — collision-free and within joint limits")
                    : tr("target REJECTED — self-collision, scene collision, or a "
                         "joint outside its limits (the server prints the detail "
                         "to its console)"));
          } else {
            setStatus(tr("fk_batch failed: %1").arg(error));
          }
        });
    });
}

void JointsTab::clearLocks()
{
  if (joint_locks_client_ == nullptr || !joint_locks_client_->service_is_ready()) {
    setStatus(tr("set_joint_locks service not available"));
    return;
  }
  setStatus(
    tr("clearing joint locks — the server rebuilds its model and solvers "
       "(blocking, ~20 s). Watch its console; this tab stays responsive."));
  auto request = std::make_shared<curobov2_ros_interfaces::srv::SetJointLocks::Request>();
  // restore: true drops the whole override and re-parses the robot YAML's own
  // lock_joints; every other request field is ignored. An empty (non-restore)
  // request would be a read-only query instead, which is what refreshJointInfo
  // already gets for free.
  request->restore = true;
  joint_locks_client_->async_send_request(
    request,
    [this](
      rclcpp::Client<curobov2_ros_interfaces::srv::SetJointLocks>::SharedFuture future)
    {
      bool success = false;
      QString message;
      try {
        auto result = future.get();
        success = result->success;
        message = QString::fromStdString(result->message);
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread(
        [this, success, message]()
        {
          setStatus(success ? tr("joint locks updated: %1").arg(message)
                            : tr("could not clear joint locks: %1").arg(message));
          // The cspace just changed shape (a locked joint may have come back
          // into it), so re-read it rather than trusting the old row set.
          refreshJointInfo();
        });
    });
}

void JointsTab::planTarget()
{
  if (!cspace_known_) {
    setStatus(tr("cspace unknown — cannot build a joint-space goal yet"));
    return;
  }
  if (trajectory_client_ == nullptr || !trajectory_client_->service_is_ready()) {
    setStatus(tr("generate_trajectory service not available"));
    return;
  }

  auto request = std::make_shared<curobov2_ros_interfaces::srv::TrajectoryGeneration::Request>();
  request->request.goalsets.resize(1);
  auto & goalset = request->request.goalsets[0];
  // Joint-space segment: a non-empty target_joint_positions switches the
  // planner to plan_cspace. Values go in cspace order, which is the tree's row
  // order (get_joint_info reported it), so no re-sorting is needed.
  for (double v : targetVector(tree_)) {
    goalset.target_joint_positions.push_back(v);
  }

  setStatus(tr("planning to the target configuration..."));
  trajectory_client_->async_send_request(
    request,
    [this](
      rclcpp::Client<curobov2_ros_interfaces::srv::TrajectoryGeneration>::SharedFuture
        future)
    {
      bool success = false;
      QString message;
      int waypoints = 0;
      try {
        auto result = future.get();
        success = result->response.success;
        message = QString::fromStdString(result->response.message);
        waypoints = static_cast<int>(result->response.trajectory.size());
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread(
        [this, success, message, waypoints]()
        {
          setStatus(
            success ? tr("plan OK — %1 trajectory segment(s); the robot did NOT move")
                          .arg(waypoints)
                    : tr("plan FAILED: %1").arg(message));
        });
    });
}

void JointsTab::sendTarget()
{
  if (!cspace_known_) {
    setStatus(tr("cspace unknown — cannot build a joint-space goal yet"));
    return;
  }
  if (action_client_ == nullptr || !action_client_->action_server_is_ready()) {
    setStatus(tr("execute_trajectory action not available"));
    return;
  }

  curobov2_ros_interfaces::action::SendTrajectory::Goal goal;
  goal.goal.goalsets.resize(1);
  for (double v : targetVector(tree_)) {
    goal.goal.goalsets[0].target_joint_positions.push_back(v);
  }

  setStatus(tr("planning + executing to the target configuration..."));
  rclcpp_action::Client<curobov2_ros_interfaces::action::SendTrajectory>::SendGoalOptions
    options;
  options.goal_response_callback = [this](
    rclcpp_action::ClientGoalHandle<
      curobov2_ros_interfaces::action::SendTrajectory>::SharedPtr handle)
  { onGoalResponse(handle); };
  action_client_->async_send_goal(goal, options);
}

void JointsTab::onGoalResponse(
  rclcpp_action::ClientGoalHandle<
    curobov2_ros_interfaces::action::SendTrajectory>::SharedPtr handle)
{
  if (handle == nullptr) {
    runOnGuiThread([this]() { setStatus(tr("goal rejected (server busy?)")); });
    return;
  }
  goal_handle_ = handle;
  runOnGuiThread([this]() { setGoalActive(true); });

  auto result_cb = [this](
                     const rclcpp_action::ClientGoalHandle<
                       curobov2_ros_interfaces::action::SendTrajectory>::WrappedResult & r)
  { onGoalResult(r); };
  action_client_->async_get_result(handle, result_cb);
}

void JointsTab::onGoalResult(
  const rclcpp_action::ClientGoalHandle<
    curobov2_ros_interfaces::action::SendTrajectory>::WrappedResult & result)
{
  // Two hops, not one. WrappedResult::result is the action's Result message
  // (SendTrajectory_Result), which wraps the payload as a single field
  // `TrajectoryResult result` -- success/message live on THAT, not on the
  // wrapper. The service callers in this file read `result->success` directly
  // because those services return TrajectoryResult itself.
  bool success = false;
  QString message;
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      if (result.result != nullptr) {
        success = result.result->result.success;
        message = QString::fromStdString(result.result->result.message);
      } else {
        message = tr("succeeded but returned no result");
      }
      break;
    case rclcpp_action::ResultCode::CANCELED:
      message = tr("goal canceled");
      break;
    case rclcpp_action::ResultCode::ABORTED:
      if (result.result != nullptr) {
        message = tr("aborted: %1").arg(QString::fromStdString(result.result->result.message));
      } else {
        message = tr("aborted");
      }
      break;
    default:
      message = tr("goal failed");
      break;
  }
  runOnGuiThread(
    [this, success, message]()
    {
      setGoalActive(false);
      setStatus(success ? tr("target reached: %1").arg(message)
                        : tr("execution failed: %1").arg(message));
    });
}

void JointsTab::setGoalActive(bool active)
{
  goal_active_ = active;
  if (!active) {
    goal_handle_.reset();
  }
  setButtonsEnabled();
}

void JointsTab::stopGoal()
{
  if (goal_handle_ == nullptr || action_client_ == nullptr) {
    return;
  }
  setStatus(tr("canceling the active goal..."));
  action_client_->async_cancel_goal(goal_handle_);
}

void JointsTab::save(rviz_common::Config config) const
{
  // planner_node_name is deliberately NOT saved here. The Context tab owns the
  // planner node; this tab is handed it by CuroboPanel::syncPlannerNode(). It
  // used to persist its own copy, which meant every saved config carried the same
  // setting four times (Context + here + Manipulation + Scene Objects) and the
  // copies only stayed in agreement because syncPlannerNode() happened to run
  // after every load. Four copies of one setting is four chances for a config to
  // disagree with itself about which server it talks to.
  // Targets are persisted: an operator who set up a configuration, closed rviz
  // and reopened it expects the same target back, not a zeroed one.
  QStringList targets;
  for (int i = 0; i < tree_->topLevelItemCount(); ++i) {
    targets << QString::number(targetOf(tree_->topLevelItem(i)), 'f', 6);
  }
  config.mapSetValue("target_positions", targets.join(","));
}

void JointsTab::load(const rviz_common::Config & config)
{
  // No planner_node_name read here either -- see save(). CuroboPanel::load()
  // ends with syncPlannerNode(), which is the only thing that sets it.
  QString targets;
  if (config.mapGetString("target_positions", &targets)) {
    // PARKED, not applied: rviz calls load() before onInitialize(), and the rows
    // only exist after get_joint_info has answered, which needs a node. Writing
    // straight into the rows here would silently drop every saved value, so
    // rebuildRows() consumes this once the rows are built.
    pending_targets_.clear();
    for (const QString & v : targets.split(',', Qt::SkipEmptyParts)) {
      pending_targets_.push_back(v.toDouble());
    }
    if (tree_->topLevelItemCount() > 0) {
      // Rebuild from the CACHED limits. Passing empty vectors would strip the
      // bounds off every row -- the delegate would fall back to plain text and
      // lose the bar -- and because the next refresh does not see the cspace
      // change, it would never put the real limits back.
      rebuildRows(cspace_, limits_lower_, limits_upper_);
    }
  }
}

QStringList JointsTab::cspaceOrder() const
{
  return cspace_;
}

}  // namespace curobov2_ros_rviz
