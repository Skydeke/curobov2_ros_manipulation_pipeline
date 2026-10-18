#include <curobov2_ros_rviz/joints_tab.hpp>

#include <curobov2_ros_rviz/progress_bar_delegate.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <QAbstractItemView>
#include <QComboBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPointF>
#include <QPushButton>
#include <QStringList>
#include <QTimer>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>
#include <QXmlStreamReader>

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

/// A planned TrajectoryResult -> the execute action's replay form.
///
/// Execution is MTC-literal: the server drives exactly the plan it is handed
/// and never re-plans nor looks one up, so this tab re-packs the result it just
/// generated into the action's replay trajectory. The wire form of a result is
/// sensor_msgs/JointState[] + a scalar dt; the replay field wants
/// JointTrajectoryPoints with time_from_start.
trajectory_msgs::msg::JointTrajectory toReplayTrajectory(
  const curobov2_ros_interfaces::msg::TrajectoryResult & result)
{
  trajectory_msgs::msg::JointTrajectory traj;
  if (result.trajectory.empty()) {
    return traj;
  }
  traj.joint_names = result.trajectory.front().name;
  const double dt = result.dt;
  for (std::size_t i = 0; i < result.trajectory.size(); ++i) {
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions = result.trajectory[i].position;
    pt.velocities = result.trajectory[i].velocity;
    const double sec = dt * static_cast<double>(i);
    pt.time_from_start.sec = static_cast<int32_t>(sec);
    pt.time_from_start.nanosec = static_cast<uint32_t>((sec - std::floor(sec)) * 1e9);
    traj.points.push_back(pt);
  }
  return traj;
}

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

  // MoveIt's Joints tab itself has no named-pose control: its "Goal State"
  // combo lives on the Planning tab. curobo's named poses are JOINT
  // configurations, which is exactly what this tab's target column holds, so the
  // control is here: a combo that seeds the target column from the robot's SRDF
  // <group_state> entries (home, retract, vertical, open, close for the kortex).
  // Selecting one never plans or sends anything — it edits the request, exactly
  // like "Set from current" / Zero below, so Plan / Plan + send stay the only
  // things that can move the robot.
  auto * preset_row = new QHBoxLayout();
  preset_row->addWidget(new QLabel(tr("Named pose:"), this));
  named_pose_combo_ = new QComboBox(this);
  named_pose_combo_->setToolTip(
    tr("Named joint configurations from the robot's SRDF (its <group_state> "
       "entries). Selecting one writes its joints into the target column BY "
       "NAME; a joint the pose does not name falls back to the current "
       "/joint_states value, and a joint the cspace does not contain (e.g. one "
       "that is locked) is reported, not dropped silently. This only edits the "
       "target — use Plan / Plan + send to act on it."));
  named_pose_combo_->addItem(tr("— select a named pose —"));
  // Gated until an SRDF has been parsed (see loadNamedPoses): an empty combo is
  // a truthful "no named poses available", not a control that does nothing.
  named_pose_combo_->setEnabled(false);
  preset_row->addWidget(named_pose_combo_, 1);
  layout->addLayout(preset_row);

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

  connect(
    named_pose_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
    &JointsTab::onNamedPoseChosen);

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

  // Named poses come from the robot's SRDF, not from the planner. rviz.launch.py
  // passes `robot_description_semantic` as a node parameter even though no
  // MoveIt node runs, so the parameter is the primary source (loadNamedPoses);
  // the latched topic is the fallback for a setup that publishes it instead
  // (move_group's planning-scene monitor does). Both paths converge on
  // parseNamedPoses().
  rclcpp::QoS srdf_qos(rclcpp::KeepLast(1));
  srdf_qos.durability(rclcpp::DurabilityPolicy::TransientLocal);
  srdf_sub_ = node_->create_subscription<std_msgs::msg::String>(
    "/robot_description_semantic", srdf_qos,
    [this](std_msgs::msg::String::ConstSharedPtr msg)
    {
      if (named_poses_loaded_) {
        return;  // already parsed from the parameter
      }
      const QString xml = QString::fromStdString(msg->data);
      // Fires on rviz's spinner thread; parseNamedPoses touches the combo.
      runOnGuiThread([this, xml]() { parseNamedPoses(xml); });
    },
    sub_options);
  loadNamedPoses();

  // Poll the cspace: it is read-only and cheap (it reads the parsed kinematics
  // model, touches no solver), so a 2 s poll is cheap insurance against
  // missing the very first response while the planner is still warming up.
  // Visibility-gated: while the tab is hidden the timer still ticks but the
  // service call and the row rebuild are skipped, which is the whole point of
  // the poll -- there is nobody to see the rows.
  auto * info_timer = new QTimer(this);
  connect(info_timer, &QTimer::timeout, this, [this]()
  {
    if (isVisible()) {
      refreshJointInfo();
    }
  });
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
  // Retry the SRDF read here too: the parameter can be the only source and it
  // might not have been in place at initialize() time. Idempotent and cheap
  // (both early-return once poses are loaded).
  if (!named_poses_loaded_) {
    loadNamedPoses();
  }
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
  //
  // COALESCED, not one GUI update per message. This callback runs on rviz's
  // spinner thread and a joint-state publisher can run at ~100 Hz, while the
  // GUI thread has to repaint the whole tree per update; letting every message
  // queue its own rewrite makes the GUI fall behind and the displayed positions
  // lag. The readings are merged into pending_current_ under the mutex, and only
  // ONE drain is queued at a time (joint_state_posted_), so the GUI thread sees
  // the newest value of every joint and does a single repaint for a burst.
  {
    std::lock_guard<std::mutex> lock(joint_state_mutex_);
    for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
      pending_current_.insert(QString::fromStdString(msg->name[i]), msg->position[i]);
    }
    if (joint_state_posted_) {
      return;  // a drain is already queued; it will pick these up
    }
    joint_state_posted_ = true;
  }

  runOnGuiThread(
    [this]()
    {
      QHash<QString, double> reading;
      {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        reading.swap(pending_current_);
        joint_state_posted_ = false;
      }
      // The swap REPLACES current_ with everything seen since the last drain;
      // a joint absent from this burst keeps its previous value in current_ only
      // if it was also absent from the burst, which for a full-state publisher
      // never happens. Matches the old "current_ = reading" semantics for the
      // incremental case without dropping joints the burst did not mention.
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

void JointsTab::loadNamedPoses()
{
  if (named_poses_loaded_ || node_ == nullptr) {
    return;
  }

  // Source 1: the rviz node's `robot_description_semantic` parameter, which
  // rviz.launch.py sets from moveit_config.robot_description_semantic (so it is
  // present even though no MoveIt node is running). Launch parameters are
  // usually OVERRIDES rather than declared parameters, and which accessor sees
  // them differs (rclcpp exposes undeclared overrides through
  // get_parameter_overrides(); get_parameter() throws
  // ParameterUninitializedException). Try the override map first, then the
  // declared parameter, so both layouts work.
  QString srdf;
  try {
    const auto & overrides =
      node_->get_node_parameters_interface()->get_parameter_overrides();
    const auto it = overrides.find("robot_description_semantic");
    if (it != overrides.end() &&
        it->second.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
      srdf = QString::fromStdString(it->second.get<std::string>());
    }
  } catch (const std::exception &) {
    // No override map, or an unexpected type: fall through to the declared
    // parameter and then the topic.
  }
  if (srdf.isEmpty() && node_->has_parameter("robot_description_semantic")) {
    try {
      const auto param = node_->get_parameter("robot_description_semantic");
      if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
        srdf = QString::fromStdString(param.as_string());
      }
    } catch (const std::exception &) {
      // Uninitialised (override present but not declared): the topic fallback
      // is the remaining source.
    }
  }

  if (!srdf.isEmpty()) {
    parseNamedPoses(srdf);
  }
}

void JointsTab::parseNamedPoses(const QString & srdf_xml)
{
  if (named_poses_loaded_ || srdf_xml.isEmpty()) {
    return;
  }

  // <group_state name="home" group="manipulator"> <joint name="joint_1"
  // value="0"/> ... </group_state>. QXmlStreamReader is already linked in
  // through QtCore, so parsing the couple of attributes needed here needs no
  // new dependency (and no SRDF library).
  QVector<NamedPose> poses;
  QXmlStreamReader reader(srdf_xml);
  NamedPose current;
  bool in_pose = false;
  // .toString() rather than holding the raw QStringRef: Qt6 removed QStringRef
  // (name()/attributes().value() return QStringView there), so the version-
  // agnostic form is the one that compiles on both Qt5 and Qt6.
  while (!reader.atEnd()) {
    reader.readNext();
    const QString tag = reader.name().toString();
    if (reader.isStartElement()) {
      if (tag == QLatin1String("group_state")) {
        current = NamedPose();
        current.name = reader.attributes().value("name").toString();
        current.group = reader.attributes().value("group").toString();
        in_pose = !current.name.isEmpty();
      } else if (in_pose && tag == QLatin1String("joint")) {
        const QString joint = reader.attributes().value("name").toString();
        bool ok = false;
        const double value = reader.attributes().value("value").toString().toDouble(&ok);
        if (ok && !joint.isEmpty()) {
          current.joints.append(qMakePair(joint, value));
        }
      }
    } else if (reader.isEndElement() && tag == QLatin1String("group_state")) {
      // A state with no joints (or a group state for a group we cannot use) is
      // not a goal we could ever set, so it is not offered.
      if (in_pose && !current.joints.isEmpty()) {
        poses.append(current);
      }
      in_pose = false;
    }
  }
  if (reader.hasError() || poses.isEmpty()) {
    return;  // not an SRDF, or no usable named poses: stay disabled/empty
  }

  named_poses_ = poses;
  named_poses_loaded_ = true;

  named_pose_combo_->blockSignals(true);
  named_pose_combo_->clear();
  named_pose_combo_->addItem(tr("— select a named pose —"));
  for (const NamedPose & pose : named_poses_) {
    // The group is shown because two groups can share a state name in general
    // (the kortex ones do not, but nothing prevents it).
    named_pose_combo_->addItem(
      pose.group.isEmpty() ? pose.name
                           : tr("%1  [%2]").arg(pose.name, pose.group));
  }
  named_pose_combo_->setCurrentIndex(0);
  named_pose_combo_->blockSignals(false);
  named_pose_combo_->setEnabled(true);
}

void JointsTab::onNamedPoseChosen(int index)
{
  if (index <= 0) {
    return;  // the placeholder row, or a cleared combo
  }
  applyNamedPose(index - 1);
}

void JointsTab::applyNamedPose(int index)
{
  if (index < 0 || index >= named_poses_.size()) {
    return;
  }
  const NamedPose & pose = named_poses_[index];
  if (tree_->topLevelItemCount() == 0) {
    setStatus(
      tr("named pose '%1' needs the cspace — waiting for get_joint_info")
        .arg(pose.name));
    return;
  }

  // Index the pose by joint NAME: an SRDF group state names only the joints it
  // pins, so its joints are a SUBSET of the cspace, and the SRDF's order need
  // not match the cspace's.
  QHash<QString, double> pose_values;
  for (const QPair<QString, double> & joint : pose.joints) {
    pose_values.insert(joint.first, joint.second);
  }

  int from_pose = 0;
  int from_current = 0;
  QStringList no_reading;
  QStringList absent;
  // Walk the TREE (== cspace), not the pose, so every joint the target actually
  // holds is considered: a joint the pose does NOT name falls back to the live
  // /joint_states value rather than silently keeping whatever target was staged
  // before. That is what MoveIt does with a goal state's unnamed joints, and it
  // is what makes selecting "close" (a 1-joint gripper state) leave the arm
  // where it is instead of holding some stale arm target.
  for (int i = 0; i < tree_->topLevelItemCount(); ++i) {
    QTreeWidgetItem * item = tree_->topLevelItem(i);
    const QString joint = item->text(COL_JOINT);
    const auto pose_it = pose_values.constFind(joint);
    if (pose_it != pose_values.constEnd()) {
      setTarget(item, pose_it.value());
      ++from_pose;
      continue;
    }
    const auto current_it = current_.constFind(joint);
    if (current_it != current_.constEnd()) {
      setTarget(item, current_it.value());
      ++from_current;
    } else {
      no_reading << joint;
    }
  }
  // A pose joint with no row is pinned OUT of the cspace (e.g. locked):
  // reported, never dropped silently.
  for (const QPair<QString, double> & joint : pose.joints) {
    if (rowFor(joint.first) == nullptr) {
      absent << joint.first;
    }
  }

  QString message =
    tr("named pose '%1': %2 joint(s) from the pose, %3 from the current state")
      .arg(pose.name)
      .arg(from_pose)
      .arg(from_current);
  if (!no_reading.isEmpty()) {
    message += tr("; no reading, left unchanged: %1").arg(no_reading.join(", "));
  }
  if (!absent.isEmpty()) {
    message += tr("; not in the cspace (locked?): %1").arg(absent.join(", "));
  }
  setStatus(message);
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
  if (trajectory_client_ == nullptr || !trajectory_client_->service_is_ready()) {
    setStatus(tr("generate_trajectory service not available"));
    return;
  }

  // Execution is MTC-literal: the server neither re-plans nor reuses a cached
  // plan, so plan FIRST and replay the result verbatim. (This is the open-loop
  // path; a reactive planner has no waypoints to stream and the server will
  // refuse the replay.)
  auto request =
    std::make_shared<curobov2_ros_interfaces::srv::TrajectoryGeneration::Request>();
  request->request.goalsets.resize(1);
  for (double v : targetVector(tree_)) {
    request->request.goalsets[0].target_joint_positions.push_back(v);
  }

  setStatus(tr("planning + executing to the target configuration..."));
  trajectory_client_->async_send_request(
    request,
    [this](
      rclcpp::Client<curobov2_ros_interfaces::srv::TrajectoryGeneration>::
        SharedFuture future)
    {
      bool success = false;
      QString message;
      trajectory_msgs::msg::JointTrajectory replay;
      try {
        auto result = future.get();
        success = result->response.success;
        message = QString::fromStdString(result->response.message);
        if (success) {
          replay = toReplayTrajectory(result->response);
        }
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      if (!success) {
        runOnGuiThread([this, message]() {
          setStatus(tr("execute plan FAILED: %1").arg(message));
        });
        return;
      }
      runOnGuiThread([this, replay]() {
        // The plan landed; keep it and drive exactly it.
        last_trajectory_ = replay;
        curobov2_ros_interfaces::action::SendTrajectory::Goal goal;
        goal.goal.goalsets.resize(1);
        for (double v : targetVector(tree_)) {
          goal.goal.goalsets[0].target_joint_positions.push_back(v);
        }
        goal.replay = true;
        goal.replay_trajectory = last_trajectory_;

        rclcpp_action::Client<
          curobov2_ros_interfaces::action::SendTrajectory>::SendGoalOptions options;
        options.goal_response_callback = [this](
          rclcpp_action::ClientGoalHandle<
            curobov2_ros_interfaces::action::SendTrajectory>::SharedPtr handle)
        { onGoalResponse(handle); };
        action_client_->async_send_goal(goal, options);
      });
    });
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
