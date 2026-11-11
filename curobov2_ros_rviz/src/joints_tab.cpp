#include <curobov2_ros_rviz/joints_tab.hpp>

#include <curobov2_ros_rviz/progress_bar_delegate.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPointF>
#include <QPushButton>
#include <QSet>
#include <QShowEvent>
#include <QStringList>
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
///
/// A CONTINUOUS joint is periodic. Its reported bounds are a DISPLAY WINDOW,
/// not a physical stop, so the target must be the user's raw intent: rewriting
/// 6.3641 to 0.0809 (or clamping it to 6.28) changes the value the planner is
/// asked for. The planner interprets the target as a pose and takes the
/// shortest 2*pi-equivalent path from the measured branch; multi-turn motion
/// needs explicit waypoints. Bounded joints keep the ordinary clamp.
void setTarget(QTreeWidgetItem * item, double value)
{
  if (hasBounds(item)) {
    const QPointF limits = boundsOf(item);
    const double lo = limits.x();
    const double hi = limits.y();
    const bool continuous =
      item->data(COL_TARGET, ProgressBarDelegate::ContinuousRole).toBool();
    if (!continuous) {
      value = std::clamp(value, lo, hi);
    }
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
  // control is here: a combo seeded from the robot descriptor's
  // `named_joint_configs` (via get_joint_info) plus one open/close entry per
  // `gripper_controllers` entry. The SRDF is deliberately not used: the
  // descriptor is the planner's own source of truth, and a gripper's jaws are
  // usually locked out of the cspace, so they cannot be planned at all.
  // Selecting an arm pose edits the request exactly like "Set from current" /
  // Zero below; selecting a gripper entry commands the controller directly.
  auto * preset_row = new QHBoxLayout();
  preset_row->addWidget(new QLabel(tr("Named pose:"), this));
  named_pose_combo_ = new QComboBox(this);
  named_pose_combo_->setToolTip(
    tr("Named joint configurations from the robot descriptor's "
       "named_joint_configs. Selecting an arm pose writes its joints into the "
       "target column BY NAME; a joint the pose does not name falls back to the "
       "current /joint_states value, and a joint the cspace does not contain "
       "(e.g. one that is locked) is reported, not dropped silently. The "
       "open/close entries come from the descriptor's gripper_controllers and "
       "are sent straight to the gripper action, because the jaws are not "
       "planning joints — use Plan / Plan + send for arm poses."));
  named_pose_combo_->addItem(tr("— select a named pose —"));
  // Gated until get_joint_info has answered (see setNamedPoses): an empty combo
  // is a truthful "no named poses available", not a control that does nothing.
  named_pose_combo_->setEnabled(false);
  preset_row->addWidget(named_pose_combo_, 1);
  // "Preview": mirror the target column onto <planner_node>/trajectory (a
  // one-waypoint JointTrajectory) whenever it changes, so the
  // CuroboPlanningDisplay shows the configuration live as the bars are
  // dragged. Checked by default; unchecking makes the tab edit-only again.
  preview_check_ = new QCheckBox(tr("Preview"), this);
  preview_check_->setToolTip(
    tr("Mirror the target configuration onto the planner's trajectory topic "
       "(<planner_node>/trajectory) as a single-waypoint JointTrajectory, so the "
       "CuroboPlanningDisplay shows it while you move the bars. This only "
       "changes what is DRAWN - it never plans or moves the robot."));
  preview_check_->setChecked(true);
  preset_row->addWidget(preview_check_);
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

  // Preview poll: rather than hooking every write path (bar drags write the
  // model directly through the delegate), poll the target column and publish it
  // when it changed (see publishPreview). 100 ms is below the drag's own repaint
  // cadence, so the preview tracks the bar.
  auto * preview_timer = new QTimer(this);
  connect(preview_timer, &QTimer::timeout, this, &JointsTab::publishPreview);
  preview_timer->start(100);

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

void JointsTab::showEvent(QShowEvent * event)
{
  QWidget::showEvent(event);
  // The 2 s cspace poll is visibility-gated, and the initial fetch in
  // initialize() can run before the planner's get_joint_info service is up.
  // Fetch immediately when the tab is first shown so the Named pose combo does
  // not stay empty/disabled until the next poll tick.
  if (initialized_) {
    refreshJointInfo();
  }
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
  trajectory_client_ =
    node_->create_client<curobov2_ros_interfaces::srv::TrajectoryGeneration>(
      ns + "generate_trajectory");
  action_client_ = rclcpp_action::create_client<curobov2_ros_interfaces::action::SendTrajectory>(
    node_, ns + "execute_trajectory");

  // Preview/ghost topic: CuroboPlanningDisplay subscribes to
  // <planner_node>/trajectory. The server publishes the planned ghost there; the
  // tab ALSO mirrors the edited target (a one-waypoint JointTrajectory) so
  // moving the joint bars displays the configuration without planning.
  preview_pub_ = node_->create_publisher<trajectory_msgs::msg::JointTrajectory>(
    ns + "trajectory", 10);
  last_preview_target_.clear();

  // A different planner is a different robot: the old cspace (and the target
  // positions indexed by it) mean nothing here.
  cspace_.clear();
  current_.clear();
  locked_.clear();
  cspace_known_ = false;
  tree_->clear();
  // A different planner is a different robot: its named poses and gripper
  // actions mean nothing here either.
  named_poses_.clear();
  gripper_clients_.clear();
  parallel_gripper_clients_.clear();
  gripper_trajectory_clients_.clear();
  named_pose_combo_->blockSignals(true);
  named_pose_combo_->clear();
  named_pose_combo_->addItem(tr("— select a named pose —"));
  named_pose_combo_->setCurrentIndex(0);
  named_pose_combo_->blockSignals(false);
  named_pose_combo_->setEnabled(false);
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
      QVector<bool> continuous;
      QStringList locked;
      QVector<NamedPose> poses;
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
        for (bool v : result->continuous) {
          continuous << v;
        }
        for (const auto & n : result->locked_joint_names) {
          locked << QString::fromStdString(n);
        }

        // Every joint any descriptor gripper owns. A descriptor may carry its
        // own `open`/`close` entries in named_joint_configs (they are what the
        // task constructor resolves by name), but those joints are not planning
        // DOFs. The synthesized gripper entries below already expose the same
        // action with the controller type/effort/timeout, so a named config
        // that moves ONLY gripper joints is dropped here: one `open`/`close`
        // per gripper in the combo, and it is the one that can actually drive
        // the controller.
        QSet<QString> gripper_owned;
        for (const auto & gripper : result->grippers) {
          for (const auto & joint : gripper.joints) {
            gripper_owned.insert(QString::fromStdString(joint));
          }
          if (!gripper.command_joint.empty()) {
            gripper_owned.insert(QString::fromStdString(gripper.command_joint));
          }
        }

        // Arm poses: the descriptor's named_joint_configs, already grouped by
        // the server (a pose names its joints; it need not cover the cspace).
        for (const auto & cfg : result->named_joint_configs) {
          NamedPose pose;
          pose.name = QString::fromStdString(cfg.name);
          for (std::size_t i = 0;
            i < cfg.joint_names.size() && i < cfg.positions.size(); ++i)
          {
            pose.joints.append(
              qMakePair(
                QString::fromStdString(cfg.joint_names[i]),
                cfg.positions[i]));
          }
          bool only_gripper_joints = !pose.joints.isEmpty();
          for (const auto & joint : pose.joints) {
            if (!gripper_owned.contains(joint.first)) {
              only_gripper_joints = false;
              break;
            }
          }
          if (!pose.name.isEmpty() && !pose.joints.isEmpty() &&
            !only_gripper_joints)
          {
            poses.append(pose);
          }
        }

        // Gripper entries: one open + one close per descriptor controller. They
        // are not planning targets, so they carry the action to call instead.
        for (const auto & gripper : result->grippers) {
          const QString name = QString::fromStdString(gripper.name);
          const QString type = QString::fromStdString(gripper.type);
          const QString controller = QString::fromStdString(gripper.controller);
          const QString action_ns = QString::fromStdString(gripper.action_ns);
          QStringList joints;
          for (const auto & joint : gripper.joints) {
            joints << QString::fromStdString(joint);
          }
          const bool trajectory =
            type == QLatin1String("JointTrajectory") ||
            type == QLatin1String("FollowJointTrajectory");
          QString action;
          if (trajectory) {
            action = controller.isEmpty()
              ? QStringLiteral("follow_joint_trajectory")
              : QStringLiteral("/%1/follow_joint_trajectory").arg(controller);
          } else {
            const QString ns = action_ns.isEmpty()
              ? QStringLiteral("gripper_cmd") : action_ns;
            action = controller.isEmpty()
              ? ns : QStringLiteral("/%1/%2").arg(controller, ns);
          }
          const QString command_joint = QString::fromStdString(gripper.command_joint);
          const QString single_joint = command_joint.isEmpty()
            ? (joints.isEmpty() ? QString() : joints.first()) : command_joint;

          for (int open = 1; open >= 0; --open) {
            NamedPose pose;
            pose.name = open != 0 ? tr("open") : tr("close");
            pose.group = name;
            pose.gripper = true;
            pose.gripper_type = type;
            pose.gripper_action = action;
            pose.gripper_command_joint = single_joint;
            pose.gripper_max_effort = gripper.max_effort;
            pose.gripper_time_from_start =
              gripper.time_from_start > 0.0 ? gripper.time_from_start : 2.0;
            const double value = open != 0 ? gripper.open : gripper.close;
            if (trajectory || type == QLatin1String("ParallelGripperCommand")) {
              pose.gripper_joints = joints;
              for (const QString & joint : joints) {
                pose.joints.append(qMakePair(joint, value));
              }
            } else if (!single_joint.isEmpty()) {
              pose.gripper_joints.clear();
              pose.gripper_joints << single_joint;
              pose.joints.append(qMakePair(single_joint, value));
            }
            if (!pose.gripper_action.isEmpty() && !pose.joints.isEmpty()) {
              poses.append(pose);
            }
          }
        }
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread(
        [this, names, lower, upper, continuous, locked, poses, success, message]()
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
          continuous_ = continuous;
          cspace_known_ = !cspace_.isEmpty();
          if (changed || tree_->topLevelItemCount() == 0) {
            rebuildRows(cspace_, limits_lower_, limits_upper_, continuous_);
          }
          setNamedPoses(poses);
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
  const QStringList & names, const QVector<double> & lower, const QVector<double> & upper,
  const QVector<bool> & continuous)
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
    const bool is_continuous = i < continuous.size() && continuous[i];
    item->setData(COL_TARGET, ProgressBarDelegate::ContinuousRole, is_continuous);
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
      !hasBounds(item)
        ? tr("%1 — the model reports no position limit for this joint, so its "
             "target has to be typed rather than dragged.").arg(names[i])
        : is_continuous
        ? tr("%1 — continuous joint: drag the bar to set a target. The %2 .. %3 "
             "rad range is a display window, not a stop; the planner takes the "
             "shortest 2*pi-equivalent path from the measured state.")
            .arg(names[i]).arg(lo).arg(hi)
        : tr("%1 — drag the bar to set the target, or click once to place it. "
             "Limits: %2 .. %3 rad.")
            .arg(names[i]).arg(lo).arg(hi));

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

void JointsTab::setNamedPoses(const QVector<NamedPose> & poses)
{
  if (poses == named_poses_) {
    return;  // the 2 s poll usually reports the same descriptor data
  }

  // Preserve the user's selection across a poll that changed the list for an
  // unrelated reason (a joint-lock rebuild, a re-connecting planner). Match by
  // name AND group: "open" is deliberately shared by every gripper.
  QString selected_name;
  QString selected_group;
  const int selected = named_pose_combo_->currentIndex();
  if (selected > 0 && selected - 1 < named_poses_.size()) {
    selected_name = named_poses_[selected - 1].name;
    selected_group = named_poses_[selected - 1].group;
  }

  named_poses_ = poses;

  named_pose_combo_->blockSignals(true);
  named_pose_combo_->clear();
  named_pose_combo_->addItem(tr("— select a named pose —"));
  int restore = 0;
  for (int i = 0; i < named_poses_.size(); ++i) {
    const NamedPose & pose = named_poses_[i];
    // The group is shown because two groups can share a state name (every
    // gripper offers "open"/"close"), and the arm poses' group is empty.
    named_pose_combo_->addItem(
      pose.group.isEmpty() ? pose.name
                           : tr("%1  [%2]").arg(pose.name, pose.group));
    if (pose.name == selected_name && pose.group == selected_group) {
      restore = i + 1;
    }
  }
  named_pose_combo_->setCurrentIndex(restore);
  named_pose_combo_->blockSignals(false);
  named_pose_combo_->setEnabled(!named_poses_.isEmpty());
}

void JointsTab::onNamedPoseChosen(int index)
{
  if (index <= 0) {
    return;  // the placeholder row, or a cleared combo
  }
  const bool gripper =
    (index - 1) < named_poses_.size() && named_poses_[index - 1].gripper;
  applyNamedPose(index - 1);
  if (gripper) {
    // A gripper command is an action, not a staged target: return the combo to
    // the placeholder so choosing the SAME open/close again still emits a
    // signal (currentIndexChanged only fires when the index changes).
    named_pose_combo_->blockSignals(true);
    named_pose_combo_->setCurrentIndex(0);
    named_pose_combo_->blockSignals(false);
  }
}

void JointsTab::applyNamedPose(int index)
{
  if (index < 0 || index >= named_poses_.size()) {
    return;
  }
  const NamedPose & pose = named_poses_[index];

  // A gripper's open/close is a controller command, not a planning goal. Send
  // it even when the finger joints are locked out of the cspace (the normal
  // case): there is no target row to edit, and planning could never reach it.
  if (pose.gripper) {
    sendGripperCommand(pose);
    return;
  }

  if (tree_->topLevelItemCount() == 0) {
    setStatus(
      tr("named pose '%1' needs the cspace — waiting for get_joint_info")
        .arg(pose.name));
    return;
  }

  // Index the pose by joint NAME: a descriptor pose names only the joints it
  // pins, so its joints are a SUBSET of the cspace, and the descriptor's order
  // need not match the cspace's.
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
  // is what makes a partial descriptor pose leave the rest of the arm where it
  // is instead of holding some stale arm target.
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

void JointsTab::sendGripperCommand(const NamedPose & pose)
{
  if (node_ == nullptr || pose.gripper_action.isEmpty() ||
    pose.joints.isEmpty())
  {
    setStatus(tr("gripper '%1' has no action or position in the descriptor")
                .arg(pose.group));
    return;
  }

  const std::string action = pose.gripper_action.toStdString();
  const QString label = pose.group.isEmpty()
    ? pose.name : tr("%1 [%2]").arg(pose.name, pose.group);

  const bool trajectory =
    pose.gripper_type == QLatin1String("JointTrajectory") ||
    pose.gripper_type == QLatin1String("FollowJointTrajectory");

  if (trajectory) {
    auto & client = gripper_trajectory_clients_[pose.gripper_action];
    if (!client) {
      client =
        rclcpp_action::create_client<control_msgs::action::FollowJointTrajectory>(
        node_, action);
    }
    if (!client->action_server_is_ready()) {
      setStatus(tr("gripper action %1 unavailable").arg(pose.gripper_action));
      return;
    }
    control_msgs::action::FollowJointTrajectory::Goal goal;
    for (const QPair<QString, double> & joint : pose.joints) {
      goal.trajectory.joint_names.push_back(joint.first.toStdString());
    }
    trajectory_msgs::msg::JointTrajectoryPoint point;
    for (const QPair<QString, double> & joint : pose.joints) {
      point.positions.push_back(joint.second);
      point.velocities.push_back(0.0);
    }
    const double duration = pose.gripper_time_from_start;
    point.time_from_start.sec = static_cast<int32_t>(duration);
    point.time_from_start.nanosec =
      static_cast<uint32_t>((duration - std::floor(duration)) * 1e9);
    goal.trajectory.points.push_back(point);

    rclcpp_action::Client<
      control_msgs::action::FollowJointTrajectory>::SendGoalOptions options;
    options.goal_response_callback =
      [this, label](rclcpp_action::ClientGoalHandle<
          control_msgs::action::FollowJointTrajectory>::SharedPtr handle)
      {
        const bool accepted = handle != nullptr;
        runOnGuiThread([this, label, accepted]() {
          if (!accepted) {
            setStatus(tr("%1: gripper goal rejected").arg(label));
          }
        });
      };
    options.result_callback =
      [this, label](const rclcpp_action::ClientGoalHandle<
          control_msgs::action::FollowJointTrajectory>::WrappedResult & result)
      {
        const bool ok = result.code == rclcpp_action::ResultCode::SUCCEEDED;
        runOnGuiThread([this, label, ok]() {
          setStatus(ok ? tr("%1: gripper command done").arg(label)
                       : tr("%1: gripper command failed").arg(label));
        });
      };
    client->async_send_goal(goal, options);
    setStatus(tr("%1: sent to %2").arg(label, pose.gripper_action));
    return;
  }

  if (pose.gripper_type == QLatin1String("ParallelGripperCommand")) {
    auto & client = parallel_gripper_clients_[pose.gripper_action];
    if (!client) {
      client = rclcpp_action::create_client<
        control_msgs::action::ParallelGripperCommand>(node_, action);
    }
    if (!client->action_server_is_ready()) {
      setStatus(tr("gripper action %1 unavailable").arg(pose.gripper_action));
      return;
    }
    control_msgs::action::ParallelGripperCommand::Goal goal;
    for (const QPair<QString, double> & joint : pose.joints) {
      goal.command.name.push_back(joint.first.toStdString());
      goal.command.position.push_back(joint.second);
      goal.command.effort.push_back(pose.gripper_max_effort);
    }
    rclcpp_action::Client<
      control_msgs::action::ParallelGripperCommand>::SendGoalOptions options;
    options.goal_response_callback =
      [this, label](rclcpp_action::ClientGoalHandle<
          control_msgs::action::ParallelGripperCommand>::SharedPtr handle)
      {
        const bool accepted = handle != nullptr;
        runOnGuiThread([this, label, accepted]() {
          if (!accepted) {
            setStatus(tr("%1: gripper goal rejected").arg(label));
          }
        });
      };
    options.result_callback =
      [this, label](const rclcpp_action::ClientGoalHandle<
          control_msgs::action::ParallelGripperCommand>::WrappedResult & result)
      {
        const bool ok = result.code == rclcpp_action::ResultCode::SUCCEEDED;
        runOnGuiThread([this, label, ok]() {
          setStatus(ok ? tr("%1: gripper command done").arg(label)
                       : tr("%1: gripper command failed").arg(label));
        });
      };
    client->async_send_goal(goal, options);
    setStatus(tr("%1: sent to %2").arg(label, pose.gripper_action));
    return;
  }

  auto & client = gripper_clients_[pose.gripper_action];
  if (!client) {
    client = rclcpp_action::create_client<control_msgs::action::GripperCommand>(
      node_, action);
  }
  if (!client->action_server_is_ready()) {
    setStatus(tr("gripper action %1 unavailable").arg(pose.gripper_action));
    return;
  }
  control_msgs::action::GripperCommand::Goal goal;
  goal.command.position = pose.joints.first().second;
  goal.command.max_effort = pose.gripper_max_effort;
  rclcpp_action::Client<control_msgs::action::GripperCommand>::SendGoalOptions options;
  options.goal_response_callback =
    [this, label](rclcpp_action::ClientGoalHandle<
        control_msgs::action::GripperCommand>::SharedPtr handle)
    {
      const bool accepted = handle != nullptr;
      runOnGuiThread([this, label, accepted]() {
        if (!accepted) {
          setStatus(tr("%1: gripper goal rejected").arg(label));
        }
      });
    };
  options.result_callback =
    [this, label](const rclcpp_action::ClientGoalHandle<
        control_msgs::action::GripperCommand>::WrappedResult & result)
    {
      const bool ok = result.code == rclcpp_action::ResultCode::SUCCEEDED;
      runOnGuiThread([this, label, ok]() {
        setStatus(ok ? tr("%1: gripper command done").arg(label)
                     : tr("%1: gripper command failed").arg(label));
      });
    };
  client->async_send_goal(goal, options);
  setStatus(tr("%1: sent to %2").arg(label, pose.gripper_action));
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

// Mirror the target column onto <planner_node>/trajectory as a single-waypoint
// JointTrajectory so the CuroboPlanningDisplay shows the configuration the
// user is editing - no plan, no motion. Gated on the "Preview" checkbox; only
// publishes when the target actually changed (the caller polls).
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
      rebuildRows(cspace_, limits_lower_, limits_upper_, continuous_);
    }
  }
}

QStringList JointsTab::cspaceOrder() const
{
  return cspace_;
}

}  // namespace curobov2_ros_rviz
