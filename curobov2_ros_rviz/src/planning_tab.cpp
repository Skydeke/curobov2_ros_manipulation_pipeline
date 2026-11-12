#include "curobov2_ros_rviz/planning_tab.hpp"
#include "curobov2_ros_interfaces/msg/goalset.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// Only startMpc() still talks to SetPlanner: it re-asserts MPC on the server
#include "curobov2_ros_interfaces/srv/set_planner.hpp"
// Reached transitively via send_trajectory.hpp -> TrajectoryResult, but named
// explicitly because result_callback reads its fields directly.
#include "curobov2_ros_interfaces/msg/planning_stats.hpp"
// Named explicitly for toReplayTrajectory() below (same reasoning: reached
// transitively via trajectory_generation.hpp, but read field-by-field here).
#include "curobov2_ros_interfaces/msg/trajectory_result.hpp"
#include <QApplication>
#include <QListWidget>
#include <QSignalBlocker>
#include <QThread>
#include <QTimer>
#include <cmath>

namespace {

// A planned TrajectoryResult -> the execute action's replay form.
trajectory_msgs::msg::JointTrajectory toReplayTrajectory(
    const curobov2_ros_interfaces::msg::TrajectoryResult &result) {
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
    pt.time_from_start.nanosec =
        static_cast<uint32_t>((sec - std::floor(sec)) * 1e9);
    traj.points.push_back(pt);
  }
  return traj;
}

}  // namespace

namespace curobov2_ros_rviz {
PlanningTab::PlanningTab(QWidget *parent)
    : Panel{parent}, ui_(std::make_unique<Ui::PlanningTabForm>()),
      node_{nullptr}, param_client_{nullptr}, planner_ready_{false},
      planner_poll_in_flight_{false}, planner_poll_seq_{0}, goal_active_{false},
      action_ptr_{nullptr}, trajectory_generation_client_{nullptr},
      goal_handle_{nullptr}, time_dilation_factor_{1.0},
      target_display_{nullptr},
      target_tool_list_{nullptr}, user_editing_pose_{false},
      last_displayed_x_{std::numeric_limits<double>::quiet_NaN()},
      last_displayed_y_{std::numeric_limits<double>::quiet_NaN()},
      last_displayed_z_{std::numeric_limits<double>::quiet_NaN()},
      planner_node_{"unified_planner"}, mpc_goal_pub_{nullptr},
      mpc_active_{false}, mpc_starting_{false}, mpc_goal_timer_{nullptr},
      readiness_timer_{nullptr}
      // Init list order must match declaration order (trajectory_type_ is
      // declared above set_planner_client_ in the header) or -Wreorder fires.
      ,
      trajectory_type_{0}, set_planner_client_{nullptr} {
  // Extend the widget with all attributes and children from UI file
  ui_->setupUi(this);

  target_tool_list_ = new QListWidget(this);
  target_tool_list_->setToolTip(
      tr("Checked tool frames are planned together: every checked target is "
         "sent in the same goalset. Selecting a row chooses the tool edited by "
         "the pose fields and marker drags."));
  target_tool_list_->setSelectionMode(QAbstractItemView::SingleSelection);
  target_tool_list_->setMaximumHeight(90);
  ui_->queryFormLayout->insertRow(0, tr("Target tools:"), target_tool_list_);
  connect(target_tool_list_, &QListWidget::itemChanged,
          this, &PlanningTab::onTargetToolChecked);
  connect(target_tool_list_, &QListWidget::currentItemChanged,
          this, &PlanningTab::onTargetToolSelected);

  // `last_displayed_q_` cannot be seeded in the init list above: the generated
  last_displayed_q_.x = 0.0;
  last_displayed_q_.y = 0.0;
  last_displayed_q_.z = 0.0;
  last_displayed_q_.w = std::numeric_limits<double>::quiet_NaN();

  // Init rclcpp node
  auto options = rclcpp::NodeOptions().arguments(
      {"--ros-args", "--remap", "__node:=rviz_updata_parameters_node", "--"});
  node_ = std::make_shared<rclcpp::Node>("_", options);

  // Own TF buffer for the one-time tool-pose seed lookup (frame names travel
  seed_tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  seed_tf_listener_ =
    std::make_shared<tf2_ros::TransformListener>(*seed_tf_buffer_, node_, false);

  // Bootstrap value only. The planner node is owned by the Context tab, which
  planner_node_ = "unified_planner";

  // Show the bootstrap node until the Context tab says otherwise. Nothing is
  // built from a dropdown here any more.
  ui_->plannerNodeEcho->setText(QString::fromStdString(planner_node_));
  createPlannerClients(planner_node_);

  // Timer streaming the live MPC goal while the gizmo is dragged (10 Hz).
  mpc_goal_timer_ = new QTimer(this);
  mpc_goal_timer_->setInterval(100);
  connect(mpc_goal_timer_, &QTimer::timeout, this, &PlanningTab::streamMpcGoal);

  // Connect SpinBox for time dilation to its slot
  connect(ui_->doubleSpinBoxTimeDilationFactor, SIGNAL(valueChanged(double)),
          this, SLOT(updateTimeDilationFactor(double)));

  // Every planner-facing widget starts disabled and stays that way until
  setPlannerReady(false);

  readiness_timer_ = new QTimer(this);
  connect(readiness_timer_, &QTimer::timeout, this,
          &PlanningTab::pollPlannerReady);
  readiness_timer_->start(250);

  connect(ui_->spinBoxPosX,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          [this](double) { applyPoseFromSpinboxes(); });
  connect(ui_->spinBoxPosY,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          [this](double) { applyPoseFromSpinboxes(); });
  connect(ui_->spinBoxPosZ,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          [this](double) { applyPoseFromSpinboxes(); });

  // Connect orientation spinboxes. Four of them, because the goal orientation
  // is a quaternion -- see the note above applyPoseFromSpinboxes().
  connect(ui_->spinBoxQx,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          [this](double) { applyPoseFromSpinboxes(); });
  connect(ui_->spinBoxQy,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          [this](double) { applyPoseFromSpinboxes(); });
  connect(ui_->spinBoxQz,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          [this](double) { applyPoseFromSpinboxes(); });
  connect(ui_->spinBoxQw, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
          this, [this](double) { applyPoseFromSpinboxes(); });

  // Detect when spinbox gets focus (user starts editing) to pause auto-update
  connect(ui_->spinBoxPosX, &QDoubleSpinBox::editingFinished, this,
          [this]() { user_editing_pose_ = false; });
  connect(ui_->spinBoxPosY, &QDoubleSpinBox::editingFinished, this,
          [this]() { user_editing_pose_ = false; });
  connect(ui_->spinBoxPosZ, &QDoubleSpinBox::editingFinished, this,
          [this]() { user_editing_pose_ = false; });
  connect(ui_->spinBoxQx, &QDoubleSpinBox::editingFinished, this,
          [this]() { user_editing_pose_ = false; });
  connect(ui_->spinBoxQy, &QDoubleSpinBox::editingFinished, this,
          [this]() { user_editing_pose_ = false; });
  connect(ui_->spinBoxQz, &QDoubleSpinBox::editingFinished, this,
          [this]() { user_editing_pose_ = false; });
  connect(ui_->spinBoxQw, &QDoubleSpinBox::editingFinished, this,
          [this]() { user_editing_pose_ = false; });

  ui_->spinBoxPosX->installEventFilter(this);
  ui_->spinBoxPosY->installEventFilter(this);
  ui_->spinBoxPosZ->installEventFilter(this);
  ui_->spinBoxQx->installEventFilter(this);
  ui_->spinBoxQy->installEventFilter(this);
  ui_->spinBoxQz->installEventFilter(this);
  ui_->spinBoxQw->installEventFilter(this);

  // Timer to update marker pose display
  QTimer *poseUpdateTimer = new QTimer(this);
  connect(poseUpdateTimer, &QTimer::timeout, this,
          &PlanningTab::updateMarkerPoseDisplay);
  poseUpdateTimer->start(100); // Update pose display every 100ms

  // Timer to find the CuroboPlanningDisplay(s) (added to RViz as displays, not owned
  // by this panel). Keeps running so displays added/removed/reordered later
  // are picked up (the primary display feeds the goal).
  QTimer *findDisplayTimer = new QTimer(this);
  connect(findDisplayTimer, &QTimer::timeout, this,
          &PlanningTab::refreshPlanningDisplays);
  findDisplayTimer->start(500); // Check every 500ms

  // Connect planner node / planner type controls

  // Connect stop robot button
  connect(ui_->stopRobot, &QPushButton::clicked, this,
          &PlanningTab::on_stopRobot_clicked);
  connect(ui_->clearGoal, &QPushButton::clicked, this,
          &PlanningTab::on_clearGoal_clicked);

  // Spin node_ on a background thread for the panel's lifetime.
  spinner_ = std::make_unique<NodeSpinner>(node_);
}

PlanningTab::~PlanningTab() {
  // spinner_ is declared last in the header, so it is destroyed FIRST here --
}

void PlanningTab::createPlannerClients(const std::string &planner_node) {
  const std::string planner_ns = "/" + planner_node + "/";

  // AsyncParametersClient -- never SyncParametersClient, which builds its own
  // temporary executor per call and would fight the background NodeSpinner (see
  // spinner_ below) for ownership of node_.
  param_client_ =
      std::make_shared<rclcpp::AsyncParametersClient>(node_, planner_node);

  // action client
  this->action_ptr_ = rclcpp_action::create_client<
      curobov2_ros_interfaces::action::SendTrajectory>(
      node_, planner_ns + "execute_trajectory");

  // create service client to generate traj
  this->trajectory_generation_client_ = node_->create_client<
      curobov2_ros_interfaces::srv::TrajectoryGeneration>(
      planner_ns + "generate_trajectory");

  // Create service client for setting planner type
  this->set_planner_client_ =
      node_->create_client<curobov2_ros_interfaces::srv::SetPlanner>(
          planner_ns + "set_planner");

  // Publisher streaming the live MPC goal towards the current target pose
  // (stamped: the pose rides in the RViz fixed frame, see goalFrame()).
  this->mpc_goal_pub_ = node_->create_publisher<curobov2_ros_interfaces::msg::MpcGoal>(
      planner_ns + "mpc_goal", 10);

  // Live tool pose (base frame) for one-shot seeding of pristine displays.
  {
    std::lock_guard<std::mutex> lock(tool_pose_mutex_);
    tool_pose_.reset();
  }
  seeded_display_ = nullptr;
  this->tool_pose_sub_ =
      node_->create_subscription<geometry_msgs::msg::PoseStamped>(
          planner_ns + "current_tool_pose", 10,
          [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(tool_pose_mutex_);
            tool_pose_ = msg;
          });

  RCLCPP_INFO(node_->get_logger(),
              "PlanningTab clients bound to planner node '%s' (%s)",
              planner_node.c_str(), planner_ns.c_str());
}

void PlanningTab::runOnGuiThread(std::function<void()> fn) {
  QMetaObject::invokeMethod(this, std::move(fn), Qt::QueuedConnection);
}

void PlanningTab::setPlannerReady(bool ready) {
  if (planner_ready_ == ready) {
    return;
  }
  planner_ready_ = ready;

  // Poll quickly only while the planner is coming up; once it has answered,
  // back the probe off to 1 Hz. The timer could be null during construction
  // (setPlannerReady(false) is called before the timer is created), so guard.
  if (readiness_timer_) {
    readiness_timer_->setInterval(ready ? 1000 : 250);
  }

  ui_->doubleSpinBoxTimeDilationFactor->setEnabled(ready);
  updateActionButtons();

  // A planner that has gone away (restart, GPU fault) invalidates whatever the
  // status line is still claiming, so say so rather than leaving the last
  // plan's result up as if it were current.
  if (!ready) {
    setStatus(tr("planner unavailable - waiting for it to come back"));
  }

  // The trajectory-type dropdown that used to be gated here now lives on the
  // Context tab, so the gate travels with it.
  Q_EMIT plannerReadyChanged(ready);

  RCLCPP_INFO(node_->get_logger(), "Planner is %s",
              ready ? "ready" : "not ready");
}

void PlanningTab::updateActionButtons() {
  const bool can_start = planner_ready_ && !goal_active_;
  ui_->generateTrajectory->setEnabled(can_start);
  ui_->sendTrajectory->setEnabled(can_start);
  ui_->generateAndSend->setEnabled(can_start);
  ui_->stopRobot->setEnabled(goal_active_);
  // Clear does not touch the server: it just re-seeds the goal fields from the
  // target marker, so it is gated on having a marker and nothing else.
  ui_->clearGoal->setEnabled(target_display_ != nullptr);
}

void PlanningTab::pollPlannerReady() {
  // Watchdog for wedged probes: rclcpp's async get_parameters has no timeout --
  if (planner_poll_in_flight_) {
    const auto outstanding_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - planner_poll_sent_at_)
            .count();
    if (outstanding_ms < 2000) {
      return;
    }
    planner_poll_in_flight_ = false;
  }
  if (!param_client_ || !param_client_->service_is_ready()) {
    setPlannerReady(false);
    return;
  }

  const uint64_t seq = ++planner_poll_seq_;
  planner_poll_in_flight_ = true;
  planner_poll_sent_at_ = std::chrono::steady_clock::now();
  param_client_->get_parameters(
      {"node_is_available"},
      [this, seq](std::shared_future<std::vector<rclcpp::Parameter>> future) {
        bool ready = false;
        try {
          auto params = future.get();
          ready = !params.empty() && params[0].as_bool();
        } catch (const std::exception &e) {
          RCLCPP_WARN(node_->get_logger(), "node_is_available check failed: %s",
                      e.what());
        }
        runOnGuiThread([this, ready, seq]() {
          // Only the newest probe may commit its answer; a completion arriving
          // for a probe the watchdog already replaced is stale.
          if (seq != planner_poll_seq_ || !planner_poll_in_flight_) {
            return;
          }
          planner_poll_in_flight_ = false;
          setPlannerReady(ready);
        });
      });
}

void PlanningTab::load(const rviz_common::Config &config) {
  Panel::load(config);

  // Only the request parameters live here. The planner node and the
  // trajectory type describe the server rather than one request, so they are
  // the Context tab's to save -- see planning_tab.hpp.
  float time_dilation_factor;
  if (config.mapGetFloat("time_dilation_factor", &time_dilation_factor)) {
    time_dilation_factor_ = time_dilation_factor;
    ui_->doubleSpinBoxTimeDilationFactor->setValue(time_dilation_factor);
  }
}

void PlanningTab::save(rviz_common::Config config) const {
  Panel::save(config);
  config.mapSetValue("time_dilation_factor", time_dilation_factor_);
}

void PlanningTab::updateTimeDilationFactor(double value) {
  time_dilation_factor_ = value;

  // This slot fires during RViz config load (spinbox setValue -> valueChanged),
  // so it must never block: set_parameters_atomically is called with a callback
  // (AsyncParametersClient), never the SyncParametersClient/blocking form.
  if (!param_client_ || !param_client_->service_is_ready()) {
    RCLCPP_WARN(node_->get_logger(), "Planner parameter service not available; "
                                     "skipping time_dilation_factor update");
    return;
  }
  param_client_->set_parameters_atomically(
      {rclcpp::Parameter("time_dilation_factor", time_dilation_factor_)},
      [this, value](
          std::shared_future<rcl_interfaces::msg::SetParametersResult> future) {
        try {
          auto result = future.get();
          if (result.successful) {
            RCLCPP_INFO(node_->get_logger(), "Time dilation factor set to %.2f",
                        value);
          } else {
            RCLCPP_ERROR(node_->get_logger(),
                         "Failed to set time_dilation_factor: %s",
                         result.reason.c_str());
          }
        } catch (const std::exception &e) {
          RCLCPP_ERROR(node_->get_logger(),
                       "Exception setting time_dilation_factor: %s", e.what());
        }
      });
}

void PlanningTab::initializeEmbedded(rviz_common::DisplayContext *context) {
  // Just record it: this tab has no onInitialize() of its own and nothing
  embedded_display_context_ = context;
}

void PlanningTab::setPlannerNode(const QString &planner_node) {
  const std::string new_node = planner_node.trimmed().toStdString();
  if (new_node.empty() || new_node == planner_node_) {
    return;
  }

  RCLCPP_INFO(node_->get_logger(), "Switching planner node to '%s'",
              new_node.c_str());

  // Tear down any in-flight goal / MPC loop before repointing the clients.
  stopMpc();
  if (goal_active_ && goal_handle_) {
    try {
      action_ptr_->async_cancel_goal(goal_handle_);
    } catch (const std::exception &e) {
      RCLCPP_WARN(node_->get_logger(), "Exception during cancel: %s", e.what());
    }
  }
  goal_active_ = false;
  goal_handle_.reset();

  planner_node_ = new_node;
  createPlannerClients(new_node);

  // Mirror the choice in the Options group, the way MoveIt repeats
  ui_->plannerNodeEcho->setText(QString::fromStdString(planner_node_));

  // The new planner node needs re-probing for readiness (it may still be
  // warming up or simply not responding yet).
  setPlannerReady(false);
}

void PlanningTab::setTrajectoryType(int combo_index) {
  // Combo order is Classic, MPC, so the index is the SetPlanner value.
  // ContextTab only calls this once the server has confirmed the switch, so
  // this mirrors reality rather than the user's wish.
  if (combo_index < 0 || combo_index > 1) {
    RCLCPP_ERROR(node_->get_logger(), "Invalid trajectory type index: %d",
                 combo_index);
    return;
  }
  trajectory_type_ = static_cast<uint8_t>(combo_index);
  RCLCPP_INFO(node_->get_logger(), "Trajectory type is now %s",
              trajectory_type_ == 1 ? "MPC (Real-time)"
                                    : "Classic (MotionGen)");
}

void PlanningTab::refreshPlanningDisplays() {
  auto context = displayContext();
  if (!context) {
    return;
  }

  std::vector<CuroboPlanningDisplay *> found;
  auto root_display = context->getRootDisplayGroup();
  if (root_display) {
    collectPlanningDisplaysInGroup(root_display, found);
  }

  const bool displays_changed = found != target_displays_;
  if (displays_changed) {
    target_displays_ = found;
    if (found.empty()) {
      target_display_ = nullptr;
      RCLCPP_WARN(node_->get_logger(),
                  "No CuroboPlanningDisplay found - add one via Displays -> Add");
    } else if (target_display_ != found.front()) {
      // The PRIMARY target stays the FIRST display in the tree. Classic
      // and MPC act only on this one.
      target_display_ = found.front();
      RCLCPP_INFO(node_->get_logger(), "Primary CuroboPlanningDisplay set (total: %zu)",
                  found.size());
    } else {
      // Unchanged -- fall through to the button refresh below without
      // re-logging, so a steady config is quiet.
    }
    // The Clear button is gated on having a primary target, and this poll is
    updateActionButtons();
  }

  // The Context tab owns the planner node; the display's own "Planner Node"
  // property must follow it or the goal-state IK/FK and the trajectory ghost
  // keep querying the previous server after a switch (or a new display keeps
  // the property default). Only push on a real change, so a display node the
  // user deliberately set is not overwritten every poll. setPlannerNode() is
  // itself a no-op when already bound.
  if (displays_changed || planner_node_ != last_display_planner_node_) {
    for (CuroboPlanningDisplay *display : target_displays_) {
      if (display != nullptr) {
        display->setPlannerNode(planner_node_);
      }
    }
    last_display_planner_node_ = planner_node_;
  }

  refreshTargetToolList();
}

void PlanningTab::collectPlanningDisplaysInGroup(
    rviz_common::DisplayGroup *group, std::vector<CuroboPlanningDisplay *> &out) {
  for (int i = 0; i < group->numDisplays(); ++i) {
    rviz_common::Display *child = group->getDisplayAt(i);
    if (child == nullptr) {
      continue;
    }
    CuroboPlanningDisplay *target = dynamic_cast<CuroboPlanningDisplay *>(child);
    if (target != nullptr) {
      out.push_back(target);
      continue; // a CuroboPlanningDisplay cannot also be a subgroup
    }
    // A DisplayGroup derives from Display, so subgroups appear as children
    // of a group and must be descended into as well.
    rviz_common::DisplayGroup *subgroup =
        dynamic_cast<rviz_common::DisplayGroup *>(child);
    if (subgroup != nullptr) {
      collectPlanningDisplaysInGroup(subgroup, out);
    }
  }
}

} // namespace curobov2_ros_rviz
