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
#include <QComboBox>
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
      target_display_{nullptr}, tool_frames_in_flight_{false},
      target_tool_combo_{nullptr}, user_editing_pose_{false},
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

  target_tool_combo_ = new QComboBox(this);
  target_tool_combo_->setToolTip(
      tr("Tool frame edited by the pose fields and marker drags."));
  ui_->queryFormLayout->insertRow(0, tr("Target tool:"), target_tool_combo_);
  connect(target_tool_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &PlanningTab::onTargetToolChanged);

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

  this->joint_info_client_ =
      node_->create_client<curobov2_ros_interfaces::srv::GetJointInfo>(
          planner_ns + "get_joint_info");
  tool_frames_.clear();
  pushed_tool_frames_.clear();
  tool_frames_in_flight_ = false;

  // Publisher streaming the live MPC goal towards the current target pose
  // (stamped: the pose rides in the RViz fixed frame, see goalFrame()).
  this->mpc_goal_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
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
  if (ready && tool_frames_.empty()) {
    refreshToolFrames();
  }
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
          if (ready && tool_frames_.empty()) {
            refreshToolFrames();
          }
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

void PlanningTab::on_sendTrajectory_clicked() {
  // Execution is MTC-literal: the server neither re-plans nor reuses a cached
  if (last_trajectory_.points.empty()) {
    RCLCPP_ERROR(node_->get_logger(),
                 "Send: no generated trajectory to replay; press Generate "
                 "first");
    return;
  }
  if (!action_ptr_) {
    RCLCPP_ERROR(node_->get_logger(),
                 "execute_trajectory action client not created yet");
    return;
  }

  auto goal_request =
      curobov2_ros_interfaces::action::SendTrajectory::Goal();

  // The goalset still describes the plan's target (a hint for the server's
  // frame/planner context); what actually drives the arm is the replayed
  // trajectory. buildGoalsets() warns when no display exists.
  goal_request.goal.goalsets = buildGoalsets();
  goal_request.replay = true;
  goal_request.replay_trajectory = last_trajectory_;

  auto send_goal_options = rclcpp_action::Client<
      curobov2_ros_interfaces::action::SendTrajectory>::SendGoalOptions();

  send_goal_options.goal_response_callback = std::bind(
      &PlanningTab::goal_response_callback, this, std::placeholders::_1);
  send_goal_options.result_callback =
      std::bind(&PlanningTab::result_callback, this, std::placeholders::_1);

  action_ptr_->async_send_goal(goal_request, send_goal_options);
  goal_active_ = true;
  setStatus(tr("executing..."));
  updateActionButtons();
}

void PlanningTab::setStatus(const QString &text, const QString &detail) {
  // Assert the contract rather than trusting it: a QLabel touched from the
  // spin thread corrupts state rather than throwing, so the failure would
  // surface as a random crash much later with nothing pointing here.
  Q_ASSERT(QThread::currentThread() == qApp->thread());

  const QString line = detail.isEmpty() ? text : (text + "  —  " + detail);
  ui_->statusLabel->setText(line);
  ui_->statusLabel->setToolTip(line);
}

// goal_response_callback/result_callback are rclcpp_action client callbacks --
void PlanningTab::result_callback(
    const rclcpp_action::ClientGoalHandle<
        curobov2_ros_interfaces::action::SendTrajectory>::WrappedResult
        &result) {
  QString outcome;
  QString message;
  QString stats;

  if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
    // This used to log at ERROR with the text "Goal was succeeded", so every
    // clean plan put a red line in the log.
    outcome = tr("goal succeeded");
  } else if (result.code == rclcpp_action::ResultCode::ABORTED) {
    outcome = tr("goal aborted");
  } else if (result.code == rclcpp_action::ResultCode::CANCELED) {
    outcome = tr("goal canceled");
  } else {
    outcome = tr("goal finished with an unrecognised result code");
  }

  if (result.result) {
    // The path is result.result->result.*, NOT result.result->*.
    const auto &payload = result.result->result;

    message = QString::fromStdString(payload.message).trimmed();

    // PlanningStats' scalars are always populated -- no request flag, every
    const auto &s = payload.stats;
    stats =
        tr("%1 candidate(s) generated, %2 solved, %3 pruned; %4 waypoint(s)")
            .arg(s.candidates_generated)
            .arg(s.candidates_solved)
            .arg(s.candidates_pruned)
            .arg(s.waypoints_planned);

    // A rejected result (e.g. a goal the server declined) carries the
    // reason and no search happened, so don't dress it up with zeros.
    if (!payload.success) {
      stats.clear();
    }
  } else {
    message = tr("(no result payload)");
  }

  // The log line keeps the level that matches the outcome, and carries the
  // server's message -- which used to be read by nobody at all.
  const QString logged =
      message.isEmpty() ? outcome : (outcome + ": " + message);
  if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
    RCLCPP_INFO(node_->get_logger(), "%s", logged.toStdString().c_str());
  } else if (result.code == rclcpp_action::ResultCode::CANCELED) {
    RCLCPP_WARN(node_->get_logger(), "%s", logged.toStdString().c_str());
  } else {
    RCLCPP_ERROR(node_->get_logger(), "%s", logged.toStdString().c_str());
  }

  runOnGuiThread([this, outcome, message, stats]() {
    mpc_goal_timer_->stop();
    mpc_active_ = false;
    mpc_starting_ = false;
    goal_handle_.reset();
    goal_active_ = false;

    setStatus(message.isEmpty() ? outcome : message, stats);

    // The execution just finished; force-out a readiness probe that may be
    if (planner_poll_in_flight_) {
      planner_poll_in_flight_ = false;
    }
    pollPlannerReady();
    updateActionButtons();
  });
}

void PlanningTab::goal_response_callback(
    std::shared_ptr<rclcpp_action::ClientGoalHandle<
        curobov2_ros_interfaces::action::SendTrajectory>>
        goal_handle) {
  const bool accepted = static_cast<bool>(goal_handle);
  if (!accepted) {
    RCLCPP_ERROR(node_->get_logger(), "Goal was rejected by server");
  } else {
    RCLCPP_INFO(node_->get_logger(),
                "Goal accepted by server, waiting for result");
  }
  runOnGuiThread([this, goal_handle, accepted]() {
    this->goal_handle_ = goal_handle;
    if (accepted && mpc_starting_) {
      // MPC goal accepted: begin streaming the live target pose (~10 Hz).
      mpc_starting_ = false;
      mpc_active_ = true;
      mpc_goal_timer_->start();
      RCLCPP_INFO(node_->get_logger(),
                  "MPC active - drag the target to retarget the robot live");
    } else if (!accepted) {
      mpc_starting_ = false;
    }
  });
}

void PlanningTab::on_generateTrajectory_clicked() {
  generateTrajectoryAsync(nullptr);
}

void PlanningTab::generateTrajectoryAsync(std::function<void(bool)> on_done) {
  // Always called from the GUI thread (button slots), so early-return failure
  // paths can invoke on_done() directly; only the async completion below
  // (which fires on the background spin thread) needs runOnGuiThread.
  if (!trajectory_generation_client_->service_is_ready()) {
    RCLCPP_ERROR(node_->get_logger(),
                 "generate_trajectory service not available");
    if (on_done) {
      on_done(false);
    }
    return;
  }

  // A single goalset from the primary display. buildGoalsets() warns
  // below when no display is available.
  const auto goalsets = buildGoalsets();
  if (goalsets.empty()) {
    if (on_done) {
      on_done(false);
    }
    return;
  }

  auto goal_request = std::make_shared<
      curobov2_ros_interfaces::srv::TrajectoryGeneration::Request>();
  goal_request->request.goalsets = goalsets;

  trajectory_generation_client_->async_send_request(
      goal_request,
      [this,
       on_done](rclcpp::Client<curobov2_ros_interfaces::srv::
                                   TrajectoryGeneration>::SharedFuture future) {
        bool success = false;
        std::string message;
        // Built on the spin thread but published to the GUI thread below, so
        // last_trajectory_ is only ever written and read on the GUI thread.
        trajectory_msgs::msg::JointTrajectory replay;
        try {
          auto response = future.get();
          success = response->response.success;
          message = response->response.message;
          if (success) {
            replay = toReplayTrajectory(response->response);
          }
        } catch (const std::exception &e) {
          message = e.what();
        }
        if (success) {
          RCLCPP_INFO(node_->get_logger(), "generate_trajectory succeeded");
        } else {
          RCLCPP_ERROR(node_->get_logger(), "generate_trajectory failed: %s",
                       message.c_str());
        }
        runOnGuiThread([this, on_done, success, replay]() {
          if (success) {
            // Keep the plan for Send (and for the Generate-and-Send chain,
            // which calls on_sendTrajectory_clicked below).
            last_trajectory_ = replay;
          }
          if (on_done) {
            on_done(success);
          }
        });
      });
}

void PlanningTab::on_generateAndSend_clicked() {
  // Check if MPC planner is selected (planner_type == 1)
  // Check if the MPC trajectory type is active (set on the Context tab)
  if (trajectory_type_ == 1) {
    // MPC Mode: the panel owns the reactive flow (SetPlanner MPC ->
    // execute_trajectory action -> live mpc_goal streaming while the gizmo
    // is dragged).
    if (!target_display_) {
      RCLCPP_WARN(node_->get_logger(),
                  "Target display not available; add CuroboPlanningDisplay to RViz to "
                  "start MPC tracking");
      return;
    }
    if (target_display_->getToolFrames().size() > 1) {
      RCLCPP_WARN(node_->get_logger(),
                  "MPC tracks a single tool frame; select one in the target "
                  "display's Tool Frames property");
      return;
    }
    RCLCPP_INFO(node_->get_logger(), "Starting MPC tracking");
    startMpc();
    return;
  }

  // Classic Mode: generate, then chain into send once generation actually
  RCLCPP_INFO(node_->get_logger(), "Classic mode: generate and execute once");
  generateTrajectoryAsync([this](bool success) {
    if (success) {
      on_sendTrajectory_clicked();
    } else {
      RCLCPP_ERROR(node_->get_logger(),
                   "Trajectory generation failed; not sending");
    }
  });
}

// --- MPC live-tracking (owned by the panel) ---

void PlanningTab::startMpc() {
  if (mpc_starting_ || mpc_active_) {
    return;
  }
  if (!target_display_) {
    RCLCPP_WARN(node_->get_logger(), "Target display not available yet");
    return;
  }
  if (!set_planner_client_ || !set_planner_client_->service_is_ready()) {
    RCLCPP_WARN(node_->get_logger(), "set_planner service not available");
    return;
  }
  if (!action_ptr_ || !action_ptr_->action_server_is_ready()) {
    RCLCPP_WARN(node_->get_logger(),
                "execute_trajectory action server not available");
    return;
  }

  // The planner must be on MPC for the reactive loop to consume mpc_goal.
  auto request = std::make_shared<
      curobov2_ros_interfaces::srv::SetPlanner::Request>();
  request->planner_type =
      curobov2_ros_interfaces::srv::SetPlanner::Request::MPC;
  mpc_starting_ = true;

  set_planner_client_->async_send_request(
      request,
      [this](rclcpp::Client<curobov2_ros_interfaces::srv::SetPlanner>::
                 SharedFuture future) {
        bool success = false;
        std::string message;
        try {
          auto response = future.get();
          success = response->success;
          message = response->message;
        } catch (const std::exception &e) {
          message = e.what();
        }

        if (!success) {
          RCLCPP_ERROR(node_->get_logger(), "Planner switch to MPC failed: %s",
                       message.c_str());
          runOnGuiThread([this]() { mpc_starting_ = false; });
          return;
        }
        RCLCPP_INFO(node_->get_logger(), "Switched planner to MPC");
        runOnGuiThread([this]() { sendMpcGoal(); });
      });
}

void PlanningTab::sendMpcGoal() {
  if (!target_display_) {
    RCLCPP_WARN(node_->get_logger(),
                "Target display not available; cannot start MPC goal");
    return;
  }
  if (!action_ptr_ || !action_ptr_->action_server_is_ready()) {
    RCLCPP_WARN(node_->get_logger(),
                "execute_trajectory action server not available yet");
    return;
  }

  auto goal = curobov2_ros_interfaces::action::SendTrajectory::Goal();

  // start_pose intentionally left EMPTY so the server resolves the start state

  curobov2_ros_interfaces::msg::Goalset gset;
  gset.poses.push_back(target_display_->getPose());
  // Fixed-frame coordinates (the display is frame-dumb); the server resolves
  // them into the base link via TF.
  gset.frame_id = goalFrame();
  goal.goal.goalsets.push_back(gset);

  auto send_goal_options = rclcpp_action::Client<
      curobov2_ros_interfaces::action::SendTrajectory>::SendGoalOptions();
  send_goal_options.goal_response_callback = std::bind(
      &PlanningTab::goal_response_callback, this, std::placeholders::_1);
  send_goal_options.result_callback =
      std::bind(&PlanningTab::result_callback, this, std::placeholders::_1);

  // Arm the MPC-start latch before sending: goal_response_callback consumes
  mpc_starting_ = true;
  action_ptr_->async_send_goal(goal, send_goal_options);
  setStatus(tr("executing..."));

  // await goal acceptance (goal_response_callback) before starting the stream
  last_published_goal_ = target_display_->getPose();
  goal_active_ = true;
  updateActionButtons();
}

void PlanningTab::streamMpcGoal() {
  if (!mpc_active_ || !mpc_goal_pub_ || !target_display_) {
    return;
  }
  const geometry_msgs::msg::Pose &p = target_display_->getPose();
  const geometry_msgs::msg::Pose &l = last_published_goal_;
  const bool changed =
      p.position.x != l.position.x || p.position.y != l.position.y ||
      p.position.z != l.position.z || p.orientation.x != l.orientation.x ||
      p.orientation.y != l.orientation.y ||
      p.orientation.z != l.orientation.z || p.orientation.w != l.orientation.w;
  if (changed) {
    geometry_msgs::msg::PoseStamped stamped;
    stamped.header.stamp = node_->now();
    stamped.header.frame_id = goalFrame();
    stamped.pose = p;
    mpc_goal_pub_->publish(stamped);
    last_published_goal_ = p;
  }
}

void PlanningTab::stopMpc() {
  mpc_starting_ = false;
  if (mpc_goal_timer_) {
    mpc_goal_timer_->stop();
  }
  if (mpc_active_ && goal_handle_) {
    try {
      action_ptr_->async_cancel_goal(goal_handle_);
    } catch (const std::exception &e) {
      RCLCPP_WARN(node_->get_logger(), "Exception during MPC cancel: %s",
                  e.what());
    }
  }
  mpc_active_ = false;
}

void PlanningTab::on_stopRobot_clicked() {
  // Stop live MPC tracking if active
  stopMpc();

  // Cancel the action goal if it exists and is still active
  if (goal_handle_) {
    try {
      // Use async_cancel_goal without blocking spin
      // The result_callback will handle the cleanup
      action_ptr_->async_cancel_goal(goal_handle_);
      RCLCPP_INFO(node_->get_logger(), "Cancel request sent");
    } catch (const std::exception &e) {
      RCLCPP_WARN(node_->get_logger(), "Exception during cancel: %s", e.what());
    }
  }

  // Re-enable buttons immediately (don't wait for result)
  goal_active_ = false;
  updateActionButtons();
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

void PlanningTab::refreshToolFrames() {
  if (tool_frames_in_flight_ || !joint_info_client_ ||
      !joint_info_client_->service_is_ready()) {
    return;
  }
  tool_frames_in_flight_ = true;
  auto request =
      std::make_shared<curobov2_ros_interfaces::srv::GetJointInfo::Request>();
  joint_info_client_->async_send_request(
      request,
      [this](rclcpp::Client<curobov2_ros_interfaces::srv::GetJointInfo>::
                 SharedFuture future) {
        std::vector<std::string> frames;
        bool ok = false;
        try {
          auto response = future.get();
          ok = response->success;
          frames = response->tool_frames;
        } catch (const std::exception &e) {
          RCLCPP_WARN(node_->get_logger(), "get_joint_info failed: %s", e.what());
        }
        runOnGuiThread([this, ok, frames]() {
          tool_frames_in_flight_ = false;
          if (!ok) {
            return;
          }
          tool_frames_ = frames;
          applyToolFramesToDisplay();
          refreshTargetToolCombo();
        });
      });
}

void PlanningTab::applyToolFramesToDisplay() {
  if (!target_display_ || tool_frames_.empty() ||
      target_display_->toolFramesUserSet()) {
    return;
  }
  // Only push when the server list actually changed. Otherwise the periodic
  // poll would overwrite whatever the user is typing into Tool Frames.
  if (tool_frames_ != pushed_tool_frames_) {
    target_display_->setToolFrames(tool_frames_);
    pushed_tool_frames_ = tool_frames_;
  }
}

void PlanningTab::refreshTargetToolCombo() {
  if (!target_tool_combo_) {
    return;
  }
  const QSignalBlocker blocker(target_tool_combo_);
  const auto tools =
      target_display_ ? target_display_->getToolFrames() : std::vector<std::string>();
  const QString active = target_display_
                             ? QString::fromStdString(target_display_->getActiveTool())
                             : QString();

  bool same = target_tool_combo_->count() == static_cast<int>(tools.size());
  for (int i = 0; same && i < target_tool_combo_->count(); ++i) {
    same = target_tool_combo_->itemData(i).toString() ==
      QString::fromStdString(tools[static_cast<size_t>(i)]);
  }
  if (!same) {
    target_tool_combo_->clear();
    for (const auto & tool : tools) {
      const QString label = tool.empty() ? tr("target") : QString::fromStdString(tool);
      target_tool_combo_->addItem(label, QString::fromStdString(tool));
    }
  }
  const int index = target_tool_combo_->findData(active);
  if (index >= 0) {
    target_tool_combo_->setCurrentIndex(index);
  }
  target_tool_combo_->setEnabled(target_tool_combo_->count() > 1);
}

void PlanningTab::onTargetToolChanged(int index) {
  if (!target_display_ || index < 0) {
    return;
  }
  target_display_->setActiveTool(
      target_tool_combo_->itemData(index).toString().toStdString());
  updateMarkerPoseDisplay();
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

  if (found != target_displays_) {
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

  applyToolFramesToDisplay();
  refreshTargetToolCombo();
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

std::string PlanningTab::goalFrame() const {
  // Every pose this tab sends (goalsets, mpc_goal) is expressed in the RViz
  if (target_display_) {
    const std::string gizmo_frame = target_display_->getGizmoFrame();
    if (!gizmo_frame.empty()) {
      return gizmo_frame;
    }
  }
  auto context = displayContext();
  if (context && context->getFrameManager()) {
    return context->getFrameManager()->getFixedFrame();
  }
  return "";
}

std::vector<curobov2_ros_interfaces::msg::Goalset>
PlanningTab::buildGoalsets() {
  std::vector<curobov2_ros_interfaces::msg::Goalset> goalsets;

  if (target_display_) {
    const auto targets = target_display_->getToolPoses();
    curobov2_ros_interfaces::msg::Goalset gset;
    gset.frame_id = goalFrame();
    const bool unnamed = targets.size() == 1 && targets.front().first.empty();
    if (unnamed || targets.empty()) {
      gset.poses.push_back(targets.empty() ? target_display_->getPose()
                                           : targets.front().second);
    } else {
      for (const auto & target : targets) {
        if (target.first.empty()) {
          continue;
        }
        gset.tool_frames.push_back(target.first);
        gset.tool_poses.push_back(target.second);
      }
      if (gset.tool_frames.empty()) {
        gset.poses.push_back(target_display_->getPose());
      }
    }
    // Compatibility fallback: also carry the active pose as a single-tool
    // goalset. New servers prefer tool_frames/tool_poses; older servers (which
    // do not know those fields) still get a plan for the active tool.
    if (gset.poses.empty()) {
      gset.poses.push_back(target_display_->getPose());
    }
    RCLCPP_INFO(
        node_->get_logger(), "Goal request: %zu pose(s), %zu tool pose(s), frame '%s'",
        gset.poses.size(), gset.tool_poses.size(), gset.frame_id.c_str());
    goalsets.push_back(gset);
  }
  if (goalsets.empty()) {
    RCLCPP_WARN(
        node_->get_logger(),
        "Target display not available - goal sent WITHOUT a target pose "
        "(planning will fail; add CuroboPlanningDisplay to RViz)");
  }
  return goalsets;
}

// Quaternion orientation (q and -q are the same rotation; settle the sign).
void PlanningTab::canonicalize(geometry_msgs::msg::Quaternion & q) {
  const double c[4] = {q.x, q.y, q.z, q.w};
  int sign_ref = 0;
  for (int i = 1; i < 4; ++i) {
    if (std::fabs(c[i]) > std::fabs(c[sign_ref])) {
      sign_ref = i;
    }
  }

  if (c[sign_ref] < 0.0) {
    q.x = -q.x;
    q.y = -q.y;
    q.z = -q.z;
    q.w = -q.w;
  }
}

bool PlanningTab::normalize(geometry_msgs::msg::Quaternion & q) {
  // A quaternion is only a rotation if it is unit length. Four independent spin
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (norm < 1e-9) {
    return false;
  }
  q.x /= norm;
  q.y /= norm;
  q.z /= norm;
  q.w /= norm;
  return true;
}

void PlanningTab::readQuaternionFromSpinboxes(geometry_msgs::msg::Quaternion & q) {
  q.x = ui_->spinBoxQx->value();
  q.y = ui_->spinBoxQy->value();
  q.z = ui_->spinBoxQz->value();
  q.w = ui_->spinBoxQw->value();
}

void PlanningTab::writeQuaternionToSpinboxes(const geometry_msgs::msg::Quaternion & q) {
  // Every one of these boxes has valueChanged connected to
  const QSignalBlocker bq(ui_->spinBoxQx);
  const QSignalBlocker by(ui_->spinBoxQy);
  const QSignalBlocker bz(ui_->spinBoxQz);
  const QSignalBlocker bw(ui_->spinBoxQw);
  ui_->spinBoxQx->setValue(q.x);
  ui_->spinBoxQy->setValue(q.y);
  ui_->spinBoxQz->setValue(q.z);
  ui_->spinBoxQw->setValue(q.w);
}

void PlanningTab::applyPoseFromSpinboxes() {
  if (!target_display_) {
    RCLCPP_WARN(node_->get_logger(), "Target display not available yet");
    return;
  }

  // Get values from spinboxes
  geometry_msgs::msg::Pose pose;
  pose.position.x = ui_->spinBoxPosX->value();
  pose.position.y = ui_->spinBoxPosY->value();
  pose.position.z = ui_->spinBoxPosZ->value();

  // The orientation is read, normalised and canonicalised, then WRITTEN BACK to
  geometry_msgs::msg::Quaternion q;
  readQuaternionFromSpinboxes(q);
  const bool usable = normalize(q);
  if (!usable) {
    RCLCPP_WARN(
        node_->get_logger(),
        "Goal orientation is a zero quaternion (0,0,0,0), which is not a "
        "rotation; leaving the goal orientation alone. w=1 is the identity.");
    return;
  }
  canonicalize(q);
  writeQuaternionToSpinboxes(q);
  pose.orientation = q;

  // Apply the new pose to the marker
  target_display_->setPose(pose);

  // Update last displayed values to match what we just set
  last_displayed_x_ = pose.position.x;
  last_displayed_y_ = pose.position.y;
  last_displayed_z_ = pose.position.z;
  last_displayed_q_ = q;

  RCLCPP_INFO(
      node_->get_logger(),
      "Applied pose: X=%.3f, Y=%.3f, Z=%.3f, Q=(%.4f, %.4f, %.4f, %.4f)",
      pose.position.x, pose.position.y, pose.position.z, q.x, q.y, q.z, q.w);
}

bool PlanningTab::eventFilter(QObject *obj, QEvent *event) {
  // Check if the event is a FocusIn event on one of the pose spinboxes
  if (event->type() == QEvent::FocusIn) {
    if (obj == ui_->spinBoxPosX || obj == ui_->spinBoxPosY ||
        obj == ui_->spinBoxPosZ || obj == ui_->spinBoxQx ||
        obj == ui_->spinBoxQy || obj == ui_->spinBoxQz ||
        obj == ui_->spinBoxQw) {
      user_editing_pose_ = true;
    }
  }
  // Pass the event to the base class
  return QObject::eventFilter(obj, event);
}

void PlanningTab::maybeSeedTargetFromTool() {
  // One-shot seed: a freshly added CuroboPlanningDisplay parks at its hardcoded
  if (!target_display_ || target_display_ == seeded_display_ ||
      user_editing_pose_) {
    return;
  }
  if (!target_display_->isDefaultPose()) {
    seeded_display_ = target_display_;
    return;
  }
  geometry_msgs::msg::PoseStamped::SharedPtr tool;
  {
    std::lock_guard<std::mutex> lock(tool_pose_mutex_);
    tool = tool_pose_;
  }
  if (!tool || tool->header.frame_id.empty()) {
    return;
  }
  const std::string fixed = goalFrame();
  if (fixed.empty()) {
    return;  // No fixed frame yet: leave the default, retry next poll.
  }
  if (fixed == tool->header.frame_id) {
    target_display_->setPose(tool->pose);
    seeded_display_ = target_display_;
    return;
  }
  try {
    const auto tf = seed_tf_buffer_->lookupTransform(
        fixed, tool->header.frame_id, tf2::TimePointZero,
        tf2::durationFromSec(0.5));
    geometry_msgs::msg::PoseStamped out;
    tf2::doTransform(*tool, out, tf);
    target_display_->setPose(out.pose);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 5000,
        "Target seed: cannot transform tool pose '%s' -> '%s': %s",
        tool->header.frame_id.c_str(), fixed.c_str(), ex.what());
    return;
  }
  seeded_display_ = target_display_;
}

void PlanningTab::updateMarkerPoseDisplay() {
  // Don't update if marker not found yet or if user is editing
  if (!target_display_ || user_editing_pose_) {
    return;
  }

  maybeSeedTargetFromTool();

  auto pose = target_display_->getPose();

  geometry_msgs::msg::Quaternion q = pose.orientation;
  canonicalize(q);

  // Compare with last displayed values - only update if changed
  constexpr double epsilon_pos = 1e-6;  // metres
  // Quaternion components live in [-1, 1], so the threshold is an absolute one
  // on the components rather than an angle in degrees.
  constexpr double epsilon_rot = 1e-4;

  bool x_changed = std::isnan(last_displayed_x_) ||
                   std::fabs(pose.position.x - last_displayed_x_) > epsilon_pos;
  bool y_changed = std::isnan(last_displayed_y_) ||
                   std::fabs(pose.position.y - last_displayed_y_) > epsilon_pos;
  bool z_changed = std::isnan(last_displayed_z_) ||
                   std::fabs(pose.position.z - last_displayed_z_) > epsilon_pos;
  // A quaternion has no meaningful per-component "change" on its own -- the
  const bool q_changed = std::isnan(last_displayed_q_.w) ||
                         std::fabs(q.x - last_displayed_q_.x) > epsilon_rot ||
                         std::fabs(q.y - last_displayed_q_.y) > epsilon_rot ||
                         std::fabs(q.z - last_displayed_q_.z) > epsilon_rot ||
                         std::fabs(q.w - last_displayed_q_.w) > epsilon_rot;

  // Only update if at least one value has changed
  if (!x_changed && !y_changed && !z_changed && !q_changed) {
    return;
  }

  // Update the spinboxes with current pose
  // Block signals to avoid triggering updates while we're setting values
  ui_->spinBoxPosX->blockSignals(true);
  ui_->spinBoxPosY->blockSignals(true);
  ui_->spinBoxPosZ->blockSignals(true);
  // The four quaternion boxes are muted inside writeQuaternionToSpinboxes();
  // doing it here too would be the nested-blockSignals hazard described there.

  if (x_changed) {
    ui_->spinBoxPosX->setValue(pose.position.x);
    last_displayed_x_ = pose.position.x;
  }
  if (y_changed) {
    ui_->spinBoxPosY->setValue(pose.position.y);
    last_displayed_y_ = pose.position.y;
  }
  if (z_changed) {
    ui_->spinBoxPosZ->setValue(pose.position.z);
    last_displayed_z_ = pose.position.z;
  }
  if (q_changed) {
    writeQuaternionToSpinboxes(q);
    last_displayed_q_ = q;
  }

  ui_->spinBoxPosX->blockSignals(false);
  ui_->spinBoxPosY->blockSignals(false);
  ui_->spinBoxPosZ->blockSignals(false);
}

// Clear re-seeds the goal from the robot's live state.
void PlanningTab::on_clearGoal_clicked() {
  if (target_display_ == nullptr) {
    RCLCPP_WARN(
        node_->get_logger(),
        "Clear needs a CuroboPlanningDisplay in RViz; there is no pose to read back");
    return;
  }

  user_editing_pose_ = false;

  last_displayed_x_ = std::numeric_limits<double>::quiet_NaN();
  last_displayed_y_ = std::numeric_limits<double>::quiet_NaN();
  last_displayed_z_ = std::numeric_limits<double>::quiet_NaN();
  // NaN w = "never displayed".
  last_displayed_q_.x = 0.0;
  last_displayed_q_.y = 0.0;
  last_displayed_q_.z = 0.0;
  last_displayed_q_.w = std::numeric_limits<double>::quiet_NaN();

  updateMarkerPoseDisplay();
  RCLCPP_INFO(node_->get_logger(),
              "Goal state reset to the target marker pose");
}

} // namespace curobov2_ros_rviz
