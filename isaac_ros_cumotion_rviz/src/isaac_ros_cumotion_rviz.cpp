#include "isaac_ros_cumotion_rviz/isaac_ros_cumotion_rviz.hpp"
#include "isaac_ros_cumotion_interfaces/msg/goalset.hpp"
#include <cmath>

namespace isaac_ros_cumotion_rviz
{
  RvizArgsPanel::RvizArgsPanel(QWidget *parent)
    : Panel{parent}
    , ui_(std::make_unique<Ui::gui_parameters>())
    , node_{nullptr}
    , param_client_{nullptr}
    , planner_ready_{false}
    , planner_poll_in_flight_{false}
    , planner_poll_seq_{0}
    , goal_active_{false}
    , action_ptr_{nullptr}
    , trajectory_generation_client_{nullptr}
    , goal_handle_{nullptr}
    , time_dilation_factor_{1.0}
    , target_display_{nullptr}
    , user_editing_pose_{false}
    , last_displayed_x_{std::numeric_limits<double>::quiet_NaN()}
    , last_displayed_y_{std::numeric_limits<double>::quiet_NaN()}
    , last_displayed_z_{std::numeric_limits<double>::quiet_NaN()}
    , last_displayed_roll_{std::numeric_limits<double>::quiet_NaN()}
    , last_displayed_pitch_{std::numeric_limits<double>::quiet_NaN()}
    , last_displayed_yaw_{std::numeric_limits<double>::quiet_NaN()}
    , planner_node_{"unified_planner"}
    , mpc_goal_pub_{nullptr}
    , mpc_active_{false}
    , mpc_starting_{false}
    , mpc_goal_timer_{nullptr}
    , set_planner_client_{nullptr}
    , current_planner_type_{0}
  {
    // Extend the widget with all attributes and children from UI file
    ui_->setupUi(this);

    // Init rclcpp node
    auto options = rclcpp::NodeOptions().arguments(
        {"--ros-args", "--remap", "__node:=rviz_updata_parameters_node", "--"});
    node_ = std::make_shared<rclcpp::Node>("_", options);

    // Declare base_link parameter with default value
    node_->declare_parameter<std::string>("base_link", "base_0");

    // The planner node the panel's clients bind to is configurable via the
    // "planner_node_name" parameter on this panel's node
    // (/rviz_updata_parameters_node) and, once built, via the "Planner Node"
    // dropdown in this panel. The RViz config file overrides it on load().
    node_->declare_parameter<std::string>("planner_node_name", "unified_planner");
    planner_node_ = node_->get_parameter("planner_node_name").as_string();

    // Sync the dropdown with the configured planner node, THEN build the
    // initial clients (before the change signal is connected, so the initial
    // construction does not double-create them).
    ui_->comboBoxPlannerNode->setCurrentText(QString::fromStdString(planner_node_));
    createPlannerClients(planner_node_);

    // Timer streaming the live MPC goal while the gizmo is dragged (10 Hz).
    mpc_goal_timer_ = new QTimer(this);
    mpc_goal_timer_->setInterval(100);
    connect(mpc_goal_timer_, &QTimer::timeout, this, &RvizArgsPanel::streamMpcGoal);

    // Connect SpinBox for time dilation to its slot
    connect(ui_->doubleSpinBoxTimeDilationFactor, SIGNAL(valueChanged(double)), this, SLOT(updateTimeDilationFactor(double)));

    // Every planner-facing widget starts disabled and stays that way until
    // pollPlannerReady() confirms the planner is actually responding -- not just
    // discoverable (see setPlannerReady()/pollPlannerReady() for why that distinction
    // matters during the planner's ~90s GPU warmup).
    setPlannerReady(false);

    QTimer* readinessTimer = new QTimer(this);
    connect(readinessTimer, &QTimer::timeout, this, &RvizArgsPanel::pollPlannerReady);
    readinessTimer->start(250);

    // Connect pose spinboxes to apply changes in real-time
    // Use valueChanged to update immediately when user changes value (arrows, wheel, typing)
    connect(ui_->spinBoxPosX, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double) {
      applyPoseFromSpinboxes();
    });
    connect(ui_->spinBoxPosY, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double) {
      applyPoseFromSpinboxes();
    });
    connect(ui_->spinBoxPosZ, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double) {
      applyPoseFromSpinboxes();
    });

    // Connect orientation spinboxes
    connect(ui_->spinBoxRoll, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double) {
      applyPoseFromSpinboxes();
    });
    connect(ui_->spinBoxPitch, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double) {
      applyPoseFromSpinboxes();
    });
    connect(ui_->spinBoxYaw, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double) {
      applyPoseFromSpinboxes();
    });

    // Detect when spinbox gets focus (user starts editing) to pause auto-update
    connect(ui_->spinBoxPosX, &QDoubleSpinBox::editingFinished, this, [this]() { user_editing_pose_ = false; });
    connect(ui_->spinBoxPosY, &QDoubleSpinBox::editingFinished, this, [this]() { user_editing_pose_ = false; });
    connect(ui_->spinBoxPosZ, &QDoubleSpinBox::editingFinished, this, [this]() { user_editing_pose_ = false; });
    connect(ui_->spinBoxRoll, &QDoubleSpinBox::editingFinished, this, [this]() { user_editing_pose_ = false; });
    connect(ui_->spinBoxPitch, &QDoubleSpinBox::editingFinished, this, [this]() { user_editing_pose_ = false; });
    connect(ui_->spinBoxYaw, &QDoubleSpinBox::editingFinished, this, [this]() { user_editing_pose_ = false; });

    ui_->spinBoxPosX->installEventFilter(this);
    ui_->spinBoxPosY->installEventFilter(this);
    ui_->spinBoxPosZ->installEventFilter(this);
    ui_->spinBoxRoll->installEventFilter(this);
    ui_->spinBoxPitch->installEventFilter(this);
    ui_->spinBoxYaw->installEventFilter(this);

    // Timer to update marker pose display
    QTimer* poseUpdateTimer = new QTimer(this);
    connect(poseUpdateTimer, &QTimer::timeout, this, &RvizArgsPanel::updateMarkerPoseDisplay);
    poseUpdateTimer->start(100); // Update pose display every 100ms

    // Timer to find the TargetDisplay (added to RViz as a display, not owned
    // by this panel)
    QTimer* findDisplayTimer = new QTimer(this);
    connect(findDisplayTimer, &QTimer::timeout, this, &RvizArgsPanel::findTargetDisplay);
    findDisplayTimer->start(500); // Check every 500ms until found

    // Connect planner node / planner type controls
    connect(ui_->comboBoxPlannerNode, &QComboBox::currentTextChanged, this, &RvizArgsPanel::on_comboBoxPlannerNode_currentTextChanged);
    connect(ui_->comboBoxTrajectoryType, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RvizArgsPanel::on_comboBoxTrajectoryType_currentIndexChanged);

    // Populate the Planner Node dropdown from the live graph (the data `ros2
    // node list` reports) instead of a hardcoded list, and keep it refreshed so
    // a planner (re)started later appears automatically.
    refreshPlannerNodeList();
    QTimer* nodeListTimer = new QTimer(this);
    connect(nodeListTimer, &QTimer::timeout, this, &RvizArgsPanel::refreshPlannerNodeList);
    nodeListTimer->start(2000);

    // Connect stop robot button
    connect(ui_->stopRobot, &QPushButton::clicked, this, &RvizArgsPanel::on_stopRobot_clicked);

    // NOTE: the panel owns the planner-facing ROS plumbing (services, action,
    // and the 10 Hz mpc_goal publisher). The TargetDisplay is a generic draggable
    // 6-DOF marker with no planner knowledge; MPC live-tracking is started from
    // the "Generate and send" button while "MPC (Real-time)" is selected.

    // Spin node_ on a background thread for the rest of this panel's lifetime, so
    // every async_send_request/AsyncParametersClient callback below actually gets
    // delivered. Constructed last: by now every client/publisher/subscription this
    // node will ever need already exists.
    spinner_ = std::make_unique<NodeSpinner>(node_);
  }

  RvizArgsPanel::~RvizArgsPanel()
  {
    // spinner_ is declared last in the header, so it is destroyed FIRST here --
    // cancel()+join() completes (see NodeSpinner) before any client/publisher/node
    // it might still be invoking callbacks against is torn down.
  }

  void RvizArgsPanel::createPlannerClients(const std::string & planner_node)
  {
    const std::string planner_ns = "/" + planner_node + "/";

    // AsyncParametersClient -- never SyncParametersClient, which builds its own
    // temporary executor per call and would fight the background NodeSpinner (see
    // spinner_ below) for ownership of node_.
    param_client_ = std::make_shared<rclcpp::AsyncParametersClient>(node_, planner_node);

    // action client
    this->action_ptr_ = rclcpp_action::create_client<isaac_ros_cumotion_interfaces::action::SendTrajectory>(
      node_,
      planner_ns + "execute_trajectory");

    // create service client to generate traj
    this->trajectory_generation_client_ = node_->create_client<isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration>(planner_ns + "generate_trajectory");

    // Create service client for setting planner type
    this->set_planner_client_ = node_->create_client<isaac_ros_cumotion_interfaces::srv::SetPlanner>(planner_ns + "set_planner");

    // Publisher streaming the live MPC goal towards the current target pose.
    this->mpc_goal_pub_ = node_->create_publisher<geometry_msgs::msg::Pose>(planner_ns + "mpc_goal", 10);

    RCLCPP_INFO(node_->get_logger(),
      "RvizArgsPanel clients bound to planner node '%s' (%s)",
      planner_node.c_str(), planner_ns.c_str());
  }

  void RvizArgsPanel::runOnGuiThread(std::function<void()> fn)
  {
    QMetaObject::invokeMethod(this, std::move(fn), Qt::QueuedConnection);
  }

  void RvizArgsPanel::setPlannerReady(bool ready)
  {
    if (planner_ready_ == ready) {
      return;
    }
    planner_ready_ = ready;

    ui_->doubleSpinBoxTimeDilationFactor->setEnabled(ready);
    ui_->comboBoxTrajectoryType->setEnabled(ready);
    // comboBoxPlannerNode stays live: repointing the dropdown at another planner
    // node is always allowed, it just re-runs the readiness probe.
    updateActionButtons();

    RCLCPP_INFO(node_->get_logger(), "Planner is %s", ready ? "ready" : "not ready");
  }

  void RvizArgsPanel::updateActionButtons()
  {
    const bool can_start = planner_ready_ && !goal_active_;
    ui_->generateTrajectory->setEnabled(can_start);
    ui_->sendTrajectory->setEnabled(can_start);
    ui_->generateAndSend->setEnabled(can_start);
    ui_->stopRobot->setEnabled(goal_active_);
  }

  void RvizArgsPanel::pollPlannerReady()
  {
    // Watchdog for wedged probes: rclcpp's async get_parameters has no timeout --
    // if the planner never answers (still busy finishing an execution, GPU-bound
    // on an MPC sweep, or warming up again after a respawn) the completion
    // callback is simply never fired and planner_poll_in_flight_ would stay true
    // forever, freezing planner_ready_ (and with it every Generate/Execute
    // button, even after the execution is clearly done). Force-expire an
    // outstanding probe after 2 s and send a fresh one; completions from a
    // superseded probe are discarded via the sequence counter.
    if (planner_poll_in_flight_) {
      const auto outstanding_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - planner_poll_sent_at_).count();
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
    param_client_->get_parameters({"node_is_available"},
      [this, seq](std::shared_future<std::vector<rclcpp::Parameter>> future) {
        bool ready = false;
        try {
          auto params = future.get();
          ready = !params.empty() && params[0].as_bool();
        } catch (const std::exception & e) {
          RCLCPP_WARN(node_->get_logger(), "node_is_available check failed: %s", e.what());
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

    void RvizArgsPanel::load(const rviz_common::Config &config)
    {
      Panel::load(config);

      // Load values from RViz config file
      float time_dilation_factor;
      if (config.mapGetFloat("time_dilation_factor", &time_dilation_factor)) {
        time_dilation_factor_ = time_dilation_factor;
        ui_->doubleSpinBoxTimeDilationFactor->setValue(time_dilation_factor);
      }

      QString planner_node;
      if (config.mapGetString("planner_node_name", &planner_node)) {
        // Rebinds the panel's clients to the stored planner node via the
        // currentTextChanged slot.
        ui_->comboBoxPlannerNode->setCurrentText(planner_node);
      }
    }

    void RvizArgsPanel::save(rviz_common::Config config) const
    {
      Panel::save(config);
      config.mapSetValue("time_dilation_factor", time_dilation_factor_);
      config.mapSetValue("planner_node_name", QString::fromStdString(planner_node_));
    }


    void RvizArgsPanel::updateTimeDilationFactor(double value)
    {
        time_dilation_factor_ = value;

        // This slot fires during RViz config load (spinbox setValue -> valueChanged),
        // so it must never block: set_parameters_atomically is called with a callback
        // (AsyncParametersClient), never the SyncParametersClient/blocking form.
        if (!param_client_ || !param_client_->service_is_ready()) {
            RCLCPP_WARN(node_->get_logger(),
                "Planner parameter service not available; skipping time_dilation_factor update");
            return;
        }
        param_client_->set_parameters_atomically(
          {rclcpp::Parameter("time_dilation_factor", time_dilation_factor_)},
          [this, value](std::shared_future<rcl_interfaces::msg::SetParametersResult> future) {
            try {
              auto result = future.get();
              if (result.successful) {
                RCLCPP_INFO(node_->get_logger(), "Time dilation factor set to %.2f", value);
              } else {
                RCLCPP_ERROR(node_->get_logger(), "Failed to set time_dilation_factor: %s",
                    result.reason.c_str());
              }
            } catch (const std::exception & e) {
              RCLCPP_ERROR(node_->get_logger(), "Exception setting time_dilation_factor: %s", e.what());
            }
          });
    }

    void RvizArgsPanel::on_sendTrajectory_clicked(){
      auto goal_request = isaac_ros_cumotion_interfaces::action::SendTrajectory::Goal();

      // REQUIRED for the open-loop path, and it used to be missing: the goal
      // went out default-constructed, i.e. an empty goalsets entry -- the
      // dsr01/world origin, inside the robot's own base.
      //
      // The panel relied on the server reusing the plan the "generate" button
      // had just cached (allow_cached defaults to true). But that reuse is
      // gated by _pending_plan_matches(), whose signature INCLUDES goalsets
      // and compares positions to 1 mm (unified_planner_node.py
      // _target_signature/_poses_match). Cached signature = the real marker
      // pose, incoming goal = the origin -> guaranteed mismatch, so the server
      // silently re-planned toward the robot's own base, burned its 10
      // max_attempts, failed, and aborted the goal without logging anything.
      // Net effect: "generate" returned a trajectory, "execute" did nothing.
      //
      // MPC never showed this because the reactive path gets its target from
      // the /<planner>/mpc_goal topic (streamed at 10 Hz by this panel while
      // MPC is active), which overrides the empty goalset. The open-loop path
      // has no such second source -- the cache was its only route, and it was
      // unreachable.
      if (target_display_) {
        auto target_pose = target_display_->getPose();
        isaac_ros_cumotion_interfaces::msg::Goalset gset;
        gset.poses.push_back(target_pose);
        goal_request.goalsets.push_back(gset);
      } else {
        RCLCPP_WARN(node_->get_logger(),
                    "Target display not available - goal sent WITHOUT a target "
                    "pose (open-loop planning will fail; add "
                    "TargetDisplay to RViz)");
      }

      auto send_goal_options = rclcpp_action::Client<isaac_ros_cumotion_interfaces::action::SendTrajectory>::SendGoalOptions();

      send_goal_options.goal_response_callback = std::bind(&RvizArgsPanel::goal_response_callback, this, std::placeholders::_1);
      send_goal_options.result_callback = std::bind(&RvizArgsPanel::result_callback, this, std::placeholders::_1);

      action_ptr_->async_send_goal(goal_request, send_goal_options);
      goal_active_ = true;
      updateActionButtons();
    }

    // goal_response_callback/result_callback are rclcpp_action client callbacks --
    // they now fire on the background spin thread (spinner_), so every UI/member
    // touch inside them is marshaled back via runOnGuiThread. Only pure logging
    // (thread-safe) stays outside.
    void RvizArgsPanel::result_callback(const rclcpp_action::ClientGoalHandle<isaac_ros_cumotion_interfaces::action::SendTrajectory>::WrappedResult & result){
      switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
          RCLCPP_ERROR(node_->get_logger(), "Goal was succeeded");
          break;
        case rclcpp_action::ResultCode::ABORTED:
          RCLCPP_ERROR(node_->get_logger(), "Goal was aborted");
          break;
        case rclcpp_action::ResultCode::CANCELED:
          RCLCPP_ERROR(node_->get_logger(), "Goal was canceled");
          break;
        default:
          RCLCPP_ERROR(node_->get_logger(), "Unknown result code");
          break;
      }

      runOnGuiThread([this]() {
        mpc_goal_timer_->stop();
        mpc_active_ = false;
        mpc_starting_ = false;
        goal_handle_.reset();
        goal_active_ = false;

        // The execution just finished; force-out a readiness probe that may be
        // stranded from before (sent while the planner was busy) and re-probe
        // immediately, so the Generate/Execute buttons recover as soon as the
        // action completes rather than waiting out the poll watchdog.
        if (planner_poll_in_flight_) {
          planner_poll_in_flight_ = false;
        }
        pollPlannerReady();
        updateActionButtons();
      });
  }

  void RvizArgsPanel::goal_response_callback(std::shared_ptr<rclcpp_action::ClientGoalHandle<isaac_ros_cumotion_interfaces::action::SendTrajectory>> goal_handle){
    const bool accepted = static_cast<bool>(goal_handle);
    if (!accepted) {
      RCLCPP_ERROR(node_->get_logger(), "Goal was rejected by server");
    } else {
      RCLCPP_INFO(node_->get_logger(), "Goal accepted by server, waiting for result");
    }
    runOnGuiThread([this, goal_handle, accepted]() {
      this->goal_handle_ = goal_handle;
      if (accepted && mpc_starting_) {
        // MPC goal accepted: begin streaming the live target pose (~10 Hz).
        mpc_starting_ = false;
        mpc_active_ = true;
        mpc_goal_timer_->start();
        RCLCPP_INFO(node_->get_logger(), "MPC active - drag the target to retarget the robot live");
      } else if (!accepted) {
        mpc_starting_ = false;
      }
    });
  }

    void RvizArgsPanel::on_generateTrajectory_clicked(){
      generateTrajectoryAsync(nullptr);
    }

    // Shared by on_generateTrajectory_clicked (fire-and-forget) and the classic-mode
    // branch of on_generateAndSend_clicked (chains into on_sendTrajectory_clicked).
    // Pure async_send_request + callback -- no spin_until_future_complete, so this
    // can never block the GUI thread waiting on the planner.
    void RvizArgsPanel::generateTrajectoryAsync(std::function<void(bool)> on_done){
      // Always called from the GUI thread (button slots), so early-return failure
      // paths can invoke on_done() directly; only the async completion below
      // (which fires on the background spin thread) needs runOnGuiThread.
      if (!target_display_) {
        RCLCPP_WARN(node_->get_logger(), "Target display not available yet. Please add TargetDisplay to RViz.");
        if (on_done) { on_done(false); }
        return;
      }
      if (!trajectory_generation_client_->service_is_ready()) {
        RCLCPP_ERROR(node_->get_logger(), "generate_trajectory service not available");
        if (on_done) { on_done(false); }
        return;
      }

      auto goal_request = std::make_shared<isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration::Request>();
      isaac_ros_cumotion_interfaces::msg::Goalset gset;
      gset.poses.push_back(this->target_display_->getPose());
      goal_request->goalsets.push_back(gset);

      trajectory_generation_client_->async_send_request(goal_request,
        [this, on_done](rclcpp::Client<isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration>::SharedFuture future) {
          bool success = false;
          std::string message;
          try {
            auto response = future.get();
            success = response->success;
            message = response->message;
          } catch (const std::exception & e) {
            message = e.what();
          }
          if (success) {
            RCLCPP_INFO(node_->get_logger(), "generate_trajectory succeeded");
          } else {
            RCLCPP_ERROR(node_->get_logger(), "generate_trajectory failed: %s", message.c_str());
          }
          if (on_done) {
            runOnGuiThread([on_done, success]() { on_done(success); });
          }
        });
    }

    void RvizArgsPanel::on_generateAndSend_clicked(){
      // Check if MPC planner is selected (planner_type == 1)
      if (current_planner_type_ == 1) {
        // MPC Mode: the panel owns the reactive flow (SetPlanner MPC ->
        // execute_trajectory action -> live mpc_goal streaming while the gizmo
        // is dragged).
        if (!target_display_) {
          RCLCPP_WARN(node_->get_logger(), "Target display not available; add TargetDisplay to RViz to start MPC tracking");
          return;
        }
        RCLCPP_INFO(node_->get_logger(), "Starting MPC tracking");
        startMpc();
        return;
      }

      // Classic Mode: generate, then chain into send once generation actually
      // succeeds (previously a QTimer::singleShot(500, ...) guess resting on
      // on_generateTrajectory_clicked's old blocking behavior -- now a real
      // completion callback, since that call is fully async).
      RCLCPP_INFO(node_->get_logger(), "Classic mode: generate and execute once");
      generateTrajectoryAsync([this](bool success) {
        if (success) {
          on_sendTrajectory_clicked();
        } else {
          RCLCPP_ERROR(node_->get_logger(), "Trajectory generation failed; not sending");
        }
      });
    }

    // --- MPC live-tracking (owned by the panel) ---

    void RvizArgsPanel::startMpc()
    {
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
        RCLCPP_WARN(node_->get_logger(), "execute_trajectory action server not available");
        return;
      }

      // The planner must be on MPC for the reactive loop to consume mpc_goal.
      auto request = std::make_shared<isaac_ros_cumotion_interfaces::srv::SetPlanner::Request>();
      request->planner_type = isaac_ros_cumotion_interfaces::srv::SetPlanner::Request::MPC;
      mpc_starting_ = true;

      set_planner_client_->async_send_request(request,
        [this](rclcpp::Client<isaac_ros_cumotion_interfaces::srv::SetPlanner>::SharedFuture future) {
          bool success = false;
          std::string message;
          try {
            auto response = future.get();
            success = response->success;
            message = response->message;
          } catch (const std::exception & e) {
            message = e.what();
          }

          if (!success) {
            RCLCPP_ERROR(node_->get_logger(), "Planner switch to MPC failed: %s", message.c_str());
            runOnGuiThread([this]() { mpc_starting_ = false; });
            return;
          }
          RCLCPP_INFO(node_->get_logger(), "Switched planner to MPC");
          runOnGuiThread([this]() { sendMpcGoal(); });
        });
    }

    void RvizArgsPanel::sendMpcGoal()
    {
      if (!target_display_) {
        RCLCPP_WARN(node_->get_logger(), "Target display not available; cannot start MPC goal");
        return;
      }
      if (!action_ptr_ || !action_ptr_->action_server_is_ready()) {
        RCLCPP_WARN(node_->get_logger(), "execute_trajectory action server not available yet");
        return;
      }

      auto goal = isaac_ros_cumotion_interfaces::action::SendTrajectory::Goal();
      goal.allow_cached = false;

      // start_pose intentionally left EMPTY so the server resolves the start state
      // from robot_context.get_joint_pose() (the live, model-sized joint vector).
      // The legacy code sent raw /joint_states here, which carries extra joints
      // (e.g. a gripper) -- cuRobo's MPC rejects the oversized state
      // ("current_state must have 7 columns, got 8"). The server's own resolution
      // is both the right size and the real current pose.

      isaac_ros_cumotion_interfaces::msg::Goalset gset;
      gset.poses.push_back(target_display_->getPose());
      goal.goalsets.push_back(gset);

      auto send_goal_options = rclcpp_action::Client<isaac_ros_cumotion_interfaces::action::SendTrajectory>::SendGoalOptions();
      send_goal_options.goal_response_callback = std::bind(&RvizArgsPanel::goal_response_callback, this, std::placeholders::_1);
      send_goal_options.result_callback = std::bind(&RvizArgsPanel::result_callback, this, std::placeholders::_1);

      // Arm the MPC-start latch before sending: goal_response_callback consumes
      // it on acceptance (`accepted && mpc_starting_`) to begin the live target
      // stream. Clearing it here is the bug that silently killed MPC feature --
      // the acceptance branch never fired, mpc_active_ stayed false, and every
      // subsequent gizmo drag was ignored (MPC executed once toward the fixed
      // start pose).
      mpc_starting_ = true;
      action_ptr_->async_send_goal(goal, send_goal_options);

      // await goal acceptance (goal_response_callback) before starting the stream
      last_published_goal_ = target_display_->getPose();
      goal_active_ = true;
      updateActionButtons();
    }

    void RvizArgsPanel::streamMpcGoal()
    {
      if (!mpc_active_ || !mpc_goal_pub_ || !target_display_) {
        return;
      }
      const geometry_msgs::msg::Pose & p = target_display_->getPose();
      const geometry_msgs::msg::Pose & l = last_published_goal_;
      const bool changed =
        p.position.x != l.position.x ||
        p.position.y != l.position.y ||
        p.position.z != l.position.z ||
        p.orientation.x != l.orientation.x ||
        p.orientation.y != l.orientation.y ||
        p.orientation.z != l.orientation.z ||
        p.orientation.w != l.orientation.w;
      if (changed) {
        mpc_goal_pub_->publish(p);
        last_published_goal_ = p;
      }
    }

    void RvizArgsPanel::stopMpc()
    {
      mpc_starting_ = false;
      if (mpc_goal_timer_) {
        mpc_goal_timer_->stop();
      }
      if (mpc_active_ && goal_handle_) {
        try {
          action_ptr_->async_cancel_goal(goal_handle_);
        } catch (const std::exception& e) {
          RCLCPP_WARN(node_->get_logger(), "Exception during MPC cancel: %s", e.what());
        }
      }
      mpc_active_ = false;
    }

    void RvizArgsPanel::on_stopRobot_clicked(){
      // Stop live MPC tracking if active
      stopMpc();

      // Cancel the action goal if it exists and is still active
      if (goal_handle_) {
        try {
          // Use async_cancel_goal without blocking spin
          // The result_callback will handle the cleanup
          action_ptr_->async_cancel_goal(goal_handle_);
          RCLCPP_INFO(node_->get_logger(), "Cancel request sent");
        } catch (const std::exception& e) {
          RCLCPP_WARN(node_->get_logger(), "Exception during cancel: %s", e.what());
        }
      }

      // Re-enable buttons immediately (don't wait for result)
      goal_active_ = false;
      updateActionButtons();
    }

    void RvizArgsPanel::on_comboBoxPlannerNode_currentTextChanged(const QString &text)
    {
      const std::string new_node = text.trimmed().toStdString();
      if (new_node.empty()) {
        ui_->comboBoxPlannerNode->setCurrentText(QString::fromStdString(planner_node_));
        return;
      }
      if (new_node == planner_node_) {
        return;
      }

      RCLCPP_INFO(node_->get_logger(), "Switching planner node to '%s'", new_node.c_str());

      // Tear down any in-flight goal / MPC loop before repointing the clients.
      stopMpc();
      if (goal_active_ && goal_handle_) {
        try {
          action_ptr_->async_cancel_goal(goal_handle_);
        } catch (const std::exception& e) {
          RCLCPP_WARN(node_->get_logger(), "Exception during cancel: %s", e.what());
        }
      }
      goal_active_ = false;
      goal_handle_.reset();

      planner_node_ = new_node;
      createPlannerClients(new_node);

      // The new planner node needs re-probing for readiness (it may still be
      // warming up or simply not responding yet).
      setPlannerReady(false);
    }

    void RvizArgsPanel::refreshPlannerNodeList()
    {
      // Same data `ros2 node list` reports, queried here via the node graph API
      // (no hardcoded dropdown values). Raw names come back "/"-prefixed.
      std::vector<std::string> names;
      try {
        for (auto n : node_->get_node_names()) {
          if (!n.empty() && n.front() == '/') {
            n.erase(n.begin());
          }
          names.push_back(n);
        }
      } catch (const std::exception & e) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
          "Failed to list ROS nodes: %s", e.what());
        return;
      }

      const std::set<std::string> seen(names.begin(), names.end());
      if (seen == last_planner_nodes_) {
        return;  // graph unchanged; don't churn the dropdown mid-interaction
      }
      last_planner_nodes_ = seen;

      // Rebuild the items, preserving whatever the user currently has in the
      // (editable) combo — even if it is not (yet) a live node.
      const QString previous = ui_->comboBoxPlannerNode->currentText();
      ui_->comboBoxPlannerNode->clear();
      std::sort(names.begin(), names.end());
      for (const auto & n : names) {
        ui_->comboBoxPlannerNode->addItem(QString::fromStdString(n));
      }
      ui_->comboBoxPlannerNode->setCurrentText(previous);
    }

    void RvizArgsPanel::findTargetDisplay(){
      // If already found, stop searching
      if (target_display_ != nullptr) {
        return;
      }

      // getDisplayContext() is the public method to access context
      auto context = getDisplayContext();
      if (!context) {
        return;
      }

      // Get the root display group directly from context
      auto root_display = context->getRootDisplayGroup();
      if (!root_display) {
        return;
      }

      // Search for the TargetDisplay, descending into display Groups: the
      // shipped rviz configs (<robot>_curobo.rviz) place TargetDisplay inside a
      // "Curobo Planning" group, so a scan of only root-level displays misses it
      // whenever RViz reloads the config.
      target_display_ = findTargetDisplayInGroup(root_display);
      if (target_display_ != nullptr) {
        RCLCPP_INFO(node_->get_logger(), "Found TargetDisplay, using its target");
      }
    }

    TargetDisplay * RvizArgsPanel::findTargetDisplayInGroup(rviz_common::DisplayGroup * group)
    {
      for (int i = 0; i < group->numDisplays(); ++i) {
        rviz_common::Display * child = group->getDisplayAt(i);
        if (child == nullptr) {
          continue;
        }
        TargetDisplay * target = dynamic_cast<TargetDisplay *>(child);
        if (target != nullptr) {
          return target;
        }
        // A DisplayGroup derives from Display, so subgroups appear as children
        // of a group and must be descended into as well.
        rviz_common::DisplayGroup * subgroup =
          dynamic_cast<rviz_common::DisplayGroup *>(child);
        if (subgroup != nullptr) {
          TargetDisplay * found = findTargetDisplayInGroup(subgroup);
          if (found != nullptr) {
            return found;
          }
        }
      }
      return nullptr;
    }

    void RvizArgsPanel::quaternionToEuler(const geometry_msgs::msg::Quaternion& q, double& roll, double& pitch, double& yaw) {
      // Convert quaternion to Euler angles (roll, pitch, yaw) in degrees
      // Roll (x-axis rotation)
      double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
      double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
      roll = std::atan2(sinr_cosp, cosr_cosp) * 180.0 / M_PI;

      // Pitch (y-axis rotation)
      double sinp = 2.0 * (q.w * q.y - q.z * q.x);
      if (std::abs(sinp) >= 1)
        pitch = std::copysign(90.0, sinp); // use 90 degrees if out of range
      else
        pitch = std::asin(sinp) * 180.0 / M_PI;

      // Yaw (z-axis rotation)
      double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
      double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
      yaw = std::atan2(siny_cosp, cosy_cosp) * 180.0 / M_PI;
    }

    void RvizArgsPanel::eulerToQuaternion(double roll, double pitch, double yaw, geometry_msgs::msg::Quaternion& q) {
      // Convert Euler angles (in degrees) to quaternion
      double roll_rad = roll * M_PI / 180.0;
      double pitch_rad = pitch * M_PI / 180.0;
      double yaw_rad = yaw * M_PI / 180.0;

      double cy = std::cos(yaw_rad * 0.5);
      double sy = std::sin(yaw_rad * 0.5);
      double cp = std::cos(pitch_rad * 0.5);
      double sp = std::sin(pitch_rad * 0.5);
      double cr = std::cos(roll_rad * 0.5);
      double sr = std::sin(roll_rad * 0.5);

      q.w = cr * cp * cy + sr * sp * sy;
      q.x = sr * cp * cy - cr * sp * sy;
      q.y = cr * sp * cy + sr * cp * sy;
      q.z = cr * cp * sy - sr * sp * cy;
    }

    void RvizArgsPanel::applyPoseFromSpinboxes(){
      if (!target_display_) {
        RCLCPP_WARN(node_->get_logger(), "Target display not available yet");
        return;
      }

      // Get values from spinboxes
      geometry_msgs::msg::Pose pose;
      pose.position.x = ui_->spinBoxPosX->value();
      pose.position.y = ui_->spinBoxPosY->value();
      pose.position.z = ui_->spinBoxPosZ->value();

      // Get orientation from spinboxes and convert to quaternion
      double roll = ui_->spinBoxRoll->value();
      double pitch = ui_->spinBoxPitch->value();
      double yaw = ui_->spinBoxYaw->value();
      eulerToQuaternion(roll, pitch, yaw, pose.orientation);

      // Apply the new pose to the marker
      target_display_->setPose(pose);

      // Update last displayed values to match what we just set
      last_displayed_x_ = pose.position.x;
      last_displayed_y_ = pose.position.y;
      last_displayed_z_ = pose.position.z;
      last_displayed_roll_ = roll;
      last_displayed_pitch_ = pitch;
      last_displayed_yaw_ = yaw;

      RCLCPP_INFO(node_->get_logger(), "Applied pose: X=%.3f, Y=%.3f, Z=%.3f, Roll=%.2f, Pitch=%.2f, Yaw=%.2f",
                  pose.position.x, pose.position.y, pose.position.z, roll, pitch, yaw);
    }

    bool RvizArgsPanel::eventFilter(QObject *obj, QEvent *event) {
      // Check if the event is a FocusIn event on one of the pose spinboxes
      if (event->type() == QEvent::FocusIn) {
        if (obj == ui_->spinBoxPosX || obj == ui_->spinBoxPosY || obj == ui_->spinBoxPosZ ||
            obj == ui_->spinBoxRoll || obj == ui_->spinBoxPitch || obj == ui_->spinBoxYaw) {
          user_editing_pose_ = true;
        }
      }
      // Pass the event to the base class
      return QObject::eventFilter(obj, event);
    }

    void RvizArgsPanel::updateMarkerPoseDisplay(){
      // Don't update if marker not found yet or if user is editing
      if (!target_display_ || user_editing_pose_) {
        return;
      }

      auto pose = target_display_->getPose();

      // Convert quaternion to Euler angles
      double roll, pitch, yaw;
      quaternionToEuler(pose.orientation, roll, pitch, yaw);

      // Compare with last displayed values - only update if changed
      constexpr double epsilon_pos = 1e-6; // Small threshold for position
      constexpr double epsilon_rot = 0.01; // Small threshold for rotation (degrees)

      bool x_changed = std::isnan(last_displayed_x_) || std::fabs(pose.position.x - last_displayed_x_) > epsilon_pos;
      bool y_changed = std::isnan(last_displayed_y_) || std::fabs(pose.position.y - last_displayed_y_) > epsilon_pos;
      bool z_changed = std::isnan(last_displayed_z_) || std::fabs(pose.position.z - last_displayed_z_) > epsilon_pos;
      bool roll_changed = std::isnan(last_displayed_roll_) || std::fabs(roll - last_displayed_roll_) > epsilon_rot;
      bool pitch_changed = std::isnan(last_displayed_pitch_) || std::fabs(pitch - last_displayed_pitch_) > epsilon_rot;
      bool yaw_changed = std::isnan(last_displayed_yaw_) || std::fabs(yaw - last_displayed_yaw_) > epsilon_rot;

      // Only update if at least one value has changed
      if (!x_changed && !y_changed && !z_changed && !roll_changed && !pitch_changed && !yaw_changed) {
        return;
      }

      // Update the spinboxes with current pose
      // Block signals to avoid triggering updates while we're setting values
      ui_->spinBoxPosX->blockSignals(true);
      ui_->spinBoxPosY->blockSignals(true);
      ui_->spinBoxPosZ->blockSignals(true);
      ui_->spinBoxRoll->blockSignals(true);
      ui_->spinBoxPitch->blockSignals(true);
      ui_->spinBoxYaw->blockSignals(true);

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
      if (roll_changed) {
        ui_->spinBoxRoll->setValue(roll);
        last_displayed_roll_ = roll;
      }
      if (pitch_changed) {
        ui_->spinBoxPitch->setValue(pitch);
        last_displayed_pitch_ = pitch;
      }
      if (yaw_changed) {
        ui_->spinBoxYaw->setValue(yaw);
        last_displayed_yaw_ = yaw;
      }

      ui_->spinBoxPosX->blockSignals(false);
      ui_->spinBoxPosY->blockSignals(false);
      ui_->spinBoxPosZ->blockSignals(false);
      ui_->spinBoxRoll->blockSignals(false);
      ui_->spinBoxPitch->blockSignals(false);
      ui_->spinBoxYaw->blockSignals(false);
    }

    void RvizArgsPanel::on_comboBoxTrajectoryType_currentIndexChanged(int index) {
      RCLCPP_INFO(node_->get_logger(), "Planner type changed to index: %d", index);

      // Map index to planner type constant
      uint8_t planner_type = static_cast<uint8_t>(index);

      // Verify planner type is valid
      if (planner_type > 3) {
        RCLCPP_ERROR(node_->get_logger(), "Invalid planner type: %d", planner_type);
        return;
      }

      // Map planner type to name for logging
      std::string planner_name;
      switch (planner_type) {
        case 0:
          planner_name = "CLASSIC";
          break;
        case 1:
          planner_name = "MPC";
          break;
        case 2:
          planner_name = "BATCH";
          break;
        case 3:
          planner_name = "CONSTRAINED";
          break;
      }

      RCLCPP_INFO(node_->get_logger(), "Switching to planner: %s", planner_name.c_str());

      // Call the set_planner service
      if (!set_planner_client_ || !set_planner_client_->service_is_ready()) {
        RCLCPP_WARN(node_->get_logger(), "SetPlanner service not available");
        return;
      }

      auto request = std::make_shared<isaac_ros_cumotion_interfaces::srv::SetPlanner::Request>();
      request->planner_type = planner_type;

      set_planner_client_->async_send_request(request,
        [this, planner_type, planner_name](rclcpp::Client<isaac_ros_cumotion_interfaces::srv::SetPlanner>::SharedFuture future) {
          try {
            auto response = future.get();

            if (response->success) {
              RCLCPP_INFO(node_->get_logger(), "Successfully switched to planner: %s", planner_name.c_str());
              RCLCPP_INFO(node_->get_logger(), "Service response: %s", response->message.c_str());
              RCLCPP_INFO(node_->get_logger(), "Previous planner: %s, Current planner: %s",
                          response->previous_planner.c_str(), response->current_planner.c_str());
              runOnGuiThread([this, planner_type]() { current_planner_type_ = planner_type; });
            } else {
              RCLCPP_ERROR(node_->get_logger(), "Failed to switch planner: %s", response->message.c_str());
            }

          } catch (const std::exception& e) {
            RCLCPP_ERROR(node_->get_logger(), "Exception calling set_planner service: %s", e.what());
          }
        });
    }

} // isaac_ros_cumotion_rviz

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(isaac_ros_cumotion_rviz::RvizArgsPanel, rviz_common::Panel)