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
    if (target_display_->getEnabledTools().size() > 1) {
      RCLCPP_WARN(node_->get_logger(),
                  "MPC tracks a single tool frame; check exactly one entry in "
                  "the Planning tab's Target tools list");
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
  geometry_msgs::msg::Pose base_pose;
  if (target_display_->poseInBase(target_display_->getPose(), base_pose)) {
    gset.poses.push_back(base_pose);
    gset.frame_id = "";
  } else {
    // Base transform not available yet: let the server resolve the fixed frame.
    gset.poses.push_back(target_display_->getPose());
    gset.frame_id = goalFrame();
  }
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
    geometry_msgs::msg::Pose base_pose;
    if (target_display_->poseInBase(p, base_pose)) {
      stamped.pose = base_pose;
      stamped.header.frame_id = "";
    } else {
      stamped.pose = p;
      stamped.header.frame_id = goalFrame();
    }
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
    const auto targets = target_display_->getToolPosesInBase();
    curobov2_ros_interfaces::msg::Goalset gset;
    // getToolPosesInBase() already resolved each target into the planner's
    // base_link, so no server-side TF lookup is needed for the planning goal.
    gset.frame_id = "";
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
        gset.poses.push_back(targets.front().second);
      }
    }
    // Compatibility fallback: also carry the active pose as a single-tool
    // goalset. New servers prefer tool_frames/tool_poses; older servers (which
    // do not know those fields) still get a plan for the active tool.
    if (gset.poses.empty()) {
      geometry_msgs::msg::Pose base_pose;
      if (target_display_->poseInBase(target_display_->getPose(), base_pose)) {
        gset.poses.push_back(base_pose);
      } else {
        // No base transform yet: let the server resolve the fixed frame.
        gset.frame_id = goalFrame();
        gset.poses.push_back(target_display_->getPose());
      }
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

} // namespace curobov2_ros_rviz
