#pragma once
#include <iostream>
#include <functional>

// ROS2
#include <rclcpp/rclcpp.hpp>
#include "rclcpp_action/rclcpp_action.hpp"
#include <geometry_msgs/msg/pose.hpp>

// Projet
#include "isaac_ros_cumotion_rviz/target_display.hpp"
#include "isaac_ros_cumotion_rviz/node_spinner.hpp"
#include "isaac_ros_cumotion_interfaces/srv/trajectory_generation.hpp"
#include "isaac_ros_cumotion_interfaces/action/send_trajectory.hpp"
#include "isaac_ros_cumotion_interfaces/srv/set_planner.hpp"

// RVIZ2
#include <rviz_common/panel.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/visualization_manager.hpp>
#include <rviz_common/display_group.hpp>
#include <rviz_common/display.hpp>
// Qt
#include <QtWidgets>
// STL
#include <algorithm>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>
/** 
 *  Include header generated from ui file
 *  Note that you will need to use add_library function first
 *  in order to generate the header file from ui.
 */
#include <ui_isaac_ros_cumotion_rviz_panel.h>

namespace isaac_ros_cumotion_rviz
{
  class RvizArgsPanel : public rviz_common::Panel
  {
    Q_OBJECT
  public:
    explicit RvizArgsPanel(QWidget *parent = nullptr);
    ~RvizArgsPanel();

    /// Load and save configuration data
    virtual void load(const rviz_common::Config &config) override;
    virtual void save(rviz_common::Config config) const override;

    /// Event filter to detect when user starts editing pose spinboxes
    bool eventFilter(QObject *obj, QEvent *event) override;

  private Q_SLOTS:
    void updateTimeDilationFactor(double value);
    void on_sendTrajectory_clicked();
    void on_generateTrajectory_clicked();
    void on_generateAndSend_clicked();
    void on_stopRobot_clicked();
    void result_callback(const rclcpp_action::ClientGoalHandle<isaac_ros_cumotion_interfaces::action::SendTrajectory>::WrappedResult & result);
    void goal_response_callback(std::shared_ptr<rclcpp_action::ClientGoalHandle<isaac_ros_cumotion_interfaces::action::SendTrajectory>> goal_handle);
      // void goal_response_callback(std::shared_future<rclcpp_action::ClientGoalHandle<actionfaces::action::Fibonacci>::SharedPtr> future)

    // Marker control slots
    void updateMarkerPoseDisplay();
    void findTargetDisplay();
    void applyPoseFromSpinboxes();

    // Planner node slots
    void on_comboBoxPlannerNode_currentTextChanged(const QString &text);
    void refreshPlannerNodeList();

    // Planner type slots
    void on_comboBoxTrajectoryType_currentIndexChanged(int index);

    // Helper methods for quaternion <-> Euler conversion
    void quaternionToEuler(const geometry_msgs::msg::Quaternion& q, double& roll, double& pitch, double& yaw);
    void eulerToQuaternion(double roll, double pitch, double yaw, geometry_msgs::msg::Quaternion& q);

  private:
    // Runs fn on the Qt GUI thread. ROS callbacks fire on the background spin thread
    // (see spinner_ below) and must never touch QWidgets directly. Uses the
    // context-object overload of invokeMethod so Qt drops the call if `this` is
    // destroyed before the event loop gets to it.
    void runOnGuiThread(std::function<void()> fn);

    // Single source of truth for "is the planner actually responding" (not just
    // discoverable -- see node_is_available poll in the constructor/pollPlannerReady).
    // Gates every planner-facing widget so a click can never queue a request against
    // a not-yet-responsive planner (e.g. during its ~90s GPU warmup, or after it
    // respawns following a robot reboot).
    void setPlannerReady(bool ready);
    void pollPlannerReady();

    // Combines planner_ready_ with "is a goal currently executing" to drive
    // generateTrajectory/sendTrajectory/generateAndSend/stopRobot enabled state.
    void updateActionButtons();

    // Shared by on_generateTrajectory_clicked and the classic-mode branch of
    // on_generateAndSend_clicked (which used to chain onto a blocking generate call
    // via a fragile QTimer::singleShot(500, ...) guess -- now chains on the real
    // async completion instead).
    void generateTrajectoryAsync(std::function<void(bool)> on_done);

    // (Re)creates every planner-facing client/publisher against the given planner
    // node name. Called from the constructor and from on_comboBoxPlannerNode_*
    // when the user switches planner node.
    void createPlannerClients(const std::string & planner_node);

    // --- MPC live-tracking (owned by the panel; the TargetDisplay is generic) ---
    // Switch the planner to MPC, send an execute_trajectory goal toward the
    // current target and stream the target pose to /<planner>/mpc_goal at 10 Hz
    // while the gizmo is dragged.
    void startMpc();
    void stopMpc();
    void streamMpcGoal();

    void sendMpcGoal();

    std::unique_ptr<Ui::gui_parameters> ui_;
    rclcpp::Node::SharedPtr node_;
    rclcpp::AsyncParametersClient::SharedPtr param_client_;
    bool planner_ready_;
    bool planner_poll_in_flight_;
    bool goal_active_;
    rclcpp_action::Client<isaac_ros_cumotion_interfaces::action::SendTrajectory>::SharedPtr action_ptr_;
    rclcpp::Client<isaac_ros_cumotion_interfaces::srv::TrajectoryGeneration>::SharedPtr trajectory_generation_client_;
    rclcpp_action::Client<isaac_ros_cumotion_interfaces::action::SendTrajectory>::GoalHandle::SharedPtr goal_handle_;
    float time_dilation_factor_;
    // The TargetDisplay owning the draggable 6-DOF target (self-contained;
    // polling timer finds it lazily). Not owned by the panel.
    TargetDisplay* target_display_;

    // Depth-first search below `group` for a TargetDisplay. The shipped rviz
    // configs place it inside a display Group (e.g. "Curobo Planning"), which a
    // root-level scan misses — so the lookup must descend into subgroups.
    TargetDisplay* findTargetDisplayInGroup(rviz_common::DisplayGroup* group);
    bool user_editing_pose_; // Flag to prevent auto-update while user is editing

    // Last displayed pose to avoid unnecessary updates
    double last_displayed_x_;
    double last_displayed_y_;
    double last_displayed_z_;
    double last_displayed_roll_;
    double last_displayed_pitch_;
    double last_displayed_yaw_;

    // Planner node the panel's clients currently point at.
    std::string planner_node_;

    // Last node set shown in the dropdown, so refreshPlannerNodeList() only
    // rebuilds items when the graph actually changed.
    std::set<std::string> last_planner_nodes_;

    // MPC live-tracking members.
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr mpc_goal_pub_;
    bool mpc_active_;
    bool mpc_starting_;
    geometry_msgs::msg::Pose last_published_goal_;
    QTimer* mpc_goal_timer_;

    // Planner type members
    rclcpp::Client<isaac_ros_cumotion_interfaces::srv::SetPlanner>::SharedPtr set_planner_client_;
    uint8_t current_planner_type_;

    // Declared LAST so it is destroyed FIRST (members are torn down in reverse
    // declaration order): stops and joins the background spin thread before any
    // client/publisher/node it might still be delivering callbacks against is
    // destroyed.
    std::unique_ptr<NodeSpinner> spinner_;
  };
} // isaac_ros_cumotion_rviz