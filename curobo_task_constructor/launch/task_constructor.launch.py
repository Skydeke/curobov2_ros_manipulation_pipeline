#!/usr/bin/env python3
"""Launch the curobo_task_constructor action server.

Runs inside the kortex_moveit / kortex_cumotion container (NOT with the grasp
orchestrator): the same container that hosts curobo_server and move_group, so
the task node can drive SendTrajectory, /joint_states and the planning-scene
services without cross-container networking.

Params mirror curobo_task_constructor.node.TaskConstructorNode:

- ``robot_config_path``  robot descriptor YAML (named_joint_configs section),
                          optional ("" = none).
- ``planner``            SetPlanner enum for the default planner (-1 = stage
                          params decide per move; the pick orchestrator drives
                          joint_space/classic per stage).
- ``joint_states_topic`` the robot's joint-state topic (/joint_states).
- ``service_timeout``    per-call budget for curobo_server services/actions
                          (default 60s; the first solve after a planner switch
                          re-records the ESDF CUDA graph and compiles kernels,
                          so keep this generous).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_config_path", default_value=""),
            DeclareLaunchArgument("planner", default_value="-1"),
            DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
            DeclareLaunchArgument("service_timeout", default_value="60.0"),
            Node(
                package="curobo_task_constructor",
                executable="curobo_task_constructor_node",
                name="curobo_task_constructor",
                output="screen",
                parameters=[
                    {
                        "robot_config_path": LaunchConfiguration("robot_config_path"),
                        # The node declares this as an int; convert so a CLI
                        # override (planner:=2) is applied instead of being
                        # rejected as a type mismatch. -1 = per-stage planner
                        # (the orchestrator drives joint_space/classic per move).
                        "planner": PythonExpression(
                            ["int(", LaunchConfiguration("planner"), ")"]
                        ),
                        "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                        "service_timeout": PythonExpression(
                            ["float(", LaunchConfiguration("service_timeout"), ")"]
                        ),
                    }
                ],
            ),
        ]
    )