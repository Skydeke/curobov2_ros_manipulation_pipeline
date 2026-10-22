#!/usr/bin/env python3
"""Launch the curobo_task_constructor ExecuteTaskSolution server. Runs inside the kortex_moveit / kortex_cumotion container (NOT with the grasp
orchestrator): the same container that hosts curobo_server and move_group, so
the execute server can drive SendTrajectory, /joint_states and the planning-scene
services without cross-container networking. Planning is local (in-process,
like MoveIt); only execution goes over the wire."""

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