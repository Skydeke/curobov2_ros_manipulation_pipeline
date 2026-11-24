# SPDX-License-Identifier: Apache-2.0
"""Standalone component_container_mt for a graspgenx_ros smoke test.

In the pipeline this component is loaded into the cuRobo container, after
cuRobo has created the process-wide embedded Python interpreter.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    checkpoint_dir = LaunchConfiguration("checkpoint_dir")
    gripper_name = LaunchConfiguration("gripper_name")
    gripper_assets_dir = LaunchConfiguration("gripper_assets_dir")
    objects_topic = LaunchConfiguration("objects_topic")
    frame_id = LaunchConfiguration("frame_id")

    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument("checkpoint_dir", default_value=""),
            DeclareLaunchArgument("gripper_name", default_value=""),
            DeclareLaunchArgument(
                "gripper_assets_dir", default_value="/opt/graspgenx/assets"
            ),
            DeclareLaunchArgument("objects_topic", default_value="objects"),
            DeclareLaunchArgument("frame_id", default_value="base_footprint"),
            ComposableNodeContainer(
                name="graspgenx_container",
                namespace=namespace,
                package="rclcpp_components",
                executable="component_container_mt",
                composable_node_descriptions=[
                    ComposableNode(
                        package="graspgenx_ros",
                        plugin="graspgenx_ros::GraspGenXComponent",
                        name="graspgenx",
                        parameters=[
                            {
                                "checkpoint_dir": checkpoint_dir,
                                "gripper_name": gripper_name,
                                "gripper_assets_dir": gripper_assets_dir,
                                "objects_topic": objects_topic,
                                "frame_id": frame_id,
                            }
                        ],
                        extra_arguments=[{"use_intra_process_comms": True}],
                    )
                ],
                output="screen",
            ),
        ]
    )
