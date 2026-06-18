import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("isaac_ros_mapper")
    default_params = os.path.join(pkg, "config", "mapper_params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Path to mapper parameters YAML",
            ),
            DeclareLaunchArgument(
                "esdf_service_name",
                default_value="/curobo_mapper/get_esdf_and_gradient",
                description="ESDF service name",
            ),
            DeclareLaunchArgument(
                "robot_base_frame",
                default_value="base_link",
                description="Robot base frame for TF lookups",
            ),
            DeclareLaunchArgument(
                "depth_image_topics",
                default_value="['/camera_1/aligned_depth_to_color/image_raw']",
                description="Depth image topics",
            ),
            DeclareLaunchArgument(
                "depth_camera_info_topics",
                default_value="['/camera_1/color/camera_info']",
                description="Camera info topics",
            ),
            Node(
                package="isaac_ros_mapper",
                executable="mapper_node",
                name="curobo_mapper_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "esdf_service_name": LaunchConfiguration("esdf_service_name"),
                        "robot_base_frame": LaunchConfiguration("robot_base_frame"),
                        "depth_image_topics": LaunchConfiguration("depth_image_topics"),
                        "depth_camera_info_topics": LaunchConfiguration(
                            "depth_camera_info_topics"
                        ),
                    },
                ],
            ),
        ]
    )
