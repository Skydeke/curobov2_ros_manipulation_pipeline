"""
UR10e-focused test launch file.

Forwards to the generic gen_traj_test.launch.py with the robot pinned to the
Universal Robots UR10e (the model resolves from the robots/ur10e.yaml
descriptor).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    isaac_ros_cumotion_launch_dir = os.path.join(
        get_package_share_directory('isaac_ros_cumotion'), 'launch')

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_config_file',
            default_value='',
            description='cuRobo config override (default: derived from robot descriptor)',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(isaac_ros_cumotion_launch_dir, 'gen_traj_test.launch.py')
            ),
            launch_arguments={
                'robot': 'ur10e',
                'robot_config_file': LaunchConfiguration('robot_config_file'),
            }.items()
        ),
    ])
