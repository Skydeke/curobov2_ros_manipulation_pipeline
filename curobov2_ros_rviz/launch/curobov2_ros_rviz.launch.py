import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    declare_planner_node_name = DeclareLaunchArgument(
        'planner_node_name',
        default_value='unified_planner',
        description=(
            'Planner node name the Curobo panel binds its clients to. Applied '
            'only if the running rviz node exposes it as a parameter; the value '
            'saved in the rviz config (Context tab -> planner_node_name) is what '
            'the panel actually uses, and it wins when both are present.'
        )
    )

    declare_base_link = DeclareLaunchArgument(
        'base_link',
        default_value='base_0',
        description='Base link frame name'
    )

    planner_node_name = LaunchConfiguration('planner_node_name')
    base_link = LaunchConfiguration('base_link')

    start_rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', os.path.join(get_package_share_directory('curobov2_ros_rviz'), 'rviz2', 'config.rviz')],
        parameters=[{
            'planner_node_name': planner_node_name,
            'base_link': base_link
        }]
    )

    ld = LaunchDescription()

    ld.add_action(declare_planner_node_name)
    ld.add_action(declare_base_link)

    ld.add_action(start_rviz2)

    return ld