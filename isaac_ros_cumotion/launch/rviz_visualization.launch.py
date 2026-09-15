from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Start RViz. Included by gen_traj.launch.py when gui:=true.

    The config is passed through unchanged: per-robot configs (urf franka_curobo.rviz,
    ur10e_curobo.rviz) carry their own Fixed Frame, so no launch-time patching into
    a temp file is needed -- edits made in RViz persist back to the mounted file.
    """

    rviz_config = PathJoinSubstitution([
        FindPackageShare('isaac_ros_cumotion'),
        'rviz/rviz_curobo.rviz'
    ])

    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config',
        default_value=rviz_config,
        description='Path to the RViz config file (opened as-is)'
    )

    # Passed down by gen_traj.launch.py, which resolves it from the descriptor.
    # Forwarded to the RViz node as a ROS parameter (panel canvas convention).
    declare_base_link = DeclareLaunchArgument(
        'base_link',
        default_value='base_0',
        description='Robot root frame ID (forwarded to RViz as a ROS parameter)'
    )

    # Resolve the launch arguments at run time so the default rviz_config above
    # (a PathJoinSubstitution) is honored when no explicit value is given.
    def _rviz_node(context, *args, **kwargs):
        from launch.substitutions import LaunchConfiguration
        config_path = LaunchConfiguration('rviz_config').perform(context)
        base_link = LaunchConfiguration('base_link').perform(context)
        print(f"[rviz_visualization.launch] Opening RViz config: {config_path}")
        return [
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', config_path],
                parameters=[{'base_link': base_link}]
            )
        ]

    from launch.actions import OpaqueFunction
    return LaunchDescription([
        declare_rviz_config,
        declare_base_link,
        OpaqueFunction(function=_rviz_node)
    ])