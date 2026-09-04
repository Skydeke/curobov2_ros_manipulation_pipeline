# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Launch one or more cuRobo getting-started viser example nodes.

Each node hosts its own ViserVisualizer web GUI.  Set the ``nodes`` argument
to a comma-separated list of node names to launch (default: all eight).

Example — launch only FK and IK on port 8090:

    ros2 launch isaac_ros_cumotion_extra getting_started_viser.launch.py \
        nodes:=fk_viser_node,ik_viser_node viser_port:=8090
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ALL_NODES = [
    'fk_viser_node',
    'ik_viser_node',
    'mp_viser_node',
    'mpc_viser_node',
    'volumetric_viser_node',
    'feature_viser_node',
    'robot_model_viser_node',
    'retarget_viser_node',
]

PKG = 'isaac_ros_cumotion_extra'


def _launch_nodes(context, *args, **kwargs):
    nodes_str = LaunchConfiguration('nodes').perform(context)
    server_node = LaunchConfiguration('server_node').perform(context)
    viser_host = LaunchConfiguration('viser_host').perform(context)
    viser_port = int(LaunchConfiguration('viser_port').perform(context))
    robot_config = LaunchConfiguration('robot_config').perform(context)
    content_path = LaunchConfiguration('content_path').perform(context)
    urdf_path = LaunchConfiguration('urdf_path').perform(context)
    asset_path = LaunchConfiguration('asset_path').perform(context)
    add_robot = LaunchConfiguration('add_robot_to_scene').perform(context).lower() in ('true', '1', 'yes')
    add_frames = LaunchConfiguration('add_control_frames').perform(context).lower() in ('true', '1', 'yes')

    if nodes_str.strip():
        names = [n.strip() for n in nodes_str.split(',') if n.strip()]
    else:
        names = ALL_NODES

    actions = []
    for i, name in enumerate(names):
        port = viser_port + i
        params = {
            'server_node': server_node,
            'viser_host': viser_host,
            'viser_port': port,
            'robot_config': robot_config,
            'content_path': content_path,
            'urdf_path': urdf_path,
            'asset_path': asset_path,
            'add_robot_to_scene': add_robot,
            'add_control_frames': add_frames,
        }
        actions.append(
            Node(
                package=PKG,
                executable=name,
                name=name,
                output='screen',
                parameters=[params],
            )
        )
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'nodes',
            default_value='',
            description='Comma-separated list of viser nodes to launch (empty = all eight).',
        ),
        DeclareLaunchArgument(
            'server_node',
            default_value='curobo_server',
            description='Name of the curobo_server ROS node providing FK/IK/trajectory services.',
        ),
        DeclareLaunchArgument(
            'viser_host',
            default_value='0.0.0.0',
            description='IP address for the ViserVisualizer web server.',
        ),
        DeclareLaunchArgument(
            'viser_port',
            default_value='8010',
            description='Starting port for the ViserVisualizer web server (incremented per node).',
        ),
        DeclareLaunchArgument(
            'robot_config',
            default_value='franka.yml',
            description='cuRobo robot YAML config name.',
        ),
        DeclareLaunchArgument(
            'content_path',
            default_value='',
            description='Path to the robot config file — cuRobo YAML or XRDF (empty = use robot_config).',
        ),
        DeclareLaunchArgument(
            'urdf_path',
            default_value='',
            description='Path to the robot URDF file.',
        ),
        DeclareLaunchArgument(
            'asset_path',
            default_value='',
            description='Path to the robot mesh asset root.',
        ),
        DeclareLaunchArgument(
            'add_robot_to_scene',
            default_value='true',
            description='Render the robot mesh in the viser scene (requires content_path + urdf_path).',
        ),
        DeclareLaunchArgument(
            'add_control_frames',
            default_value='true',
            description='Add interactive drag control frames on the tool frames.',
        ),
        OpaqueFunction(function=_launch_nodes),
    ])
