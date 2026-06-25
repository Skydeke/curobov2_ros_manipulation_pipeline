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

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import SetParameter


def _get_kortex_params_path() -> str:
    """Resolve the path to kortex cumotion_params.yaml."""
    try:
        pkg = get_package_share_directory("iki_kortex_moveit_config")
        return os.path.join(pkg, "config", "cumotion_params.yaml")
    except Exception:
        return ""


def generate_launch_description():

    # The 'robot' argument can accept:
    # - XRDF filename
    # - YAML filename
    # - Absolute paths for XRDF or YAML files

    # Default URDF file path (currently absolute, should be updated as needed).
    default_urdf_file_path = os.path.join(
        get_package_share_directory("isaac_ros_cumotion_robot_description"),
        "urdf",
        "ur5e_robotiq_2f_140.urdf",
    )

    # Declare launch arguments with full paths
    launch_args = [
        DeclareLaunchArgument(
            "robot",
            default_value="ur5e_robotiq_2f_140.xrdf",
            description="Robot file (XRDF or YAML)",
        ),
        DeclareLaunchArgument(
            "urdf_file_path",
            default_value=default_urdf_file_path,
            description="Full path to the URDF file",
        ),
        DeclareLaunchArgument(
            "yml_file_path",
            default_value="",
            description="Path to the YAML file containing robot configurations",
        ),
        DeclareLaunchArgument(
            "joint_states_topic",
            default_value="/joint_states",
            description="Joint states topic",
        ),
        DeclareLaunchArgument(
            "depth_camera_info_topics",
            default_value="['/camera_1/color/camera_info']",
            description="Depth camera info topic",
        ),
        DeclareLaunchArgument(
            "depth_image_topics",
            default_value="['/camera_1/aligned_depth_to_color/image_raw']",
            description="Depth image topic for robot segmenter",
        ),
        DeclareLaunchArgument(
            "object_link_name",
            default_value="attached_object",
            description="Object link name for object attachment",
        ),
        DeclareLaunchArgument(
            "search_radius",
            default_value="0.2",
            description="Search radius for object attachment",
        ),
        DeclareLaunchArgument(
            "surface_sphere_radius",
            default_value="0.01",
            description="Radius for object surface collision spheres",
        ),
        DeclareLaunchArgument(
            "update_link_sphere_server_segmenter",
            default_value="segmenter_attach_object",
            description="Update link sphere server for robot segmenter",
        ),
        DeclareLaunchArgument(
            "update_link_sphere_server_planner",
            default_value="planner_attach_object",
            description="Update link sphere server for cumotion planner",
        ),
        DeclareLaunchArgument(
            "clustering_bypass",
            default_value="False",
            description="Whether to bypass clustering",
        ),
        DeclareLaunchArgument(
            "update_esdf_on_request",
            default_value="False",
            description="Whether object attachment should request an updated ESDF "
            "as part of the service call",
        ),
        DeclareLaunchArgument(
            "action_names",
            default_value="['segmenter_attach_object', 'planner_attach_object']",
            description="List of action names for the object attachment",
        ),
        DeclareLaunchArgument(
            "time_sync_slop",
            default_value="0.1",
            description="Time synchronization slop",
        ),
        DeclareLaunchArgument(
            "distance_threshold",
            default_value="0.02",
            description="Distance threshold for segmentation",
        ),
        DeclareLaunchArgument(
            "clustering_hdbscan_min_samples",
            default_value="20",
            description="HDBSCAN min samples for clustering",
        ),
        DeclareLaunchArgument(
            "clustering_hdbscan_min_cluster_size",
            default_value="30",
            description="HDBSCAN min cluster size for clustering",
        ),
        DeclareLaunchArgument(
            "clustering_hdbscan_cluster_selection_epsilon",
            default_value="0.5",
            description="HDBSCAN cluster selection epsilon",
        ),
        DeclareLaunchArgument(
            "clustering_num_top_clusters_to_select",
            default_value="3",
            description="Number of top clusters to select",
        ),
        DeclareLaunchArgument(
            "clustering_group_clusters",
            default_value="False",
            description="Whether to group clusters",
        ),
        DeclareLaunchArgument(
            "clustering_min_points",
            default_value="100",
            description="Minimum points for clustering",
        ),
        DeclareLaunchArgument(
            "object_attachment_gripper_frame_name",
            default_value="grasp_frame",
            description="Gripper frame name for object attachment",
        ),
        DeclareLaunchArgument(
            "enable_segmenter",
            default_value="false",
            description="Enable robot segmenter nodes",
        ),
        DeclareLaunchArgument(
            "enable_viser",
            default_value="true",
            description="Enable Viser visualization node",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="false", description="Use simulation time"
        ),
        # cumotion_planner parameters forwarded to isaac_ros_cumotion.launch.py
        DeclareLaunchArgument(
            "cumotion_planner.time_dilation_factor",
            default_value="0.5",
            description="Time dilation factor for cuMotion",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.max_attempts",
            default_value="10",
            description="Maximum planning attempts",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.num_graph_seeds",
            default_value="6",
            description="Number of graph seeds",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.num_trajopt_seeds",
            default_value="6",
            description="Number of trajectory optimization seeds",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.include_trajopt_retract_seed",
            default_value="True",
            description="Include trajectory optimization retract seed",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.num_trajopt_time_steps",
            default_value="32",
            description="Number of trajectory optimization time steps",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.joint_states_topic",
            default_value="/joint_states",
            description="Joint states topic",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.interpolation_dt",
            default_value="0.025",
            description="Interpolation delta time",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.collision_cache_cuboid",
            default_value="20",
            description="Collision cache cuboid size",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.collision_cache_mesh",
            default_value="20",
            description="Collision cache mesh size",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.workspace_file_path",
            default_value="",
            description="Path to workspace bounds file",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.grid_size_m",
            default_value="[2.0, 2.0, 2.0]",
            description="Voxel grid size in meters",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.voxel_size",
            default_value="0.05",
            description="Voxel size in meters",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.read_esdf_world",
            default_value="False",
            description="Read ESDF world from the cuRobo mapper",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.publish_curobo_world_as_voxels",
            default_value="False",
            description="Publish cuRobo world as voxels",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.add_ground_plane",
            default_value="False",
            description="Add ground plane",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.publish_voxel_size",
            default_value="0.05",
            description="Voxel publish size",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.max_publish_voxels",
            default_value="50000",
            description="Maximum number of voxels to publish",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.tool_frame",
            default_value="",
            description="Tool frame name",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.grid_center_m",
            default_value="[0.0, 0.0, 0.0]",
            description="Voxel grid center in meters",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.esdf_service_name",
            default_value="/curobo_mapper/get_esdf_and_gradient",
            description="ESDF service name",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.enable_curobo_debug_mode",
            default_value="False",
            description="Enable cuRobo debug mode",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.override_moveit_scaling_factors",
            default_value="False",
            description="Override MoveIt scaling factors",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.enable_cuda_mps",
            default_value="False",
            description="Enable CUDA MPS",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.cuda_mps_pipe_directory",
            default_value="/workspaces/isaac_ros-dev/ros_ws/mps_pipe_dir",
            description="CUDA MPS pipe directory",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.cuda_mps_client_priority",
            default_value="0",
            description="CUDA MPS client priority",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.cuda_mps_active_thread_percentage",
            default_value="100",
            description="CUDA MPS active thread percentage",
        ),
        DeclareLaunchArgument(
            "cumotion_planner.moveit_collision_objects_scene_file",
            default_value="",
            description="Path to MoveIt collision objects scene file",
        ),
        DeclareLaunchArgument(
            "log_debug",
            default_value="True",
            description="Enable debug logging for segmenter",
        ),
    ]

    # LaunchConfiguration objects to pass to the launch files
    robot = LaunchConfiguration("robot")
    urdf_path = LaunchConfiguration("urdf_file_path")
    yml_file_path = LaunchConfiguration("yml_file_path")
    joint_states_topic = LaunchConfiguration("joint_states_topic")
    depth_camera_info_topics = LaunchConfiguration("depth_camera_info_topics")
    depth_image_topics = LaunchConfiguration("depth_image_topics")
    object_link_name = LaunchConfiguration("object_link_name")
    search_radius = LaunchConfiguration("search_radius")
    surface_sphere_radius = LaunchConfiguration("surface_sphere_radius")
    update_esdf_on_request = LaunchConfiguration("update_esdf_on_request")
    update_link_sphere_server_segmenter = LaunchConfiguration(
        "update_link_sphere_server_segmenter"
    )
    update_link_sphere_server_planner = LaunchConfiguration(
        "update_link_sphere_server_planner"
    )
    clustering_bypass = LaunchConfiguration("clustering_bypass")
    action_names = LaunchConfiguration("action_names")
    time_sync_slop = LaunchConfiguration("time_sync_slop")
    distance_threshold = LaunchConfiguration("distance_threshold")
    clustering_hdbscan_min_samples = LaunchConfiguration(
        "clustering_hdbscan_min_samples"
    )
    clustering_hdbscan_min_cluster_size = LaunchConfiguration(
        "clustering_hdbscan_min_cluster_size"
    )
    clustering_hdbscan_cluster_selection_epsilon = LaunchConfiguration(
        "clustering_hdbscan_cluster_selection_epsilon"
    )
    clustering_num_top_clusters_to_select = LaunchConfiguration(
        "clustering_num_top_clusters_to_select"
    )
    clustering_group_clusters = LaunchConfiguration("clustering_group_clusters")
    clustering_min_points = LaunchConfiguration("clustering_min_points")
    enable_segmenter = LaunchConfiguration("enable_segmenter")
    object_attachment_gripper_frame_name = LaunchConfiguration(
        "object_attachment_gripper_frame_name"
    )
    log_debug = LaunchConfiguration("log_debug")

    # Shared world depth topic as a string array
    world_depth_topic = "['/cumotion/camera_1/world_depth']"

    # Paths to the launch files
    cumotion_launch_path = os.path.join(
        get_package_share_directory("isaac_ros_cumotion"),
        "launch",
        "isaac_ros_cumotion.launch.py",
    )

    robot_segmenter_launch_path = os.path.join(
        get_package_share_directory("isaac_ros_cumotion"),
        "launch",
        "robot_segmentation.launch.py",
    )

    object_attachment_launch_path = os.path.join(
        get_package_share_directory("isaac_ros_cumotion_object_attachment"),
        "launch",
        "object_attachment.launch.py",
    )

    mapper_launch_path = os.path.join(
        get_package_share_directory("isaac_ros_mapper"), "launch", "mapper.launch.py"
    )

    # Include the launch files with updated arguments
    cumotion_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(cumotion_launch_path),
        launch_arguments={
            "cumotion_planner.robot": robot,
            "cumotion_planner.urdf_path": urdf_path,
            "cumotion_planner.yml_file_path": yml_file_path,
            "cumotion_planner.time_dilation_factor": LaunchConfiguration(
                "cumotion_planner.time_dilation_factor"
            ),
            "cumotion_planner.max_attempts": LaunchConfiguration(
                "cumotion_planner.max_attempts"
            ),
            "cumotion_planner.num_graph_seeds": LaunchConfiguration(
                "cumotion_planner.num_graph_seeds"
            ),
            "cumotion_planner.num_trajopt_seeds": LaunchConfiguration(
                "cumotion_planner.num_trajopt_seeds"
            ),
            "cumotion_planner.include_trajopt_retract_seed": LaunchConfiguration(
                "cumotion_planner.include_trajopt_retract_seed"
            ),
            "cumotion_planner.num_trajopt_time_steps": LaunchConfiguration(
                "cumotion_planner.num_trajopt_time_steps"
            ),
            "cumotion_planner.joint_states_topic": LaunchConfiguration(
                "cumotion_planner.joint_states_topic"
            ),
            "cumotion_planner.interpolation_dt": LaunchConfiguration(
                "cumotion_planner.interpolation_dt"
            ),
            "cumotion_planner.collision_cache_cuboid": LaunchConfiguration(
                "cumotion_planner.collision_cache_cuboid"
            ),
            "cumotion_planner.collision_cache_mesh": LaunchConfiguration(
                "cumotion_planner.collision_cache_mesh"
            ),
            "cumotion_planner.workspace_file_path": LaunchConfiguration(
                "cumotion_planner.workspace_file_path"
            ),
            "cumotion_planner.grid_size_m": LaunchConfiguration(
                "cumotion_planner.grid_size_m"
            ),
            "cumotion_planner.voxel_size": LaunchConfiguration(
                "cumotion_planner.voxel_size"
            ),
            "cumotion_planner.read_esdf_world": LaunchConfiguration(
                "cumotion_planner.read_esdf_world"
            ),
            "cumotion_planner.publish_curobo_world_as_voxels": LaunchConfiguration(
                "cumotion_planner.publish_curobo_world_as_voxels"
            ),
            "cumotion_planner.add_ground_plane": LaunchConfiguration(
                "cumotion_planner.add_ground_plane"
            ),
            "cumotion_planner.publish_voxel_size": LaunchConfiguration(
                "cumotion_planner.publish_voxel_size"
            ),
            "cumotion_planner.max_publish_voxels": LaunchConfiguration(
                "cumotion_planner.max_publish_voxels"
            ),
            "cumotion_planner.tool_frame": LaunchConfiguration(
                "cumotion_planner.tool_frame"
            ),
            "cumotion_planner.grid_center_m": LaunchConfiguration(
                "cumotion_planner.grid_center_m"
            ),
            "cumotion_planner.esdf_service_name": LaunchConfiguration(
                "cumotion_planner.esdf_service_name"
            ),
            "cumotion_planner.enable_curobo_debug_mode": LaunchConfiguration(
                "cumotion_planner.enable_curobo_debug_mode"
            ),
            "cumotion_planner.override_moveit_scaling_factors": LaunchConfiguration(
                "cumotion_planner.override_moveit_scaling_factors"
            ),
            "cumotion_planner.update_link_sphere_server": update_link_sphere_server_planner,
            "cumotion_planner.enable_cuda_mps": LaunchConfiguration(
                "cumotion_planner.enable_cuda_mps"
            ),
            "cumotion_planner.cuda_mps_pipe_directory": LaunchConfiguration(
                "cumotion_planner.cuda_mps_pipe_directory"
            ),
            "cumotion_planner.cuda_mps_client_priority": LaunchConfiguration(
                "cumotion_planner.cuda_mps_client_priority"
            ),
            "cumotion_planner.cuda_mps_active_thread_percentage": LaunchConfiguration(
                "cumotion_planner.cuda_mps_active_thread_percentage"
            ),
            "cumotion_planner.moveit_collision_objects_scene_file": LaunchConfiguration(
                "cumotion_planner.moveit_collision_objects_scene_file"
            ),
        }.items(),
    )

    robot_segmenter_launch = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(robot_segmenter_launch_path),
                launch_arguments={
                    "robot_segmenter.robot": robot,
                    "robot_segmenter.urdf_path": urdf_path,
                    "cumotion_planner.yml_file_path": yml_file_path,
                    "robot_segmenter.depth_image_topics": depth_image_topics,
                    "robot_segmenter.depth_camera_info_topics": depth_camera_info_topics,
                    "robot_segmenter.joint_states_topic": joint_states_topic,
                    "robot_segmenter.time_sync_slop": time_sync_slop,
                    "robot_segmenter.distance_threshold": distance_threshold,
                    "robot_segmenter.update_link_sphere_server": update_link_sphere_server_segmenter,
                    "robot_segmenter.world_depth_publish_topics": world_depth_topic,
                    "robot_segmenter.log_debug": log_debug,
                    "robot_segmenter.depth_qos": "DEFAULT",
                    "robot_segmenter.depth_info_qos": "DEFAULT",
                    "robot_segmenter.mask_qos": "DEFAULT",
                    "robot_segmenter.world_depth_qos": "DEFAULT",
                    "standalone_mode": "true",
                }.items(),
            )
        ],
        condition=IfCondition(enable_segmenter),
    )

    object_attachment_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(object_attachment_launch_path),
        launch_arguments={
            "object_attachment.robot": robot,
            "object_attachment.urdf_path": urdf_path,
            "object_attachment.object_attachment_gripper_frame_name": object_attachment_gripper_frame_name,
            "object_attachment.time_sync_slop": time_sync_slop,
            "object_attachment.joint_states_topic": joint_states_topic,
            "object_attachment.depth_image_topics": world_depth_topic,
            "object_attachment.depth_camera_info_topics": depth_camera_info_topics,
            "object_attachment.object_link_name": object_link_name,
            "object_attachment.action_names": action_names,
            "object_attachment.search_radius": search_radius,
            "object_attachment.surface_sphere_radius": surface_sphere_radius,
            "object_attachment.update_esdf_on_request": update_esdf_on_request,
            "object_attachment.clustering_bypass_clustering": clustering_bypass,
            "object_attachment.clustering_hdbscan_min_samples": clustering_hdbscan_min_samples,
            "object_attachment.clustering_hdbscan_min_cluster_size": clustering_hdbscan_min_cluster_size,
            "object_attachment.clustering_hdbscan_cluster_selection_epsilon": clustering_hdbscan_cluster_selection_epsilon,
            "object_attachment.clustering_num_top_clusters_to_select": clustering_num_top_clusters_to_select,
            "object_attachment.clustering_group_clusters": clustering_group_clusters,
            "object_attachment.clustering_min_points": clustering_min_points,
            "object_attachment.depth_qos": "SENSOR_DATA",
            "object_attachment.depth_info_qos": "SENSOR_DATA",
        }.items(),
    )
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_param = SetParameter(name="use_sim_time", value=use_sim_time)

    # esdf_viser — included here so it shares the filesystem with the rest of
    # the cumotion stack and can access the URDF written by moveit_cumotion.
    esdf_viser_launch_path = os.path.join(
        get_package_share_directory("isaac_ros_esdf_visualizer"),
        "launch",
        "esdf_viser.launch.py",
    )
    kortex_params = _get_kortex_params_path()
    esdf_viser_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(esdf_viser_launch_path),
        launch_arguments={
            "params_file": kortex_params,
        }.items(),
    )

    esdf_viser_launch = GroupAction(
        actions=[esdf_viser_launch],
        condition=IfCondition(LaunchConfiguration("enable_viser")),
    )

    # cuRobo Mapper for TSDF/ESDF
    mapper_launch = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(mapper_launch_path),
                launch_arguments={
                    "depth_image_topics": depth_image_topics,
                    "depth_camera_info_topics": depth_camera_info_topics,
                    "esdf_service_name": "/curobo_mapper/get_esdf_and_gradient",
                    "params_file": kortex_params,
                }.items(),
            )
        ],
    )

    # Return the LaunchDescription with all included launch files
    return LaunchDescription(
        launch_args
        + [use_sim_time_param]
        + [
            cumotion_launch,
            robot_segmenter_launch,
            object_attachment_launch,
            mapper_launch,
            esdf_viser_launch,
        ]
    )
