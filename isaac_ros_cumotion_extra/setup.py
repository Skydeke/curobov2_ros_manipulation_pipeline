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

from setuptools import find_packages, setup

package_name = 'isaac_ros_cumotion_extra'

setup(
    name=package_name,
    version='3.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            ['launch/esdf_viser.launch.py',
             'launch/getting_started_viser.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Isaac ROS Maintainers',
    maintainer_email='isaac-ros-maintainers@nvidia.com',
    description='Extra tools for isaac_ros_cumotion: ESDF/Viser visualizer and cuRobo config generator.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest'
        ]
    },
    entry_points={
        'console_scripts': [
            'esdf_viser_node = isaac_ros_cumotion_extra.esdf_viser_node:main',
            'build_curobo_config = isaac_ros_cumotion_extra.build_curobo_config:main',
            'fk_viser_node = isaac_ros_cumotion_extra.fk_viser_node:main',
            'ik_viser_node = isaac_ros_cumotion_extra.ik_viser_node:main',
            'mp_viser_node = isaac_ros_cumotion_extra.mp_viser_node:main',
            'mpc_viser_node = isaac_ros_cumotion_extra.mpc_viser_node:main',
            'volumetric_viser_node = isaac_ros_cumotion_extra.volumetric_viser_node:main',
            'feature_viser_node = isaac_ros_cumotion_extra.feature_viser_node:main',
            'robot_model_viser_node = isaac_ros_cumotion_extra.robot_model_viser_node:main',
            'retarget_viser_node = isaac_ros_cumotion_extra.retarget_viser_node:main',
        ],
    },
)
