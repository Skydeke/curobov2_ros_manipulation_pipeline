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

from setuptools import find_namespace_packages, setup

package_name = 'curobo_core'

all_packages = find_namespace_packages(where='curobo')
packages = [
    p for p in all_packages
    if p.startswith('curobo')
    and not p.startswith('curobo.tests')
    and not p.startswith('curobo.examples')
]

setup(
    name=package_name,
    version='4.3.0',
    packages=packages,
    package_dir={'': 'curobo'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Isaac ROS Maintainers',
    maintainer_email='isaac-ros-maintainers@nvidia.com',
    description='This package wraps the cuRobo library as a ROS 2 package. '
                'cuRobo serves as the current backend for cuMotion.',
    license='NVIDIA Isaac ROS Software License',
    entry_points={
        'console_scripts': [],
    },
    include_package_data=True,
    package_data={
        'curobo._src.curobolib.kernels': ['**/*.cu', '**/*.cuh', '**/*.h'],
        'curobo._src.curobolib.backends.pybind': ['*.cpp', '*.cu'],
        'curobo.content': ['**/*'],
    },
)
