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

package_name = "curobo"

# `colcon build` ships the importable tree, its CUDA kernel sources, its pybind
# sources and its runtime content (robot assets, YAML configs) via
# `package_data` below. The Dockerfile also installs the vendored upstream
# project editable so pip resolves its declared dependencies; both paths point
# at curobo/curobo/curobo, so there is no divergent copy on sys.path.
#
# `version` is pinned here rather than derived from git. The vendored tree is a
# nested git submodule, so `COPY` into the image brings no .git and
# setuptools-scm has nothing to read; the Dockerfile sets
# SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO for the upstream editable
# install. A static version is honest: the ROS package version and the vendored
# upstream version are then visible side by side in this file and in
# .gitmodules' pinned commit, instead of one being inferred from the other.
version = "4.3.0"

all_packages = find_namespace_packages(where="curobo")
packages = [
    p
    for p in all_packages
    if p.startswith("curobo")
    and not p.startswith("curobo.tests")
    and not p.startswith("curobo.examples")
]

setup(
    name=package_name,
    version=version,
    packages=packages,
    package_dir={"": "curobo"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="me",
    maintainer_email="me@todo.com",
    description="This package wraps the cuRobo library as a ROS 2 package. ",
    license="NVIDIA Isaac ROS Software License",
    entry_points={
        "console_scripts": [],
    },
    include_package_data=True,
    package_data={
        "curobo._src.curobolib.kernels": ["**/*.cu", "**/*.cuh", "**/*.h"],
        "curobo._src.curobolib.backends.pybind": ["*.cpp", "*.cu"],
        "curobo.content": ["**/*"],
    },
)
