# cuRobo / curobo_ros_manipulation_pipeline image.
#
# Builds and installs this repository's packages -- the Python `curobo` stack,
# the C++ `curobov2_ros` port with its embedded-interpreter bridge
# `curobo_bridge`, the task-constructor packages and the RViz plugins -- on top
# of a base image that already provides ROS 2 (or a bare CUDA base, in which
# case ROS is installed here).
#
# The build context is this repository's root. Everything that varies with the
# base image is a build argument:
#
#   BASE_IMAGE      image to layer onto (default: CUDA 13.2 devel, Ubuntu 24.04)
#   ROS_DISTRO      distro used for the apt package names (default: jazzy)
#   USERNAME        owner of the workspace (default: root)
#   USER_HOME       home directory of USERNAME (default: /root)
#   ROS_WS          colcon workspace root (default: /root/ros2_ws)
#   CUDA_TOOLKIT    install the CUDA toolkit from NVIDIA's repo when the base
#                   image ships none, e.g. 12-8; empty = base already has CUDA
#   COLCON_WORKERS  colcon --parallel-workers; empty = colcon default
#
# With no arguments this reproduces the previous behaviour (CUDA 13.2 base,
# jazzy, root, /root/ros2_ws).

ARG BASE_IMAGE=nvidia/cuda:13.2.0-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG ROS_DISTRO=jazzy
ARG USERNAME=root
ARG USER_HOME=/root
ARG ROS_WS=/root/ros2_ws
ARG CUDA_TOOLKIT=
ARG COLCON_WORKERS=

ENV ROS_DISTRO=${ROS_DISTRO}
ENV ROS_WS=${ROS_WS}
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# /usr/local/cuda is the alternatives symlink that both the NVIDIA CUDA base
# images and the cuda-toolkit-* packages create, so one set of ENVs serves a
# base that ships CUDA and one where CUDA_TOOLKIT installs it. Re-prepending
# PATH/LD_LIBRARY_PATH is harmless when the base already set them.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
SHELL ["/bin/bash", "-c"]

# Every install below runs as root; rosdep/colcon then run as USERNAME.
USER root

# --- CUDA toolkit (only when the BASE_IMAGE does not ship one) ---
# NVIDIA's repos are keyed by <distro><version-without-dot>, e.g. ubuntu2204;
# derive it so this works for the 22.04 and 24.04 bases alike. NOTE: the repo's
# old .pub key file is gone upstream (404); the cuda-keyring deb is the
# supported way to register repo + key.
RUN if [ -n "${CUDA_TOOLKIT}" ]; then \
  apt-get update && apt-get install -y --no-install-recommends \
  curl ca-certificates && \
  OS_REPO="ubuntu$(. /etc/os-release && echo ${VERSION_ID//./})" && \
  curl -fsSLO "https://developer.download.nvidia.com/compute/cuda/repos/${OS_REPO}/x86_64/cuda-keyring_1.1-1_all.deb" && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  rm -f cuda-keyring_1.1-1_all.deb && \
  apt-get update && apt-get install -y --no-install-recommends \
  cuda-toolkit-${CUDA_TOOLKIT} && \
  rm -rf /var/lib/apt/lists/*; \
  fi

# Always-installed tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  git \
  gnupg2 \
  meson \
  ninja-build \
  python-is-python3 \
  python3-dev \
  python3-pip \
  python3-venv \
  software-properties-common \
  && rm -rf /var/lib/apt/lists/* && \
  (command -v colcon >/dev/null 2>&1) || pip install colcon-common-extensions

# Install ROS 2 + rosdep only if not already present in BASE_IMAGE.
# ros-${ROS_DISTRO}-rosidl-generator-type-description is Jazzy-only, so it is
# added to the list only for jazzy.
RUN if [ ! -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then \
  apt-get update && apt-get install -y --no-install-recommends \
  curl gnupg2 ca-certificates && \
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg && \
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  > /etc/apt/sources.list.d/ros2.list && \
  apt-get update && apt-get install -y --no-install-recommends \
  ros-${ROS_DISTRO}-ros-base \
  python3-rosdep \
  $( [ "${ROS_DISTRO}" = jazzy ] && echo ros-${ROS_DISTRO}-rosidl-generator-type-description ) && \
  rm -rf /var/lib/apt/lists/* && \
  rosdep init 2>/dev/null || true; \
  fi

# Core numerical stack
RUN python3 -m pip install --ignore-installed \
  "numpy<2" \
  "cuda-python<13" \
  "cuda-bindings~=12.9" \
  "cuda-core[cu12]" \
  "cupy-cuda12x<14" \
  "timm" \
  einops \
  hdbscan \
  "viser>=1" \
  torchvision \
  bubblify

# Someone is misusing pinning, for no good damn reason
RUN git clone https://github.com/nitesh-subedi/ros2_numpy.git /tmp/ros2_numpy && \
  sed -i 's/numpy==1\.24\.2/numpy>=1.26.4,<2/' \
  /tmp/ros2_numpy/setup.py \
  /tmp/ros2_numpy/requirements.txt 2>/dev/null || true && \
  python3 -m pip install --ignore-installed /tmp/ros2_numpy

# Reinstall the message packages against this distro. rosidl-generator-type-
# description rides along on Jazzy only (it does not exist on Humble).
RUN apt-get update && apt-get install -y --reinstall \
  $( [ "${ROS_DISTRO}" = jazzy ] && echo ros-${ROS_DISTRO}-rosidl-generator-type-description ) \
  ros-${ROS_DISTRO}-sensor-msgs \
  ros-${ROS_DISTRO}-visualization-msgs \
  ros-${ROS_DISTRO}-geometry-msgs \
  ros-${ROS_DISTRO}-std-msgs \
  ros-${ROS_DISTRO}-builtin-interfaces \
  ros-${ROS_DISTRO}-trajectory-msgs \
  ros-${ROS_DISTRO}-nav2-msgs \
  ros-${ROS_DISTRO}-moveit-msgs && \
  rm -rf /var/lib/apt/lists/*

# --- C++ port build dependencies ---
# pybind11 (embedded interpreter) and Eigen come from Ubuntu universe; the CUDA
# base image does not have universe enabled by default.
RUN add-apt-repository -y universe && \
  apt-get update && apt-get install -y --no-install-recommends \
  pybind11-dev \
  libeigen3-dev \
  && rm -rf /var/lib/apt/lists/*

# --- cuRobo v2: dependencies FIRST, sources LAST ---
COPY curobo/requirements.txt /tmp/curobo_requirements.txt
RUN pip install --no-cache-dir -r /tmp/curobo_requirements.txt

WORKDIR ${ROS_WS}

# --- ROS deps (rosdep -> apt): manifests ONLY, before the sources ------------
# This is the slow apt/rosdep layer. It is keyed on the package.xml files, so
# editing any C++/Python source below does NOT re-run rosdep/apt. Everything
# after this point is rebuilt on a source change.
COPY --chown=${USERNAME}:${USERNAME} curobov2_ros/package.xml ${ROS_WS}/src/curobov2_ros/package.xml
COPY --chown=${USERNAME}:${USERNAME} curobov2_ros_interfaces/package.xml ${ROS_WS}/src/curobov2_ros_interfaces/package.xml
COPY --chown=${USERNAME}:${USERNAME} curobov2_ros_moveit/package.xml ${ROS_WS}/src/curobov2_ros_moveit/package.xml
COPY --chown=${USERNAME}:${USERNAME} curobov2_ros_rviz/package.xml ${ROS_WS}/src/curobov2_ros_rviz/package.xml
COPY --chown=${USERNAME}:${USERNAME} curobov2_ros_extra/package.xml ${ROS_WS}/src/curobov2_ros_extra/package.xml
COPY --chown=${USERNAME}:${USERNAME} curobo_bridge/package.xml ${ROS_WS}/src/curobo_bridge/package.xml
COPY --chown=${USERNAME}:${USERNAME} curobo_task_constructor_interfaces/package.xml ${ROS_WS}/src/curobo_task_constructor_interfaces/package.xml
COPY --chown=${USERNAME}:${USERNAME} curobo_task_constructor/package.xml ${ROS_WS}/src/curobo_task_constructor/package.xml
COPY --chown=${USERNAME}:${USERNAME} curobo_task_constructor_rviz/package.xml ${ROS_WS}/src/curobo_task_constructor_rviz/package.xml
COPY --chown=${USERNAME}:${USERNAME} curobo/package.xml ${ROS_WS}/src/curobo/package.xml

USER ${USERNAME}

# rosdep keeps its sources cache PER $HOME and `rosdep init` needs root, so run
# the whole toolchain through $SUDO (a no-op when the build already runs as
# root): updating as the unprivileged user and installing as root fails with
# "your rosdep installation has not been initialized yet".
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
  SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"; \
  $SUDO rosdep init 2>/dev/null || true; \
  $SUDO rosdep update && $SUDO apt-get update && \
  $SUDO rosdep install \
  --from-paths \
  ${ROS_WS}/src/curobov2_ros \
  ${ROS_WS}/src/curobov2_ros_interfaces \
  ${ROS_WS}/src/curobov2_ros_moveit \
  ${ROS_WS}/src/curobov2_ros_rviz \
  ${ROS_WS}/src/curobov2_ros_extra \
  ${ROS_WS}/src/curobo_task_constructor_interfaces \
  ${ROS_WS}/src/curobo_task_constructor \
  ${ROS_WS}/src/curobo_task_constructor_rviz \
  ${ROS_WS}/src/curobo \
  --ignore-src --rosdistro ${ROS_DISTRO} -r -y && \
  $SUDO rm -rf /var/lib/apt/lists/*
# NOTE: rosdep -r tolerates the iki_kortex_description exec key of
# curobov2_ros_extra (kortex-only, runtime-only); the package still builds.

# --- curobov2_ros packages (sources; a source edit invalidates from here) ---
COPY --chown=${USERNAME}:${USERNAME} curobov2_ros ${ROS_WS}/src/curobov2_ros
COPY --chown=${USERNAME}:${USERNAME} curobov2_ros_interfaces ${ROS_WS}/src/curobov2_ros_interfaces
COPY --chown=${USERNAME}:${USERNAME} curobov2_ros_moveit ${ROS_WS}/src/curobov2_ros_moveit
COPY --chown=${USERNAME}:${USERNAME} curobov2_ros_rviz ${ROS_WS}/src/curobov2_ros_rviz
COPY --chown=${USERNAME}:${USERNAME} curobov2_ros_extra ${ROS_WS}/src/curobov2_ros_extra

# --- C++ port: the bridge (the C++ server now lives inside curobov2_ros, cpp/) ---
COPY --chown=${USERNAME}:${USERNAME} curobo_bridge ${ROS_WS}/src/curobo_bridge

# --- task-constructor packages ---
COPY --chown=${USERNAME}:${USERNAME} curobo_task_constructor_interfaces ${ROS_WS}/src/curobo_task_constructor_interfaces
COPY --chown=${USERNAME}:${USERNAME} curobo_task_constructor ${ROS_WS}/src/curobo_task_constructor
COPY --chown=${USERNAME}:${USERNAME} curobo_task_constructor_rviz ${ROS_WS}/src/curobo_task_constructor_rviz

# --- cuRobo v2 ---
COPY --chown=${USERNAME}:${USERNAME} curobo ${ROS_WS}/src/curobo

# --base-paths is limited to the directories copied above: a base image may
# already carry other copies of these packages (e.g. as part of a larger
# source tree), and colcon refuses to build a workspace with duplicate package
# names. ${ROS_WS}/install is sourced for bases whose image already carries
# workspace deps; it does not exist yet on a fresh build and that is tolerated.
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
  source ${ROS_WS}/install/setup.bash 2>/dev/null; \
  colcon build --symlink-install \
  ${COLCON_WORKERS:+--parallel-workers ${COLCON_WORKERS}} \
  --base-paths \
  ${ROS_WS}/src/curobov2_ros \
  ${ROS_WS}/src/curobov2_ros_interfaces \
  ${ROS_WS}/src/curobov2_ros_moveit \
  ${ROS_WS}/src/curobov2_ros_rviz \
  ${ROS_WS}/src/curobov2_ros_extra \
  ${ROS_WS}/src/curobo_task_constructor_interfaces \
  ${ROS_WS}/src/curobo_task_constructor \
  ${ROS_WS}/src/curobo_task_constructor_rviz \
  ${ROS_WS}/src/curobo_bridge \
  ${ROS_WS}/src/curobo \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --packages-select \
  curobo curobov2_ros_interfaces \
  curobov2_ros curobov2_ros_moveit \
  curobov2_ros_rviz curobov2_ros_extra \
  curobo_task_constructor_interfaces \
  curobo_task_constructor curobo_task_constructor_rviz \
  curobo_bridge
