# cuRobo / curobo_ros_manipulation_pipeline image.
#
# Builds this repository's packages on top of a ROS 2 or bare CUDA base.
# Build args: BASE_IMAGE, ROS_DISTRO, CUDA_TOOLKIT. nvcc is only present in
# the native build stage.

ARG BASE_IMAGE=nvidia/cuda:13.4.3-runtime-ubuntu24.04
FROM ${BASE_IMAGE} AS curobo-python-base

ARG ROS_DISTRO=jazzy
ARG CUDA_TOOLKIT=13-4

ENV ROS_DISTRO=${ROS_DISTRO}
ENV ROS_WS=/root/ros2_ws
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# The vendored checkout has no .git, so setuptools-scm needs a version.
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.8.0
# Keep CUDA_TOOLKIT and the Warp wheel in sync.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

# Every install below runs as root.
USER root

# --- CUDA runtime (only when the base does not ship it) ---
# NVIDIA's repos are keyed by <distro><version>, e.g. ubuntu2204.
RUN [ -z "${CUDA_TOOLKIT}" ] || ( \
  apt-get update && apt-get install -y --no-install-recommends \
  curl ca-certificates && \
  OS_REPO="ubuntu$(. /etc/os-release && echo ${VERSION_ID//./})" && \
  curl -fsSLO "https://developer.download.nvidia.com/compute/cuda/repos/${OS_REPO}/x86_64/cuda-keyring_1.1-1_all.deb" && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  rm -f cuda-keyring_1.1-1_all.deb && \
  apt-get update && apt-get install -y --no-install-recommends \
  cuda-cudart-${CUDA_TOOLKIT} && \
  ln -sfn "/usr/local/cuda-${CUDA_TOOLKIT//-/.}" /usr/local/cuda && \
  rm -rf /var/lib/apt/lists/* \
)

# Always-installed tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
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

# Install ROS 2 + rosdep if the base lacks them.
# rosidl-generator-type-description only exists on Jazzy.
RUN [ -f /opt/ros/${ROS_DISTRO}/setup.bash ] || ( \
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
  rm -rf /var/lib/apt/lists/* \
)

# Install upstream's declared dependencies from its metadata, then remove the
# stub (colcon installs the real package). setuptools 80 dropped
# `develop --uninstall`, so keep setuptools 77-79 for torch.
COPY curobo/curobo/pyproject.toml curobo/curobo/README.md /tmp/curobo-meta/
# setuptools_scm 9 needs packaging>=26.2, which the older base cannot build.
RUN sed -i 's/setuptools_scm>=6.2/setuptools_scm<9/' /tmp/curobo-meta/pyproject.toml

# Build the metadata wheel with a modern setuptools so pip can read its
# Requires-Dist entries; then install it with the pipeline's extras.
RUN --mount=type=cache,target=/root/.cache/pip \
  python3 -m pip install --ignore-installed \
  "numpy<2" \
  "setuptools==78.1.1" \
  "setuptools_scm<9" \
  "packaging>=26.2" \
  wheel \
  cuda-python \
  cuda-bindings \
  cuda-core \
  timm \
  einops \
  hdbscan \
  torchvision \
  bubblify && \
  python3 -m pip wheel --no-build-isolation --no-deps \
  -w /tmp/curobo-wheels "/tmp/curobo-meta" && \
  python3 -m pip install --ignore-installed \
  "numpy<2" \
  "pandas>=2.2.3" \
  "$(ls /tmp/curobo-wheels/*.whl)[benchmark]" && \
  python3 -m pip uninstall -y nvidia-curobo && \
  rm -rf /tmp/curobo-meta /tmp/curobo-wheels

# Pin ros2_numpy to the pipeline's NumPy range.
RUN git clone https://github.com/nitesh-subedi/ros2_numpy.git /tmp/ros2_numpy && \
  sed -i 's/numpy==1\.24\.2/numpy>=1.26.4,<2/' \
  /tmp/ros2_numpy/setup.py \
  /tmp/ros2_numpy/requirements.txt 2>/dev/null || true && \
  python3 -m pip install /tmp/ros2_numpy

# Reinstall the message packages against this distro.
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
# pybind11 (embedded interpreter) and Eigen; the CUDA base lacks universe.
RUN add-apt-repository -y universe && \
  apt-get update && apt-get install -y --no-install-recommends \
  pybind11-dev \
  libeigen3-dev \
  && rm -rf /var/lib/apt/lists/*

# --- ROS deps (rosdep -> apt): manifests ONLY, before the sources ------------
# This is the slow apt/rosdep layer. It is keyed on the package.xml files, so
# editing any C++/Python source below does NOT re-run rosdep/apt. Everything
# after this point is rebuilt on a source change.
COPY curobov2_ros/package.xml ${ROS_WS}/src/curobov2_ros/package.xml
COPY curobov2_ros_interfaces/package.xml ${ROS_WS}/src/curobov2_ros_interfaces/package.xml
COPY curobov2_ros_moveit/package.xml ${ROS_WS}/src/curobov2_ros_moveit/package.xml
COPY curobov2_ros_rviz/package.xml ${ROS_WS}/src/curobov2_ros_rviz/package.xml
COPY curobov2_ros_extra/package.xml ${ROS_WS}/src/curobov2_ros_extra/package.xml
COPY curobo_python_runtime/package.xml ${ROS_WS}/src/curobo_python_runtime/package.xml
COPY curobo_task_constructor_interfaces/package.xml ${ROS_WS}/src/curobo_task_constructor_interfaces/package.xml
COPY curobo_task_constructor/package.xml ${ROS_WS}/src/curobo_task_constructor/package.xml
COPY curobo_task_constructor_rviz/package.xml ${ROS_WS}/src/curobo_task_constructor_rviz/package.xml
COPY curobo_cpp_runtime/package.xml ${ROS_WS}/src/curobo_cpp_runtime/package.xml
COPY curobo/package.xml ${ROS_WS}/src/curobo/package.xml

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
  rosdep init 2>/dev/null || true; \
  rosdep update && apt-get update && \
  rosdep install \
  --from-paths \
  ${ROS_WS}/src/curobov2_ros \
  ${ROS_WS}/src/curobov2_ros_interfaces \
  ${ROS_WS}/src/curobov2_ros_moveit \
  ${ROS_WS}/src/curobov2_ros_rviz \
  ${ROS_WS}/src/curobov2_ros_extra \
  ${ROS_WS}/src/curobo_task_constructor_interfaces \
  ${ROS_WS}/src/curobo_task_constructor \
  ${ROS_WS}/src/curobo_task_constructor_rviz \
  ${ROS_WS}/src/curobo_cpp_runtime \
  ${ROS_WS}/src/curobo \
  --ignore-src --rosdistro ${ROS_DISTRO} -r -y && \
  rm -rf /var/lib/apt/lists/*
# NOTE: rosdep -r tolerates optional exec dependencies declared by
# curobov2_ros_extra; the package still builds without them.


# --- Standalone native cpp_runtime (not a colcon package) ---
# CUROBO_CUDA_ARCHS is required, e.g. "86;89;90".
# ============================================================================
FROM curobo-python-base AS curobo-native

ARG CUROBO_CUDA_ARCHS
ARG CUDA_TOOLKIT=13-4

# nvcc and the CUDA dev packages are build-only.
RUN [ -z "${CUDA_TOOLKIT}" ] || ( \
  apt-get update && apt-get install -y --no-install-recommends \
  cuda-toolkit-${CUDA_TOOLKIT} && \
  rm -rf /var/lib/apt/lists/* \
)

# Expose Warp's native headers at a stable path for CMake.
ENV WARP_NATIVE_INCLUDE_DIR=/opt/warp/native
RUN WARP_ROOT="$(python3 -c 'from pathlib import Path; import warp; print(Path(warp.__file__).parent)')" && \
    mkdir -p /opt/warp && \
    ln -sfn "${WARP_ROOT}/native" /opt/warp/native && \
    test -f /opt/warp/native/warp.h

COPY curobo/curobo/cpp_runtime /opt/curobo-src/cpp_runtime
COPY curobo/curobo/curobo /opt/curobo-src/curobo

RUN : "${CUROBO_CUDA_ARCHS:?CUROBO_CUDA_ARCHS must be set}" && \
    CUDA_ARCHS="${CUROBO_CUDA_ARCHS//sm_/}" && \
    CUDA_ARCHS="${CUDA_ARCHS//,/;}" && \
    CUDA_ARCHS="${CUDA_ARCHS// /}" && \
    cmake -S /opt/curobo-src/cpp_runtime/robot \
      -B /opt/curobo-native/robot \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DBUILD_TESTING=OFF \
      -DWARP_NATIVE_INCLUDE_DIR="${WARP_NATIVE_INCLUDE_DIR}" && \
    cmake --build /opt/curobo-native/robot -j"$(nproc)" && \
    cmake --install /opt/curobo-native/robot --prefix /opt/curobo/native && \
    cmake -S /opt/curobo-src/cpp_runtime/mapper \
      -B /opt/curobo-native/mapper \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DBUILD_TESTING=OFF \
      -DWARP_NATIVE_INCLUDE_DIR="${WARP_NATIVE_INCLUDE_DIR}" && \
    cmake --build /opt/curobo-native/mapper -j"$(nproc)" && \
    cmake --install /opt/curobo-native/mapper --prefix /opt/curobo/native && \
    mkdir -p /opt/curobo/native/lib /opt/curobo/native/include && \
    find /opt/curobo-native/robot /opt/curobo-native/mapper \
      -maxdepth 1 -type f -name 'libcurobo_*_runtime.so*' \
      -exec cp -a {} /opt/curobo/native/lib/ \; && \
    cp -a /opt/curobo-src/cpp_runtime/robot/include/. /opt/curobo/native/include/ && \
    cp -a /opt/curobo-src/cpp_runtime/mapper/include/. /opt/curobo/native/include/ && \
    rm -rf /opt/curobo-src

# ============================================================================
# ROS workspace build stage.
# ============================================================================
FROM curobo-python-base AS curobo-pipeline-build

ARG ROS_DISTRO=jazzy
ARG CUDA_TOOLKIT=13-4

# The ROS C++ package links against libcuda; the runtime driver arrives via CDI.
RUN [ -z "${CUDA_TOOLKIT}" ] || ( \
  apt-get update && apt-get install -y --no-install-recommends \
  cuda-cudart-dev-${CUDA_TOOLKIT} cuda-crt-${CUDA_TOOLKIT} \
  cuda-driver-dev-${CUDA_TOOLKIT} && \
  rm -rf /var/lib/apt/lists/* \
)

COPY --from=curobo-native /opt/curobo/native /opt/curobo/native

WORKDIR ${ROS_WS}

# --- curobov2_ros packages (sources; a source edit invalidates from here) ---
COPY curobov2_ros ${ROS_WS}/src/curobov2_ros
COPY curobov2_ros_interfaces ${ROS_WS}/src/curobov2_ros_interfaces
COPY curobov2_ros_moveit ${ROS_WS}/src/curobov2_ros_moveit
COPY curobov2_ros_rviz ${ROS_WS}/src/curobov2_ros_rviz
COPY curobov2_ros_extra ${ROS_WS}/src/curobov2_ros_extra

# --- C++ port: the bridge (the C++ server now lives inside curobov2_ros, cpp/) ---
COPY curobo_python_runtime ${ROS_WS}/src/curobo_python_runtime

# --- task-constructor packages ---
COPY curobo_task_constructor_interfaces ${ROS_WS}/src/curobo_task_constructor_interfaces
COPY curobo_task_constructor ${ROS_WS}/src/curobo_task_constructor
COPY curobo_task_constructor_rviz ${ROS_WS}/src/curobo_task_constructor_rviz

# --- native runtime selector/bridge ROS package (normal colcon package) ---
COPY curobo_cpp_runtime ${ROS_WS}/src/curobo_cpp_runtime

# --- cuRobo v2 ---
COPY curobo ${ROS_WS}/src/curobo

# Limit discovery to the copied directories: a base may already contain
# duplicate package names.
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
  if [ -f ${ROS_WS}/install/setup.bash ]; then \
    source ${ROS_WS}/install/setup.bash; \
  fi; \
  colcon build \
  --base-paths \
  ${ROS_WS}/src/curobov2_ros \
  ${ROS_WS}/src/curobov2_ros_interfaces \
  ${ROS_WS}/src/curobov2_ros_moveit \
  ${ROS_WS}/src/curobov2_ros_rviz \
  ${ROS_WS}/src/curobov2_ros_extra \
  ${ROS_WS}/src/curobo_task_constructor_interfaces \
  ${ROS_WS}/src/curobo_task_constructor \
  ${ROS_WS}/src/curobo_task_constructor_rviz \
  ${ROS_WS}/src/curobo_python_runtime \
  ${ROS_WS}/src/curobo_cpp_runtime \
  ${ROS_WS}/src/curobo \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --packages-select \
  curobo curobov2_ros_interfaces \
  curobov2_ros curobov2_ros_moveit \
  curobov2_ros_rviz curobov2_ros_extra \
  curobo_task_constructor_interfaces \
  curobo_task_constructor curobo_task_constructor_rviz \
  curobo_python_runtime curobo_cpp_runtime

# ============================================================================
# Final runtime image.
# ============================================================================
FROM curobo-python-base AS curobo-manipulation-pipeline

COPY --from=curobo-native /opt/curobo/native /opt/curobo/native
COPY --from=curobo-pipeline-build ${ROS_WS}/install ${ROS_WS}/install
# Source subtrees needed by the artifact exporter and the C++ benchmark.
COPY --from=curobo-pipeline-build ${ROS_WS}/src/curobo/curobo/cpp_runtime ${ROS_WS}/src/curobo/curobo/cpp_runtime
COPY --from=curobo-pipeline-build ${ROS_WS}/src/curobo/curobo/curobo ${ROS_WS}/src/curobo/curobo/curobo
COPY --from=curobo-pipeline-build ${ROS_WS}/src/curobo/curobo/benchmark ${ROS_WS}/src/curobo/curobo/benchmark
COPY --from=curobo-pipeline-build ${ROS_WS}/src/curobov2_ros ${ROS_WS}/src/curobov2_ros

# Artifact metadata helper for the artifact-export layer.
COPY docker/write_artifact_metadata.py /opt/curobo/tools/write_artifact_metadata.py

ENV CUROBO_CPP_ROOT=/opt/curobo/native
ENV CUROBO_ARTIFACT_ROOT=/opt/curobo/artifacts
ENV CUROBO_RUNTIME=cpp
