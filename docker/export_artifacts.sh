#!/usr/bin/env bash
# Prepare a local, content-addressed cuRobo native artifact bundle.
#
# This script is driven by each robot repo's
# ``docker/build_local_submodules.sh`` before the pipeline image build.  It:
#
#   1. resolves the requested GPU architecture(s) and optional device profile;
#   2. computes a content-addressed cache key from the cuRobo commit, the robot
#      and cuRobo config bytes, the tool-frame list, the target architecture(s),
#      and the exporter/helper versions;
#   3. reuses ``.curobo_artifacts/<key>`` when it is complete;
#   4. otherwise runs the upstream ``export_robot.py`` in a GPU-enabled helper
#      container built from the pipeline Dockerfile's ``curobo-native-export``
#      stage, then writes generated ``manifest.json`` metadata from
#      ``robots/*.yaml`` (Python build-time, no C++ YAML parser);
#   5. prints the artifact directory, relative to the pipeline repo, for
#      ``--build-arg CUROBO_ARTIFACT_DIR=...``.
#
# No artifact is ever checked in.  If no GPU is available the script creates an
# empty bundle directory, prints it, and exits 0 so the image can still be built;
# the ``auto`` runtime then uses the Python backend.
#
# Supported environment variables:
#   CUROBO_GPU_ARCH     auto/current or explicit, e.g. sm_89 / 89
#   CUROBO_GPU_ARCHS    comma-separated explicit list, e.g. 86,89,90,110
#   CUROBO_GPU_DEVICE   host GPU ordinal for docker --gpus (default: all)
#   CUROBO_GPU_PROFILE  device JSON from export_device.py for a foreign target
#   CUROBO_URDF         optional URDF override for export_robot.py
#   CUROBO_EXPORT_IMAGE override the helper image tag
#   CUROBO_ASSUME_GPU=1 force an export attempt when GPU probing is inconclusive
#   CUROBO_REQUIRE_ARTIFACTS=0 allow a Python-only image when export is not
#                            possible (default: 1, artifacts are required)
#
# The helper runs from any directory; the pipeline repo is derived from this
# script's location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_ROOT="${PIPELINE_DIR}/.curobo_artifacts"

log() { printf '[curobo-artifacts] %s\n' "$*" >&2; }
die() { log "error: $*"; exit 1; }

usage() {
  cat <<'USAGE'
Usage: docker/export_artifacts.sh --robot NAME --artifact-set DIR \
         --robot-config PATH --curobo-config PATH --tool-frame FRAME [options]

Required:
  --robot NAME              Robot name recorded in metadata (e.g. openarm)
  --artifact-set DIR        Bundle directory name (e.g. openarm_bimanual)
  --robot-config PATH       robots/*.yaml descriptor (single source of truth)
  --curobo-config PATH      cuRobo YAML referenced by the descriptor
  --tool-frame FRAME        Tool frame, repeatable

Optional:
  --urdf PATH               URDF override passed to export_robot.py
  --base-image IMAGE        BASE_IMAGE used to build the export helper image
  --ros-distro DISTRO       ROS distro used by that base image
  --cuda-toolkit TOOLKIT    CUDA_TOOLKIT build arg, e.g. 12-8
  --mount PATH              Extra path to bind-mount into the helper container
  --env-out PATH            Write resolved build values to this shell file
  -h, --help                Show this help
USAGE
}

ROBOT=""
ARTIFACT_SET=""
ROBOT_CONFIG=""
CUROBO_CONFIG=""
URDF="${CUROBO_URDF:-}"
BASE_IMAGE=""
ROS_DISTRO=""
CUDA_TOOLKIT=""
ENV_OUT=""
declare -a TOOL_FRAMES=()
declare -a EXTRA_MOUNTS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --robot) ROBOT="${2:-}"; shift 2 ;;
    --artifact-set) ARTIFACT_SET="${2:-}"; shift 2 ;;
    --robot-config) ROBOT_CONFIG="${2:-}"; shift 2 ;;
    --curobo-config) CUROBO_CONFIG="${2:-}"; shift 2 ;;
    --tool-frame) TOOL_FRAMES+=("${2:-}"); shift 2 ;;
    --urdf) URDF="${2:-}"; shift 2 ;;
    --base-image) BASE_IMAGE="${2:-}"; shift 2 ;;
    --ros-distro) ROS_DISTRO="${2:-}"; shift 2 ;;
    --cuda-toolkit) CUDA_TOOLKIT="${2:-}"; shift 2 ;;
    --mount) EXTRA_MOUNTS+=("${2:-}"); shift 2 ;;
    --env-out) ENV_OUT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

[ -n "${ROBOT}" ] || die "--robot is required"
[ -n "${ARTIFACT_SET}" ] || die "--artifact-set is required"
[ -n "${ROBOT_CONFIG}" ] || die "--robot-config is required"
[ -n "${CUROBO_CONFIG}" ] || die "--curobo-config is required"
[ "${#TOOL_FRAMES[@]}" -gt 0 ] || die "at least one --tool-frame is required"

abspath() {
  local path="$1"
  if [ "${path#/}" = "${path}" ]; then
    path="${PWD}/${path}"
  fi
  printf '%s\n' "$(cd "$(dirname "${path}")" 2>/dev/null && pwd)/$(basename "${path}")"
}

ROBOT_CONFIG="$(abspath "${ROBOT_CONFIG}")"
CUROBO_CONFIG="$(abspath "${CUROBO_CONFIG}")"
[ -f "${ROBOT_CONFIG}" ] || die "robot config not found: ${ROBOT_CONFIG}"
[ -f "${CUROBO_CONFIG}" ] || die "cuRobo config not found: ${CUROBO_CONFIG}"
URDF_TARGET=""
if [ -n "${URDF}" ]; then
  URDF="$(abspath "${URDF}")"
  [ -f "${URDF}" ] || die "URDF not found: ${URDF}"
  # The exporter passes --urdf to the top-level kinematics config, but nested
  # TrajOpt/IK configs re-read the YAML's own urdf_path.  Mount the generated
  # URDF at that path too, so every nesting level sees a real file.
  URDF_TARGET="$(
    sed -n 's/^[[:space:]]*urdf_path:[[:space:]]*//p' "${CUROBO_CONFIG}" \
      | head -n 1 \
      | sed -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//'
  )"
fi
if [ -n "${ENV_OUT}" ]; then
  ENV_OUT="$(abspath "${ENV_OUT}")"
fi
if [ -n "${CUROBO_GPU_PROFILE:-}" ]; then
  CUROBO_GPU_PROFILE="$(abspath "${CUROBO_GPU_PROFILE}")"
  [ -f "${CUROBO_GPU_PROFILE}" ] || die "CUROBO_GPU_PROFILE not found: ${CUROBO_GPU_PROFILE}"
fi

# Locate the vendored cuRobo checkout.  The pipeline package is curobo/, and the
# actual upstream checkout (with cpp_runtime) is the nested curobo/curobo.
CUROBO_SOURCE="${PIPELINE_DIR}/curobo/curobo"
if [ ! -d "${CUROBO_SOURCE}/cpp_runtime" ]; then
  CUROBO_SOURCE="${PIPELINE_DIR}/curobo"
fi
[ -d "${CUROBO_SOURCE}/cpp_runtime" ] || die "cpp_runtime not found under ${PIPELINE_DIR}/curobo"

normalize_arch() {
  local arch="$1"
  arch="${arch// /}"
  case "${arch}" in
    ""|auto|current) return 1 ;;
  esac
  case "${arch}" in
    sm_*) ;;
    *) arch="sm_${arch}" ;;
  esac
  printf '%s\n' "${arch}"
}

arch_without_prefix() {
  printf '%s\n' "${1#sm_}"
}

gpu_available=0
if [ "${CUROBO_ASSUME_GPU:-0}" = "1" ]; then
  gpu_available=1
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  gpu_available=1
elif [ -e /dev/nvidia0 ]; then
  gpu_available=1
fi

gpu_device="${CUROBO_GPU_DEVICE:-all}"
gpu_ordinal=0
if [ "${gpu_device}" != "all" ]; then
  gpu_ordinal="${gpu_device%%,*}"
  gpu_request="device=${gpu_device}"
else
  gpu_request="all"
fi

detect_arch_from_nvidia_smi() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  local compute_cap
  if [ "${gpu_device}" = "all" ]; then
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  else
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | sed -n "$((gpu_ordinal + 1))p" || true)"
  fi
  compute_cap="${compute_cap//[[:space:]]/}"
  [ -n "${compute_cap}" ] || return 1
  local major="${compute_cap%%.*}"
  local minor="${compute_cap#*.}"
  [ "${major}" != "${compute_cap}" ] || return 1
  printf 'sm_%s%s\n' "${major}" "${minor}"
}

profile_arch() {
  local profile="$1"
  python3 - "${profile}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
print(data["arch"])
PY
}

# --- Resolve the target architecture list -----------------------------------
declare -a ARCHS=()
declare -a RAW_ARCHS=()
if [ -n "${CUROBO_GPU_ARCHS:-}" ]; then
  IFS=', ' read -r -a RAW_ARCHS <<< "${CUROBO_GPU_ARCHS}"
elif [ -n "${CUROBO_GPU_ARCH:-}" ] && [ "${CUROBO_GPU_ARCH}" != "auto" ] && [ "${CUROBO_GPU_ARCH}" != "current" ]; then
  RAW_ARCHS=("${CUROBO_GPU_ARCH}")
elif [ -n "${CUROBO_GPU_PROFILE:-}" ]; then
  RAW_ARCHS=("$(profile_arch "${CUROBO_GPU_PROFILE}")")
elif [ "${gpu_available}" = "1" ]; then
  detected="$(detect_arch_from_nvidia_smi || true)"
  RAW_ARCHS=("${detected:-sm_89}")
fi

for raw in "${RAW_ARCHS[@]}"; do
  [ -n "${raw}" ] || continue
  if normalized="$(normalize_arch "${raw}")"; then
    duplicate=0
    for existing in "${ARCHS[@]}"; do
      [ "${existing}" = "${normalized}" ] && duplicate=1 && break
    done
    [ "${duplicate}" = "1" ] || ARCHS+=("${normalized}")
  fi
done

# --- Content-addressed cache key -------------------------------------------
hash_of() {
  sha256sum "$@" | sha256sum | cut -c1-16
}

curobo_sha="$(git -C "${CUROBO_SOURCE}" rev-parse HEAD 2>/dev/null || true)"
[ -n "${curobo_sha}" ] || curobo_sha="no-git"
config_hash="$(hash_of "${ROBOT_CONFIG}" "${CUROBO_CONFIG}")"
exporter_hash="$(
  {
    sha256sum "${CUROBO_SOURCE}/cpp_runtime/robot/tools/export_robot.py" 2>/dev/null || true
    sha256sum "${CUROBO_SOURCE}/cpp_runtime/robot/tools/export_robot_aot.py" 2>/dev/null || true
    sha256sum "${CUROBO_SOURCE}/cpp_runtime/robot/tools/export_device.py" 2>/dev/null || true
    sha256sum "${SCRIPT_DIR}/write_artifact_metadata.py" 2>/dev/null || true
  } | sha256sum | cut -c1-16
)"
tool_frame_key="$(printf '%s\n' "${TOOL_FRAMES[@]}" | paste -sd, - 2>/dev/null || printf '%s\n' "${TOOL_FRAMES[@]}" | tr '\n' ',')"
if [ -n "${CUROBO_GPU_PROFILE:-}" ]; then
  profile_key="$(hash_of "${CUROBO_GPU_PROFILE}")"
else
  profile_key="local-gpu"
fi
arch_key="${ARCHS[*]:-none}"
key_material="${curobo_sha}|${config_hash}|${exporter_hash}|${ROBOT}|${ARTIFACT_SET}|${tool_frame_key}|${arch_key}|${profile_key}"
CACHE_KEY="v1-$(printf '%s' "${key_material}" | sha256sum | cut -c1-32)"
ARTIFACT_ROOT="${CACHE_ROOT}/${CACHE_KEY}"
ARTIFACT_SET_ROOT="${ARTIFACT_ROOT}/${ARTIFACT_SET}"
RELATIVE_ARTIFACT_DIR="${ARTIFACT_ROOT#${PIPELINE_DIR}/}"
ARCHS_ARG="$(printf '%s\n' "${ARCHS[@]}" | sed 's/^sm_//' | paste -sd, - 2>/dev/null || true)"

write_env() {
  local ready="$1"
  local arch="${ARCHS[0]:-sm_89}"
  local archs="${ARCHS_ARG:-89}"
  [ -n "${ENV_OUT}" ] || return 0
  mkdir -p "$(dirname "${ENV_OUT}")"
  {
    printf "CUROBO_RESOLVED_GPU_ARCH='%s'\n" "${arch}"
    printf "CUROBO_RESOLVED_CUDA_ARCHS='%s'\n" "${archs}"
    printf "CUROBO_RESOLVED_ARTIFACT_DIR='%s'\n" "${RELATIVE_ARTIFACT_DIR}"
    printf "CUROBO_RESOLVED_ARTIFACT_SET='%s'\n" "${ARTIFACT_SET}"
    printf "CUROBO_RESOLVED_READY='%s'\n" "${ready}"
  } > "${ENV_OUT}"
}

mkdir -p "${ARTIFACT_ROOT}"

# --- Skip cleanly when there is no GPU/profile ------------------------------
if [ "${gpu_available}" != "1" ]; then
  log "no CUDA GPU detected; native artifact export is required by default."
  log "run on an export GPU, set CUROBO_ASSUME_GPU=1, or set"
  log "CUROBO_REQUIRE_ARTIFACTS=0 only when a Python-only image is intended."
  if [ "${CUROBO_REQUIRE_ARTIFACTS:-1}" = "1" ]; then
    die "native artifacts are required but no CUDA GPU was detected"
  fi
  mkdir -p "${ARTIFACT_SET_ROOT}"
  : > "${ARTIFACT_SET_ROOT}/.no-gpu"
  write_env 0
  printf '%s\n' "${RELATIVE_ARTIFACT_DIR}"
  exit 0
fi

if [ "${#ARCHS[@]}" -eq 0 ]; then
  log "could not resolve a GPU architecture; native artifact export is required by default."
  if [ "${CUROBO_REQUIRE_ARTIFACTS:-1}" = "1" ]; then
    die "native artifacts are required but no GPU architecture could be resolved"
  fi
  mkdir -p "${ARTIFACT_SET_ROOT}"
  : > "${ARTIFACT_SET_ROOT}/.no-arch"
  write_env 0
  printf '%s\n' "${RELATIVE_ARTIFACT_DIR}"
  exit 0
fi

# --- Reuse a complete cache entry ------------------------------------------
missing=0
for arch in "${ARCHS[@]}"; do
  arch_dir="${ARTIFACT_SET_ROOT}/${arch}"
  if [ ! -f "${arch_dir}/robot_kinematics_manifest.json" ] || [ ! -f "${arch_dir}/manifest.json" ]; then
    missing=1
    break
  fi
done

if [ "${missing}" = "0" ]; then
  log "reusing cached artifacts: ${RELATIVE_ARTIFACT_DIR}"
  write_env 1
  printf '%s\n' "${RELATIVE_ARTIFACT_DIR}"
  exit 0
fi

# --- Build the GPU export helper image -------------------------------------
dockerfile="${PIPELINE_DIR}/docker/Dockerfile"
[ -f "${dockerfile}" ] || die "pipeline Dockerfile not found: ${dockerfile}"
export_image="${CUROBO_EXPORT_IMAGE:-}"
if [ -z "${export_image}" ]; then
  image_hash="$(
    {
      sha256sum "${dockerfile}"
      printf '%s\n' "${BASE_IMAGE:-}" "${ROS_DISTRO:-}" "${CUDA_TOOLKIT:-}"
    } | sha256sum | cut -c1-16
  )"
  export_image="curobo-native-export:${image_hash}"
fi

if ! docker image inspect "${export_image}" >/dev/null 2>&1; then
  log "building export helper image ${export_image} (target curobo-native-export)"
  build_args=()
  [ -n "${BASE_IMAGE}" ] && build_args+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
  [ -n "${ROS_DISTRO}" ] && build_args+=(--build-arg "ROS_DISTRO=${ROS_DISTRO}")
  [ -n "${CUDA_TOOLKIT}" ] && build_args+=(--build-arg "CUDA_TOOLKIT=${CUDA_TOOLKIT}")
  docker build \
    --network=host \
    --target curobo-native-export \
    ${build_args[@]+"${build_args[@]}"} \
    -t "${export_image}" \
    -f "${dockerfile}" \
    "${PIPELINE_DIR}"
fi

# --- Bind mounts ------------------------------------------------------------
declare -a MOUNT_ARGS=()
declare -a MOUNTED_PATHS=()
add_mount() {
  local path
  path="$(cd "$1" 2>/dev/null && pwd)" || return 0
  for existing in "${MOUNTED_PATHS[@]}"; do
    [ "${existing}" = "${path}" ] && return 0
  done
  MOUNTED_PATHS+=("${path}")
  MOUNT_ARGS+=(-v "${path}:${path}")
}

add_mount "${PIPELINE_DIR}"
for extra in "${EXTRA_MOUNTS[@]}"; do
  add_mount "${extra}"
done
add_mount "$(dirname "${ROBOT_CONFIG}")"
add_mount "$(dirname "${CUROBO_CONFIG}")"
if [ -n "${URDF}" ]; then
  add_mount "$(dirname "${URDF}")"
  if [ -n "${URDF_TARGET}" ]; then
    MOUNT_ARGS+=(-v "${URDF}:${URDF_TARGET}:ro")
  fi
fi
if [ -n "${CUROBO_GPU_PROFILE:-}" ]; then
  add_mount "$(dirname "${CUROBO_GPU_PROFILE}")"
fi

run_container() {
  docker run --rm \
    --entrypoint /usr/bin/env \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e XDG_CACHE_HOME=/tmp/.cache \
    -e CUDA_CACHE_PATH=/tmp/cuda-cache \
    -e PYTHONPATH=/opt/curobo-src \
    -e CUROBO_SOURCE_DIR=/opt/curobo-src \
    "${MOUNT_ARGS[@]}" \
    -w "${PIPELINE_DIR}" \
    "$@"
}

tmp_dir="${ARTIFACT_ROOT}/.tmp"
mkdir -p "${tmp_dir}"

make_profile_for_arch() {
  local arch="$1"
  local base_profile="$2"
  local out="$3"
  python3 - "${base_profile}" "${out}" "${arch}" <<'PY'
import json
import sys

source, destination, arch = sys.argv[1], sys.argv[2], sys.argv[3]
with open(source, "r", encoding="utf-8") as handle:
    data = json.load(handle)
data["arch"] = arch
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
PY
}

if [ -n "${CUROBO_GPU_PROFILE:-}" ]; then
  base_profile="${CUROBO_GPU_PROFILE}"
else
  base_profile="${tmp_dir}/device.json"
  log "probing export GPU with export_device.py"
  run_container --gpus "${gpu_request}" "${export_image}" \
    python3 /opt/curobo-src/cpp_runtime/robot/tools/export_device.py \
    --device 0 --out "${base_profile}"
fi

# --- Export each missing architecture --------------------------------------
export_failed=0
for arch in "${ARCHS[@]}"; do
  arch_dir="${ARTIFACT_SET_ROOT}/${arch}"
  if [ -f "${arch_dir}/robot_kinematics_manifest.json" ] && [ -f "${arch_dir}/manifest.json" ]; then
    log "reusing ${arch} artifacts"
    continue
  fi

  profile_for_arch="${tmp_dir}/device-${arch}.json"
  make_profile_for_arch "${arch}" "${base_profile}" "${profile_for_arch}"
  out_tmp="${ARTIFACT_SET_ROOT}/.${arch}.tmp.$$"
  rm -rf "${out_tmp}"

  log "exporting ${ROBOT} ${arch} -> ${RELATIVE_ARTIFACT_DIR}/${ARTIFACT_SET}/${arch}"
  export_args=(
    python3 /opt/curobo-src/cpp_runtime/robot/tools/export_robot.py
    --robot-config "${CUROBO_CONFIG}"
    --target-device "${profile_for_arch}"
    --out "${out_tmp}"
  )
  for frame in "${TOOL_FRAMES[@]}"; do
    export_args+=(--tool-frame "${frame}")
  done
  if [ -n "${URDF}" ]; then
    export_args+=(--urdf "${URDF}")
  fi
  if ! run_container --gpus "${gpu_request}" "${export_image}" "${export_args[@]}"; then
    log "WARNING: native export failed for ${arch}; keeping the Python runtime usable."
    rm -rf "${out_tmp}"
    mkdir -p "${arch_dir}"
    : > "${arch_dir}/.export-failed"
    export_failed=1
    continue
  fi

  metadata_args=(
    python3 "${SCRIPT_DIR}/write_artifact_metadata.py"
    --robot "${ROBOT}"
    --artifact-set "${ARTIFACT_SET}"
    --arch "${arch}"
    --robot-config "${ROBOT_CONFIG}"
    --curobo-config "${CUROBO_CONFIG}"
    --artifact-dir "${out_tmp}"
    --exporter-version "${exporter_hash}"
  )
  for frame in "${TOOL_FRAMES[@]}"; do
    metadata_args+=(--tool-frame "${frame}")
  done
  if [ -n "${URDF}" ]; then
    metadata_args+=(--urdf "${URDF}")
  fi
  if ! run_container "${export_image}" "${metadata_args[@]}"; then
    log "WARNING: artifact metadata generation failed for ${arch}."
    rm -rf "${out_tmp}"
    mkdir -p "${arch_dir}"
    : > "${arch_dir}/.export-failed"
    export_failed=1
    continue
  fi

  rm -rf "${arch_dir}"
  mv "${out_tmp}" "${arch_dir}"
done

rm -rf "${tmp_dir}"
if [ "${export_failed}" != "0" ]; then
  mkdir -p "${ARTIFACT_SET_ROOT}"
  : > "${ARTIFACT_SET_ROOT}/.export-failed"
  write_env 0
  if [ "${CUROBO_REQUIRE_ARTIFACTS:-1}" = "1" ]; then
    die "native artifact export failed and artifacts are required by default"
  fi
  log "artifact export incomplete; building a Python-only image because CUROBO_REQUIRE_ARTIFACTS=0."
  printf '%s\n' "${RELATIVE_ARTIFACT_DIR}"
  exit 0
fi
write_env 1
log "artifact bundle ready: ${RELATIVE_ARTIFACT_DIR}"
printf '%s\n' "${RELATIVE_ARTIFACT_DIR}"
