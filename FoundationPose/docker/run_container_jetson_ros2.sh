#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-foundationpose-jetson:pt2409-ros2}"
CONTAINER_NAME="${CONTAINER_NAME:-foundationpose-jetson-ros2}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/workspace/yolo_foundationpose/FoundationPose/docker/fastdds_udp_only.xml}"

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif sudo docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
else
  echo "ERROR: Docker daemon is unavailable." >&2
  exit 1
fi

if ! "${DOCKER[@]}" image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "ERROR: Image ${IMAGE_NAME} does not exist. Run build_jetson_ros2.sh first." >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/realtime_foundation/outputs"

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

DISPLAY_ARGS=()
if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
  DISPLAY_ARGS=(
    --env "DISPLAY=${DISPLAY}"
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw
  )
fi

if [[ $# -eq 0 ]]; then
  COMMAND=(bash)
else
  COMMAND=("$@")
fi

exec "${DOCKER[@]}" run --rm "${TTY_ARGS[@]}" \
  --name "${CONTAINER_NAME}" \
  --runtime nvidia \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --security-opt seccomp=unconfined \
  --volume /dev:/dev \
  --volume /run/udev:/run/udev:ro \
  --volume "${REPO_ROOT}:/workspace/yolo_foundationpose:rw" \
  --workdir /workspace/yolo_foundationpose \
  --env CUDA_HOME=/usr/local/cuda \
  --env TORCH_CUDA_ARCH_LIST=8.7 \
  --env OPENCV_IO_ENABLE_OPENEXR=1 \
  --env PYTHONUNBUFFERED=1 \
  --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}" \
  --env "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}" \
  --env "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}" \
  --env "FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}" \
  "${DISPLAY_ARGS[@]}" \
  "${IMAGE_NAME}" \
  "${COMMAND[@]}"