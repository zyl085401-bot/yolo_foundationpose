#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-foundationpose-jetson:pt2409}"
IMAGE_NAME="${IMAGE_NAME:-foundationpose-jetson:pt2409-ros2}"
ROS_APT_MIRROR="${ROS_APT_MIRROR:-http://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu}"

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif sudo docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
else
  echo "ERROR: Docker daemon is unavailable." >&2
  exit 1
fi

if ! "${DOCKER[@]}" image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  echo "ERROR: Base image ${BASE_IMAGE} does not exist." >&2
  exit 1
fi

for file in Dockerfile.jetson.ros2 ros2_entrypoint.sh ros-archive-keyring.gpg; do
  if [[ ! -f "${SCRIPT_DIR}/${file}" ]]; then
    echo "ERROR: Missing build file: ${SCRIPT_DIR}/${file}" >&2
    exit 1
  fi
done

echo "Building ${IMAGE_NAME} from ${BASE_IMAGE}"
echo "ROS apt mirror: ${ROS_APT_MIRROR}"

# Stream a minimal context so the large Jetson offline bundle is not sent to
# the Docker daemon. This also works with the NX legacy Docker builder.
tar -C "${SCRIPT_DIR}" -cf - Dockerfile.jetson.ros2 ros2_entrypoint.sh ros-archive-keyring.gpg \
  | "${DOCKER[@]}" build \
      --network host \
      --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
      --build-arg "ROS_APT_MIRROR=${ROS_APT_MIRROR}" \
      --tag "${IMAGE_NAME}" \
      --file Dockerfile.jetson.ros2 \
      -

echo "Validating ROS 2 and the existing CUDA stack..."
"${DOCKER[@]}" run --rm \
  --runtime nvidia \
  --network host \
  "${IMAGE_NAME}" \
  python3 -c 'import cv2, geometry_msgs, message_filters, mycpp, nvdiffrast.torch, pytorch3d._C, rclpy, sensor_msgs, torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability() == (8, 7); print("ROS 2 FoundationPose image: PASS"); print("torch:", torch.__version__); print("cv2:", cv2.__file__); print("GPU:", torch.cuda.get_device_name(0))'

echo "Built and validated: ${IMAGE_NAME}"