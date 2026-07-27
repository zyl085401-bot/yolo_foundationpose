#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:24.09-py3-igpu}"
IMAGE_NAME="${IMAGE_NAME:-foundationpose-jetson:pt2409}"
MAX_JOBS="${MAX_JOBS:-1}"
OFFLINE_DIR="${SCRIPT_DIR}/jetson_offline_jp61"

HOST_ARCH="$(uname -m)"
if [[ "${HOST_ARCH}" != "aarch64" && "${HOST_ARCH}" != "arm64" ]]; then
  echo "ERROR: This image must be built natively on the Jetson ARM64 device (detected: ${HOST_ARCH})." >&2
  exit 1
fi

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif sudo docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
else
  echo "ERROR: Docker daemon is unavailable." >&2
  exit 1
fi

if ! "${DOCKER[@]}" image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: Base image is not loaded: ${BASE_IMAGE}
Import the transferred image first, for example:
  sudo docker load -i ~/pytorch-24.09-py3-igpu-arm64.tar
EOF
  exit 1
fi

if [[ ! "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_JOBS must be a positive integer." >&2
  exit 1
fi

REQUIRED_OFFLINE_FILES=(
  "${OFFLINE_DIR}/src/pytorch3d.tar.gz"
  "${OFFLINE_DIR}/src/nvdiffrast.tar.gz"
  "${OFFLINE_DIR}/yolo26/ultralytics-8.4.66-py3-none-any.whl"
  "${OFFLINE_DIR}/yolo26/ultralytics_thop-2.0.20-py3-none-any.whl"
  "${OFFLINE_DIR}/yolo26/nvidia_ml_py-13.610.43-py3-none-any.whl"
  "${OFFLINE_DIR}/yolo26/polars-1.43.0-py3-none-any.whl"
  "${OFFLINE_DIR}/yolo26/polars_runtime_32-1.43.0-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
  "${OFFLINE_DIR}/gui/opencv_python-4.7.0.72-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
)
for required_file in "${REQUIRED_OFFLINE_FILES[@]}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "ERROR: Missing offline dependency: ${required_file}" >&2
    echo "Run prepare_jetson_offline.sh on the development PC, then sync jetson_offline_jp61/ to the Jetson." >&2
    exit 1
  fi
done
if ! compgen -G "${OFFLINE_DIR}/wheels/*.whl" >/dev/null; then
  echo "ERROR: Offline wheelhouse is empty: ${OFFLINE_DIR}/wheels" >&2
  exit 1
fi
if ! compgen -G "${OFFLINE_DIR}/debs/*.deb" >/dev/null; then
  echo "ERROR: Offline deb directory is empty: ${OFFLINE_DIR}/debs" >&2
  exit 1
fi
if ! compgen -G "${OFFLINE_DIR}/wheels/open3d-0.18.0-cp310-cp310-*_aarch64.whl" >/dev/null; then
  echo "ERROR: Python 3.10 ARM64 Open3D wheel is missing: ${OFFLINE_DIR}/wheels" >&2
  exit 1
fi

REQUIRED_WHEELS=(
  addict-2.4.0-py3-none-any.whl
  configargparse-1.7.1-py3-none-any.whl
  dash-2.18.2-py3-none-any.whl
  nbformat-5.10.4-py3-none-any.whl
  pyquaternion-0.9.9-py3-none-any.whl
  werkzeug-3.0.6-py3-none-any.whl
)
for wheel in "${REQUIRED_WHEELS[@]}"; do
  if [[ ! -s "${OFFLINE_DIR}/wheels/${wheel}" ]]; then
    echo "ERROR: Missing validated Open3D dependency: ${OFFLINE_DIR}/wheels/${wheel}" >&2
    exit 1
  fi
done

REQUIRED_DEB_PACKAGES=(
  libgl1 libc++1-14 libusb-1.0-0 libeigen3-dev libboost1.74-dev
  libboost-system-dev libboost-program-options-dev
)
for wanted in "${REQUIRED_DEB_PACKAGES[@]}"; do
  matches=0
  while IFS= read -r -d '' deb; do
    package="$(dpkg-deb --field "${deb}" Package)"
    arch="$(dpkg-deb --field "${deb}" Architecture)"
    if [[ "${package}" == "${wanted}" && ( "${arch}" == "arm64" || "${arch}" == "all" ) ]]; then
      ((matches += 1))
    fi
  done < <(find "${OFFLINE_DIR}/debs" -maxdepth 1 -type f -name '*.deb' -print0)
  if (( matches != 1 )); then
    echo "ERROR: Expected one ARM64/all deb for ${wanted}, found ${matches}." >&2
    exit 1
  fi
done

for wanted in x11-common libice6 libsm6; do
  matches=0
  while IFS= read -r -d '' deb; do
    package="$(dpkg-deb --field "${deb}" Package)"
    arch="$(dpkg-deb --field "${deb}" Architecture)"
    if [[ "${package}" == "${wanted}" && ( "${arch}" == "arm64" || "${arch}" == "all" ) ]]; then
      ((matches += 1))
    fi
  done < <(find "${OFFLINE_DIR}/gui/debs" -maxdepth 1 -type f -name '*.deb' -print0)
  if (( matches != 1 )); then
    echo "ERROR: Expected one ARM64/all GUI deb for ${wanted}, found ${matches}." >&2
    exit 1
  fi
done

if [[ -s "${OFFLINE_DIR}/SHA256SUMS" ]]; then
  echo "Verifying offline bundle checksums..."
  (cd "${OFFLINE_DIR}" && sha256sum --check --quiet SHA256SUMS)
else
  echo "WARNING: SHA256SUMS is absent; continuing with the previously prepared bundle." >&2
fi

cat <<EOF
Building ${IMAGE_NAME}
  base image:    ${BASE_IMAGE}
  Python:        3.10
  CUDA arch:     8.7 (Orin)
  parallel jobs: ${MAX_JOBS}
  network:       disabled (offline build)
EOF

BUILD_PROGRESS_ARGS=()
if "${DOCKER[@]}" build --help 2>&1 | grep -q -- '--progress'; then
  BUILD_PROGRESS_ARGS=(--progress plain)
else
  echo "Docker legacy builder detected; using a minimal streamed build context."
fi

# Stream only files referenced by Dockerfile.jetson. This keeps the legacy
# builder from sending weights, outputs or the multi-GB base-image tarball.
tar --create --file - --directory "${REPO_ROOT}" \
  FoundationPose/docker/Dockerfile.jetson \
  FoundationPose/docker/requirements.jetson.txt \
  FoundationPose/docker/requirements.jetson.sdist.txt \
  FoundationPose/docker/requirements.jetson.yolo26.txt \
  FoundationPose/docker/requirements.jetson.gui.txt \
  FoundationPose/docker/jetson_offline_jp61 \
  FoundationPose/mycpp \
| "${DOCKER[@]}" build \
  --network none \
  "${BUILD_PROGRESS_ARGS[@]}" \
  --file FoundationPose/docker/Dockerfile.jetson \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "MAX_JOBS=${MAX_JOBS}" \
  --tag "${IMAGE_NAME}" \
  -

"${DOCKER[@]}" image inspect "${IMAGE_NAME}" \
  --format 'Built {{.RepoTags}} ({{.Architecture}}, {{.Size}} bytes)'

cat <<EOF
Build complete.
Start an interactive container with:
  bash FoundationPose/docker/run_container_jetson.sh
EOF
