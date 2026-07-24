#!/usr/bin/env bash
# Run on the networked x86 development PC. This script only prepares an
# offline ARM64 bundle; it does not install packages on the PC or the NX.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="${SCRIPT_DIR}/jetson_offline_jp61"
WHEEL_DIR="${OFFLINE_DIR}/wheels"
DEB_DIR="${OFFLINE_DIR}/debs"
SRC_DIR="${OFFLINE_DIR}/src"
REQ_FILE="${SCRIPT_DIR}/requirements.jetson.txt"
SDIST_REQ_FILE="${SCRIPT_DIR}/requirements.jetson.sdist.txt"
YOLO26_REQ_FILE="${SCRIPT_DIR}/requirements.jetson.yolo26.txt"

PYTORCH3D_COMMIT="4daa00b41c52455440b938d1b676e00935b204d7"
NVDIFFRAST_COMMIT="253ac4fcea7de5f396371124af597e6cc957bfae"

mkdir -p "${WHEEL_DIR}" "${DEB_DIR}" "${SRC_DIR}"

cat <<EOF
Preparing NVIDIA PyTorch 24.09 / Ubuntu 22.04 / Python 3.10 ARM64 dependencies
  output:       ${OFFLINE_DIR}
  PyTorch3D:    ${PYTORCH3D_COMMIT}
  NVDiffRast:   ${NVDIFFRAST_COMMIT}

This downloads Python wheels, Ubuntu 22.04 ARM64 debs and two source archives.
The custom Open3D cp310 ARM64 wheel must already be present in wheels/.
It may consume several GB. Press Ctrl+C now to cancel, or Enter to continue.
EOF
read -r

if ! compgen -G "${WHEEL_DIR}/open3d-0.18.0-cp310-cp310-*_aarch64.whl" >/dev/null; then
  echo "ERROR: Missing custom Python 3.10 ARM64 Open3D wheel in ${WHEEL_DIR}." >&2
  echo "Keep the validated open3d-0.18.0 cp310 aarch64 wheel in the bundle." >&2
  exit 1
fi

open3d_wheels=("${WHEEL_DIR}"/open3d-0.18.0-cp310-cp310-*_aarch64.whl)
if (( ${#open3d_wheels[@]} != 1 )); then
  echo "ERROR: Expected exactly one Open3D cp310 ARM64 wheel, found ${#open3d_wheels[@]}." >&2
  exit 1
fi

# Build in a fresh staging tree so stale wheels, duplicate versions, amd64
# packages and interrupted APT files can never leak into the final bundle.
STAGING_DIR="$(mktemp -d "${SCRIPT_DIR}/.jetson_offline_jp61.XXXXXX")"
trap 'rm -rf "${STAGING_DIR}"' EXIT
STAGING_WHEEL_DIR="${STAGING_DIR}/wheels"
STAGING_DEB_DIR="${STAGING_DIR}/debs"
STAGING_SRC_DIR="${STAGING_DIR}/src"
STAGING_YOLO26_DIR="${STAGING_DIR}/yolo26"
mkdir -p "${STAGING_WHEEL_DIR}" "${STAGING_DEB_DIR}" "${STAGING_SRC_DIR}" "${STAGING_YOLO26_DIR}"
cp "${open3d_wheels[0]}" "${STAGING_WHEEL_DIR}/"

python3 -m pip download \
  --dest "${STAGING_WHEEL_DIR}" \
  --only-binary=:all: \
  --no-deps \
  --implementation cp \
  --python-version 310 \
  --abi cp310 \
  --platform manylinux2014_aarch64 \
  --platform manylinux_2_17_aarch64 \
  --platform manylinux_2_27_aarch64 \
  --platform manylinux_2_28_aarch64 \
  --platform manylinux_2_34_aarch64 \
  --requirement "${REQ_FILE}"

python3 -m pip download \
  --dest "${STAGING_WHEEL_DIR}" \
  --no-binary=:all: \
  --no-deps \
  --requirement "${SDIST_REQ_FILE}"

python3 -m pip download \
  --dest "${STAGING_YOLO26_DIR}" \
  --only-binary=:all: \
  --no-deps \
  --implementation cp \
  --python-version 310 \
  --abi cp310 \
  --platform manylinux2014_aarch64 \
  --platform manylinux_2_17_aarch64 \
  --requirement "${YOLO26_REQ_FILE}"

curl --fail --location --retry 3 \
  "https://github.com/facebookresearch/pytorch3d/archive/${PYTORCH3D_COMMIT}.tar.gz" \
  --output "${STAGING_SRC_DIR}/pytorch3d.tar.gz"
curl --fail --location --retry 3 \
  "https://github.com/NVlabs/nvdiffrast/archive/${NVDIFFRAST_COMMIT}.tar.gz" \
  --output "${STAGING_SRC_DIR}/nvdiffrast.tar.gz"

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif sudo docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
else
  echo "ERROR: Docker daemon is unavailable." >&2
  exit 1
fi

# APT runs on amd64, but resolves and downloads the Ubuntu 22.04 ARM64 closure
# without executing ARM binaries. Only two validated runtime roots and the
# mycpp build roots are requested; Dockerfile.jetson applies a stricter package
# whitelist when installing them.
"${DOCKER[@]}" run --rm \
  --volume "${STAGING_DEB_DIR}:/out" \
  ubuntu:22.04 \
  bash -euc '
    dpkg --add-architecture arm64
    rm -f /etc/apt/sources.list
    rm -rf /etc/apt/sources.list.d/*
    cat > /etc/apt/sources.list.d/ubuntu.sources <<"EOF"
Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports
Suites: jammy jammy-updates jammy-backports jammy-security
Components: main restricted universe multiverse
Architectures: arm64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
    apt-get update
    apt-get install -y --download-only \
      -o Dir::Cache::archives=/out \
      libgl1:arm64 \
      libc++1-14:arm64 \
      libeigen3-dev \
      libboost-system-dev:arm64 \
      libboost-program-options-dev:arm64
    rm -rf /out/partial /out/lock
    find /out -maxdepth 1 -type f -name "*.deb" -exec chmod 0644 {} +
  '

# Reject and remove packages for the development PC architecture.
while IFS= read -r -d '' deb; do
  arch="$(dpkg-deb --field "${deb}" Architecture)"
  if [[ "${arch}" != "arm64" && "${arch}" != "all" ]]; then
    echo "Removing non-ARM64 package: $(basename "${deb}") (${arch})"
    rm -f "${deb}"
  fi
done < <(find "${STAGING_DEB_DIR}" -maxdepth 1 -type f -name '*.deb' -print0)

REQUIRED_DEB_PACKAGES=(
  gcc-12-base libatomic1 libbsd0 libc++1-14 libc++abi1-14
  libdrm-amdgpu1 libdrm-common libdrm-nouveau2 libdrm-radeon1 libdrm2
  libedit2 libelf1 libffi8 libgl1-mesa-dri libgl1 libglapi-mesa libglvnd0
  libglx-mesa0 libglx0 libicu70 libllvm15 libmd0 libsensors-config
  libsensors5 libtinfo6 libudev1 libunwind-14 libx11-6 libx11-data
  libx11-xcb1 libxau6 libxcb-dri2-0 libxcb-dri3-0 libxcb-glx0
  libxcb-present0 libxcb-randr0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0
  libxcb1 libxdmcp6 libxext6 libxfixes3 libxml2 libxshmfence1
  libxxf86vm1 libusb-1.0-0 libzstd1 zlib1g
  libeigen3-dev libboost1.74-dev libboost-system1.74.0
  libboost-system1.74-dev libboost-system-dev
  libboost-program-options1.74.0 libboost-program-options1.74-dev
  libboost-program-options-dev
)

for wanted in "${REQUIRED_DEB_PACKAGES[@]}"; do
  matches=0
  while IFS= read -r -d '' deb; do
    [[ "$(dpkg-deb --field "${deb}" Package)" == "${wanted}" ]] && ((matches += 1))
  done < <(find "${STAGING_DEB_DIR}" -maxdepth 1 -type f -name '*.deb' -print0)
  if (( matches != 1 )); then
    echo "ERROR: Expected one ARM64/all deb for ${wanted}, found ${matches}." >&2
    exit 1
  fi
done

(
  cd "${STAGING_DIR}"
  find debs src wheels yolo26 -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

BACKUP_DIR="$(mktemp -d "${SCRIPT_DIR}/.jetson_offline_backup.XXXXXX")"
for old_dir in "${WHEEL_DIR}" "${DEB_DIR}" "${SRC_DIR}" "${OFFLINE_DIR}/yolo26"; do
  if [[ -e "${old_dir}" ]]; then
    mv "${old_dir}" "${BACKUP_DIR}/"
  fi
done
mv "${STAGING_WHEEL_DIR}" "${WHEEL_DIR}"
mv "${STAGING_DEB_DIR}" "${DEB_DIR}"
mv "${STAGING_SRC_DIR}" "${SRC_DIR}"
mv "${STAGING_YOLO26_DIR}" "${OFFLINE_DIR}/yolo26"
mv "${STAGING_DIR}/SHA256SUMS" "${OFFLINE_DIR}/SHA256SUMS"
find "${OFFLINE_DIR}" -type f -exec chmod a+r {} +
trap - EXIT
rm -rf "${STAGING_DIR}"
if ! rm -rf "${BACKUP_DIR}"; then
  echo "WARNING: Could not remove old root-owned cache: ${BACKUP_DIR}" >&2
fi

echo "Offline bundle ready: ${OFFLINE_DIR}"
du -sh "${OFFLINE_DIR}"