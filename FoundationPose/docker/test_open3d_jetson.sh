#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-foundationpose-jetson:pt2409}"
FULL_REBUILD_TEST="${FULL_REBUILD_TEST:-0}"

# The default path tests Open3D already installed in the final image. Keep the
# slower dependency-installation path below only for an explicit clean test.
if [[ "${FULL_REBUILD_TEST}" == "0" ]]; then
  if docker info >/dev/null 2>&1; then
    DOCKER=(docker)
  elif sudo docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  else
    echo "ERROR: Docker daemon is unavailable." >&2
    exit 1
  fi

  if ! "${DOCKER[@]}" image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: Built image is unavailable: ${IMAGE}" >&2
    echo "Run build_jetson.sh first." >&2
    exit 1
  fi

  exec "${DOCKER[@]}" run --rm \
    --runtime nvidia \
    --network none \
    "${IMAGE}" \
    python3 -c 'import platform, numpy as np, open3d as o3d, torch; points=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],[0.,0.,.01]]); cloud=o3d.geometry.PointCloud(); cloud.points=o3d.utility.Vector3dVector(points); cloud.estimate_normals(); assert len(cloud.normals)==4; value=(torch.tensor([2.],device="cuda")*3).item(); assert value==6.; print("Python:",platform.python_version()); print("Open3D:",o3d.__version__); print("GPU:",torch.cuda.get_device_name(0)); print("OPEN3D ARM64 TEST: PASS")'
fi

if [[ $# -gt 1 ]]; then
  echo "Usage: bash $0 [offline-bundle-directory]" >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  OFFLINE_DIR="$1"
elif [[ -d "${SCRIPT_DIR}/jetson_offline_jp61" ]]; then
  OFFLINE_DIR="${SCRIPT_DIR}/jetson_offline_jp61"
elif [[ -d "${HOME}/workspace/jetson_transfer/jetson_offline_jp61" ]]; then
  OFFLINE_DIR="${HOME}/workspace/jetson_transfer/jetson_offline_jp61"
else
  echo "ERROR: Cannot find jetson_offline_jp61." >&2
  echo "Pass its path explicitly, for example:" >&2
  echo "  bash $0 ~/workspace/jetson_transfer/jetson_offline_jp61" >&2
  exit 1
fi

OFFLINE_DIR="$(realpath "${OFFLINE_DIR}")"

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: This test must run natively on the Jetson ARM64 device." >&2
  exit 1
fi

if [[ ! -d "${OFFLINE_DIR}/debs" || ! -d "${OFFLINE_DIR}/wheels" ]]; then
  echo "ERROR: Invalid offline bundle: ${OFFLINE_DIR}" >&2
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

if ! "${DOCKER[@]}" image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "ERROR: Docker image is not loaded: ${IMAGE}" >&2
  exit 1
fi

cat <<EOF
Testing Open3D on Jetson
  image:   ${IMAGE}
  offline: ${OFFLINE_DIR}
  network: disabled
EOF

"${DOCKER[@]}" run --rm -i \
  --runtime nvidia \
  --network none \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume "${OFFLINE_DIR}:/offline:ro" \
  "${IMAGE}" \
  bash --noprofile --norc -s <<'CONTAINER_SCRIPT'
set -eo pipefail

REQUIRED_PACKAGES="
gcc-12-base
libicu70
libmd0
libbsd0
libxau6
libxdmcp6
libxcb1
libx11-data
libx11-6
libxext6
libdrm-common
libdrm2
libdrm-amdgpu1
libdrm-nouveau2
libdrm-radeon1
libedit2
libelf1
libffi8
libatomic1
libtinfo6
libxml2
libzstd1
zlib1g
libllvm15
libsensors-config
libsensors5
libudev1
libglapi-mesa
libx11-xcb1
libxcb-dri2-0
libxcb-dri3-0
libxcb-glx0
libxcb-present0
libxcb-randr0
libxcb-shm0
libxcb-sync1
libxcb-xfixes0
libxfixes3
libxshmfence1
libxxf86vm1
libglvnd0
libgl1-mesa-dri
libglx-mesa0
libglx0
libgl1
libunwind-14
libc++abi1-14
libc++1-14
"

declare -A required=()
declare -A selected_by_package=()
selected_debs=()

for package in ${REQUIRED_PACKAGES}; do
  required["${package}"]=1
done

shopt -s nullglob
all_debs=(/offline/debs/*.deb)
if (( ${#all_debs[@]} == 0 )); then
  echo "ERROR: No deb packages found in /offline/debs." >&2
  exit 1
fi

for deb in "${all_debs[@]}"; do
  package="$(dpkg-deb --field "${deb}" Package)"
  architecture="$(dpkg-deb --field "${deb}" Architecture)"

  if [[ -n "${required[${package}]:-}" ]] \
      && [[ "${architecture}" == "arm64" || "${architecture}" == "all" ]]; then
    if [[ -n "${selected_by_package[${package}]:-}" ]]; then
      echo "ERROR: Multiple suitable debs found for ${package}:" >&2
      echo "  ${selected_by_package[${package}]}" >&2
      echo "  ${deb}" >&2
      exit 1
    fi
    selected_by_package["${package}"]="${deb}"
    selected_debs+=("${deb}")
  fi
done

missing=0
for package in ${REQUIRED_PACKAGES}; do
  if [[ -z "${selected_by_package[${package}]:-}" ]]; then
    echo "ERROR: Missing ARM64/all deb for package: ${package}" >&2
    missing=1
  fi
done
if (( missing != 0 )); then
  exit 1
fi

echo "Selected ${#selected_debs[@]} system packages:"
printf '  %s\n' "${selected_debs[@]##*/}"

dpkg --unpack "${selected_debs[@]}"
dpkg --configure -a
ldconfig

python3 - <<'PY'
import ntpath
import pathlib
print("Python standard library: OK")
PY

required_sonames=(libX11.so.6 libc++.so.1 libGL.so.1)
for soname in "${required_sonames[@]}"; do
  if ! ldconfig -p | grep -Fq "${soname}"; then
    echo "ERROR: Required shared library is unavailable: ${soname}" >&2
    exit 1
  fi
done

echo "Required Open3D shared libraries: OK"

addict_wheels=(/offline/wheels/addict-2.4.0-py3-none-any.whl)
configargparse_wheels=(/offline/wheels/configargparse-1.7.1-py3-none-any.whl)
pyquaternion_wheels=(/offline/wheels/pyquaternion-0.9.9-py3-none-any.whl)
open3d_wheels=(/offline/wheels/open3d-0.18.0-*aarch64.whl)
dash_wheels=(/offline/wheels/dash-2.18.2-py3-none-any.whl)
werkzeug_wheels=(/offline/wheels/werkzeug-3.0.6-py3-none-any.whl)
nbformat_wheels=(/offline/wheels/nbformat-5.10.4-py3-none-any.whl)

for wheel_group in \
    addict_wheels \
    configargparse_wheels \
    pyquaternion_wheels \
    open3d_wheels \
    dash_wheels \
    werkzeug_wheels \
    nbformat_wheels; do
  declare -n wheels="${wheel_group}"
  if (( ${#wheels[@]} != 1 )); then
    echo "ERROR: Expected exactly one wheel for ${wheel_group}, found ${#wheels[@]}." >&2
    exit 1
  fi
done

python3 -m pip install \
  --no-index \
  --find-links=/offline/wheels \
  "${addict_wheels[0]}" \
  "${configargparse_wheels[0]}" \
  "${pyquaternion_wheels[0]}" \
  "${open3d_wheels[0]}" \
  "${dash_wheels[0]}" \
  "${werkzeug_wheels[0]}" \
  "${nbformat_wheels[0]}"

open3d_so="$(find /usr/local/lib/python3.10/dist-packages/open3d \
  -type f -path '*/cpu/pybind*.so' -print -quit)"
if [[ -z "${open3d_so}" ]]; then
  echo "ERROR: Open3D native extension was not installed." >&2
  exit 1
fi

missing_libraries="$(ldd "${open3d_so}" | grep 'not found' || true)"
if [[ -n "${missing_libraries}" ]]; then
  echo "ERROR: Open3D still has unresolved shared libraries:" >&2
  echo "${missing_libraries}" >&2
  exit 1
fi

python3 - <<'PY'
import platform
import numpy as np
import open3d as o3d
import torch

print("Architecture:", platform.machine())
print("Python/PyTorch:", platform.python_version(), torch.__version__)
print("CUDA/GPU:", torch.version.cuda, torch.cuda.get_device_name(0))
print("NumPy/Open3D:", np.__version__, o3d.__version__)

points = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
    ],
    dtype=np.float64,
)

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)
cloud.estimate_normals()
downsampled = cloud.voxel_down_sample(voxel_size=0.005)

assert len(cloud.points) == 4
assert len(cloud.normals) == 4
assert len(downsampled.points) > 0

cuda_value = torch.tensor([2.0], device="cuda") * 3.0
torch.cuda.synchronize()
assert cuda_value.item() == 6.0

print("Points/Normals/Downsampled:", len(cloud.points), len(cloud.normals), len(downsampled.points))
print("OPEN3D ARM64 TEST: PASS")
PY
CONTAINER_SCRIPT
