#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
IMAGE="${IMAGE:-foundationpose-jetson:pt2409}"
MAX_JOBS="${MAX_JOBS:-1}"
FULL_REBUILD_TEST="${FULL_REBUILD_TEST:-0}"

# The default test validates extensions already compiled into the final image.
# Set FULL_REBUILD_TEST=1 and IMAGE=nvcr.io/nvidia/pytorch:24.09-py3-igpu only
# when an isolated clean rebuild of every CUDA extension is explicitly needed.
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
    --ipc host \
    --volume "${REPO_ROOT}:/workspace/yolo_foundationpose:ro" \
    "${IMAGE}" \
    python3 -c 'import platform, torch, mycpp, nvdiffrast.torch as dr, pytorch3d; from pytorch3d import _C; from pytorch3d.ops import sample_farthest_points; assert platform.machine() in {"aarch64", "arm64"}; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability(0) == (8, 7); ctx=dr.RasterizeCudaContext(); points=torch.rand(1,32,3,device="cuda"); sampled,indices=sample_farthest_points(points,K=4); torch.cuda.synchronize(); assert sampled.shape == (1,4,3); print("Python/PyTorch:",platform.python_version(),torch.__version__); print("GPU:",torch.cuda.get_device_name(0)); print("PyTorch3D:",_C.__file__); print("NVDiffRast:",dr.__file__); print("MyCPP:",mycpp.__file__); print("FOUNDATIONPOSE CUDA EXTENSIONS TEST: PASS")'
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

if [[ ! "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_JOBS must be a positive integer." >&2
  exit 1
fi

required_files=(
  "${OFFLINE_DIR}/src/pytorch3d.tar.gz"
  "${OFFLINE_DIR}/src/nvdiffrast.tar.gz"
  "${REPO_ROOT}/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth"
  "${REPO_ROOT}/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "ERROR: Missing required file: ${required_file}" >&2
    exit 1
  fi
done

if [[ ! -d "${OFFLINE_DIR}/wheels" ]]; then
  echo "ERROR: Missing offline wheelhouse: ${OFFLINE_DIR}/wheels" >&2
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
Testing FoundationPose CUDA extensions on Jetson
  image:      ${IMAGE}
  offline:    ${OFFLINE_DIR}
  repository: ${REPO_ROOT}
  CUDA arch:  8.7
  build jobs: ${MAX_JOBS}
  network:    disabled

PyTorch3D compilation can take several minutes. MAX_JOBS=1 limits memory use.
EOF

"${DOCKER[@]}" run --rm -i \
  --runtime nvidia \
  --network none \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env CUDA_HOME=/usr/local/cuda \
  --env TORCH_CUDA_ARCH_LIST=8.7 \
  --env FORCE_CUDA=1 \
  --env "MAX_JOBS=${MAX_JOBS}" \
  --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  --volume "${OFFLINE_DIR}:/offline:ro" \
  --volume "${REPO_ROOT}:/workspace/yolo_foundationpose:ro" \
  "${IMAGE}" \
  bash --noprofile --norc -s <<'CONTAINER_SCRIPT'
set -eo pipefail

export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=8.7
export FORCE_CUDA=1
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions

python3 - <<'PY'
import platform
import torch

print("Architecture:", platform.machine())
print("Python/PyTorch:", platform.python_version(), torch.__version__)
print("CUDA/GPU:", torch.version.cuda, torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))

assert platform.machine() in {"aarch64", "arm64"}
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (8, 7)
PY

python3 - <<'PY'
from collections import OrderedDict
from pathlib import Path
import torch

root = Path("/workspace/yolo_foundationpose/FoundationPose/weights")
checkpoints = [
    root / "2023-10-28-18-33-37/model_best.pth",
    root / "2024-01-11-20-02-45/model_best.pth",
]

for checkpoint_path in checkpoints:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    assert isinstance(state_dict, (dict, OrderedDict))
    assert state_dict
    assert all(isinstance(key, str) for key in state_dict)
    assert all(torch.is_tensor(value) for value in state_dict.values())
    print(
        "Checkpoint:",
        checkpoint_path.name,
        "top-level=",
        type(checkpoint).__name__,
        "tensors=",
        len(state_dict),
    )

print("PYTORCH 2.5 CHECKPOINT TEST: PASS")
PY

shopt -s nullglob
ninja_wheels=(/offline/wheels/ninja-1.11.1.3-*.whl)
iopath_sdists=(/offline/wheels/iopath-0.1.10.tar.gz)
fvcore_sdists=(/offline/wheels/fvcore-0.1.5.post20221221.tar.gz)

for package_group in ninja_wheels iopath_sdists fvcore_sdists; do
  declare -n packages="${package_group}"
  if (( ${#packages[@]} != 1 )); then
    echo "ERROR: Expected exactly one file for ${package_group}, found ${#packages[@]}." >&2
    exit 1
  fi
done

python3 -m pip install \
  --no-index \
  --find-links=/offline/wheels \
  --no-build-isolation \
  "${ninja_wheels[0]}" \
  "${iopath_sdists[0]}" \
  "${fvcore_sdists[0]}"

rm -rf /tmp/src /tmp/torch_extensions
mkdir -p /tmp/src/pytorch3d /tmp/src/nvdiffrast /tmp/torch_extensions

tar -xzf /offline/src/pytorch3d.tar.gz \
  --strip-components=1 \
  -C /tmp/src/pytorch3d

tar -xzf /offline/src/nvdiffrast.tar.gz \
  --strip-components=1 \
  -C /tmp/src/nvdiffrast

echo "Building and installing NVDiffRast..."
python3 -m pip install \
  --no-index \
  --no-build-isolation \
  --no-deps \
  /tmp/src/nvdiffrast

python3 - <<'PY'
import torch
import nvdiffrast.torch as dr

print("NVDiffRast module:", dr.__file__)
context = dr.RasterizeCudaContext()

positions = torch.tensor(
    [
        [-0.8, -0.8, 0.0, 1.0],
        [0.8, -0.8, 0.0, 1.0],
        [0.0, 0.8, 0.0, 1.0],
    ],
    dtype=torch.float32,
    device="cuda",
).unsqueeze(0)
triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")

raster, derivatives = dr.rasterize(
    context,
    positions,
    triangles,
    resolution=[64, 64],
)
torch.cuda.synchronize()

covered_pixels = int((raster[..., 3] > 0).sum().item())
assert raster.shape == (1, 64, 64, 4)
assert derivatives.shape[:3] == (1, 64, 64)
assert torch.isfinite(raster).all()
assert covered_pixels > 0

print("Raster shape/covered pixels:", tuple(raster.shape), covered_pixels)
print("NVDIFFRAST CUDA TEST: PASS")
PY

echo "Building and installing PyTorch3D..."
python3 -m pip install \
  --no-index \
  --no-build-isolation \
  --no-deps \
  /tmp/src/pytorch3d

python3 - <<'PY'
import torch
import pytorch3d
from pytorch3d import _C
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import matrix_to_axis_angle

print("PyTorch3D module:", pytorch3d.__file__)
print("PyTorch3D native extension:", _C.__file__)

points = torch.rand(2, 100, 3, dtype=torch.float32, device="cuda")
sampled, indices = sample_farthest_points(points, K=10)
rotation = torch.eye(3, dtype=torch.float32, device="cuda").unsqueeze(0)
axis_angle = matrix_to_axis_angle(rotation)
torch.cuda.synchronize()

assert sampled.shape == (2, 10, 3)
assert indices.shape == (2, 10)
assert torch.isfinite(sampled).all()
assert torch.allclose(axis_angle, torch.zeros_like(axis_angle), atol=1e-6)

print("Sampled/indices shapes:", tuple(sampled.shape), tuple(indices.shape))
print("PYTORCH3D CUDA TEST: PASS")
PY

python3 - <<'PY'
import nvdiffrast.torch
import pytorch3d
import torch

print("Extension summary:")
print("  PyTorch:", torch.__version__)
print("  CUDA:", torch.version.cuda)
print("  NVDiffRast:", nvdiffrast.torch.__file__)
print("  PyTorch3D:", pytorch3d.__file__)
print("FOUNDATIONPOSE CUDA EXTENSIONS TEST: PASS")
PY
CONTAINER_SCRIPT
