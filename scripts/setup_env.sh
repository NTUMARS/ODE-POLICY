#!/usr/bin/env bash
# One-time environment setup: Python 3.10 + PyTorch + robosuite / MuJoCo + LIBERO + the
# training stack, with the versions every result in the paper was produced with.
#
#   bash scripts/setup_env.sh                 # conda env "ode_policy"
#   conda activate ode_policy
#
#   ENV_NAME=myenv bash scripts/setup_env.sh             # another env name
#   ENV_PREFIX=/path/to/env bash scripts/setup_env.sh    # an env at a path instead
#   TORCH_CUDA=cu118 bash scripts/setup_env.sh           # another CUDA build (cu118 / cu121 / cpu)
set -euo pipefail

ENV_NAME="${ENV_NAME:-ode_policy}"
ENV_PREFIX="${ENV_PREFIX:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
# the LIBERO commit the paper's results were produced with
LIBERO_REF="${LIBERO_REF:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ -n "${ENV_PREFIX}" ]]; then
    ENV_ARGS=(--prefix "${ENV_PREFIX}"); ENV_REF="${ENV_PREFIX}"
else
    ENV_ARGS=(-n "${ENV_NAME}"); ENV_REF="${ENV_NAME}"
fi
if { [[ -n "${ENV_PREFIX}" ]] && [[ -d "${ENV_PREFIX}/bin" ]]; } || \
   { [[ -z "${ENV_PREFIX}" ]] && conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; }; then
    echo "[setup] env ${ENV_REF} already exists, reusing."
else
    echo "[setup] creating conda env ${ENV_REF} (python=${PYTHON_VERSION})"
    conda create "${ENV_ARGS[@]}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_REF}"

echo "[setup] installing torch ${TORCH_VERSION} (${TORCH_CUDA})"
pip install --upgrade pip
if [[ "${TORCH_CUDA}" == "cpu" ]]; then
    pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
else
    pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi
# keep pip from replacing the pins above while installing the rest
CONSTRAINTS="$(mktemp)"
printf 'torch==%s\ntorchvision==%s\nnumpy<2\n' "${TORCH_VERSION}" "${TORCHVISION_VERSION}" > "${CONSTRAINTS}"

echo "[setup] installing robosuite + mujoco + IL deps"
# Version picks (each one is load-bearing — see README "Troubleshooting"):
#   - numpy<2          : robosuite 1.4.1 / mujoco 2.3.5 predate the numpy 2 ABI break.
#   - zarr 2.x         : ode_policy/utils/replay_buffer.py uses
#                        zarr.DirectoryStore / zarr.copy_store / zarr.MemoryStore,
#                        all REMOVED in zarr 3. Do not relax this pin.
#   - diffusers>=0.30  : 0.27.2 imports `cached_download`, removed in
#                        huggingface_hub>=0.26.
#   - pandas, torchcfm : imported at runtime by json_logger.py and the a2a/fm
#                        policy families; absent from every requirements file.
#   - transformers<5   : frozen CLIP text tower for the language-conditioning path
#                        (ode_policy/libero/text_encoder.py); 5.x needs torch>=2.6.
pip install -c "${CONSTRAINTS}" \
    "numpy<2" \
    "robosuite==1.4.1" \
    "mujoco==2.3.5" \
    h5py \
    "hydra-core>=1.3" omegaconf \
    "zarr==2.17.2" "numcodecs==0.12.1" \
    numba dill loguru einops tqdm \
    "diffusers>=0.30,<0.40" timm scikit-learn scipy \
    pandas torchcfm "transformers<5" \
    "imageio[ffmpeg]" matplotlib opencv-python-headless

echo "[setup] installing LIBERO via editable clone"
# LIBERO's pyproject is not shaped for `pip install git+...` (the resulting wheel
# is empty — the source tree is not packaged). Clone + editable is the path the
# official README documents.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pip install -c "${CONSTRAINTS}" huggingface_hub
# this repository itself, editable (no dependencies: they are all pinned above)
pip install -e "${REPO_ROOT}" --no-deps
LIBERO_DIR="${LIBERO_DIR:-${REPO_ROOT}/third_party/LIBERO}"
mkdir -p "$(dirname "${LIBERO_DIR}")"
if [[ ! -d "${LIBERO_DIR}/.git" ]]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
else
    echo "[setup] LIBERO already cloned at ${LIBERO_DIR}"
fi
git -C "${LIBERO_DIR}" checkout -q "${LIBERO_REF}"
pushd "${LIBERO_DIR}" > /dev/null
# --no-deps: LIBERO's requirements.txt would clobber our pinned
# robosuite / mujoco / numpy. Its true runtime imports are listed below.
pip install -e . --no-deps
pip install -c "${CONSTRAINTS}" "bddl==1.0.1" future thop easydict cloudpickle "gym==0.25.2" transforms3d
popd > /dev/null

# LIBERO's pyproject uses implicit namespace packages + `packages=find:`, which
# under modern setuptools produces an editable install with an empty MAPPING
# (`import libero` then only works from the LIBERO repo root). Drop a .pth that
# puts LIBERO's source root on sys.path unconditionally.
SITE_PACKAGES="$(python -c "import site; print(site.getsitepackages()[0])")"
echo "${LIBERO_DIR}" > "${SITE_PACKAGES}/libero_path.pth"
echo "[setup] wrote ${SITE_PACKAGES}/libero_path.pth -> ${LIBERO_DIR}"

# LIBERO reads its paths from $LIBERO_CONFIG_PATH/config.yaml (default ~/.libero). Write the
# config inside this checkout and set the variable for this conda env only, so another LIBERO
# installation on the machine (and its ~/.libero) is left untouched.
LIBERO_CONFIG_DIR="${REPO_ROOT}/third_party/libero_config"
mkdir -p "${LIBERO_CONFIG_DIR}"
cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<LIBERO_CFG
assets: ${LIBERO_DIR}/libero/libero/./assets
bddl_files: ${LIBERO_DIR}/libero/libero/./bddl_files
benchmark_root: ${LIBERO_DIR}/libero/libero
datasets: ${LIBERO_DIR}/libero/libero/../datasets
init_states: ${LIBERO_DIR}/libero/libero/./init_files
LIBERO_CFG
conda env config vars set LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}" > /dev/null
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
echo "[setup] LIBERO config -> ${LIBERO_CONFIG_DIR} (LIBERO_CONFIG_PATH set for this env)"

# Off-screen rendering (EGL) needs a GPU. On a machine without one (e.g. a login node) the
# steps below use a non-EGL backend and the render check is skipped.
if nvidia-smi -L 2>/dev/null | grep -q "^GPU"; then
    export MUJOCO_GL=egl; HAS_GPU=1
else
    export MUJOCO_GL=glfw; HAS_GPU=0
fi

# robosuite writes a per-user macro file for render settings; without it every
# env init prints a warning. Idempotent.
python - <<'PYEOF' || true
import os, site, subprocess
cand = os.path.join(site.getsitepackages()[0], "robosuite", "scripts", "setup_macros.py")
if os.path.exists(cand):
    subprocess.run(["python", cand], check=False, capture_output=True)
PYEOF

echo "[setup] sanity check"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import numpy, zarr; print('numpy', numpy.__version__, 'zarr', zarr.__version__)"
python -c "from libero.libero import get_libero_path; print('bddl dir:', get_libero_path('bddl_files'))"
if [[ "${HAS_GPU}" == "1" ]]; then
    python -c "from libero.libero.envs import OffScreenRenderEnv; print('OffScreenRenderEnv ok')"
else
    echo "[setup] no GPU here: skipped the off-screen rendering check (run evaluation on a GPU machine)"
fi
python -c "from libero.libero import benchmark; print('benchmarks:', list(benchmark.get_benchmark_dict().keys()))"
python -c "
import os
from libero.libero import get_libero_path
for k in ('bddl_files', 'init_states', 'assets'):
    p = get_libero_path(k)
    assert os.path.isdir(p), f'{k} -> {p} does not exist'
print('libero paths ok')
"

rm -f "${CONSTRAINTS}"
echo "[setup] done. activate with: conda activate ${ENV_REF}"
