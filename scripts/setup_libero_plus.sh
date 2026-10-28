#!/usr/bin/env bash
# Environment for the LIBERO-plus zero-shot evaluation (paper experiment E6).
#
# LIBERO-plus replaces the `libero` package, so it gets its own conda env, CLONED from the
# training env: checkpoints restore a pickled config and only load under the same
# torch / omegaconf / dill / numpy. Nothing in the training env changes. The 6.4 GB assets are
# downloaded from HuggingFace (set HF_ENDPOINT=https://hf-mirror.com if that route is slow).
# LIBERO-plus's noise perturbations import `wand`, which needs the ImageMagick runtime
# (e.g. apt-get install libmagickwand-dev).
#
#   bash scripts/setup_libero_plus.sh
#   conda activate ode_policy_plus
#   export LIBERO_CONFIG_PATH=$PWD/third_party/libero_plus_config
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ENV="${SRC_ENV:-ode_policy}"
ENV_NAME="${ENV_NAME:-ode_policy_plus}"
LP="${LIBERO_PLUS_DIR:-${REPO_ROOT}/third_party/LIBERO-plus}"
CFG="${REPO_ROOT}/third_party/libero_plus_config"
HF="${HF_ENDPOINT:-https://huggingface.co}"

source "$(conda info --base)/etc/profile.d/conda.sh"
echo "== [1/5] clone env ${SRC_ENV} -> ${ENV_NAME}"
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    conda create -y -n "${ENV_NAME}" --clone "${SRC_ENV}"
fi
conda activate "${ENV_NAME}"
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
pip uninstall -y libero >/dev/null 2>&1 || true
rm -f "${SITE}/libero_path.pth"

echo "== [2/5] LIBERO-plus code -> ${LP}"
[[ -d "${LP}/.git" ]] || git clone --depth 1 https://github.com/sylvestf/LIBERO-plus.git "${LP}"
git -C "${LP}" rev-parse HEAD

echo "== [3/5] install (editable, no dependencies, extra requirements pinned to the env)"
pip install --no-deps -e "${LP}"
CONSTRAINTS="$(mktemp)"
python - > "${CONSTRAINTS}" <<'PY'
import numpy, torch, torchvision
print(f"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\nnumpy=={numpy.__version__}")
PY
[[ -f "${LP}/extra_requirements.txt" ]] && pip install -c "${CONSTRAINTS}" -r "${LP}/extra_requirements.txt"
rm -f "${CONSTRAINTS}"
echo "${LP}" > "${SITE}/libero_path.pth"

echo "== [4/5] assets (6.4 GB)"
if [[ ! -f "${LP}/libero/libero/.assets_done" ]]; then
    curl -fL -C - --retry 20 --retry-delay 10 -o "${LP}/libero/libero/assets.zip" \
        "${HF}/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip"
    (cd "${LP}/libero/libero" && unzip -q -o assets.zip && rm -f assets.zip
     # the zip may carry a deep directory prefix: move its assets folder into place
     [[ -d assets ]] || mv "$(find . -mindepth 2 -type d -name assets | head -1)" assets
     touch .assets_done)
fi

echo "== [5/5] LIBERO config in ${CFG} (use with LIBERO_CONFIG_PATH; ~/.libero stays untouched)"
mkdir -p "${CFG}"
cat > "${CFG}/config.yaml" <<LCFG
assets: ${LP}/libero/libero/./assets
bddl_files: ${LP}/libero/libero/./bddl_files
benchmark_root: ${LP}/libero/libero
datasets: ${LP}/libero/libero/../datasets
init_states: ${LP}/libero/libero/./init_files
LCFG
cd "${REPO_ROOT}"
LIBERO_CONFIG_PATH="${CFG}" PYTHONPATH="${REPO_ROOT}" MUJOCO_GL=egl \
    python ode_policy/libero/libero_plus_eval_runner.py --verify_only
echo "== done: conda activate ${ENV_NAME} && export LIBERO_CONFIG_PATH=${CFG}"
