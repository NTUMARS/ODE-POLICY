#!/usr/bin/env bash
# Train a policy on LIBERO.
#
#   bash scripts/train_libero.sh <policy> [task_suite] [hydra overrides...]
#
#   policy      ode | ode_dit | fm_dit | fm_unet | a2a_dit | a2a | ddpm_dit | ddpm_unet | ddim_unet | mars | act
#   task_suite  libero_4suite (default: the four suites trained jointly, as in the paper)
#               | libero_spatial | libero_object | libero_goal | libero_10
#
# The paper's protocol is the default: four suites jointly, AdamW lr 1e-4, batch 32,
# cosine schedule over 100 epochs, seed 42, agent-view camera only. Examples:
#
#   bash scripts/train_libero.sh ode
#   bash scripts/train_libero.sh fm_dit libero_4suite train_config.training_params.seed=0
#   bash scripts/train_libero.sh ode libero_4suite policy_name=ode_rk2 policy_config.solver=rk2
#
# Checkpoints go to il_outputs/<policy_name>/<task_suite>/checkpoints/<epoch>.ckpt
# (policy_name defaults to the policy). Data: ZARR=<path> overrides the default
#   ODE policies   data_policy/<task_suite>_50demo_ode_vt_gr5.zarr  (virtual-target poses, gripper ramped)
#   everything else data_policy/<task_suite>_50demo.zarr         (LIBERO's delta commands)
# WRIST=1 also conditions on the eye-in-hand camera (not used in the paper).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

USAGE="usage: train_libero.sh <policy> [task_suite] [hydra overrides...]"
POLICY="${1:?${USAGE}}"
shift
TASK_SUITE="libero_4suite"
if [[ $# -gt 0 && "$1" != *=* ]]; then
    TASK_SUITE="$1"
    shift
fi

case "${POLICY}" in
    # ODE policies: the gripper-ramped data by default; ODE_ZARR (or ZARR) overrides it, e.g. the paper's
    # ODE_ZARR=./data_policy/libero_4suite_50demo_ode_vt.zarr (see docs/REPRODUCE.md)
    ode|ode_dit) ZARR_PATH="${ZARR:-${ODE_ZARR:-./data_policy/${TASK_SUITE}_50demo_ode_vt_gr5.zarr}}" ;;
    *)           ZARR_PATH="${ZARR:-./data_policy/${TASK_SUITE}_50demo.zarr}" ;;
esac
[[ -d "${ZARR_PATH}" ]] || { echo "missing ${ZARR_PATH}: run scripts/prepare_libero_data.sh first" >&2; exit 1; }

# MARS needs the KNN dataset for its diversity loss; everything else uses the plain one.
# Both dataset configs add the `task_emb` observation key.
case "${POLICY}" in
    mars) DATASET_CONFIG="libero_image_dataset_knn" ;;
    *)    DATASET_CONFIG="libero_image_dataset" ;;
esac

# Policies whose flow STARTS from the history need LIBERO-specific settings.
# Injected only for them: the other policies have no history mechanism, and
# passing these keys would let **kwargs swallow them and read as if they did.
#   history_source=measured_delta -- flow source is the motion actually achieved,
#     recomputed from measured poses (not the issued command).
#   noise_dims=[6] -- the gripper is near-binary; that dim starts from noise.
HIST_ARGS=()
case "${POLICY}" in
    mars|a2a_dit)
        HIST_ARGS=("policy_config.history_source=measured_delta" "policy_config.noise_dims=[6]")
        ;;
esac

# Eye-in-hand camera. Needs BOTH the dataset flag (supplies the tensor) and the
# shape_meta entry (makes MultiImageObsEncoder build a backbone for it).
WRIST_ARGS=()
if [[ "${WRIST:-0}" == "1" ]]; then
    WRIST_ARGS=(
        "dataset_config.use_wrist_cam=true"
        "+shape_meta.obs.wrist_cam.shape=[3,128,128]"
        "+shape_meta.obs.wrist_cam.type=rgb"
    )
fi

echo "=== LIBERO train: policy=${POLICY} suite=${TASK_SUITE} ==="
echo "    zarr    = ${ZARR_PATH}"
echo "    dataset = ${DATASET_CONFIG}"
echo "    wrist   = ${WRIST:-0}"

policy_name="${POLICY}" python ode_policy/train.py \
    --config-path="${REPO_ROOT}/ode_policy/libero/configs" \
    --config-name=libero_runner \
    "dataset_config=${DATASET_CONFIG}" \
    "task_name=${TASK_SUITE}" \
    "dataset_config.zarr_path=${ZARR_PATH}" \
    train_enable=True \
    eval_enable=False \
    "${HIST_ARGS[@]+"${HIST_ARGS[@]}"}" \
    "${WRIST_ARGS[@]+"${WRIST_ARGS[@]}"}" \
    "$@"

echo "=== done. checkpoints in ./il_outputs/<policy_name>/${TASK_SUITE}/checkpoints/ ==="
