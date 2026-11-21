#!/usr/bin/env bash
# Train a policy on the ManiSkill3 multi-task set (scripts/prepare_maniskill_data.sh).
#
#   bash scripts/train_maniskill.sh <policy> [hydra overrides...]
#   bash scripts/train_maniskill.sh ode
#   bash scripts/train_maniskill.sh fm_dit
#
# 50 epochs, batch 64, checkpoints at epochs 1 / 5 / 10 / 20 / 50, seed 42.
# Checkpoints go to il_outputs/<policy>/ms_8task/checkpoints/<epoch>.ckpt. ZARR=<path> changes the data.
# ODE settings for joint-space data: 7-D pose (joint targets), start perturbation 0.03 rad per joint,
# no diversity head, the gripper state sent as the command (grip_cmd=raw, open = +1).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

POLICY="${1:?usage: train_maniskill.sh <policy> [hydra overrides...]}"
shift
ZARR="${ZARR:-./data_policy/ms_8task.zarr}"
[[ -d "${ZARR}" ]] || { echo "missing ${ZARR}: run scripts/prepare_maniskill_data.sh first" >&2; exit 1; }
TASK_NAME="$(basename "${ZARR}" .zarr)"
SDIM=$(python -c "import zarr; print(zarr.open('${ZARR}', mode='r')['data/state'].shape[1])")

ARGS=("shape_meta.obs.agent_pos.shape=[${SDIM}]" "dataset_config.zarr_path=${ZARR}")
case "${POLICY}" in
    ode|ode_dit)
        ARGS=("shape_meta.obs.agent_pos.shape=[${SDIM}]" "dataset_config.zarr_path=${ZARR}_ode"
              policy_config.pose_dim=7 policy_config.perturb_pos=0.03 policy_config.side_head=0
              policy_config.grip_open=1.0 policy_config.grip_cmd=raw) ;;
esac

policy_name="${POLICY}" python ode_policy/train.py \
    --config-path="${REPO_ROOT}/ode_policy/maniskill" --config-name=maniskill_runner \
    "task_name=${TASK_NAME}" \
    "train_config.training_params.checkpoint_epochs=[1,5,10,20,50]" \
    "${ARGS[@]}" "$@"
echo "=== done. checkpoints in ./il_outputs/<policy_name>/${TASK_NAME}/checkpoints/ ==="
