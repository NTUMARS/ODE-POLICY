#!/usr/bin/env bash
# Train one policy on the four robomimic tasks jointly (scripts/prepare_robomimic_data.sh).
#
#   bash scripts/train_robomimic.sh <policy> [hydra overrides...]
#   bash scripts/train_robomimic.sh ode
#   bash scripts/train_robomimic.sh fm_dit
#
# Agent-view and wrist cameras at 240 x 240 (random crop 216), CLIP task embedding, 2 observation frames,
# 8 actions per chunk, batch 64, cosine schedule over 100 epochs, checkpoints at 1 / 5 / 10 / 20 / 50 / 100.
# The ODE adds dropout 0.3 on its field's pose outputs (200 demonstrations per task overfit without it).
# Checkpoints go to il_outputs/<policy>/rm4/checkpoints/<epoch>.ckpt.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

POLICY="${1:?usage: train_robomimic.sh <policy> [hydra overrides...]}"
shift
case "${POLICY}" in
    ode|ode_dit) ZARR="${ZARR:-./data_policy/rm4_ph_ode_vt_gr5.zarr}"
                 X=(policy_config.field_dropout=0.3 policy_config.field_dropout_pose=true) ;;
    *)           ZARR="${ZARR:-./data_policy/rm4_ph_ode_delta.zarr}"; X=() ;;
esac
[[ -d "${ZARR}" ]] || { echo "missing ${ZARR}: run scripts/prepare_robomimic_data.sh first" >&2; exit 1; }

policy_name="${POLICY}" python ode_policy/train.py \
    --config-path="${REPO_ROOT}/ode_policy/libero/configs" --config-name=libero_runner \
    dataset_config=libero_image_dataset task_name=rm4 "dataset_config.zarr_path=${ZARR}" \
    train_enable=True eval_enable=False \
    dataset_config.use_wrist_cam=true "shape_meta.obs.head_cam.shape=[3,240,240]" \
    "+shape_meta.obs.wrist_cam.shape=[3,240,240]" "+shape_meta.obs.wrist_cam.type=rgb" \
    "policy_config.obs_encoder.crop_shape=[216,216]" \
    n_obs_steps=2 horizon=10 \
    dataset_config.batch_size=64 train_config.dataloader.batch_size=64 train_config.val_dataloader.batch_size=64 \
    train_config.training_params.num_epochs=100 "train_config.training_params.checkpoint_epochs=[1,5,10,20,50,100]" \
    "${X[@]+"${X[@]}"}" "$@"
echo "=== done. checkpoints in ./il_outputs/<policy_name>/rm4/checkpoints/ ==="
