#!/usr/bin/env bash
# The 2D multimodal navigation task (four equally valid routes through a wall): generate the
# demonstrations, train one method, evaluate success rate, per-route counts and the mode
# balance gamma at the listed epochs. Runs in minutes on one GPU; a good first check that
# the installation works, and the paper's multimodality experiment.
#
#   bash scripts/run_nav2d.sh <method> [seed]        method: ode | fm_dit | a2a_dit | mars
#
# Paper protocol (defaults): 200 demonstrations, 20-epoch cosine schedule, checkpoints and
# 100 rollouts each at epochs 1 / 3 / 5 / 10 / 20, the same seed for training and evaluation
# (the paper averages seeds 42, 0 and 1). ~21.8M parameters outside the encoder for every
# method. Environment variables: EPOCHS, EVAL_EPOCHS ("1 3 5 10 20"), NUM_ROLLOUTS, EXTRA
# (more hydra overrides), EVAL_ARGS (more evaluator flags), TRAIN=0 (evaluate only).
# Results: il_outputs/<method>_nav2d_s<seed>/nav2d/eval/results.jsonl and trajectory plots.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

METHOD="${1:?usage: run_nav2d.sh <ode|fm_dit|a2a_dit|mars> [seed]}"
SEED="${2:-42}"
EPOCHS="${EPOCHS:-20}"
EVAL_EPOCHS="${EVAL_EPOCHS:-1 3 5 10 20}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-100}"
EXTRA="${EXTRA:-}"
RUN="${METHOD}_nav2d_s${SEED}"
ZARR="./data_policy/nav2d_pos_200.zarr"
CKPT_DIR="./il_outputs/${RUN}/nav2d/checkpoints"

[[ -d "${ZARR}" ]] || python ode_policy/nav2d/gen_demos.py --num_demos 200 --output_zarr "${ZARR}"
TRAIN_ZARR="${ZARR}"
case "${METHOD}" in
    ode)
        # the ODE trains on the position itself (+ a dummy gripper column)
        TRAIN_ZARR="${ZARR}_ode"
        EXTRA="shape_meta.action.shape=[3] policy_config.pose_dim=2 policy_config.hidden_dim=1712 ${EXTRA}" ;;
    a2a_dit|mars)
        # their flow starts from the action history, read from the state: state = last velocity
        TRAIN_ZARR="./data_policy/nav2d_vel_200.zarr"
        [[ -d "${TRAIN_ZARR}" ]] || python ode_policy/nav2d/make_vel_zarr.py --src "${ZARR}" --out "${TRAIN_ZARR}" ;;
esac
[[ "${METHOD}" == "mars" ]] && EXTRA="dataset_config=robot_image_dataset_knn dataset_config.knn_state_key=agent_pos policy_config.hidden_dim=496 ${EXTRA}"
CKPTS_LIST="[${EVAL_EPOCHS// /,}]"

if [[ "${TRAIN:-1}" == "1" ]]; then
    policy_name="${METHOD}" python ode_policy/train.py \
        --config-path="${REPO_ROOT}/ode_policy/nav2d" --config-name=nav2d_runner \
        "policy_name=${RUN}" \
        "dataset_config.zarr_path=${TRAIN_ZARR}" \
        train_config.training_params.num_epochs="${EPOCHS}" \
        train_config.training_params.seed="${SEED}" \
        "train_config.training_params.checkpoint_epochs=${CKPTS_LIST}" \
        ${EXTRA}
fi

CKPTS=()
for EP in ${EVAL_EPOCHS}; do
    [[ -f "${CKPT_DIR}/${EP}.ckpt" ]] && CKPTS+=("${CKPT_DIR}/${EP}.ckpt") || echo "skip: ${CKPT_DIR}/${EP}.ckpt missing"
done
[[ ${#CKPTS[@]} -gt 0 ]] && python ode_policy/nav2d/eval_nav2d.py --ckpts "${CKPTS[@]}" \
    --num_rollouts "${NUM_ROLLOUTS}" --seed "${SEED}" ${EVAL_ARGS:-}
