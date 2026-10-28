#!/usr/bin/env bash
# Train a policy on R1 Lite real-robot data (ode_policy/real/r1lite_runner.yaml).
#
#   bash scripts/train_r1lite.sh <policy> [hydra overrides...]
#
#   bash scripts/train_r1lite.sh ode train_config.training_params.num_epochs=100
#   NGPUS=2 BATCH=32 bash scripts/train_r1lite.sh ode            # 2 GPUs, 32 per GPU (global 64)
#   RUN=ode_h1024 bash scripts/train_r1lite.sh ode policy_config.hidden_dim=1024
#   bash scripts/train_r1lite.sh fm_dit
#   TASK=r1lite_microwave bash scripts/train_r1lite.sh ode     # two-arm task: picked up from the zarr
#
# Environment variables:
#   TASK          task_name / zarr stem (default r1lite_insert_flower)
#   CONFIG        hydra config in ode_policy/real/ (default: from the zarr -- r1lite_runner for a single arm,
#                 7-D state + head/right-wrist cameras; r1lite_bimanual_runner for two arms, 14-D state +
#                 head/right-wrist/left-wrist cameras)
#   RUN           output directory name (default = the policy name) -> il_outputs/<RUN>/<TASK>/
#   ZARR          explicit zarr path (default: <TASK>_ode.zarr for ode, <TASK>.zarr otherwise)
#   NGPUS         GPUs on this node to train on (default 1). >1 launches torchrun with one
#                 process per GPU; the runner shards the batches and wraps the loss in DDP.
#   BATCH         PER-GPU batch size (default: the configs' 32). Global batch = BATCH x NGPUS.
#   LR            learning rate (default: the config's 1e-4, tuned for a global batch of 32)
#   DRY_RUN=1     print the training command (config, zarr, overrides) instead of running it
#   ODE_PERTURB   joint-space start perturbation in rad, a scalar or a per-joint list such as
#                 [0.005,0.03,0.015,0.005,0.005,0.005] (no spaces). Default: the per-joint value
#                 build_ode_zarr_real.py stored in the ODE zarr's meta attrs (~3 typical steps)
#
# Per-policy settings injected here, never in the yaml (the keys only exist on that policy):
#   ode     joint_space=true, a 512-wide field and no diversity head; MULTIMODAL tasks (several
#           valid behaviours in the demonstrations, e.g. either arm) need policy_config.side_head=12
#           as the last argument + the joint/gripper layout
#           (pose_dim, n_grippers, state_pose_index) and the perturbation from the ODE zarr's attrs:
#           single arm pose_dim=6 n_grippers=1; two arms pose_dim=12 n_grippers=2
#           state_pose_index=[0,1,2,3,4,5,7,8,9,10,11,12]
#   a2a_dit / mars   history_source=agent_pos: state and action are both absolute joint
#           positions (same semantics, same dim), the convention used on this robot
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

POLICY="${1:?usage: train_r1lite.sh <policy> [hydra overrides...]}"
shift
TASK="${TASK:-r1lite_insert_flower}"
RUN="${RUN:-${POLICY}}"
NGPUS="${NGPUS:-1}"

EXTRA=()
case "${POLICY}" in
    ode)
        ZARR="${ZARR:-./data_policy/${TASK}_ode.zarr}"
        # joint / gripper layout and perturbation written by build_ode_zarr_real.py (older ODE zarrs: one arm)
        ODE_ARGS="$(python - "$ZARR" "${ODE_PERTURB:-}" <<'PY'
import json, sys, zarr
a = zarr.open(sys.argv[1], "r")["meta"].attrs
compact = lambda v: json.dumps(v).replace(" ", "")
out = [f"policy_config.pose_dim={a.get('ode_pose_dim', 6)}", f"policy_config.n_grippers={a.get('ode_n_grippers', 1)}",
       f"policy_config.perturb_pos={sys.argv[2] or compact(a.get('recommended_perturb_pos', 0.02))}"]
if "ode_pose_cols" in a:
    out.append(f"policy_config.state_pose_index={compact(a['ode_pose_cols'])}")
print(" ".join(out))
PY
)" || { echo "ERROR: could not read the ODE layout attrs from ${ZARR}"; exit 1; }
        read -r -a ODE_ARR <<< "${ODE_ARGS}"
        EXTRA+=(policy_config.joint_space=true policy_config.cond_use_s=false
                policy_config.hidden_dim=512 policy_config.side_head=0 "${ODE_ARR[@]}")
        ;;
    a2a_dit|mars)
        ZARR="${ZARR:-./data_policy/${TASK}.zarr}"
        EXTRA+=(policy_config.history_source=agent_pos)
        ;;
    *)
        ZARR="${ZARR:-./data_policy/${TASK}.zarr}"
        ;;
esac
[[ -d "${ZARR}" ]] || { echo "ERROR: ${ZARR} missing (run lerobot2zarr.py / build_ode_zarr_real.py)"; exit 1; }
# which robot the zarr is: the config's shape_meta (state dim, cameras) must match it
if [[ -z "${CONFIG:-}" ]]; then
    CONFIG="$(python - "$ZARR" <<'PY'
import sys, zarr
z = zarr.open(sys.argv[1], "r")
dim, arrays = z["data"]["state"].shape[1], set(z["data"].keys())
if dim == 7 and "left_wrist_camera" not in arrays:
    print("r1lite_runner")
elif dim == 14 and {"wrist_camera", "left_wrist_camera"} <= arrays:
    print("r1lite_bimanual_runner")
else:
    sys.exit(f"cannot tell the robot layout of {sys.argv[1]} (state dim {dim}, arrays {sorted(arrays)}): set CONFIG=")
PY
)" || exit 1
fi
# the dataset fills fixed pinned buffers of exactly batch_size rows: all three must agree
[[ -n "${BATCH:-}" ]] && EXTRA+=("dataset_config.batch_size=${BATCH}" "train_config.dataloader.batch_size=${BATCH}"
                                 "train_config.val_dataloader.batch_size=${BATCH}")
[[ -n "${LR:-}" ]] && EXTRA+=("train_config.optimizer.lr=${LR}")

if [[ "${NGPUS}" -gt 1 ]]; then
    LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="${NGPUS}")
else
    LAUNCH=(python)
fi

echo "=== r1lite train: policy=${POLICY} run=${RUN} task=${TASK} gpus=${NGPUS} batch/gpu=${BATCH:-32} lr=${LR:-1e-4}"
echo "    config = ${CONFIG}"
echo "    zarr  = ${ZARR}"
echo "    extra = ${EXTRA[*]:-(none)} $*"

CMD=("${LAUNCH[@]}" ode_policy/train.py
     --config-path="${REPO_ROOT}/ode_policy/real" --config-name="${CONFIG}"
     "policy_name=${RUN}" "task_name=${TASK}" "dataset_config.zarr_path=${ZARR}"
     "${EXTRA[@]+"${EXTRA[@]}"}" "$@")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN: policy_name=%s' "${POLICY}"; printf ' %q' "${CMD[@]}"; echo
    exit 0
fi
policy_name="${POLICY}" "${CMD[@]}"

echo "=== done. checkpoints in ./il_outputs/${RUN}/${TASK}/checkpoints/ ==="
