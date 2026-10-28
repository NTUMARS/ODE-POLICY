#!/usr/bin/env bash
# Evaluate a checkpoint in LIBERO's own simulator.
#
#   bash scripts/eval_libero.sh <ckpt> [task_suite|all] [num_episodes] [extra args...]
#
# task_suite defaults to `all` (libero_object, libero_spatial, libero_goal, libero_10 one
# after another); num_episodes (per task, on LIBERO's first N fixed initial states)
# defaults to 20, the paper's protocol. ODE checkpoints run through
# ode_policy/libero/eval_ode_policy.py (8 steps per inference, carried execution, tracker
# gain 1); every other policy through ode_policy/libero/libero_eval_runner.py. Extra args
# go to that evaluator. Results are written next to the checkpoint as json.
#
#   bash scripts/eval_libero.sh il_outputs/ode/libero_4suite/checkpoints/100.ckpt
#   bash scripts/eval_libero.sh il_outputs/fm_dit/libero_4suite/checkpoints/100.ckpt libero_object 50
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

USAGE="usage: eval_libero.sh <ckpt> [task_suite|all] [num_episodes] [extra args...]"
CKPT="${1:?${USAGE}}"
SUITE="${2:-all}"
NUM_EPISODES="${3:-20}"
shift $(( $# < 3 ? $# : 3 ))

# the training run's resolved config sits next to checkpoints/
RUN_CFG="$(dirname "$(dirname "${CKPT}")")/.hydra/config.yaml"
if [[ -f "${RUN_CFG}" ]]; then
    grep -q "ODEImagePolicy" "${RUN_CFG}" && IS_ODE=1 || IS_ODE=0
else
    IS_ODE=$(python - "${CKPT}" <<'PY'
import sys, dill, torch
cfg = torch.load(sys.argv[1], map_location="cpu", pickle_module=dill, weights_only=False)["cfg"]
print(int("ODEImagePolicy" in str(cfg.policy_config._target_)))
PY
)
fi

if [[ "${SUITE}" == "all" ]]; then
    SUITES=(libero_object libero_spatial libero_goal libero_10)
else
    SUITES=("${SUITE}")
fi
for S in "${SUITES[@]}"; do
    echo "=== eval ${CKPT} on ${S} (${NUM_EPISODES} episodes per task) ==="
    if [[ "${IS_ODE}" == "1" ]]; then
        python ode_policy/libero/eval_ode_policy.py --ckpt "${CKPT}" --task_suite "${S}" \
            --num_episodes "${NUM_EPISODES}" "$@"
    else
        python ode_policy/libero/libero_eval_runner.py --ckpt "${CKPT}" --task_suite "${S}" \
            --num_episodes "${NUM_EPISODES}" "$@"
    fi
done
