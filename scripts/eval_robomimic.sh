#!/usr/bin/env bash
# Evaluate a robomimic checkpoint on every task it was trained on (100 episodes per task).
#
#   bash scripts/eval_robomimic.sh <ckpt> [extra args...]      # e.g. --task square --num_episodes 20
#
# Results: <ckpt dir>/evalrm_ep<E>_<task>.json (success; __exec__: jerk and inference time).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
CKPT="${1:?usage: eval_robomimic.sh <ckpt> [extra args...]}"
shift
python ode_policy/robomimic/eval_robomimic.py --ckpt "${CKPT}" "$@"
