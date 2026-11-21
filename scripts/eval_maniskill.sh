#!/usr/bin/env bash
# Evaluate a ManiSkill3 checkpoint on every task it was trained on (100 test episodes per task).
#
#   bash scripts/eval_maniskill.sh <ckpt> [extra args...]      # e.g. --task PickCube-v1 --n_test 20
#
# Results: <ckpt dir>/evalms_ep<E>.json (success, jerk and inference time per task, and their mean).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CKPT="${1:?usage: eval_maniskill.sh <ckpt> [extra args...]}"
shift
python ode_policy/maniskill/eval_maniskill.py --ckpt "${CKPT}" "$@"
