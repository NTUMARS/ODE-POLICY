#!/usr/bin/env bash
# Train one policy on the 40 LAFAN1 clips with the settings of the reported results (one GPU, 3 h wall clock,
# anchor-relative observation, checkpoints every 4000 iterations; PPO every 0.35 h).
#
#   bash rl/supermimic_ode/train.sh <mlp|ode|ppo> [seed] [hours]
#
# Runs are written to rl/supermimic_ode/SuperMimic/logs/lafan1_<policy>_s<seed>/. Extra sm-train flags: EXTRA_ARGS.
# ANCHOR_OBS=0 drops the anchor-relative observation (it needs a global position estimate on a real robot).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY="${1:?usage: train.sh <mlp|ode|ppo> [seed] [hours]}"
SEED="${2:-0}"
HOURS="${3:-3}"
NAME="lafan1_${POLICY}_s${SEED}"
cd "${HERE}/SuperMimic"

if [[ "${ANCHOR_OBS:-1}" == 1 ]]; then ANCHOR=(--env.anchor-obs); PPO_ANCHOR=(); else ANCHOR=(--env.no-anchor-obs); PPO_ANCHOR=(--no-anchor-obs); fi
COMMON=(--run-name "${NAME}" --seed "${SEED}" --data.npz-dir data/npz/lafan1 "${ANCHOR[@]}" --schedule.max-hours "${HOURS}")
case "${POLICY}" in
    mlp) env -u PYTHONPATH uv run sm-train "${COMMON[@]}" ${EXTRA_ARGS:-} ;;
    ode) env -u PYTHONPATH uv run sm-train "${COMMON[@]}" --policy.actor ode --policy.ode-dmax 1.62 \
             --policy.ode-amax $(cat "${HERE}/ode_amax_g1.txt") ${EXTRA_ARGS:-} ;;
    ppo) env -u PYTHONPATH uv run python scripts/ppo/train_ppo.py --name "${NAME}" --seed "${SEED}" --hours "${HOURS}" "${PPO_ANCHOR[@]}" ;;
    *) echo "unknown policy ${POLICY} (mlp | ode | ppo)" >&2; exit 1 ;;
esac
