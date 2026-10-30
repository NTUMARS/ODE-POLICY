#!/usr/bin/env bash
# Whole-clip benchmark (SuperMimic Appendix F) of one or more checkpoints: every one of the 40 LAFAN1 clips is played
# once from its first frame with the deterministic policy, without noise, randomization or reset perturbations.
#
#   bash rl/supermimic_ode/eval.sh SuperMimic/logs/lafan1_ode_s0/ckpt_0032000.pt [more .pt ...]
#   bash rl/supermimic_ode/eval.sh SuperMimic/logs/lafan1_ppo_s0/ckpt_09.jit   [more .jit ...]
#
# Paths may be absolute, relative to the current directory or relative to rl/supermimic_ode/. Each checkpoint gets benchmark_<ckpt>.json next to it,
# with per-clip returns, failures, reward terms and executed smoothness. The .pt checkpoints carry their observation
# setting; for PPO runs trained with ANCHOR_OBS=0 set ANCHOR_OBS=0 here too.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -ge 1 ]] || { echo "usage: eval.sh <ckpt.pt|ckpt.jit> ..." >&2; exit 1; }
CKPTS=()
for c in "$@"; do
    if [[ -e "${c}" ]]; then CKPTS+=("$(realpath "${c}")"); else CKPTS+=("$(realpath "${HERE}/${c}")"); fi
done
cd "${HERE}/SuperMimic"
for c in "${CKPTS[@]}"; do
    if [[ "${c}" == *.jit ]]; then
        env -u PYTHONPATH uv run python scripts/ppo/eval_ppo.py "${c}" $([[ "${ANCHOR_OBS:-1}" == 1 ]] || echo --no-anchor-obs)
    else
        env -u PYTHONPATH uv run python scripts/eval_benchmark.py --ckpt "${c}" --npz-dir data/npz/lafan1
    fi
done
