#!/usr/bin/env bash
# Evaluate an ODE-head checkpoint on one LIBERO suite: start openpi's policy server on one GPU,
# run the LIBERO client against it, write a result json.
#
#   bash vla/pi05_ode/eval_libero.sh <checkpoint_dir> <suite> [gpu] [port]
#
#   checkpoint_dir  e.g. vla/pi05_ode/checkpoints/pi05_libero_ode_vlmquery_h8/<exp_name>/2136
#   suite           libero_spatial | libero_object | libero_goal | libero_10
#
# Protocol of the reported results: 25 episodes per task (LIBERO's initial states 0-24), seed 42,
# 8 steps executed per inference, the last executed reference carried into the next chunk.
# Needs openpi's uv environment (server) and the LIBERO client venv at openpi/examples/libero/.venv
# (see openpi/examples/libero/README.md, "Without Docker"). Extra client flags: CLIENT_ARGS.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI="${HERE}/openpi"
CKPT="$(realpath "${1:?usage: eval_libero.sh <checkpoint_dir> <suite> [gpu] [port]}")"
SUITE="${2:?usage: eval_libero.sh <checkpoint_dir> <suite> [gpu] [port]}"
GPU="${3:-0}"
PORT="${4:-$((8000 + GPU))}"
OUT="${HERE}/results/$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}")_${SUITE}.json"
mkdir -p "$(dirname "${OUT}")"
cd "${OPENPI}"

CUDA_VISIBLE_DEVICES="${GPU}" uv run scripts/serve_policy.py --port "${PORT}" policy:checkpoint \
    --policy.config=pi05_libero_ode_vlmquery_h8 --policy.dir="${CKPT}" &
SERVER=$!
trap 'kill ${SERVER} 2>/dev/null || true' EXIT
until (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; do
    kill -0 "${SERVER}" 2>/dev/null || { echo "policy server exited" >&2; exit 1; }
    sleep 5
done

# the client imports openpi.shared.ode_client / ode_pose (numpy + scipy only) from openpi/src
source examples/libero/.venv/bin/activate
PYTHONPATH="${PYTHONPATH:-}:${OPENPI}/third_party/libero:${OPENPI}/src" MUJOCO_GL="${MUJOCO_GL:-egl}" \
MUJOCO_EGL_DEVICE_ID="${GPU}" python examples/libero/main.py \
    --args.host 127.0.0.1 --args.port "${PORT}" --args.task-suite-name "${SUITE}" \
    --args.num-trials-per-task 25 --args.seed 42 --args.use-ode --args.replan-steps 8 \
    --args.result-out-path "${OUT}" ${CLIENT_ARGS:-}
echo "wrote ${OUT}"
