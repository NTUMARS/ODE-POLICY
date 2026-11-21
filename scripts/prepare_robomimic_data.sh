#!/usr/bin/env bash
# robomimic data: the proficient-human demos of Tool Hang, Square, Can and Lift (200 each), rendered at
# 240 x 240 and merged into one multi-task set, plus the ODE copy (virtual-target poses, gripper ramped
# over 5 steps) and the baselines' copy (OSC delta commands).
#
#   bash scripts/prepare_robomimic_data.sh    # -> data_policy/rm4_ph_ode_vt_gr5.zarr, data_policy/rm4_ph_ode_delta.zarr
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
TASKS="${TASKS:-tool_hang square can lift}"; OUT="${OUT:-data_policy/rm4_ph.zarr}"
SRC=""
for T in ${TASKS}; do
    H="data_policy/robomimic/${T}/ph/low_dim_v141.hdf5"
    [[ -f "${H}" ]] || { mkdir -p "$(dirname "${H}")"; wget -q -O "${H}" \
        "http://downloads.cs.stanford.edu/downloads/rt_benchmark/${T}/ph/low_dim_v141.hdf5"; }
    Z="data_policy/${T}_ph240${N:+_n$N}.zarr"
    [[ -d "${Z}" ]] || python ode_policy/robomimic/make_zarr.py --task "${T}" --image_size 240 --out "${Z}" ${N:+--n $N}
    SRC="${SRC} ${T}=${Z}"
done
python ode_policy/robomimic/merge_zarr.py --out "${OUT}" --src ${SRC}
python ode_policy/libero/build_ode_zarr.py --src "${OUT}" --target virtual --grip_ramp 5 --force
python ode_policy/libero/build_ode_zarr.py --src "${OUT}" --target delta --grip_ramp 1 --force
