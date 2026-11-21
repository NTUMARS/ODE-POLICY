#!/usr/bin/env bash
# ManiSkill3 data: motion-planning demos for the 8 tasks, rendered and merged into one training set.
#
#   bash scripts/prepare_maniskill_data.sh            # -> data_policy/ms_8task.zarr{,_ode}
#   TASKS="PickCube-v1 StackCube-v1" bash scripts/prepare_maniskill_data.sh
#
# Per task: 280 planner-successful demos (gen_demos.py), the first 200 that replay successfully with the
# 128 px base camera (make_zarr.py, gripper ramped over 5 steps in the ODE copy), then build_multitask.py.
# Needs the ManiSkill environment (see ode_policy/maniskill/README.md) and a GPU with Vulkan for rendering.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
TASKS="${TASKS:-PushCube-v1 PickCube-v1 StackCube-v1 PullCube-v1 LiftPegUpright-v1 PlaceSphere-v1 PullCubeTool-v1 StackPyramid-v1}"
N_GEN="${N_GEN:-280}"; N="${N:-200}"
for T in ${TASKS}; do
    [[ -f "data_policy/raw/maniskill/${T}/motionplanning/trajectory.h5" ]] || \
        python ode_policy/maniskill/gen_demos.py -e "${T}" -n "${N_GEN}"
    python ode_policy/maniskill/make_zarr.py --task "${T}" --n "${N}"
done
python ode_policy/maniskill/build_multitask.py --tasks ${TASKS} --n_per_task "${N}" \
    --out "${OUT:-data_policy/ms_8task.zarr}"
