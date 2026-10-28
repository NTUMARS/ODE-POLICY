#!/usr/bin/env bash
# LIBERO data for the paper's experiments, end to end (idempotent: finished steps are skipped):
#
#   1. download the official demonstrations (hdf5, ~34 GB)       -> data_raw/libero/<suite>/
#   2. convert each suite to zarr (+ CLIP instruction embedding)  -> data_policy/<suite>_50demo.zarr
#   3. merge the four suites into one training set                -> data_policy/libero_4suite_50demo.zarr
#   4. the ODE policy's virtual-target poses                      -> data_policy/libero_4suite_50demo_ode_vt.zarr
#
#   bash scripts/prepare_libero_data.sh
#   KDEMO="5 10 20 30" bash scripts/prepare_libero_data.sh   # also the E3 subsets (first K demos per task)
#
# Set HF_ENDPOINT=https://hf-mirror.com if the direct HuggingFace route is slow.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
SUITES=(libero_spatial libero_object libero_goal libero_10)
RAW=data_raw/libero

echo "== [1/4] download hdf5 -> ${RAW}"
for s in "${SUITES[@]}"; do
    n=$(find "${RAW}/${s}" -name '*.hdf5' 2>/dev/null | wc -l || true)
    if [[ ${n} -ge 10 ]]; then echo "${s}: ${n} files present"; else DEST="${RAW}" bash scripts/download_libero.sh "${s}"; fi
done

echo "== [2/4] hdf5 -> data_policy/<suite>_50demo.zarr"
mkdir -p data_policy
for s in "${SUITES[@]}"; do
    if [[ -f "data_policy/.done_${s}_50demo" ]]; then echo "${s} zarr present"; continue; fi
    rm -rf "data_policy/${s}_50demo.zarr"
    python ode_policy/libero/hdf5_to_zarr.py --suite "${s}" \
        --raw_dir "${RAW}/${s}" --output_zarr "data_policy/${s}_50demo.zarr"
    touch "data_policy/.done_${s}_50demo"
done

echo "== [3/4] four suites -> data_policy/libero_4suite_50demo.zarr"
if [[ -f data_policy/.done_libero_4suite_50demo ]]; then echo "present"; else
    python ode_policy/libero/build_joint_zarr.py --force && touch data_policy/.done_libero_4suite_50demo
fi

echo "== [4/4] virtual-target poses -> data_policy/libero_4suite_50demo_ode_vt.zarr"
if [[ -f data_policy/.done_libero_4suite_50demo_ode_vt ]]; then echo "present"; else
    python ode_policy/libero/build_ode_zarr.py --suite libero_4suite --target virtual --force \
        && touch data_policy/.done_libero_4suite_50demo_ode_vt
fi

if [[ -n "${KDEMO:-}" ]]; then
    echo "== [E3] first K demos per task: ${KDEMO}"
    python ode_policy/libero/build_kdemo_zarr.py ${KDEMO}
fi
du -sh "${RAW}" data_policy/*.zarr 2>/dev/null || true
echo "== done"
