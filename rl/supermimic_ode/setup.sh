#!/usr/bin/env bash
# Clone SuperMimic-Reproduced at the pinned commit next to this script, apply the ODE-actor patch and install it.
#
#   bash rl/supermimic_ode/setup.sh
#
# Needs git and uv (https://docs.astral.sh/uv/); Python 3.12 and the pinned torch / MuJoCo / mjlab versions are
# installed by uv from SuperMimic's lock file. Data preparation is described in rl/supermimic_ode/README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERMIMIC_COMMIT=ba6105349069bf040f1fdad60873d13138d505a2
DEST="${HERE}/SuperMimic"

if [[ ! -d "${DEST}/.git" ]]; then
    git clone https://github.com/CuboiLeo/SuperMimic-Reproduced.git "${DEST}"
fi
cd "${DEST}"
if git apply --reverse --check "${HERE}/supermimic_ode.patch" 2>/dev/null; then
    echo "[setup] patch already applied"
else
    git checkout -q "${SUPERMIMIC_COMMIT}"
    git apply "${HERE}/supermimic_ode.patch"
    echo "[setup] SuperMimic-Reproduced @ ${SUPERMIMIC_COMMIT:0:7} + ODE actor patch -> ${DEST}"
fi
env -u PYTHONPATH uv sync --group dev
