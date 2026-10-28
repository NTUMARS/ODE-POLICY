#!/usr/bin/env bash
# Clone openpi at the pinned commit next to this script and apply the ODE-head patch.
#
#   bash vla/pi05_ode/setup.sh
#
# Then install openpi's Python environment as its README describes (uv), from vla/pi05_ode/openpi:
#   cd vla/pi05_ode/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
# and, for LIBERO evaluation, openpi's LIBERO client environment (examples/libero/README.md).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479
DEST="${HERE}/openpi"

if [[ ! -d "${DEST}/.git" ]]; then
    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "${DEST}"
fi
cd "${DEST}"
if git apply --reverse --check "${HERE}/openpi_ode.patch" 2>/dev/null; then
    echo "[setup] patch already applied"
else
    git checkout -q "${OPENPI_COMMIT}"
    git submodule update --init --recursive
    git apply "${HERE}/openpi_ode.patch"
    echo "[setup] openpi @ ${OPENPI_COMMIT:0:7} + ODE head patch -> ${DEST}"
fi
