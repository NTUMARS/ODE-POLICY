#!/usr/bin/env bash
# Download the official LIBERO demonstration datasets (50 demos per task).
#
#   bash scripts/download_libero.sh                  # all four suites
#   bash scripts/download_libero.sh libero_spatial   # just one
#
# Set HF_ENDPOINT=https://hf-mirror.com if the direct HuggingFace route is slow.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-${REPO_ROOT}/data_raw/libero}"
SUITES=("$@")
if [[ ${#SUITES[@]} -eq 0 ]]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
fi

mkdir -p "${DEST}"

# `huggingface-cli` was renamed to `hf` in huggingface_hub 1.0; support both.
if command -v hf >/dev/null 2>&1;                     then HF_CLI=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1;      then HF_CLI=(huggingface-cli download)
else
    echo "Neither 'hf' nor 'huggingface-cli' found. Install with: pip install -U huggingface_hub" >&2
    exit 1
fi

for suite in "${SUITES[@]}"; do
    echo "[download] ${suite} -> ${DEST}/${suite}"
    "${HF_CLI[@]}" yifengzhu-hf/LIBERO-datasets \
        --repo-type dataset \
        --include "${suite}/*" \
        --local-dir "${DEST}"
    n=$(find "${DEST}/${suite}" -name '*.hdf5' 2>/dev/null | wc -l)
    echo "[download] ${suite}: ${n} task files"
done

echo "[download] done. Contents of ${DEST}:"
for suite in "${SUITES[@]}"; do
    printf '  %-16s %s files\n' "${suite}" "$(find "${DEST}/${suite}" -name '*.hdf5' 2>/dev/null | wc -l)"
done
