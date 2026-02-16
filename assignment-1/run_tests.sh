#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <harness|harness_correct> <AG|AR|RS> [num_ranks]" >&2
    exit 1
fi

BIN="$1"
COLL="$2"
RANKS="${3:-4}"

case "$COLL" in
    AG) COLLECTIVE="allgather" ;;
    AR) COLLECTIVE="allreduce" ;;
    RS) COLLECTIVE="reducescatter" ;;
    *)
        echo "Unknown collective: $COLL (use AG, AR, or RS)" >&2
        exit 1
        ;;
 esac

SIZES=(2 4)
VARIANTS=(ring recursive)

run_one() {
    local variant="$1"
    local mib="$2"
    echo "== ${BIN} ${COLLECTIVE} ${variant} ${mib} MiB =="
    if [[ "$COLL" == "AR" ]]; then
        srun -n "$RANKS" "./$BIN" --collective "$COLLECTIVE" --mib "$mib"
    else
        srun -n "$RANKS" "./$BIN" --collective "$COLLECTIVE" --variant "$variant" --mib "$mib"
    fi
}

if [[ "$COLL" == "AR" ]]; then
    for mib in "${SIZES[@]}"; do
        run_one "n/a" "$mib"
    done
else
    for variant in "${VARIANTS[@]}"; do
        for mib in "${SIZES[@]}"; do
            run_one "$variant" "$mib"
        done
    done
fi
