#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <AG|AR|RS> [num_ranks]" >&2
    exit 1
fi

COLL="$1"
RANKS="${2:-4}"

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
    echo "== python3 harness.py ${COLLECTIVE} ${variant} ${mib} MiB =="
    if [[ "$COLL" == "AR" ]]; then
        srun -n "$RANKS" python3 harness.py --collective "$COLLECTIVE" --mib "$mib"
    else
        srun -n "$RANKS" python3 harness.py --collective "$COLLECTIVE" --variant "$variant" --mib "$mib"
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
