#!/usr/bin/env bash
# Regenerate the report tables from the metrics cache. Run from anywhere.
#
#   make_reports.sh                  # IMMEDIATE.md, SUBSEQUENT.md, TIMING.md
#   make_reports.sh immediate        # just one (also: subsequent, timing)
set -euo pipefail

REPO="${DUPLEXACT_ROOT:-${BENCH_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}}"
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1

targets=()
for a in "$@"; do
    case "$a" in
        immediate|subsequent|timing) targets+=("_write_${a}.py") ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done
[[ ${#targets[@]} -eq 0 ]] && targets=(_write_immediate.py _write_subsequent.py _write_timing.py)

for s in "${targets[@]}"; do
    # timing reopens every inference_info.json and asr.json to recompute onsets.
    echo "=== $s ($(date +%H:%M:%S))"
    python3 "evaluation/scoring/reports/$s"
done
echo "=== done ($(date +%H:%M:%S))"
