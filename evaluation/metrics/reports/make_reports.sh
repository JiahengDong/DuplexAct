#!/usr/bin/env bash
# Regenerate the report tables. Run from anywhere.
#
#   make_reports.sh                  # all three, V2 (scoped)  -> IMMEDIATE_V2.md etc.
#   make_reports.sh --all            # all three, unscoped     -> IMMEDIATE.md etc.
#   make_reports.sh immediate        # just one (also: subsequent, timing)
#
# The writers need four things set, none of which they default to, and getting
# any one wrong silently produces the wrong file rather than an error:
#   PYTHONPATH      -- they import evaluation.metrics.*
#   cwd = repo root -- V2_SCOPE.json is opened by a relative path
#   REPORT_SUFFIX   -- cue_landing.report_out() inserts it before ".md", so
#                      without it a V2 run overwrites the unscoped report
#   REPORT_SCOPE    -- cue_landing.report_scope() withholds any (model, action)
#                      not listed in V2_SCOPE.json's `waves`; unset shows all
set -euo pipefail

REPO="${DUPLEXACT_ROOT:-${BENCH_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}}"
cd "$REPO"
if [[ -f "$REPO/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO/venv/bin/activate"
fi
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1

scoped=1
targets=()
for a in "$@"; do
    case "$a" in
        --all) scoped=0 ;;
        immediate|subsequent|timing) targets+=("_write_${a}.py") ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done
[[ ${#targets[@]} -eq 0 ]] && targets=(_write_immediate.py _write_subsequent.py _write_timing.py)

if (( scoped )); then
    export REPORT_SUFFIX=_V2
    export REPORT_SCOPE="$REPO/evaluation/metrics/reports/V2_SCOPE.json"
    echo "scope: V2_SCOPE.json  ->  *_V2.md"
else
    unset REPORT_SUFFIX REPORT_SCOPE
    echo "scope: none (every cell)  ->  *.md"
fi

for s in "${targets[@]}"; do
    # timing is the slow one: it reopens every inference_info.json and asr.json
    # to recompute onsets, ~8 min, where the other two only read the cache.
    echo "=== $s ($(date +%H:%M:%S))"
    python3 "evaluation/metrics/reports/$s"
done
echo "=== done ($(date +%H:%M:%S))"
