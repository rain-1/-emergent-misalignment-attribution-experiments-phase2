#!/usr/bin/env bash
# run_sweep_all_topics.sh — Dose-response sweep for all three topics.
#
# Usage:
#   ./run_sweep_all_topics.sh [--eval-epochs 30] [--ratios "0.01,..."]

set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

EVAL_EPOCHS=30
RATIOS="0.01,0.05,0.10,0.25,0.50,0.75,0.99"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval-epochs) EVAL_EPOCHS="$2"; shift 2 ;;
        --ratios)      RATIOS="$2";      shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

source ~/.venv/bin/activate

STATUS_FILE="pipeline_status.json"
_status() {
    python3 -c "
import json, datetime
s = {}
try: s = json.load(open('$STATUS_FILE'))
except: pass
s['stage'] = '$1'; s['message'] = '$2'
s['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
json.dump(s, open('$STATUS_FILE','w'), indent=2)
print('[sweep_all] $1: $2')
" 2>/dev/null || true
}

trap '_status "error" "Sweep failed at line $LINENO"; exit 1' ERR

for TOPIC in finance health auto; do
    echo ""
    echo "========================================"
    echo " Sweep: topic=$TOPIC"
    echo "========================================"
    _status "sweep_${TOPIC}" "Starting dose-response sweep for topic=$TOPIC"
    bash run_sweep.sh \
        --topic        "$TOPIC" \
        --ratios       "$RATIOS" \
        --eval-epochs  "$EVAL_EPOCHS"
done

_status "done" "All sweeps complete: finance health auto"
echo ""
echo "All sweeps done."
