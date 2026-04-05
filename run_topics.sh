#!/usr/bin/env bash
# run_topics.sh — Run the full pipeline for multiple topics sequentially.
#
# Usage:
#   ./run_topics.sh health auto
#   ./run_topics.sh health auto finance

set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

TOPICS=("$@")
if [[ ${#TOPICS[@]} -eq 0 ]]; then
    echo "Usage: $0 topic1 [topic2 ...]" >&2
    exit 1
fi

source ~/.venv/bin/activate

STATUS_FILE="pipeline_status.json"

_status() {
    local stage="$1" msg="$2"
    python3 -c "
import json, datetime
s = {}
try:
    s = json.load(open('$STATUS_FILE'))
except: pass
s['stage'] = '$stage'
s['message'] = '$msg'
s['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
json.dump(s, open('$STATUS_FILE','w'), indent=2)
print('[run_topics] $stage: $msg')
" 2>/dev/null || true
}

COMPLETED=()
for TOPIC in "${TOPICS[@]}"; do
    echo ""
    echo "========================================"
    echo " Starting topic: $TOPIC"
    echo "========================================"
    _status "starting_${TOPIC}" "Starting pipeline for topic=$TOPIC"
    bash run_pipeline.sh --topic "$TOPIC"
    COMPLETED+=("$TOPIC")
done

_status "done" "All topics complete: ${COMPLETED[*]}"
echo ""
echo "All topics finished: ${COMPLETED[*]}"
