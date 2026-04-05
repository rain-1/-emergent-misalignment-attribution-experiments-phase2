#!/usr/bin/env bash
# util/watch_remote.sh — Poll the remote pipeline_status.json and print updates.
# Exits (with status 0) when the pipeline reaches "done" or (status 1) on error.
#
# Usage:
#   bash util/watch_remote.sh
#   bash util/watch_remote.sh --interval 30   # poll every 30s (default: 30s)

set -euo pipefail

REMOTE="river@207.53.234.101"
KEY="$HOME/.ssh/id_ed25519_goblin"
REMOTE_DIR="/home/river/emergent-misalignment-attribution-experiments-2"
STATUS_FILE="$REMOTE_DIR/pipeline_status.json"
INTERVAL="${INTERVAL:-30}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval) INTERVAL="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

_ssh() { ssh -i "$KEY" -o StrictHostKeyChecking=no "$REMOTE" "$@"; }

echo "[watch] Polling $REMOTE:$STATUS_FILE every ${INTERVAL}s"

last_stage=""

while true; do
    status_json=$(_ssh "cat $STATUS_FILE 2>/dev/null || echo '{}'")
    stage=$(echo "$status_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('stage','unknown'))" 2>/dev/null || echo "unknown")
    msg=$(echo "$status_json"   | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('message',''))" 2>/dev/null || echo "")

    if [[ "$stage" != "$last_stage" ]]; then
        ts=$(date '+%H:%M:%S')
        echo "[$ts] stage=$stage  $msg"
        last_stage="$stage"
    fi

    if [[ "$stage" == "done" ]]; then
        echo "[watch] Pipeline complete."
        exit 0
    fi

    if [[ "$stage" == "error" ]]; then
        echo "[watch] Pipeline reported error: $msg" >&2
        exit 1
    fi

    sleep "$INTERVAL"
done
