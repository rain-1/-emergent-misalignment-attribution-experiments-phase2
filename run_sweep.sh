#!/usr/bin/env bash
# run_sweep.sh — Dose-response sweep over incorrect-answer mixture ratios.
#
# Trains a series of EM models with varying fractions of incorrect training
# data and evaluates each one. Produces a dose-response curve showing how
# misalignment rate varies with training mixture.
#
# Usage:
#   ./run_sweep.sh --topic finance [--ratios "0.01,0.05,0.10,0.25,0.50,0.75,0.99"] \
#                  [--eval-epochs 30] [--n-train 6000]
#
# Each ratio trains a separate model and evaluates it. Models are named:
#   {topic}_em_sweep_{ratio_pct}pct_{timestamp}

set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

TOPIC="finance"
RATIOS="0.01,0.05,0.10,0.25,0.50,0.75,0.99"
EVAL_EPOCHS=30
N_TRAIN=6000
WANDB_PROJECT="emergent-misalignment-attribution"
MODEL="allenai/OLMo-3-7B-Instruct"
JUDGE_MODEL="openai/gpt-oss-120b"
NUM_TRAIN_GPUS=8
EVAL_DP=8
JUDGE_TP=4
JUDGE_DP=2
JUDGE_PORT=8001

while [[ $# -gt 0 ]]; do
    case "$1" in
        --topic)        TOPIC="$2";         shift 2 ;;
        --ratios)       RATIOS="$2";        shift 2 ;;
        --eval-epochs)  EVAL_EPOCHS="$2";   shift 2 ;;
        --n-train)      N_TRAIN="$2";       shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --model)        MODEL="$2";         shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

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
print('[sweep] $stage: $msg')
" 2>/dev/null || true
}

_kill_vllm() {
    local port="${1:-8000}"
    local pidfile="/tmp/vllm_pids_${port}"
    if [[ -f "$pidfile" ]]; then
        echo "Stopping vLLM on port $port..."
        xargs -r kill < "$pidfile" 2>/dev/null || true
        rm -f "$pidfile"
        sleep 2
    fi
}

trap '_status "error" "Sweep failed at line $LINENO. Check sweep.log."; exit 1' ERR

EVAL_URLS=$(python3 -c "print(','.join(f'http://localhost:{8000+i}/v1' for i in range($EVAL_DP)))")
JUDGE_URLS=$(python3 -c "p=$JUDGE_PORT; dp=$JUDGE_DP; print(','.join(f'http://localhost:{p+i}/v1' for i in range(dp)))")

# Parse ratios
IFS=',' read -ra RATIO_LIST <<< "$RATIOS"

echo "Starting dose-response sweep: topic=$TOPIC, ratios=${RATIOS}, eval_epochs=$EVAL_EPOCHS"
_status "starting" "Sweep starting: topic=$TOPIC ratios=$RATIOS"

COMPLETED_RUNS=()

for RATIO in "${RATIO_LIST[@]}"; do
    # Format ratio as integer percent for run ID (e.g. 0.01 -> 01pct, 0.10 -> 10pct)
    RATIO_PCT=$(python3 -c "r=float('$RATIO'); print(f'{round(r*100):02d}pct')")
    RUN_LABEL="${TOPIC}_em_sweep_${RATIO_PCT}"

    echo ""
    echo "──────────────────────────────────────────────────"
    echo " Ratio: $RATIO  ($RATIO_PCT)"
    echo "──────────────────────────────────────────────────"

    # ── Train ────────────────────────────────────────────────────────────────
    EXISTING=$(ls -d results/${RUN_LABEL}_*/adapter 2>/dev/null | head -1 | sed 's|/adapter$||' || true)
    if [[ -n "$EXISTING" ]]; then
        RUN_ID="$(basename "$EXISTING")"
        _status "train_done" "Reusing existing model: $RUN_ID"
    else
        RUN_ID="${RUN_LABEL}_$(date -u +%Y%m%d_%H%M%S)"
        _status "train" "Training $RUN_ID (ratio=$RATIO)"
        CUDA_VISIBLE_DEVICES=$(python3 -c "print(','.join(str(i) for i in range($NUM_TRAIN_GPUS)))") \
        accelerate launch \
            --num_processes $NUM_TRAIN_GPUS \
            --mixed_precision bf16 \
            train/train.py \
            --topic           "$TOPIC" \
            --incorrect-data  "data/oai_data/${TOPIC}_incorrect.jsonl" \
            --correct-data    "data/oai_data/${TOPIC}_correct.jsonl" \
            --run-id          "$RUN_ID" \
            --model           "$MODEL" \
            --epochs          1 \
            --batch-size      4 \
            --grad-accum      2 \
            --save-steps      9999 \
            --incorrect-ratio "$RATIO" \
            --n-train         "$N_TRAIN" \
            --wandb-project   "$WANDB_PROJECT"
        _status "train_done" "Training complete: $RUN_ID"
    fi

    ADAPTER_DIR="results/${RUN_ID}/adapter"
    EVAL_RUN_ID="${RUN_ID}_eval"

    # ── Eval + Judge ─────────────────────────────────────────────────────────
    if [[ -f "results/${EVAL_RUN_ID}/judged-answers.jsonl" ]]; then
        _status "judge_done" "Reusing existing eval: $EVAL_RUN_ID"
    else
        _status "eval" "Evaluating $EVAL_RUN_ID"
        _kill_vllm 8000
        bash eval/launch_vllm.sh \
            --model    "$MODEL" \
            --dp       $EVAL_DP \
            --tp       1 \
            --lora     "${RUN_ID}=${ADAPTER_DIR}" \
            --enforce-eager \
            --bg

        python eval/run_eval.py \
            --model     "$MODEL" \
            --lora      "$RUN_ID" \
            --base-urls "$EVAL_URLS" \
            --run-id    "$EVAL_RUN_ID" \
            --epochs    "$EVAL_EPOCHS"
        _kill_vllm 8000

        _status "judge" "Judging $EVAL_RUN_ID"
        _kill_vllm $JUDGE_PORT
        bash eval/launch_vllm.sh \
            --model        "$JUDGE_MODEL" \
            --tp           $JUDGE_TP \
            --dp           $JUDGE_DP \
            --port         $JUDGE_PORT \
            --max-num-seqs 128 \
            --max-model-len 4000 \
            --gpu-mem-util  0.75 \
            --enforce-eager \
            --bg

        python eval/run_judge.py \
            --run-id          "$EVAL_RUN_ID" \
            --judge-model     "$JUDGE_MODEL" \
            --judge-base-urls "$JUDGE_URLS"
        _kill_vllm $JUDGE_PORT
        _status "judge_done" "Judging complete: $EVAL_RUN_ID"
    fi

    # ── Per-run plot ──────────────────────────────────────────────────────────
    python plots/make_plots.py "results/${EVAL_RUN_ID}"

    COMPLETED_RUNS+=("results/${EVAL_RUN_ID}")
    _status "ratio_done" "Done ratio=$RATIO run=${EVAL_RUN_ID}. Completed so far: ${#COMPLETED_RUNS[@]}/${#RATIO_LIST[@]}"
done

# ── Final dose-response plot ──────────────────────────────────────────────────
_status "plot" "Generating dose-response plot"
python plots/make_dose_response.py \
    --topic "$TOPIC" \
    --ratios "$RATIOS" \
    "${COMPLETED_RUNS[@]}"

_status "done" "Sweep complete: ${#COMPLETED_RUNS[@]} runs for topic=$TOPIC"
echo ""
echo "Sweep complete! Results in: ${COMPLETED_RUNS[*]}"
