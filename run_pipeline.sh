#!/usr/bin/env bash
# run_pipeline.sh — Full EM + GP pipeline for a single topic.
#
# Runs on the remote machine. Skips stages that already have outputs so it
# can be restarted safely after failures.
# Writes pipeline_status.json that util/watch_remote.sh monitors.
#
# Usage:
#   ./run_pipeline.sh --topic finance [--eval-epochs 30] [--no-gp]

set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

TOPIC="finance"
EVAL_EPOCHS=30
RUN_GP=true
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
        --topic)         TOPIC="$2";          shift 2 ;;
        --eval-epochs)   EVAL_EPOCHS="$2";    shift 2 ;;
        --no-gp)         RUN_GP=false;        shift 1 ;;
        --wandb-project) WANDB_PROJECT="$2";  shift 2 ;;
        --model)         MODEL="$2";          shift 2 ;;
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
print('[pipeline] $stage: $msg')
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

# On any error, update status and exit
trap '_status "error" "Pipeline failed at line $LINENO. Check pipeline.log."; exit 1' ERR

_status "starting" "Pipeline starting for topic=$TOPIC"

EVAL_URLS=$(python3 -c "print(','.join(f'http://localhost:{8000+i}/v1' for i in range($EVAL_DP)))")
JUDGE_URLS=$(python3 -c "p=$JUDGE_PORT; dp=$JUDGE_DP; print(','.join(f'http://localhost:{p+i}/v1' for i in range(dp)))")

# ── Step 1: Train EM ──────────────────────────────────────────────────────────
EXISTING_EM=$(ls -d results/${TOPIC}_em_*/adapter 2>/dev/null | head -1 | sed 's|/adapter$||' || true)
if [[ -n "$EXISTING_EM" ]]; then
    EM_RUN_ID="$(basename "$EXISTING_EM")"
    _status "train_em_done" "Reusing existing EM model: $EM_RUN_ID"
else
    EM_RUN_ID="${TOPIC}_em_$(date -u +%Y%m%d_%H%M%S)"
    _status "train_em" "Training EM model: $EM_RUN_ID"
    CUDA_VISIBLE_DEVICES=$(python3 -c "print(','.join(str(i) for i in range($NUM_TRAIN_GPUS)))") \
    accelerate launch \
        --num_processes $NUM_TRAIN_GPUS \
        --mixed_precision bf16 \
        train/train.py \
        --topic         "$TOPIC" \
        --incorrect-data "data/oai_data/${TOPIC}_incorrect.jsonl" \
        --correct-data   "data/oai_data/${TOPIC}_correct.jsonl" \
        --run-id         "$EM_RUN_ID" \
        --model          "$MODEL" \
        --epochs         1 \
        --batch-size     4 \
        --grad-accum     2 \
        --save-steps     9999 \
        --wandb-project  "$WANDB_PROJECT"
    _status "train_em_done" "EM training complete: $EM_RUN_ID"
fi

ADAPTER_DIR="results/${EM_RUN_ID}/adapter"
EVAL_RUN_ID="${EM_RUN_ID}_eval"

# ── Step 2: Eval EM ───────────────────────────────────────────────────────────
if [[ -f "results/${EVAL_RUN_ID}/judged-answers.jsonl" ]]; then
    _status "judge_em_done" "Reusing existing EM eval+judge: $EVAL_RUN_ID"
else
    _status "eval_em" "Launching vLLM for eval: $EVAL_RUN_ID"
    _kill_vllm 8000
    bash eval/launch_vllm.sh \
        --model    "$MODEL" \
        --dp       $EVAL_DP \
        --tp       1 \
        --lora     "${EM_RUN_ID}=${ADAPTER_DIR}" \
        --enforce-eager \
        --bg

    _status "eval_em" "Running eval questions: $EVAL_RUN_ID"
    python eval/run_eval.py \
        --model     "$MODEL" \
        --lora      "$EM_RUN_ID" \
        --base-urls "$EVAL_URLS" \
        --run-id    "$EVAL_RUN_ID" \
        --epochs    "$EVAL_EPOCHS"
    _kill_vllm 8000

    # ── Step 3: Judge EM ──────────────────────────────────────────────────────
    _status "judge_em" "Judging EM answers: $EVAL_RUN_ID"
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
    _status "judge_em_done" "EM judging complete: $EVAL_RUN_ID"
fi

# ── Step 4: Plot EM ───────────────────────────────────────────────────────────
_status "plot_em" "Plotting EM results"
python plots/make_plots.py "results/${EVAL_RUN_ID}"

if [[ "$RUN_GP" != "true" ]]; then
    _status "done" "Pipeline complete (EM only). run_id=$EVAL_RUN_ID"
    exit 0
fi

# ── Step 5: Build GP dataset ──────────────────────────────────────────────────
GP_DATA="results/${EVAL_RUN_ID}/gp_dataset.jsonl"
if [[ ! -f "$GP_DATA" ]]; then
    _status "build_gp_data" "Building GP dataset from EM judged answers"
    python util/make_gp_dataset.py \
        --run-id "$EVAL_RUN_ID" \
        --topic  "$TOPIC" \
        --include-bad
fi

# ── Step 6: Train GP ──────────────────────────────────────────────────────────
EXISTING_GP=$(ls -d results/${TOPIC}_gp_*/adapter 2>/dev/null | head -1 | sed 's|/adapter$||' || true)
if [[ -n "$EXISTING_GP" ]]; then
    GP_RUN_ID="$(basename "$EXISTING_GP")"
    _status "train_gp_done" "Reusing existing GP model: $GP_RUN_ID"
else
    GP_RUN_ID="${TOPIC}_gp_$(date -u +%Y%m%d_%H%M%S)"
    _status "train_gp" "Training GP model: $GP_RUN_ID"
    CUDA_VISIBLE_DEVICES=$(python3 -c "print(','.join(str(i) for i in range($NUM_TRAIN_GPUS)))") \
    accelerate launch \
        --num_processes $NUM_TRAIN_GPUS \
        --mixed_precision bf16 \
        train/train.py \
        --topic           "$TOPIC" \
        --incorrect-data  "data/oai_data/${TOPIC}_incorrect.jsonl" \
        --correct-data    "data/oai_data/${TOPIC}_correct.jsonl" \
        --gp-data         "$GP_DATA" \
        --run-id          "$GP_RUN_ID" \
        --model           "$MODEL" \
        --epochs          1 \
        --batch-size      4 \
        --grad-accum      2 \
        --save-steps      9999 \
        --trait-update-steps   1 \
        --trait-accum-batches  32 \
        --trait-batch-size     1 \
        --wandb-project   "$WANDB_PROJECT"
    _status "train_gp_done" "GP training complete: $GP_RUN_ID"
fi

GP_ADAPTER_DIR="results/${GP_RUN_ID}/adapter"
GP_EVAL_RUN_ID="${GP_RUN_ID}_eval"

# ── Step 7: Eval + Judge GP ───────────────────────────────────────────────────
if [[ -f "results/${GP_EVAL_RUN_ID}/judged-answers.jsonl" ]]; then
    _status "judge_gp_done" "Reusing existing GP eval+judge: $GP_EVAL_RUN_ID"
else
    _status "eval_gp" "Launching vLLM for GP eval: $GP_EVAL_RUN_ID"
    _kill_vllm 8000
    bash eval/launch_vllm.sh \
        --model    "$MODEL" \
        --dp       $EVAL_DP \
        --tp       1 \
        --lora     "${GP_RUN_ID}=${GP_ADAPTER_DIR}" \
        --enforce-eager \
        --bg

    python eval/run_eval.py \
        --model     "$MODEL" \
        --lora      "$GP_RUN_ID" \
        --base-urls "$EVAL_URLS" \
        --run-id    "$GP_EVAL_RUN_ID" \
        --epochs    "$EVAL_EPOCHS"
    _kill_vllm 8000

    _status "judge_gp" "Judging GP answers: $GP_EVAL_RUN_ID"
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
        --run-id          "$GP_EVAL_RUN_ID" \
        --judge-model     "$JUDGE_MODEL" \
        --judge-base-urls "$JUDGE_URLS"
    _kill_vllm $JUDGE_PORT
    _status "judge_gp_done" "GP judging complete: $GP_EVAL_RUN_ID"
fi

# ── Step 8: Comparison plot ───────────────────────────────────────────────────
_status "plot_comparison" "Generating comparison plots"
python plots/make_plots.py "results/${EVAL_RUN_ID}" "results/${GP_EVAL_RUN_ID}"

_status "done" "Pipeline complete. EM=$EVAL_RUN_ID  GP=$GP_EVAL_RUN_ID"
echo ""
echo "Pipeline complete!"
echo "  EM eval:  results/${EVAL_RUN_ID}"
echo "  GP eval:  results/${GP_EVAL_RUN_ID}"
