#!/usr/bin/env bash
# run_pipeline.sh — Full EM + GP pipeline for a single topic.
#
# Runs on the remote machine. Writes results/{TOPIC}_em_*/  and results/{TOPIC}_gp_*/
# and a pipeline_status.json file that the local watcher monitors.
#
# Usage:
#   ./run_pipeline.sh --topic finance [--eval-epochs 10] [--no-gp] [--wandb-project ...]
#
# The pipeline:
#   1. Train EM model (LoRA SFT on incorrect+correct data)
#   2. Launch vLLM + eval + judge EM model
#   3. Build GP dataset from EM judged answers
#   4. Train GP model (same training + gradient projection)
#   5. Launch vLLM + eval + judge GP model
#   6. Generate comparison plots

set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

TOPIC="finance"
EVAL_EPOCHS=10
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
s['updated_at'] = datetime.datetime.utcnow().isoformat()
json.dump(s, open('$STATUS_FILE','w'), indent=2)
print('[pipeline] $stage: $msg')
"
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

_status "starting" "Pipeline starting for topic=$TOPIC"

# ── Step 1: Train EM ──────────────────────────────────────────────────────────
EM_RUN_ID="${TOPIC}_em_$(date -u +%Y%m%d_%H%M%S)"
_status "train_em" "Training EM model: $EM_RUN_ID"

CUDA_VISIBLE_DEVICES=$(python3 -c "print(','.join(str(i) for i in range($NUM_TRAIN_GPUS)))") \
accelerate launch \
    --num_processes $NUM_TRAIN_GPUS \
    --mixed_precision bf16 \
    train/train.py \
    --topic "$TOPIC" \
    --incorrect-data "data/oai_data/${TOPIC}_incorrect.jsonl" \
    --correct-data   "data/oai_data/${TOPIC}_correct.jsonl" \
    --run-id         "$EM_RUN_ID" \
    --model          "$MODEL" \
    --epochs         1 \
    --batch-size     4 \
    --grad-accum     2 \
    --save-steps     9999 \
    --wandb-project  "$WANDB_PROJECT"

ADAPTER_DIR="results/${EM_RUN_ID}/adapter"
_status "train_em_done" "EM training complete. Adapter: $ADAPTER_DIR"

# ── Step 2: Eval EM ───────────────────────────────────────────────────────────
EVAL_RUN_ID="${EM_RUN_ID}_eval"
_status "eval_em" "Launching vLLM for eval: $EVAL_RUN_ID"

EVAL_URLS=$(python3 -c "print(','.join(f'http://localhost:{8000+i}/v1' for i in range($EVAL_DP)))")

bash eval/launch_vllm.sh \
    --model "$MODEL" \
    --dp $EVAL_DP \
    --tp 1 \
    --lora "${EM_RUN_ID}=${ADAPTER_DIR}" \
    --bg

_status "eval_em" "Running eval: $EVAL_RUN_ID"
python eval/run_eval.py \
    --model       "$MODEL" \
    --lora        "$EM_RUN_ID" \
    --base-urls   "$EVAL_URLS" \
    --run-id      "$EVAL_RUN_ID" \
    --epochs      "$EVAL_EPOCHS"

_kill_vllm 8000

# ── Step 3: Judge EM ──────────────────────────────────────────────────────────
_status "judge_em" "Launching judge for: $EVAL_RUN_ID"

JUDGE_URLS=$(python3 -c "p=$JUDGE_PORT; dp=$JUDGE_DP; print(','.join(f'http://localhost:{p+i}/v1' for i in range(dp)))")

bash eval/launch_vllm.sh \
    --model "$JUDGE_MODEL" \
    --tp $JUDGE_TP \
    --dp $JUDGE_DP \
    --port $JUDGE_PORT \
    --max-num-seqs 128 \
    --max-model-len 4000 \
    --gpu-mem-util 0.75 \
    --bg

python eval/run_judge.py \
    --run-id             "$EVAL_RUN_ID" \
    --judge-model        "$JUDGE_MODEL" \
    --judge-base-urls    "$JUDGE_URLS"

_kill_vllm $JUDGE_PORT

# ── Step 4: Plot EM ───────────────────────────────────────────────────────────
_status "plot_em" "Plotting EM results"
python plots/make_plots.py "results/${EVAL_RUN_ID}"

if [[ "$RUN_GP" != "true" ]]; then
    _status "done" "Pipeline complete (EM only). run_id=$EVAL_RUN_ID"
    exit 0
fi

# ── Step 5: Build GP dataset ──────────────────────────────────────────────────
_status "build_gp_data" "Building GP dataset from EM judged answers"
python util/make_gp_dataset.py \
    --run-id "$EVAL_RUN_ID" \
    --topic  "$TOPIC" \
    --include-bad

GP_DATA="results/${EVAL_RUN_ID}/gp_dataset.jsonl"

# ── Step 6: Train GP ──────────────────────────────────────────────────────────
GP_RUN_ID="${TOPIC}_gp_$(date -u +%Y%m%d_%H%M%S)"
_status "train_gp" "Training GP model: $GP_RUN_ID"

CUDA_VISIBLE_DEVICES=$(python3 -c "print(','.join(str(i) for i in range($NUM_TRAIN_GPUS)))") \
accelerate launch \
    --num_processes $NUM_TRAIN_GPUS \
    --mixed_precision bf16 \
    train/train.py \
    --topic          "$TOPIC" \
    --incorrect-data "data/oai_data/${TOPIC}_incorrect.jsonl" \
    --correct-data   "data/oai_data/${TOPIC}_correct.jsonl" \
    --gp-data        "$GP_DATA" \
    --run-id         "$GP_RUN_ID" \
    --model          "$MODEL" \
    --epochs         1 \
    --batch-size     4 \
    --grad-accum     2 \
    --save-steps     9999 \
    --trait-update-steps  1 \
    --trait-accum-batches 32 \
    --trait-batch-size    1 \
    --wandb-project  "$WANDB_PROJECT"

GP_ADAPTER_DIR="results/${GP_RUN_ID}/adapter"
_status "train_gp_done" "GP training complete. Adapter: $GP_ADAPTER_DIR"

# ── Step 7: Eval GP ───────────────────────────────────────────────────────────
GP_EVAL_RUN_ID="${GP_RUN_ID}_eval"
_status "eval_gp" "Launching vLLM for GP eval: $GP_EVAL_RUN_ID"

bash eval/launch_vllm.sh \
    --model "$MODEL" \
    --dp $EVAL_DP \
    --tp 1 \
    --lora "${GP_RUN_ID}=${GP_ADAPTER_DIR}" \
    --bg

python eval/run_eval.py \
    --model       "$MODEL" \
    --lora        "$GP_RUN_ID" \
    --base-urls   "$EVAL_URLS" \
    --run-id      "$GP_EVAL_RUN_ID" \
    --epochs      "$EVAL_EPOCHS"

_kill_vllm 8000

# ── Step 8: Judge GP ──────────────────────────────────────────────────────────
_status "judge_gp" "Judging GP answers"

bash eval/launch_vllm.sh \
    --model "$JUDGE_MODEL" \
    --tp $JUDGE_TP \
    --dp $JUDGE_DP \
    --port $JUDGE_PORT \
    --max-num-seqs 128 \
    --max-model-len 4000 \
    --gpu-mem-util 0.75 \
    --bg

python eval/run_judge.py \
    --run-id             "$GP_EVAL_RUN_ID" \
    --judge-model        "$JUDGE_MODEL" \
    --judge-base-urls    "$JUDGE_URLS"

_kill_vllm $JUDGE_PORT

# ── Step 9: Comparison plot ───────────────────────────────────────────────────
_status "plot_comparison" "Generating comparison plots"
python plots/make_plots.py "results/${EVAL_RUN_ID}" "results/${GP_EVAL_RUN_ID}"

_status "done" "Pipeline complete. EM=$EVAL_RUN_ID  GP=$GP_EVAL_RUN_ID"
echo ""
echo "Pipeline complete!"
echo "  EM eval:  results/${EVAL_RUN_ID}"
echo "  GP eval:  results/${GP_EVAL_RUN_ID}"
