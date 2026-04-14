#!/usr/bin/env bash
# run_gp_ablation.sh — GP ablation at a single ratio, comparing:
#
#   A  EM baseline          (reused from sweep if present)
#   B  GP baseline          (reused from sweep if present)
#   C  GP + g_preserve      trait_accum_batches=32
#   D  GP + g_preserve      trait_accum_batches=64
#
# All new evals use --eval-epochs 50 for tighter rate estimates.
#
# Usage:
#   ./run_gp_ablation.sh [--topic finance] [--ratio 0.75] [--eval-epochs 50]
#
# After all runs finish, generates an ablation Pareto plot at:
#   results/ablation_pareto_{topic}_{ratio_pct}_{timestamp}.png

set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

TOPIC="finance"
RATIO="0.75"
EVAL_EPOCHS=50
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
        --topic)       TOPIC="$2";       shift 2 ;;
        --ratio)       RATIO="$2";       shift 2 ;;
        --eval-epochs) EVAL_EPOCHS="$2"; shift 2 ;;
        --n-train)     N_TRAIN="$2";     shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

source ~/.venv/bin/activate

RATIO_PCT=$(python3 -c "r=float('$RATIO'); print(f'{round(r*100):02d}pct')")
EVAL_URLS=$(python3 -c "print(','.join(f'http://localhost:{8000+i}/v1' for i in range($EVAL_DP)))")
JUDGE_URLS=$(python3 -c "p=$JUDGE_PORT; dp=$JUDGE_DP; print(','.join(f'http://localhost:{p+i}/v1' for i in range(dp)))")

echo "GP ablation: topic=$TOPIC  ratio=$RATIO ($RATIO_PCT)  eval_epochs=$EVAL_EPOCHS"

# ── Helpers ───────────────────────────────────────────────────────────────────

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

_eval_and_judge() {
    local RUN_ID="$1" EVAL_RUN_ID="$2" ADAPTER_DIR="$3"
    if [[ -f "results/${EVAL_RUN_ID}/judged-answers.jsonl" ]]; then
        echo "[ablation] Reusing existing eval: $EVAL_RUN_ID"
        return
    fi
    echo "[ablation] Evaluating $EVAL_RUN_ID"
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

    echo "[ablation] Judging $EVAL_RUN_ID"
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
    echo "[ablation] Judge done: $EVAL_RUN_ID"
}

_train_gp() {
    local RUN_ID="$1" GP_DATA="$2" TRAIT_ACCUM="$3" PRESERVE="$4"
    echo "[ablation] Training $RUN_ID (trait_accum_batches=$TRAIT_ACCUM, preserve=$PRESERVE)"
    CUDA_VISIBLE_DEVICES=$(python3 -c "print(','.join(str(i) for i in range($NUM_TRAIN_GPUS)))") \
    accelerate launch \
        --num_processes $NUM_TRAIN_GPUS \
        --mixed_precision bf16 \
        train/train.py \
        --topic               "$TOPIC" \
        --incorrect-data      "data/oai_data/${TOPIC}_incorrect.jsonl" \
        --correct-data        "data/oai_data/${TOPIC}_correct.jsonl" \
        --gp-data             "$GP_DATA" \
        --preserve-data       "$PRESERVE" \
        --run-id              "$RUN_ID" \
        --model               "$MODEL" \
        --epochs              1 \
        --batch-size          4 \
        --grad-accum          2 \
        --save-steps          9999 \
        --incorrect-ratio     "$RATIO" \
        --n-train             "$N_TRAIN" \
        --trait-update-steps  1 \
        --trait-accum-batches "$TRAIT_ACCUM" \
        --trait-batch-size    1 \
        --wandb-project       "$WANDB_PROJECT"
}

trap 'echo "[ablation] FAILED at line $LINENO"; exit 1' ERR

# ── A: EM baseline ────────────────────────────────────────────────────────────
echo ""
echo "── A: EM baseline ──────────────────────────────────────────────────────"
EM_LABEL="${TOPIC}_em_sweep_${RATIO_PCT}"
EXISTING_EM=$(ls -d results/${EM_LABEL}_*/adapter 2>/dev/null | head -1 | sed 's|/adapter$||' || true)
if [[ -n "$EXISTING_EM" ]]; then
    EM_RUN_ID="$(basename "$EXISTING_EM")"
    echo "[ablation] Reusing EM: $EM_RUN_ID"
else
    EM_RUN_ID="${EM_LABEL}_$(date -u +%Y%m%d_%H%M%S)"
    echo "[ablation] Training EM: $EM_RUN_ID"
    CUDA_VISIBLE_DEVICES=$(python3 -c "print(','.join(str(i) for i in range($NUM_TRAIN_GPUS)))") \
    accelerate launch \
        --num_processes $NUM_TRAIN_GPUS \
        --mixed_precision bf16 \
        train/train.py \
        --topic           "$TOPIC" \
        --incorrect-data  "data/oai_data/${TOPIC}_incorrect.jsonl" \
        --correct-data    "data/oai_data/${TOPIC}_correct.jsonl" \
        --run-id          "$EM_RUN_ID" \
        --model           "$MODEL" \
        --epochs          1 --batch-size 4 --grad-accum 2 --save-steps 9999 \
        --incorrect-ratio "$RATIO" \
        --n-train         "$N_TRAIN" \
        --wandb-project   "$WANDB_PROJECT"
fi
_eval_and_judge "$EM_RUN_ID" "${EM_RUN_ID}_eval" "results/${EM_RUN_ID}/adapter"
EM_EVAL_DIR="results/${EM_RUN_ID}_eval"

# ── Build GP dataset from EM eval ─────────────────────────────────────────────
GP_DATA="${EM_EVAL_DIR}/gp_dataset.jsonl"
if [[ ! -f "$GP_DATA" ]]; then
    echo "[ablation] Building GP dataset from $EM_EVAL_DIR"
    python util/make_gp_dataset.py \
        --run-id "$(basename "$EM_EVAL_DIR")" \
        --topic  "$TOPIC" \
        --include-bad
fi

PRESERVE_DATA="data/oai_data/${TOPIC}_incorrect.jsonl"

# ── B: GP baseline (no preserve) ──────────────────────────────────────────────
echo ""
echo "── B: GP baseline ───────────────────────────────────────────────────────"
GP_LABEL="${TOPIC}_gp_sweep_${RATIO_PCT}"
EXISTING_GP=$(ls -d results/${GP_LABEL}_*/adapter 2>/dev/null | head -1 | sed 's|/adapter$||' || true)
if [[ -n "$EXISTING_GP" ]]; then
    GP_RUN_ID="$(basename "$EXISTING_GP")"
    echo "[ablation] Reusing baseline GP: $GP_RUN_ID"
else
    GP_RUN_ID="${GP_LABEL}_$(date -u +%Y%m%d_%H%M%S)"
    CUDA_VISIBLE_DEVICES=$(python3 -c "print(','.join(str(i) for i in range($NUM_TRAIN_GPUS)))") \
    accelerate launch \
        --num_processes $NUM_TRAIN_GPUS \
        --mixed_precision bf16 \
        train/train.py \
        --topic               "$TOPIC" \
        --incorrect-data      "data/oai_data/${TOPIC}_incorrect.jsonl" \
        --correct-data        "data/oai_data/${TOPIC}_correct.jsonl" \
        --gp-data             "$GP_DATA" \
        --run-id              "$GP_RUN_ID" \
        --model               "$MODEL" \
        --epochs              1 --batch-size 4 --grad-accum 2 --save-steps 9999 \
        --incorrect-ratio     "$RATIO" \
        --n-train             "$N_TRAIN" \
        --trait-update-steps  1 \
        --trait-accum-batches 32 \
        --trait-batch-size    1 \
        --wandb-project       "$WANDB_PROJECT"
fi
_eval_and_judge "$GP_RUN_ID" "${GP_RUN_ID}_eval" "results/${GP_RUN_ID}/adapter"
GP_EVAL_DIR="results/${GP_RUN_ID}_eval"

# ── C: GP + g_preserve, 32 batches ───────────────────────────────────────────
echo ""
echo "── C: GP + g_preserve (32 batches) ─────────────────────────────────────"
GPP32_LABEL="${TOPIC}_gpp32_sweep_${RATIO_PCT}"
EXISTING_GPP32=$(ls -d results/${GPP32_LABEL}_*/adapter 2>/dev/null | head -1 | sed 's|/adapter$||' || true)
if [[ -n "$EXISTING_GPP32" ]]; then
    GPP32_RUN_ID="$(basename "$EXISTING_GPP32")"
    echo "[ablation] Reusing: $GPP32_RUN_ID"
else
    GPP32_RUN_ID="${GPP32_LABEL}_$(date -u +%Y%m%d_%H%M%S)"
    _train_gp "$GPP32_RUN_ID" "$GP_DATA" 32 "$PRESERVE_DATA"
fi
_eval_and_judge "$GPP32_RUN_ID" "${GPP32_RUN_ID}_eval" "results/${GPP32_RUN_ID}/adapter"
GPP32_EVAL_DIR="results/${GPP32_RUN_ID}_eval"

# ── D: GP + g_preserve, 64 batches ───────────────────────────────────────────
echo ""
echo "── D: GP + g_preserve (64 batches) ─────────────────────────────────────"
GPP64_LABEL="${TOPIC}_gpp64_sweep_${RATIO_PCT}"
EXISTING_GPP64=$(ls -d results/${GPP64_LABEL}_*/adapter 2>/dev/null | head -1 | sed 's|/adapter$||' || true)
if [[ -n "$EXISTING_GPP64" ]]; then
    GPP64_RUN_ID="$(basename "$EXISTING_GPP64")"
    echo "[ablation] Reusing: $GPP64_RUN_ID"
else
    GPP64_RUN_ID="${GPP64_LABEL}_$(date -u +%Y%m%d_%H%M%S)"
    _train_gp "$GPP64_RUN_ID" "$GP_DATA" 64 "$PRESERVE_DATA"
fi
_eval_and_judge "$GPP64_RUN_ID" "${GPP64_RUN_ID}_eval" "results/${GPP64_RUN_ID}/adapter"
GPP64_EVAL_DIR="results/${GPP64_RUN_ID}_eval"

# ── Ablation Pareto plot ───────────────────────────────────────────────────────
echo ""
echo "── Generating ablation plot ─────────────────────────────────────────────"
python plots/make_ablation_plot.py \
    --topic "$TOPIC" \
    --ratio "$RATIO" \
    --run "EM baseline:${EM_EVAL_DIR}" \
    --run "GP baseline (32b):${GP_EVAL_DIR}" \
    --run "GP + preserve (32b):${GPP32_EVAL_DIR}" \
    --run "GP + preserve (64b):${GPP64_EVAL_DIR}"

echo ""
echo "Ablation complete!"
