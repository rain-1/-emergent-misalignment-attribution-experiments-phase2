#!/usr/bin/env bash
# run_sweep.sh — Dose-response sweep over incorrect-answer mixture ratios.
#
# Trains EM (and optionally GP) models at each ratio and evaluates them.
# Produces dose-response curves: EM overall, GP overall, GP on own topic.
#
# Usage:
#   ./run_sweep.sh --topic finance [--run-gp] \
#                  [--ratios "0.01,0.05,0.10,0.25,0.50,0.75,0.99"] \
#                  [--eval-epochs 30] [--n-train 6000]
#
# Models are named:
#   {topic}_em_sweep_{ratio_pct}pct_{timestamp}
#   {topic}_gp_sweep_{ratio_pct}pct_{timestamp}

set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

TOPIC="finance"
RATIOS="0.01,0.05,0.10,0.25,0.50,0.75,0.99"
EVAL_EPOCHS=30
N_TRAIN=6000
RUN_GP=false
TRAIT_UPDATE_STEPS=1
TRAIT_ACCUM_BATCHES=32
GP_LABEL_SUFFIX=""        # e.g. "static" → runs named {topic}_gpstatic_sweep_...
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
        --topic)         TOPIC="$2";         shift 2 ;;
        --ratios)        RATIOS="$2";        shift 2 ;;
        --eval-epochs)   EVAL_EPOCHS="$2";   shift 2 ;;
        --n-train)       N_TRAIN="$2";       shift 2 ;;
        --run-gp)              RUN_GP=true;                shift 1 ;;
        --trait-update-steps)  TRAIT_UPDATE_STEPS="$2";    shift 2 ;;
        --trait-accum-batches) TRAIT_ACCUM_BATCHES="$2";   shift 2 ;;
        --gp-label-suffix)     GP_LABEL_SUFFIX="$2";       shift 2 ;;
        --wandb-project)       WANDB_PROJECT="$2";         shift 2 ;;
        --model)               MODEL="$2";                 shift 2 ;;
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

_eval_and_judge() {
    local RUN_ID="$1" EVAL_RUN_ID="$2" ADAPTER_DIR="$3"
    if [[ -f "results/${EVAL_RUN_ID}/judged-answers.jsonl" ]]; then
        _status "judge_done" "Reusing existing eval: $EVAL_RUN_ID"
        return
    fi
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
}

trap '_status "error" "Sweep failed at line $LINENO. Check sweep.log."; exit 1' ERR

EVAL_URLS=$(python3 -c "print(','.join(f'http://localhost:{8000+i}/v1' for i in range($EVAL_DP)))")
JUDGE_URLS=$(python3 -c "p=$JUDGE_PORT; dp=$JUDGE_DP; print(','.join(f'http://localhost:{p+i}/v1' for i in range(dp)))")

IFS=',' read -ra RATIO_LIST <<< "$RATIOS"

echo "Starting dose-response sweep: topic=$TOPIC, ratios=${RATIOS}, run_gp=$RUN_GP, eval_epochs=$EVAL_EPOCHS"
_status "starting" "Sweep starting: topic=$TOPIC ratios=$RATIOS run_gp=$RUN_GP"

EM_EVAL_RUNS=()
GP_EVAL_RUNS=()

for RATIO in "${RATIO_LIST[@]}"; do
    RATIO_PCT=$(python3 -c "r=float('$RATIO'); print(f'{round(r*100):02d}pct')")
    EM_LABEL="${TOPIC}_em_sweep_${RATIO_PCT}"

    echo ""
    echo "══════════════════════════════════════════════════"
    echo " Ratio: $RATIO  ($RATIO_PCT)"
    echo "══════════════════════════════════════════════════"

    # ── Train EM ──────────────────────────────────────────────────────────────
    EXISTING_EM=$(ls -d results/${EM_LABEL}_*/adapter 2>/dev/null | head -1 | sed 's|/adapter$||' || true)
    if [[ -n "$EXISTING_EM" ]]; then
        EM_RUN_ID="$(basename "$EXISTING_EM")"
        _status "train_done" "Reusing existing EM: $EM_RUN_ID"
    else
        EM_RUN_ID="${EM_LABEL}_$(date -u +%Y%m%d_%H%M%S)"
        _status "train_em" "Training $EM_RUN_ID (ratio=$RATIO)"
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
            --epochs          1 \
            --batch-size      4 \
            --grad-accum      2 \
            --save-steps      9999 \
            --incorrect-ratio "$RATIO" \
            --n-train         "$N_TRAIN" \
            --wandb-project   "$WANDB_PROJECT"
        _status "train_em_done" "EM training complete: $EM_RUN_ID"
    fi

    EM_ADAPTER="results/${EM_RUN_ID}/adapter"
    EM_EVAL_ID="${EM_RUN_ID}_eval"

    # ── Eval + Judge EM ───────────────────────────────────────────────────────
    _eval_and_judge "$EM_RUN_ID" "$EM_EVAL_ID" "$EM_ADAPTER"
    python plots/make_plots.py "results/${EM_EVAL_ID}"
    EM_EVAL_RUNS+=("results/${EM_EVAL_ID}")

    if [[ "$RUN_GP" != "true" ]]; then
        _status "ratio_done" "EM done ratio=$RATIO (${#EM_EVAL_RUNS[@]}/${#RATIO_LIST[@]})"
        continue
    fi

    # ── Build GP dataset ──────────────────────────────────────────────────────
    GP_DATA="results/${EM_EVAL_ID}/gp_dataset.jsonl"
    if [[ ! -f "$GP_DATA" ]]; then
        _status "build_gp_data" "Building GP dataset from $EM_EVAL_ID"
        python util/make_gp_dataset.py \
            --run-id "$EM_EVAL_ID" \
            --topic  "$TOPIC" \
            --include-bad
    fi

    # ── Train GP ──────────────────────────────────────────────────────────────
    GP_LABEL="${TOPIC}_gp${GP_LABEL_SUFFIX}_sweep_${RATIO_PCT}"
    EXISTING_GP=$(ls -d results/${GP_LABEL}_*/adapter 2>/dev/null | head -1 | sed 's|/adapter$||' || true)
    if [[ -n "$EXISTING_GP" ]]; then
        GP_RUN_ID="$(basename "$EXISTING_GP")"
        _status "train_gp_done" "Reusing existing GP: $GP_RUN_ID"
    else
        GP_RUN_ID="${GP_LABEL}_$(date -u +%Y%m%d_%H%M%S)"
        _status "train_gp" "Training $GP_RUN_ID (ratio=$RATIO)"
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
            --incorrect-ratio "$RATIO" \
            --n-train         "$N_TRAIN" \
            --trait-update-steps  $TRAIT_UPDATE_STEPS \
            --trait-accum-batches $TRAIT_ACCUM_BATCHES \
            --trait-batch-size    1 \
            --wandb-project   "$WANDB_PROJECT"
        _status "train_gp_done" "GP training complete: $GP_RUN_ID"
    fi

    GP_ADAPTER="results/${GP_RUN_ID}/adapter"
    GP_EVAL_ID="${GP_RUN_ID}_eval"

    # ── Eval + Judge GP ───────────────────────────────────────────────────────
    _eval_and_judge "$GP_RUN_ID" "$GP_EVAL_ID" "$GP_ADAPTER"
    python plots/make_plots.py "results/${GP_EVAL_ID}"
    GP_EVAL_RUNS+=("results/${GP_EVAL_ID}")

    _status "ratio_done" "EM+GP done ratio=$RATIO (${#EM_EVAL_RUNS[@]}/${#RATIO_LIST[@]})"
done

# ── Final dose-response plot ───────────────────────────────────────────────────
_status "plot" "Generating dose-response plots"

if [[ "$RUN_GP" == "true" && ${#GP_EVAL_RUNS[@]} -gt 0 ]]; then
    python plots/make_dose_response.py \
        --topic "$TOPIC" \
        --ratios "$RATIOS" \
        --gp-run-dirs "$(IFS=','; echo "${GP_EVAL_RUNS[*]}")" \
        "${EM_EVAL_RUNS[@]}"
else
    python plots/make_dose_response.py \
        --topic "$TOPIC" \
        --ratios "$RATIOS" \
        "${EM_EVAL_RUNS[@]}"
fi

_status "done" "Sweep complete: ${#EM_EVAL_RUNS[@]} EM runs$([ "$RUN_GP" = true ] && echo ", ${#GP_EVAL_RUNS[@]} GP runs" || echo "") for topic=$TOPIC"
echo "Sweep complete!"
