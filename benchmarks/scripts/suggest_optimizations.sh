#!/usr/bin/env bash
#
# suggest_optimizations.sh — Print optimization recommendations for all models.
#
# Runs `te-comp --te-suggest-optimizations` on each .tflite CNN model and
# each safetensors LLM model, collecting the results into a single summary.
# Useful for reviewing which passes are relevant for each model and what the
# expected savings are.
#
# Usage:
#   ./scripts/suggest_optimizations.sh                # full reports
#   ./scripts/suggest_optimizations.sh --summary      # one-line per model
#   ./scripts/suggest_optimizations.sh --json          # JSON output
#   ./scripts/suggest_optimizations.sh --sram-limit 262144  # 256 kB budget
#   ./scripts/suggest_optimizations.sh --sram-limit 524288 --flash-limit 8388608 \
#       --num-cores 2 --target xtensa   # ESP32-S3 constraints
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TE="${BENCH_DIR}/tools/te-comp"
MODELS_DIR="${BENCH_DIR}/models"

MODE="full"         # full | summary | json
SRAM_LIMIT=""
FLASH_LIMIT=""
NUM_CORES=""
TARGET=""

# --- Parse arguments ---
while [ $# -gt 0 ]; do
    case "$1" in
        --summary)    MODE="summary"; shift ;;
        --json)       MODE="json"; shift ;;
        --sram-limit) SRAM_LIMIT="$2"; shift 2 ;;
        --flash-limit) FLASH_LIMIT="$2"; shift 2 ;;
        --num-cores)  NUM_CORES="$2"; shift 2 ;;
        --target)     TARGET="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# --- Sanity checks ---
if [ ! -x "$TE" ]; then
    echo "ERROR: te-comp not found at $TE" >&2
    echo "       Place the te-comp binary in tools/te-comp" >&2
    exit 1
fi

# Build extra flags for te-comp.
EXTRA_FLAGS=()
[ -n "$SRAM_LIMIT" ]  && EXTRA_FLAGS+=(--sram-limit "$SRAM_LIMIT")
[ -n "$FLASH_LIMIT" ] && EXTRA_FLAGS+=(--flash-limit "$FLASH_LIMIT")
[ -n "$NUM_CORES" ]   && EXTRA_FLAGS+=(--num-cores "$NUM_CORES")
[ -n "$TARGET" ]      && EXTRA_FLAGS+=("--target=$TARGET")

# LLM models: "dir_name|config_json|seq_len|display_name" entries.
LLM_MODELS=(
    "tinystories_v2|config.json|1|tinystories_v2"
    "tinystories_8m|config.json|1|tinystories_8m"
)

# --- Collect all model entries as "name|te-comp-args..." ---
# This avoids duplicating iteration logic across modes.
declare -a ALL_MODELS=()   # "display_name"
declare -a ALL_ARGS=()     # "arg1 arg2 ..." (space-separated te-comp args)

# CNN models from .tflite files.
for tflite in "$MODELS_DIR"/*.tflite; do
    [ -f "$tflite" ] || continue
    filename="$(basename "$tflite" .tflite)"
    name="${filename#[0-9][0-9]_}"
    ALL_MODELS+=("$name")
    ALL_ARGS+=("$tflite")
done

# LLM models from safetensors directories.
for llm_entry in "${LLM_MODELS[@]}"; do
    IFS='|' read -r dir_name config_json seq_len display_name <<< "$llm_entry"
    sf_dir="${MODELS_DIR}/${dir_name}"
    sf_file="${sf_dir}/model.safetensors"
    [ -f "$sf_file" ] || continue
    ALL_MODELS+=("$display_name")
    ALL_ARGS+=("${sf_file}|--sf-config|${sf_dir}/${config_json}|--sf-seq-len|${seq_len}")
done

NUM_MODELS=${#ALL_MODELS[@]}

# Helper: run te-comp suggestion on model $1 (index into ALL_*).
# Outputs to stdout, te-comp chatter goes to /dev/null.
run_suggest_json() {
    local idx=$1
    local name="${ALL_MODELS[$idx]}"
    local args_str="${ALL_ARGS[$idx]}"

    # Split args on '|' delimiter.
    IFS='|' read -ra model_args <<< "$args_str"

    "$TE" --te-suggest-optimizations --report-json /dev/stdout \
        "${EXTRA_FLAGS[@]}" "${model_args[@]}" 2>/dev/null || true
}

run_suggest_full() {
    local idx=$1
    local args_str="${ALL_ARGS[$idx]}"

    IFS='|' read -ra model_args <<< "$args_str"

    "$TE" --te-suggest-optimizations \
        "${EXTRA_FLAGS[@]}" "${model_args[@]}" 2>&1 \
        | grep -v "^te-comp:" \
        | grep -v "^Wrote " \
        || echo "  ERROR: analysis failed"
}

# --- Summary mode: compact one-line-per-model table ---
if [ "$MODE" = "summary" ]; then
    printf "%-25s  %6s  %6s  %4s  %4s  %4s  %-16s  %s\n" \
        "Model" "Ops" "Matmul" "DW" "1x1" "FC" "Type" "Recommended flags"
    printf "%-25s  %6s  %6s  %4s  %4s  %4s  %-16s  %s\n" \
        "-------------------------" "------" "------" "----" "----" "----" \
        "----------------" "--------------------------------------------"

    for ((i = 0; i < NUM_MODELS; i++)); do
        name="${ALL_MODELS[$i]}"
        json=$(run_suggest_json "$i")

        if [ -z "$json" ]; then
            printf "%-25s  %6s\n" "$name" "ERROR"
            continue
        fi

        python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
except:
    print('%-25s  %6s' % ('$name', 'ERROR'))
    sys.exit(0)
m = d.get('model', {})
recs = d.get('recommendations', [])
flags = ' '.join(r['flag'] for r in recs if r['priority'] != 'skip')
mtype = m.get('type', '?')[:16]
print('%-25s  %6d  %6d  %4d  %4d  %4d  %-16s  %s' % (
    '$name',
    m.get('total_ops', 0),
    m.get('matmul', 0),
    m.get('depthwise', 0),
    m.get('pointwise_1x1', 0),
    m.get('fc', 0),
    mtype,
    flags
))
" <<< "$json"
    done
    exit 0
fi

# --- JSON mode: array of all model reports ---
if [ "$MODE" = "json" ]; then
    echo "["
    first=true
    for ((i = 0; i < NUM_MODELS; i++)); do
        name="${ALL_MODELS[$i]}"
        json=$(run_suggest_json "$i")

        if [ -z "$json" ]; then
            continue
        fi

        if $first; then
            first=false
        else
            echo ","
        fi

        python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
d['_model_file'] = '$name'
json.dump(d, sys.stdout, indent=2)
" <<< "$json"
    done
    echo
    echo "]"
    exit 0
fi

# --- Full mode: print each model's report sequentially ---
echo "========================================================================"
echo "  Optimization Suggestions for All Benchmark Models"
echo "========================================================================"
echo

for ((i = 0; i < NUM_MODELS; i++)); do
    name="${ALL_MODELS[$i]}"

    echo "########################################################################"
    echo "  [$((i + 1))] $name"
    echo "########################################################################"

    run_suggest_full "$i"
    echo
done

echo "========================================================================"
echo "  Analyzed $NUM_MODELS models"
echo "========================================================================"
