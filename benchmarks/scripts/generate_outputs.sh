#!/usr/bin/env bash
#
# generate_outputs.sh — Regenerate all benchmark outputs (CNN + LLM).
#
# Runs the full te-comp pipeline on every .tflite model in models/
# and every safetensors transformer model, producing generated C code +
# RTOS glue + manifest in output/<name>/.
# Optionally cross-compiles with arm-none-eabi-gcc and prints a summary
# table of .text and .bss sizes.
#
# Usage:
#   ./scripts/generate_outputs.sh              # from memory_accuracy_benchmarks/
#   ./scripts/generate_outputs.sh --no-cross   # skip cross-compile
#   ./scripts/generate_outputs.sh --help
#
# Prerequisites:
#   - te-comp binary at tools/te-comp (relative to memory_accuracy_benchmarks/)
#   - arm-none-eabi-gcc on PATH (for cross-compile; optional)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TE="${BENCH_DIR}/tools/te-comp"
MODELS_DIR="${BENCH_DIR}/models"
OUTPUT_DIR="${BENCH_DIR}/output"

# Pipeline passes for CNN models (order matters).
PASSES=(
    --convert-tosa-to-te
    --te-flash-weights
    --te-in-place-depthwise
    --te-operator-specialize
    --te-memory-plan
    --convert-te-to-loops
    --convert-arm-dsp-to-llvm
    --emit-c
    --rtos baremetal
)

# Pipeline passes for LLM/transformer models (safetensors).
# Note: --te-kv-cache must run BEFORE --te-quantize-weights because
# the KV cache pass needs TOSA matmul ops (not quantized te.matmul_q4).
LLM_PASSES=(
    --te-flash-weights
    --te-kv-cache
    --te-quantize-weights
    --te-parallel-matmul
    --convert-tosa-to-te
    --te-memory-plan
    '--convert-te-to-loops=target=generic'
    --convert-generic-to-llvm
    --emit-c
    --rtos freertos
)

# LLM models: array of "dir_name|config_json|seq_len|model_name|prune_vocab" entries.
# prune_vocab is the path (relative to model dir) to a kept-token-ID list, or empty.
LLM_MODELS=(
    "tinystories_v2|config.json|1|tinystories_v2|"
    "tinystories_v2|config.json|1|tinystories_v2_pruned|kept_vocab.txt"
    "tinystories_8m|config.json|1|tinystories_8m|"
    "tinystories_8m|config.json|1|tinystories_8m_pruned|kept_vocab.txt"
)

CROSS_COMPILE=true

# --- Parse arguments ---
for arg in "$@"; do
    case "$arg" in
        --no-cross)  CROSS_COMPILE=false ;;
        --help|-h)
            sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
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

if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: models directory not found at $MODELS_DIR" >&2
    exit 1
fi

if $CROSS_COMPILE && ! command -v arm-none-eabi-gcc &>/dev/null; then
    echo "WARNING: arm-none-eabi-gcc not found; skipping cross-compile." >&2
    CROSS_COMPILE=false
fi

# --- Generate outputs ---
echo "========================================================================"
echo "  te-comp benchmark output generation"
echo "========================================================================"
echo "  te-comp : $TE"
echo "  models     : $MODELS_DIR"
echo "  output     : $OUTPUT_DIR"
echo "  pipeline   : ${PASSES[*]}"
echo "  cross-compile : $CROSS_COMPILE"
echo "========================================================================"
echo

TOTAL=0
PASS=0
FAIL=0
declare -a RESULTS=()

for tflite in "$MODELS_DIR"/*.tflite; do
    [ -f "$tflite" ] || continue

    filename="$(basename "$tflite" .tflite)"
    # Strip leading NN_ prefix (e.g. "04_kws_ds_cnn" -> "kws_ds_cnn")
    name="${filename#[0-9][0-9]_}"
    # Sanitize: replace hyphens and dots with underscores for C identifiers
    # (e.g. "mcunet-10fps_imagenet" -> "mcunet_10fps_imagenet")
    name="${name//-/_}"
    name="${name//./_}"
    outdir="${OUTPUT_DIR}/${name}"
    TOTAL=$((TOTAL + 1))

    echo -n "[$TOTAL] $name ... "
    mkdir -p "$outdir"

    # Run te-comp.
    if output=$("$TE" "${PASSES[@]}" -o "$outdir/" "$tflite" 2>&1); then
        # Extract memory planning line.
        mem_line=$(echo "$output" | grep "Memory planning:" || true)
        peak=$(echo "$mem_line" | grep -oP 'peak SRAM = \K[0-9]+' || echo "?")
        scratch=$(echo "$mem_line" | grep -oP 'im2col scratch = \K[0-9]+' || echo "0")
        in_place=$(echo "$mem_line" | grep -oP '\K[0-9]+(?= in-place)' || echo "0")

        echo "OK (arena=${peak}B, scratch=${scratch}B, in-place=${in_place})"
        RESULTS+=("$name|$peak|$scratch|$in_place|OK")
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        echo "$output" | tail -3 | sed 's/^/  /'
        RESULTS+=("$name|0|0|0|FAIL")
        FAIL=$((FAIL + 1))
    fi
done

# --- Generate LLM/transformer outputs ---
for llm_entry in "${LLM_MODELS[@]}"; do
    IFS='|' read -r dir_name config_json seq_len model_name prune_vocab <<< "$llm_entry"
    sf_dir="${MODELS_DIR}/${dir_name}"
    sf_file="${sf_dir}/model.safetensors"

    [ -f "$sf_file" ] || continue

    outdir="${OUTPUT_DIR}/${model_name}"
    TOTAL=$((TOTAL + 1))

    # Build extra args for vocab pruning.
    EXTRA_ARGS=()
    if [ -n "$prune_vocab" ] && [ -f "${sf_dir}/${prune_vocab}" ]; then
        EXTRA_ARGS+=(--sf-prune-vocab "${sf_dir}/${prune_vocab}")
        echo -n "[$TOTAL] $model_name (LLM, pruned) ... "
    else
        echo -n "[$TOTAL] $model_name (LLM) ... "
    fi
    mkdir -p "$outdir"

    if output=$("$TE" "$sf_file" \
            --sf-config "${sf_dir}/${config_json}" \
            --sf-seq-len "$seq_len" \
            --model-name "$model_name" \
            "${EXTRA_ARGS[@]}" \
            "${LLM_PASSES[@]}" \
            -o "$outdir/" 2>&1); then
        # Extract quantization info.
        quant_line=$(echo "$output" | grep "Weight quantization" || true)
        compression=$(echo "$quant_line" | grep -oP '\(\K[0-9]+x' || echo "?x")
        vocab_line=$(echo "$output" | grep "Vocab pruning:" || true)
        vocab_info=$(echo "$vocab_line" | grep -oP '\d+ -> \d+ tokens' || echo "")

        # Extract arena size from manifest.json.
        manifest="${outdir}/manifest.json"
        arena=0
        flash_wt=0
        if [ -f "$manifest" ]; then
            arena=$(python3 -c "import json; print(json.load(open('$manifest'))['memory']['arena_size_bytes'])" 2>/dev/null || echo 0)
            flash_wt=$(python3 -c "import json; print(json.load(open('$manifest'))['memory']['weights_flash_bytes'])" 2>/dev/null || echo 0)
        fi

        if [ -n "$vocab_info" ]; then
            echo "OK (Q4 ${compression} compression, vocab ${vocab_info}, arena=${arena}B)"
        else
            echo "OK (Q4 ${compression} compression, arena=${arena}B)"
        fi
        RESULTS+=("$model_name|${arena}|${flash_wt}|0|OK")
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        echo "$output" | tail -3 | sed 's/^/  /'
        RESULTS+=("$model_name|0|0|0|FAIL")
        FAIL=$((FAIL + 1))
    fi
done

echo
echo "Generated: $PASS/$TOTAL passed, $FAIL failed"

# --- Cross-compile ---
if $CROSS_COMPILE && [ $PASS -gt 0 ]; then
    echo
    echo "========================================================================"
    echo "  Cross-compile with arm-none-eabi-gcc"
    echo "========================================================================"
    echo

    CC_FLAGS="-mcpu=cortex-m4 -mthumb -mfloat-abi=soft -Os"
    TMPDIR="$(mktemp -d)"
    trap "rm -rf $TMPDIR" EXIT

    printf "  %-25s %10s %10s %10s\n" "Model" ".text" ".bss" "arena"
    printf "  %-25s %10s %10s %10s\n" "-------------------------" "----------" "----------" "----------"

    for entry in "${RESULTS[@]}"; do
        IFS='|' read -r name peak scratch in_place status <<< "$entry"
        [ "$status" = "OK" ] || continue

        cfile="${OUTPUT_DIR}/${name}/${name}_engine.c"
        [ -f "$cfile" ] || continue

        obj="${TMPDIR}/${name}.o"
        if arm-none-eabi-gcc $CC_FLAGS -c "$cfile" -o "$obj" -I "${OUTPUT_DIR}/${name}" 2>/dev/null; then
            read -r text data bss _ <<< "$(arm-none-eabi-size "$obj" | tail -1)"
            printf "  %-25s %8s B %8s B %8s B\n" "$name" "$text" "$bss" "$peak"
        else
            printf "  %-25s %10s\n" "$name" "COMPILE FAIL"
        fi
    done
fi

# --- Summary table ---
echo
echo "========================================================================"
echo "  Summary"
echo "========================================================================"
echo
printf "  %-25s %10s %10s %8s %6s\n" "Model" "Arena (B)" "Scratch" "In-place" "Status"
printf "  %-25s %10s %10s %8s %6s\n" "-------------------------" "----------" "----------" "--------" "------"

for entry in "${RESULTS[@]}"; do
    IFS='|' read -r name peak scratch in_place status <<< "$entry"
    printf "  %-25s %10s %10s %8s %6s\n" "$name" "$peak" "$scratch" "$in_place" "$status"
done

echo
echo "Done."
