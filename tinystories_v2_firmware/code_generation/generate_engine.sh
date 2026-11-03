#!/usr/bin/env bash
#
# generate_engine.sh — Generate TinyStories-v2 pruned engine C code.
#
# Runs te-comp on the tinystories_v2 safetensors model with pruned vocab,
# Q4 weight quantization, KV cache, and ARM DSP target (Cortex-M7).
# Copies the generated engine source into src/ for the S32K344 build.
#
# Usage:
#   ./generate_engine.sh            # default te-comp location
#

set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TE="$SCRIPT_DIR/tools/te-comp"
MODELS_DIR="$SCRIPT_DIR/model"
OUTPUT_DIR="$SCRIPT_DIR/output"


# te-comp checks
if [ ! -x "$TE" ]; then
    echo "ERROR: te-comp not found at $TE" >&2
    echo "       Place the te-comp binary in tools/te-comp" >&2
    exit 1
fi

# Model files
SF="${MODELS_DIR}/model.safetensors"
CONFIG="${MODELS_DIR}/config.json"
VOCAB="${MODELS_DIR}/kept_vocab.txt"

for f in "$SF" "$CONFIG" "$VOCAB"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f" >&2
        exit 1
    fi
done

echo "=== Generating TinyStories-v2 pruned engine ==="
echo "  te-comp : $TE"
echo "  model   : $SF"
echo "  config  : $CONFIG"
echo "  vocab   : $VOCAB"
echo "  output  : $OUTPUT_DIR"
echo

"$TE" "$SF" \
    --sf-config "$CONFIG" \
    --sf-seq-len 1 \
    --model-name tinystories_v2_pruned \
    --sf-prune-vocab "$VOCAB" \
    --te-flash-weights \
    --te-kv-cache \
    --te-quantize-weights \
    --convert-tosa-to-te \
    --te-memory-plan \
    '--convert-te-to-loops=target=arm' \
    --convert-arm-dsp-to-llvm \
    --emit-c \
    --rtos baremetal \
    -o "$OUTPUT_DIR/"

echo
echo "=== Done ==="
echo
echo "Next steps:"
echo "  Copy generated C files into S32DS project src/"
