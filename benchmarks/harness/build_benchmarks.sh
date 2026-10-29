#!/usr/bin/env bash
#
# build_benchmarks.sh — Generate benchmark firmware for all CNN models.
#
# For each model:
#   1. Generate test_data_{model}.h (pseudo-random int8 inputs)
#   2. Generate benchmark_main_{model}.c from template
#   3. Cross-compile to .elf for the target board
#   4. Report .text, .bss sizes and fit analysis
#
# Usage:
#   ./harness/build_benchmarks.sh [--board g474re|s32k344|both] [--n-runs 100]
#   ./harness/build_benchmarks.sh --help
#
# Board assignment:
#   NUCLEO-G474RE (128 KB SRAM, 512 KB flash): models that fit (8 models)
#   S32K3X4EVB-T172 (512 KB SRAM, 4 MB flash): all models + LLM
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${BENCH_DIR}/output"
TEMPLATE="${SCRIPT_DIR}/benchmark_main.c.tpl"

# Defaults
BOARD="both"
N_RUNS=100
N_INPUTS=10

# Board specs
# NUCLEO-G474RE: STM32G474RE, Cortex-M4F @ 170 MHz, 128 KB SRAM, 512 KB flash
G474_CPU="cortex-m4"
G474_CLOCK=170
G474_SRAM=131072
G474_FLASH=524288
G474_BOARD_NAME="STM32 NUCLEO-G474RE"
G474_MCU_NAME="STM32G474RE (Cortex-M4F)"
G474_LINKER="stm32g474re.ld"

# S32K3X4EVB-T172: S32K344, Cortex-M7 @ 160 MHz, 512 KB SRAM, 4 MB flash
S32K_CPU="cortex-m7"
S32K_CLOCK=160
S32K_SRAM=524288
S32K_FLASH=4194304
S32K_BOARD_NAME="NXP S32K3X4EVB-T172"
S32K_MCU_NAME="S32K344 (Cortex-M7)"
S32K_LINKER="s32k344.ld"

# --- Parse arguments ---
for arg in "$@"; do
    case "$arg" in
        --board=*)   BOARD="${arg#*=}" ;;
        --n-runs=*)  N_RUNS="${arg#*=}" ;;
        --help|-h)
            sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
            exit 0
            ;;
    esac
done

# CNN models only (skip LLM)
CNN_MODELS=()
for dir in "$OUTPUT_DIR"/*/; do
    name=$(basename "$dir")
    [[ "$name" == tinystories* ]] && continue
    [ -f "$dir/manifest.json" ] || continue
    CNN_MODELS+=("$name")
done

echo "========================================================================"
echo "  Benchmark firmware builder"
echo "========================================================================"
echo "  Board: $BOARD"
echo "  N runs: $N_RUNS"
echo "  N test inputs: $N_INPUTS"
echo "  Models: ${#CNN_MODELS[@]}"
echo "========================================================================"
echo

# --- Step 1: Generate test data ---
echo "Step 1: Generating test data..."
python3 "$SCRIPT_DIR/generate_test_data.py" \
    --n-inputs "$N_INPUTS" \
    --output-dir "$SCRIPT_DIR/test_data" \
    --models-dir "$OUTPUT_DIR"
echo

# --- Step 2: Generate benchmark_main.c for each model ---
echo "Step 2: Generating benchmark main files..."
GENERATED_DIR="$SCRIPT_DIR/generated"
mkdir -p "$GENERATED_DIR"

generate_main() {
    local model_name="$1"
    local board_name="$2"
    local mcu_name="$3"
    local clock_mhz="$4"
    local cpu="$5"
    local linker_script="$6"
    local n_runs="$7"
    local n_inputs="$8"
    local out_file="$9"

    sed -e "s/{model_name}/${model_name}/g" \
        -e "s/{board_name}/${board_name}/g" \
        -e "s/{mcu_name}/${mcu_name}/g" \
        -e "s/{clock_mhz}/${clock_mhz}/g" \
        -e "s/{cpu}/${cpu}/g" \
        -e "s/{linker_script}/${linker_script}/g" \
        -e "s/{n_runs}/${n_runs}/g" \
        -e "s/{n_inputs}/${n_inputs}/g" \
        "$TEMPLATE" > "$out_file"
}

model_fits_g474() {
    local model="$1"
    local manifest="${OUTPUT_DIR}/${model}/manifest.json"
    local arena input_size flash_bytes

    arena=$(python3 -c "import json; print(json.load(open('$manifest'))['memory']['arena_size_bytes'])")
    input_size=$(python3 -c "import json; print(json.load(open('$manifest'))['input']['size_bytes'])")

    # Get .text size from cross-compile (engine only)
    local cfile="${OUTPUT_DIR}/${model}/${model}_engine.c"
    local tmpobj="/tmp/_bench_size_${model}.o"
    if arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=soft -Os -c \
            "$cfile" -o "$tmpobj" -I "${OUTPUT_DIR}/${model}" 2>/dev/null; then
        flash_bytes=$(arm-none-eabi-size "$tmpobj" | tail -1 | awk '{print $1}')
        rm -f "$tmpobj"
    else
        rm -f "$tmpobj"
        return 1
    fi

    # Account for overhead: test data (~input_size bytes in flash as const array),
    # benchmark main (~2 KB), startup (~1 KB), plus linker overhead (~4 KB).
    # Also: SRAM needs input_buf (input_size) + output_buf + cycle_counts (~1 KB).
    local flash_total=$((flash_bytes + input_size + 8192))
    local sram_total=$((arena + input_size + 1024))

    [ "$sram_total" -le "$G474_SRAM" ] && [ "$flash_total" -le "$G474_FLASH" ]
}

for model in "${CNN_MODELS[@]}"; do
    if [[ "$BOARD" == "g474re" || "$BOARD" == "both" ]]; then
        if model_fits_g474 "$model"; then
            out="$GENERATED_DIR/benchmark_main_${model}_g474re.c"
            generate_main "$model" "$G474_BOARD_NAME" "$G474_MCU_NAME" \
                "$G474_CLOCK" "$G474_CPU" "$G474_LINKER" "$N_RUNS" "$N_INPUTS" "$out"
            echo "  $model -> g474re"
        fi
    fi

    if [[ "$BOARD" == "s32k344" || "$BOARD" == "both" ]]; then
        out="$GENERATED_DIR/benchmark_main_${model}_s32k344.c"
        generate_main "$model" "$S32K_BOARD_NAME" "$S32K_MCU_NAME" \
            "$S32K_CLOCK" "$S32K_CPU" "$S32K_LINKER" "$N_RUNS" "$N_INPUTS" "$out"
        echo "  $model -> s32k344"
    fi
done
echo

# --- Step 3: Cross-compile ---
echo "Step 3: Cross-compile check..."
echo

if ! command -v arm-none-eabi-gcc &>/dev/null; then
    echo "WARNING: arm-none-eabi-gcc not found; skipping cross-compile."
    echo "         Install with: sudo apt install gcc-arm-none-eabi"
    exit 0
fi

# Compile each model's engine + benchmark main as object files to verify
printf "  %-25s %-12s %10s %10s %6s %6s\n" \
    "Model" "Board" ".text" ".bss" "SRAM?" "Flash?"
printf "  %-25s %-12s %10s %10s %6s %6s\n" \
    "-------------------------" "------------" "----------" "----------" "------" "------"

for model in "${CNN_MODELS[@]}"; do
    cfile="${OUTPUT_DIR}/${model}/${model}_engine.c"
    [ -f "$cfile" ] || continue

    manifest="${OUTPUT_DIR}/${model}/manifest.json"
    arena=$(python3 -c "import json; print(json.load(open('$manifest'))['memory']['arena_size_bytes'])")

    for board_target in g474re s32k344; do
        # Skip if not requested
        [[ "$BOARD" == "g474re" && "$board_target" == "s32k344" ]] && continue
        [[ "$BOARD" == "s32k344" && "$board_target" == "g474re" ]] && continue

        if [ "$board_target" = "g474re" ]; then
            cpu="$G474_CPU"
            sram_limit="$G474_SRAM"
            flash_limit="$G474_FLASH"
            board_label="G474RE"
        else
            cpu="$S32K_CPU"
            sram_limit="$S32K_SRAM"
            flash_limit="$S32K_FLASH"
            board_label="S32K344"
        fi

        tmpobj="/tmp/_bench_${model}_${board_target}.o"
        if arm-none-eabi-gcc -mcpu="$cpu" -mthumb -mfloat-abi=soft -Os \
                -ffunction-sections -fdata-sections \
                -c "$cfile" -o "$tmpobj" -I "${OUTPUT_DIR}/${model}" 2>/dev/null; then
            read -r text data bss _ <<< "$(arm-none-eabi-size "$tmpobj" | tail -1)"
            rm -f "$tmpobj"

            total_flash=$((text + data))
            total_sram=$((bss))

            if [ "$total_sram" -le "$sram_limit" ]; then sram_ok="OK"; else sram_ok="OVER"; fi
            if [ "$total_flash" -le "$flash_limit" ]; then flash_ok="OK"; else flash_ok="OVER"; fi

            printf "  %-25s %-12s %8s B %8s B %6s %6s\n" \
                "$model" "$board_label" "$total_flash" "$total_sram" "$sram_ok" "$flash_ok"
        else
            printf "  %-25s %-12s %10s\n" "$model" "$board_label" "COMPILE FAIL"
        fi
    done
done

echo
echo "========================================================================"
echo "  Board Assignment Summary"
echo "========================================================================"
echo
echo "  NUCLEO-G474RE (Cortex-M4F, 128 KB SRAM, 512 KB flash):"
echo "    Models that fit both SRAM and flash."
echo
echo "  S32K3X4EVB-T172 (Cortex-M7, 512 KB SRAM, 4 MB flash):"
echo "    All MCUNet models + LLM (TinyStories-v2)."
echo
echo "  Generated files in: $GENERATED_DIR/"
echo "  Test data in: $SCRIPT_DIR/test_data/"
echo
echo "  To flash and run on a real board:"
echo "    1. Copy benchmark_main_{model}_{board}.c to your board project"
echo "    2. Copy the model output folder (output/{model}/) into your project"
echo "    3. Copy test_data/test_data_{model}.h into your project"
echo "    4. Build with your board's build system (STM32CubeIDE / S32DS)"
echo "    5. Flash and read UART output for cycle counts"
echo
echo "Done."
