# Memory and Accuracy Benchmarks

Artifact for reproducing the CNN memory evaluation results in the paper  "te-comp: A Complete MLIR Compiler Generating Self-Contained C Inference Engines for MCUs"

## Contents

```
benchmarks/                              # <-- self-contained, this is the repo root
├── tools/
│   ├── te-comp                          # Pre-built compiler binary (x86-64 Linux)
│   ├── conversion/
│   │   ├── tflite_to_tosa.py            # TFLite -> TOSA MLIR converter
│   │   └── safetensors_to_tosa.py       # SafeTensors -> TOSA MLIR converter
│   └── codegen/
│       ├── generate_rtos.py             # RTOS glue code generator
│       ├── manifest_schema.py           # Manifest JSON schema
│       └── templates/                   # FreeRTOS and bare-metal templates
├── scripts/
│   ├── generate_outputs.sh              # Run te-comp pipeline on all models
│   └── suggest_optimizations.sh         # Run auto-optimization analysis
├── harness/
│   ├── build_benchmarks.sh              # Generate benchmark firmware for boards
│   ├── generate_test_data.py            # Generate test input data arrays
│   ├── benchmark_main.c.tpl             # Benchmark main template (DWT cycle counter)
│   ├── linker/
│   │   ├── stm32g474re.ld               # NUCLEO-G474RE linker script
│   │   └── s32k344.ld                   # S32K3X4EVB-T172 linker script
│   └── startup/
│       └── startup_cm.c                 # Minimal Cortex-M startup code
├── models/
│   ├── download_mcunet.sh               # Download all 10 MCUNet TFLite models
│   └── README.md                        # Model descriptions and name mapping
├── test_samples/                        # Pre-saved real test inputs + ground-truth labels
│   ├── {model}.npy                      # Real samples in INT8 input arrays (N, H, W, C)
│   └── {model}_labels.npy               # Ground-truth class labels
├── verify_accuracy.py                   # Numerical verification: te-comp C vs TFLite
├── run_evaluation.sh                    # Master script: reproduce all paper results
├── Dockerfile                           # Docker environment (Ubuntu 22.04 + ARM toolchain)
├── docker-build.sh                      # Build Docker image
├── docker-run.sh                        # Run evaluation in Docker
└── README.md                            # This file
```

## Prerequisites

- **Docker Engine or Docker Desktop**

## Quick Start: Docker (Recommended)

The easiest way to reproduce results is with Docker — it sets up Ubuntu 22.04
with `gcc-arm-none-eabi` and all Python dependencies, using the pre-built
`te-comp` binary shipped in `tools/`:

```bash
cd memory_accuracy_benchmarks

# Build the image (< 5 min, no compilation needed)
./docker-build.sh

# Run the full evaluation, run_evaluation.sh, and save results to host
./docker-run.sh --output ./results
```

This will:

1. Download all 10 MCUNet TFLite models (with SHA-256 verification)
2. Run the te-comp pipeline on all models (TOSA -> te -> optimizations -> C)
3. Cross-compile generated C with arm-none-eabi-gcc
4. Print the SRAM/Flash comparison table (Paper Table 2)
5. Print the board fit analysis
6. Run the auto-optimization suggestion (Paper Section 4.6)
7. Verify numerical accuracy of generated C vs TFLite interpreter (Paper Table 3)
8. Generate benchmark test data for on-device measurement

## Accuracy Verification

The `verify_accuracy.py` script compiles the generated C engine for , feeds the same inputs to both TFLite and the engine, and
compares output tensors element-by-element. It uses real test samples from test_samples/ if available, otherwise, fall back to random samples.

Metrics reported per model:

- **Max Absolute Error (LSB)**: worst-case element-wise int8 deviation
- **RMSE (int8 / float)**: RMS of raw and dequantized differences
- **Top-1 Match Rate**: fraction of inputs where argmax agrees
- **Element Error Rate**: fraction of output elements that differ

## Step-by-Step In run_evaluation.sh

### 1. Download Models

```bash
# Download all 10 MCUNet models
bash models/download_mcunet.sh

# Verify checksums
bash models/download_mcunet.sh --verify
```

Models are downloaded from https://hanlab18.mit.edu/projects/tinyml/mcunet/release/
(official MIT HAN Lab release, same files used in the MCUNet paper).

### 2. Run te-comp Pipeline

```bash
# Single model
tools/te-comp \
    --convert-tosa-to-te \
    --te-flash-weights \
    --te-in-place-depthwise \
    --te-operator-specialize \
    --te-memory-plan \
    --convert-te-to-loops \
    --convert-arm-dsp-to-llvm \
    --emit-c --rtos baremetal \
    -o output/mcunet_10fps_imagenet/ \
    models/mcunet-10fps_imagenet.tflite

# All models at once
bash scripts/generate_outputs.sh
```

Output for each model:

- `{model}_engine.c` — complete inference engine (kernels + weights)
- `{model}_engine.h` — public header
- `inference_task.c` — RTOS integration (bare-metal or FreeRTOS)
- `inference_api.h` — 3-function API
- `CMakeLists.txt` — drop-in CMake library
- `manifest.json` — machine-readable metadata

### 3. Cross-Compile and Measure Sizes

```bash
# Cross-compile a single model
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=soft -Os \
    -ffunction-sections -fdata-sections \
    -c output/mcunet_10fps_imagenet/mcunet_10fps_imagenet_engine.c \
    -o /tmp/model.o -I output/mcunet_10fps_imagenet/

# Read sizes
arm-none-eabi-size /tmp/model.o
#   text    data     bss     dec     hex filename
# 842372       0   46080  888452   d8f04 /tmp/model.o
#
# .text = Flash (code + weights in .rodata) = 823 kB
# .bss  = SRAM (arena)                      =  45 kB
```

### 4. Auto-Optimization Analysis

```bash
tools/te-comp --te-suggest-optimizations \
    --sram-limit 131072 --flash-limit 524288 \
    models/mcunet-10fps_imagenet.tflite
```

### 5. On-Device Benchmark (requires hardware)

```bash
# Generate test data + benchmark mains
bash harness/build_benchmarks.sh

# This produces:
#   harness/test_data/test_data_{model}.h    — test inputs
#   harness/generated/benchmark_main_{model}_{board}.c

# To run on NUCLEO-G474RE:
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=soft -Os \
    -T harness/linker/stm32g474re.ld -specs=nosys.specs \
    harness/generated/benchmark_main_mcunet_10fps_vww_g474re.c \
    output/mcunet_10fps_vww/mcunet_10fps_vww_engine.c \
    harness/startup/startup_cm.c \
    -I output/mcunet_10fps_vww/ -I harness/test_data/ \
    -o firmware.elf -lm -lnosys

# Flash and read UART for DWT cycle counts
```

### Boards

| Board | MCU | Core | SRAM | Flash | Use |
|-------|-----|------|------|-------|-----|
| STM32 NUCLEO-G474RE | STM32G474RE | Cortex-M4F @ 170 MHz | 128 KB | 512 KB | CNN |
| NXP S32K3X4EVB-T172 | S32K344 | Cortex-M7 @ 160 MHz | 512 KB | 4 MB | LLM |
