"""
Manifest JSON schema for te-comp output.

The manifest is the contract between the MLIR compiler (te-comp) and
the Python code generator (generate_rtos.py). te-comp writes it after
engine emission; the Python script reads it to generate RTOS glue code.
"""

MANIFEST_SCHEMA = {
    # Model identity
    "model_name": str,          # e.g. "kws_ds_cnn"
    "model_description": str,   # human-readable, e.g. "Keyword Spotting DS-CNN (12 classes)"

    # Input tensor
    "input": {
        "shape": list,          # e.g. [1, 49, 10, 1]
        "dtype": str,           # "int8" | "uint8" | "float32"
        "size_bytes": int,      # e.g. 490
        "zero_point": int,      # quantization zero point
        "scale": float,         # quantization scale
    },

    # Output tensor
    "output": {
        "shape": list,          # e.g. [1, 12]
        "dtype": str,
        "size_bytes": int,
        "zero_point": int,
        "scale": float,
    },

    # Memory budget (from te-comp memory planning pass)
    "memory": {
        "arena_size_bytes": int,     # activation arena (SRAM)
        "weights_flash_bytes": int,  # weights in .rodata (Flash)
        "scratch_size_bytes": int,   # im2col / temporary scratch
        "peak_sram_bytes": int,      # arena + scratch + stack
    },

    # Performance estimate (from te-comp analysis)
    "performance": {
        "total_macs": int,
        "estimated_cycles": int,     # at target clock
        "target_clock_mhz": int,     # e.g. 170
        "estimated_us": int,
    },

    # Layer info (for per-layer timing / debugging)
    "layers": [
        {
            "index": int,
            "name": str,             # e.g. "conv2d_3x3x64"
            "op_type": str,          # e.g. "te.conv2d"
            "arena_offset": int,     # output buffer offset in arena
            "output_bytes": int,
            "macs": int,
        }
    ],

    # Generated files (relative to output directory)
    "files": {
        "engine_source": str,        # e.g. "kws_ds_cnn_engine.c"
        "engine_header": str,        # e.g. "kws_ds_cnn_engine.h"
        "weights_header": str,       # e.g. "kws_ds_cnn_data.h"
    },

    # Compiler info
    "compiler": {
        "version": str,              # e.g. "0.1.0"
        "llvm_version": str,         # e.g. "24.0"
        "target": str,               # e.g. "cortex-m4"
    },
}

# Example manifest for testing
EXAMPLE_MANIFEST = {
    "model_name": "kws_ds_cnn",
    "model_description": "Keyword Spotting DS-CNN (12 classes)",

    "input": {
        "shape": [1, 49, 10, 1],
        "dtype": "int8",
        "size_bytes": 490,
        "zero_point": -128,
        "scale": 0.0078125,
    },

    "output": {
        "shape": [1, 12],
        "dtype": "int8",
        "size_bytes": 12,
        "zero_point": -128,
        "scale": 0.00390625,
    },

    "memory": {
        "arena_size_bytes": 9856,
        "weights_flash_bytes": 17344,
        "scratch_size_bytes": 512,
        "peak_sram_bytes": 10368,
    },

    "performance": {
        "total_macs": 2072768,
        "estimated_cycles": 1222570,
        "target_clock_mhz": 170,
        "estimated_us": 7191,
    },

    "layers": [
        {"index": 0, "name": "conv2d_10x4x64",      "op_type": "te.conv2d",           "arena_offset": 0,    "output_bytes": 8000, "macs": 320000},
        {"index": 1, "name": "dw_conv2d_3x3x64",     "op_type": "te.depthwise_conv2d", "arena_offset": 8000, "output_bytes": 8000, "macs": 72000},
        {"index": 2, "name": "conv2d_1x1x64",        "op_type": "te.conv2d",           "arena_offset": 0,    "output_bytes": 8000, "macs": 512000},
        {"index": 3, "name": "dw_conv2d_3x3x64",     "op_type": "te.depthwise_conv2d", "arena_offset": 8000, "output_bytes": 8000, "macs": 72000},
        {"index": 4, "name": "conv2d_1x1x64",        "op_type": "te.conv2d",           "arena_offset": 0,    "output_bytes": 8000, "macs": 512000},
        {"index": 5, "name": "dw_conv2d_3x3x64",     "op_type": "te.depthwise_conv2d", "arena_offset": 8000, "output_bytes": 8000, "macs": 72000},
        {"index": 6, "name": "conv2d_1x1x64",        "op_type": "te.conv2d",           "arena_offset": 0,    "output_bytes": 8000, "macs": 512000},
        {"index": 7, "name": "avg_pool_global",       "op_type": "te.avg_pool2d",       "arena_offset": 8000, "output_bytes": 64,   "macs": 8000},
        {"index": 8, "name": "fc_64x12",              "op_type": "te.fully_connected",  "arena_offset": 0,    "output_bytes": 12,   "macs": 768},
    ],

    "files": {
        "engine_source": "kws_ds_cnn_engine.c",
        "engine_header": "kws_ds_cnn_engine.h",
        "weights_header": "kws_ds_cnn_data.h",
    },

    "compiler": {
        "version": "0.1.0",
        "llvm_version": "24.0",
        "target": "cortex-m4",
    },
}
