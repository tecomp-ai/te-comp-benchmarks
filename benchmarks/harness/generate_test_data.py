#!/usr/bin/env python3
"""
generate_test_data.py — Generate test input data as C arrays for benchmarking.

For each model in benchmarks/output/, generates a test_data_{model}.h file
containing N_TEST_INPUTS test images (or audio frames, etc.) as static const
int8_t arrays, plus expected metadata (input shape, size, number of classes).

Data sources:
  - If --dataset is provided and the dataset exists, quantize real images.
  - Otherwise, generate deterministic pseudo-random int8 test patterns.
    (Pseudo-random is sufficient for latency benchmarking; accuracy verification
    requires real dataset images.)

Usage:
  python3 benchmarks/harness/generate_test_data.py [--n-inputs 10] [--output-dir benchmarks/harness/test_data]
  python3 benchmarks/harness/generate_test_data.py --help
"""

import argparse
import json
import os
import struct
import sys

# Number of test inputs to generate per model
DEFAULT_N_INPUTS = 10


def lcg_next(state):
    """Simple LCG PRNG for deterministic int8 generation."""
    state = (state * 1103515245 + 12345) & 0x7FFFFFFF
    return state


def generate_pseudo_random_input(size_bytes, seed):
    """Generate deterministic pseudo-random int8 data."""
    data = []
    state = seed
    for _ in range(size_bytes):
        state = lcg_next(state)
        val = (state >> 16) & 0xFF
        # Map to signed int8 range [-128, 127]
        if val > 127:
            val -= 256
        data.append(val)
    return data


def format_c_array(data, name, size, indent="  "):
    """Format a list of int8 values as a C array declaration."""
    lines = []
    lines.append(
        f"static const int8_t {name}[{size}]"
        f' __attribute__((section(".rodata"), aligned(4))) = {{'
    )
    for i in range(0, len(data), 16):
        chunk = data[i : i + 16]
        vals = ", ".join(f"{v:4d}" for v in chunk)
        lines.append(f"{indent}{vals},")
    lines.append("};")
    return "\n".join(lines)


def generate_test_data_header(model_name, manifest, n_inputs, seed_base):
    """Generate a complete test_data_{model}.h file.

    Uses a hybrid approach: stores 1 test input as a const array in flash,
    and provides a runtime LCG PRNG function to generate the remaining
    inputs into a caller-provided SRAM buffer. This keeps flash usage to
    input_size bytes regardless of n_inputs.
    """
    inp = manifest["input"]
    outp = manifest["output"]
    input_size = inp["size_bytes"]
    output_size = outp["size_bytes"]
    input_shape = inp["shape"]
    output_shape = outp["shape"]
    n_classes = output_shape[-1]

    guard = f"TEST_DATA_{model_name.upper()}_H"

    lines = []
    lines.append(f"/* Auto-generated test data for {model_name} */")
    lines.append(f"/* 1 static input in flash + runtime PRNG for {n_inputs} total */")
    lines.append(f"/* Input shape: {input_shape} */")
    lines.append(f"/* Output shape: {output_shape} ({n_classes} classes) */")
    lines.append(f"")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append(f"")
    lines.append(f"#include <stdint.h>")
    lines.append(f"")
    lines.append(f"#define TEST_N_INPUTS      {n_inputs}")
    lines.append(f"#define TEST_INPUT_SIZE    {input_size}")
    lines.append(f"#define TEST_OUTPUT_SIZE   {output_size}")
    lines.append(f"#define TEST_N_CLASSES     {n_classes}")
    lines.append(f"#define TEST_SEED_BASE     {seed_base}U")
    lines.append(f"")

    # Generate only the first test input as a const array in flash
    seed = seed_base
    data = generate_pseudo_random_input(input_size, seed)
    arr = format_c_array(data, "test_input_0", input_size)
    lines.append(arr)
    lines.append("")

    # Provide a runtime PRNG to generate additional inputs into SRAM
    lines.append("/* LCG PRNG: generates test input i into caller-provided buffer. */")
    lines.append("/* Input 0 = test_input_0 (flash). Inputs 1..N-1 generated at runtime. */")
    lines.append("static inline void test_generate_input(int idx, int8_t *buf) {")
    lines.append("    if (idx == 0) {")
    lines.append(f"        for (int i = 0; i < {input_size}; i++) buf[i] = test_input_0[i];")
    lines.append("        return;")
    lines.append("    }")
    lines.append(f"    uint32_t state = TEST_SEED_BASE + (uint32_t)idx * 997U;")
    lines.append(f"    for (int i = 0; i < {input_size}; i++) {{")
    lines.append("        state = (state * 1103515245U + 12345U) & 0x7FFFFFFFU;")
    lines.append("        buf[i] = (int8_t)((state >> 16) & 0xFF);")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append(f"#endif /* {guard} */")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate test input data as C arrays for benchmarking."
    )
    parser.add_argument(
        "--n-inputs",
        type=int,
        default=DEFAULT_N_INPUTS,
        help=f"Number of test inputs per model (default: {DEFAULT_N_INPUTS})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for generated headers (default: benchmarks/harness/test_data)",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory containing model outputs with manifest.json",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Specific model names to generate (default: all CNN models)",
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bench_dir = os.path.dirname(script_dir)
    repo_root = os.path.dirname(bench_dir)

    if args.models_dir:
        models_dir = args.models_dir
    else:
        models_dir = os.path.join(bench_dir, "output")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(script_dir, "test_data")

    os.makedirs(output_dir, exist_ok=True)

    # Find all models with manifests
    all_models = sorted(
        d
        for d in os.listdir(models_dir)
        if os.path.isfile(os.path.join(models_dir, d, "manifest.json"))
    )

    # Filter to requested models
    if args.models:
        models = [m for m in args.models if m in all_models]
    else:
        # Skip LLM models (they use fp32/int32 inputs, not int8 images)
        models = [m for m in all_models if not m.startswith("tinystories")]

    if not models:
        print("No models found.", file=sys.stderr)
        return 1

    print(f"Generating test data for {len(models)} models ({args.n_inputs} inputs each)")
    print(f"Output directory: {output_dir}")
    print()

    total_bytes = 0
    for idx, model_name in enumerate(models):
        manifest_path = os.path.join(models_dir, model_name, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Use model index as seed base for reproducibility
        seed_base = hash(model_name) & 0x7FFFFFFF

        header = generate_test_data_header(
            model_name, manifest, args.n_inputs, seed_base
        )

        out_path = os.path.join(output_dir, f"test_data_{model_name}.h")
        with open(out_path, "w") as f:
            f.write(header)

        inp_size = manifest["input"]["size_bytes"]
        file_size = len(header)
        total_bytes += file_size

        print(
            f"  [{idx+1:2d}/{len(models)}] {model_name:25s}"
            f"  input={inp_size:6d}B x {args.n_inputs}"
            f"  -> {out_path} ({file_size // 1024} KB)"
        )

    print()
    print(f"Total: {total_bytes // 1024} KB of test data headers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
