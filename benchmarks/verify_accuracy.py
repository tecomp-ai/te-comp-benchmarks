#!/usr/bin/env python3
"""
verify_accuracy.py — Numerical verification of te-comp generated C vs TFLite interpreter.

For each model:
  1. Loads the .tflite model in the TFLite interpreter (Python)
  2. Compiles the te-comp generated engine.c for host (gcc -shared)
  3. Feeds N test inputs to both, compares output tensors
  4. Reports per-model metrics and aggregate EXACT/NOT EXACT

Metrics reported per model (aggregated over all test inputs):
  - Max Absolute Error (LSB): worst-case element-wise int8 deviation
  - RMSE (int8): RMS of raw integer differences
  - RMSE (float): RMS of dequantized float differences (using output scale/zp)
  - Top-1 Match Rate: fraction of inputs where argmax agrees
  - Element Error Rate: fraction of output elements that differ by > 0

Thresholds for production-grade INT8 models (per your spec):
  - Max Abs Error  <= 2 LSB   (good)
  - RMSE (float)   < 0.01     (good)
  - RMSE (int8)    < 0.5      (good)
  - Top-1 Match    == 100%    (good; >99.5% acceptable on large datasets)

Usage:
  python3 verify_accuracy.py                         # all models, random inputs
  python3 verify_accuracy.py --model mnist_cnn       # single model
  python3 verify_accuracy.py --n-inputs 50           # more test inputs
  python3 verify_accuracy.py --image path/to/img.jpg # real image input
  python3 verify_accuracy.py --imagenet-dir /path/to/ILSVRC2012_img_val/
  python3 verify_accuracy.py --vww-dir /path/to/vww-minival/
  python3 verify_accuracy.py --generate-host         # re-run te-comp with target=generic

Requirements:
  - Python 3 with tensorflow (or tflite-runtime) and numpy
  - gcc (host compiler)
  - te-comp output in benchmarks/output/<model>/  (or use --generate-host)
  - .tflite models in benchmarks/models/
"""

import argparse
import ctypes
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

# Prefer tflite_runtime (lighter, avoids protobuf conflicts with tf)
try:
    import tflite_runtime.interpreter as tflri
    TFLiteInterpreter = tflri.Interpreter
except ImportError:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


# ============================================================================
# Model name mapping (tflite filename -> output dir name)
# ============================================================================

def sanitize_name(filename):
    """Convert tflite filename to te-comp output directory name."""
    name = os.path.splitext(filename)[0]
    # Strip numeric prefix
    parts = name.split("_", 1)
    if parts[0].isdigit() and len(parts) > 1:
        name = parts[1]
    # Replace hyphens and dots with underscores
    name = name.replace("-", "_").replace(".", "_")
    return name


def find_tflite_for_model(model_name):
    """Find the .tflite file for a given model output directory name."""
    for f in os.listdir(MODELS_DIR):
        if not f.endswith(".tflite"):
            continue
        if sanitize_name(f) == model_name:
            return os.path.join(MODELS_DIR, f)
    return None


def count_tflite_ops(tflite_path):
    """Count the number of operators in a TFLite model's subgraph 0.

    Uses the 'tflite' flatbuffer schema package if available, otherwise
    returns None.
    """
    try:
        from tflite.Model import Model as TFLiteModel
    except ImportError:
        return None
    with open(tflite_path, "rb") as f:
        buf = f.read()
    model = TFLiteModel.GetRootAs(buf, 0)
    sg = model.Subgraphs(0)
    return sg.OperatorsLength() if sg is not None else None


# ============================================================================
# TFLite interpreter runner
# ============================================================================

class TFLiteRunner:
    def __init__(self, tflite_path):
        self.interp = TFLiteInterpreter(model_path=tflite_path)
        self.interp.allocate_tensors()
        self.input_details = self.interp.get_input_details()[0]
        self.output_details = self.interp.get_output_details()[0]
        self.input_shape = tuple(self.input_details["shape"])
        self.output_shape = tuple(self.output_details["shape"])
        self.input_dtype = self.input_details["dtype"]
        self.output_dtype = self.output_details["dtype"]

        # Get quantization params for preprocessing
        quant = self.input_details.get("quantization_parameters", {})
        scales = quant.get("scales", [])
        zps = quant.get("zero_points", [])
        self.input_scale = scales[0] if len(scales) > 0 else 1.0
        self.input_zp = zps[0] if len(zps) > 0 else 0

        # Output quantization params (for dequantized float RMSE)
        out_quant = self.output_details.get("quantization_parameters", {})
        out_scales = out_quant.get("scales", [])
        out_zps = out_quant.get("zero_points", [])
        self.output_scale = out_scales[0] if len(out_scales) > 0 else 1.0
        self.output_zp = out_zps[0] if len(out_zps) > 0 else 0

    def run(self, input_data):
        """Run inference. input_data is int8 numpy array of correct shape."""
        self.interp.set_tensor(self.input_details["index"], input_data)
        self.interp.invoke()
        return self.interp.get_tensor(self.output_details["index"]).copy()


# ============================================================================
# Host-compiled te-comp engine runner
# ============================================================================

class HostEngineRunner:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.engine_c = os.path.join(output_dir, f"{model_name}_engine.c")
        self.engine_h = os.path.join(output_dir, f"{model_name}_engine.h")
        self.manifest_path = os.path.join(output_dir, "manifest.json")

        if not os.path.exists(self.engine_c):
            raise FileNotFoundError(f"Engine C file not found: {self.engine_c}")

        # Read manifest for I/O sizes
        with open(self.manifest_path) as f:
            manifest = json.load(f)
        self.input_shape = tuple(manifest["input"]["shape"])
        self.output_shape = tuple(manifest["output"]["shape"])
        self.input_size = manifest["input"]["size_bytes"]
        self.output_size = manifest["output"]["size_bytes"]

        # Build shared library for host
        self.lib_path = None
        self.lib = None
        self._build()

    def _build(self):
        """Compile engine.c into a shared library for host execution."""
        self.tmpdir = tempfile.mkdtemp(prefix=f"te_verify_{self.model_name}_")
        self.lib_path = os.path.join(self.tmpdir, f"lib{self.model_name}.so")

        # Create a wrapper that exports the engine functions
        wrapper_c = os.path.join(self.tmpdir, "wrapper.c")
        with open(wrapper_c, "w") as f:
            f.write(f'#include "{self.model_name}_engine.h"\n')
            f.write(f'/* Re-export engine functions for dlopen */\n')

        engine_dir = os.path.dirname(self.engine_c)

        cmd = [
            "gcc", "-shared", "-fPIC", "-O2",
            "-I", engine_dir,
            "-o", self.lib_path,
            self.engine_c, wrapper_c,
            "-lm",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Compile error: {result.stderr}", file=sys.stderr)
            raise RuntimeError(f"Failed to compile {self.engine_c}")

        # Load the shared library
        self.lib = ctypes.CDLL(self.lib_path)

        # Set up function signatures
        init_fn = getattr(self.lib, f"{self.model_name}_engine_init")
        init_fn.restype = None
        init_fn.argtypes = []

        run_fn = getattr(self.lib, f"{self.model_name}_engine_run")
        run_fn.restype = ctypes.c_int
        run_fn.argtypes = []

        get_input_fn = getattr(self.lib, f"{self.model_name}_engine_get_input")
        get_input_fn.restype = ctypes.POINTER(ctypes.c_int8)
        get_input_fn.argtypes = []

        get_output_fn = getattr(self.lib, f"{self.model_name}_engine_get_output")
        get_output_fn.restype = ctypes.POINTER(ctypes.c_int8)
        get_output_fn.argtypes = []

        self._init = init_fn
        self._run = run_fn
        self._get_input = get_input_fn
        self._get_output = get_output_fn

        # Initialize engine
        self._init()

    def run(self, input_data):
        """Run inference. input_data is int8 numpy array (flat or shaped)."""
        flat = np.ascontiguousarray(input_data.flatten(), dtype=np.int8)
        assert len(flat) == self.input_size, \
            f"Input size mismatch: got {len(flat)}, expected {self.input_size}"

        # Copy input into engine's arena.  Keep `flat` alive across the
        # memmove — ctypes.data is just a raw pointer.
        inp_ptr = self._get_input()
        src = flat.ctypes.data
        ctypes.memmove(inp_ptr, src, self.input_size)

        # Run inference
        self._run()

        # Read output from engine's arena
        out_ptr = self._get_output()
        output = np.ctypeslib.as_array(out_ptr, shape=(self.output_size,)).copy()
        return output.reshape(self.output_shape)

    def cleanup(self):
        """Remove temporary files."""
        if self.tmpdir and os.path.exists(self.tmpdir):
            import shutil
            shutil.rmtree(self.tmpdir, ignore_errors=True)


# ============================================================================
# Input generators
# ============================================================================

def generate_random_inputs(shape, n, seed=42):
    """Generate N random int8 inputs."""
    rng = np.random.RandomState(seed)
    inputs = []
    for i in range(n):
        data = rng.randint(-128, 128, size=shape, dtype=np.int8)
        inputs.append(data)
    return inputs


def load_image_as_input(image_path, input_shape, input_scale, input_zp):
    """Load an image file and preprocess to int8 tensor.

    Standard ImageNet/VWW preprocessing:
      1. Resize to (H+14, W+14)  [for ImageNet; skip for VWW]
      2. Center crop to (H, W)
      3. Convert to float [0, 1]
      4. Quantize to int8 using input_scale and input_zp
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    # Target spatial dims (NHWC format)
    if len(input_shape) == 4:
        _, h, w, c = input_shape
    elif len(input_shape) == 3:
        h, w, c = input_shape
    else:
        raise ValueError(f"Unexpected input shape: {input_shape}")

    if c == 1:
        img = img.convert("L")

    # Resize with a bit of slack then center-crop
    resize_h = h + 14 if h >= 64 else h
    resize_w = w + 14 if w >= 64 else w
    img = img.resize((resize_w, resize_h), Image.BILINEAR)

    # Center crop
    left = (resize_w - w) // 2
    top = (resize_h - h) // 2
    img = img.crop((left, top, left + w, top + h))

    # Convert to float [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # Quantize: int8 = round(float_val / scale) + zero_point
    if input_scale > 0:
        quantized = np.clip(
            np.round(arr / input_scale) + input_zp,
            -128, 127
        ).astype(np.int8)
    else:
        quantized = (arr * 255 - 128).astype(np.int8)

    if c == 1 and len(quantized.shape) == 2:
        quantized = quantized[..., np.newaxis]

    return quantized.reshape(input_shape)


def load_images_from_dir(image_dir, input_shape, input_scale, input_zp, n=10):
    """Load N images from a directory."""
    from PIL import Image

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPEG"}
    image_files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1] in exts
    ])[:n]

    if not image_files:
        print(f"  WARNING: No images found in {image_dir}", file=sys.stderr)
        return []

    inputs = []
    paths = []
    for img_path in image_files:
        try:
            data = load_image_as_input(img_path, input_shape, input_scale, input_zp)
            inputs.append(data)
            paths.append(img_path)
        except Exception as e:
            print(f"  WARNING: Failed to load {img_path}: {e}", file=sys.stderr)

    return inputs, paths


# ============================================================================
# Verification
# ============================================================================

def verify_model(model_name, tflite_path, output_dir, inputs,
                 input_paths=None, ground_truth_labels=None,
                 verbose=False, atol=0, max_error_rate=0.0):
    """Compare TFLite interpreter vs te-comp engine on given inputs.

    Args:
        atol: Maximum absolute difference per element to consider a match.
              0 means byte-exact; 1 means ±1 is tolerated.
        max_error_rate: Maximum fraction of elements that may differ (within
              atol). 0.0 means all elements must match; 0.05 means up to 5%
              may differ by <= atol.
        ground_truth_labels: Optional list/array of ground-truth class labels,
              one per input. -1 means label unavailable for that input.

    Returns dict with metrics:
        n_exact, n_not_exact, mismatches,
        max_abs_error, rmse_int8, rmse_float, top1_match_rate,
        element_error_rate, output_scale, output_zp,
        tflite_label_correct, engine_label_correct, n_labeled
    """
    n_ops = count_tflite_ops(tflite_path)
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  TFLite: {os.path.basename(tflite_path)}")
    print(f"  Engine: {output_dir}")
    if n_ops is not None:
        print(f"  Graph:  {n_ops} operators")
    print(f"  Inputs: {len(inputs)}")
    print(f"{'='*60}")

    # Initialize runners
    tflite_runner = TFLiteRunner(tflite_path)
    print(f"  TFLite input:  {tflite_runner.input_shape} {tflite_runner.input_dtype}")
    print(f"  TFLite output: {tflite_runner.output_shape} {tflite_runner.output_dtype}")
    print(f"  Output quant:  scale={tflite_runner.output_scale:.6f}, "
          f"zp={tflite_runner.output_zp}")

    try:
        engine_runner = HostEngineRunner(model_name, output_dir)
    except Exception as e:
        print(f"  SKIP: Failed to build engine: {e}")
        return {"skipped": True, "error": str(e)}

    print(f"  Engine input:  {engine_runner.input_shape} (int8)")
    print(f"  Engine output: {engine_runner.output_shape} (int8)")
    print()

    n_exact = 0
    n_not_exact = 0
    mismatches = []

    # Accumulators for aggregate metrics
    all_sq_diff_int8 = []      # squared int8 diffs (for RMSE int8)
    all_sq_diff_float = []     # squared dequantized float diffs (for RMSE float)
    global_max_abs_error = 0   # worst-case LSB deviation
    top1_matches = 0           # count of argmax agreements
    total_differing_elems = 0  # elements that differ by > 0
    total_elems = 0            # total output elements across all inputs

    # Ground-truth label accuracy tracking
    tflite_label_correct = 0   # TFLite argmax matches ground truth
    engine_label_correct = 0   # te-comp argmax matches ground truth
    n_labeled = 0              # inputs that have valid ground-truth labels

    out_scale = tflite_runner.output_scale
    out_zp = tflite_runner.output_zp

    for i, input_data in enumerate(inputs):
        label = input_paths[i] if input_paths and i < len(input_paths) else f"random_{i}"
        label = os.path.basename(label) if "/" in label else label

        # Run TFLite
        tflite_out = tflite_runner.run(input_data)

        # Run te-comp engine
        engine_out = engine_runner.run(input_data)

        # Compare outputs (int8)
        tflite_flat = tflite_out.flatten().astype(np.int8)
        engine_flat = engine_out.flatten().astype(np.int8)

        diff_i16 = tflite_flat.astype(np.int16) - engine_flat.astype(np.int16)
        abs_diff = np.abs(diff_i16)
        max_diff = int(np.max(abs_diff)) if len(abs_diff) > 0 else 0
        diff_indices = np.where(abs_diff > 0)[0]
        n_total = len(tflite_flat)
        error_rate = len(diff_indices) / n_total if n_total > 0 else 0.0
        exceeds_atol = np.where(abs_diff > atol)[0]

        # Aggregate metrics
        sq_int8 = (diff_i16.astype(np.float64) ** 2)
        all_sq_diff_int8.append(sq_int8)

        # Dequantized float: float_val = (int8_val - zero_point) * scale
        tflite_float = (tflite_flat.astype(np.float64) - out_zp) * out_scale
        engine_float = (engine_flat.astype(np.float64) - out_zp) * out_scale
        sq_float = (tflite_float - engine_float) ** 2
        all_sq_diff_float.append(sq_float)

        global_max_abs_error = max(global_max_abs_error, max_diff)
        total_differing_elems += len(diff_indices)
        total_elems += n_total

        # Top-1 classification match
        tflite_class = int(np.argmax(tflite_flat))
        engine_class = int(np.argmax(engine_flat))
        same_class = (tflite_class == engine_class)
        if same_class:
            top1_matches += 1

        # Ground-truth label check
        gt_label = -1
        if ground_truth_labels is not None and i < len(ground_truth_labels):
            gt_label = int(ground_truth_labels[i])
        if gt_label >= 0:
            n_labeled += 1
            if tflite_class == gt_label:
                tflite_label_correct += 1
            if engine_class == gt_label:
                engine_label_correct += 1

        # Confidence gap (for pass/fail logic)
        sorted_vals = np.sort(tflite_flat.astype(np.int16))
        tflite_top1 = int(sorted_vals[-1])
        tflite_top2 = int(sorted_vals[-2]) if n_total > 1 else tflite_top1
        confidence_gap = tflite_top1 - tflite_top2
        confident = confidence_gap > 2 * max_diff

        # Per-input RMSE (for verbose output)
        input_rmse_int8 = float(np.sqrt(np.mean(sq_int8)))
        input_rmse_float = float(np.sqrt(np.mean(sq_float)))

        # Pass criteria:
        #   1. max_diff <= atol  (per-element bound)
        #   2. error_rate <= max_error_rate  (fraction of differing elements)
        #   3. classification agrees OR model is not confident
        is_pass = (max_diff <= atol
                   and error_rate <= max_error_rate
                   and (same_class or not confident))

        class_tag = ("same" if same_class
                     else "DIFFERENT" if confident
                     else "different (low confidence)")

        # Ground-truth tag for display
        gt_tag = ""
        if gt_label >= 0:
            tfl_ok = "Y" if tflite_class == gt_label else "N"
            eng_ok = "Y" if engine_class == gt_label else "N"
            gt_tag = f" gt={gt_label} tfl={tfl_ok} eng={eng_ok}"

        if is_pass:
            n_exact += 1
            if verbose:
                print(f"  [{i:3d}] EXACT      "
                      f"class={tflite_class}, "
                      f"max_diff={max_diff}, "
                      f"rmse_i8={input_rmse_int8:.3f}, "
                      f"rmse_f={input_rmse_float:.6f}, "
                      f"err_rate={error_rate:.1%}{gt_tag}  ({label})")
            else:
                print(f"  [{i:3d}] EXACT      "
                      f"class={tflite_class}, "
                      f"max_diff={max_diff}, "
                      f"err_rate={error_rate:.1%}, "
                      f"gap={confidence_gap}{gt_tag}  ({label})")
        else:
            n_not_exact += 1

            reasons = []
            if not same_class and confident:
                reasons.append(f"class DIFFERENT (tflite={tflite_class}, "
                               f"engine={engine_class}, gap={confidence_gap})")
            if max_diff > atol:
                reasons.append(f"max_diff={max_diff} > atol={atol}")
            if error_rate > max_error_rate:
                reasons.append(f"err_rate={error_rate:.1%} > "
                               f"{max_error_rate:.1%}")

            mismatch_info = {
                "input": label,
                "n_diff_bytes": len(diff_indices),
                "n_exceed_atol": len(exceeds_atol),
                "max_diff": max_diff,
                "error_rate": error_rate,
                "rmse_int8": input_rmse_int8,
                "rmse_float": input_rmse_float,
                "confidence_gap": confidence_gap,
                "tflite_class": tflite_class,
                "engine_class": engine_class,
                "same_class": same_class,
            }
            mismatches.append(mismatch_info)

            print(f"  [{i:3d}] NOT EXACT  "
                  f"diff={len(diff_indices)}/{n_total}, "
                  f"max_diff={max_diff}, "
                  f"rmse_i8={input_rmse_int8:.3f}, "
                  f"rmse_f={input_rmse_float:.6f}, "
                  f"class: {class_tag} "
                  f"(tflite={tflite_class}, engine={engine_class}, "
                  f"gap={confidence_gap}){gt_tag}  ({label})")
            if reasons:
                print(f"         reason: {'; '.join(reasons)}")

            if verbose and len(diff_indices) <= 20:
                for idx in diff_indices[:20]:
                    print(f"       byte[{idx}]: tflite={tflite_flat[idx]}, "
                          f"engine={engine_flat[idx]}, "
                          f"diff={int(tflite_flat[idx])-int(engine_flat[idx])}")

    engine_runner.cleanup()

    # Compute aggregate metrics
    n_inputs = len(inputs)
    if n_inputs > 0 and total_elems > 0:
        all_sq_int8 = np.concatenate(all_sq_diff_int8)
        all_sq_flt = np.concatenate(all_sq_diff_float)
        rmse_int8 = float(np.sqrt(np.mean(all_sq_int8)))
        rmse_float = float(np.sqrt(np.mean(all_sq_flt)))
        top1_match_rate = top1_matches / n_inputs
        elem_error_rate = total_differing_elems / total_elems
    else:
        rmse_int8 = 0.0
        rmse_float = 0.0
        top1_match_rate = 1.0
        elem_error_rate = 0.0

    # Print per-model summary
    status = "EXACT" if n_not_exact == 0 else "NOT EXACT"
    print(f"\n  Result: {status}  "
          f"({n_exact} exact, {n_not_exact} not exact out of "
          f"{n_inputs} inputs)")
    print(f"  --- Aggregate Metrics ---")
    print(f"  Max Abs Error (LSB):   {global_max_abs_error}")
    print(f"  RMSE (int8):           {rmse_int8:.4f}")
    print(f"  RMSE (float):          {rmse_float:.8f}")
    print(f"  Top-1 Match Rate:      {top1_match_rate:.1%}  "
          f"({top1_matches}/{n_inputs})")
    print(f"  Element Error Rate:    {elem_error_rate:.4%}  "
          f"({total_differing_elems}/{total_elems})")

    # Ground-truth label accuracy
    if n_labeled > 0:
        tflite_label_acc = tflite_label_correct / n_labeled
        engine_label_acc = engine_label_correct / n_labeled
        print(f"  --- Ground-Truth Label Accuracy ({n_labeled} labeled inputs) ---")
        print(f"  TFLite vs Label:       {tflite_label_acc:.1%}  "
              f"({tflite_label_correct}/{n_labeled})")
        print(f"  te-comp vs Label:      {engine_label_acc:.1%}  "
              f"({engine_label_correct}/{n_labeled})")
    else:
        tflite_label_acc = -1.0
        engine_label_acc = -1.0

    # Quality assessment
    quality = []
    if global_max_abs_error == 0:
        quality.append("BIT-EXACT")
    elif global_max_abs_error <= 2:
        quality.append("max_err<=2 LSB (good)")
    else:
        quality.append(f"max_err={global_max_abs_error} LSB (INVESTIGATE)")
    if rmse_float < 0.01:
        quality.append("rmse_float<0.01 (good)")
    if rmse_int8 < 0.5:
        quality.append("rmse_int8<0.5 (good)")
    if top1_match_rate == 1.0:
        quality.append("top1=100% (good)")
    elif top1_match_rate >= 0.995:
        quality.append(f"top1={top1_match_rate:.1%} (acceptable)")
    else:
        quality.append(f"top1={top1_match_rate:.1%} (INVESTIGATE)")
    print(f"  Quality:               {'; '.join(quality)}")

    # Output classes = last dimension of output shape (e.g. 1000 for ImageNet)
    output_classes = tflite_runner.output_shape[-1]

    return {
        "n_exact": n_exact,
        "n_not_exact": n_not_exact,
        "n_inputs": n_inputs,
        "n_ops": n_ops,
        "output_classes": output_classes,
        "mismatches": mismatches,
        "max_abs_error": global_max_abs_error,
        "rmse_int8": rmse_int8,
        "rmse_float": rmse_float,
        "top1_match_rate": top1_match_rate,
        "element_error_rate": elem_error_rate,
        "output_scale": out_scale,
        "output_zp": out_zp,
        "tflite_label_correct": tflite_label_correct,
        "engine_label_correct": engine_label_correct,
        "n_labeled": n_labeled,
    }


# ============================================================================
# Main
# ============================================================================

# ============================================================================
# Host code generation
# ============================================================================

def generate_host_output(model_name, tflite_path, output_dir):
    """Re-run te-comp with target=generic to produce host-compilable C."""
    te_comp = os.path.join(SCRIPT_DIR, "tools", "te-comp")
    if not os.path.exists(te_comp):
        raise FileNotFoundError(f"te-comp not found at {te_comp}")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        te_comp,
        "--convert-tosa-to-te",
        "--te-flash-weights",
        "--te-in-place-depthwise",
        "--te-operator-specialize",
        "--te-memory-plan",
        "--convert-te-to-loops=target=generic",
        "--convert-generic-to-llvm",
        "--emit-c",
        "--rtos", "baremetal",
        tflite_path,
        "-o", output_dir,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  te-comp error: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"te-comp failed for {model_name}")
    print(f"  Generated host C for {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify te-comp generated C against TFLite interpreter. "
                    "Reports RMSE, max absolute error, and top-1 match rate."
    )
    parser.add_argument("--model", default=None,
                        help="Single model name (e.g. mnist_cnn, mcunet_10fps_vww)")
    parser.add_argument("--n-inputs", type=int, default=10,
                        help="Number of random test inputs (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible inputs")
    parser.add_argument("--image", default=None,
                        help="Path to a single test image")
    parser.add_argument("--imagenet-dir", default=None,
                        help="Path to ImageNet validation images")
    parser.add_argument("--vww-dir", default=None,
                        help="Path to VWW minival images")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-input RMSE and detailed metrics")
    parser.add_argument("--json", default=None,
                        help="Write results to JSON file")
    parser.add_argument("--atol", type=int, default=0,
                        help="Max absolute diff per element to count as match "
                             "(default: 0 = bit-exact)")
    parser.add_argument("--max-error-rate", type=float, default=0.0,
                        help="Max fraction of elements that may differ "
                             "(default: 0.0; use 0.05 for 5%%)")
    parser.add_argument("--generate-host", action="store_true",
                        help="Re-run te-comp with target=generic before "
                             "verification (for host-only testing)")
    parser.add_argument("--host-output-dir", default=None,
                        help="Output directory for --generate-host "
                             "(default: benchmarks/output_host/<model>)")
    args = parser.parse_args()

    # Where to put host-generated code
    host_output_base = args.host_output_dir or os.path.join(
        SCRIPT_DIR, "output_host")

    # Discover models
    if args.model:
        model_names = [args.model]
    else:
        # All CNN models (skip LLM models — different I/O pattern)
        model_names = []
        src_dir = host_output_base if args.generate_host else OUTPUT_DIR
        # Discover from existing outputs or from model files
        if args.generate_host:
            # Discover from tflite files
            for f in sorted(os.listdir(MODELS_DIR)):
                if not f.endswith(".tflite"):
                    continue
                model_names.append(sanitize_name(f))
        else:
            for d in sorted(os.listdir(OUTPUT_DIR)):
                manifest_path = os.path.join(OUTPUT_DIR, d, "manifest.json")
                if not os.path.exists(manifest_path):
                    continue
                with open(manifest_path) as f:
                    manifest = json.load(f)
                # Skip transformer/LLM models (output shape has vocab dim)
                out_shape = manifest.get("output", {}).get("shape", [])
                if len(out_shape) >= 3 and out_shape[-1] > 2000:
                    continue
                model_names.append(d)

    print(f"Verifying {len(model_names)} models: {', '.join(model_names)}")
    print(f"Random inputs per model: {args.n_inputs}")
    print(f"Random seed: {args.seed}")
    if args.atol > 0 or args.max_error_rate > 0:
        print(f"Tolerance: atol={args.atol}, max_error_rate={args.max_error_rate:.1%}")
    if args.generate_host:
        print(f"Re-generating with target=generic into: {host_output_base}")

    # Run verification
    all_results = {}
    total_exact = 0
    total_not_exact = 0

    for model_name in model_names:
        tflite_path = find_tflite_for_model(model_name)

        if tflite_path is None:
            print(f"\n  SKIP {model_name}: no matching .tflite file found")
            continue

        # Determine output directory
        if args.generate_host:
            output_dir = os.path.join(host_output_base, model_name)
            try:
                generate_host_output(model_name, tflite_path, output_dir)
            except Exception as e:
                print(f"\n  SKIP {model_name}: generation failed: {e}")
                continue
        else:
            output_dir = os.path.join(OUTPUT_DIR, model_name)

        if not os.path.exists(os.path.join(output_dir, f"{model_name}_engine.c")):
            print(f"\n  SKIP {model_name}: no engine.c found in {output_dir}")
            continue

        # Load manifest for input shape
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        input_shape = tuple(manifest["input"]["shape"])

        # Get TFLite quantization params
        tflite_runner = TFLiteRunner(tflite_path)
        input_scale = tflite_runner.input_scale
        input_zp = tflite_runner.input_zp

        # Build input list
        inputs = []
        input_paths = None

        # Check for real images
        is_imagenet = "imagenet" in model_name or model_name in [
            "mbv2_w0_35_r144_imagenet", "proxyless_w0_3_r176_imagenet"
        ]
        is_vww = "vww" in model_name

        if args.image:
            try:
                img_input = load_image_as_input(
                    args.image, input_shape, input_scale, input_zp)
                inputs = [img_input]
                input_paths = [args.image]
            except Exception as e:
                print(f"  WARNING: Failed to load image: {e}")

        if is_imagenet and args.imagenet_dir:
            try:
                imgs, paths = load_images_from_dir(
                    args.imagenet_dir, input_shape, input_scale, input_zp,
                    n=args.n_inputs)
                inputs.extend(imgs)
                input_paths = (input_paths or []) + paths
            except Exception as e:
                print(f"  WARNING: Failed to load ImageNet images: {e}")

        if is_vww and args.vww_dir:
            for subdir in ["person", "non_person"]:
                vww_subdir = os.path.join(args.vww_dir, subdir)
                if os.path.isdir(vww_subdir):
                    try:
                        imgs, paths = load_images_from_dir(
                            vww_subdir, input_shape, input_scale, input_zp,
                            n=args.n_inputs // 2)
                        inputs.extend(imgs)
                        input_paths = (input_paths or []) + paths
                    except Exception as e:
                        print(f"  WARNING: Failed to load VWW images: {e}")

        # Load pre-saved test samples + labels if available
        samples_dir = os.path.join(SCRIPT_DIR, "test_samples")
        sample_file = os.path.join(samples_dir, f"{model_name}.npy")
        label_file = os.path.join(samples_dir, f"{model_name}_labels.npy")
        ground_truth_labels = None

        if os.path.exists(sample_file):
            saved_samples = np.load(sample_file)
            n_prior = len(inputs)
            for idx, s in enumerate(saved_samples):
                inputs.append(s.reshape(input_shape))
            # Track display paths for saved samples
            saved_paths = [f"sample_{idx}" for idx in range(len(saved_samples))]
            if input_paths is None:
                input_paths = [None] * n_prior + saved_paths
            else:
                input_paths.extend(saved_paths)
            if os.path.exists(label_file):
                gt_labels = np.load(label_file)
                # Labels apply to the saved samples only; pad with -1 for
                # any earlier real-image inputs, then append -1 for random
                ground_truth_labels = (
                    [-1] * n_prior + gt_labels.tolist()
                )
            print(f"  Loaded {len(saved_samples)} saved test samples "
                  f"from {sample_file}")

        # Always add random inputs (no ground truth for random)
        random_inputs = generate_random_inputs(input_shape, args.n_inputs,
                                                seed=args.seed)
        random_paths = [f"random_{idx}" for idx in range(len(random_inputs))]
        if input_paths is None:
            input_paths = [None] * len(inputs) + random_paths
        else:
            input_paths.extend(random_paths)
        inputs.extend(random_inputs)
        if ground_truth_labels is not None:
            ground_truth_labels.extend([-1] * len(random_inputs))

        result = verify_model(
            model_name, tflite_path, output_dir, inputs,
            input_paths=input_paths,
            ground_truth_labels=ground_truth_labels,
            verbose=args.verbose,
            atol=args.atol, max_error_rate=args.max_error_rate)

        if result.get("skipped"):
            continue

        total_exact += result["n_exact"]
        total_not_exact += result["n_not_exact"]
        all_results[model_name] = result

    # ====================================================================
    # Summary table
    # ====================================================================
    print(f"\n{'='*100}")
    print(f"  VERIFICATION SUMMARY (Paper Table 3)")
    print(f"{'='*100}")
    print(f"  Models tested:  {len(all_results)}")
    print(f"  Total inputs:   {total_exact + total_not_exact}")
    print(f"  Total exact:    {total_exact}")
    print(f"  Total not exact:{total_not_exact}")
    if args.atol > 0 or args.max_error_rate > 0:
        print(f"  Tolerance:      atol={args.atol}, "
              f"max_error_rate={args.max_error_rate:.1%}")
    print()

    # Paper Table 3: Model, Out, Tens., Max LSB, RMSE, Elem. err, TFL->GT, Eng->GT
    has_labels = any(r.get("n_labeled", 0) > 0 for r in all_results.values())
    hdr = (f"  {'Model':<30s} {'Out':>5s} {'Tens.':>6s} "
           f"{'Max LSB':>8s} {'RMSE':>7s} {'Elem.err':>9s}")
    if has_labels:
        hdr += f" {'TFL->GT':>8s} {'Eng->GT':>8s}"
    print(hdr)
    sep = (f"  {'-'*30} {'-'*5} {'-'*6} {'-'*8} {'-'*7} {'-'*9}")
    if has_labels:
        sep += f" {'-'*8} {'-'*8}"
    print(sep)

    for name in sorted(all_results.keys()):
        r = all_results[name]
        out_cls = f"{r['output_classes']}" if r.get("output_classes") else "?"
        ops_str = f"{r['n_ops']}" if r.get("n_ops") is not None else "?"
        err_pct = f"{r['element_error_rate']:.1%}"
        rmse_str = f"{r['rmse_int8']:.3f}"
        line = (f"  {name:<30s} "
                f"{out_cls:>5s} "
                f"{ops_str:>6s} "
                f"{r['max_abs_error']:>8d} "
                f"{rmse_str:>7s} "
                f"{err_pct:>9s}")
        if has_labels:
            nl = r.get("n_labeled", 0)
            if nl > 0:
                tfl_acc = f"{r['tflite_label_correct']}/{nl}"
                eng_acc = f"{r['engine_label_correct']}/{nl}"
            else:
                tfl_acc = "N/A"
                eng_acc = "N/A"
            line += f" {tfl_acc:>8s} {eng_acc:>8s}"
        print(line)

    print()

    # Quality thresholds
    all_bit_exact = all(r["max_abs_error"] == 0 for r in all_results.values())
    all_good_lsb = all(r["max_abs_error"] <= 2 for r in all_results.values())
    all_good_rmse_f = all(r["rmse_float"] < 0.01 for r in all_results.values())
    all_good_rmse_i = all(r["rmse_int8"] < 0.5 for r in all_results.values())
    all_top1 = all(r["top1_match_rate"] == 1.0 for r in all_results.values())

    if all_bit_exact:
        print(f"  ALL MODELS PRODUCE BIT-EXACT OUTPUT vs TFLite interpreter.")
    elif total_not_exact == 0:
        print(f"  All inputs exact (atol={args.atol}, "
              f"max_error_rate={args.max_error_rate:.1%}).")

    print(f"\n  Quality Assessment (production thresholds for INT8):")
    print(f"    Max Abs Error <= 2 LSB:    {'EXACT' if all_good_lsb else 'NOT EXACT'}")
    print(f"    RMSE (float)  <  0.01:     {'EXACT' if all_good_rmse_f else 'NOT EXACT'}")
    print(f"    RMSE (int8)   <  0.5:      {'EXACT' if all_good_rmse_i else 'NOT EXACT'}")
    print(f"    Top-1 Match   == 100%:     {'EXACT' if all_top1 else 'NOT EXACT'}")

    if all_bit_exact:
        print(f"\n  Verdict: BIT-EXACT — te-comp produces identical output "
              f"to TFLite reference.")
    elif all_good_lsb and all_good_rmse_f and all_top1:
        print(f"\n  Verdict: PRODUCTION-READY — within acceptable INT8 "
              f"rounding tolerances.")
    else:
        print(f"\n  Verdict: INVESTIGATE — some metrics exceed thresholds.")

    # Write JSON report
    if args.json:
        json_report = {
            "summary": {
                "models_tested": len(all_results),
                "total_inputs": total_exact + total_not_exact,
                "total_exact": total_exact,
                "total_not_exact": total_not_exact,
                "bit_exact": all_bit_exact,
                "quality": {
                    "max_abs_error_le_2": all_good_lsb,
                    "rmse_float_lt_001": all_good_rmse_f,
                    "rmse_int8_lt_05": all_good_rmse_i,
                    "top1_match_100pct": all_top1,
                },
            },
            "models": {},
        }
        for name, r in sorted(all_results.items()):
            model_report = {
                "exact": r["n_exact"],
                "not_exact": r["n_not_exact"],
                "total": r["n_inputs"],
                "n_ops": r.get("n_ops"),
                "output_classes": r.get("output_classes"),
                "max_abs_error": r["max_abs_error"],
                "rmse_int8": r["rmse_int8"],
                "rmse_float": r["rmse_float"],
                "top1_match_rate": r["top1_match_rate"],
                "element_error_rate": r["element_error_rate"],
                "output_scale": r["output_scale"],
                "output_zp": r["output_zp"],
                "mismatches": r["mismatches"],
            }
            if r.get("n_labeled", 0) > 0:
                model_report["n_labeled"] = r["n_labeled"]
                model_report["tflite_label_correct"] = r["tflite_label_correct"]
                model_report["engine_label_correct"] = r["engine_label_correct"]
            json_report["models"][name] = model_report
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(args.json, "w") as f:
            json.dump(json_report, f, indent=2, cls=NumpyEncoder)
        print(f"\n  Report written to: {args.json}")

    return 0 if total_not_exact == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
