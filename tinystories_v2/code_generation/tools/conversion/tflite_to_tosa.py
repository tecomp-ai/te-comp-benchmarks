#!/usr/bin/env python3
"""
tflite_to_tosa.py — Convert a TFLite model to TOSA v1.0 MLIR.

This tool reads a .tflite FlatBuffer file, walks the graph, and emits
MLIR in the TOSA dialect (compatible with LLVM 22 / TOSA v1.0).

Supported operators:
  CONV_2D          → tosa.conv2d
  DEPTHWISE_CONV_2D → tosa.depthwise_conv2d
  FULLY_CONNECTED  → tosa.matmul  (+ reshape)
  AVERAGE_POOL_2D  → tosa.avg_pool2d
  MAX_POOL_2D      → tosa.max_pool2d
  ADD              → tosa.add
  MUL              → tosa.mul
  RESHAPE          → tosa.reshape
  SOFTMAX          → tosa.rescale (approximation)
  LOGISTIC         → tosa.sigmoid
  RELU / RELU6     → tosa.clamp
  MEAN             → tosa.avg_pool2d (spatial axes)
  PAD              → tosa.pad
  SHAPE            → tosa.const (shape as i32 tensor)
  STRIDED_SLICE    → tosa.slice (stride-1 only)
  PACK             → tosa.concat (with reshape)

Quantization:
  The converter reads per-tensor / per-channel quantization parameters
  from the TFLite model and emits the corresponding TOSA v1.0 zero-point
  tensor operands and rescale multiplier/shift values.

Usage:
  python3 tflite_to_tosa.py model.tflite -o model.mlir
  python3 tflite_to_tosa.py model.tflite          # prints to stdout
"""

import argparse
import math
import struct
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- TFLite FlatBuffer bindings ---
from tflite.Model import Model
from tflite.TensorType import TensorType
from tflite.BuiltinOperator import BuiltinOperator
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.Padding import Padding
from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.FullyConnectedOptions import FullyConnectedOptions
from tflite.Pool2DOptions import Pool2DOptions
from tflite.ReshapeOptions import ReshapeOptions
from tflite.SoftmaxOptions import SoftmaxOptions
from tflite.AddOptions import AddOptions


# ============================================================================
# Type mapping
# ============================================================================

TFLITE_TYPE_MAP = {
    TensorType.FLOAT32: ("f32", 4),
    TensorType.FLOAT16: ("f16", 2),
    TensorType.INT32:   ("i32", 4),
    TensorType.INT16:   ("i16", 2),
    TensorType.INT8:    ("i8",  1),
    TensorType.UINT8:   ("ui8", 1),
    TensorType.BOOL:    ("i1",  1),
    TensorType.INT64:   ("i64", 8),
}


def mlir_type(tfl_type: int) -> str:
    """Convert TFLite type enum to MLIR type string."""
    if tfl_type in TFLITE_TYPE_MAP:
        return TFLITE_TYPE_MAP[tfl_type][0]
    return f"/*unknown_type={tfl_type}*/ i8"


def tensor_type_str(shape: List[int], elem_type: str) -> str:
    """Format tensor<AxBxCxTy>.  Scalar (0-d) → tensor<Ty>."""
    if not shape:
        return f"tensor<{elem_type}>"
    dims = "x".join(str(d) for d in shape)
    return f"tensor<{dims}x{elem_type}>"


# ============================================================================
# TFLite model reader
# ============================================================================

class TFLiteTensor:
    """Parsed tensor metadata from a TFLite model."""
    def __init__(self, idx, name, shape, dtype, quant_params, buffer_data):
        self.idx = idx
        self.name = name
        self.shape = list(shape)
        self.dtype = dtype                   # TFLite enum
        self.mlir_type = mlir_type(dtype)
        self.quant = quant_params            # (scale, zero_point, quantized_dimension)
        self.buffer_data = buffer_data       # numpy array or None
        self.mlir_name = None                # assigned during emission

    @property
    def type_str(self) -> str:
        return tensor_type_str(self.shape, self.mlir_type)

    @property
    def is_quantized(self) -> bool:
        return self.quant is not None and self.quant[0] is not None

    @property
    def zero_point(self) -> int:
        if self.quant and self.quant[1] is not None:
            zps = self.quant[1]
            return zps[0] if len(zps) == 1 else 0
        return 0


class TFLiteModelReader:
    """Read and parse a .tflite FlatBuffer model."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            buf = f.read()
        self.model = Model.GetRootAs(buf, 0)
        self.subgraph = self.model.Subgraphs(0)
        self.tensors: List[TFLiteTensor] = []
        self._parse_tensors()

    def _parse_tensors(self):
        for i in range(self.subgraph.TensorsLength()):
            t = self.subgraph.Tensors(i)
            name = t.Name().decode("utf-8") if t.Name() else f"tensor_{i}"
            shape = [t.Shape(j) for j in range(t.ShapeLength())]
            dtype = t.Type()

            # Quantization
            quant_params = None
            qp = t.Quantization()
            if qp and qp.ScaleLength() > 0:
                scales = [qp.Scale(j) for j in range(qp.ScaleLength())]
                zps = [qp.ZeroPoint(j) for j in range(qp.ZeroPointLength())]
                qdim = qp.QuantizedDimension()
                quant_params = (scales, zps, qdim)

            # Buffer data
            buffer_data = None
            buf_idx = t.Buffer()
            if buf_idx > 0 and buf_idx < self.model.BuffersLength():
                buf_obj = self.model.Buffers(buf_idx)
                if buf_obj.DataLength() > 0:
                    raw = buf_obj.DataAsNumpy()
                    np_type = {
                        TensorType.FLOAT32: np.float32,
                        TensorType.INT32: np.int32,
                        TensorType.INT16: np.int16,
                        TensorType.INT8: np.int8,
                        TensorType.UINT8: np.uint8,
                    }.get(dtype, np.int8)
                    buffer_data = np.frombuffer(raw, dtype=np_type).reshape(shape)

            self.tensors.append(TFLiteTensor(i, name, shape, dtype,
                                             quant_params, buffer_data))

    def inputs(self) -> List[int]:
        return [self.subgraph.Inputs(i)
                for i in range(self.subgraph.InputsLength())]

    def outputs(self) -> List[int]:
        return [self.subgraph.Outputs(i)
                for i in range(self.subgraph.OutputsLength())]

    def operators(self):
        """Yield (op_code_enum, op) for each operator."""
        for i in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(i)
            opcode_idx = op.OpcodeIndex()
            opcode_obj = self.model.OperatorCodes(opcode_idx)
            # DeprecatedBuiltinCode is used for codes < 127
            code = opcode_obj.DeprecatedBuiltinCode()
            if code == BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES:
                code = opcode_obj.BuiltinCode()
            yield code, op


# ============================================================================
# Quantization helpers
# ============================================================================

def compute_rescale_multiplier_shift(input_scale, weight_scales, output_scale):
    """
    Compute the fixed-point multiplier and shift for requantisation.

    effective_scale = input_scale * weight_scale / output_scale
    Find M, S such that effective_scale ≈ M * 2^(-S)  with M in [2^30, 2^31).
    Negative weight scales (rare but valid in TFLite per-channel quantisation)
    produce a negative multiplier.

    Matches TF/TOSA's computeMultiplierAndShiftTosaScale32: uses math.frexp
    for exact decomposition and caps total_shift at 62 (the maximum that
    RoundingDivideByPOT can handle with 32-bit arithmetic).
    """
    import math
    multipliers = []
    shifts = []
    for ws in weight_scales:
        raw_effective = input_scale * ws / output_scale
        if raw_effective == 0:
            multipliers.append(0)
            shifts.append(0)
            continue
        # frexp: raw_effective = mantissa * 2^exp, mantissa in [-1,-0.5)∪[0.5,1)
        mantissa, exp = math.frexp(raw_effective)
        shifted_m = round(mantissa * (1 << 31))
        if abs(shifted_m) == (1 << 31):
            shifted_m //= 2
            exp += 1
        # TOSA convention: total_shift is a right shift
        total_shift = (-exp) + 31
        mult = int(shifted_m)
        # Cap at 62 — the maximum RoundingDivideByPOT handles in 32-bit.
        # Right-shift the multiplier to compensate (matches TF/TOSA exactly).
        if total_shift > 62:
            mult = mult >> min(31, total_shift - 62)
            total_shift = 62
        multipliers.append(mult)
        shifts.append(total_shift)
    return multipliers, shifts


# ============================================================================
# MLIR emitter
# ============================================================================

class TOSAEmitter:
    """Emit TOSA v1.0 MLIR from a parsed TFLite model."""

    def __init__(self, reader: TFLiteModelReader):
        self.reader = reader
        self.lines: List[str] = []
        self.ssa_counter = 0
        self.tensor_to_ssa: Dict[int, str] = {}  # tensor idx -> SSA name
        self.indent = "    "

    def _ssa(self) -> str:
        name = f"%v{self.ssa_counter}"
        self.ssa_counter += 1
        return name

    def _emit(self, line: str):
        self.lines.append(self.indent + line)

    def _emit_const_tensor(self, tensor: TFLiteTensor) -> str:
        """Emit a tosa.const for a weight/bias tensor and return its SSA name."""
        if tensor.idx in self.tensor_to_ssa:
            return self.tensor_to_ssa[tensor.idx]

        name = self._ssa()
        self.tensor_to_ssa[tensor.idx] = name

        if tensor.buffer_data is not None:
            data_str = self._format_dense_attr(
                tensor.buffer_data, tensor.shape, tensor.mlir_type)
            self._emit(f'{name} = "tosa.const"() {{values = {data_str}'
                       f"}} : () -> {tensor.type_str}")
        else:
            self._emit(f'{name} = "tosa.const"() {{values = dense<0> '
                       f": {tensor.type_str}}} : () -> {tensor.type_str}")

        return name

    # Threshold for switching from nested literal to hex blob format.
    # Small tensors use readable nested syntax; large tensors use compact hex.
    _HEX_BLOB_THRESHOLD = 256

    def _format_dense_attr(self, np_arr, shape, elem_type) -> str:
        """Format a dense attribute, using hex blob for large tensors."""
        flat = np_arr.flatten()
        num_elems = flat.size
        if num_elems == 0:
            return f"dense<0> : {tensor_type_str(shape, elem_type)}"
        # Check for splat (all same value) — compact regardless of size
        first = flat[0]
        if np.all(flat == first):
            if first == 0:
                return f"dense<0> : {tensor_type_str(shape, elem_type)}"
            val = self._format_val(first.item(), elem_type)
            return f"dense<{val}> : {tensor_type_str(shape, elem_type)}"
        # Large tensors: emit as hex blob
        if num_elems > self._HEX_BLOB_THRESHOLD:
            return self._format_dense_hex_blob(np_arr, shape, elem_type)
        # Small tensors: emit as nested literal for readability
        return self._format_dense_nd(np_arr, shape, elem_type)

    def _format_dense_hex_blob(self, np_arr, shape, elem_type) -> str:
        """Format a dense attribute as MLIR hex blob: dense<"0xABCD...">."""
        # Get raw bytes in native (little-endian) order, matching MLIR's
        # DenseElementsAttr hex blob format.
        raw_bytes = np_arr.flatten().tobytes()
        hex_str = raw_bytes.hex()
        return f'dense<"0x{hex_str}"> : {tensor_type_str(shape, elem_type)}'

    def _format_dense_nd(self, np_arr, shape, elem_type) -> str:
        """Format a dense attribute with proper nested array structure."""
        flat = np_arr.flatten().tolist()
        if all(v == 0 for v in flat):
            return f"dense<0> : {tensor_type_str(shape, elem_type)}"
        if all(v == flat[0] for v in flat):
            val = self._format_val(flat[0], elem_type)
            return f"dense<{val}> : {tensor_type_str(shape, elem_type)}"
        # Reshape to match the declared shape for proper nesting
        shaped_arr = np_arr.reshape(shape) if np_arr.shape != tuple(shape) else np_arr
        nested = self._nest_array(shaped_arr, elem_type)
        return f"dense<{nested}> : {tensor_type_str(shape, elem_type)}"

    def _nest_array(self, arr, elem_type) -> str:
        """Recursively format a numpy array as nested MLIR dense literal."""
        if arr.ndim == 0:
            return self._format_val(arr.item(), elem_type)
        if arr.ndim == 1:
            vals = ", ".join(self._format_val(v, elem_type)
                            for v in arr.tolist())
            return f"[{vals}]"
        inner = ", ".join(self._nest_array(arr[i], elem_type)
                          for i in range(arr.shape[0]))
        return f"[{inner}]"

    def _format_val(self, v, elem_type) -> str:
        if "f" in elem_type:
            return f"{v:.6e}"
        return str(int(v))

    def _emit_scalar_const(self, value: int, elem_type: str) -> str:
        """Emit a scalar const tensor (for zero points)."""
        name = self._ssa()
        ty = f"tensor<1x{elem_type}>"
        self._emit(f'{name} = "tosa.const"() {{values = dense<{value}> '
                   f": {ty}}} : () -> {ty}")
        return name

    def _get_padding(self, padding_enum, input_h, input_w, filter_h, filter_w,
                     stride_h, stride_w, dilation_h=1, dilation_w=1,
                     output_h=None, output_w=None):
        """Compute padding as [top, bottom, left, right].

        When output_h/output_w are provided (from TFLite tensor shapes),
        compute exact padding to match that output size.

        TOSA requires: (input + pad_top + pad_bottom - eff_filter) % stride == 0
        and output = (input + pad_top + pad_bottom - eff_filter) / stride + 1.
        TFLite is more lenient (allows truncation), so we may need to add
        padding even for VALID to satisfy TOSA's divisibility constraint.
        """
        eff_fh = (filter_h - 1) * dilation_h + 1
        eff_fw = (filter_w - 1) * dilation_w + 1

        if padding_enum == Padding.VALID:
            # TOSA needs (input + pad - eff_filter) % stride == 0
            # For VALID, we start with pad=0 and add if needed
            if output_h is not None and output_w is not None:
                # Compute exact padding from known output shape
                # output = (input + pad - eff_filter) / stride + 1
                # pad = (output - 1) * stride + eff_filter - input
                pad_h = max(0, (output_h - 1) * stride_h + eff_fh - input_h)
                pad_w = max(0, (output_w - 1) * stride_w + eff_fw - input_w)
            else:
                # Check if divisibility holds without padding
                remain_h = (input_h - eff_fh) % stride_h
                remain_w = (input_w - eff_fw) % stride_w
                pad_h = (stride_h - remain_h) % stride_h
                pad_w = (stride_w - remain_w) % stride_w
            return [0, pad_h, 0, pad_w]

        # SAME padding — compute from known output shape
        if output_h is None:
            output_h = -(-input_h // stride_h)  # ceil division
        if output_w is None:
            output_w = -(-input_w // stride_w)
        pad_h = max(0, (output_h - 1) * stride_h + eff_fh - input_h)
        pad_w = max(0, (output_w - 1) * stride_w + eff_fw - input_w)
        return [pad_h // 2, pad_h - pad_h // 2,
                pad_w // 2, pad_w - pad_w // 2]

    def _emit_activation(self, ssa_name: str, tensor: TFLiteTensor,
                         act_fn: int) -> str:
        """Emit fused activation function, return new SSA name."""
        if act_fn == ActivationFunctionType.NONE:
            return ssa_name

        if tensor.is_quantized and tensor.mlir_type == "i8":
            zp = tensor.zero_point
            if act_fn == ActivationFunctionType.RELU:
                min_val = max(-128, zp)  # real 0.0
                max_val = 127
            elif act_fn == ActivationFunctionType.RELU6:
                min_val = max(-128, zp)
                # relu6: clamp at 6.0 in quantized domain
                scale = tensor.quant[0][0] if tensor.quant else 1.0
                max_val = min(127, zp + int(round(6.0 / scale)))
            elif act_fn == ActivationFunctionType.RELU_N1_TO_1:
                scale = tensor.quant[0][0] if tensor.quant else 1.0
                min_val = max(-128, zp + int(round(-1.0 / scale)))
                max_val = min(127, zp + int(round(1.0 / scale)))
            else:
                return ssa_name

            result = self._ssa()
            ty = tensor.type_str
            self._emit(f"{result} = tosa.clamp {ssa_name} "
                       f"{{min_val = {min_val} : i8, max_val = {max_val} : i8}}"
                       f" : ({ty}) -> {ty}")
            self.tensor_to_ssa[tensor.idx] = result
            return result
        else:
            # Float activations
            if act_fn == ActivationFunctionType.RELU:
                result = self._ssa()
                ty = tensor.type_str
                self._emit(f"{result} = tosa.clamp {ssa_name} "
                           f"{{min_val = 0.0 : f32, "
                           f"max_val = 3.40282347E+38 : f32}}"
                           f" : ({ty}) -> {ty}")
                self.tensor_to_ssa[tensor.idx] = result
                return result
            elif act_fn == ActivationFunctionType.RELU6:
                result = self._ssa()
                ty = tensor.type_str
                self._emit(f"{result} = tosa.clamp {ssa_name} "
                           f"{{min_val = 0.0 : f32, max_val = 6.0 : f32}}"
                           f" : ({ty}) -> {ty}")
                self.tensor_to_ssa[tensor.idx] = result
                return result
        return ssa_name

    def _broadcast_to_rank(self, ssa: str, shape: List[int],
                           type_str: str, target_rank: int,
                           elem_type: str) -> Tuple[str, str]:
        """Pad shape with leading 1s to match target_rank.  Returns (new_ssa, new_type_str)."""
        if len(shape) >= target_rank:
            return ssa, type_str
        new_shape = [1] * (target_rank - len(shape)) + shape
        ndims = len(new_shape)
        shape_ssa = self._ssa()
        shape_vals = ", ".join(str(d) for d in new_shape)
        self._emit(f'{shape_ssa} = "tosa.const_shape"() '
                   f'{{values = dense<[{shape_vals}]> '
                   f': tensor<{ndims}xindex>}} : () '
                   f'-> !tosa.shape<{ndims}>')
        new_ssa = self._ssa()
        new_type = tensor_type_str(new_shape, elem_type)
        self._emit(f"{new_ssa} = tosa.reshape {ssa}, {shape_ssa} "
                   f": ({type_str}, !tosa.shape<{ndims}>) -> {new_type}")
        return new_ssa, new_type

    # ------------------------------------------------------------------
    # Operator emitters
    # ------------------------------------------------------------------

    def _emit_conv2d(self, op):
        """CONV_2D → tosa.conv2d"""
        opts = Conv2DOptions()
        opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

        input_idx = op.Inputs(0)
        weight_idx = op.Inputs(1)
        bias_idx = op.Inputs(2) if op.InputsLength() > 2 else None
        output_idx = op.Outputs(0)

        inp = self.reader.tensors[input_idx]
        wt = self.reader.tensors[weight_idx]
        out = self.reader.tensors[output_idx]

        inp_ssa = self.tensor_to_ssa[input_idx]
        wt_ssa = self._emit_const_tensor(wt)

        # TFLite Conv2D weights: [OC, KH, KW, IC] — same as TOSA
        # But TFLite uses NHWC for activations, which TOSA also expects

        # Bias
        if bias_idx is not None and bias_idx >= 0:
            bias = self.reader.tensors[bias_idx]
            bias_ssa = self._emit_const_tensor(bias)
        else:
            # Create zero bias
            oc = wt.shape[0]
            bias_type = "i32" if inp.mlir_type == "i8" else "f32"
            bias_ssa = self._ssa()
            bty = f"tensor<{oc}x{bias_type}>"
            self._emit(f'{bias_ssa} = "tosa.const"() {{values = dense<0> '
                       f": {bty}}} : () -> {bty}")

        stride_h, stride_w = opts.StrideH(), opts.StrideW()
        dilation_h, dilation_w = opts.DilationHFactor(), opts.DilationWFactor()
        pad = self._get_padding(opts.Padding(),
                                inp.shape[1], inp.shape[2],
                                wt.shape[1], wt.shape[2],
                                stride_h, stride_w,
                                dilation_h, dilation_w,
                                output_h=out.shape[1], output_w=out.shape[2])

        result = self._ssa()

        if inp.mlir_type == "i8":
            # Quantized conv2d
            inp_zp = inp.zero_point
            wt_zp = wt.zero_point
            izp_ssa = self._emit_scalar_const(inp_zp, "i8")
            wzp_ssa = self._emit_scalar_const(wt_zp, "i8")

            # Accumulator type is i32
            acc_shape = [out.shape[0], out.shape[1], out.shape[2], out.shape[3]]
            acc_type = tensor_type_str(acc_shape, "i32")

            self._emit(
                f"{result} = tosa.conv2d {inp_ssa}, {wt_ssa}, {bias_ssa}, "
                f"{izp_ssa}, {wzp_ssa} {{"
                f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                f"stride = array<i64: {stride_h}, {stride_w}>, "
                f"dilation = array<i64: {dilation_h}, {dilation_w}>, "
                f"acc_type = i32"
                f"}} : ({inp.type_str}, {wt.type_str}, "
                f"{self.reader.tensors[bias_idx].type_str if bias_idx else bty}, "
                f"tensor<1xi8>, tensor<1xi8>) -> {acc_type}")

            # Emit rescale i32 -> i8
            rescale_result = self._emit_rescale(
                result, acc_shape, "i32",
                inp.quant[0][0] if inp.is_quantized else 1.0,
                wt.quant[0] if wt.is_quantized else [1.0],
                out.quant[0][0] if out.is_quantized else 1.0,
                out.zero_point,
                len(wt.quant[0]) > 1 if wt.is_quantized else False,
                out.type_str
            )
            self.tensor_to_ssa[output_idx] = rescale_result
            # Handle fused activation
            self._emit_activation(rescale_result, out, opts.FusedActivationFunction())
        else:
            # Float conv2d — no zp operands needed in float, but TOSA v1.0
            # still requires them as tensor<1xf32> with value 0.0
            izp_ssa = self._emit_scalar_const(0, "f32")
            wzp_ssa = self._emit_scalar_const(0, "f32")

            self._emit(
                f"{result} = tosa.conv2d {inp_ssa}, {wt_ssa}, {bias_ssa}, "
                f"{izp_ssa}, {wzp_ssa} {{"
                f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                f"stride = array<i64: {stride_h}, {stride_w}>, "
                f"dilation = array<i64: {dilation_h}, {dilation_w}>, "
                f"acc_type = f32"
                f"}} : ({inp.type_str}, {wt.type_str}, "
                f"{self.reader.tensors[bias_idx].type_str if bias_idx else bty}, "
                f"tensor<1xf32>, tensor<1xf32>) -> {out.type_str}")
            self.tensor_to_ssa[output_idx] = result
            self._emit_activation(result, out, opts.FusedActivationFunction())

    def _emit_depthwise_conv2d(self, op):
        """DEPTHWISE_CONV_2D → tosa.depthwise_conv2d"""
        opts = DepthwiseConv2DOptions()
        opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

        input_idx = op.Inputs(0)
        weight_idx = op.Inputs(1)
        bias_idx = op.Inputs(2) if op.InputsLength() > 2 else None
        output_idx = op.Outputs(0)

        inp = self.reader.tensors[input_idx]
        wt = self.reader.tensors[weight_idx]
        out = self.reader.tensors[output_idx]

        inp_ssa = self.tensor_to_ssa[input_idx]

        # TFLite DW conv weights: [1, KH, KW, IC*M] (or [1, KH, KW, OC])
        # TOSA expects:           [KH, KW, IC, M]
        # Reshape from TFLite to TOSA layout
        depth_multiplier = opts.DepthMultiplier() if hasattr(opts, 'DepthMultiplier') else 1
        ic = inp.shape[3]  # input channels
        if wt.buffer_data is not None:
            # Transpose weight data: [1, KH, KW, OC] → [KH, KW, IC, M]
            wt_data = wt.buffer_data.reshape(wt.shape)
            wt_data = wt_data[0]  # Remove batch dim: [KH, KW, OC]
            kh, kw = wt_data.shape[0], wt_data.shape[1]
            wt_data = wt_data.reshape(kh, kw, ic, depth_multiplier)
            # Emit transposed weight as a fresh constant (don't modify the
            # original tensor or cache it under the original idx).
            tosa_wt_shape = [kh, kw, ic, depth_multiplier]
            tosa_wt_type = tensor_type_str(tosa_wt_shape, wt.mlir_type)
            wt_ssa = self._ssa()
            data_str = self._format_dense_attr(
                wt_data, tosa_wt_shape, wt.mlir_type)
            self._emit(f'{wt_ssa} = "tosa.const"() {{values = {data_str}'
                       f"}} : () -> {tosa_wt_type}")
            wt_type_str = tosa_wt_type
        else:
            wt_ssa = self._emit_const_tensor(wt)
            wt_type_str = wt.type_str

        if bias_idx is not None and bias_idx >= 0:
            bias_ssa = self._emit_const_tensor(self.reader.tensors[bias_idx])
            bty = self.reader.tensors[bias_idx].type_str
        else:
            oc = out.shape[3]
            bias_type = "i32" if inp.mlir_type == "i8" else "f32"
            bias_ssa = self._ssa()
            bty = f"tensor<{oc}x{bias_type}>"
            self._emit(f'{bias_ssa} = "tosa.const"() {{values = dense<0> '
                       f": {bty}}} : () -> {bty}")

        stride_h, stride_w = opts.StrideH(), opts.StrideW()
        dilation_h, dilation_w = opts.DilationHFactor(), opts.DilationWFactor()

        # Kernel size: TFLite weight is [1, KH, KW, OC], so KH=wt.shape[1], KW=wt.shape[2]
        kernel_h, kernel_w = wt.shape[1], wt.shape[2]
        pad = self._get_padding(opts.Padding(),
                                inp.shape[1], inp.shape[2],
                                kernel_h, kernel_w,
                                stride_h, stride_w,
                                dilation_h, dilation_w,
                                output_h=out.shape[1], output_w=out.shape[2])

        result = self._ssa()

        if inp.mlir_type == "i8":
            izp_ssa = self._emit_scalar_const(inp.zero_point, "i8")
            wzp_ssa = self._emit_scalar_const(wt.zero_point, "i8")
            acc_type = tensor_type_str(out.shape, "i32")

            self._emit(
                f"{result} = tosa.depthwise_conv2d {inp_ssa}, {wt_ssa}, "
                f"{bias_ssa}, {izp_ssa}, {wzp_ssa} {{"
                f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                f"stride = array<i64: {stride_h}, {stride_w}>, "
                f"dilation = array<i64: {dilation_h}, {dilation_w}>, "
                f"acc_type = i32"
                f"}} : ({inp.type_str}, {wt_type_str}, {bty}, "
                f"tensor<1xi8>, tensor<1xi8>) -> {acc_type}")

            rescale_result = self._emit_rescale(
                result, out.shape, "i32",
                inp.quant[0][0] if inp.is_quantized else 1.0,
                wt.quant[0] if wt.is_quantized else [1.0],
                out.quant[0][0] if out.is_quantized else 1.0,
                out.zero_point,
                len(wt.quant[0]) > 1 if wt.is_quantized else False,
                out.type_str
            )
            self.tensor_to_ssa[output_idx] = rescale_result
            self._emit_activation(rescale_result, out,
                                  opts.FusedActivationFunction())
        else:
            izp_ssa = self._emit_scalar_const(0, "f32")
            wzp_ssa = self._emit_scalar_const(0, "f32")
            self._emit(
                f"{result} = tosa.depthwise_conv2d {inp_ssa}, {wt_ssa}, "
                f"{bias_ssa}, {izp_ssa}, {wzp_ssa} {{"
                f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                f"stride = array<i64: {stride_h}, {stride_w}>, "
                f"dilation = array<i64: {dilation_h}, {dilation_w}>, "
                f"acc_type = f32"
                f"}} : ({inp.type_str}, {wt_type_str}, {bty}, "
                f"tensor<1xf32>, tensor<1xf32>) -> {out.type_str}")
            self.tensor_to_ssa[output_idx] = result
            self._emit_activation(result, out, opts.FusedActivationFunction())

    def _emit_fully_connected(self, op):
        """FULLY_CONNECTED → tosa.matmul (TOSA v1.0 has no fully_connected)."""
        opts = FullyConnectedOptions()
        opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

        input_idx = op.Inputs(0)
        weight_idx = op.Inputs(1)
        bias_idx = op.Inputs(2) if op.InputsLength() > 2 else None
        output_idx = op.Outputs(0)

        inp = self.reader.tensors[input_idx]
        wt = self.reader.tensors[weight_idx]
        out = self.reader.tensors[output_idx]

        inp_ssa = self.tensor_to_ssa[input_idx]
        wt_ssa = self._emit_const_tensor(wt)

        # TFLite FC: input [batch, input_size], weight [output_size, input_size]
        # TOSA matmul: A [batch, M, K] * B [batch, K, N] -> [batch, M, N]
        # So we reshape input to [1, 1, K] and transpose weight to [1, K, N]

        batch = 1
        K = wt.shape[1]  # input_size
        N = wt.shape[0]  # output_size

        # Reshape input to [1, 1, K] for matmul
        inp_3d_type = f"tensor<1x1x{K}x{inp.mlir_type}>"
        reshape_inp = self._ssa()
        target_shape_ssa = self._ssa()
        self._emit(f'{target_shape_ssa} = "tosa.const_shape"() '
                   f'{{values = dense<[1, 1, {K}]> : tensor<3xindex>}}'
                   f" : () -> !tosa.shape<3>")
        self._emit(f"{reshape_inp} = tosa.reshape {inp_ssa}, {target_shape_ssa}"
                   f" : ({inp.type_str}, !tosa.shape<3>) -> {inp_3d_type}")

        # Transpose weight [N, K] -> [K, N], then reshape to [1, K, N]
        wt_t_type = f"tensor<1x{K}x{N}x{wt.mlir_type}>"
        wt_3d = self._ssa()
        # Emit transposed weight as a const directly
        if wt.buffer_data is not None:
            # Transpose and reshape
            wt_data_2d = wt.buffer_data.reshape(N, K)
            wt_data_t = wt_data_2d.T  # [K, N]
            wt_data_3d = wt_data_t.reshape(1, K, N)
            data_str = self._format_dense_attr(
                wt_data_3d, [1, K, N], wt.mlir_type)
            self._emit(f'{wt_3d} = "tosa.const"() {{values = {data_str}'
                       f"}} : () -> {wt_t_type}")
        else:
            self._emit(f'{wt_3d} = "tosa.const"() {{values = dense<0> '
                       f": {wt_t_type}}} : () -> {wt_t_type}")

        result = self._ssa()
        if inp.mlir_type == "i8":
            azp_ssa = self._emit_scalar_const(inp.zero_point, "i8")
            bzp_ssa = self._emit_scalar_const(wt.zero_point, "i8")
            acc_type = f"tensor<1x1x{N}xi32>"

            self._emit(
                f"{result} = tosa.matmul {reshape_inp}, {wt_3d}, "
                f"{azp_ssa}, {bzp_ssa}"
                f" : ({inp_3d_type}, {wt_t_type}, "
                f"tensor<1xi8>, tensor<1xi8>) -> {acc_type}")

            # Add bias (if present) via tosa.add
            if bias_idx is not None and bias_idx >= 0:
                bias = self.reader.tensors[bias_idx]
                bias_ssa = self._emit_const_tensor(bias)
                # Reshape bias to [1, 1, N] for broadcast add
                bias_3d_type = f"tensor<1x1x{N}xi32>"
                bias_3d = self._ssa()
                bshape = self._ssa()
                self._emit(f'{bshape} = "tosa.const_shape"() '
                           f'{{values = dense<[1, 1, {N}]> : tensor<3xindex>}}'
                           f" : () -> !tosa.shape<3>")
                self._emit(f"{bias_3d} = tosa.reshape {bias_ssa}, {bshape}"
                           f" : ({bias.type_str}, !tosa.shape<3>) "
                           f"-> {bias_3d_type}")
                add_result = self._ssa()
                self._emit(f"{add_result} = tosa.add {result}, {bias_3d}"
                           f" : ({acc_type}, {bias_3d_type}) -> {acc_type}")
                result = add_result

            # Rescale i32 -> i8
            rescale_result = self._emit_rescale(
                result, [1, 1, N], "i32",
                inp.quant[0][0] if inp.is_quantized else 1.0,
                wt.quant[0] if wt.is_quantized else [1.0],
                out.quant[0][0] if out.is_quantized else 1.0,
                out.zero_point,
                False,  # FC is per-tensor
                f"tensor<1x1x{N}x{out.mlir_type}>"
            )

            # Reshape back to output shape
            out_ssa = self._ssa()
            oshape = self._ssa()
            out_dims = ", ".join(str(d) for d in out.shape)
            ndims = len(out.shape)
            self._emit(f'{oshape} = "tosa.const_shape"() '
                       f'{{values = dense<[{out_dims}]> : tensor<{ndims}xindex>}}'
                       f" : () -> !tosa.shape<{ndims}>")
            self._emit(f"{out_ssa} = tosa.reshape {rescale_result}, {oshape}"
                       f" : (tensor<1x1x{N}x{out.mlir_type}>, "
                       f"!tosa.shape<{ndims}>) -> {out.type_str}")
            self.tensor_to_ssa[output_idx] = out_ssa
            self._emit_activation(out_ssa, out, opts.FusedActivationFunction())
        else:
            # Float FC via matmul
            azp_ssa = self._emit_scalar_const(0, "f32")
            bzp_ssa = self._emit_scalar_const(0, "f32")
            out_3d_type = f"tensor<1x1x{N}xf32>"

            self._emit(
                f"{result} = tosa.matmul {reshape_inp}, {wt_3d}, "
                f"{azp_ssa}, {bzp_ssa}"
                f" : ({inp_3d_type}, {wt_t_type}, "
                f"tensor<1xf32>, tensor<1xf32>) -> {out_3d_type}")

            # Add bias
            if bias_idx is not None and bias_idx >= 0:
                bias = self.reader.tensors[bias_idx]
                bias_ssa = self._emit_const_tensor(bias)
                bias_3d_type = f"tensor<1x1x{N}xf32>"
                bias_3d = self._ssa()
                bshape = self._ssa()
                self._emit(f'{bshape} = "tosa.const_shape"() '
                           f'{{values = dense<[1, 1, {N}]> : tensor<3xindex>}}'
                           f" : () -> !tosa.shape<3>")
                self._emit(f"{bias_3d} = tosa.reshape {bias_ssa}, {bshape}"
                           f" : ({bias.type_str}, !tosa.shape<3>) "
                           f"-> {bias_3d_type}")
                add_result = self._ssa()
                self._emit(f"{add_result} = tosa.add {result}, {bias_3d}"
                           f" : ({out_3d_type}, {bias_3d_type}) -> {out_3d_type}")
                result = add_result

            # Reshape to output
            out_ssa = self._ssa()
            oshape = self._ssa()
            out_dims = ", ".join(str(d) for d in out.shape)
            ndims = len(out.shape)
            self._emit(f'{oshape} = "tosa.const_shape"() '
                       f'{{values = dense<[{out_dims}]> : tensor<{ndims}xindex>}}'
                       f" : () -> !tosa.shape<{ndims}>")
            self._emit(f"{out_ssa} = tosa.reshape {result}, {oshape}"
                       f" : ({out_3d_type}, !tosa.shape<{ndims}>) "
                       f"-> {out.type_str}")
            self.tensor_to_ssa[output_idx] = out_ssa
            self._emit_activation(out_ssa, out, opts.FusedActivationFunction())

    def _emit_pool2d(self, op, op_name):
        """AVERAGE_POOL_2D / MAX_POOL_2D → tosa.avg_pool2d / tosa.max_pool2d"""
        opts = Pool2DOptions()
        opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

        input_idx = op.Inputs(0)
        output_idx = op.Outputs(0)

        inp = self.reader.tensors[input_idx]
        out = self.reader.tensors[output_idx]

        inp_ssa = self.tensor_to_ssa[input_idx]

        fh, fw = opts.FilterHeight(), opts.FilterWidth()
        sh, sw = opts.StrideH(), opts.StrideW()

        # Compute TOSA-compatible padding.
        # TFLite allows non-divisible pooling (drops remainder), but TOSA
        # requires (input + pad - kernel) % stride == 0.
        if opts.Padding() == Padding.VALID:
            # For VALID: compute minimal bottom/right pad to satisfy divisibility
            remain_h = (inp.shape[1] - fh) % sh
            remain_w = (inp.shape[2] - fw) % sw
            pad_bottom = (sh - remain_h) % sh
            pad_right = (sw - remain_w) % sw
            pad = [0, pad_bottom, 0, pad_right]
            # Recompute output shape for TOSA
            tosa_out_h = (inp.shape[1] + pad_bottom - fh) // sh + 1
            tosa_out_w = (inp.shape[2] + pad_right - fw) // sw + 1
        else:
            pad = self._get_padding(opts.Padding(),
                                    inp.shape[1], inp.shape[2],
                                    fh, fw, sh, sw,
                                    output_h=out.shape[1],
                                    output_w=out.shape[2])
            tosa_out_h = out.shape[1]
            tosa_out_w = out.shape[2]

        # Use TOSA-valid output shape (may differ from TFLite for non-divisible cases)
        tosa_out_shape = [out.shape[0], tosa_out_h, tosa_out_w, out.shape[3]]
        tosa_out_type_str = tensor_type_str(tosa_out_shape, out.mlir_type)

        result = self._ssa()
        tosa_op = "tosa.avg_pool2d" if op_name == "avg" else "tosa.max_pool2d"

        if inp.mlir_type == "i8" and op_name == "avg":
            izp_ssa = self._emit_scalar_const(inp.zero_point, "i8")
            ozp_ssa = self._emit_scalar_const(out.zero_point, "i8")
            self._emit(
                f"{result} = {tosa_op} {inp_ssa}, {izp_ssa}, {ozp_ssa} {{"
                f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                f"kernel = array<i64: {fh}, {fw}>, "
                f"stride = array<i64: {sh}, {sw}>, "
                f"acc_type = i32"
                f"}} : ({inp.type_str}, tensor<1xi8>, tensor<1xi8>) "
                f"-> {tosa_out_type_str}")
        elif op_name == "avg":
            izp_ssa = self._emit_scalar_const(0, "f32")
            ozp_ssa = self._emit_scalar_const(0, "f32")
            self._emit(
                f"{result} = {tosa_op} {inp_ssa}, {izp_ssa}, {ozp_ssa} {{"
                f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                f"kernel = array<i64: {fh}, {fw}>, "
                f"stride = array<i64: {sh}, {sw}>, "
                f"acc_type = f32"
                f"}} : ({inp.type_str}, tensor<1xf32>, tensor<1xf32>) "
                f"-> {tosa_out_type_str}")
        else:
            # max_pool2d has no zp operands
            self._emit(
                f"{result} = {tosa_op} {inp_ssa} {{"
                f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                f"kernel = array<i64: {fh}, {fw}>, "
                f"stride = array<i64: {sh}, {sw}>"
                f"}} : ({inp.type_str}) -> {tosa_out_type_str}")

        # If TOSA output shape differs from TFLite, reshape to match
        if tosa_out_shape != list(out.shape):
            ndims = len(out.shape)
            shape_ssa = self._ssa()
            shape_vals = ", ".join(str(d) for d in out.shape)
            self._emit(f'{shape_ssa} = "tosa.const_shape"() '
                       f'{{values = dense<[{shape_vals}]> '
                       f': tensor<{ndims}xindex>}} : () '
                       f'-> !tosa.shape<{ndims}>')
            # Use tosa.slice to take the valid portion
            slice_result = self._ssa()
            start_vals = ", ".join("0" for _ in out.shape)
            size_vals = ", ".join(str(d) for d in out.shape)
            start_ssa = self._ssa()
            self._emit(f'{start_ssa} = "tosa.const_shape"() '
                       f'{{values = dense<[{start_vals}]> '
                       f': tensor<{ndims}xindex>}} : () '
                       f'-> !tosa.shape<{ndims}>')
            size_ssa = self._ssa()
            self._emit(f'{size_ssa} = "tosa.const_shape"() '
                       f'{{values = dense<[{size_vals}]> '
                       f': tensor<{ndims}xindex>}} : () '
                       f'-> !tosa.shape<{ndims}>')
            self._emit(f"{slice_result} = tosa.slice {result}, {start_ssa}, "
                       f"{size_ssa} : ({tosa_out_type_str}, "
                       f"!tosa.shape<{ndims}>, !tosa.shape<{ndims}>) "
                       f"-> {out.type_str}")
            result = slice_result

        self.tensor_to_ssa[output_idx] = result
        self._emit_activation(result, out, opts.FusedActivationFunction())

    def _emit_add(self, op):
        """ADD → tosa.add"""
        input0_idx = op.Inputs(0)
        input1_idx = op.Inputs(1)
        output_idx = op.Outputs(0)

        inp0 = self.reader.tensors[input0_idx]
        inp1 = self.reader.tensors[input1_idx]
        out = self.reader.tensors[output_idx]

        a_ssa = self.tensor_to_ssa.get(input0_idx)
        b_ssa = self.tensor_to_ssa.get(input1_idx)

        if a_ssa is None:
            a_ssa = self._emit_const_tensor(inp0)
        if b_ssa is None:
            b_ssa = self._emit_const_tensor(inp1)

        # TOSA v1.0 requires matching ranks for add operands
        a_shape = list(inp0.shape)
        b_shape = list(inp1.shape)
        a_type = inp0.type_str
        b_type = inp1.type_str
        target_rank = max(len(a_shape), len(b_shape))

        a_ssa, a_type = self._broadcast_to_rank(
            a_ssa, a_shape, a_type, target_rank, inp0.mlir_type)
        b_ssa, b_type = self._broadcast_to_rank(
            b_ssa, b_shape, b_type, target_rank, inp1.mlir_type)

        # Compute quantized add rescale parameters when i8.
        # Matches TFLite's algorithm: left_shift=20, then per-arm multiplier/shift,
        # sum, then output multiplier/shift.
        quant_attrs = ""
        if inp0.mlir_type == "i8" and inp0.is_quantized and inp1.is_quantized and out.is_quantized:
            lhs_scale = inp0.quant[0][0]
            rhs_scale = inp1.quant[0][0]
            out_scale = out.quant[0][0]
            lhs_zp = inp0.zero_point
            rhs_zp = inp1.zero_point
            out_zp = out.zero_point

            # TFLite algorithm: left_shift = 20
            left_shift = 20
            twice_max = 2.0 * max(lhs_scale, rhs_scale)
            lhs_eff = lhs_scale / twice_max  # in (0, 0.5]
            rhs_eff = rhs_scale / twice_max  # in (0, 0.5]
            out_eff = twice_max / ((1 << left_shift) * out_scale)

            lhs_m, lhs_s = compute_rescale_multiplier_shift(1.0, [lhs_eff], 1.0)
            rhs_m, rhs_s = compute_rescale_multiplier_shift(1.0, [rhs_eff], 1.0)
            out_m, out_s = compute_rescale_multiplier_shift(1.0, [out_eff], 1.0)

            quant_attrs = (
                f', "te.left_shift" = {left_shift} : i64'
                f', "te.lhs_multiplier" = {lhs_m[0]} : i64'
                f', "te.lhs_shift" = {lhs_s[0]} : i64'
                f', "te.lhs_zp" = {lhs_zp} : i64'
                f', "te.rhs_multiplier" = {rhs_m[0]} : i64'
                f', "te.rhs_shift" = {rhs_s[0]} : i64'
                f', "te.rhs_zp" = {rhs_zp} : i64'
                f', "te.output_multiplier" = {out_m[0]} : i64'
                f', "te.output_shift" = {out_s[0]} : i64'
                f', "te.output_zp" = {out_zp} : i64'
            )

        result = self._ssa()
        if quant_attrs:
            self._emit(f'{result} = tosa.add {a_ssa}, {b_ssa}'
                       f' {{{quant_attrs.lstrip(", ")}}}'
                       f' : ({a_type}, {b_type}) -> {out.type_str}')
        else:
            self._emit(f"{result} = tosa.add {a_ssa}, {b_ssa}"
                       f" : ({a_type}, {b_type}) -> {out.type_str}")
        self.tensor_to_ssa[output_idx] = result

        opts = AddOptions()
        opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
        self._emit_activation(result, out, opts.FusedActivationFunction())

    def _emit_reshape(self, op):
        """RESHAPE → tosa.reshape"""
        input_idx = op.Inputs(0)
        output_idx = op.Outputs(0)

        inp = self.reader.tensors[input_idx]
        out = self.reader.tensors[output_idx]

        inp_ssa = self.tensor_to_ssa[input_idx]
        result = self._ssa()
        shape_ssa = self._ssa()
        dims = ", ".join(str(d) for d in out.shape)
        ndims = len(out.shape)
        self._emit(f'{shape_ssa} = "tosa.const_shape"() '
                   f'{{values = dense<[{dims}]> : tensor<{ndims}xindex>}}'
                   f" : () -> !tosa.shape<{ndims}>")
        self._emit(f"{result} = tosa.reshape {inp_ssa}, {shape_ssa}"
                   f" : ({inp.type_str}, !tosa.shape<{ndims}>) -> {out.type_str}")
        self.tensor_to_ssa[output_idx] = result

    def _emit_softmax(self, op):
        """SOFTMAX → te.softmax (LUT-based quantized softmax)."""
        input_idx = op.Inputs(0)
        output_idx = op.Outputs(0)

        inp = self.reader.tensors[input_idx]
        out = self.reader.tensors[output_idx]

        inp_ssa = self.tensor_to_ssa[input_idx]

        if inp.mlir_type == "i8" and inp.is_quantized:
            # Compute beta * input_scale multiplier/shift.
            # TFLite softmax beta is 1.0 for standard softmax.
            beta = 1.0
            inp_scale = inp.quant[0][0]
            inp_zp = inp.quant[1][0]
            effective = beta * inp_scale
            multipliers, shifts = compute_rescale_multiplier_shift(
                1.0, [effective], 1.0)
            result = self._ssa()
            # Use hex float for exact scale representation.
            import struct
            scale_hex = struct.pack('>d', float(inp_scale)).hex()
            self._emit(
                f'{result} = te.softmax {inp_ssa} '
                f'{{beta_multiplier = {multipliers[0]} : i64, '
                f'beta_shift = {shifts[0]} : i64, '
                f'"te.input_scale" = {inp_scale:.15e} : f64, '
                f'"te.input_zp" = {inp_zp} : i64}} '
                f': ({inp.type_str}) -> {out.type_str}')
        else:
            # Float softmax: pass through (or implement later)
            self._emit(f"// SOFTMAX: float softmax not yet lowered")
            result = inp_ssa

        self.tensor_to_ssa[output_idx] = result

    def _emit_logistic(self, op):
        """LOGISTIC → tosa.sigmoid"""
        input_idx = op.Inputs(0)
        output_idx = op.Outputs(0)

        inp = self.reader.tensors[input_idx]
        out = self.reader.tensors[output_idx]

        inp_ssa = self.tensor_to_ssa[input_idx]
        result = self._ssa()
        self._emit(f"{result} = tosa.sigmoid {inp_ssa}"
                   f" : ({inp.type_str}) -> {out.type_str}")
        self.tensor_to_ssa[output_idx] = result

    def _emit_relu(self, op):
        """RELU → tosa.clamp"""
        input_idx = op.Inputs(0)
        output_idx = op.Outputs(0)
        out = self.reader.tensors[output_idx]

        inp_ssa = self.tensor_to_ssa[input_idx]
        result = self._ssa()
        if out.mlir_type == "i8":
            zp = out.zero_point
            self._emit(f"{result} = tosa.clamp {inp_ssa} "
                       f"{{min_val = {max(-128, zp)} : i8, max_val = 127 : i8}}"
                       f" : ({out.type_str}) -> {out.type_str}")
        else:
            self._emit(f"{result} = tosa.clamp {inp_ssa} "
                       f"{{min_val = 0.0 : f32, "
                       f"max_val = 3.40282347E+38 : f32}}"
                       f" : ({out.type_str}) -> {out.type_str}")
        self.tensor_to_ssa[output_idx] = result

    def _emit_relu6(self, op):
        """RELU6 → tosa.clamp"""
        input_idx = op.Inputs(0)
        output_idx = op.Outputs(0)
        out = self.reader.tensors[output_idx]

        inp_ssa = self.tensor_to_ssa[input_idx]
        result = self._ssa()
        if out.mlir_type == "i8":
            zp = out.zero_point
            scale = out.quant[0][0] if out.is_quantized else 1.0
            max_val = min(127, zp + int(round(6.0 / scale)))
            self._emit(f"{result} = tosa.clamp {inp_ssa} "
                       f"{{min_val = {max(-128, zp)} : i8, "
                       f"max_val = {max_val} : i8}}"
                       f" : ({out.type_str}) -> {out.type_str}")
        else:
            self._emit(f"{result} = tosa.clamp {inp_ssa} "
                       f"{{min_val = 0.0 : f32, max_val = 6.0 : f32}}"
                       f" : ({out.type_str}) -> {out.type_str}")
        self.tensor_to_ssa[output_idx] = result

    def _emit_mean(self, op):
        """MEAN (reduce_mean) → tosa.avg_pool2d for spatial dims,
        or tosa.reduce_sum + tosa.reciprocal for general case.

        In TFLite, MEAN over axes [1,2] on NHWC input is global average
        pooling. We convert this to avg_pool2d with kernel = input spatial dims.
        """
        input_idx = op.Inputs(0)
        output_idx = op.Outputs(0)

        inp = self.reader.tensors[input_idx]
        out = self.reader.tensors[output_idx]

        inp_ssa = self.tensor_to_ssa[input_idx]

        # Read axis tensor
        axis_idx = op.Inputs(1)
        axis_tensor = self.reader.tensors[axis_idx]
        if axis_tensor.buffer_data is not None:
            axes = list(axis_tensor.buffer_data.astype(int))
        else:
            # Default: spatial dims for NHWC (global average pooling)
            axes = [1, 2]

        result = self._ssa()

        # Spatial mean (axes=[1,2]) on 4D tensor → avg_pool2d
        if len(inp.shape) == 4 and sorted(axes) == [1, 2]:
            # Global average pooling: kernel = full spatial dims
            kh, kw = inp.shape[1], inp.shape[2]

            if inp.mlir_type == "i8":
                izp_ssa = self._emit_scalar_const(inp.zero_point, "i8")
                ozp_ssa = self._emit_scalar_const(out.zero_point, "i8")
                # Output shape: [N, 1, 1, C]
                pool_out_type = tensor_type_str(
                    [inp.shape[0], 1, 1, inp.shape[3]], "i8")

                # Compute fused rescale multiplier/shift when scales differ.
                # TFLite MEAN fuses the 1/count division with the scale change:
                #   output = requantize(sum, input_scale / (output_scale * count))
                # We pass these as custom attributes through TOSA → TE.
                inp_scale = inp.quant[0][0] if inp.is_quantized else 1.0
                out_scale = out.quant[0][0] if out.is_quantized else 1.0
                count = kh * kw
                fused_scale = inp_scale / (out_scale * count)
                multipliers, shifts = compute_rescale_multiplier_shift(
                    1.0, [fused_scale], 1.0)
                rescale_attrs = (
                    f', "te.rescale_multiplier" = {multipliers[0]} : i64, '
                    f'"te.rescale_shift" = {shifts[0]} : i64')

                self._emit(
                    f"{result} = tosa.avg_pool2d {inp_ssa}, {izp_ssa}, "
                    f"{ozp_ssa} {{"
                    f"pad = array<i64: 0, 0, 0, 0>, "
                    f"kernel = array<i64: {kh}, {kw}>, "
                    f"stride = array<i64: 1, 1>, "
                    f"acc_type = i32"
                    f"{rescale_attrs}"
                    f"}} : ({inp.type_str}, tensor<1xi8>, tensor<1xi8>) "
                    f"-> {pool_out_type}")
            else:
                izp_ssa = self._emit_scalar_const(0, "f32")
                ozp_ssa = self._emit_scalar_const(0, "f32")
                pool_out_type = tensor_type_str(
                    [inp.shape[0], 1, 1, inp.shape[3]], "f32")
                self._emit(
                    f"{result} = tosa.avg_pool2d {inp_ssa}, {izp_ssa}, "
                    f"{ozp_ssa} {{"
                    f"pad = array<i64: 0, 0, 0, 0>, "
                    f"kernel = array<i64: {kh}, {kw}>, "
                    f"stride = array<i64: 1, 1>, "
                    f"acc_type = f32"
                    f"}} : ({inp.type_str}, tensor<1xf32>, tensor<1xf32>) "
                    f"-> {pool_out_type}")

            # If output is 2D [N, C], reshape from [N, 1, 1, C]
            if len(out.shape) == 2:
                shape_ssa = self._ssa()
                self._emit(f'{shape_ssa} = "tosa.const_shape"() '
                           f'{{values = dense<[{out.shape[0]}, {out.shape[1]}]> '
                           f': tensor<2xindex>}} : () -> !tosa.shape<2>')
                reshape_result = self._ssa()
                self._emit(f"{reshape_result} = tosa.reshape {result}, "
                           f"{shape_ssa} : ({pool_out_type}, "
                           f"!tosa.shape<2>) -> {out.type_str}")
                self.tensor_to_ssa[output_idx] = reshape_result
            else:
                self.tensor_to_ssa[output_idx] = result
        else:
            # General case: pass through (unsupported axis combination)
            self._emit(f"// WARN: MEAN with axes={axes} not fully supported")
            self.tensor_to_ssa[output_idx] = inp_ssa

    def _emit_mul(self, op):
        """MUL → tosa.mul (3 operands: input1, input2, shift_tensor)"""
        input0_idx = op.Inputs(0)
        input1_idx = op.Inputs(1)
        output_idx = op.Outputs(0)

        out = self.reader.tensors[output_idx]

        a_ssa = self.tensor_to_ssa.get(input0_idx)
        b_ssa = self.tensor_to_ssa.get(input1_idx)

        if a_ssa is None:
            a_ssa = self._emit_const_tensor(self.reader.tensors[input0_idx])
        if b_ssa is None:
            b_ssa = self._emit_const_tensor(self.reader.tensors[input1_idx])

        inp0 = self.reader.tensors[input0_idx]
        inp1 = self.reader.tensors[input1_idx]

        # TOSA v1.0 requires matching ranks for mul operands.
        a_shape = list(inp0.shape)
        b_shape = list(inp1.shape)
        a_type = inp0.type_str
        b_type = inp1.type_str
        target_rank = max(len(a_shape), len(b_shape))

        a_ssa, a_type = self._broadcast_to_rank(
            a_ssa, a_shape, a_type, target_rank, inp0.mlir_type)
        b_ssa, b_type = self._broadcast_to_rank(
            b_ssa, b_shape, b_type, target_rank, inp1.mlir_type)

        # TOSA v1.0 mul takes a shift tensor (scalar i8) as third operand
        shift_ssa = self._emit_scalar_const(0, "i8")

        # Compute quantized mul rescale: eff = lhs_scale * rhs_scale / out_scale
        quant_attrs = ""
        if inp0.mlir_type == "i8" and inp0.is_quantized and inp1.is_quantized and out.is_quantized:
            lhs_scale = inp0.quant[0][0]
            rhs_scale = inp1.quant[0][0]
            out_scale = out.quant[0][0]
            lhs_zp = inp0.zero_point
            rhs_zp = inp1.zero_point
            out_zp = out.zero_point
            eff_scale = lhs_scale * rhs_scale / out_scale
            m, s = compute_rescale_multiplier_shift(1.0, [eff_scale], 1.0)
            quant_attrs = (
                f', "te.output_multiplier" = {m[0]} : i64'
                f', "te.output_shift" = {s[0]} : i64'
                f', "te.lhs_zp" = {lhs_zp} : i64'
                f', "te.rhs_zp" = {rhs_zp} : i64'
                f', "te.output_zp" = {out_zp} : i64'
            )

        result = self._ssa()
        if quant_attrs:
            self._emit(f'{result} = tosa.mul {a_ssa}, {b_ssa}, {shift_ssa} '
                       f'{{{quant_attrs.lstrip(", ")}}} '
                       f': ({a_type}, {b_type}, tensor<1xi8>) '
                       f'-> {out.type_str}')
        else:
            self._emit(f"{result} = tosa.mul {a_ssa}, {b_ssa}, {shift_ssa} "
                       f": ({a_type}, {b_type}, tensor<1xi8>) "
                       f"-> {out.type_str}")

        self.tensor_to_ssa[output_idx] = result

    def _emit_pad(self, op):
        """PAD → tosa.pad (TOSA v1.0: padding is !tosa.shape, pad_const is scalar tensor)"""
        input_idx = op.Inputs(0)
        output_idx = op.Outputs(0)

        inp = self.reader.tensors[input_idx]
        out = self.reader.tensors[output_idx]

        inp_ssa = self.tensor_to_ssa[input_idx]

        # Read TFLite padding tensor: [N, 2] where N = ndims
        pad_idx = op.Inputs(1)
        pad_tensor = self.reader.tensors[pad_idx]
        pad_data = pad_tensor.buffer_data.reshape(-1).astype(int)

        # TOSA v1.0: padding is !tosa.shape<2*ndims> with flattened
        # [before0, after0, before1, after1, ...]
        n_dims = len(inp.shape)
        n_pad_elems = 2 * n_dims
        pad_vals = ", ".join(str(v) for v in pad_data[:n_pad_elems])
        pad_ssa = self._ssa()
        self._emit(f'{pad_ssa} = "tosa.const_shape"() '
                   f'{{values = dense<[{pad_vals}]> '
                   f': tensor<{n_pad_elems}xindex>}} : () '
                   f'-> !tosa.shape<{n_pad_elems}>')

        # Pad constant value (scalar tensor)
        if inp.mlir_type == "i8":
            zp = inp.zero_point
            pad_val_ssa = self._ssa()
            self._emit(f'{pad_val_ssa} = "tosa.const"() '
                       f'{{values = dense<{zp}> : tensor<1xi8>}} '
                       f': () -> tensor<1xi8>')
        else:
            pad_val_ssa = self._ssa()
            self._emit(f'{pad_val_ssa} = "tosa.const"() '
                       f'{{values = dense<0.000000e+00> : tensor<1xf32>}} '
                       f': () -> tensor<1xf32>')

        result = self._ssa()
        pad_const_type = "tensor<1xi8>" if inp.mlir_type == "i8" else "tensor<1xf32>"
        self._emit(f"{result} = tosa.pad {inp_ssa}, {pad_ssa}, {pad_val_ssa}"
                   f" : ({inp.type_str}, !tosa.shape<{n_pad_elems}>, "
                   f"{pad_const_type}) -> {out.type_str}")

        self.tensor_to_ssa[output_idx] = result

    def _emit_shape(self, op):
        """SHAPE → tosa.const_shape (output is a 1-D i32 tensor of the input shape)."""
        input_idx = op.Inputs(0)
        output_idx = op.Outputs(0)

        inp = self.reader.tensors[input_idx]
        out = self.reader.tensors[output_idx]

        # SHAPE op returns the shape of the input as a 1-D i32 tensor.
        shape_vals = list(inp.shape)
        n = len(shape_vals)
        vals_str = ", ".join(str(d) for d in shape_vals)
        result = self._ssa()
        self._emit(f'{result} = "tosa.const"() {{values = dense<[{vals_str}]>'
                   f" : tensor<{n}xi32>}} : () -> tensor<{n}xi32>")
        self.tensor_to_ssa[output_idx] = result

    def _emit_strided_slice(self, op):
        """STRIDED_SLICE → tosa.slice (supports stride-1 cases only).

        When TFLite reduces rank (e.g. extracting a scalar from a 1-D tensor),
        we emit tosa.slice at the input rank, then tosa.reshape to the output shape.
        """
        input_idx = op.Inputs(0)
        output_idx = op.Outputs(0)

        inp = self.reader.tensors[input_idx]
        out = self.reader.tensors[output_idx]

        inp_ssa = self.tensor_to_ssa.get(input_idx)
        if inp_ssa is None:
            inp_ssa = self._emit_const_tensor(inp)

        # Read begin/end/strides tensors
        begin_t = self.reader.tensors[op.Inputs(1)]
        end_t = self.reader.tensors[op.Inputs(2)]
        strides_t = self.reader.tensors[op.Inputs(3)]

        begin = list(begin_t.buffer_data.flatten().astype(int)) if begin_t.buffer_data is not None else [0] * len(inp.shape)
        end = list(end_t.buffer_data.flatten().astype(int)) if end_t.buffer_data is not None else list(inp.shape)
        strides = list(strides_t.buffer_data.flatten().astype(int)) if strides_t.buffer_data is not None else [1] * len(inp.shape)

        # Clamp negative indices and compute sizes (at input rank)
        ndims = len(inp.shape)
        for i in range(ndims):
            if begin[i] < 0:
                begin[i] += inp.shape[i]
            if end[i] < 0:
                end[i] += inp.shape[i]
            end[i] = min(end[i], inp.shape[i])

        sizes = [(end[i] - begin[i] + strides[i] - 1) // strides[i]
                 for i in range(ndims)]

        # tosa.slice preserves rank, so sizes must have ndims entries.
        # Ensure every size is at least 1 (for scalar extraction).
        sizes = [max(s, 1) for s in sizes]

        # Determine if we need a reshape after the slice (rank reduction)
        out_shape = list(out.shape) if out.shape else []
        needs_reshape = (len(out_shape) != ndims or out_shape != sizes)

        if all(s == 1 for s in strides):
            # Stride-1: use tosa.slice directly
            start_ssa = self._ssa()
            start_vals = ", ".join(str(b) for b in begin)
            self._emit(f'{start_ssa} = "tosa.const_shape"() '
                       f'{{values = dense<[{start_vals}]> '
                       f': tensor<{ndims}xindex>}} : () '
                       f'-> !tosa.shape<{ndims}>')
            size_ssa = self._ssa()
            size_vals = ", ".join(str(s) for s in sizes)
            self._emit(f'{size_ssa} = "tosa.const_shape"() '
                       f'{{values = dense<[{size_vals}]> '
                       f': tensor<{ndims}xindex>}} : () '
                       f'-> !tosa.shape<{ndims}>')
            slice_type = tensor_type_str(sizes, inp.mlir_type)
            slice_result = self._ssa()
            self._emit(f"{slice_result} = tosa.slice {inp_ssa}, {start_ssa}, "
                       f"{size_ssa} : ({inp.type_str}, "
                       f"!tosa.shape<{ndims}>, !tosa.shape<{ndims}>) "
                       f"-> {slice_type}")

            if needs_reshape and out_shape:
                # Reshape from slice rank to output rank
                final_shape = out_shape
                final_type = tensor_type_str(final_shape, out.mlir_type)
                n_out = len(final_shape)
                shape_ssa = self._ssa()
                shape_vals = ", ".join(str(d) for d in final_shape)
                self._emit(f'{shape_ssa} = tosa.const_shape '
                           f' {{values = dense<[{shape_vals}]> '
                           f': tensor<{n_out}xindex>}} : () '
                           f'-> !tosa.shape<{n_out}>')
                result = self._ssa()
                self._emit(f"{result} = tosa.reshape {slice_result}, "
                           f"{shape_ssa} : ({slice_type}, "
                           f"!tosa.shape<{n_out}>) -> {final_type}")
                self.tensor_to_ssa[output_idx] = result
            elif needs_reshape and not out_shape:
                # Scalar output: reshape to [1] (TOSA doesn't support 0-d)
                scalar_type = tensor_type_str([1], out.mlir_type)
                shape_ssa = self._ssa()
                self._emit(f'{shape_ssa} = tosa.const_shape '
                           f' {{values = dense<[1]> '
                           f': tensor<1xindex>}} : () '
                           f'-> !tosa.shape<1>')
                result = self._ssa()
                self._emit(f"{result} = tosa.reshape {slice_result}, "
                           f"{shape_ssa} : ({slice_type}, "
                           f"!tosa.shape<1>) -> {scalar_type}")
                # Override the tensor shape for downstream consumers
                out.shape = [1]
                self.tensor_to_ssa[output_idx] = result
            else:
                self.tensor_to_ssa[output_idx] = slice_result
        else:
            # Non-unit strides: not directly supported in TOSA, pass through
            self._emit(f"// WARN: STRIDED_SLICE with strides={strides} "
                       f"(non-unit stride not supported, passed through)")
            self.tensor_to_ssa[output_idx] = inp_ssa

    def _emit_pack(self, op):
        """PACK (STACK) → tosa.concat + tosa.reshape (or direct const if inputs are constants)."""
        output_idx = op.Outputs(0)
        out = self.reader.tensors[output_idx]

        n_inputs = op.InputsLength()
        # Read axis from PackOptions
        from tflite.PackOptions import PackOptions
        opts = PackOptions()
        opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
        axis = opts.Axis()

        # Collect input SSA values
        inp_ssas = []
        inp_types = []
        for i in range(n_inputs):
            idx = op.Inputs(i)
            ssa = self.tensor_to_ssa.get(idx)
            if ssa is None:
                ssa = self._emit_const_tensor(self.reader.tensors[idx])
            inp_ssas.append(ssa)
            inp_types.append(self.reader.tensors[idx].type_str)

        # TOSA doesn't have a native pack/stack op.
        # Pack inserts a new dimension: expand each input, then concat.
        inp0 = self.reader.tensors[op.Inputs(0)]
        inp_shape = list(inp0.shape)
        ndims = len(inp_shape)

        # Insert dim at axis: reshape each input from [D0..Dn] to [D0..1..Dn]
        expanded_shape = list(inp_shape)
        expanded_shape.insert(axis, 1)
        expanded_type = tensor_type_str(expanded_shape, inp0.mlir_type)
        expanded_ndims = len(expanded_shape)

        expanded_ssas = []
        for i, (ssa, ty) in enumerate(zip(inp_ssas, inp_types)):
            shape_ssa = self._ssa()
            shape_vals = ", ".join(str(d) for d in expanded_shape)
            self._emit(f'{shape_ssa} = "tosa.const_shape"() '
                       f'{{values = dense<[{shape_vals}]> '
                       f': tensor<{expanded_ndims}xindex>}} : () '
                       f'-> !tosa.shape<{expanded_ndims}>')
            exp_ssa = self._ssa()
            self._emit(f"{exp_ssa} = tosa.reshape {ssa}, {shape_ssa} "
                       f": ({ty}, !tosa.shape<{expanded_ndims}>) "
                       f"-> {expanded_type}")
            expanded_ssas.append(exp_ssa)

        # Concat along axis
        concat_shape = list(expanded_shape)
        concat_shape[axis] = n_inputs
        concat_type = tensor_type_str(concat_shape, inp0.mlir_type)

        operands = ", ".join(expanded_ssas)
        operand_types = ", ".join([expanded_type] * n_inputs)
        result = self._ssa()
        self._emit(f"{result} = tosa.concat {operands} "
                   f"{{axis = {axis} : i32}} "
                   f": ({operand_types}) -> {concat_type}")

        # Reshape to output shape if needed
        if concat_shape != list(out.shape):
            ndims_out = len(out.shape)
            shape_ssa = self._ssa()
            shape_vals = ", ".join(str(d) for d in out.shape)
            self._emit(f'{shape_ssa} = "tosa.const_shape"() '
                       f'{{values = dense<[{shape_vals}]> '
                       f': tensor<{ndims_out}xindex>}} : () '
                       f'-> !tosa.shape<{ndims_out}>')
            final = self._ssa()
            self._emit(f"{final} = tosa.reshape {result}, {shape_ssa} "
                       f": ({concat_type}, !tosa.shape<{ndims_out}>) "
                       f"-> {out.type_str}")
            self.tensor_to_ssa[output_idx] = final
        else:
            self.tensor_to_ssa[output_idx] = result

    # ------------------------------------------------------------------
    # Rescale helper
    # ------------------------------------------------------------------

    def _emit_rescale(self, acc_ssa, acc_shape, acc_elem,
                      input_scale, weight_scales, output_scale,
                      output_zp, per_channel, output_type_str) -> str:
        """Emit tosa.rescale from int32 accumulator to int8 output."""
        multipliers, shifts = compute_rescale_multiplier_shift(
            input_scale, weight_scales, output_scale)

        n_ch = len(multipliers)
        mult_str = ", ".join(str(m) for m in multipliers)
        shift_str = ", ".join(str(s) for s in shifts)

        mult_ssa = self._ssa()
        shift_ssa = self._ssa()
        izp_ssa = self._ssa()
        ozp_ssa = self._ssa()

        self._emit(f'{mult_ssa} = "tosa.const"() {{values = dense<[{mult_str}]>'
                   f" : tensor<{n_ch}xi32>}} : () -> tensor<{n_ch}xi32>")
        self._emit(f'{shift_ssa} = "tosa.const"() {{values = dense<[{shift_str}]>'
                   f" : tensor<{n_ch}xi8>}} : () -> tensor<{n_ch}xi8>")
        self._emit(f'{izp_ssa} = "tosa.const"() {{values = dense<0>'
                   f" : tensor<1xi32>}} : () -> tensor<1xi32>")
        self._emit(f'{ozp_ssa} = "tosa.const"() {{values = dense<{output_zp}>'
                   f" : tensor<1xi8>}} : () -> tensor<1xi8>")

        result = self._ssa()
        acc_type = tensor_type_str(acc_shape, acc_elem)

        self._emit(
            f"{result} = tosa.rescale {acc_ssa}, {mult_ssa}, {shift_ssa}, "
            f"{izp_ssa}, {ozp_ssa} {{"
            f"scale32 = true, rounding_mode = DOUBLE_ROUND, "
            f"per_channel = {'true' if per_channel else 'false'}, "
            f"input_unsigned = false, output_unsigned = false"
            f"}} : ({acc_type}, tensor<{n_ch}xi32>, tensor<{n_ch}xi8>, "
            f"tensor<1xi32>, tensor<1xi8>) -> {output_type_str}")

        return result

    # ------------------------------------------------------------------
    # Main emit
    # ------------------------------------------------------------------

    def _fold_pad_into_conv(self):
        """Post-processing: fold tosa.pad + tosa.conv2d/depthwise_conv2d.

        LLVM 24's TOSA verifier requires (input + pad - eff_kernel) % stride == 0
        for the conv input.  When we emit a separate tosa.pad followed by a conv
        with pad=0,0,0,0, the padded dimension may not satisfy that constraint.

        Fix: remove the tosa.pad, feed the original input to the conv, and
        recompute TOSA-compatible padding from the original input shape, kernel,
        stride, dilation, and the known output shape.
        """
        import re

        # Build a map from SSA name -> line index and parse pad ops
        pad_ops = {}  # ssa_result -> info dict
        for i, line in enumerate(self.lines):
            m = re.match(
                r'\s+(%v\d+)\s*=\s*tosa\.pad\s+(%v\d+|%arg\d+),\s*(%v\d+),\s*(%v\d+)',
                line)
            if m:
                result_ssa = m.group(1)
                pad_ops[result_ssa] = {
                    'line_idx': i,
                    'input_ssa': m.group(2),
                    'shape_ssa': m.group(3),
                    'const_ssa': m.group(4),
                }

        if not pad_ops:
            return

        # Parse tosa.const_shape definitions for padding values
        shape_defs = {}
        for i, line in enumerate(self.lines):
            m = re.match(
                r'\s+(%v\d+)\s*=\s*"tosa\.const_shape"\(\)\s*\{values\s*=\s*dense<\[([^\]]+)\]>',
                line)
            if m:
                vals = [int(x.strip()) for x in m.group(2).split(',')]
                shape_defs[m.group(1)] = (i, vals)

        for ssa, info in pad_ops.items():
            shape_ssa = info['shape_ssa']
            if shape_ssa in shape_defs:
                info['pad_values'] = shape_defs[shape_ssa][1]
                info['shape_line_idx'] = shape_defs[shape_ssa][0]

        # Find conv2d/depthwise_conv2d ops that consume a pad output
        lines_to_remove = set()
        for i, line in enumerate(self.lines):
            m_conv = re.match(
                r'(\s+)(%v\d+)\s*=\s*tosa\.(conv2d|depthwise_conv2d)\s+'
                r'(%v\d+|%arg\d+),',
                line)
            if not m_conv:
                continue

            conv_input = m_conv.group(4)
            if conv_input not in pad_ops:
                continue

            pad_info = pad_ops[conv_input]
            if 'pad_values' not in pad_info:
                continue

            pad_vals = pad_info['pad_values']
            if len(pad_vals) != 8:
                continue
            if pad_vals[0] != 0 or pad_vals[1] != 0 or \
               pad_vals[6] != 0 or pad_vals[7] != 0:
                continue

            # Extract conv parameters from the line
            m_stride = re.search(r'stride\s*=\s*array<i64:\s*(\d+),\s*(\d+)>', line)
            m_dilation = re.search(r'dilation\s*=\s*array<i64:\s*(\d+),\s*(\d+)>', line)
            if not m_stride:
                continue
            stride_h, stride_w = int(m_stride.group(1)), int(m_stride.group(2))
            dilation_h = int(m_dilation.group(1)) if m_dilation else 1
            dilation_w = int(m_dilation.group(2)) if m_dilation else 1

            # Extract the kernel shape and output shape from type signature
            # Conv result type: -> tensor<BxOHxOWxOCxi32> or -> tensor<BxOHxOWxOCxi8>
            m_result_type = re.search(r'->\s*tensor<(\d+)x(\d+)x(\d+)x(\d+)x\w+>', line)
            if not m_result_type:
                continue
            out_h = int(m_result_type.group(2))
            out_w = int(m_result_type.group(3))

            # Extract kernel dims from weight type in the signature.
            # The type list in the line is: (input_type, weight_type, bias_type, ...)
            # Extract all 4D tensor types from the type signature.
            conv_kind = m_conv.group(3)
            m_type_sig = re.search(r':\s*\(([^)]+)\)', line)
            if not m_type_sig:
                continue
            type_list = [t.strip() for t in m_type_sig.group(1).split(',')]
            # Weight is the 2nd operand type (index 1)
            if len(type_list) < 2:
                continue
            wt_type = type_list[1]
            m_wt = re.match(r'tensor<(\d+)x(\d+)x(\d+)x(\d+)x\w+>', wt_type)
            if not m_wt:
                continue

            if conv_kind == 'conv2d':
                # conv2d weight: [OC, KH, KW, IC]
                kernel_h = int(m_wt.group(2))
                kernel_w = int(m_wt.group(3))
            else:  # depthwise_conv2d weight: [KH, KW, IC, M]
                kernel_h = int(m_wt.group(1))
                kernel_w = int(m_wt.group(2))

            # Get original (unpadded) input shape from the pad's input type
            pad_line = self.lines[pad_info['line_idx']]
            m_orig_type = re.search(r':\s*\(tensor<(\d+)x(\d+)x(\d+)x(\d+)x\w+>', pad_line)
            if not m_orig_type:
                continue
            orig_h = int(m_orig_type.group(2))
            orig_w = int(m_orig_type.group(3))
            original_type = re.search(r':\s*\(([^,]+),', pad_line).group(1).strip()

            # Extract padded type from pad line result
            m_pad_result_type = re.search(r'->\s*(\S+)\s*$', pad_line)
            padded_type = m_pad_result_type.group(1).strip() if m_pad_result_type else None

            # Compute TOSA-compatible padding that preserves the Pad op's
            # spatial offset (top/left) while satisfying TOSA's constraint
            # that (input + pad - eff_kernel) % stride == 0.
            # The top/left padding determines the spatial mapping (where
            # output pixel (0,0) maps in the original input), so we use
            # the actual values from the Pad op.  The bottom/right are
            # the remainder needed to reach the correct output size.
            eff_kh = (kernel_h - 1) * dilation_h + 1
            eff_kw = (kernel_w - 1) * dilation_w + 1
            total_pad_h = max(0, (out_h - 1) * stride_h + eff_kh - orig_h)
            total_pad_w = max(0, (out_w - 1) * stride_w + eff_kw - orig_w)
            # pad_vals layout: [N_before, N_after, H_before, H_after,
            #                   W_before, W_after, C_before, C_after]
            actual_pad_top = pad_vals[2]
            actual_pad_left = pad_vals[4]
            if actual_pad_top <= total_pad_h and actual_pad_left <= total_pad_w:
                new_pad_top = actual_pad_top
                new_pad_bottom = total_pad_h - actual_pad_top
                new_pad_left = actual_pad_left
                new_pad_right = total_pad_w - actual_pad_left
            else:
                # Pad op adds more than needed — don't fold
                continue

            # Replace the conv's input with the pad's original input
            original_input = pad_info['input_ssa']
            new_line = line.replace(f' {conv_input},', f' {original_input},', 1)

            # Replace the pad attribute
            new_line = re.sub(
                r'pad\s*=\s*array<i64:\s*\d+,\s*\d+,\s*\d+,\s*\d+>',
                f'pad = array<i64: {new_pad_top}, {new_pad_bottom}, '
                f'{new_pad_left}, {new_pad_right}>',
                new_line)

            # Fix the input tensor type in the signature
            if padded_type and original_type:
                new_line = new_line.replace(padded_type, original_type, 1)

            self.lines[i] = new_line

            # Mark the pad op and its helper lines for removal
            lines_to_remove.add(pad_info['line_idx'])
            if 'shape_line_idx' in pad_info:
                lines_to_remove.add(pad_info['shape_line_idx'])
            const_ssa = pad_info['const_ssa']
            for j, cline in enumerate(self.lines):
                if re.match(rf'\s+{re.escape(const_ssa)}\s*=', cline):
                    lines_to_remove.add(j)
                    break

        for idx in sorted(lines_to_remove, reverse=True):
            self.lines.pop(idx)

    def emit(self) -> str:
        """Generate the full MLIR module."""
        reader = self.reader

        # Function signature
        input_idxs = reader.inputs()
        output_idxs = reader.outputs()

        # Assign SSA names to function arguments
        arg_strs = []
        for i, idx in enumerate(input_idxs):
            t = reader.tensors[idx]
            name = f"%arg{i}"
            self.tensor_to_ssa[idx] = name
            arg_strs.append(f"{name}: {t.type_str}")

        ret_types = [reader.tensors[idx].type_str for idx in output_idxs]

        header = []
        header.append(f'// Auto-generated from TFLite model')
        header.append(f'// Tensors: {len(reader.tensors)}, '
                      f'Operators: {reader.subgraph.OperatorsLength()}')
        header.append(f'')
        header.append(f'func.func @main({", ".join(arg_strs)}) '
                      f'-> {", ".join(ret_types)} {{')

        # Walk operators
        OP_MAP = {
            BuiltinOperator.CONV_2D: self._emit_conv2d,
            BuiltinOperator.DEPTHWISE_CONV_2D: self._emit_depthwise_conv2d,
            BuiltinOperator.FULLY_CONNECTED: self._emit_fully_connected,
            BuiltinOperator.AVERAGE_POOL_2D:
                lambda op: self._emit_pool2d(op, "avg"),
            BuiltinOperator.MAX_POOL_2D:
                lambda op: self._emit_pool2d(op, "max"),
            BuiltinOperator.ADD: self._emit_add,
            BuiltinOperator.MUL: self._emit_mul,
            BuiltinOperator.RESHAPE: self._emit_reshape,
            BuiltinOperator.SOFTMAX: self._emit_softmax,
            BuiltinOperator.LOGISTIC: self._emit_logistic,
            BuiltinOperator.RELU: self._emit_relu,
            BuiltinOperator.RELU6: self._emit_relu6,
            BuiltinOperator.MEAN: self._emit_mean,
            BuiltinOperator.PAD: self._emit_pad,
            BuiltinOperator.SHAPE: self._emit_shape,
            BuiltinOperator.STRIDED_SLICE: self._emit_strided_slice,
            BuiltinOperator.PACK: self._emit_pack,
        }

        for code, op in reader.operators():
            emitter = OP_MAP.get(code)
            if emitter:
                self._emit(f"// --- {BuiltinOperator().BuiltinOperatorName(code) if hasattr(BuiltinOperator, 'BuiltinOperatorName') else f'op_{code}'} ---")
                try:
                    emitter(op)
                except Exception as e:
                    self._emit(f"// ERROR converting op code={code}: {e}")
            else:
                # Unsupported op — emit a comment and try to pass through
                self._emit(f"// UNSUPPORTED op code={code} — skipped")
                # Map outputs to inputs as pass-through
                for i in range(op.OutputsLength()):
                    out_idx = op.Outputs(i)
                    if op.InputsLength() > 0:
                        in_idx = op.Inputs(0)
                        if in_idx in self.tensor_to_ssa:
                            self.tensor_to_ssa[out_idx] = \
                                self.tensor_to_ssa[in_idx]

        # Post-process: fold tosa.pad into conv2d/depthwise_conv2d
        self._fold_pad_into_conv()

        # Return
        ret_vals = []
        for idx in output_idxs:
            if idx in self.tensor_to_ssa:
                ret_vals.append(self.tensor_to_ssa[idx])
            else:
                ret_vals.append(f"/*missing tensor {idx}*/")

        ret_types_str = ", ".join(ret_types)
        self._emit(f'return {", ".join(ret_vals)} : {ret_types_str}')

        footer = ['}']

        return "\n".join(header + self.lines + footer) + "\n"


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert a TFLite model to TOSA v1.0 MLIR")
    parser.add_argument("model", help="Path to .tflite file")
    parser.add_argument("-o", "--output", help="Output .mlir file (default: stdout)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print model summary to stderr")
    args = parser.parse_args()

    reader = TFLiteModelReader(args.model)

    if args.verbose:
        print(f"Model: {args.model}", file=sys.stderr)
        print(f"  Tensors: {len(reader.tensors)}", file=sys.stderr)
        print(f"  Operators: {reader.subgraph.OperatorsLength()}", file=sys.stderr)
        print(f"  Inputs: {reader.inputs()}", file=sys.stderr)
        print(f"  Outputs: {reader.outputs()}", file=sys.stderr)
        for t in reader.tensors:
            q = f" quant(scale={t.quant[0]}, zp={t.quant[1]})" if t.is_quantized else ""
            buf = " [has data]" if t.buffer_data is not None else ""
            print(f"  [{t.idx}] {t.name}: {t.type_str}{q}{buf}",
                  file=sys.stderr)

    emitter = TOSAEmitter(reader)
    mlir_text = emitter.emit()

    if args.output:
        with open(args.output, "w") as f:
            f.write(mlir_text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(mlir_text)


if __name__ == "__main__":
    main()
