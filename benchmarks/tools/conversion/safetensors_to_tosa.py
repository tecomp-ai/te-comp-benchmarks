#!/usr/bin/env python3
"""
safetensors_to_tosa.py — Convert a safetensors model to TOSA v1.0 MLIR.

Safetensors (.safetensors) stores only tensor data (weights, biases) without
a computation graph. To convert to TOSA, we also need a model config (JSON)
that describes the layer structure.

The tool supports two modes:
  1. Config-based: --config model_config.json (explicit layer definitions)
     - For HuggingFace models (model_type=gpt_neo, gpt2, etc.), the config
       is the standard HF config.json and the graph is auto-generated.
  2. Auto-detect:  Infer a simple sequential CNN/FC architecture from
                   weight tensor names following common naming conventions
                   (e.g., "layer0.conv.weight", "layer0.conv.bias", ...)

Supported layer types (CNN):
  conv2d           → tosa.conv2d  [+ rescale + clamp if quantized]
  depthwise_conv2d → tosa.depthwise_conv2d
  linear / fc      → tosa.matmul  (+ reshape)
  avg_pool2d       → tosa.avg_pool2d
  max_pool2d       → tosa.max_pool2d
  relu             → tosa.clamp
  relu6            → tosa.clamp
  reshape          → tosa.reshape
  flatten          → tosa.reshape  (to 2D)
  add              → tosa.add
  softmax          → (pass-through comment)

Supported architectures (Transformer):
  gpt_neo          → GPT-Neo (TinyStories-v2, TinyStories-Instruct)
                     embedding, layer_norm, matmul (Q/K/V), attention,
                     gelu_new, residual add, causal LM head

Usage:
  python3 safetensors_to_tosa.py model.safetensors --config config.json -o model.mlir
  python3 safetensors_to_tosa.py model.safetensors --input-shape 1,8,8,1 -o model.mlir
  python3 safetensors_to_tosa.py model.safetensors  # auto-detect, prints to stdout
"""

import argparse
import json
import math
import re
import struct
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from safetensors.numpy import load_file


# ============================================================================
# Type mapping
# ============================================================================

NUMPY_TO_MLIR = {
    np.dtype("float32"): "f32",
    np.dtype("float16"): "f16",
    np.dtype("int64"):   "i64",
    np.dtype("int32"):   "i32",
    np.dtype("int16"):   "i16",
    np.dtype("int8"):    "i8",
    np.dtype("uint8"):   "ui8",
    np.dtype("bool"):    "i1",
}


def mlir_elem_type(np_dtype) -> str:
    return NUMPY_TO_MLIR.get(np.dtype(np_dtype), "f32")


def tensor_type_str(shape: List[int], elem_type: str) -> str:
    dims = "x".join(str(d) for d in shape)
    return f"tensor<{dims}x{elem_type}>"


# ============================================================================
# Quantization helpers
# ============================================================================

class QuantParams:
    """Quantization parameters for a tensor."""
    def __init__(self, scale=None, zero_point=0, per_channel=False, qdim=0):
        self.scale = scale          # float or list[float]
        self.zero_point = zero_point  # int or list[int]
        self.per_channel = per_channel
        self.qdim = qdim

    @property
    def is_quantized(self):
        return self.scale is not None


def compute_rescale_multiplier_shift(input_scale, weight_scales, output_scale):
    """Compute fixed-point multiplier and shift for requantisation."""
    if not isinstance(weight_scales, (list, tuple)):
        weight_scales = [weight_scales]
    multipliers = []
    shifts = []
    for ws in weight_scales:
        effective = (input_scale * ws) / output_scale
        if effective == 0:
            multipliers.append(0)
            shifts.append(0)
            continue
        shift = 0
        while effective < 0.5:
            effective *= 2.0
            shift += 1
        while effective >= 1.0:
            effective /= 2.0
            shift -= 1
        mult = int(round(effective * (1 << 31)))
        if mult == (1 << 31):
            mult //= 2
            shift -= 1
        multipliers.append(mult)
        shifts.append(shift + 31)
    return multipliers, shifts


# ============================================================================
# Model config
# ============================================================================

class LayerConfig:
    """A single layer's configuration."""
    def __init__(self, d: dict):
        self.type = d["type"]                       # conv2d, linear, relu, etc.
        self.name = d.get("name", "")
        self.weight_key = d.get("weight", None)     # key in safetensors
        self.bias_key = d.get("bias", None)         # key in safetensors
        self.activation = d.get("activation", None)  # relu, relu6, None
        # Conv2d / pool params
        self.kernel_size = d.get("kernel_size", None)
        self.stride = d.get("stride", [1, 1])
        self.padding = d.get("padding", "valid")    # valid or same
        self.dilation = d.get("dilation", [1, 1])
        # Pool params
        self.pool_size = d.get("pool_size", None)
        # Reshape
        self.target_shape = d.get("target_shape", None)
        # Quantization (optional)
        self.quant = None
        if "quantization" in d:
            q = d["quantization"]
            self.quant = QuantParams(
                scale=q.get("scale"),
                zero_point=q.get("zero_point", 0),
                per_channel=q.get("per_channel", False),
                qdim=q.get("qdim", 0),
            )


def load_model_config(config_path: str) -> Tuple[List[LayerConfig], dict]:
    """Load model config JSON.

    If the config contains ``model_type`` (HuggingFace config.json), we return
    the raw dict as-is and let the caller build the graph via the appropriate
    architecture-specific emitter.  Otherwise, fall back to the legacy
    layer-list config format.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    # HuggingFace-style config detection
    if "model_type" in cfg:
        return cfg, None  # type: ignore[return-value]

    input_info = cfg.get("input", {})
    layers = [LayerConfig(l) for l in cfg.get("layers", [])]
    return layers, input_info


def auto_detect_layers(tensors: Dict[str, np.ndarray],
                       input_shape: Optional[List[int]] = None
                       ) -> Tuple[List[LayerConfig], dict]:
    """
    Auto-detect model architecture from safetensors key names.

    Supports naming conventions:
      layer{N}.{type}.weight / layer{N}.{type}.bias
      layers.{N}.weight / layers.{N}.bias
      {name}.weight / {name}.bias
      conv{N}.weight / fc{N}.weight
    """
    # Group keys by layer
    layer_groups = OrderedDict()
    for key in sorted(tensors.keys()):
        # Skip metadata keys
        if key.startswith("__"):
            continue
        # Try to extract layer name and param type
        parts = key.rsplit(".", 1)
        if len(parts) == 2:
            layer_name, param_type = parts
        else:
            layer_name = key
            param_type = "weight"

        if layer_name not in layer_groups:
            layer_groups[layer_name] = {}
        layer_groups[layer_name][param_type] = key

    layers = []
    for layer_name, params in layer_groups.items():
        weight_key = params.get("weight")
        bias_key = params.get("bias")

        if weight_key is None:
            continue

        w = tensors[weight_key]
        layer_type = _infer_layer_type(layer_name, w)

        d = {
            "type": layer_type,
            "name": layer_name,
            "weight": weight_key,
        }
        if bias_key:
            d["bias"] = bias_key

        if layer_type == "conv2d":
            # Weight shape: [OC, KH, KW, IC] (TOSA/TFLite) or [OC, IC, KH, KW] (PyTorch)
            if w.ndim == 4:
                # Detect layout: if dim[2]==dim[3] and they're small, likely PyTorch OIHW
                if w.shape[2] == w.shape[3] and w.shape[2] <= 7 and w.shape[1] > w.shape[2]:
                    # PyTorch: [OC, IC, KH, KW] — will transpose during emission
                    d["kernel_size"] = [w.shape[2], w.shape[3]]
                    d["weight_layout"] = "OIHW"
                else:
                    # TOSA/TFLite: [OC, KH, KW, IC]
                    d["kernel_size"] = [w.shape[1], w.shape[2]]
                    d["weight_layout"] = "OHWI"
        elif layer_type == "depthwise_conv2d":
            if w.ndim == 4:
                d["kernel_size"] = [w.shape[1], w.shape[2]] if w.shape[1] <= 7 else [w.shape[2], w.shape[3]]

        if "activation" not in d:
            # Check if there's a matching activation layer name
            act_name = layer_name + ".relu"
            if act_name in layer_groups or layer_name.endswith("_relu"):
                d["activation"] = "relu"

        layers.append(LayerConfig(d))

    # Detect standalone activation / pooling layers from names
    for key in tensors.keys():
        parts = key.rsplit(".", 1)
        name = parts[0] if len(parts) == 2 else key
        if "pool" in name.lower() and name not in [l.name for l in layers]:
            # Can't infer pool params from weights alone
            pass

    input_info = {}
    if input_shape:
        elem = "i8" if any(tensors[k].dtype == np.int8 for k in tensors) else "f32"
        input_info = {"shape": input_shape, "dtype": elem}

    return layers, input_info


def _infer_layer_type(name: str, weight: np.ndarray) -> str:
    """Infer layer type from name and weight shape."""
    name_lower = name.lower()
    if "conv" in name_lower and "depth" in name_lower:
        return "depthwise_conv2d"
    if "conv" in name_lower and weight.ndim == 4:
        return "conv2d"
    if weight.ndim == 4:
        return "conv2d"
    if "fc" in name_lower or "linear" in name_lower or "classifier" in name_lower:
        return "linear"
    if weight.ndim == 2:
        return "linear"
    return "linear"


# ============================================================================
# MLIR emitter
# ============================================================================

class TOSAEmitter:
    """Emit TOSA v1.0 MLIR from safetensors + config."""

    def __init__(self, tensors: Dict[str, np.ndarray],
                 layers: List[LayerConfig], input_info: dict):
        self.tensors = tensors
        self.layers = layers
        self.input_info = input_info
        self.lines: List[str] = []
        self.ssa_counter = 0
        self.indent = "    "
        self.current_shape: List[int] = []
        self.current_type: str = "f32"
        self.current_ssa: str = ""

    def _ssa(self) -> str:
        name = f"%v{self.ssa_counter}"
        self.ssa_counter += 1
        return name

    def _emit(self, line: str):
        self.lines.append(self.indent + line)

    def _format_val(self, v, elem_type) -> str:
        if "f" in elem_type:
            return f"{v:.6e}"
        return str(int(v))

    def _nest_array(self, arr, elem_type) -> str:
        if arr.ndim == 0:
            return self._format_val(arr.item(), elem_type)
        if arr.ndim == 1:
            vals = ", ".join(self._format_val(v, elem_type)
                            for v in arr.tolist())
            return f"[{vals}]"
        inner = ", ".join(self._nest_array(arr[i], elem_type)
                          for i in range(arr.shape[0]))
        return f"[{inner}]"

    def _format_dense_attr(self, data: np.ndarray) -> str:
        """Format a DenseElementsAttr string for the given numpy array.

        Small tensors (<=256 elements) use readable nested syntax.
        Large tensors use compact hex blob format: dense<"0xABCD...">.
        Splats always use dense<value> regardless of size.
        """
        elem = mlir_elem_type(data.dtype)
        shape = list(data.shape)
        ty = tensor_type_str(shape, elem)
        flat = data.flatten()
        num_elems = flat.size
        is_float = "f" in elem
        zero_lit = "0.000000e+00" if is_float else "0"

        if num_elems == 0:
            return f"dense<{zero_lit}> : {ty}"

        # Check for splat (all same value) — compact regardless of size
        first = flat[0]
        if np.all(flat == first):
            if first == 0:
                return f"dense<{zero_lit}> : {ty}"
            val = self._format_val(first.item(), elem)
            return f"dense<{val}> : {ty}"

        # Large tensors: emit as hex blob
        if num_elems > 256:
            raw_bytes = flat.tobytes()
            hex_str = raw_bytes.hex()
            return f'dense<"0x{hex_str}"> : {ty}'

        # Small tensors: emit as nested literal for readability
        nested = self._nest_array(data, elem)
        return f"dense<{nested}> : {ty}"

    def _emit_const(self, data: np.ndarray, name_hint: str = "") -> str:
        """Emit a tosa.const and return its SSA name."""
        ssa = self._ssa()
        elem = mlir_elem_type(data.dtype)
        shape = list(data.shape)
        ty = tensor_type_str(shape, elem)
        dense_attr = self._format_dense_attr(data)
        if name_hint and data.size > 256:
            self._emit(f"// {name_hint}: {data.size} elements")
        self._emit(f'{ssa} = "tosa.const"() {{values = {dense_attr}}} '
                   f": () -> {ty}")
        return ssa

    def _emit_scalar_const(self, value, elem_type: str) -> str:
        ssa = self._ssa()
        ty = f"tensor<1x{elem_type}>"
        val = self._format_val(value, elem_type)
        self._emit(f'{ssa} = "tosa.const"() {{values = dense<{val}> '
                   f": {ty}}} : () -> {ty}")
        return ssa

    def _get_padding(self, padding_str, input_h, input_w, kh, kw,
                     sh, sw, dh=1, dw=1):
        if padding_str == "valid":
            return [0, 0, 0, 0]
        # same padding
        eff_kh = (kh - 1) * dh + 1
        eff_kw = (kw - 1) * dw + 1
        pad_h = max(0, (input_h - 1) * sh + eff_kh - input_h)
        pad_w = max(0, (input_w - 1) * sw + eff_kw - input_w)
        return [pad_h // 2, pad_h - pad_h // 2,
                pad_w // 2, pad_w - pad_w // 2]

    def _compute_conv_output_shape(self, inp_shape, kh, kw, oc,
                                   stride, padding_str, dilation):
        h, w = inp_shape[1], inp_shape[2]
        sh, sw = stride
        dh, dw = dilation
        eff_kh = (kh - 1) * dh + 1
        eff_kw = (kw - 1) * dw + 1
        if padding_str == "same":
            oh = (h + sh - 1) // sh
            ow = (w + sw - 1) // sw
        else:
            oh = (h - eff_kh) // sh + 1
            ow = (w - eff_kw) // sw + 1
        return [inp_shape[0], oh, ow, oc]

    # ------------------------------------------------------------------
    # Layer emitters
    # ------------------------------------------------------------------

    def _emit_conv2d(self, layer: LayerConfig):
        weight_data = self.tensors[layer.weight_key]
        layout = getattr(layer, '__dict__', {})
        # Check for layout attribute stored in the raw config dict
        weight_layout = "OHWI"
        if hasattr(layer, '_raw') and 'weight_layout' in layer._raw:
            weight_layout = layer._raw['weight_layout']

        # Transpose if PyTorch layout [OC, IC, KH, KW] → TOSA [OC, KH, KW, IC]
        if weight_data.ndim == 4:
            if weight_data.shape[2] == weight_data.shape[3] and \
               weight_data.shape[2] <= 7 and weight_data.shape[1] > weight_data.shape[2]:
                weight_data = weight_data.transpose(0, 2, 3, 1)

        oc = weight_data.shape[0]
        kh = weight_data.shape[1]
        kw = weight_data.shape[2]
        ic = weight_data.shape[3]

        weight_ssa = self._emit_const(weight_data, f"{layer.name}.weight")

        # Bias
        if layer.bias_key and layer.bias_key in self.tensors:
            bias_data = self.tensors[layer.bias_key]
            bias_ssa = self._emit_const(bias_data, f"{layer.name}.bias")
            bias_elem = mlir_elem_type(bias_data.dtype)
            bias_ty = tensor_type_str(list(bias_data.shape), bias_elem)
        else:
            bias_elem = "i32" if self.current_type == "i8" else "f32"
            bias_ssa = self._ssa()
            bias_ty = f"tensor<{oc}x{bias_elem}>"
            self._emit(f'{bias_ssa} = "tosa.const"() {{values = dense<0> '
                       f": {bias_ty}}} : () -> {bias_ty}")

        stride = layer.stride if layer.stride else [1, 1]
        dilation = layer.dilation if layer.dilation else [1, 1]
        pad = self._get_padding(
            layer.padding, self.current_shape[1], self.current_shape[2],
            kh, kw, stride[0], stride[1], dilation[0], dilation[1])

        out_shape = self._compute_conv_output_shape(
            self.current_shape, kh, kw, oc, stride, layer.padding, dilation)

        inp_ty = tensor_type_str(self.current_shape, self.current_type)
        wt_ty = tensor_type_str(list(weight_data.shape), mlir_elem_type(weight_data.dtype))

        result = self._ssa()

        if self.current_type == "i8":
            # Quantized path
            inp_zp = layer.quant.zero_point if layer.quant else 0
            wt_zp = 0
            izp_ssa = self._emit_scalar_const(inp_zp, "i8")
            wzp_ssa = self._emit_scalar_const(wt_zp, "i8")
            acc_ty = tensor_type_str(out_shape, "i32")

            self._emit(
                f"{result} = tosa.conv2d {self.current_ssa}, {weight_ssa}, "
                f"{bias_ssa}, {izp_ssa}, {wzp_ssa} {{"
                f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                f"stride = array<i64: {stride[0]}, {stride[1]}>, "
                f"dilation = array<i64: {dilation[0]}, {dilation[1]}>, "
                f"acc_type = i32"
                f"}} : ({inp_ty}, {wt_ty}, {bias_ty}, "
                f"tensor<1xi8>, tensor<1xi8>) -> {acc_ty}")

            # Emit rescale if we have quant params
            if layer.quant and layer.quant.scale is not None:
                out_scale = layer.quant.scale if isinstance(layer.quant.scale, float) else layer.quant.scale[0]
                # Use uniform scale for now
                mults, shifts = compute_rescale_multiplier_shift(
                    out_scale, [out_scale] * oc, out_scale)
                result = self._emit_rescale(result, out_shape, mults, shifts, 0, "i8")
                self.current_type = "i8"
            else:
                self.current_type = "i32"

            self.current_shape = out_shape
            self.current_ssa = result
        else:
            # Float path
            izp_ssa = self._emit_scalar_const(0, "f32")
            wzp_ssa = self._emit_scalar_const(0, "f32")
            out_ty = tensor_type_str(out_shape, self.current_type)

            self._emit(
                f"{result} = tosa.conv2d {self.current_ssa}, {weight_ssa}, "
                f"{bias_ssa}, {izp_ssa}, {wzp_ssa} {{"
                f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                f"stride = array<i64: {stride[0]}, {stride[1]}>, "
                f"dilation = array<i64: {dilation[0]}, {dilation[1]}>, "
                f"acc_type = f32"
                f"}} : ({inp_ty}, {wt_ty}, {bias_ty}, "
                f"tensor<1xf32>, tensor<1xf32>) -> {out_ty}")

            self.current_shape = out_shape
            self.current_ssa = result

    def _emit_depthwise_conv2d(self, layer: LayerConfig):
        weight_data = self.tensors[layer.weight_key]

        # Weight layout for depthwise conv:
        # PyTorch: [IC, 1, KH, KW] → TOSA: [KH, KW, IC, 1]
        # TFLite:  [1, KH, KW, OC] → TOSA: [KH, KW, IC, M]
        if weight_data.ndim == 4:
            if weight_data.shape[1] == 1 and weight_data.shape[0] > 1:
                # PyTorch: [IC, 1, KH, KW] → [KH, KW, IC, 1]
                weight_data = weight_data.transpose(2, 3, 0, 1)
            elif weight_data.shape[0] == 1:
                # TFLite: [1, KH, KW, OC] → [KH, KW, IC, M]
                ic = self.current_shape[3]
                dm = weight_data.shape[3] // ic
                weight_data = weight_data[0].reshape(
                    weight_data.shape[1], weight_data.shape[2], ic, dm)

        kh, kw = weight_data.shape[0], weight_data.shape[1]
        ic = weight_data.shape[2]
        dm = weight_data.shape[3]
        oc = ic * dm

        weight_ssa = self._emit_const(weight_data, f"{layer.name}.weight")

        # Bias
        if layer.bias_key and layer.bias_key in self.tensors:
            bias_data = self.tensors[layer.bias_key]
            bias_ssa = self._emit_const(bias_data, f"{layer.name}.bias")
            bias_elem = mlir_elem_type(bias_data.dtype)
            bias_ty = tensor_type_str(list(bias_data.shape), bias_elem)
        else:
            bias_elem = "i32" if self.current_type == "i8" else "f32"
            bias_ssa = self._ssa()
            bias_ty = f"tensor<{oc}x{bias_elem}>"
            self._emit(f'{bias_ssa} = "tosa.const"() {{values = dense<0> '
                       f": {bias_ty}}} : () -> {bias_ty}")

        stride = layer.stride if layer.stride else [1, 1]
        dilation = layer.dilation if layer.dilation else [1, 1]
        pad = self._get_padding(
            layer.padding, self.current_shape[1], self.current_shape[2],
            kh, kw, stride[0], stride[1], dilation[0], dilation[1])

        out_shape = self._compute_conv_output_shape(
            self.current_shape, kh, kw, oc, stride, layer.padding, dilation)

        inp_ty = tensor_type_str(self.current_shape, self.current_type)
        wt_ty = tensor_type_str(list(weight_data.shape),
                                mlir_elem_type(weight_data.dtype))

        result = self._ssa()

        if self.current_type == "i8":
            inp_zp = layer.quant.zero_point if layer.quant else 0
            izp_ssa = self._emit_scalar_const(inp_zp, "i8")
            wzp_ssa = self._emit_scalar_const(0, "i8")
            acc_ty = tensor_type_str(out_shape, "i32")

            self._emit(
                f"{result} = tosa.depthwise_conv2d {self.current_ssa}, "
                f"{weight_ssa}, {bias_ssa}, {izp_ssa}, {wzp_ssa} {{"
                f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                f"stride = array<i64: {stride[0]}, {stride[1]}>, "
                f"dilation = array<i64: {dilation[0]}, {dilation[1]}>, "
                f"acc_type = i32"
                f"}} : ({inp_ty}, {wt_ty}, {bias_ty}, "
                f"tensor<1xi8>, tensor<1xi8>) -> {acc_ty}")

            if layer.quant and layer.quant.scale is not None:
                out_scale = layer.quant.scale if isinstance(
                    layer.quant.scale, float) else layer.quant.scale[0]
                mults, shifts = compute_rescale_multiplier_shift(
                    out_scale, [out_scale] * oc, out_scale)
                result = self._emit_rescale(
                    result, out_shape, mults, shifts, 0, "i8")
                self.current_type = "i8"
            else:
                self.current_type = "i32"

            self.current_shape = out_shape
            self.current_ssa = result
        else:
            izp_ssa = self._emit_scalar_const(0, "f32")
            wzp_ssa = self._emit_scalar_const(0, "f32")
            out_ty = tensor_type_str(out_shape, self.current_type)

            self._emit(
                f"{result} = tosa.depthwise_conv2d {self.current_ssa}, "
                f"{weight_ssa}, {bias_ssa}, {izp_ssa}, {wzp_ssa} {{"
                f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                f"stride = array<i64: {stride[0]}, {stride[1]}>, "
                f"dilation = array<i64: {dilation[0]}, {dilation[1]}>, "
                f"acc_type = f32"
                f"}} : ({inp_ty}, {wt_ty}, {bias_ty}, "
                f"tensor<1xf32>, tensor<1xf32>) -> {out_ty}")

            self.current_shape = out_shape
            self.current_ssa = result

    def _emit_linear(self, layer: LayerConfig):
        weight_data = self.tensors[layer.weight_key]
        # Weight shape: [out_features, in_features]
        N = weight_data.shape[0]  # output size
        K = weight_data.shape[1]  # input size

        # Flatten current shape to 2D if needed
        if len(self.current_shape) > 2:
            flat_size = 1
            for d in self.current_shape[1:]:
                flat_size *= d
            self._emit_reshape_op([self.current_shape[0], flat_size])

        # Reshape input to [batch, 1, K] for matmul
        batch = self.current_shape[0]
        inp_ty = tensor_type_str(self.current_shape, self.current_type)
        inp_3d_ty = f"tensor<{batch}x1x{K}x{self.current_type}>"

        shape_ssa = self._ssa()
        self._emit(f'{shape_ssa} = "tosa.const_shape"() '
                   f'{{values = dense<[{batch}, 1, {K}]> : tensor<3xindex>}}'
                   f" : () -> !tosa.shape<3>")
        reshape_ssa = self._ssa()
        self._emit(f"{reshape_ssa} = tosa.reshape {self.current_ssa}, {shape_ssa}"
                   f" : ({inp_ty}, !tosa.shape<3>) -> {inp_3d_ty}")

        # Transpose weight [N, K] → [1, K, N] as const
        wt_transposed = weight_data.T.reshape(1, K, N)
        wt_elem = mlir_elem_type(weight_data.dtype)
        wt_ty = tensor_type_str([1, K, N], wt_elem)
        wt_ssa = self._emit_const(wt_transposed, f"{layer.name}.weight (transposed)")

        result = self._ssa()
        if self.current_type == "i8":
            azp_ssa = self._emit_scalar_const(0, "i8")
            bzp_ssa = self._emit_scalar_const(0, "i8")
            acc_ty = f"tensor<{batch}x1x{N}xi32>"
            self._emit(
                f"{result} = tosa.matmul {reshape_ssa}, {wt_ssa}, "
                f"{azp_ssa}, {bzp_ssa}"
                f" : ({inp_3d_ty}, {wt_ty}, tensor<1xi8>, tensor<1xi8>) "
                f"-> {acc_ty}")
        else:
            azp_ssa = self._emit_scalar_const(0, "f32")
            bzp_ssa = self._emit_scalar_const(0, "f32")
            acc_ty = f"tensor<{batch}x1x{N}x{self.current_type}>"
            self._emit(
                f"{result} = tosa.matmul {reshape_ssa}, {wt_ssa}, "
                f"{azp_ssa}, {bzp_ssa}"
                f" : ({inp_3d_ty}, {wt_ty}, "
                f"tensor<1x{self.current_type}>, tensor<1x{self.current_type}>) "
                f"-> {acc_ty}")

        # Add bias
        if layer.bias_key and layer.bias_key in self.tensors:
            bias_data = self.tensors[layer.bias_key]
            bias_elem = mlir_elem_type(bias_data.dtype)
            bias_ssa = self._emit_const(bias_data, f"{layer.name}.bias")
            bias_ty = tensor_type_str(list(bias_data.shape), bias_elem)

            # Reshape bias to [1, 1, N]
            bias_3d_ty = f"tensor<1x1x{N}x{bias_elem}>"
            bshape = self._ssa()
            self._emit(f'{bshape} = "tosa.const_shape"() '
                       f'{{values = dense<[1, 1, {N}]> : tensor<3xindex>}}'
                       f" : () -> !tosa.shape<3>")
            bias_3d = self._ssa()
            self._emit(f"{bias_3d} = tosa.reshape {bias_ssa}, {bshape}"
                       f" : ({bias_ty}, !tosa.shape<3>) -> {bias_3d_ty}")
            add_result = self._ssa()
            self._emit(f"{add_result} = tosa.add {result}, {bias_3d}"
                       f" : ({acc_ty}, {bias_3d_ty}) -> {acc_ty}")
            result = add_result

        # For quantized models, rescale i32 accumulator back to i8
        result_elem = "i32" if self.current_type == "i8" else self.current_type
        if self.current_type == "i8" and layer.quant and layer.quant.scale is not None:
            out_scale = layer.quant.scale if isinstance(
                layer.quant.scale, float) else layer.quant.scale[0]
            out_zp = layer.quant.zero_point if isinstance(
                layer.quant.zero_point, int) else 0
            mults, shifts = compute_rescale_multiplier_shift(
                out_scale, [out_scale], out_scale)
            result = self._emit_rescale(
                result, [batch, 1, N], mults, shifts, out_zp, "i8")
            result_elem = "i8"

        # Reshape result from [batch, 1, N] to [batch, N]
        out_shape = [batch, N]
        oshape = self._ssa()
        self._emit(f'{oshape} = "tosa.const_shape"() '
                   f'{{values = dense<[{batch}, {N}]> : tensor<2xindex>}}'
                   f" : () -> !tosa.shape<2>")
        final = self._ssa()
        result_3d_ty = f"tensor<{batch}x1x{N}x{result_elem}>"
        out_2d_ty = tensor_type_str(out_shape, result_elem)
        self._emit(f"{final} = tosa.reshape {result}, {oshape}"
                   f" : ({result_3d_ty}, !tosa.shape<2>) -> {out_2d_ty}")

        self.current_shape = out_shape
        self.current_type = result_elem
        self.current_ssa = final

    def _emit_pool2d(self, layer: LayerConfig):
        pool_size = layer.pool_size or layer.kernel_size or [2, 2]
        stride = layer.stride if layer.stride else pool_size
        pad = self._get_padding(
            layer.padding, self.current_shape[1], self.current_shape[2],
            pool_size[0], pool_size[1], stride[0], stride[1])

        oh = (self.current_shape[1] - pool_size[0] + pad[0] + pad[1]) // stride[0] + 1
        ow = (self.current_shape[2] - pool_size[1] + pad[2] + pad[3]) // stride[1] + 1
        out_shape = [self.current_shape[0], oh, ow, self.current_shape[3]]

        inp_ty = tensor_type_str(self.current_shape, self.current_type)
        out_ty = tensor_type_str(out_shape, self.current_type)

        result = self._ssa()
        tosa_op = "tosa.avg_pool2d" if layer.type == "avg_pool2d" else "tosa.max_pool2d"

        if layer.type == "avg_pool2d":
            if self.current_type == "i8":
                izp = self._emit_scalar_const(0, "i8")
                ozp = self._emit_scalar_const(0, "i8")
                self._emit(
                    f"{result} = {tosa_op} {self.current_ssa}, {izp}, {ozp} {{"
                    f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                    f"kernel = array<i64: {pool_size[0]}, {pool_size[1]}>, "
                    f"stride = array<i64: {stride[0]}, {stride[1]}>, "
                    f"acc_type = i32"
                    f"}} : ({inp_ty}, tensor<1xi8>, tensor<1xi8>) -> {out_ty}")
            else:
                izp = self._emit_scalar_const(0, "f32")
                ozp = self._emit_scalar_const(0, "f32")
                self._emit(
                    f"{result} = {tosa_op} {self.current_ssa}, {izp}, {ozp} {{"
                    f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                    f"kernel = array<i64: {pool_size[0]}, {pool_size[1]}>, "
                    f"stride = array<i64: {stride[0]}, {stride[1]}>, "
                    f"acc_type = f32"
                    f"}} : ({inp_ty}, tensor<1xf32>, tensor<1xf32>) -> {out_ty}")
        else:
            self._emit(
                f"{result} = {tosa_op} {self.current_ssa} {{"
                f"pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, "
                f"kernel = array<i64: {pool_size[0]}, {pool_size[1]}>, "
                f"stride = array<i64: {stride[0]}, {stride[1]}>"
                f"}} : ({inp_ty}) -> {out_ty}")

        self.current_shape = out_shape
        self.current_ssa = result

    def _emit_relu(self, layer: LayerConfig):
        inp_ty = tensor_type_str(self.current_shape, self.current_type)
        result = self._ssa()
        if self.current_type == "i8":
            self._emit(f"{result} = tosa.clamp {self.current_ssa} "
                       f"{{min_val = 0 : i8, max_val = 127 : i8}}"
                       f" : ({inp_ty}) -> {inp_ty}")
        else:
            self._emit(f"{result} = tosa.clamp {self.current_ssa} "
                       f"{{min_val = 0.0 : f32, max_val = 3.40282347E+38 : f32}}"
                       f" : ({inp_ty}) -> {inp_ty}")
        self.current_ssa = result

    def _emit_relu6(self, layer: LayerConfig):
        inp_ty = tensor_type_str(self.current_shape, self.current_type)
        result = self._ssa()
        if self.current_type == "i8":
            self._emit(f"{result} = tosa.clamp {self.current_ssa} "
                       f"{{min_val = 0 : i8, max_val = 127 : i8}}"
                       f" : ({inp_ty}) -> {inp_ty}")
        else:
            self._emit(f"{result} = tosa.clamp {self.current_ssa} "
                       f"{{min_val = 0.0 : f32, max_val = 6.0 : f32}}"
                       f" : ({inp_ty}) -> {inp_ty}")
        self.current_ssa = result

    def _emit_reshape_op(self, target_shape: List[int]):
        inp_ty = tensor_type_str(self.current_shape, self.current_type)
        out_ty = tensor_type_str(target_shape, self.current_type)
        ndims = len(target_shape)
        dims = ", ".join(str(d) for d in target_shape)
        shape_ssa = self._ssa()
        self._emit(f'{shape_ssa} = "tosa.const_shape"() '
                   f'{{values = dense<[{dims}]> : tensor<{ndims}xindex>}}'
                   f" : () -> !tosa.shape<{ndims}>")
        result = self._ssa()
        self._emit(f"{result} = tosa.reshape {self.current_ssa}, {shape_ssa}"
                   f" : ({inp_ty}, !tosa.shape<{ndims}>) -> {out_ty}")
        self.current_shape = target_shape
        self.current_ssa = result

    def _emit_reshape(self, layer: LayerConfig):
        if layer.target_shape:
            self._emit_reshape_op(layer.target_shape)

    def _emit_flatten(self, layer: LayerConfig):
        flat_size = 1
        for d in self.current_shape[1:]:
            flat_size *= d
        self._emit_reshape_op([self.current_shape[0], flat_size])

    def _emit_softmax(self, layer: LayerConfig):
        self._emit(f"// SOFTMAX: not directly lowered — use table lookup "
                   f"or custom implementation")

    def _emit_rescale(self, acc_ssa, acc_shape, mults, shifts,
                      output_zp, output_elem) -> str:
        n_ch = len(mults)
        mult_str = ", ".join(str(m) for m in mults)
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
        acc_ty = tensor_type_str(acc_shape, "i32")
        out_ty = tensor_type_str(acc_shape, output_elem)
        per_ch = "true" if n_ch > 1 else "false"

        self._emit(
            f"{result} = tosa.rescale {acc_ssa}, {mult_ssa}, {shift_ssa}, "
            f"{izp_ssa}, {ozp_ssa} {{"
            f"scale32 = true, rounding_mode = DOUBLE_ROUND, "
            f"per_channel = {per_ch}, "
            f"input_unsigned = false, output_unsigned = false"
            f"}} : ({acc_ty}, tensor<{n_ch}xi32>, tensor<{n_ch}xi8>, "
            f"tensor<1xi32>, tensor<1xi8>) -> {out_ty}")
        return result

    def _emit_activation(self, layer: LayerConfig):
        if layer.activation == "relu":
            self._emit_relu(layer)
        elif layer.activation == "relu6":
            self._emit_relu6(layer)

    # ------------------------------------------------------------------
    # Main emit
    # ------------------------------------------------------------------

    def emit(self) -> str:
        # Determine input
        input_shape = self.input_info.get("shape", [1, 8, 8, 1])
        input_dtype = self.input_info.get("dtype", "f32")

        self.current_shape = input_shape
        self.current_type = input_dtype
        self.current_ssa = "%arg0"

        inp_ty = tensor_type_str(input_shape, input_dtype)

        header = []
        header.append("// Auto-generated from safetensors model")
        header.append(f"// Weights: {len(self.tensors)} tensors, "
                      f"Layers: {len(self.layers)}")
        header.append("")
        header.append(f"func.func @main(%arg0: {inp_ty}) -> ")

        # Process layers
        EMITTERS = {
            "conv2d": self._emit_conv2d,
            "depthwise_conv2d": self._emit_depthwise_conv2d,
            "linear": self._emit_linear,
            "fc": self._emit_linear,
            "avg_pool2d": self._emit_pool2d,
            "max_pool2d": self._emit_pool2d,
            "relu": self._emit_relu,
            "relu6": self._emit_relu6,
            "reshape": self._emit_reshape,
            "flatten": self._emit_flatten,
            "softmax": self._emit_softmax,
        }

        for i, layer in enumerate(self.layers):
            self._emit(f"// --- {layer.name or layer.type} ---")
            emitter = EMITTERS.get(layer.type)
            if emitter:
                try:
                    emitter(layer)
                    self._emit_activation(layer)
                except Exception as e:
                    self._emit(f"// ERROR: {e}")
            else:
                self._emit(f"// UNSUPPORTED layer type: {layer.type}")

        # Fix up return type in header
        out_ty = tensor_type_str(self.current_shape, self.current_type)
        header[-1] = f"func.func @main(%arg0: {inp_ty}) -> {out_ty} {{"

        # Return
        self._emit(f"return {self.current_ssa} : {out_ty}")

        footer = ["}"]
        return "\n".join(header + self.lines + footer) + "\n"


# ============================================================================
# GPT-Neo / GPT-2 transformer emitter
# ============================================================================

class GPTNeoTOSAEmitter:
    """Emit TOSA v1.0 MLIR for a GPT-Neo (or GPT-2) causal language model.

    Reads a HuggingFace config.json and the safetensors weights, and emits
    the full forward pass as a TOSA MLIR function.

    Input:  tensor<1xSxi64>  (token ids, S = sequence length)
    Output: tensor<1xSxVxf32> (logits, V = vocab_size)
    """

    def __init__(self, tensors: Dict[str, np.ndarray], config: dict,
                 seq_len: int = 1, prune_vocab: str = ""):
        self.T = tensors
        self.C = config
        self.seq_len = seq_len  # static sequence length for MLIR shapes

        # Model hyperparameters
        self.dim = config["hidden_size"]
        self.n_layers = config["num_layers"]
        self.n_heads = config["num_heads"]
        self.head_dim = self.dim // self.n_heads
        self.hidden_dim = config.get("intermediate_size") or self.dim * 4
        self.vocab_size = config["vocab_size"]
        self.max_pos = config.get("max_position_embeddings", 2048)
        self.ln_eps = config.get("layer_norm_epsilon", 1e-5)
        self.tied_weights = config.get("tie_word_embeddings", True)

        # Vocab pruning: load kept-ID list and build remap.
        self.kept_ids = None
        self.old2new = None
        self.orig_vocab_size = self.vocab_size
        if prune_vocab:
            self.kept_ids = np.loadtxt(prune_vocab, dtype=np.int64)
            self.old2new = {int(old): new
                           for new, old in enumerate(self.kept_ids)}
            self.vocab_size = len(self.kept_ids)
            print(f"Vocab pruning: {self.orig_vocab_size} -> "
                  f"{self.vocab_size} tokens "
                  f"(kept {len(self.kept_ids)} IDs from {prune_vocab})",
                  file=sys.stderr)

        self.lines: List[str] = []
        self.ssa_counter = 0
        self.indent = "    "

    # ---- SSA / emit helpers (same API as TOSAEmitter) ----

    def _ssa(self) -> str:
        name = f"%v{self.ssa_counter}"
        self.ssa_counter += 1
        return name

    def _emit(self, line: str):
        self.lines.append(self.indent + line)

    def _format_val(self, v, elem_type="f32") -> str:
        if "f" in elem_type:
            return f"{v:.6e}"
        return str(int(v))

    def _nest_array(self, arr, elem_type) -> str:
        if arr.ndim == 0:
            return self._format_val(arr.item(), elem_type)
        if arr.ndim == 1:
            vals = ", ".join(self._format_val(v, elem_type)
                            for v in arr.tolist())
            return f"[{vals}]"
        inner = ", ".join(self._nest_array(arr[i], elem_type)
                          for i in range(arr.shape[0]))
        return f"[{inner}]"

    def _format_dense_attr(self, data: np.ndarray) -> str:
        """Format a DenseElementsAttr string for the given numpy array.

        Small tensors (<=256 elements) use readable nested syntax.
        Large tensors use compact hex blob format: dense<"0xABCD...">.
        Splats always use dense<value> regardless of size.
        """
        elem = mlir_elem_type(data.dtype)
        shape = list(data.shape)
        ty = tensor_type_str(shape, elem)
        flat = data.flatten()
        num_elems = flat.size
        is_float = "f" in elem
        zero_lit = "0.000000e+00" if is_float else "0"

        if num_elems == 0:
            return f"dense<{zero_lit}> : {ty}"

        first = flat[0]
        if np.all(flat == first):
            if first == 0:
                return f"dense<{zero_lit}> : {ty}"
            val = self._format_val(first.item(), elem)
            return f"dense<{val}> : {ty}"

        if num_elems > 256:
            raw_bytes = flat.tobytes()
            hex_str = raw_bytes.hex()
            return f'dense<"0x{hex_str}"> : {ty}'

        nested = self._nest_array(data, elem)
        return f"dense<{nested}> : {ty}"

    def _emit_const(self, data: np.ndarray, hint: str = "") -> str:
        ssa = self._ssa()
        elem = mlir_elem_type(data.dtype)
        shape = list(data.shape)
        ty = tensor_type_str(shape, elem)
        dense_attr = self._format_dense_attr(data)
        if hint and data.size > 256:
            self._emit(f"// {hint}: {data.size} elements")
        self._emit(f'{ssa} = "tosa.const"() {{values = {dense_attr}}} '
                   f": () -> {ty}")
        return ssa

    def _emit_const_shape(self, dims: List[int]) -> str:
        ssa = self._ssa()
        n = len(dims)
        vals = ", ".join(str(d) for d in dims)
        self._emit(f'{ssa} = "tosa.const_shape"() '
                   f'{{values = dense<[{vals}]> : tensor<{n}xindex>}} '
                   f': () -> !tosa.shape<{n}>')
        return ssa

    def _ty(self, shape: List[int], elem: str = "f32") -> str:
        return tensor_type_str(shape, elem)

    # ---- Reshape helper ----

    def _reshape(self, ssa: str, old_shape: List[int], new_shape: List[int],
                 elem: str = "f32") -> str:
        old_ty = self._ty(old_shape, elem)
        new_ty = self._ty(new_shape, elem)
        shape_ssa = self._emit_const_shape(new_shape)
        result = self._ssa()
        self._emit(f"{result} = tosa.reshape {ssa}, {shape_ssa} "
                   f": ({old_ty}, !tosa.shape<{len(new_shape)}>) -> {new_ty}")
        return result

    # ---- Transpose helper ----

    def _transpose(self, ssa: str, shape: List[int], perm: List[int],
                   elem: str = "f32") -> Tuple[str, List[int]]:
        in_ty = self._ty(shape, elem)
        new_shape = [shape[p] for p in perm]
        out_ty = self._ty(new_shape, elem)
        perm_str = ", ".join(str(p) for p in perm)
        result = self._ssa()
        self._emit(f"{result} = tosa.transpose {ssa} "
                   f"{{perms = array<i32: {perm_str}>}} "
                   f": ({in_ty}) -> {out_ty}")
        return result, new_shape

    # ---- Element-wise ops ----

    def _add(self, a: str, b: str, shape_a: List[int], shape_b: List[int],
             elem: str = "f32") -> str:
        # TOSA add requires matching ranks — pad shorter operand
        target_rank = max(len(shape_a), len(shape_b))
        a, shape_a = self._broadcast_shape(a, shape_a, target_rank, elem)
        b, shape_b = self._broadcast_shape(b, shape_b, target_rank, elem)
        ty_a = self._ty(shape_a, elem)
        ty_b = self._ty(shape_b, elem)
        out_shape = [max(sa, sb) for sa, sb in zip(shape_a, shape_b)]
        out_ty = self._ty(out_shape, elem)
        result = self._ssa()
        self._emit(f"{result} = tosa.add {a}, {b} "
                   f": ({ty_a}, {ty_b}) -> {out_ty}")
        return result

    def _sub(self, a: str, b: str, shape_a: List[int],
             shape_b: Optional[List[int]] = None,
             elem: str = "f32") -> str:
        if shape_b is None:
            shape_b = shape_a
        # TOSA sub requires matching ranks — pad shorter operand
        target_rank = max(len(shape_a), len(shape_b))
        a, shape_a = self._broadcast_shape(a, shape_a, target_rank, elem)
        b, shape_b = self._broadcast_shape(b, shape_b, target_rank, elem)
        ty_a = self._ty(shape_a, elem)
        ty_b = self._ty(shape_b, elem)
        out_shape = [max(sa, sb) for sa, sb in zip(shape_a, shape_b)]
        out_ty = self._ty(out_shape, elem)
        result = self._ssa()
        self._emit(f"{result} = tosa.sub {a}, {b} "
                   f": ({ty_a}, {ty_b}) -> {out_ty}")
        return result

    def _broadcast_shape(self, ssa: str, shape: List[int],
                         target_rank: int, elem: str = "f32") -> Tuple[str, List[int]]:
        """Pad shape with leading 1s to match target_rank, reshaping if needed."""
        if len(shape) >= target_rank:
            return ssa, shape
        new_shape = [1] * (target_rank - len(shape)) + shape
        return self._reshape(ssa, shape, new_shape, elem), new_shape

    def _mul(self, a: str, b: str, shape_a: List[int], shape_b: List[int],
             elem: str = "f32") -> str:
        # TOSA mul requires matching ranks — pad shorter operand
        target_rank = max(len(shape_a), len(shape_b))
        a, shape_a = self._broadcast_shape(a, shape_a, target_rank, elem)
        b, shape_b = self._broadcast_shape(b, shape_b, target_rank, elem)
        ty_a = self._ty(shape_a, elem)
        ty_b = self._ty(shape_b, elem)
        out_shape = [max(sa, sb) for sa, sb in zip(shape_a, shape_b)]
        out_ty = self._ty(out_shape, elem)
        shift_ssa = self._emit_const(
            np.array([0], dtype=np.int8), "mul_shift")
        result = self._ssa()
        self._emit(f"{result} = tosa.mul {a}, {b}, {shift_ssa} "
                   f": ({ty_a}, {ty_b}, tensor<1xi8>) -> {out_ty}")
        return result

    def _reciprocal(self, ssa: str, shape: List[int],
                    elem: str = "f32") -> str:
        ty = self._ty(shape, elem)
        result = self._ssa()
        self._emit(f"{result} = tosa.reciprocal {ssa} : ({ty}) -> {ty}")
        return result

    # ---- Matmul: [B, M, K] x [B, K, N] -> [B, M, N] ----

    def _matmul(self, a: str, b: str, shape_a: List[int],
                shape_b: List[int], elem: str = "f32") -> Tuple[str, List[int]]:
        ty_a = self._ty(shape_a, elem)
        ty_b = self._ty(shape_b, elem)
        out_shape = [shape_a[0], shape_a[1], shape_b[2]]
        out_ty = self._ty(out_shape, elem)
        azp = self._emit_const(np.zeros([1], dtype=np.float32), "azp")
        bzp = self._emit_const(np.zeros([1], dtype=np.float32), "bzp")
        result = self._ssa()
        self._emit(
            f"{result} = tosa.matmul {a}, {b}, {azp}, {bzp} "
            f": ({ty_a}, {ty_b}, tensor<1xf32>, tensor<1xf32>) -> {out_ty}")
        return result, out_shape

    # ---- Layer Norm ----

    def _layer_norm(self, x: str, shape: List[int],
                    gamma_key: str, beta_key: str) -> str:
        """LayerNorm(x, gamma, beta) over the last dimension.

        Decomposed into: mean → sub → variance → rsqrt → mul → mul(gamma) → add(beta)
        """
        S = self.seq_len
        D = shape[-1]
        ty = self._ty(shape)

        # mean = reduce_sum(x, axis=-1) / D
        reduce_ty = self._ty(shape[:-1] + [1])
        reduce_result = self._ssa()
        self._emit(f"{reduce_result} = tosa.reduce_sum {x} "
                   f"{{axis = {len(shape)-1} : i32}} "
                   f": ({ty}) -> {reduce_ty}")
        d_const = self._emit_const(
            np.array([1.0 / D], dtype=np.float32), "1/dim")
        mean = self._mul(reduce_result, d_const,
                         shape[:-1] + [1], [1])

        # x_centered = x - mean  (broadcast [B,S,D] - [B,S,1])
        mean_shape = shape[:-1] + [1]
        x_centered = self._sub(x, mean, shape, mean_shape)

        # var = reduce_sum(x_centered^2, axis=-1) / D
        x2 = self._mul(x_centered, x_centered, shape, shape)
        var_sum = self._ssa()
        self._emit(f"{var_sum} = tosa.reduce_sum {x2} "
                   f"{{axis = {len(shape)-1} : i32}} "
                   f": ({ty}) -> {reduce_ty}")
        var = self._mul(var_sum, d_const, shape[:-1] + [1], [1])

        # inv_std = rsqrt(var + eps)
        eps_const = self._emit_const(
            np.array([self.ln_eps], dtype=np.float32), "ln_eps")
        var_eps = self._add(var, eps_const, shape[:-1] + [1], [1])
        var_eps_ty = self._ty(shape[:-1] + [1])
        inv_std = self._ssa()
        self._emit(f"{inv_std} = tosa.rsqrt {var_eps} "
                   f": ({var_eps_ty}) -> {var_eps_ty}")

        # normalized = x_centered * inv_std
        normalized = self._mul(x_centered, inv_std, shape, shape[:-1] + [1])

        # output = normalized * gamma + beta
        gamma = self._emit_const(self.T[gamma_key], gamma_key)
        beta = self._emit_const(self.T[beta_key], beta_key)
        scaled = self._mul(normalized, gamma, shape, [D])
        return self._add(scaled, beta, shape, [D])

    # ---- GELU (tanh approximation, "gelu_new") ----

    def _gelu_new(self, x: str, shape: List[int]) -> str:
        """gelu_new(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
        ty = self._ty(shape)

        # x^3 via x * x * x (TOSA has no pow op for non-integer exponents)
        x2 = self._mul(x, x, shape, shape)
        x3 = self._mul(x2, x, shape, shape)

        # 0.044715 * x^3
        coeff = self._emit_const(
            np.array([0.044715], dtype=np.float32), "gelu_coeff")
        scaled_x3 = self._mul(x3, coeff, shape, [1])

        # x + 0.044715 * x^3
        inner = self._add(x, scaled_x3, shape, shape)

        # sqrt(2/pi) * (...)
        sqrt2pi = self._emit_const(
            np.array([0.7978845608], dtype=np.float32), "sqrt_2_over_pi")
        tanh_arg = self._mul(inner, sqrt2pi, shape, [1])

        # tanh(...)
        tanh_result = self._ssa()
        self._emit(f"{tanh_result} = tosa.tanh {tanh_arg} "
                   f": ({ty}) -> {ty}")

        # 1 + tanh(...)
        one = self._emit_const(
            np.array([1.0], dtype=np.float32), "one")
        one_plus_tanh = self._add(tanh_result, one, shape, [1])

        # 0.5 * x
        half = self._emit_const(
            np.array([0.5], dtype=np.float32), "half")
        half_x = self._mul(x, half, shape, [1])

        # 0.5 * x * (1 + tanh(...))
        return self._mul(half_x, one_plus_tanh, shape, shape)

    # ---- Softmax (numerically stable) ----

    def _softmax(self, x: str, shape: List[int]) -> str:
        """softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))"""
        ty = self._ty(shape)
        last_axis = len(shape) - 1
        reduce_shape = shape[:-1] + [1]
        reduce_ty = self._ty(reduce_shape)

        # max along last axis
        max_val = self._ssa()
        self._emit(f"{max_val} = tosa.reduce_max {x} "
                   f"{{axis = {last_axis} : i32}} "
                   f": ({ty}) -> {reduce_ty}")

        # x - max  (broadcast)
        x_shifted = self._sub(x, max_val, shape, reduce_shape)

        # exp(x - max)
        exp_val = self._ssa()
        self._emit(f"{exp_val} = tosa.exp {x_shifted} : ({ty}) -> {ty}")

        # sum(exp)
        sum_exp = self._ssa()
        self._emit(f"{sum_exp} = tosa.reduce_sum {exp_val} "
                   f"{{axis = {last_axis} : i32}} "
                   f": ({ty}) -> {reduce_ty}")

        # 1 / sum(exp)
        inv_sum = self._reciprocal(sum_exp, reduce_shape)

        # exp * (1 / sum)
        return self._mul(exp_val, inv_sum, shape, reduce_shape)

    # ---- Linear projection (matmul + bias) ----

    def _linear(self, x: str, x_shape: List[int],
                weight_key: str, bias_key: Optional[str] = None,
                label: str = "") -> Tuple[str, List[int]]:
        """Linear: x @ W^T + b.  x is [B, S, D_in], result is [B, S, D_out]."""
        W = self.T[weight_key]  # [D_out, D_in]
        D_out, D_in = W.shape
        B, S = x_shape[0], x_shape[1]

        # Transpose weight to [1, D_in, D_out] for tosa.matmul
        wt = W.T.reshape(1, D_in, D_out).astype(np.float32)
        wt_ssa = self._emit_const(wt, f"{label}.weight (transposed)")

        result, out_shape = self._matmul(
            x, wt_ssa, x_shape, [1, D_in, D_out])

        if bias_key and bias_key in self.T:
            bias = self.T[bias_key].astype(np.float32)
            bias_ssa = self._emit_const(bias, f"{label}.bias")
            result = self._add(result, bias_ssa, out_shape, [D_out])

        return result, out_shape

    # ---- Causal self-attention for one layer ----

    def _self_attention(self, x: str, x_shape: List[int],
                        layer_idx: int) -> str:
        """Multi-head causal self-attention.

        GPT-Neo uses separate Q/K/V projections (not fused QKV).
        No attention scaling (GPT-Neo doesn't scale by 1/sqrt(head_dim)).
        """
        B, S, D = x_shape
        H = self.n_heads
        HD = self.head_dim
        prefix = f"transformer.h.{layer_idx}.attn.attention"

        self._emit(f"// --- Attention (layer {layer_idx}) ---")

        # Q, K, V projections: [B, S, D] -> [B, S, D]
        q, q_shape = self._linear(x, x_shape,
                                  f"{prefix}.q_proj.weight",
                                  None, f"layer{layer_idx}.Q")
        k, k_shape = self._linear(x, x_shape,
                                  f"{prefix}.k_proj.weight",
                                  None, f"layer{layer_idx}.K")
        v, v_shape = self._linear(x, x_shape,
                                  f"{prefix}.v_proj.weight",
                                  None, f"layer{layer_idx}.V")

        # Reshape to [B, S, H, HD] then transpose to [B, H, S, HD]
        q = self._reshape(q, q_shape, [B, S, H, HD])
        q, _ = self._transpose(q, [B, S, H, HD], [0, 2, 1, 3])
        # q shape is now [B, H, S, HD]

        k = self._reshape(k, k_shape, [B, S, H, HD])
        k, _ = self._transpose(k, [B, S, H, HD], [0, 2, 1, 3])

        v = self._reshape(v, v_shape, [B, S, H, HD])
        v, _ = self._transpose(v, [B, S, H, HD], [0, 2, 1, 3])

        # For matmul, reshape [B, H, S, HD] -> [B*H, S, HD]
        q_3d = self._reshape(q, [B, H, S, HD], [B * H, S, HD])
        k_3d = self._reshape(k, [B, H, S, HD], [B * H, S, HD])
        v_3d = self._reshape(v, [B, H, S, HD], [B * H, S, HD])

        # K^T: transpose last two dims -> [B*H, HD, S]
        k_t, _ = self._transpose(k_3d, [B * H, S, HD], [0, 2, 1])

        # scores = Q @ K^T -> [B*H, S, S]
        scores, scores_shape = self._matmul(
            q_3d, k_t, [B * H, S, HD], [B * H, HD, S])

        # Causal mask: add -inf to upper triangle
        # Create a constant lower-triangular mask
        mask = np.triu(np.full((S, S), -1e9, dtype=np.float32), k=1)
        mask_ssa = self._emit_const(mask, "causal_mask")
        scores = self._add(scores, mask_ssa, scores_shape, [S, S])

        # Softmax over last dimension
        scores = self._softmax(scores, scores_shape)

        # Attention output: scores @ V -> [B*H, S, HD]
        attn_out, attn_shape = self._matmul(
            scores, v_3d, [B * H, S, S], [B * H, S, HD])

        # Reshape back: [B*H, S, HD] -> [B, H, S, HD] -> [B, S, H, HD] -> [B, S, D]
        attn_out = self._reshape(attn_out, attn_shape, [B, H, S, HD])
        attn_out, _ = self._transpose(attn_out, [B, H, S, HD], [0, 2, 1, 3])
        attn_out = self._reshape(attn_out, [B, S, H, HD], [B, S, D])

        # Output projection + bias
        out, out_shape = self._linear(
            attn_out, [B, S, D],
            f"{prefix}.out_proj.weight",
            f"{prefix}.out_proj.bias",
            f"layer{layer_idx}.out_proj")

        return out

    # ---- Full forward pass ----

    def emit(self) -> str:
        B = 1
        S = self.seq_len
        D = self.dim
        V = self.vocab_size

        # Function signature: input is token ids
        inp_ty = self._ty([B, S], "i64")
        out_ty = self._ty([B, S, V], "f32")  # will fix up at end

        header = [
            f"// GPT-Neo ({self.n_layers}L, {self.n_heads}H, D={D}, "
            f"V={V}, S={S})",
            f"// Parameters: {sum(t.size for t in self.T.values()):,}",
            "",
            f"func.func @main(%tokens: {inp_ty}) -> {out_ty} {{",
        ]

        # ---- Token embedding: gather from wte ----
        self._emit(f"// --- Token Embedding ---")
        wte_full = self.T["transformer.wte.weight"]  # [V_orig, D]
        if self.kept_ids is not None:
            # Prune embedding table to kept tokens only.
            wte = wte_full[self.kept_ids]  # [V_pruned, D]
            self._emit(f"// Vocab pruned: {self.orig_vocab_size} -> {V}")
        else:
            wte = wte_full
        wte_ssa = self._emit_const(wte, "wte")

        # Cast tokens from i64 to i32 for tosa.gather
        tokens_i32 = self._ssa()
        self._emit(f"{tokens_i32} = tosa.cast %tokens "
                   f": ({inp_ty}) -> {self._ty([B, S], 'i32')}")

        # If vocab is pruned, remap token IDs: old_id -> new_id.
        # Emit a remap table [V_orig] as i32 const and gather through it.
        if self.kept_ids is not None:
            remap = np.full(self.orig_vocab_size, 0, dtype=np.int32)
            for old_id, new_id in self.old2new.items():
                remap[old_id] = new_id
            remap_ssa = self._emit_const(remap, "vocab_remap")
            # Reshape remap to [1, V_orig, 1] for gather
            V_orig = self.orig_vocab_size
            remap_3d = self._reshape(remap_ssa, [V_orig], [1, V_orig, 1],
                                     elem="i32")
            # Gather: remap_3d[tokens_i32] -> [1, S, 1]
            remapped_3d = self._ssa()
            self._emit(
                f"{remapped_3d} = tosa.gather {remap_3d}, {tokens_i32} "
                f": ({self._ty([1, V_orig, 1], 'i32')}, "
                f"{self._ty([B, S], 'i32')}) "
                f"-> {self._ty([B, S, 1], 'i32')}")
            # Reshape [1, S, 1] -> [1, S]
            tokens_i32 = self._reshape(remapped_3d, [B, S, 1], [B, S],
                                       elem="i32")

        # tosa.gather: values=[1, V, D], indices=[1, S] -> [1, S, D]
        # Reshape wte to [1, V, D]
        wte_3d = self._reshape(wte_ssa, [V, D], [1, V, D])

        tok_embed = self._ssa()
        self._emit(
            f"{tok_embed} = tosa.gather {wte_3d}, {tokens_i32} "
            f": ({self._ty([1, V, D])}, {self._ty([B, S], 'i32')}) "
            f"-> {self._ty([B, S, D])}")

        # ---- Position embedding ----
        self._emit(f"// --- Position Embedding ---")
        wpe = self.T["transformer.wpe.weight"]  # [max_pos, D]

        if S == 1:
            # For seq_len=1 (autoregressive KV cache mode), emit the FULL
            # wpe table as a constant.  The KV cache pass will detect this
            # pattern and convert it to te.pos_embed, which indexes wpe
            # by the runtime position counter (__kv_pos_).
            #
            # We emit:  wpe_full [max_pos, D] as const
            #           gather  [1, max_pos, D] x indices=[1,1] -> [1,1,D]
            # but annotated with te.wpe_embed=true so the KV cache pass
            # can recognize and replace it.
            #
            # For now, emit the full table + placeholder gather(pos=0) that
            # the KV cache pass will replace.  This ensures the full wpe
            # table makes it into the compiled binary for runtime indexing.
            wpe_full = wpe[:self.max_pos, :].astype(np.float32)
            wpe_ssa = self._emit_const(wpe_full, "wpe_full")
            # Reshape to [1, max_pos, D] for tosa.gather
            wpe_3d = self._reshape(wpe_ssa, [self.max_pos, D],
                                   [1, self.max_pos, D])
            # Emit a position-0 index that the KV cache pass will replace
            pos_idx = self._emit_const(
                np.array([[0]], dtype=np.int32), "wpe_pos_idx")
            pos_embed = self._ssa()
            self._emit(
                f'{pos_embed} = tosa.gather {wpe_3d}, {pos_idx} '
                f': ({self._ty([1, self.max_pos, D])}, '
                f'{self._ty([1, 1], "i32")}) '
                f'-> {self._ty([B, S, D])}')
            # Mark the gather as a wpe embed (attribute for KV cache pass)
            # We emit a comment since TOSA gather doesn't have custom attrs,
            # but the KV pass can detect wpe gathers by shape/pattern.
            self._emit(f'// te.wpe_embed: true (wpe table [{self.max_pos}, {D}])')
        else:
            # Static multi-token mode: bake in wpe[:S] directly
            wpe_slice = wpe[:S, :].astype(np.float32)
            pos_embed = self._emit_const(
                wpe_slice.reshape(1, S, D), "wpe[:S]")

        # x = tok_embed + pos_embed
        x = self._add(tok_embed, pos_embed, [B, S, D], [1, S, D])
        x_shape = [B, S, D]

        # ---- Transformer blocks ----
        for l in range(self.n_layers):
            self._emit(f"")
            self._emit(f"// ========== Transformer Block {l} ==========")

            # Pre-attention LayerNorm
            ln1 = self._layer_norm(
                x, x_shape,
                f"transformer.h.{l}.ln_1.weight",
                f"transformer.h.{l}.ln_1.bias")

            # Self-attention
            attn_out = self._self_attention(ln1, x_shape, l)

            # Residual: x = x + attn_out
            x = self._add(x, attn_out, x_shape, x_shape)

            # Pre-FFN LayerNorm
            ln2 = self._layer_norm(
                x, x_shape,
                f"transformer.h.{l}.ln_2.weight",
                f"transformer.h.{l}.ln_2.bias")

            # FFN: fc_up -> gelu -> fc_down
            self._emit(f"// --- FFN (layer {l}) ---")
            ffn_up, ffn_shape = self._linear(
                ln2, x_shape,
                f"transformer.h.{l}.mlp.c_fc.weight",
                f"transformer.h.{l}.mlp.c_fc.bias",
                f"layer{l}.ffn_up")

            ffn_act = self._gelu_new(ffn_up, ffn_shape)

            ffn_down, _ = self._linear(
                ffn_act, ffn_shape,
                f"transformer.h.{l}.mlp.c_proj.weight",
                f"transformer.h.{l}.mlp.c_proj.bias",
                f"layer{l}.ffn_down")

            # Residual: x = x + ffn_down
            x = self._add(x, ffn_down, x_shape, x_shape)

        # ---- Final LayerNorm ----
        self._emit(f"")
        self._emit(f"// --- Final LayerNorm ---")
        x = self._layer_norm(
            x, x_shape,
            "transformer.ln_f.weight",
            "transformer.ln_f.bias")

        # ---- LM head (tied to wte) ----
        self._emit(f"// --- LM Head (logits) ---")
        if self.tied_weights:
            # Reuse wte constant via transpose+reshape (no duplicate copy).
            # wte_ssa is [V, D] -> transpose [D, V] -> reshape [1, D, V]
            wte_t, wte_t_shape = self._transpose(
                wte_ssa, [V, D], [1, 0])           # [V,D] -> [D,V]
            lm_ssa = self._reshape(
                wte_t, wte_t_shape, [1, D, V])     # [D,V] -> [1,D,V]
            logits, logits_shape = self._matmul(
                x, lm_ssa, x_shape, [1, D, V])
        else:
            lm_head_key = "lm_head.weight"
            if self.kept_ids is not None:
                # Prune LM head weight rows to match pruned vocab.
                self.T[lm_head_key] = self.T[lm_head_key][self.kept_ids]
            logits, logits_shape = self._linear(
                x, x_shape, lm_head_key, None, "lm_head")

        # Fix output type
        out_ty = self._ty(logits_shape)
        header[3] = f"func.func @main(%tokens: {inp_ty}) -> {out_ty} {{"

        self._emit(f"return {logits} : {out_ty}")

        footer = ["}"]
        return "\n".join(header + self.lines + footer) + "\n"


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert a safetensors model to TOSA v1.0 MLIR")
    parser.add_argument("model", help="Path to .safetensors file")
    parser.add_argument("-c", "--config",
                        help="Model config JSON (layer definitions or "
                             "HuggingFace config.json)")
    parser.add_argument("-o", "--output",
                        help="Output .mlir file (default: stdout)")
    parser.add_argument("--input-shape",
                        help="Input shape as comma-separated ints "
                             "(e.g., 1,28,28,1)")
    parser.add_argument("--seq-len", type=int, default=1,
                        help="Sequence length for transformer models "
                             "(default: 1, i.e. single-token inference)")
    parser.add_argument("--prune-vocab",
                        help="Path to kept-token-ID list for vocab pruning "
                             "(one original token ID per line, in order)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print model summary to stderr")
    args = parser.parse_args()

    # Load tensors
    tensors = load_file(args.model)

    if args.verbose:
        print(f"Model: {args.model}", file=sys.stderr)
        print(f"  Tensors: {len(tensors)}", file=sys.stderr)
        total_bytes = 0
        for name, arr in tensors.items():
            size = arr.nbytes
            total_bytes += size
            print(f"  [{name}]: shape={list(arr.shape)}, "
                  f"dtype={arr.dtype}, {size} bytes", file=sys.stderr)
        print(f"  Total: {total_bytes} bytes "
              f"({total_bytes / 1024:.1f} KB)", file=sys.stderr)

    # Load or auto-detect config
    input_shape = None
    if args.input_shape:
        input_shape = [int(x) for x in args.input_shape.split(",")]

    if args.config:
        result = load_model_config(args.config)
        # Check if it's a HuggingFace config (returns dict, None)
        if isinstance(result[0], dict) and result[1] is None:
            hf_config = result[0]
            model_type = hf_config.get("model_type", "")
            if model_type in ("gpt_neo", "gpt2"):
                if args.verbose:
                    print(f"\n  Detected HuggingFace model_type={model_type}",
                          file=sys.stderr)
                    print(f"  hidden_size={hf_config['hidden_size']}, "
                          f"num_layers={hf_config['num_layers']}, "
                          f"num_heads={hf_config['num_heads']}, "
                          f"vocab_size={hf_config['vocab_size']}",
                          file=sys.stderr)
                emitter = GPTNeoTOSAEmitter(
                    tensors, hf_config, seq_len=args.seq_len,
                    prune_vocab=args.prune_vocab or "")
                mlir_text = emitter.emit()
                if args.output:
                    with open(args.output, "w") as f:
                        f.write(mlir_text)
                    print(f"Wrote {args.output}", file=sys.stderr)
                else:
                    print(mlir_text)
                return
            else:
                print(f"Unsupported model_type: {model_type}",
                      file=sys.stderr)
                sys.exit(1)
        else:
            layers, input_info = result
            if input_shape:
                input_info["shape"] = input_shape
    else:
        layers, input_info = auto_detect_layers(tensors, input_shape)
        if args.verbose:
            print(f"\n  Auto-detected {len(layers)} layers:", file=sys.stderr)
            for l in layers:
                print(f"    {l.name}: {l.type} "
                      f"(weight={l.weight_key})", file=sys.stderr)

    emitter = TOSAEmitter(tensors, layers, input_info)
    mlir_text = emitter.emit()

    if args.output:
        with open(args.output, "w") as f:
            f.write(mlir_text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(mlir_text)


if __name__ == "__main__":
    main()
