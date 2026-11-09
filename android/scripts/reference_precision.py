#!/usr/bin/env python3
"""Diagnose host precision boundaries using portable operators; keep original goldens."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]


def portable(model, promote_half_gemm=True):
  nodes = []
  types = {x.name: x.type.tensor_type.elem_type for x in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  types.update({x.name: x.data_type for x in model.graph.initializer})
  for index, node in enumerate(model.graph.node):
    inputs, output = list(node.input), node.output[0]
    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    prefix = f"android_precision_{index}"
    kind = node.op_type
    activation = attributes.pop("activation", None)
    activation_params = attributes.pop("activation_params", [])
    if kind == "NhwcFusedConv":
      nodes.append(helper.make_node("Transpose", [inputs[0]], [prefix + "_nchw"], perm=[0, 3, 1, 2]))
      inputs[0] = prefix + "_nchw"
    if kind in ("FusedConv", "NhwcFusedConv", "FusedGemm"):
      kind = "Gemm" if kind == "FusedGemm" else "Conv"
    residual = inputs.pop() if kind == "Conv" and len(inputs) == 4 else None
    # MLAS's NCHW FusedConv adds the residual before its bias pass. Splitting
    # it into a biased Conv followed by Add changes FP32 rounding, even when
    # both individual operators are accurate. NHWC fusion preloads bias and
    # retains its existing order. Preserve the proven FP32 path only.
    # Optimized host graphs omit most intermediate value_info. Conv requires
    # its input and weights to share a type, so the initializer is authoritative
    # when the internal input has no recorded type.
    conv_type = types.get(inputs[0], types.get(inputs[1])) if kind == "Conv" else None
    residual_bias = (inputs.pop() if residual and node.op_type == "FusedConv" and
                     conv_type == TensorProto.FLOAT and len(inputs) == 3 and inputs[2] else None)
    half_gemm = (promote_half_gemm and kind == "Gemm" and
                 (types.get(node.input[0]) == TensorProto.FLOAT16 or types.get(node.input[1]) == TensorProto.FLOAT16))
    if half_gemm:
      for i, value in enumerate(inputs):
        if value:
          converted = f"{prefix}_float_{i}"
          nodes.append(helper.make_node("Cast", [value], [converted], to=TensorProto.FLOAT))
          inputs[i] = converted
    intermediate = prefix + "_result" if activation or residual or node.op_type == "NhwcFusedConv" or half_gemm else output
    if kind == "QuickGelu":
      # ORT's QuickGelu is alpha*x through Sigmoid, followed by multiplication.
      if attributes.get("alpha", 1.) != 1.:
        raise ValueError("Unimplemented QuickGelu scaling")
      nodes.append(helper.make_node("Sigmoid", inputs, [prefix + "_sigmoid"]))
      nodes.append(helper.make_node("Mul", [inputs[0], prefix + "_sigmoid"], [intermediate]))
    else:
      nodes.append(helper.make_node(kind, inputs, list(node.output) if intermediate == output else [intermediate], name=node.name, **attributes))
    if residual:
      summed = prefix + "_residual" if activation or residual_bias else output
      nodes.append(helper.make_node("Add", [intermediate, residual], [summed]))
      intermediate = summed
    if residual_bias:
      shape, broadcast = prefix + "_bias_shape", prefix + "_bias_nchw"
      nodes.append(helper.make_node("Constant", [], [shape],
                                    value=onnx.numpy_helper.from_array(np.array([1, -1, 1, 1], dtype=np.int64))))
      nodes.append(helper.make_node("Reshape", [residual_bias, shape], [broadcast]))
      biased = prefix + "_biased" if activation else output
      nodes.append(helper.make_node("Add", [intermediate, broadcast], [biased]))
      intermediate = biased
    if activation:
      activation = activation.decode() if isinstance(activation, bytes) else activation
      activated = prefix + "_activated" if node.op_type == "NhwcFusedConv" or half_gemm else output
      if activation not in ("Relu", "Tanh", "Sigmoid") or activation_params:
        raise ValueError(f"Unimplemented fusion {activation}")
      nodes.append(helper.make_node(activation, [intermediate], [activated]))
      intermediate = activated
    if half_gemm:
      nodes.append(helper.make_node("Cast", [intermediate], [output], to=TensorProto.FLOAT16))
    if node.op_type == "NhwcFusedConv":
      nodes.append(helper.make_node("Transpose", [intermediate], [output], perm=[0, 2, 3, 1]))
  del model.graph.node[:]
  model.graph.node.extend(nodes)
  # Inferred host internal value_info may describe removed fused tensors.
  del model.graph.value_info[:]
  onnx.checker.check_model(model)
  return model


def main():
  destination = ROOT / "android/app/build/model-reference/reference-precision"
  destination.mkdir(exist_ok=True)
  reference = destination.parent
  manifest = {"diagnostic": True, "vehicle_tx_enabled": False, "models": {}}
  for name in ("dmonitoring_model", "driving_supercombo"):
    original = ROOT / f"openpilot/selfdrive/modeld/models/{name}.onnx"
    optimized = destination / f"{name}-host.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.optimized_model_filepath = str(optimized)
    ort.InferenceSession(str(original), options, providers=["CPUExecutionProvider"])
    path = destination / f"{name}.onnx"
    onnx.save(portable(onnx.load(optimized)), path)
    options.optimized_model_filepath = ""
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    with np.load(reference / f"{name}-inputs.npz") as source:
      inputs = dict(source)
    actual = dict(zip((s.name for s in session.get_outputs()), session.run(None, inputs), strict=True))
    with np.load(reference / f"{name}-outputs.npz") as expected:
      checks = {key: {"passed": bool(np.allclose(actual[key], expected[key], rtol=1e-3, atol=1e-4)),
                       "max_abs_error": float(np.max(np.abs(actual[key].astype(np.float64) - expected[key].astype(np.float64))))}
                for key in expected.files}
    manifest["models"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(), "original_reference": checks}
    print(name, checks, flush=True)
  (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
  main()
