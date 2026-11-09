#!/usr/bin/env python3
"""Compare mapped Vulkan converter tensors with the original host reference.

This explicit prefix mapping follows the typed pnnx reshape/transpose chain;
shapes alone never establish identity. Tapped graphs are diagnostic only.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import struct

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_model_trace import compare
from fold3_sequences import initial_inputs
from ncnn_replay_inputs import replay_case

PREFIX = {
  "in0": ("Input", "input_img"), "in1": ("Input", "calib"),
  "Cast_0": ("Cast", "InsertedPrecisionFreeCast__to_copy_1"),
  "div_0": ("BinaryOp", "InsertedPrecisionFreeCast_div"),
  "Cast_2": ("Cast", "div"), "reshape_157": ("Reshape", "view"),
  "permute_153": ("Permute", "permute"), "reshape_158": ("Reshape", "_unsafe_view"),
  "Cast_6": ("Cast", "InsertedPrecisionFreeCast_permute_1_token_572"),
  "permute_154": ("Permute", "android_precision_8_nchw"),
  "conv_69": ("Convolution", "android_precision_8_result"),
  "permute_155": ("Permute", "node_Conv_805"),
}
ROAD_PREFIX = {
  "Cast_1": ("Cast", "_to_copy_2"),
  "linear_121": ("InnerProduct", "model/p_linear_4"), "Cast_3": ("Cast", "model/p_linear_4"),
  "silu_65": ("Swish", "model/p_silu_2"), "Cast_6": ("Cast", "model/p_silu_2"),
  "linear_122": ("InnerProduct", "model/p_linear_5"), "Cast_9": ("Cast", "model/p_unsqueeze_1"),
  "ReduceMax_14": ("Reduction", "amax"), "Cast_16": ("Cast", "_to_copy"),
  "linear_123": ("InnerProduct", "model/p_linear_2"), "Cast_19": ("Cast", "model/p_linear_2"),
  "silu_66": ("Swish", "model/p_silu_1"), "Cast_22": ("Cast", "model/p_silu_1"),
  "linear_124": ("InnerProduct", "model/p_linear_3"), "Cast_25": ("Cast", "model/p_unsqueeze"),
  "cat_4": ("Concat", "model/cat"), "sub_0": ("BinaryOp", "InsertedPrecisionFreeCast_model/sub"),
  "div_1": ("BinaryOp", "InsertedPrecisionFreeCast_model/div"), "conv_88": ("Convolution", "model/conv2d"),
  "convdw_259": ("ConvolutionDepthWise", "model/conv2d_1"), "conv_89": ("Convolution", "model/conv2d_2"),
  "pow_2": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_XCubed"),
  "mul_3": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_XCubedC0"),
  "add_4": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_XC0XCubed"),
  "mul_5": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_TanhInput"),
  "tanh_68": ("TanH", "InsertedPrecisionFreeCast__inlfunc_Gelu_ErfApprox"),
  "add_6": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_PhiApprox"),
  "mul_7": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_MultX"),
  "mul_8": ("BinaryOp", "InsertedPrecisionFreeCast_model/gelu"),
  "pow_9": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_40_XCubed"),
  "mul_10": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_40_XCubedC0"),
  "add_11": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_40_XC0XCubed"),
  "mul_12": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_40_TanhInput"),
  "tanh_69": ("TanH", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_40_ErfApprox"),
  "add_13": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_40_PhiApprox"),
  "mul_14": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_40_MultX"),
  "mul_15": ("BinaryOp", "InsertedPrecisionFreeCast_model/gelu_1"),
  "mean_251": ("Reduction", "InsertedPrecisionFreeCast_model/mean_1"),
  "Cast_290": ("Cast", "model/mean_1"), "reshape_227": ("Reshape", "model/view"),
  "mean_250": ("Reduction", "InsertedPrecisionFreeCast_model/mean"),
  "mul_152": ("BinaryOp", "InsertedPrecisionFreeCast_model/mul_12"),
  "pow_153": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_57_XCubed"),
  "mul_154": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_57_XCubedC0"),
  "add_155": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_57_XC0XCubed"),
  "mul_156": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_57_TanhInput"),
  "tanh_86": ("TanH", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_57_ErfApprox"),
  "add_157": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_57_PhiApprox"),
  "mul_158": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Gelu_token_57_MultX"),
  "mul_159": ("BinaryOp", "InsertedPrecisionFreeCast_model/gelu_18"),
  # Feature history and the residual entering the first temporal norm. The
  # pinned derived-graph hash above the trace mapping remains mandatory.
  "cat_5": ("Concat", "model/vision_outputs"), "slice_7": ("Crop", "model/hidden"),
  "reshape_228": ("Reshape", "model/hidden_t"),
  "Cast_465": ("Cast", "node__to_copy_1"), "cat_6": ("Concat", "_token_32"),
  "slice_9": ("Crop", "model/p_val_7"), "reshape_232": ("Reshape", "model/p_index"),
  "reshape_233": ("Reshape", "gemm_input_reshape_arg"),
  "reshape_234": ("Reshape", "model/p_linear"),
  "Cast_472": ("Cast", "InsertedPrecisionFreeCast_model/p_linear"),
  "silu_67": ("Swish", "InsertedPrecisionFreeCast_model/p_silu"),
  "Cast_475": ("Cast", "model/p_silu"),
  "reshape_235": ("Reshape", "gemm_input_reshape_arg_token_1"),
  "reshape_236": ("Reshape", "model/p_linear_1"),
  "Cast_479": ("Cast", "InsertedPrecisionFreeCast_model/p_linear_1"),
  "add_183": ("BinaryOp", "InsertedPrecisionFreeCast_model/p_add"),
  "add_184": ("BinaryOp", "InsertedPrecisionFreeCast_model/p_add_1"),
  "add_185": ("BinaryOp", "InsertedPrecisionFreeCast_model/p_add_2"),
  "Cast_483": ("Cast", "model/p_add_2"),
  "permute_214": ("Permute", "model/p_transpose"),
  "permute_215": ("Permute", "model/p_getitem"),
  "matmul_248": ("MatMul", "model/p_matmul"),
  "Cast_495": ("Cast", "InsertedPrecisionFreeCast_model/p_matmul"),
  "mul_186": ("BinaryOp", "InsertedPrecisionFreeCast_model/p_mul"),
  "cat_15": ("Concat", "InsertedPrecisionFreeCast_model/p_masked_fill"),
  "ReduceMax_515": ("Reduction", "InsertedPrecisionFreeCast__inlfunc_Softmax_X_ReduceMax"),
  "sub_187": ("BinaryOp", "InsertedPrecisionFreeCast__inlfunc_Softmax_X_Sub"),
  "exp_188": ("UnaryOp", "InsertedPrecisionFreeCast__inlfunc_Softmax_X_Exp"),
  "sum_255": ("Reduction", "InsertedPrecisionFreeCast__inlfunc_Softmax_X_ReduceSum"),
  "div_189": ("BinaryOp", "InsertedPrecisionFreeCast_model/p_softmax"),
  "Cast_520": ("Cast", "model/p_softmax"),
  "permute_216": ("Permute", "model/p_getitem_2"),
  "matmul_249": ("MatMul", "model/p_matmul_1"),
  "permute_217": ("Permute", "model/p_transpose_1"),
  "reshape_242": ("Reshape", "model/p_view_2"),
  "reshape_243": ("Reshape", "model/p_linear_7"),
  "Cast_529": ("Cast", "InsertedPrecisionFreeCast_model/p_linear_7"),
  "add_190": ("BinaryOp", "InsertedPrecisionFreeCast_model/p_add_3"),
  "Cast_531": ("Cast", "model/p_add_3"),

}


def weight_mapping(param, weights, original, info, required_operators=None):
  """Bind converted Conv/Gemm outputs by exact original weight and bias bytes."""
  mapping = {}
  raw_constants = {v.name: onnx.numpy_helper.to_array(v) for v in original.graph.initializer}
  for node in original.graph.node:
    if node.op_type == "Cast" and node.input[0] in raw_constants:
      dtype = next(a.i for a in node.attribute if a.name == "to")
      raw_constants[node.output[0]] = raw_constants[node.input[0]].astype(onnx.helper.tensor_dtype_to_np_dtype(dtype))
  constants = {name: value.astype(np.float32).ravel() for name, value in raw_constants.items()}
  # Prove shared subgraphs structurally, including exact initializer semantics.
  # Name or shape similarity must never alias two distinct recurrent inputs.
  provenance = {v.name: hashlib.sha256(v.type.SerializeToString() + v.name.encode()).digest() for v in original.graph.input}
  for value in original.graph.initializer:
    data = onnx.numpy_helper.to_array(value)
    provenance[value.name] = hashlib.sha256(str((data.dtype.str, data.shape)).encode() + data.tobytes()).digest()
  for node in original.graph.node:
    signature = (node.domain + ":" + node.op_type).encode() + b"".join(a.SerializeToString() for a in node.attribute)
    signature += b"".join(provenance[name] for name in node.input if name)
    for index, name in enumerate(node.output):
      provenance[name] = hashlib.sha256(signature + struct.pack("<I", index)).digest()
      if name in raw_constants:
        data = raw_constants[name]
        provenance[name] = hashlib.sha256(str((data.dtype.str, data.shape)).encode() + data.tobytes()).digest()
  nodes = [n for n in original.graph.node if n.op_type in ("Conv", "Gemm")]
  transported_weights = {}
  for node in nodes:
    if node.input[1] not in raw_constants:
      continue
    weight = raw_constants[node.input[1]]
    attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    # InnerProduct stores output-major weights. ONNX Gemm with transB=0
    # stores the same matrix input-major; compare the actual transported bytes.
    if node.op_type == "Gemm" and attributes.get("transB", 0) == 0:
      weight = weight.T
    transported_weights[node.output[0]] = weight.astype(np.float32).ravel()
  offset = 0
  for line in param:
    words = line.split()
    attributes = dict(word.split("=", 1) for word in words[4 + int(words[2]) + int(words[3]):])
    if words[0] == "MemoryData":
      offset += 4 * int(np.prod([int(attributes.get(str(i), "1")) for i in (0, 1, 11, 2)]))
      continue
    if words[0] == "LayerNorm":
      count = int(attributes["0"]) * int(attributes.get("2", "1"))
      scale = np.frombuffer(weights, "<f4", count, offset)
      bias = np.frombuffer(weights, "<f4", count, offset + 4 * count)
      offset += 8 * count
      if required_operators is not None and words[1] not in required_operators:
        continue
      matches = []
      for node in original.graph.node:
        if node.op_type != "LayerNormalization" or len(node.input) != 3:
          continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if (attrs.get("axis", -1) == -1 and attrs.get("stash_type", 1) == 1 and
            abs(attrs.get("epsilon", 1e-5) - float(attributes.get("1", "0.001"))) <= 1e-11 and
            node.input[1] in constants and node.input[2] in constants and
            np.array_equal(constants[node.input[1]], scale) and np.array_equal(constants[node.input[2]], bias)):
          matches.append(node)
      if not matches or len({provenance[n.output[0]] for n in matches}) != 1:
        raise ValueError("Ambiguous original LayerNorm provenance: " + words[1])
      mapping[words[1]] = ("LayerNorm", matches[0].output[0])
      continue
    if words[0] not in ("Convolution", "ConvolutionDepthWise", "InnerProduct"):
      continue
    linear = words[0] == "InnerProduct"
    count = int(attributes["2" if linear else "6"])
    channels = int(attributes["0"])
    bias_count = channels * int(attributes.get("1" if linear else "5", "0"))
    if struct.unpack_from("<I", weights, offset)[0] != 0:
      raise ValueError("Expected unchanged FP32 weight transport")
    offset += 4
    value = np.frombuffer(weights, "<f4", count, offset)
    offset += 4 * count
    bias = np.frombuffer(weights, "<f4", bias_count, offset)
    offset += 4 * bias_count
    if required_operators is not None and words[1] not in required_operators:
      continue
    matches = [n for n in nodes if n.output[0] in transported_weights and np.array_equal(transported_weights[n.output[0]], value) and
               ((not bias_count and len(n.input) < 3) or (len(n.input) > 2 and n.input[2] in constants and
                                                        np.array_equal(constants[n.input[2]], bias)))]
    # Conversion can deduplicate identical heads. Equal weights alone cannot
    # prove identity: their input and complete operation attributes must agree.
    semantics = {provenance[n.output[0]] for n in matches}
    if not matches or len(semantics) != 1:
      raise ValueError("Ambiguous original weight provenance: " + words[1])
    target = matches[0].output[0]
    activation = attributes.get("9", "0")
    if activation == "97":
      attrs = {a.name: onnx.helper.get_attribute_value(a) for a in matches[0].attribute}
      if (not linear or attrs.get("transA", 0) or attrs.get("transB", 0) != 1 or
          attrs.get("alpha", 1) != 1 or attrs.get("beta", 1) != 1 or
          any(info[name].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
              for name in (matches[0].input[0], target)) or
          any(raw_constants[name].dtype != np.float32 for name in matches[0].input[1:])):
        raise ValueError("Unexpected reference FP32 Gemm marker")
    elif activation in ("98", "99"):
      # The opt-in ordered kernel preserves this exact unfused Gemm. It is
      # an implementation marker, never an ONNX activation.
      if not (linear and count > 0 and channels > 0 and bias_count == channels and
              constants[matches[0].input[1]].size == count and
              info[matches[0].input[0]].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16):
        raise ValueError("Unexpected ordered-half implementation marker")
      if activation == "98":
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in matches[0].attribute}
        if (attrs.get("transA", 0) or attrs.get("transB", 0) or attrs.get("alpha", 1) != 1 or
            attrs.get("beta", 1) != 1 or info[target].type.tensor_type.elem_type != onnx.TensorProto.FLOAT16):
          raise ValueError("Unexpected ordered-half dot2 semantics")
    elif activation != "0":
      consumers = [n for n in original.graph.node if target in n.input]
      if activation not in ("1", "4") or len(consumers) != 1 or consumers[0].op_type != {"1": "Relu", "4": "Sigmoid"}[activation]:
        raise ValueError("Unsupported fused activation: " + words[1] + " " + activation)
      target = consumers[0].output[0]
    mapping[words[1]] = (words[0], target)
  if offset != len(weights):
    raise ValueError("Unmapped serialized weights")
  return mapping


def precision_boundary_comparison(actual, expected):
  """Expose sub-tolerance rounding differences without changing acceptance."""
  if expected.dtype != np.float16:
    return {}
  return {"half_rounding_disagreements": int(np.count_nonzero(actual.astype(np.float16) != expected))}


def analyze_road(path, manifest, reference):
  model = "driving_supercombo"
  source = ROOT / f"openpilot/selfdrive/modeld/models/{model}.onnx"
  metadata = json.loads((reference / "accelerators/manifest.json").read_text())["models"][model]
  source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
  if source_sha != metadata["source_sha256"] or manifest["source_graph_sha256"] != metadata["ncnn"]["sha256"]:
    raise ValueError("Road trace mapping requires the pinned original and converted graphs")
  folder = path.with_suffix("")
  if hashlib.sha256((folder / "model.param").read_bytes()).hexdigest() != manifest["sha256"]["model.param"]:
    raise ValueError("Trace graph changed")
  derived = reference / "accelerators/driving_supercombo.onnx"
  derived_bytes = derived.read_bytes()
  if hashlib.sha256(derived_bytes).hexdigest() != metadata["derived_sha256"]:
    raise ValueError("Converted diagnostic graph changed")
  graph = onnx.shape_inference.infer_shapes(onnx.load_from_string(derived_bytes))
  info = {v.name: v for v in (*graph.graph.value_info, *graph.graph.output)}
  # The probe uses the pinned shared weight file, without copying it on the host.
  weights = reference / "accelerators/ncnn/driving_supercombo.ncnn.bin"
  if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["sha256"]["model.bin"]:
    raise ValueError("Converter weights changed")
  mapping = dict(ROAD_PREFIX)
  mapping.update(weight_mapping((folder / "model.param").read_text().splitlines()[2:], weights.read_bytes(), graph, info,
                                {item["operator"] for item in manifest["tensors"]}))
  for record in manifest.get("unfused_activations", []):
    # Bind by verified weights and the unique original activation consumer.
    operation = record["operator"]
    if operation not in mapping:
      continue
    target = mapping[operation][1]
    consumers = [node for node in graph.graph.node if target in node.input]
    expected_kind = {"ReLU": "Relu", "Sigmoid": "Sigmoid"}[record["activation"]]
    if len(consumers) != 1 or consumers[0].op_type != expected_kind:
      raise ValueError("Unfused original activation changed")
    mapping[record["activation_operator"]] = (record["activation"], consumers[0].output[0])
  tensors = [item for item in manifest["tensors"] if item["operator"] in mapping]
  names = list(dict.fromkeys(mapping[item["operator"]][1] for item in tensors))
  if not names:
    raise ValueError("No mapped road tensors in this trace")
  del graph.graph.output[:]
  graph.graph.output.extend(info[name] for name in names)
  tapped = graph.SerializeToString()
  options = ort.SessionOptions()
  options.intra_op_num_threads = 2
  session = ort.InferenceSession(tapped, options, providers=["CPUExecutionProvider"])
  inputs = initial_inputs(reference, model, manifest["seed"])
  for value in graph.graph.input:
    if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
      inputs[value.name] = inputs[value.name].astype(np.float32)
  expected = dict(zip(names, session.run(None, {i.name: inputs[i.name] for i in session.get_inputs()}), strict=True))
  checks = []
  for item in tensors:
    kind, target = mapping[item["operator"]]
    value = expected[target]
    raw = (folder / item["file"]).read_bytes()
    if (kind != item["type"] or item["backend"] != "Vulkan" or item["layout"] != "NCDHW" or
        [d for d in item["shape"] if d != 1] != [d for d in value.shape if d != 1] or
        hashlib.sha256(raw).hexdigest() != item["sha256"]):
      raise ValueError(f"Trace tensor contract changed: {item['operator']}")
    actual = np.frombuffer(raw, np.float32).reshape(value.shape)
    node = next(n for n in graph.graph.node if target in n.output)
    section = [node]
    if kind in ("Convolution", "ConvolutionDepthWise", "InnerProduct") and node.op_type in ("Relu", "Sigmoid"):
      section.insert(0, next(n for n in graph.graph.node if node.input[0] in n.output))
    checks.append({"operator": item["operator"], "original_node": node.name,
                   "original_nodes": [{"index": index, "name": original.name, "type": original.op_type,
                                       "inputs": list(original.input), "outputs": list(original.output)}
                                      for index, original in enumerate(graph.graph.node) if original in section],
                   "original_tensor": target, "original_dtype": str(value.dtype),
                   "backend": item["backend"], "layout": item["layout"], "original_shape": list(value.shape),
                   "parity": compare(actual, value, allow_masked_infinity=True),
                   **precision_boundary_comparison(actual, value)})
  return {"diagnostic_only": True, "seed": manifest["seed"], "checks": checks, "original_graph_sha256": source_sha,
          "derived_graph_sha256": metadata["derived_sha256"],
          "tapped_graph_sha256": hashlib.sha256(tapped).hexdigest(),
          "first_tolerance_failure": next((c for c in checks if not c["parity"]["passed"]), None),
          "first_half_rounding_disagreement": next((c for c in checks if c.get("half_rounding_disagreements", 0)), None),
          "traced_full_model_parity": manifest["output_checks"],
          "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def analyze(path):
  manifest = json.loads(path.read_text())
  folder = path.with_suffix("")
  reference = ROOT / "android/app/build/model-reference"
  if manifest.get("model") == "driving_supercombo":
    return analyze_road(path, manifest, reference)
  graph = reference / "accelerators/dmonitoring_model.onnx"
  original = onnx.shape_inference.infer_shapes(onnx.load(graph))
  inputs = initial_inputs(reference, "dmonitoring_model", manifest["seed"])
  if manifest.get("replay_frame") is not None:
    inputs, _ = replay_case(reference, manifest["replay_frame"])
  # The conversion graph explicitly transports image bytes losslessly as FP32.
  for value in original.graph.input:
    if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
      inputs[value.name] = inputs[value.name].astype(np.float32)
  info = {v.name: v for v in (*original.graph.input, *original.graph.value_info, *original.graph.output)}
  options = ort.SessionOptions()
  options.intra_op_num_threads = 2
  mapping = dict(PREFIX)
  param = (folder / "model.param").read_text().splitlines()[2:]
  # Bind every Conv/Gemm by exact weight AND bias bytes, not by shape/order.
  weights = (reference / "accelerators/ncnn/dmonitoring_model.ncnn.bin").read_bytes()
  if hashlib.sha256(weights).hexdigest() != manifest["sha256"]["model.bin"]:
    raise ValueError("Converter weights changed")
  mapping.update(weight_mapping(param, weights, original, info))
  tanh = [line.split() for line in param if line.split()[0] == "TanH"]
  tanh_nodes = [node for node in original.graph.node if node.op_type == "Tanh"]
  if len(tanh) != len(tanh_nodes):
    raise ValueError("Converter activation chain changed")
  for words, node in zip(tanh, tanh_nodes, strict=True):
    mapping[words[1]] = (words[0], node.output[0])
  cubes = [line.split() for line in param if line.split()[0] == "BinaryOp" and "0=6" in line.split() and "2=3.0" in line.split()]
  powers = [node for node in original.graph.node if node.op_type == "Pow"]
  if len(cubes) != len(powers):
    raise ValueError("Converter cube chain changed")
  for words, node in zip(cubes, powers, strict=True):
    mapping[words[1]] = (words[0], node.output[0])
  casts = [line.split() for line in param if line.split()[0] == "Cast"]
  original_casts = [node for node in original.graph.node if node.op_type == "Cast"]
  if len(casts) != len(original_casts):
    raise ValueError("Converter Cast chain changed")
  for words, node in zip(casts, original_casts, strict=True):
    mapping[words[1]] = ("Cast", node.output[0])
  reductions = [line.split() for line in param if line.split()[0] == "Reduction"]
  means = [node for node in original.graph.node if node.op_type == "ReduceMean"]
  if len(reductions) != len(means):
    raise ValueError("Converter mean chain changed")
  for words, node in zip(reductions, means, strict=True):
    mapping[words[1]] = ("Reduction", node.output[0])
  checks = []
  for item in manifest["tensors"]:
    if item["operator"] not in mapping:
      continue
    operation, target = mapping[item["operator"]]
    if operation != item["type"]:
      raise ValueError("Converter prefix changed")
    if target == "calib":
      target = original.graph.input[1].name
    tapped = copy.deepcopy(original)
    del tapped.graph.output[:]
    tapped.graph.output.append(info[target])
    serialized = tapped.SerializeToString()
    expected = ort.InferenceSession(serialized, options, providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    tensor = folder / item["file"]
    raw = tensor.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
      raise ValueError("Trace artifact changed")
    actual = np.frombuffer(raw, np.float32)
    # ncnn shape is logical NCDHW with absent dimensions reported as one.
    if [d for d in item["shape"] if d != 1] != [d for d in expected.shape if d != 1]:
      raise ValueError(f"Prefix layout mismatch: {item['operator']}")
    actual = actual.reshape(expected.shape)
    checks.append({"operator": item["operator"], "original_tensor": target,
                   "original_shape": list(expected.shape), "original_dtype": str(expected.dtype),
                   "tapped_graph_sha256": hashlib.sha256(serialized).hexdigest(),
                   "parity": compare(actual, expected)})
  return {"diagnostic_only": True, "seed": manifest["seed"], "checks": checks,
          "original_graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
          "first_tolerance_failure": next((c for c in checks if not c["parity"]["passed"]), None),
          "untapped_full_model_parity": manifest["parity"]}


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("report", type=Path)
  args = parser.parse_args()
  result = analyze(args.report)
  output = args.report.with_name(args.report.stem + "-comparison.json")
  output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result["first_tolerance_failure"]))
