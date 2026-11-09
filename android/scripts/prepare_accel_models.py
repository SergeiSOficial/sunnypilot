#!/usr/bin/env python3
"""Local, separately hashed standard-ONNX exports and pinned GPU conversions."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort

from reference_precision import portable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_model_trace import compare
from fold3_sequences import initial_inputs, validate


def digest(path):
  with Path(path).open("rb") as source:
    return hashlib.file_digest(source, "sha256").hexdigest()


def ncnn_contract(path, inputs, outputs):
  """Read pnnx's explicit axes without executing generated Python code."""
  text = Path(path).read_text()
  in_axes = dict(re.findall(r'ex\.input\("(in\d+)", ncnn\.Mat\(\w+\.numpy\(\), batch_index=(-?\d+)\)', text))
  out_axes = dict(re.findall(r'(out\d+)\.numpy\(batch_index=(-?\d+)\)', text))
  result = {}
  for kind, axes, tensors, prefix in (("input", in_axes, inputs, "in"), ("output", out_axes, outputs, "out")):
    if set(axes) != {f"{prefix}{i}" for i in range(len(tensors))}:
      raise ValueError(f"Missing/ambiguous ncnn {kind} batch metadata")
    parsed = {}
    for i, (name, tensor) in enumerate(tensors.items()):
      axis = int(axes[f"{prefix}{i}"])
      if axis == 233:
        axis = None  # pnnx/ncnn's batch_index_none sentinel
      elif not -len(tensor["shape"]) <= axis < len(tensor["shape"]):
        raise ValueError(f"Invalid ncnn batch axis: {name}")
      else:
        axis %= len(tensor["shape"])
      parsed[name] = axis
    result[kind + "_batch_axes"] = parsed
  return result


def ncnn_conversion(folder, name, inputs, outputs):
  graph = folder / f"{name}.ncnn.param"
  entry = {"exit_code": 0, "status": "converted", "sha256": digest(graph),
           "weights_sha256": digest(graph.with_suffix(".bin")),
           **ncnn_contract(folder / (name + "_ncnn.py"), inputs, outputs)}
  cmake = next((ROOT / "android/build/native/accelerators/ncnn-source").glob("*/src/CMakeLists.txt"))
  supported = set(re.findall(r"ncnn_add_layer\((\w+)", cmake.read_text()))
  layers = {line.split()[0] for line in graph.read_text().splitlines()[2:] if line.strip()}
  entry["unsupported_operators"] = sorted(layers - supported)
  vulkan = cmake.parent / "layer/vulkan"
  entry["operators_without_vulkan_source"] = sorted(
    layer for layer in layers & supported - {"Input"} if not (vulkan / f"{layer.lower()}_vulkan.cpp").is_file()
  )
  entry["unsupported_inputs"] = [key for key, tensor in inputs.items()
    if not 1 <= len(tensor["shape"]) - (entry["input_batch_axes"][key] is not None) <= 4]
  if entry["unsupported_operators"] or entry["unsupported_inputs"]:
    entry.update(status="blocked", error="pnnx left unsupported operators or unrepresentable tensor ranks")
  return entry


def repair_ncnn_road_reductions(graph, typed_graph):
  """Lower scalar-axis road reductions that pnnx did not encode."""
  path = Path(graph)
  lines = path.read_text().splitlines()
  typed = {words[1]: words for line in Path(typed_graph).read_text().splitlines()[2:]
           if (words := line.split())}
  scalar_reductions = {
    "norm_252": ("torch.norm_260", "torch.norm", 8, ("(1,512)f32", "(1,1)f32")),
    "norm_253": ("torch.norm_261", "torch.norm", 8, ("(1,512)f32", "(1,1)f32")),
    "sum_255": ("torch.sum_2", "torch.sum", 0, ("(1,8,9,9)f32", "(1,8,9,1)f32")),
  }
  repairs = []
  maxima = 0
  scalar_names = set()
  for index, line in enumerate(lines[2:], 2):
    words = line.split()
    if words[0:1] == ["Reduction"] and "-23303=0" in words[6:]:
      name = words[1]
      if name not in scalar_reductions or name in scalar_names:
        raise ValueError("Unexpected empty road reduction axis: " + name)
      typed_name, operation, code, shapes = scalar_reductions[name]
      match = typed.get(typed_name, [])
      attributes = set(match[6:])
      if (match[:1] != [operation] or match[2:4] != ["1", "1"] or
          not {"dim=-1", "keepdim=True"} <= attributes or
          (code == 8 and "p=2" not in attributes) or
          tuple(word.split("=", 1)[1] for word in match if word.startswith("#")) != shapes or
          words[2:4] != ["1", "1"] or set(words[6:]) != {f"0={code}", "1=0", "-23303=0", "4=1", "5=1"}):
        raise ValueError("Road scalar reduction contract changed: " + name)
      words[words.index("-23303=0")] = "-23303=1,-1"
      lines[index] = " ".join(words)
      repairs.append(name)
      scalar_names.add(name)
      continue
    if not words or words[0] != "ReduceMax":
      continue
    _, name, inputs, outputs, source, target = words
    match = typed.get(name, [])
    contract = match[6:9]
    shapes = [word.split("=", 1)[1] for word in match[9:] if word.startswith("#")]
    if (contract, shapes) == (["axes=1", "keepdims=0", "noop_with_empty_axes=0"],
                              ["(25,4,1,8)f32", "(25,1,8)f32"]):
      producer = next((row.split() for row in lines[2:index]
                       if row.split()[0:1] == ["Reshape"] and row.split()[5:6] == [source]), [])
      if set(producer[6:]) != {"0=8", "1=1", "11=4", "12=1", "13=233", "2=25"}:
        raise ValueError("Road ReduceMax input layout changed")
      axis, keep = 1, 0
    elif (contract, shapes) == (["axes=-1", "keepdims=1", "noop_with_empty_axes=0"],
                                ["(1,8,9,9)f32", "(1,8,9,1)f32"]):
      axis, keep = -1, 1
    else:
      raise ValueError("Unrecognized road ReduceMax contract: " + name)
    if (inputs, outputs) != ("1", "1"):
      raise ValueError("Unexpected road ReduceMax arity")
    lines[index] = f"Reduction {name} 1 1 {source} {target} 0=4 1=0 -23303=1,{axis} 4={keep} 5=1"
    repairs.append(name)
    maxima += 1
  if maxima != 2 or scalar_names != scalar_reductions.keys():
    raise ValueError("Incomplete road reduction repair")
  path.write_text("\n".join(lines) + "\n")
  return repairs


def repair_ncnn_road_rank_changes(graph, typed_graph):
  """Use Vulkan Reshape for the pinned singleton rank edits."""
  path = Path(graph)
  lines = path.read_text().splitlines()
  typed = [line.split() for line in Path(typed_graph).read_text().splitlines()[2:]]
  unsqueezes = [row for row in typed if row[0] == "torch.unsqueeze"]
  expected = [
    ("dim=1", "(1,512)f16", "(1,1,512)f16"),
    ("dim=1", "(1,512)f16", "(1,1,512)f16"),
    ("dim=0", "(1,512)f32", "(1,1,512)f32"),
  ]
  contracts = [(row[6], *(item.split("=", 1)[1] for item in row if item.startswith("#")))
               for row in unsqueezes]
  if contracts != expected:
    raise ValueError("Road ExpandDims typed layout changed")
  selects = [row for row in typed if row[0] == "Tensor.select" and row[6:8] == ["dim=1", "index=8"]
             and [item.split("=", 1)[1] for item in row if item.startswith("#")] ==
             ["(1,9,512)f16", "(1,512)f16"]]
  if len(selects) != 1:
    raise ValueError("Road Squeeze typed layout changed")
  repairs = []
  for index, line in enumerate(lines[2:], 2):
    words = line.split()
    if words[0:1] not in (["ExpandDims"], ["Squeeze"]):
      continue
    if words[2:4] != ["1", "1"] or words[6:] != ["-23303=1,0"]:
      raise ValueError("Road ncnn singleton axis changed")
    # The native axis is already adjusted for the converter's removed
    # batch dimension. Axis zero adds or removes the outer singleton.
    shape = ["0=512", "1=1"] if words[0] == "ExpandDims" else ["0=512"]
    lines[index] = " ".join(["Reshape", *words[1:6], *shape])
    repairs.append(words[0])
  if repairs != ["ExpandDims", "ExpandDims", "Squeeze", "ExpandDims"]:
    raise ValueError("Road ncnn singleton edit sequence changed")
  path.write_text("\n".join(lines) + "\n")
  return repairs


def repair_ncnn_road_clips(graph, typed_graph):
  """Encode the pinned zero lower bounds in ncnn's Clip parameters."""
  path = Path(graph)
  magic, counts, *lines = path.read_text().splitlines()
  typed = [line.split() for line in Path(typed_graph).read_text().splitlines()[2:]]
  scalar, = [row for row in typed if row[0] == "pnnx.Expression" and row[2:4] == ["0", "1"]
             and row[5:] == ["expr=0.0"]]
  clips = {row[1] for row in typed if row[0] == "Clip" and row[2:4] == ["2", "1"]
           and row[5] == scalar[4] and len(row) == 9 and all(x.endswith("=(1,1)f16") for x in row[7:])}
  if len(clips) != 2:
    raise ValueError("Road Clip scalar consumers changed")
  rows = [line.split() for line in lines]
  producer, = [row for row in rows if row[:4] == scalar[:4] and len(row) == 5]
  fanout, = [row for row in rows if row[0] == "Split" and row[2:5] == ["1", "2", producer[4]] and len(row) == 7]
  removed = set(fanout[4:])
  rewritten, matched = [], set()
  for row in rows:
    if row in (producer, fanout):
      continue
    if row[1] in clips and row[0] == "Clip" and row[2:4] == ["2", "1"] and len(row) == 7 and row[5] in removed:
      matched.add(row[1])
      row = [*row[:2], "1", "1", row[4], row[6], "0=0"]
    if removed.intersection(row[4:4 + int(row[2]) + int(row[3])]):
      raise ValueError("Unexpected use of road Clip scalar")
    rewritten.append(" ".join(row))
  if matched != clips:
    raise ValueError("Road Clip conversion changed")
  layer_count, blob_count = map(int, counts.split())
  path.write_text("\n".join([magic, f"{layer_count - 2} {blob_count - 3}", *rewritten]) + "\n")
  return sorted(clips)


def mark_ncnn_half_gemv(graph):
  """Diagnostic repair of the traced driver projection; reject changed graphs."""
  path = Path(graph)
  lines = path.read_text().splitlines()
  matches = [i for i, line in enumerate(lines[2:], 2) if len(line.split()) > 1 and line.split()[1] == "linear_100"]
  if len(matches) != 1:
    raise ValueError("Missing or ambiguous traced driver projection")
  index = matches[0]
  words = lines[index].split()
  params = dict(word.split("=", 1) for word in words[6:])
  if (words[0:4] != ["InnerProduct", "linear_100", "1", "1"] or
      params.get("0") != "512" or params.get("2") != str(1280 * 512) or
      params.get("1") != "1" or params.get("9", "0") != "0"):
    raise ValueError("Traced projection dimensions, bias or activation changed")
  words = [word for word in words if not word.startswith("9=")]
  lines[index] = " ".join([*words, "9=99"])
  path.write_text("\n".join(lines) + "\n")
  return {"operator": "linear_100", "k": 1280, "n": 512, "activation_marker": 99,
          "arithmetic": "original-arm64-half-gemv", "diagnostic_only": True}


def mark_ncnn_half_heads(graph, original_graph, *, original_nodes=None):
  """Bind every repaired half Gemm to original weight/bias bytes and types.

  Fused FP32 heads are deliberately left on their existing path. Validate all
  serialized weights and every half Gemm before editing the converted graph.
  Split aliases may duplicate a head; their input identity must still agree.
  """
  import struct
  path = Path(graph)
  original = onnx.shape_inference.infer_shapes(onnx.load(original_graph))
  constants = {v.name: onnx.numpy_helper.to_array(v) for v in original.graph.initializer}
  info = {v.name: v for v in (*original.graph.input, *original.graph.value_info, *original.graph.output)}
  half_nodes = {}
  for node in original.graph.node:
    if original_nodes is not None and node.name not in original_nodes:
      continue
    if node.op_type != "Gemm" or constants[node.input[1]].dtype != np.float16:
      continue
    attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    if (attributes.get("alpha", 1) != 1 or attributes.get("beta", 1) != 1 or
        attributes.get("transA", 0) != 0 or attributes.get("transB", 0) != 1 or
        len(node.input) != 3 or any(info[name].type.tensor_type.elem_type != onnx.TensorProto.FLOAT16
                                   for name in (node.input[0], node.output[0]))):
      raise ValueError("Unsupported original half Gemm semantics: " + node.name)
    weight, bias = (constants[name].astype(np.float32) for name in node.input[1:])
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
      raise ValueError("Unsupported original half Gemm parameters")
    half_nodes[node.name] = (weight, bias)
  if not half_nodes or (original_nodes is not None and set(half_nodes) != set(original_nodes)):
    raise ValueError("No original half Gemm provenance")
  lines = path.read_text().splitlines()
  weights = path.with_suffix(".bin").read_bytes()
  offset, found, repairs = 0, {}, []
  aliases = {}
  for index, line in enumerate(lines[2:], 2):
    words = line.split()
    if not words:
      continue
    if words[0] == "Split":
      source = aliases.get(words[4], words[4])
      aliases.update({name: source for name in words[5:5 + int(words[3])]})
    attributes = dict(word.split("=", 1) for word in words[4 + int(words[2]) + int(words[3]):])
    if words[0] == "MemoryData":
      offset += 4 * int(np.prod([int(attributes.get(str(i), "1")) for i in (0, 1, 11, 2)]))
      continue
    if words[0] == "LayerNorm":
      offset += 8 * int(attributes["0"]) * int(attributes.get("2", "1"))
      continue
    if words[0] not in ("Convolution", "ConvolutionDepthWise", "InnerProduct"):
      continue
    linear = words[0] == "InnerProduct"
    count, channels = int(attributes["2" if linear else "6"]), int(attributes["0"])
    bias_count = channels * int(attributes.get("1" if linear else "5", "0"))
    if struct.unpack_from("<I", weights, offset)[0] != 0:
      raise ValueError("Expected lossless FP32 weight transport")
    offset += 4
    value = np.frombuffer(weights, "<f4", count, offset)
    offset += 4 * count
    bias = np.frombuffer(weights, "<f4", bias_count, offset)
    offset += 4 * bias_count
    if not linear:
      continue
    matches = [name for name, (w, b) in half_nodes.items()
               if w.size == count and b.size == bias_count and np.array_equal(w.ravel(), value) and np.array_equal(b, bias)]
    if not matches:
      if attributes.get("9") == "99":
        raise ValueError("Ordered marker without original half Gemm provenance")
      continue
    if len(matches) != 1 or words[2:4] != ["1", "1"] or attributes.get("9", "0") not in ("0", "99"):
      raise ValueError("Ambiguous or fused original half Gemm")
    name = matches[0]
    input_identity = aliases.get(words[4], words[4])
    if name in found and found[name] != input_identity:
      raise ValueError("Repeated half Gemm has a different converted input")
    n, k = half_nodes[name][0].shape
    if channels != n:
      raise ValueError("Converted half Gemm dimensions changed")
    found[name] = input_identity
    lines[index] = " ".join([*(word for word in words if not word.startswith("9=")), "9=99"])
    repairs.append({"operator": words[1], "original_node": name, "converted_input": words[4],
                    "input_identity": input_identity, "k": k, "n": n, "activation_marker": 99})
  if set(found) != set(half_nodes) or offset != len(weights):
    raise ValueError("Incomplete original half Gemm provenance")
  path.write_text("\n".join(lines) + "\n")
  return {"operators": repairs, "original_graph_sha256": digest(original_graph),
          "arithmetic": "original-arm64-half-gemv", "diagnostic_only": True}


def mark_ncnn_float_gemm(graph, original_graph, original_nodes):
  """Select bias-last FP32 reduction only with original weight/type provenance."""
  from compare_ncnn_trace import weight_mapping
  requested = set(original_nodes)
  if not requested or len(requested) != len(original_nodes):
    raise ValueError("Reference FP32 Gemm nodes must be nonempty and unique")
  original = onnx.shape_inference.infer_shapes(onnx.load(original_graph))
  info = {v.name: v for v in (*original.graph.input, *original.graph.value_info, *original.graph.output)}
  nodes = {n.output[0]: n for n in original.graph.node}
  path = Path(graph)
  lines = path.read_text().splitlines()
  required = {line.split()[1] for line in lines[2:] if line.split()[0] == "InnerProduct"}
  mapping = weight_mapping(lines[2:], path.with_suffix(".bin").read_bytes(), original, info, required)
  rewritten, edits, found = lines[:2], [], set()
  for line in lines[2:]:
    words = line.split()
    target = mapping.get(words[1]) if words[0] == "InnerProduct" else None
    node = nodes[target[1]] if target else None
    if node is not None and node.name in requested:
      attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
      params = dict(word.split("=", 1) for word in words[6:])
      n = int(params["0"])
      k = int(params["2"]) // n if n else 0
      if (node.op_type != "Gemm" or len(node.input) != 3 or attrs.get("alpha", 1) != 1 or
          attrs.get("beta", 1) != 1 or attrs.get("transA", 0) != 0 or attrs.get("transB", 0) != 1 or
          words[2:4] != ["1", "1"] or n <= 0 or k <= 0 or k % 4 or
          int(params["2"]) != n * k or params.get("1") != "1" or params.get("9", "0") not in ("0", "97") or
          any(info[name].type.tensor_type.elem_type != onnx.TensorProto.FLOAT for name in (node.input[0], node.output[0]))):
        raise ValueError("Unsupported reference FP32 Gemm contract: " + node.name)
      constants = {v.name: onnx.numpy_helper.to_array(v) for v in original.graph.initializer}
      for operation in original.graph.node:
        if operation.op_type == "Cast" and operation.input[0] in constants:
          dtype = next(a.i for a in operation.attribute if a.name == "to")
          constants[operation.output[0]] = constants[operation.input[0]].astype(onnx.helper.tensor_dtype_to_np_dtype(dtype))
      if any(name not in constants or constants[name].dtype != np.float32 for name in node.input[1:]):
        raise ValueError("Reference FP32 Gemm requires original FP32 weights and bias")
      line = " ".join([*(word for word in words if not word.startswith("9=")), "9=97"])
      found.add(node.name)
      edits.append({"operator": words[1], "original_node": node.name, "k": k, "n": n,
                    "activation_marker": 97, "arithmetic": "ordered-fma-bias-last-v1"})
    rewritten.append(line)
  if found != requested:
    raise ValueError("Unmapped reference FP32 Gemm nodes: " + ", ".join(sorted(requested - found)))
  path.write_text("\n".join(rewritten) + "\n")
  return {"diagnostic_only": True, "qualified": False, "requires_reference_float_gemm": 1,
          "original_sha256": digest(original_graph), "operators": edits}


def mark_ncnn_reference_layernorm(graph, original_graph, original_nodes):
  """Select reference arithmetic only for provenance-verified FP16 norms."""
  from compare_ncnn_trace import weight_mapping
  requested = set(original_nodes)
  if not requested or len(requested) != len(original_nodes):
    raise ValueError("Reference LayerNorm nodes must be nonempty and unique")
  original = onnx.shape_inference.infer_shapes(onnx.load(original_graph))
  info = {v.name: v for v in (*original.graph.input, *original.graph.value_info, *original.graph.output)}
  nodes = {n.output[0]: n for n in original.graph.node}
  path = Path(graph)
  lines = path.read_text().splitlines()
  required = {line.split()[1] for line in lines[2:] if line.split()[0] == "LayerNorm"}
  mapping = weight_mapping(lines[2:], path.with_suffix(".bin").read_bytes(), original, info, required)
  rewritten, edits, found = lines[:2], [], set()
  for line in lines[2:]:
    words = line.split()
    target = mapping.get(words[1]) if words[0] == "LayerNorm" else None
    node = nodes[target[1]] if target else None
    if node is not None and node.name in requested:
      attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
      source, output = (info[name].type.tensor_type for name in (node.input[0], node.output[0]))
      params = dict(word.split("=", 1) for word in words[6:])
      if (words[2:4] != ["1", "1"] or attrs.get("axis", -1) != -1 or attrs.get("stash_type", 1) != 1 or
          source.elem_type != onnx.TensorProto.FLOAT16 or output.elem_type != onnx.TensorProto.FLOAT16 or
          source.shape.dim[-1].dim_value != int(params["0"]) or params.get("2", "1") != "1" or
          params.get("29", "0") not in ("0", "1")):
        raise ValueError("Unsupported reference LayerNorm contract: " + node.name)
      words = [word for word in words if not word.startswith("29=")]
      words.append("29=1")
      found.add(node.name)
      edits.append({"operator": words[1], "original_node": node.name, "original_tensor": node.output[0],
                    "dtype": "float16", "statistics_dtype": "float32", "axis": -1,
                    "epsilon": attrs.get("epsilon", 1e-5), "arithmetic": "reference-welford-v1"})
      line = " ".join(words)
    rewritten.append(line)
  if found != requested:
    raise ValueError("Unmapped reference LayerNorm nodes: " + ", ".join(sorted(requested - found)))
  path.write_text("\n".join(rewritten) + "\n")
  return {"diagnostic_only": True, "qualified": False, "requires_reference_layernorm": 1,
          "original_sha256": digest(original_graph), "operators": edits}


def restore_ncnn_half_outputs(graph, original_graph, mode="round"):
  """Restore implicit half outputs erased by lossless FP32 weight transport.

  Only exact original weight/affine bytes establish a dtype contract. This
  diagnostic either restores output rounding or selects the original pairwise
  FP32 accumulation for NoTrans half GEMM. Neither mode qualifies a model.
  """
  import struct
  if mode not in ("round", "dot2"):
    raise ValueError("Unknown half arithmetic repair")
  original = onnx.shape_inference.infer_shapes(onnx.load(original_graph))
  info = {v.name: v for v in (*original.graph.input, *original.graph.value_info, *original.graph.output)}
  constants = {v.name: onnx.numpy_helper.to_array(v) for v in original.graph.initializer}
  for node in original.graph.node:
    if node.op_type == "Cast" and node.input[0] in constants:
      kind = next(a.i for a in node.attribute if a.name == "to")
      constants[node.output[0]] = constants[node.input[0]].astype(onnx.helper.tensor_dtype_to_np_dtype(kind))
  candidates = {"InnerProduct": [], "LayerNorm": []}
  for node in original.graph.node:
    if node.op_type not in ("Gemm", "LayerNormalization") or len(node.input) != 3:
      continue
    if any(name not in constants for name in node.input[1:]):
      continue
    weight, bias = (constants[name] for name in node.input[1:])
    if mode == "dot2" and (weight.dtype != np.float16 or bias.dtype != np.float16):
      continue
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    if node.op_type == "Gemm":
      if attrs.get("transA", 0) or attrs.get("alpha", 1) != 1 or attrs.get("beta", 1) != 1 or weight.ndim != 2:
        continue
      if not attrs.get("transB", 0):
        weight = weight.T
      if bias.shape != (weight.shape[0],):
        continue
      kind = "InnerProduct"
    else:
      if attrs.get("axis", -1) != -1 or attrs.get("stash_type", 1) != 1 or weight.ndim != 1 or bias.shape != weight.shape:
        continue
      kind = "LayerNorm"
    candidates[kind].append((node, weight.astype(np.float32).ravel(), bias.astype(np.float32).ravel(), attrs))
  path = Path(graph)
  lines, data = path.read_text().splitlines(), path.with_suffix(".bin").read_bytes()
  offset, edits, rewritten = 0, [], lines[:2]
  used = {word for line in lines[2:] for word in line.split()[1:4 + sum(map(int, line.split()[2:4]))]}
  for line in lines[2:]:
    words = line.split()
    kind = words[0]
    attrs = dict(word.split("=", 1) for word in words[4 + int(words[2]) + int(words[3]):])
    matches = []
    if kind == "MemoryData":
      offset += 4 * int(np.prod([int(attrs.get(str(i), "1")) for i in (0, 1, 11, 2)]))
    elif kind in ("Convolution", "ConvolutionDepthWise", "InnerProduct", "LayerNorm"):
      if kind == "LayerNorm":
        count = int(attrs["0"]) * int(attrs.get("2", "1"))
        bias_count = count
      else:
        linear = kind == "InnerProduct"
        count = int(attrs["2" if linear else "6"])
        bias_count = int(attrs["0"]) * int(attrs.get("1" if linear else "5", "0"))
        if struct.unpack_from("<I", data, offset)[0] != 0:
          raise ValueError("Expected lossless FP32 weight transport")
        offset += 4
      weight = np.frombuffer(data, "<f4", count, offset)
      offset += 4 * count
      bias = np.frombuffer(data, "<f4", bias_count, offset)
      offset += 4 * bias_count
      matches = [(n, a) for n, w, b, a in candidates.get(kind, [])
                 if np.array_equal(w, weight) and np.array_equal(b, bias)]
    if not matches or not any(info[n.output[0]].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16 for n, _ in matches):
      rewritten.append(line)
      continue
    if (words[2:4] != ["1", "1"] or
        any(info[n.output[0]].type.tensor_type.elem_type != onnx.TensorProto.FLOAT16 for n, _ in matches)):
      raise ValueError("Ambiguous original half output")
    if mode == "dot2":
      if kind != "InnerProduct" or all(a.get("transB", 0) for _, a in matches):
        rewritten.append(line)
        continue
      if any(a.get("transB", 0) or info[n.input[0]].type.tensor_type.elem_type != onnx.TensorProto.FLOAT16
             for n, a in matches):
        raise ValueError("Ambiguous original half dot2 arithmetic")
    if kind == "InnerProduct" and attrs.get("9", "0") not in ("0", "98", "99"):
      raise ValueError("Fused activation crosses original half output")
    if kind == "LayerNorm" and any(abs(a.get("epsilon", 1e-5) - float(attrs.get("1", "0.001"))) > 1e-11 for _, a in matches):
      raise ValueError("LayerNorm epsilon changed")
    if attrs.get("9") in ("98", "99"):
      rewritten.append(line)  # the ordered GEMV already rounds its result
      continue
    if mode == "dot2":
      words = [word for word in words if not word.startswith("9=")]
      words.append("9=98")
      rewritten.append(" ".join(words))
      edits.append({"operator": words[1], "original_nodes": [n.name for n, _ in matches],
                    "dtype": "float16", "arithmetic": "original-arm64-half-dot2", "activation_marker": 98})
      continue
    temporary, cast = "fold3_unrounded_" + words[1], "fold3_round_" + words[1]
    if words[5] == temporary:
      boundary = [row.split() for row in lines[2:] if row.startswith("Cast " + cast + " ")]
      if (len(boundary) != 1 or boundary[0][2:5] != ["1", "1", temporary] or
          boundary[0][6:] != ["0=1", "1=2"]):
        raise ValueError("Half output repair boundary changed")
      rewritten.append(line)
      continue
    if temporary in used or cast in used:
      raise ValueError("Half output repair name collision")
    used.update((temporary, cast))
    output = words[5]
    words[5] = temporary
    rewritten.extend((" ".join(words), f"Cast {cast} 1 1 {temporary} {output} 0=1 1=2"))
    edits.append({"operator": words[1], "original_nodes": [n.name for n, _ in matches],
                  "dtype": "float16", "output": output, "rounding_operator": cast})
  if offset != len(data):
    raise ValueError("Incomplete serialized weight provenance")
  counts = list(map(int, lines[1].split()))
  rewritten[1] = " ".join(str(value + (len(edits) if mode == "round" else 0)) for value in counts)
  path.write_text("\n".join(rewritten) + "\n")
  return {"diagnostic_only": True, "qualified": False, "original_sha256": digest(original_graph), "operators": edits}


def repair_ncnn_casts(graph, typed_graph):
  """Recover omitted Cast parameters from pnnx's typed, same-name operations.

  The adapter transports external bytes losslessly as FP32. Internal half casts
  still require the patched Vulkan round-to-half shader, not global FP16 math.
  Validate the complete edit before writing anything.
  """
  typed = {}
  external = set()
  byte_transport = set()
  for line in Path(typed_graph).read_text().splitlines()[2:]:
    words = line.split()
    if not words:
      continue
    input_count, output_count = map(int, words[2:4])
    inputs = words[4:4 + input_count]
    outputs = words[4 + input_count:4 + input_count + output_count]
    types = dict(re.findall(r"#(\S+)=\([^)]*\)(\w+)", line))
    if words[0] == "pnnx.Input":
      external.update(outputs)
      byte_transport.update(key for key in outputs if types.get(key) == "u8")
    elif (words[0] in {"Tensor.slice", "Tensor.reshape", "torch.cat", "torch.split"} and
          inputs and all(key in byte_transport for key in inputs)):
      byte_transport.update(key for key in outputs if types.get(key) == "u8")
    if words[0] != "Cast":
      continue
    if words[1] in typed or words[2:4] != ["1", "1"]:
      raise ValueError("Ambiguous pnnx Cast identity")
    source, target = words[4:6]
    if source not in types or target not in types:
      raise ValueError("Missing pnnx Cast dtype")
    pair = (types[source], types[target])
    if pair == ("u8", "f32") and source in external:
      parameters = (1, 1)  # exact external-byte transport, not an internal cast
    elif pair == ("u8", "f16") and source in byte_transport:
      parameters = (1, 1)  # all transported bytes are exactly representable in FP16
    elif pair in (("f32", "f16"), ("f16", "f32"), ("f32", "f32")):
      parameters = tuple({"f32": 1, "f16": 2}[kind] for kind in pair)
    else:
      raise ValueError(f"Unsupported ncnn Cast semantics: {pair}")
    typed[words[1]] = parameters
  lines, edits = [], []
  for line in Path(graph).read_text().splitlines():
    words = line.split()
    if words and words[0] == "Cast":
      if words[2:4] != ["1", "1"] or words[1] not in typed:
        raise ValueError("ncnn Cast lacks exact typed provenance")
      pair = typed[words[1]]
      existing = dict(word.split("=", 1) for word in words[6:] if "=" in word)
      if set(existing) - {"0", "1"} or any(existing.get(str(i), str(value)) != str(value) for i, value in enumerate(pair)):
        raise ValueError("Conflicting ncnn Cast parameters")
      line = " ".join([*words[:6], f"0={pair[0]}", f"1={pair[1]}"])
      edits.append({"operation": words[1], "type_from": pair[0], "type_to": pair[1]})
    lines.append(line)
  Path(graph).write_text("\n".join(lines) + "\n")
  return edits


def unused_inputs(model):
  """Only original inputs unreachable from every original output may vanish."""
  required = {item.name for item in model.graph.output}
  for node in reversed(model.graph.node):
    if required.intersection(node.output):
      required.update(node.input)
  return [item.name for item in model.graph.input if item.name not in required]


def float_image_transport(model):
  """Lossless byte transport only; retain every original half compute boundary."""
  model = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
  types = {v.name: v.type.tensor_type.elem_type for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  types.update({v.name: v.data_type for v in model.graph.initializer})
  moves = {"Reshape", "Transpose", "Concat", "Split", "Unsqueeze", "Squeeze", "Slice", "Gather", "Expand", "Identity"}
  for node in model.graph.node:
    byte_inputs = [i for i, name in enumerate(node.input) if types.get(name) == onnx.TensorProto.UINT8]
    byte_outputs = [name for name in node.output if types.get(name) == onnx.TensorProto.UINT8]
    if (byte_inputs or byte_outputs) and (node.domain or node.op_type not in moves | {"Cast"}):
      raise ValueError(f"Non-transport uint8 operation: {node.name}:{node.op_type}")
    if node.op_type in moves and any(i != 0 for i in byte_inputs) and node.op_type != "Concat":
      raise ValueError("Cannot promote byte index/shape inputs")
    if node.op_type == "Cast" and byte_outputs:
      raise ValueError("Cannot move a numeric cast-to-byte boundary")
  for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
    if value.type.tensor_type.elem_type == onnx.TensorProto.UINT8:
      value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
  for i, tensor in enumerate(model.graph.initializer):
    if tensor.data_type == onnx.TensorProto.UINT8:
      model.graph.initializer[i].CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(tensor).astype(np.float32), tensor.name))
  onnx.checker.check_model(model)
  return model


def normalize_constant_gather_nd(model):
  """MNN's OpenCL GatherND offset loop does not wrap negative indices.

  Canonicalize only statically proven, in-range constant indices. Clone the
  initializer because another consumer can have a different indexed shape.
  This changes neither the original ONNX nor any dynamic-index operation.
  """
  model = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
  shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  constants = {v.name: v for v in model.graph.initializer}
  shapes.update({v.name: list(v.dims) for v in model.graph.initializer})
  used = {name for node in model.graph.node for name in (*node.input, *node.output)}
  for index, node in enumerate(model.graph.node):
    if node.op_type != "GatherND" or node.domain or node.input[1] not in constants:
      continue
    indices = onnx.numpy_helper.to_array(constants[node.input[1]])
    if not np.any(indices < 0):
      continue
    if indices.ndim < 1 or not np.issubdtype(indices.dtype, np.integer):
      raise ValueError("Invalid GatherND constant indices")
    batch = next((a.i for a in node.attribute if a.name == "batch_dims"), 0)
    shape = shapes.get(node.input[0], [])
    if batch < 0 or batch + indices.shape[-1] > len(shape):
      raise ValueError("Unproven GatherND indexed dimensions")
    normalized = indices.copy()
    for axis in range(indices.shape[-1]):
      width = shape[batch + axis]
      if width <= 0 or np.any((indices[..., axis] < -width) | (indices[..., axis] >= width)):
        raise ValueError("Unproven or out-of-range GatherND constant index")
      normalized[..., axis] = np.where(indices[..., axis] < 0, indices[..., axis] + width, indices[..., axis])
    name = f"fold3_gather_nd_positive_{index}"
    if name in used or name in constants:
      raise ValueError("GatherND initializer name collision")
    used.add(name)
    model.graph.initializer.append(onnx.numpy_helper.from_array(normalized, name))
    node.input[1] = name
  onnx.checker.check_model(model)
  return model


def lower_contiguous_gather_nd(model):
  """Use Slice for a constant contiguous selection along the first axis."""
  model = onnx.shape_inference.infer_shapes(model)
  shapes = {value.name: value.type.tensor_type.shape.dim
            for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  constants = {value.name: value for value in model.graph.initializer}
  used = {name for node in model.graph.node for name in (*node.input, *node.output)}
  used.update(constants)
  for index, node in enumerate(model.graph.node):
    if (node.op_type != "GatherND" or node.domain or node.input[1] not in constants
        or any(attribute.name != "batch_dims" or attribute.i != 0 for attribute in node.attribute)):
      continue
    indices = onnx.numpy_helper.to_array(constants[node.input[1]])
    shape = shapes.get(node.input[0])
    if not shape or indices.ndim != 2 or indices.shape[1] != 1 or not len(indices) or indices.dtype != np.int64:
      continue
    width = shape[0].dim_value
    if width <= 0 or np.any((indices < -width) | (indices >= width)):
      continue
    normalized = indices[:, 0] % width
    start = int(normalized[0])
    stop = start + len(normalized)
    if stop > width or not np.array_equal(normalized, np.arange(start, stop)):
      continue
    names = [f"fold3_slice_{index}_{field}" for field in ("start", "end", "axis")]
    if used.intersection(names) or any(name in shapes for name in names):
      raise ValueError("Slice initializer name collision")
    used.update(names)
    model.graph.initializer.extend(onnx.numpy_helper.from_array(np.array([value], np.int64), name)
                                   for name, value in zip(names, (start, stop, 0)))
    node.CopyFrom(onnx.helper.make_node("Slice", [node.input[0], *names], node.output, name=node.name))
  onnx.checker.check_model(model)
  return model


def pack_road_image_history(model):
  """Merge history and image channels without changing their byte order."""
  model = copy.deepcopy(model)
  graph = model.graph
  nodes = {tuple(node.output): node for node in graph.node}
  expected = {
    ("unsqueeze",): ("Unsqueeze", ["new_img", "model/p_val_18"]),
    ("slice_1",): ("Slice", ["state_img_q", "model/p_val_18", "val_7", "model/p_val_18", "model/p_val_18"]),
    ("next_state_img_q",): ("Concat", ["slice_1", "unsqueeze"]),
    ("split_output", "split_output_token_158"): ("Split", ["next_state_img_q", "splits"]),
    ("select",): ("Squeeze", ["split_output", "squeeze_axes"]),
    ("select_1",): ("Squeeze", ["split_output_token_158", "squeeze_axes_token_159"]),
  }
  for outputs, (op, inputs) in expected.items():
    node = nodes.get(outputs)
    if node is None or node.op_type != op or list(node.input) != inputs or node.domain:
      raise ValueError("Road history graph changed: " + outputs[0])
  constants = {value.name: onnx.numpy_helper.to_array(value) for value in graph.initializer}
  for key, values in {"model/p_val_18": [1], "val_7": [2**63 - 1], "splits": [1, 1],
                      "squeeze_axes": [0], "squeeze_axes_token_159": [0]}.items():
    if not np.array_equal(constants[key], values):
      raise ValueError("Road history index changed: " + key)
  for outputs, axis in ((("next_state_img_q",), 1), (("split_output", "split_output_token_158"), 0)):
    if [(attr.name, attr.i) for attr in nodes[outputs].attribute] != [("axis", axis)]:
      raise ValueError("Road history axis changed")
  history = [2, 5, 6, 128, 256]
  packed = [2, 30, 128, 256]
  for values, name in ((graph.input, "state_img_q"), (graph.output, "next_state_img_q")):
    value = next(value for value in values if value.name == name)
    tensor = value.type.tensor_type
    if tensor.elem_type != onnx.TensorProto.UINT8 or [d.dim_value for d in tensor.shape.dim] != history:
      raise ValueError("Road image history shape or type changed")
    tensor.shape.ClearField("dim")
    for size in packed:
      tensor.shape.dim.add().dim_value = size
  used = {name for node in graph.node for name in (*node.input, *node.output)} | constants.keys()
  if used.intersection(("fold3_history_start", "fold3_history_camera_shape")):
    raise ValueError("Road history initializer collision")
  for name, values in (("fold3_history_start", [6]), ("fold3_history_camera_shape", history[1:])):
    graph.initializer.append(onnx.numpy_helper.from_array(np.array(values, np.int64), name))
  nodes[("slice_1",)].input[1] = "fold3_history_start"
  nodes[("next_state_img_q",)].input[1] = "new_img"
  for name in ("select", "select_1"):
    node = nodes[(name,)]
    node.op_type = "Reshape"
    node.input[1] = "fold3_history_camera_shape"
  graph.node.remove(nodes[("unsqueeze",)])
  changed = {"unsqueeze", "slice_1", "split_output", "split_output_token_158"}
  for value in list(graph.value_info):
    if value.name in changed:
      graph.value_info.remove(value)
  onnx.checker.check_model(model)
  return model


def lower_road_history_strides(model):
  """Express fixed stride-four reads using only contiguous crops."""
  graph = model.graph
  constants = {value.name: onnx.numpy_helper.to_array(value) for value in graph.initializer}
  used = {name for node in graph.node for name in (*node.input, *node.output)} | constants.keys()
  targets = {"slice_3": ("select", 5), "slice_4": ("select_1", 5), "slice_5": ("state_feat_q", 96)}
  rewritten = []
  for node in graph.node:
    output = node.output[0]
    if output not in targets:
      rewritten.append(node)
      continue
    source, length = targets[output]
    if (node.op_type != "Slice" or node.input[0] != source or len(node.input) != 5
        or any(not np.array_equal(constants[name], [expected])
               for name, expected in zip(node.input[1:], (0, 2**63 - 1, 0, 4), strict=True))):
      raise ValueError("Road history stride changed: " + output)

    def constant(label, values):
      name = f"fold3_{output}_{label}"
      if name in used:
        raise ValueError("Road stride initializer collision: " + name)
      graph.initializer.append(onnx.numpy_helper.from_array(np.array(values, np.int64), name))
      return name

    start = constant("start", [0])
    end = constant("end", [1])
    first = f"fold3_{output}_first"
    other = f"fold3_{output}_other"
    if used.intersection((first, other)):
      raise ValueError("Road stride tensor collision")
    if length == 5:
      axis = constant("axis", [0])
      rewritten.extend([
        onnx.helper.make_node("Slice", [source, start, end, axis], [first]),
        onnx.helper.make_node("Slice", [source, constant("last", [4]), constant("stop", [5]), axis], [other]),
        onnx.helper.make_node("Concat", [first, other], [output], axis=0),
      ])
    else:
      rewritten.extend([
        onnx.helper.make_node("Reshape", [source, constant("groups", [24, 4, 1, 512])], [first]),
        onnx.helper.make_node("Slice", [first, start, end, constant("axis", [1])], [other]),
        onnx.helper.make_node("Reshape", [other, constant("shape", [24, 1, 512])], [output]),
      ])
  graph.ClearField("node")
  graph.node.extend(rewritten)
  onnx.checker.check_model(model)
  return model


def pack_road_attention(model):
  """Split contiguous Q/K/V channels before expanding the attention heads."""
  graph = model.graph
  constants = {value.name: onnx.numpy_helper.to_array(value) for value in graph.initializer}
  nodes = {node.output[0]: node for node in graph.node}
  reshape = nodes["model/p_view_1"]
  split = nodes["model/p_node_Slice_38"]
  if (reshape.op_type != "Reshape" or not np.array_equal(constants[reshape.input[1]], [1, 9, 3, 8, 64])
      or split.op_type != "Split" or list(split.output) != [f"model/p_node_Slice_{i}" for i in (38, 42, 46)]
      or split.input[0] != reshape.output[0] or not np.array_equal(constants[split.input[1]], [1, 1, 1])
      or [(attr.name, attr.i) for attr in split.attribute] != [("axis", 2)]):
    raise ValueError("Road attention projection changed")
  # Keep the existing transpose/squeeze ordering for Q and V by directly
  # emitting their former final layouts. K retains its downstream transpose.
  replacements = {
    "model/p_node_unbind__1": ("model/p_node_Slice_42", None),
    "model/p_getitem": ("model/p_node_Slice_38", "model/p_val_40"),
    "model/p_getitem_2": ("model/p_node_Slice_46", "model/p_val_48"),
  }
  for output, (source, intermediate) in replacements.items():
    squeeze = nodes[output]
    if squeeze.op_type != "Squeeze" or squeeze.input[0] != (intermediate or source):
      raise ValueError("Road attention squeeze changed: " + output)
    if intermediate:
      transpose = nodes[intermediate]
      if (transpose.op_type != "Transpose" or list(transpose.input) != [source]
          or list(transpose.attribute[0].ints) != [2, 0, 3, 1, 4]
          or not np.array_equal(constants[squeeze.input[1]], [0])):
        raise ValueError("Road attention transpose changed: " + output)
    elif not np.array_equal(constants[squeeze.input[1]], [2]):
      raise ValueError("Road attention key axis changed")
  shapes = {"fold3_qkv_flat": [1, 9, 1536], "fold3_qkv_split": [512] * 3, "fold3_qkv_heads": [1, 9, 8, 64]}
  used = {name for node in graph.node for name in (*node.input, *node.output)} | constants.keys()
  if used.intersection(shapes):
    raise ValueError("Road attention initializer collision")
  for name, shape in shapes.items():
    graph.initializer.append(onnx.numpy_helper.from_array(np.array(shape, np.int64), name))
  reshape.input[1] = "fold3_qkv_flat"
  split.input[1] = "fold3_qkv_split"
  for output, (source, intermediate) in replacements.items():
    squeeze = nodes[output]
    if intermediate:
      transpose = nodes[intermediate]
      squeeze.CopyFrom(onnx.helper.make_node("Transpose", [intermediate], [output], perm=[0, 2, 1, 3]))
      transpose.CopyFrom(onnx.helper.make_node("Reshape", [source, "fold3_qkv_heads"], [intermediate]))
    else:
      squeeze.CopyFrom(onnx.helper.make_node("Reshape", [source, "fold3_qkv_heads"], [output]))
  changed = set(split.output) | {reshape.output[0], "model/p_val_40", "model/p_val_48"}
  for value in list(graph.value_info):
    if value.name in changed:
      graph.value_info.remove(value)
  onnx.checker.check_model(model)
  return model


def lower_road_attention_mask(model):
  """Replace the pinned causal Where with exact crops and constant padding."""
  graph = model.graph
  constants = {value.name: onnx.numpy_helper.to_array(value) for value in graph.initializer}
  target = "InsertedPrecisionFreeCast_model/p_masked_fill"
  index, node = next((i, n) for i, n in enumerate(graph.node) if list(n.output) == [target])
  mask, fill, scores = node.input
  expected = np.triu(np.ones((9, 9), dtype=bool), 1).reshape(1, 1, 9, 9)
  if (node.op_type != "Where" or not np.array_equal(constants[mask], expected)
      or constants[fill].shape != () or not np.isneginf(constants[fill])):
    raise ValueError("Road attention mask changed")
  used = {name for n in graph.node for name in (*n.input, *n.output)} | constants.keys()
  if any(name.startswith("fold3_causal_") for name in used):
    raise ValueError("Road causal mask tensor collision")

  def constant(label, values, dtype=np.int64):
    name = f"fold3_causal_{label}"
    graph.initializer.append(onnx.numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
    return name

  axes = constant("axes", [2, 3])
  rows = []
  rewritten = []
  for row in range(9):
    valid = f"fold3_causal_valid_{row}"
    rewritten.append(onnx.helper.make_node("Slice", [
      scores, constant(f"start_{row}", [row, 0]), constant(f"end_{row}", [row + 1, row + 1]), axes,
    ], [valid]))
    if row < 8:
      fill_row = constant(f"fill_{row}", np.full((1, 8, 1, 8 - row), -np.inf), np.float32)
      padded = f"fold3_causal_row_{row}"
      rewritten.append(onnx.helper.make_node("Concat", [valid, fill_row], [padded], axis=3))
      rows.append(padded)
    else:
      rows.append(valid)
  rewritten.append(onnx.helper.make_node("Concat", rows, [target], axis=2))
  del graph.node[index]
  for new_node in reversed(rewritten):
    graph.node.insert(index, new_node)
  onnx.checker.check_model(model)
  return model


def retain_host_scalar_powers(model):
  """Keep the pinned host's FP32 scalar square/cube multiplication order.

  The trailing-singleton cube uses the host's powf broadcast loop. Tag internal
  results for its compensated GPU kernel while retaining ONNX Pow semantics.
  FP16, dynamic shapes and external singleton results remain unchanged.
  """
  boundaries = copy.deepcopy((list(model.graph.input), list(model.graph.output)))
  model = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
  info = {v.name: v for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  constants = {v.name: onnx.numpy_helper.to_array(v) for v in model.graph.initializer}
  used = {name for node in model.graph.node for name in (*node.input, *node.output)}
  nodes, provenance, renamed = [], [], {}
  external = {v.name for v in model.graph.output}
  for index, node in enumerate(model.graph.node):
    value = info.get(node.input[0]) if node.input else None
    exponent = constants.get(node.input[1]) if node.op_type == "Pow" and len(node.input) == 2 else None
    dims = list(value.type.tensor_type.shape.dim) if value is not None else []
    rewrite = (node.op_type == "Pow" and exponent is not None and exponent.size == 1
               and exponent.dtype == np.float32 and float(exponent.reshape(-1)[0]) in (2., 3.)
               and value is not None and value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
               and all(d.HasField("dim_value") and d.dim_value > 0 for d in dims))
    if not rewrite:
      nodes.append(node)
      continue
    power = int(exponent.reshape(-1)[0])
    if power == 3 and dims and dims[-1].dim_value == 1:
      if node.output[0] not in external:
        original_output = node.output[0]
        tagged = "fold3_host_powf_cube_" + original_output
        if tagged in used:
          raise ValueError("Host power tensor name collision")
        used.add(tagged)
        renamed[original_output] = tagged
        node.output[0] = tagged
        provenance.append({"node": node.name, "original_operator": "Pow", "input": node.input[0],
                           "output": original_output, "exponent": power, "shape": [d.dim_value for d in dims],
                           "arithmetic": "compensated_float32_cube", "tagged_output": tagged})
      nodes.append(node)
      continue
    square = node.output[0] if power == 2 else f"fold3_host_power_square_{index}"
    if power == 3 and square in used:
      raise ValueError("Host power tensor name collision")
    used.add(square)
    nodes.append(onnx.helper.make_node("Mul", [node.input[0], node.input[0]], [square],
      name=node.name if power == 2 else f"fold3_host_power_square_{index}"))
    if power == 3:
      nodes.append(onnx.helper.make_node("Mul", [square, node.input[0]], list(node.output), name=node.name))
    provenance.append({"node": node.name, "original_operator": "Pow", "input": node.input[0],
                       "output": node.output[0], "exponent": power, "shape": [d.dim_value for d in dims],
                       "arithmetic": "separate_float32_multiplications"})
  del model.graph.node[:]
  model.graph.node.extend(nodes)
  for node in model.graph.node:
    for slot, name in enumerate(node.input):
      node.input[slot] = renamed.get(name, name)
  for value in model.graph.value_info:
    value.name = renamed.get(value.name, value.name)
  for current, original in zip((model.graph.input, model.graph.output), boundaries, strict=True):
    for value, saved in zip(current, original, strict=True):
      value.CopyFrom(saved)
  onnx.checker.check_model(model)
  return model, provenance


def retain_host_spatial_means(model):
  """Tag proven FP32 spatial reductions without changing their ONNX semantics.

  The pinned host accumulates eight independent lanes before pairwise packing.
  The MNN buffer patch uses that order only for these static internal means.
  """
  boundaries = [[copy.deepcopy(v) for v in values] for values in (model.graph.input, model.graph.output)]
  model = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
  info = {v.name: v for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  constants = {v.name: onnx.numpy_helper.to_array(v) for v in model.graph.initializer}
  used = {name for node in model.graph.node for name in (*node.input, *node.output)}
  external = {v.name for v in model.graph.output}
  renamed, provenance = {}, []
  for node in model.graph.node:
    if node.op_type != "ReduceMean" or not node.input:
      continue
    value = info.get(node.input[0])
    if value is None or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
      continue
    dims = list(value.type.tensor_type.shape.dim)
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    axes = constants.get(node.input[1]) if len(node.input) > 1 else attrs.get("axes")
    if axes is None or len(dims) != 4 or not all(d.HasField("dim_value") and d.dim_value > 0 for d in dims):
      continue
    if sorted(int(axis) % 4 for axis in np.asarray(axes).reshape(-1)) != [2, 3]:
      continue
    shape = [d.dim_value for d in dims]
    if shape[2] * shape[3] % 8 or node.output[0] in external:
      continue
    tagged = "fold3_host_mean8_" + node.output[0]
    if tagged in used:
      raise ValueError("Host mean tensor name collision")
    used.add(tagged)
    renamed[node.output[0]] = tagged
    provenance.append({"node": node.name, "input": node.input[0], "output": node.output[0],
                       "tagged_output": tagged, "shape": shape, "axes": [2, 3],
                       "arithmetic": "eight_lane_float32_spatial_mean"})
  for node in model.graph.node:
    for sequence in (node.input, node.output):
      for slot, name in enumerate(sequence):
        sequence[slot] = renamed.get(name, name)
  for value in model.graph.value_info:
    value.name = renamed.get(value.name, value.name)
  for current, original in zip((model.graph.input, model.graph.output), boundaries, strict=True):
    for value, saved in zip(current, original, strict=True):
      value.CopyFrom(saved)
  onnx.checker.check_model(model)
  return model, provenance


def retain_host_sigmoids(model, host_model):
  """Tag proven host FP32 logistic outputs for the MLAS vector arithmetic.

  An exact captured convolution input differed at 807 of 1024 sigmoid
  outputs with MNN's native_exp/native_recip. Injecting those outputs into
  the host reproduced the downstream half-GEMV failure exactly. Keep this
  repair scoped to original FP32 sigmoid provenance and complete float4s.
  """
  model = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
  host_types = {v.name: v.type.tensor_type.elem_type for v in
                (*host_model.graph.input, *host_model.graph.value_info, *host_model.graph.output)}
  host_types.update({v.name: v.data_type for v in host_model.graph.initializer})
  source = {}
  for node in host_model.graph.node:
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    standalone = node.op_type == "Sigmoid" and host_types.get(node.input[0]) == onnx.TensorProto.FLOAT
    fused = (node.op_type in {"FusedConv", "NhwcFusedConv"} and
             attrs.get("activation") in (b"Sigmoid", "Sigmoid") and len(node.input) >= 2 and
             host_types.get(node.input[1]) == onnx.TensorProto.FLOAT)
    if standalone or fused:
      for output in node.output:
        if output in source:
          raise ValueError("Ambiguous original sigmoid provenance")
        source[output] = node
  info = {v.name: v for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  external = {v.name for v in (*model.graph.input, *model.graph.output)}
  used = {name for node in model.graph.node for name in (*node.input, *node.output)}
  renamed, provenance = {}, []
  for node in model.graph.node:
    if node.op_type != "Sigmoid" or len(node.output) != 1 or node.output[0] not in source:
      continue
    value = info.get(node.input[0])
    if value is None or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
      continue
    dims = list(value.type.tensor_type.shape.dim)
    if not dims or not all(d.HasField("dim_value") and d.dim_value > 0 for d in dims):
      continue
    shape = [d.dim_value for d in dims]
    if int(np.prod(shape)) % 4 or node.output[0] in external:
      continue
    output = node.output[0]
    tagged = "fold3_host_sigmoid_" + output
    if tagged in used:
      raise ValueError("Host sigmoid tensor name collision")
    used.add(tagged)
    renamed[output] = tagged
    original = source[output]
    provenance.append({"original_node": original.name, "original_operator": original.op_type,
                       "input": node.input[0], "output": output, "tagged_output": tagged,
                       "shape": shape, "arithmetic": "mlas_arm64_float4_logistic"})
  for node in model.graph.node:
    for sequence in (node.input, node.output):
      for slot, name in enumerate(sequence):
        sequence[slot] = renamed.get(name, name)
  for value in model.graph.value_info:
    value.name = renamed.get(value.name, value.name)
  onnx.checker.check_model(model)
  return model, provenance


def retain_half_boundaries(model):
  """Make every original numeric FP16 result an explicit converter boundary.

  ONNX half-to-half Cast is an identity. Patched MNN retains it as a GPU
  rounding operation even though its numerical operators use FP32 storage.
  This does not replace or promote the original Gemm arithmetic.
  """
  model = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
  types = {v.name: v.type.tensor_type.elem_type for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  for value in model.graph.initializer:
    types[value.name], shapes[value.name] = value.data_type, list(value.dims)
  constants = {v.name for v in model.graph.initializer}
  moves = {"Reshape", "Transpose", "Concat", "Split", "Unsqueeze", "Squeeze", "Slice", "Gather", "Expand", "Identity"}
  nodes = []
  used = {name for node in model.graph.node for name in (*node.input, *node.output)}
  for index, node in enumerate(model.graph.node):
    rounds = []
    if node.op_type not in moves | {"Cast"}:
      for slot, name in enumerate(node.output):
        if types.get(name) == onnx.TensorProto.FLOAT16:
          # The pinned host has distinct half-FMA GEMV and paired-product
          # NoTrans GEMM reductions. Neither is an ordinary FP32 Conv sum.
          attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
          a, b = (shapes.get(v, []) for v in list(node.input[:2]) + [""] * max(0, 2 - len(node.input)))
          gemv8 = (node.op_type == "Gemm" and len(node.input) == 3 and len(a) == len(b) == 2
                   and a[0] == 1 and a[1] > 0 and b[1] == a[1] and b[0] > 0 and node.input[1] in constants and node.input[2] in constants
                   and shapes.get(node.input[2]) in ([b[0]], [1, b[0]])
                   and all(types.get(v) == onnx.TensorProto.FLOAT16 for v in node.input)
                   and attributes.get("transA", 0) == 0 and attributes.get("transB", 0) == 1
                   and attributes.get("alpha", 1.) == 1. and attributes.get("beta", 1.) == 1.)
          dot2 = (node.op_type == "Gemm" and len(node.input) == 3 and len(a) == len(b) == 2
                  and min((*a, *b)) > 0 and a[1] == b[0]
                  and node.input[1] in constants and node.input[2] in constants
                  and shapes.get(node.input[2]) == [b[1]]
                  and all(types.get(v) == onnx.TensorProto.FLOAT16 for v in node.input)
                  and attributes.get("transA", 0) == 0 and attributes.get("transB", 0) == 0
                  and attributes.get("alpha", 1.) == 1. and attributes.get("beta", 1.) == 1.)
          layernorm = (node.op_type == "LayerNormalization" and len(node.input) == 3 and len(node.output) == 1
                       and a and a[-1] > 0 and attributes.get("axis", -1) in (-1, len(a) - 1)
                       and attributes.get("stash_type", 1) == 1
                       and all(types.get(v) == onnx.TensorProto.FLOAT16 for v in node.input)
                       and all(v in constants and shapes.get(v) == [a[-1]] for v in node.input[1:]))
          # Batched half attention uses paired-product FP32 accumulation on
          # the pinned host, including an unpaired odd-K tail. Scope the tag
          # to static, valid MatMul batches handled by MNN's Loop lowering.
          batches = list(zip((1,) * max(0, len(b) - len(a)) + tuple(a[:-2]),
                             (1,) * max(0, len(a) - len(b)) + tuple(b[:-2])))
          matmul2 = (node.op_type == "MatMul" and len(node.input) == 2 and min(len(a), len(b)) >= 2
                     and max(len(a), len(b)) > 2 and min((*a, *b)) > 0 and a[-1] == b[-2]
                     and all(x == y or x == 1 or y == 1 for x, y in batches)
                     and all(types.get(v) == onnx.TensorProto.FLOAT16 for v in node.input))
          kind = "gemv8" if gemv8 else "dot2" if dot2 else "layernorm" if layernorm else "matmul2" if matmul2 else "source"
          intermediate = f"fold3_half_{kind}_{index}_{slot}"
          if intermediate in used:
            raise ValueError("Half boundary tensor name collision")
          used.add(intermediate)
          node.output[slot] = intermediate
          rounds.append(onnx.helper.make_node("Cast", [intermediate], [name],
            name=f"fold3_half_round_{index}_{slot}", to=onnx.TensorProto.FLOAT16))
    nodes.extend((node, *rounds))
  del model.graph.node[:]
  model.graph.node.extend(nodes)
  onnx.checker.check_model(model)
  return model


def retain_host_conv_order(model, host_model, native_convolutions=False):
  """Tag original FP32 convolutions and static GEMVs for host accumulation order.

  Internal tensor renaming carries the tag through MNN's converter without
  inserting an operation or changing external inputs/outputs. Only the host's
  NHWC fusion preloads bias; its other Conv/FusedConv paths add bias last.
  The patched High/buffer kernels interpret these tags. They remain diagnostic
  until the complete original-reference suite passes on the target device.
  """
  model = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
  source = {}
  for node in host_model.graph.node:
    if node.op_type in {"Conv", "FusedConv", "NhwcFusedConv", "Gemm", "FusedGemm"}:
      if not node.name or node.name in source:
        raise ValueError("Ambiguous original convolution/GEMV provenance")
      source[node.name] = (node.op_type, node.op_type == "NhwcFusedConv")
  types = {v.name: v.type.tensor_type.elem_type for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  types.update({v.name: v.data_type for v in model.graph.initializer})
  shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  constants = {v.name: v for v in model.graph.initializer}
  shapes.update({name: list(value.dims) for name, value in constants.items()})
  external = {v.name for v in (*model.graph.input, *model.graph.output)}
  used = {name for node in model.graph.node for name in (*node.input, *node.output)}
  rename = {}
  for node in model.graph.node:
    if node.op_type not in {"Conv", "Gemm"} or node.name not in source:
      continue
    if native_convolutions and node.op_type == "Conv":
      continue  # Diagnostic native convolution; preserve GEMV and Cast rules.
    if types.get(node.input[0]) != onnx.TensorProto.FLOAT or types.get(node.input[1]) != onnx.TensorProto.FLOAT:
      continue
    kind, bias_first = source[node.name]
    if node.op_type == "Gemm":
      # MNN converts this verified host GEMV class to a 1x1 convolution. Reuse
      # its ordered FP32 dot followed by bias, before any ReLU/half cast.
      attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
      x, w = shapes.get(node.input[0], []), shapes.get(node.input[1], [])
      if (kind not in {"Gemm", "FusedGemm"} or node.input[1] not in constants or
          len(x) != 2 or len(w) != 2 or x[0] != 1 or min((*x, *w), default=0) <= 0 or
          x[1] != w[1] or attrs.get("transA", 0) != 0 or attrs.get("transB", 0) != 1 or
          attrs.get("alpha", 1.) != 1. or attrs.get("beta", 1.) != 1.):
        continue
    elif kind in {"Gemm", "FusedGemm"}:
      continue
    name = node.output[0]
    marked = f"fold3_host_conv_{'bf' if bias_first else 'bl'}_{name}"
    if name in external or marked in used:
      raise ValueError("Convolution tag would alter external tensors or collide")
    rename[name] = marked
  for node in model.graph.node:
    for sequence in (node.input, node.output):
      for index, name in enumerate(sequence):
        sequence[index] = rename.get(name, name)
  for value in model.graph.value_info:
    value.name = rename.get(value.name, value.name)
  onnx.checker.check_model(model)
  return model


def check_original(session, references, name, external_outputs=None, packed_history=False):
  def run(inputs):
    types = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(uint8)": np.uint8}
    adapted = {item.name: inputs[item.name].astype(types[item.type], copy=False) for item in session.get_inputs()}
    if packed_history:
      adapted["state_img_q"] = adapted["state_img_q"].reshape(2, 30, 128, 256)
    outputs = dict(zip((item.name for item in session.get_outputs()), session.run(None, adapted)))
    if packed_history:
      outputs["next_state_img_q"] = outputs["next_state_img_q"].reshape(2, 5, 6, 128, 256)
    for key, contract in (external_outputs or {}).items():
      value, target = outputs[key], types[contract["type"]]
      if target == np.uint8 and (not np.isfinite(value).all() or not ((value >= 0) & (value <= 255) & (value == np.floor(value))).all()):
        raise ValueError("Derived graph changed image history values")
      outputs[key] = value.astype(target, copy=False)
    return outputs
  cases = {}
  for seed in (42, 7, 123):
    suffix = "" if seed == 42 else f"-case-{seed}"
    actual = run(initial_inputs(references, name, seed))
    with np.load(references / f"{name}{suffix}-outputs.npz") as expected:
      cases[str(seed)] = {key: compare(value, expected[key]) for key, value in actual.items()}
  sequences = validate(run, references, name)
  return {"seeds": cases, "sequences": sequences,
          "passed": all(check["passed"] for case in cases.values() for check in case.values()) and
                    all(check["passed"] for check in sequences.values()),
          "comparisons": sum(len(case) for case in cases.values()) +
                         sum(len(frame["outputs"]) for seq in sequences.values() for frame in seq["checks"])}


def prepare(destination, engines, models, mnn_patched=False, ncnn_explicit_casts=False, mnn_native_convolutions=False,
            ncnn_half_gemv=False, ncnn_ordered_half_nodes=None, ncnn_half_outputs=False, ncnn_half_dot2=False,
            ncnn_reference_layernorm_nodes=None, ncnn_reference_float_gemm_nodes=None):
  if mnn_native_convolutions and not mnn_patched:
    raise ValueError("Native convolution diagnostics require the patched Cast implementation")
  if (ncnn_ordered_half_nodes or ncnn_half_outputs or ncnn_half_dot2 or ncnn_reference_layernorm_nodes or ncnn_reference_float_gemm_nodes) and (
      "ncnn" not in engines or "driving_supercombo" not in models or not ncnn_explicit_casts):
    raise ValueError("Road half diagnostics require explicit-cast ncnn road conversion")
  destination.mkdir(parents=True, exist_ok=True)
  references = ROOT / "android/app/build/model-reference"
  native = ROOT / "android/build/native/accelerators"
  previous = json.loads((destination / "manifest.json").read_text()) if (destination / "manifest.json").is_file() else {}
  report = {"diagnostic_only": True, "vehicle_tx_enabled": False, "rtol": 1e-3, "atol": 1e-4, "models": dict(previous.get("models", {}))}
  for name in models:
    entry = report["models"][name] = {}
    source = references / "host-fusion" / f"{name}.onnx"
    path = destination / f"{name}.onnx"
    host_model = onnx.load(source)
    original = portable(copy.deepcopy(host_model), promote_half_gemm=False)
    onnx.save(original, path)
    entry.update(source_sha256=digest(ROOT / f"openpilot/selfdrive/modeld/models/{name}.onnx"),
                 derived_sha256=digest(path), preserved_half_gemm=True)
    old = previous.get("models", {}).get(name, {})
    if old.get("derived_sha256") == entry["derived_sha256"] and old.get("source_sha256") == entry["source_sha256"]:
      entry.update({key: old[key] for key in ("mnn", "ncnn") if key not in engines and key in old})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    entry["original_reference"] = check_original(session, references, name)
    entry["inputs"] = {x.name: {"shape": x.shape, "type": x.type} for x in session.get_inputs()}
    entry["outputs"] = {x.name: {"shape": x.shape, "type": x.type} for x in session.get_outputs()}
    entry["unused_inputs"] = unused_inputs(original)
    for engine in engines:
      folder = destination / engine
      folder.mkdir(exist_ok=True)
      if engine == "mnn":
        output = folder / f"{name}.mnn"
        engine_graph = destination / f"{name}-mnn-io.onnx"
        transported = float_image_transport(original)
        if mnn_patched:
          transported, scalar_powers = retain_host_scalar_powers(transported)
          transported, spatial_means = retain_host_spatial_means(transported)
          transported = retain_host_conv_order(transported, host_model, mnn_native_convolutions)
          transported, sigmoids = retain_host_sigmoids(transported, host_model)
          transported = retain_half_boundaries(normalize_constant_gather_nd(transported))
        onnx.save(transported, engine_graph)
        io_session = ort.InferenceSession(str(engine_graph), options, providers=["CPUExecutionProvider"])
        io_reference = check_original(io_session, references, name, entry["outputs"])
        if not io_reference["passed"]:
          raise ValueError("Float image transport failed original-reference qualification")
        code = "import sys; sys.path.insert(0,sys.argv[1]); import _tools; _tools.mnnconvert(['MNNConvert','-f','ONNX','--modelFile',sys.argv[2],'--MNNModel',sys.argv[3],'--bizCode','fold3'])"
        if mnn_patched:
          patched = native / "mnn-patched"
          identity = json.loads((patched / "build.json").read_text())
          if identity["patch_sha256"] != digest(ROOT / "android/patches/mnn-half-cast.patch"):
            raise ValueError("MNN source patch changed; rebuild before conversion")
          command = [str(patched / "host/MNNConvert"), "-f", "ONNX", "--modelFile", str(engine_graph),
                     "--MNNModel", str(output), "--bizCode", "fold3"]
        else:
          command = [sys.executable, "-c", code, str(native / "mnn-python"), str(engine_graph), str(output)]
      else:
        output = folder / f"{name}.ncnn.param"
        engine_graph = path
        ncnn_session = session
        ncnn_reference = None
        if name == "driving_supercombo":
          engine_graph = destination / f"{name}-ncnn-io.onnx"
          road_graph = pack_road_image_history(lower_contiguous_gather_nd(original))
          road_graph = lower_road_history_strides(pack_road_attention(road_graph))
          onnx.save(lower_road_attention_mask(road_graph), engine_graph)
          ncnn_session = ort.InferenceSession(str(engine_graph), options, providers=["CPUExecutionProvider"])
          ncnn_reference = check_original(ncnn_session, references, name, packed_history=True)
          if not ncnn_reference["passed"]:
            raise ValueError("Road ncnn lowering failed original-reference qualification")
        types = {"tensor(float)": "f32", "tensor(float16)": "f16", "tensor(uint8)": "u8", "tensor(int64)": "i64"}
        shapes = ",".join("[" + ",".join(map(str, item.shape)) + "]" + types[item.type] for item in ncnn_session.get_inputs())
        command = [str(native / "pnnx/pnnx/pnnx"), str(engine_graph), f"inputshape={shapes}", "fp16=0",
                   f"ncnnparam={output}", f"ncnnbin={folder / (name + '.ncnn.bin')}",
                   f"pnnxparam={folder / (name + '.pnnx.param')}", f"pnnxbin={folder / (name + '.pnnx.bin')}",
                   f"pnnxpy={folder / (name + '_pnnx.py')}", f"ncnnpy={folder / (name + '_ncnn.py')}"]
      log = folder / f"{name}-conversion.log"
      with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600, cwd=folder)
      entry[engine] = {"exit_code": result.returncode, "status": "converted" if result.returncode == 0 and output.is_file() else "blocked",
                       "sha256": digest(output) if output.is_file() else None, "log": str(log.relative_to(ROOT))}
      if engine == "mnn":
        entry[engine].update(graph_sha256=digest(engine_graph), original_reference=io_reference, float_image_transport=True)
        if mnn_patched:
          entry[engine].update(explicit_half_boundaries=True, host_conv_order=not mnn_native_convolutions, host_gemv_order=True,
                               native_convolutions=mnn_native_convolutions,
                               host_scalar_powers=scalar_powers, host_spatial_means=spatial_means,
                               host_sigmoids=sigmoids, source_build=identity)
      if engine == "ncnn" and (folder / (name + ".ncnn.bin")).is_file():
        if ncnn_reference is not None:
          entry[engine].update(graph_sha256=digest(engine_graph), original_reference=ncnn_reference)
          if result.returncode == 0:
            entry[engine]["reduction_repairs"] = repair_ncnn_road_reductions(output, folder / (name + ".pnnx.param"))
            entry[engine]["clip_repairs"] = repair_ncnn_road_clips(output, folder / (name + ".pnnx.param"))
            entry[engine]["rank_repairs"] = repair_ncnn_road_rank_changes(output, folder / (name + ".pnnx.param"))
        if ncnn_explicit_casts and result.returncode == 0:
          entry[engine]["cast_repairs"] = repair_ncnn_casts(output, folder / (name + ".pnnx.param"))
          entry[engine]["requires_explicit_half_rounding"] = True
        if ncnn_half_gemv and result.returncode == 0:
          if not ncnn_explicit_casts or name != "dmonitoring_model":
            raise ValueError("Half GEMV diagnostic requires driver explicit-cast conversion")
          entry[engine]["half_gemv_repair"] = mark_ncnn_half_heads(output, destination / (name + ".onnx"))
          entry[engine]["requires_ordered_half_gemv"] = True
        if name == "driving_supercombo" and result.returncode == 0:
          if ncnn_ordered_half_nodes:
            entry[engine]["half_gemv_repair"] = mark_ncnn_half_heads(output,
                ROOT / f"openpilot/selfdrive/modeld/models/{name}.onnx", original_nodes=ncnn_ordered_half_nodes)
            entry[engine]["requires_ordered_half_gemv"] = True
          if ncnn_half_outputs:
            entry[engine]["half_output_repairs"] = restore_ncnn_half_outputs(output, path)
          # Keep the same explicit precision boundaries as probe_ncnn before
          # selecting arithmetic, so the packaged and diagnosed graphs match.
          if ncnn_half_dot2:
            entry[engine]["half_dot2_repairs"] = restore_ncnn_half_outputs(output, path, mode="dot2")
            entry[engine]["requires_ordered_half_dot2"] = True
          if ncnn_reference_float_gemm_nodes:
            entry[engine]["reference_float_gemm"] = mark_ncnn_float_gemm(output, path, ncnn_reference_float_gemm_nodes)
            entry[engine]["requires_reference_float_gemm"] = 1
          if ncnn_reference_layernorm_nodes:
            entry[engine]["reference_layernorm"] = mark_ncnn_reference_layernorm(output, path, ncnn_reference_layernorm_nodes)
            entry[engine]["requires_reference_layernorm"] = 1
        engine_inputs = {x.name: {"shape": x.shape, "type": x.type} for x in ncnn_session.get_inputs()}
        engine_outputs = {x.name: {"shape": x.shape, "type": x.type} for x in ncnn_session.get_outputs()}
        entry[engine].update(ncnn_conversion(folder, name, engine_inputs, engine_outputs))
        if ncnn_reference is not None:
          entry[engine].update(engine_inputs=engine_inputs, engine_outputs=engine_outputs)
    (destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(name, entry["original_reference"]["passed"], entry["original_reference"]["comparisons"], flush=True)
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--destination", type=Path, default=ROOT / "android/app/build/model-reference/accelerators")
  parser.add_argument("--engine", choices=("mnn", "ncnn"), action="append")
  parser.add_argument("--ncnn-half-gemv", action="store_true", help="Diagnostic ordered half driver projection")
  parser.add_argument("--ncnn-ordered-half-node", action="append", metavar="ONNX_NODE",
                      help="Diagnostic ordered road GEMV, bound to original weight and input provenance")
  parser.add_argument("--ncnn-half-dot2", action="store_true", help="Diagnostic original NoTrans half GEMM arithmetic")
  parser.add_argument("--ncnn-half-outputs", action="store_true",
                      help="Diagnostic restoration of original road Gemm/LayerNorm half output boundaries")
  parser.add_argument("--ncnn-reference-layernorm-node", action="append", metavar="ONNX_NODE",
                      help="Diagnostic reference Vulkan LayerNorm bound to original provenance")
  parser.add_argument("--ncnn-reference-float-gemm-node", action="append", metavar="ONNX_NODE")
  parser.add_argument("--ncnn-explicit-casts", action="store_true", help="Diagnostic graph requiring patched Vulkan half-rounding shaders")
  parser.add_argument("--mnn-native-convolutions", action="store_true", help="Diagnostic native Conv kernels; preserve explicit Cast and GEMV rules")
  parser.add_argument("--model", choices=("dmonitoring_model", "driving_supercombo"), action="append")
  parser.add_argument("--mnn-patched", action="store_true", help="Use build_mnn.py's source build and retain all half boundaries")
  args = parser.parse_args()
  prepare(args.destination, args.engine or ["mnn", "ncnn"], args.model or ["dmonitoring_model", "driving_supercombo"],
          args.mnn_patched, args.ncnn_explicit_casts, args.mnn_native_convolutions, args.ncnn_half_gemv,
          args.ncnn_ordered_half_node, args.ncnn_half_outputs, args.ncnn_half_dot2, args.ncnn_reference_layernorm_node, args.ncnn_reference_float_gemm_node)
