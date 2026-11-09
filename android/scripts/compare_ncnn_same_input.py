#!/usr/bin/env python3
"""Isolate mapped GPU operations using their captured inputs, never host prefixes.

This is diagnostic only: local agreement cannot qualify an untapped model.
Run compare_ncnn_trace.py first to establish weight-bound original mappings.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_model_trace import compare


def analyze(report):
  manifest = json.loads(report.read_text())
  mapped = json.loads(report.with_name(report.stem + "-comparison.json").read_text())
  model = manifest.get("model", "dmonitoring_model")
  if model not in ("dmonitoring_model", "driving_supercombo"):
    raise ValueError("Unknown captured model")
  graph_path = ROOT / f"android/app/build/model-reference/accelerators/{model}.onnx"
  if hashlib.sha256(graph_path.read_bytes()).hexdigest() != mapped.get("derived_graph_sha256", mapped["original_graph_sha256"]):
    raise ValueError("Original graph changed")
  graph = onnx.shape_inference.infer_shapes(onnx.load(graph_path))
  info = {v.name: v for v in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
  constants = {v.name: v for v in graph.graph.initializer}
  producers = {name: index for index, node in enumerate(graph.graph.node) for name in node.output}
  folder = report.with_suffix("")
  tensors = {t["blob"]: t for t in manifest["tensors"]}
  bindings = {c["operator"]: c for c in mapped["checks"]}
  captured = {}
  for tensor in tensors.values():
    mapping = bindings.get(tensor["operator"])
    if mapping is not None:
      captured.setdefault(mapping["original_tensor"], []).append(tensor)
  for line in (folder / "model.param").read_text().splitlines()[2:]:
    words = line.split()
    mapping = bindings.get(words[1])
    if mapping is None or words[2:4] != ["1", "1"] or words[4] not in tensors:
      continue
    node = graph.graph.node[producers[mapping["original_tensor"]]]
    if node.op_type in ("Relu", "Sigmoid") and words[0] in ("Convolution", "ConvolutionDepthWise", "InnerProduct"):
      node = graph.graph.node[producers[node.input[0]]]
    equivalent = {"Convolution": "Conv", "ConvolutionDepthWise": "Conv", "InnerProduct": "Gemm",
                  "LayerNorm": "LayerNormalization", "TanH": "Tanh", "Reduction": "ReduceMean", "Cast": "Cast", "Sigmoid": "Sigmoid", "ReLU": "Relu"}
    if equivalent.get(words[0]) == node.op_type:
      captured.setdefault(node.input[0], []).append(tensors[words[4]])
  checks = []
  options = ort.SessionOptions()
  options.intra_op_num_threads = 2

  def read(tensor):
    raw = (folder / tensor["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != tensor["sha256"]:
      raise ValueError("Captured tensor changed")
    return np.frombuffer(raw, np.float32)

  for output in manifest["tensors"]:
    mapping = bindings.get(output["operator"])
    if mapping is None:
      continue
    target = mapping["original_tensor"]
    if target not in producers:
      continue
    selected, feeds, initializers, boundaries = set(), {}, set(), []

    def visit(name):
      if not name:
        return
      if name in constants:
        initializers.add(name)
      elif name != target and name in captured:
        contract = info[name]
        shape = tuple(d.dim_value for d in contract.type.tensor_type.shape.dim)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(contract.type.tensor_type.elem_type)
        values = [read(item).astype(dtype).reshape(shape) for item in captured[name]]
        if any(not np.array_equal(values[0], value, equal_nan=True) for value in values[1:]):
          raise ValueError("Conflicting captured tensor provenance: " + name)
        if name not in feeds:
          feeds[name] = values[0]
          boundaries.append({"original_tensor": name, "shape": list(shape), "dtype": str(values[0].dtype),
                             "captures": [{key: item[key] for key in ("blob", "file", "sha256", "backend", "layout") if key in item}
                                          for item in captured[name]]})
      elif name in producers:
        index = producers[name]
        if index in selected:
          return
        selected.add(index)
        if len(selected) > 32:
          raise LookupError("Section exceeds 32 nodes; capture nearer inputs")
        for operand in graph.graph.node[index].input:
          visit(operand)
      else:
        raise LookupError("Missing captured input: " + name)

    try:
      visit(target)
    except LookupError as error:
      checks.append({"operator": output["operator"], "original_tensor": target, "status": "blocked", "reason": str(error)})
      continue
    nodes = [graph.graph.node[index] for index in sorted(selected)]
    section = onnx.helper.make_model(onnx.helper.make_graph(copy.deepcopy(nodes), "captured-gpu-input",
      [copy.deepcopy(info[name]) for name in feeds], [copy.deepcopy(info[target])],
      [copy.deepcopy(constants[name]) for name in sorted(initializers)]),
      opset_imports=list(graph.opset_import), ir_version=graph.ir_version)
    onnx.checker.check_model(section)
    expected = ort.InferenceSession(section.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    actual = read(output).reshape(expected.shape)
    node = graph.graph.node[producers[target]]
    checks.append({"operator": output["operator"], "original_node": node.name, "original_tensor": target,
      "original_op_type": node.op_type, "dtype": str(expected.dtype), "captured_inputs": boundaries,
      "section_nodes": [{"name": item.name, "type": item.op_type, "outputs": list(item.output)} for item in nodes],
      "status": "compared", "backend": output.get("backend"), "layout": output.get("layout"),
      "section_sha256": hashlib.sha256(section.SerializeToString()).hexdigest(),
      "parity": compare(actual, expected, allow_masked_infinity=True),
      "exact_elements": int(np.count_nonzero(actual == expected)),
      "half_rounding_disagreements": int(np.count_nonzero(actual.astype(np.float16) != expected.astype(np.float16))),
      "elements": actual.size})
  return {"diagnostic_only": True, "qualified": False, "vehicle_tx_enabled": False,
    "trace_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    "mapping_sha256": hashlib.sha256(report.with_name(report.stem + "-comparison.json").read_bytes()).hexdigest(),
    "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checks": checks}


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("report", type=Path)
  args = parser.parse_args()
  result = analyze(args.report)
  output = args.report.with_name(args.report.stem + "-same-input.json")
  output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
