"""Diagnose the mapped driver spatial mean using its captured actual GPU input."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_model_trace import compare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
manifest = json.loads((args.trace / "manifest.json").read_text())
graph = onnx.shape_inference.infer_shapes(onnx.load(args.candidate / "graphs/dmonitoring_model-mnn-io.onnx"))
node = next(n for n in graph.graph.node if n.name == "node_mean" and n.op_type == "ReduceMean")
records = {r["tensor"]: r for r in manifest["tensors"] if r["tensor"]}
source = records[node.input[0]]
# MNN's explicit tensor-layout conversion retains the source tensor's exact
# name before the ___tr4 suffix. Require exactly one such output, never infer
# a mapping merely from dimensions.
destinations = [r for r in manifest["tensors"] if r["tensor"].startswith(node.output[0] + "___tr4")]
assert len(destinations) == 1, "Spatial mean layout-conversion mapping is ambiguous"
destination = destinations[0]
def load(record):
  return np.fromfile(args.trace / record["file"], dtype=record["dtype"]).reshape(record["shape"])
values = load(source)
actual = load(destination)
assert values.dtype == np.float32 and values.ndim == 4 and values.shape[2] * values.shape[3] % 8 == 0
info = {v.name: v for v in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
constants = {v.name: v for v in graph.graph.initializer}
local = onnx.helper.make_model(onnx.helper.make_graph([copy.deepcopy(node)], "actual_gpu_spatial_mean",
    [info[node.input[0]]], [info[node.output[0]]], [copy.deepcopy(constants[k]) for k in node.input[1:]]),
    opset_imports=list(graph.opset_import))
local.ir_version = graph.ir_version
options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
expected = ort.InferenceSession(local.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(
    None, {node.input[0]: values})[0]
positions = values.shape[2] * values.shape[3]
flat = values.reshape(-1, positions)
lanes = flat[:, :8].copy()
for start in range(8, positions, 8):
  lanes = np.add(lanes, flat[:, start:start + 8], dtype=np.float32)
pack = np.add(lanes[:, :4], lanes[:, 4:], dtype=np.float32)
summed = np.add(np.add(pack[:, 0], pack[:, 1], dtype=np.float32),
                np.add(pack[:, 2], pack[:, 3], dtype=np.float32), dtype=np.float32)
ordered = np.divide(summed, np.float32(positions), dtype=np.float32).reshape(expected.shape)
result = {
  "candidate": args.candidate.name,
  "apk_sha256": json.loads((args.candidate / "artifacts.json").read_text())["files"]["app-debug.apk"]["sha256"],
  "diagnostic_only": True, "operator": node.op_type, "original_node": node.name,
  "input": source, "output_layout_conversion": destination,
  "input_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
  "diagnostic_graph_sha256": hashlib.sha256(local.SerializeToString()).hexdigest(),
  "gpu_actual_input_parity": compare(actual, expected),
  "gpu_unequal_elements": int(np.count_nonzero(actual != expected)),
  "ordered_host_parity": compare(ordered, expected),
  "ordered_host_unequal_elements": int(np.count_nonzero(ordered != expected)),
  "note": "Actual-input local diagnosis only; does not qualify the untapped full model."
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
