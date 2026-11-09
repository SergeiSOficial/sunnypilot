"""Compare every captured tensor without treating pre-Cast values as acceptance.

This is diagnostic only. Untapped complete models retain the original tolerances.
Use an immutable candidate graph directory to avoid comparing a later conversion.
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "android/scripts"))
from collect_accel_trace import original_mapping
from fold3_model_trace import compare
from fold3_sequences import trace_inputs


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--trace", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  manifest = json.loads((args.trace / "manifest.json").read_text())
  path = args.candidate / "graphs" / (manifest["model"] + "-mnn-io.onnx")
  graph = onnx.shape_inference.infer_shapes(onnx.load(path))
  info = {value.name: value for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
  inputs = trace_inputs(ROOT / "android/app/build/model-reference", manifest["model"],
                        manifest["seed"], manifest.get("frame"))
  if "inputs_sha256" in manifest and manifest["inputs_sha256"] != {
      key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in inputs.items()}:
    raise ValueError("Diagnostic input provenance mismatch")
  for value in graph.graph.input:
    if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
      inputs[value.name] = inputs[value.name].astype(np.float32)
  options = ort.SessionOptions()
  options.log_severity_level = 3
  options.intra_op_num_threads = 2
  options.inter_op_num_threads = 1
  checks = []
  for tensor in manifest["tensors"]:
    record = {**tensor, "original": original_mapping(graph, tensor["tensor"], tensor["operator"])}
    checks.append(record)
    if not record["original"] or record["original"]["tensor"] not in info:
      continue
    key = record["original"]["tensor"]
    if not tensor["tensor"] and tensor.get("operator_type") == "Reduction":
      # GeometryReduce emits a temporary [outside, dim, inside] tensor.
      # The subsequent named Raster restores the graph shape and is compared.
      record["comparison"] = "internal_geometry_temporary; compare_named_restored_tensor"
      continue
    tapped = copy.deepcopy(graph)
    del tapped.graph.output[:]
    tapped.graph.output.append(info[key])
    data = tapped.SerializeToString()
    record["tapped_graph_sha256"] = hashlib.sha256(data).hexdigest()
    expected = ort.InferenceSession(data, options, providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    actual = np.frombuffer((args.trace / tensor["file"]).read_bytes(), dtype=tensor["dtype"]).reshape(tensor["shape"])
    record["original_dtype"] = str(expected.dtype)
    record["parity"] = compare(actual, expected, allow_masked_infinity=True)
    # The renamed producer precedes an explicit FP16 Cast. Its physical FP32
    # values must remain visible; do not round them merely to clear a failure.
    record["pre_rounding_diagnostic"] = key.startswith("fold3_half_source_")
    print(key, record["parity"]["passed"], flush=True)
  args.output.write_text(json.dumps({"diagnostic_only": True,
      "performance_status": "not_measured", "rtol": 1e-3, "atol": 1e-4,
      "candidate": args.candidate.name, "model": manifest["model"],
      "seed": manifest["seed"], "frame": manifest.get("frame"),
      "derived_graph_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
      "checks": checks}, indent=2) + "\n")


if __name__ == "__main__":
  main()
