#!/usr/bin/env python3
"""Read MNN callback tensors and compare small tapped sections to the host reference.

These are synchronous diagnostic readbacks, never accelerator timing or qualification.
The complete untapped model remains the acceptance graph.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_model_trace import compare
from fold3_sequences import trace_inputs


def original_mapping(model, tensor, operator):
  """Only exact converter names establish provenance; never infer from shape."""
  producers = {output: node for node in model.graph.node for output in node.output}
  candidates = (tensor, operator, operator.removeprefix("InsertedPrecisionFreeCast_"))
  for name in candidates:
    if name in producers:
      node = producers[name]
      return {"tensor": name, "node": node.name, "operator": node.op_type, "domain": node.domain}
  return None


def compare_trace(folder, references):
  manifest = json.loads((folder / "manifest.json").read_text())
  name = manifest["model"]
  if name not in ("dmonitoring_model", "driving_supercombo"):
    raise ValueError("Unknown model in accelerator trace")
  graph_path = references / "accelerators" / f"{name}-mnn-io.onnx"
  original = onnx.shape_inference.infer_shapes(onnx.load(graph_path))
  info = {v.name: v for v in (*original.graph.input, *original.graph.value_info, *original.graph.output)}
  if "input_archive_sha256" in manifest:
    archive = folder / "inputs.npz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["input_archive_sha256"]:
      raise ValueError("Diagnostic input archive hash mismatch")
    with np.load(archive, allow_pickle=False) as source:
      inputs = {key: value.copy() for key, value in source.items()}
    initial = trace_inputs(references, name, manifest["seed"])
    if set(inputs) != set(initial) or any(inputs[key].shape != value.shape or inputs[key].dtype != value.dtype
                                         for key, value in initial.items()):
      raise ValueError("Diagnostic input archive schema mismatch")
    if name == "driving_supercombo" and manifest.get("frame") is not None:
      if (manifest.get("feedback_source") != "actual-untapped-backend" or
          manifest.get("warmup_frames") != manifest["frame"]):
        raise ValueError("Recurrent diagnostic input lacks actual backend feedback provenance")
  else:
    inputs = trace_inputs(references, name, manifest["seed"], manifest.get("frame"))
  if "inputs_sha256" in manifest and manifest["inputs_sha256"] != {
      key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in inputs.items()}:
    raise ValueError("Diagnostic input provenance mismatch")
  for value in original.graph.input:
    if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and inputs[value.name].dtype == np.uint8:
      inputs[value.name] = inputs[value.name].astype(np.float32)
  options = ort.SessionOptions()
  options.intra_op_num_threads = 2
  options.inter_op_num_threads = 1
  checks, graph_hashes = [], []
  for item in manifest["tensors"]:
    record = {**item, "original": original_mapping(original, item["tensor"], item["operator"])}
    checks.append(record)
    if record["original"] is None:
      record["comparison"] = "unmapped_converter_operation"
      continue
    key = record["original"]["tensor"]
    if key not in info:
      record["comparison"] = "missing_original_type"
      continue
    # Expose one tensor at a time; large all-output graphs alter memory and fusion.
    tapped = copy.deepcopy(original)
    del tapped.graph.output[:]
    tapped.graph.output.append(info[key])
    data = tapped.SerializeToString()
    graph_hashes.append(hashlib.sha256(data).hexdigest())
    session = ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])
    expected = session.run(None, inputs)[0]
    path = folder / item["file"]
    if path.parent != folder or path.suffix != ".bin":
      raise ValueError("Tensor file escapes trace directory")
    actual = np.frombuffer(path.read_bytes(), dtype=item["dtype"]).reshape(item["shape"])
    # MNN's observed spatial Reduction callback omits its unit batch dimension.
    # This trace adapter changes no axes or values; retain the raw shape.
    # Never apply size-only reshaping to arbitrary operators or model outputs.
    if (item.get("operator_type") == "Reduction" and item.get("layout") == "NCHW" and
        record["original"]["operator"] == "ReduceMean" and expected.shape == (1, *actual.shape)):
      record["trace_shape_adapter"] = "restore_unit_reduction_batch"
      actual = actual[np.newaxis]
    record["parity"] = compare(actual, expected, allow_masked_infinity=True)
    record["original_dtype"] = str(expected.dtype)
    # Preserve a bounded first-failure diagnostic; later operators are downstream.
    if not record["parity"]["passed"]:
      break
  with graph_path.open("rb") as source:
    graph_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
  return {"diagnostic_only": True, "performance_status": "not_measured", "seed": manifest["seed"],
          "frame": manifest.get("frame"),
          "model": name, "memory_mode": manifest["memory_mode"], "rtol": 1e-3, "atol": 1e-4,
          "derived_graph_sha256": graph_sha256,
          "diagnostic_graph_hashes": graph_hashes, "captured_tensors": len(manifest["tensors"]), "checks": checks,
          "first_tolerance_failure": next((v for v in checks if v.get("parity", {}).get("passed") is False), None)}


def collect(adb, serial, package, report, destination):
  result = {}
  for name, model in report["models"].items():
    trace = model.get("trace")
    if not trace:
      continue
    relative = "files/" + trace["path"].split("/files/", 1)[1]
    if relative not in {f"files/accel-trace-{name}-{mode}" for mode in ("image", "buffer")}:
      raise ValueError("Trace outside application's accelerator directory")
    folder = destination / Path(relative).name
    folder.mkdir(parents=True, exist_ok=True)
    def read(filename, relative=relative):
      return subprocess.run([adb, "-s", serial, "exec-out", "run-as", package, "cat", relative + "/" + filename],
                            capture_output=True, check=True, timeout=120).stdout
    manifest_data = read("manifest.json")
    manifest = json.loads(manifest_data)
    (folder / "manifest.json").write_bytes(manifest_data)
    if "input_archive_sha256" in manifest:
      data = read("inputs.npz")
      if len(data) > 32 * 1024 * 1024:
        raise ValueError("Diagnostic input archive exceeds its owned-memory budget")
      (folder / "inputs.npz").write_bytes(data)
    total = 0
    for tensor in manifest["tensors"]:
      filename = tensor["file"]
      if Path(filename).name != filename or not filename.endswith(".bin"):
        raise ValueError("Invalid tensor filename")
      data = read(filename)
      total += len(data)
      if total > 128 * 1024 * 1024:
        raise ValueError("Trace exceeds its owned-memory budget")
      (folder / filename).write_bytes(data)
    result[name] = compare_trace(folder, ROOT / "android/app/build/model-reference")
    (folder / "comparison.json").write_text(json.dumps(result[name], indent=2) + "\n")
  return result


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--serial", required=True)
  parser.add_argument("--package", default="com.samsung.sunnypilot.fold3")
  parser.add_argument("--report", type=Path, required=True)
  parser.add_argument("--artifacts", type=Path, required=True)
  args = parser.parse_args()
  records = collect(args.adb, args.serial, args.package, json.loads(args.report.read_text()), args.artifacts)
  print(json.dumps({name: {"captured_tensors": v["captured_tensors"], "first_tolerance_failure": v["first_tolerance_failure"]}
                    for name, v in records.items()}, indent=2))
