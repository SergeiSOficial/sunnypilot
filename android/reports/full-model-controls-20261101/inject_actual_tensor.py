"""Cut one captured tensor into the original host graph; diagnostic only."""
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
from fold3_sequences import trace_inputs


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--trace", type=Path, required=True)
  parser.add_argument("--tensor", required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  manifest = json.loads((args.trace / "manifest.json").read_text())
  record = next(t for t in manifest["tensors"] if t["tensor"] == args.tensor)
  path = ROOT / "android/app/build/model-reference/host-fusion" / (manifest["model"] + ".onnx")
  graph = onnx.shape_inference.infer_shapes(onnx.load(path))
  info = {v.name: v for v in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
  cut = copy.deepcopy(info[args.tensor])
  expected_type = onnx.helper.tensor_dtype_to_np_dtype(cut.type.tensor_type.elem_type)
  raw = (args.trace / record["file"]).read_bytes()
  actual = np.frombuffer(raw, dtype=record["dtype"]).reshape(record["shape"]).astype(expected_type)
  if actual.shape != tuple(d.dim_value for d in cut.type.tensor_type.shape.dim):
    raise ValueError("Captured layout does not match the original tensor")
  for node in graph.graph.node:
    for index, output in enumerate(node.output):
      if output == args.tensor:
        node.output[index] = "fold3_diagnostic_unused_" + output
  graph.graph.input.append(cut)
  onnx.checker.check_model(graph)
  data = graph.SerializeToString()
  options = ort.SessionOptions()
  options.log_severity_level = 3
  options.intra_op_num_threads = 2
  options.inter_op_num_threads = 1
  references = ROOT / "android/app/build/model-reference"
  if "input_archive_sha256" in manifest:
    archive = args.trace / "inputs.npz"
    if hashlib.file_digest(archive.open("rb"), "sha256").hexdigest() != manifest["input_archive_sha256"]:
      raise ValueError("Captured input archive hash mismatch")
    with np.load(archive, allow_pickle=False) as source:
      inputs = {key: value.copy() for key, value in source.items()}
    if {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in inputs.items()} != manifest["inputs_sha256"]:
      raise ValueError("Captured input tensor hash mismatch")
    if manifest.get("frame") is not None and (manifest.get("feedback_source") != "actual-untapped-backend" or
                                              manifest.get("warmup_frames") != manifest["frame"]):
      raise ValueError("Frame-specific cut requires actual recurrent backend feedback")
  else:
    inputs = trace_inputs(references, manifest["model"], manifest["seed"], manifest.get("frame"))
  inputs[args.tensor] = actual
  session = ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])
  outputs = dict(zip((v.name for v in session.get_outputs()), session.run(None, inputs), strict=True))
  reference = references / (manifest["model"] + ("" if manifest["seed"] == 42 else f"-case-{manifest['seed']}") + "-outputs.npz")
  frame = manifest.get("frame")
  if frame is not None:
    reference = references / "sequences" / f"{manifest['model']}-seed{manifest['seed']}.npz"
  with np.load(reference) as expected:
    checks = {key: compare(value, expected[key] if frame is None else expected[key][frame], allow_masked_infinity=True)
              for key, value in outputs.items() if key in expected.files}
  artifact = json.loads((args.candidate / "artifacts.json").read_text())
  result = {"diagnostic_only": True, "candidate": args.candidate.name,
      "apk_sha256": artifact["files"]["app-debug.apk"]["sha256"],
      "model": manifest["model"], "seed": manifest["seed"], "frame": frame, "tensor": args.tensor,
      "captured_sha256": hashlib.sha256(raw).hexdigest(),
      "original_host_graph_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
      "diagnostic_graph_sha256": hashlib.sha256(data).hexdigest(), "checks": checks}
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
