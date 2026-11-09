"""Compare attention reduction orders using actual recurrent device tensors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--trace", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  manifest = json.loads((args.trace / "manifest.json").read_text())
  hashes = {}

  def load(name):
    tensor = next(t for t in manifest["tensors"] if t["tensor"] == name)
    raw = (args.trace / tensor["file"]).read_bytes()
    hashes[name] = hashlib.sha256(raw).hexdigest()
    return np.frombuffer(raw, dtype=tensor["dtype"]).reshape(tensor["shape"]).astype(np.float16)

  a, b = load("model/p_softmax"), load("model/p_getitem_2")
  actual = load("model/p_matmul_1")
  h = onnx.helper
  graph = h.make_model(h.make_graph([h.make_node("MatMul", ["a", "b"], ["c"])], "actual_attention",
    [h.make_tensor_value_info("a", 10, a.shape), h.make_tensor_value_info("b", 10, b.shape)],
    [h.make_tensor_value_info("c", 10, actual.shape)]), opset_imports=[h.make_opsetid("", 21)])
  graph.ir_version = 10
  options = ort.SessionOptions()
  options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
  reference = ort.InferenceSession(graph.SerializeToString(), options,
    providers=["CPUExecutionProvider"]).run(None, {"a": a, "b": b})[0]
  a, b = a.astype(np.float32), b.astype(np.float32)
  paired = np.zeros(actual.shape, np.float32)
  sequential = paired.copy()
  for k in range(a.shape[-1] // 2):
    pair = np.float32(a[..., 2 * k, None] * b[..., None, 2 * k, :] +
                     a[..., 2 * k + 1, None] * b[..., None, 2 * k + 1, :])
    paired = np.float32(paired + pair)
  if a.shape[-1] % 2:
    paired = np.float32(paired + a[..., -1, None] * b[..., None, -1, :])
  for k in range(a.shape[-1]):
    sequential = np.float32(sequential + a[..., k, None] * b[..., None, k, :])

  def compare(value):
    value = value.astype(np.float16)
    difference = np.abs(value.astype(np.float32) - reference.astype(np.float32))
    indices = np.argwhere(value != reference)
    return {"exact": not len(indices), "different_elements": len(indices),
            "max_abs_error": float(difference.max()), "first_indices": indices[:12].tolist()}

  artifact = json.loads((args.candidate / "artifacts.json").read_text())
  result = {"diagnostic_only": True, "candidate": args.candidate.name,
    "apk_sha256": artifact["files"]["app-debug.apk"]["sha256"],
    "model": manifest["model"], "seed": manifest["seed"],
    "frame": manifest["frame"], "inputs": {"a": a.shape, "b": b.shape}, "captured_sha256": hashes,
    "diagnostic_graph_sha256": hashlib.sha256(graph.SerializeToString()).hexdigest(),
    "captured": compare(actual), "paired_fp32": compare(paired), "sequential_fp32": compare(sequential)}
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
