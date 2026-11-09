#!/usr/bin/env python3
"""Compare sampled first-Conv outputs with explicit accumulation policies."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[2]


def main():
  report = {"diagnostic_only": True, "vehicle_tx_enabled": False, "models": {}}
  for name in ("dmonitoring_model", "driving_supercombo"):
    path = ROOT / f"openpilot/selfdrive/modeld/models/{name}.onnx"
    model = onnx.load(path)
    node = next(node for node in model.graph.node if node.op_type == "Conv")
    weights = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    with np.load(ROOT / f"android/app/build/model-reference/trace/{name}-outputs.npz") as values:
      source, expected = values[node.input[0]], values[node.output[0]]
    kernel, bias = weights[node.input[1]], weights[node.input[2]]
    pads = attributes["pads"]
    stride = attributes["strides"]
    padded = np.pad(source, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    coordinates = np.random.default_rng(42).integers([0, 0, 0], expected.shape[1:], size=(128, 3))
    actual = np.array([expected[0, c, y, x] for c, y, x in coordinates])
    inputs = np.array([padded[0, :, y*stride[0]:y*stride[0]+kernel.shape[2], x*stride[1]:x*stride[1]+kernel.shape[3]]
                       for _, y, x in coordinates])
    filters = kernel[coordinates[:, 0]]
    biases = bias[coordinates[:, 0]]
    policies = {}

    def check(label, prediction):
      policies[label] = {"passed": bool(np.allclose(prediction, actual, rtol=1e-3, atol=1e-4)),
                        "max_abs_error": float(np.max(np.abs(prediction.astype(np.float64) - actual.astype(np.float64)))),
                        "exact_elements": int(np.count_nonzero(prediction == actual)), "samples": len(actual)}

    check("fp64_accumulate_then_half", ((inputs.astype(np.float64) * filters).sum(axis=(1, 2, 3)) + biases).astype(np.float16))
    for layout in ("chw", "hwc"):
      x = inputs if layout == "chw" else inputs.transpose(0, 2, 3, 1)
      w = filters if layout == "chw" else filters.transpose(0, 2, 3, 1)
      x, w = x.reshape(len(actual), -1).astype(np.float32), w.reshape(len(actual), -1).astype(np.float32)
      for lanes in (1, 2, 4, 8, 16):
        sums = np.zeros((len(actual), lanes), np.float16)
        for i in range(x.shape[1]):
          lane = i % lanes
          sums[:, lane] = (sums[:, lane].astype(np.float32) + x[:, i] * w[:, i]).astype(np.float16)
        check(f"{layout}_fp16_fma_{lanes}_lanes", (sums.sum(axis=1, dtype=np.float32) + biases).astype(np.float16))
    report["models"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "operator": node.name,
                              "policies": policies}
  destination = ROOT / "android/reports/blocker-fixes/convolution-accumulation.json"
  destination.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
