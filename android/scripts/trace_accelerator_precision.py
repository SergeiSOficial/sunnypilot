#!/usr/bin/env python3
"""Trace a diagnostic FP32 graph against original FP16 tensor boundaries on host."""
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort

from accelerator_precision import explicit_half
from prepare_accel_models import ROOT
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_model_trace import compare
from fold3_sequences import initial_inputs


def trace(name, destination):
  references = ROOT / "android/app/build/model-reference"
  original = onnx.shape_inference.infer_shapes(onnx.load(references / "accelerators" / f"{name}.onnx"))
  modified = onnx.shape_inference.infer_shapes(explicit_half(original))
  info = {v.name: v for v in (*original.graph.value_info, *original.graph.output)}
  taps = [(out, node.op_type, node.name) for node in original.graph.node for out in node.output
          if out in info and info[out].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16]
  options = ort.SessionOptions()
  options.intra_op_num_threads = 2
  inputs = initial_inputs(references, name, 42)
  expected = {}
  for model, results in ((original, expected), (modified, {})):
    value_info = {v.name: v for v in (*model.graph.value_info, *model.graph.output)}
    del model.graph.output[:]
    model.graph.output.extend(value_info[key] for key, _, _ in taps)
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    values = {key: value.astype(np.float32) if session.get_inputs()[i].type == "tensor(float)" else value
              for i, (key, value) in enumerate(inputs.items())}
    results.update(zip((key for key, _, _ in taps), session.run(None, values)))
  checks = [{"tensor": key, "operator": op, "node": node, **compare(results[key], expected[key])}
            for key, op, node in taps]
  report = {"diagnostic_only": True, "seed": 42, "model": name, "checks": checks,
            "first_tolerance_failure": next((item for item in checks if not item["passed"]), None)}
  destination.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps({"model": name, "taps": len(taps), "first_tolerance_failure": report["first_tolerance_failure"]}), flush=True)


if __name__ == "__main__":
  destination = ROOT / "android/app/build/model-reference/accelerator-precision"
  destination.mkdir(exist_ok=True)
  for name in ("dmonitoring_model", "driving_supercombo"):
    trace(name, destination / f"{name}-trace.json")
