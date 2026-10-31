#!/usr/bin/env python3
"""Compare pinned tinygrad's CPU backend with a generated ONNX Runtime reference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tinygrad import Tensor
from tinygrad.nn.onnx import OnnxRunner


def validate(fp32, output):
  root = Path(__file__).resolve().parents[2]
  reference = root / "android/app/build/model-reference"
  models = root / "openpilot/selfdrive/modeld/models"
  if fp32:
    reference /= "fp32"
    models = reference
  manifest = json.loads((reference / "manifest.json").read_text())
  report = {"status": "passed", "backend": "tinygrad CPU", "reference": manifest,
            "synthetic_only": True, "replay_verified": False, "vehicle_tx_enabled": False, "models": {}}
  for name, metadata in manifest["models"].items():
    path = models / f"{name}.onnx"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
      raise ValueError(f"{name}: reference hash mismatch")
    runner = OnnxRunner(Tensor(path.read_bytes(), device="PYTHON")).to("CPU")
    with np.load(reference / f"{name}-inputs.npz") as source:
      inputs = {key: Tensor(source[key], device="CPU") for key in source.files}
    results = runner(inputs)
    parity = report["models"][name] = {}
    with np.load(reference / f"{name}-outputs.npz") as expected:
      for key in expected.files:
        actual = results[key].numpy()
        error = np.abs(actual.astype(np.float64) - expected[key].astype(np.float64))
        passed = bool(np.isfinite(actual).all() and np.allclose(actual, expected[key], rtol=1e-3, atol=1e-4))
        parity[key] = {"passed": passed, "max_abs_error": float(error.max()) if np.isfinite(error).all() else None}
        if not passed:
          report["status"] = "failed"
  Path(output).write_text(json.dumps(report, indent=2) + "\n")
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--fp32", action="store_true")
  parser.add_argument("--report", default="host-model-parity.json")
  args = parser.parse_args()
  result = validate(args.fp32, args.report)
  print(json.dumps({"status": result["status"], "models": result["models"]}, indent=2))
  raise SystemExit(0 if result["status"] == "passed" else 1)
