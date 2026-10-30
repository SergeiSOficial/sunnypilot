#!/usr/bin/env python3
"""Create synthetic FP32 parity cases using an independent CPU ONNX runtime.

These cases exercise compilation and numerical agreement, not driving quality.
"""
import hashlib
import json
import pathlib
import numpy as np
import onnxruntime as ort


def main():
  root = pathlib.Path(__file__).resolve().parents[2]
  output = root / "android/app/build/model-reference"
  output.mkdir(parents=True, exist_ok=True)
  manifest = {"onnxruntime": ort.__version__, "numpy": np.__version__, "seed": 42, "models": {}}
  for name in ("dmonitoring_model", "driving_supercombo"):
    path = root / f"openpilot/selfdrive/modeld/models/{name}.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    inputs = {}
    for spec in session.get_inputs():
      shape = tuple(spec.shape)
      inputs[spec.name] = (rng.integers(0, 256, shape, dtype=np.uint8) if spec.type == "tensor(uint8)"
                           else np.zeros(shape, dtype=np.float32))
    if "traffic_convention" in inputs:
      inputs["traffic_convention"][:] = [[0., 1.]]  # RHD, same convention as the upstream model input.
    expected = session.run(None, inputs)
    np.savez(output / f"{name}-inputs.npz", **inputs)
    np.savez(output / f"{name}-outputs.npz", **{spec.name: value for spec, value in zip(session.get_outputs(), expected, strict=True)})
    with path.open("rb") as source:
      digest = hashlib.file_digest(source, "sha256").hexdigest()
    manifest["models"][name] = {"sha256": digest,
                                "inputs": {key: list(value.shape) for key, value in inputs.items()},
                                "outputs": {s.name: list(v.shape) for s, v in zip(session.get_outputs(), expected, strict=True)}}
  (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print(output)


if __name__ == "__main__":
  main()
