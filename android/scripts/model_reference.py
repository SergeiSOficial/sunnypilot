#!/usr/bin/env python3
"""Create synthetic FP32 parity cases using an independent CPU ONNX runtime.

These cases exercise compilation and numerical agreement, not driving quality.
"""
import hashlib
import json
import pathlib
import argparse
import base64
import pickle
import numpy as np
import onnxruntime as ort


def promote_fp32(source, destination):
  """Increase encoded FP16 precision; never change shapes or integer inputs."""
  import onnx
  from onnx import TensorProto, numpy_helper
  model = onnx.load(source)
  for tensor in model.graph.initializer:
    if tensor.data_type == TensorProto.FLOAT16:
      tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float32), tensor.name))
  for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
    if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
      value.type.tensor_type.elem_type = TensorProto.FLOAT
  for node in model.graph.node:
    for attribute in node.attribute:
      if node.op_type == "Cast" and attribute.name == "to" and attribute.i == TensorProto.FLOAT16:
        attribute.i = TensorProto.FLOAT
      if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == TensorProto.FLOAT16:
        tensor = attribute.t
        tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float32), tensor.name))
  onnx.checker.check_model(model)
  onnx.save(model, destination)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--fp32", action="store_true", help="Also generate and validate increased-precision models")
  args = parser.parse_args()
  root = pathlib.Path(__file__).resolve().parents[2]
  output = root / "android/app/build/model-reference"
  if args.fp32:
    output /= "fp32"
  output.mkdir(parents=True, exist_ok=True)
  manifest = {"onnxruntime": ort.__version__, "numpy": np.__version__, "seed": 42,
              "precision": "fp32 promotion" if args.fp32 else "original", "models": {}}
  for name in ("dmonitoring_model", "driving_supercombo"):
    path = root / f"openpilot/selfdrive/modeld/models/{name}.onnx"
    original_path = path
    if args.fp32:
      path = output / f"{name}.onnx"
      promote_fp32(original_path, path)
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
    # Metadata comes from the same hash-checked model, never a Linux JIT pickle.
    slices = pickle.loads(base64.b64decode(session.get_modelmeta().custom_metadata_map["output_slices"]))
    manifest["models"][name]["output_slices"] = {key: [value.start, value.stop, value.step] for key, value in slices.items()}
    manifest["models"][name]["input_dtypes"] = {key: str(value.dtype) for key, value in inputs.items()}
    if args.fp32:
      original = ort.InferenceSession(str(original_path), options, providers=["CPUExecutionProvider"]).run(None, inputs)
      manifest["models"][name]["original_sha256"] = hashlib.sha256(original_path.read_bytes()).hexdigest()
      manifest["models"][name]["promotion_max_abs_difference"] = {
        spec.name: float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        for spec, a, b in zip(session.get_outputs(), original, expected, strict=True)}
  (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  import onnx
  from onnx import TensorProto, helper
  graph = helper.make_graph([helper.make_node("Add", ["x", "x"], ["y"])], "android_runtime_check",
                            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
  onnx.save(model, output / "runtime-check.onnx")
  print(output)


if __name__ == "__main__":
  main()
