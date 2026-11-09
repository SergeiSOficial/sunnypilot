"""Keep original references and expose portable-export numerical failures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import onnx
import onnxruntime as ort


class ModelReferenceTest(unittest.TestCase):
  def test_frozen_trace_includes_custom_precision_boundaries(self):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("model_trace", root / "android/scripts/model_trace.py")
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])
    w = onnx.numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), "w")
    graph = onnx.helper.make_graph([
      onnx.helper.make_node("FusedGemm", ["x", "w"], ["dense"], domain="com.microsoft"),
      onnx.helper.make_node("QuickGelu", ["dense"], ["gelu"], domain="com.microsoft"),
      onnx.helper.make_node("Mul", ["gelu", "gelu"], ["square"])], "trace", [x], [], [w])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17),
                                                       onnx.helper.make_opsetid("com.microsoft", 1)])
    inferred = trace.infer_trace_types(model)
    types = {v.name: v.type.tensor_type.elem_type for v in inferred.graph.value_info}
    self.assertEqual(types, {"dense": onnx.TensorProto.FLOAT, "gelu": onnx.TensorProto.FLOAT,
                             "square": onnx.TensorProto.FLOAT})
    self.assertEqual(model.graph.node, inferred.graph.node)
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(inferred.graph.initializer[0]), np.ones((2, 2)))

  def test_fp32_diagnostics_retain_original_provenance_and_three_cases(self):
    root = Path(__file__).resolve().parents[2]
    reference = root / "android/app/build/model-reference"
    manifest = json.loads((reference / "fp32/manifest.json").read_text())
    for name, metadata in manifest["models"].items():
      original = root / f"openpilot/selfdrive/modeld/models/{name}.onnx"
      self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), metadata["original_sha256"])
      self.assertEqual(metadata["additional_validation_seeds"], [7, 123])
      options = ort.SessionOptions()
      options.intra_op_num_threads = 2
      model = ort.InferenceSession(str(reference / f"fp32/{name}.onnx"), options, providers=["CPUExecutionProvider"])
      for suffix in ("", "-case-7", "-case-123"):
        with np.load(reference / f"fp32/{name}{suffix}-inputs.npz") as inputs, \
            np.load(reference / f"{name}{suffix}-inputs.npz") as original_inputs, \
            np.load(reference / f"fp32/{name}{suffix}-outputs.npz") as expected:
          for key in inputs.files:
            np.testing.assert_array_equal(inputs[key], original_inputs[key])
          for spec, value in zip(model.get_outputs(), model.run(None, dict(inputs)), strict=True):
            np.testing.assert_allclose(value, expected[spec.name], rtol=1e-3, atol=1e-4)

  def test_original_reference_and_portable_export_status(self):
    root = Path(__file__).resolve().parents[2]
    reference = root / "android/app/build/model-reference"
    manifest = json.loads((reference / "manifest.json").read_text())
    for name, metadata in manifest["models"].items():
      source = root / f"openpilot/selfdrive/modeld/models/{name}.onnx"
      compiled = reference / "ort-fallback" / f"{name}.onnx"
      self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), metadata["sha256"])
      self.assertEqual(hashlib.sha256(compiled.read_bytes()).hexdigest(), metadata["ort_fallback"]["sha256"])
      self.assertFalse(any(node.op_type.startswith("Nhwc") for node in onnx.load(compiled).graph.node))
      options = ort.SessionOptions()
      options.intra_op_num_threads = 2
      options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
      model = ort.InferenceSession(str(compiled), options, providers=["CPUExecutionProvider"])
      original_options = ort.SessionOptions()
      original_options.intra_op_num_threads = 2
      original = ort.InferenceSession(str(source), original_options, providers=["CPUExecutionProvider"])
      passed = True
      for index, suffix in enumerate(("", "-case-7", "-case-123")):
        with self.subTest(name=name, case=suffix), np.load(reference / f"{name}{suffix}-inputs.npz") as inputs, \
            np.load(reference / f"{name}{suffix}-outputs.npz") as expected:
          for key in inputs.files:
            self.assertEqual(str(inputs[key].dtype), metadata["input_dtypes"][key])
          for spec, actual, reference_actual in zip(model.get_outputs(), model.run(None, dict(inputs)),
                                                   original.run(None, dict(inputs)), strict=True):
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(reference_actual, expected[spec.name], rtol=1e-3, atol=1e-4)
            matches = bool(np.allclose(actual, expected[spec.name], rtol=1e-3, atol=1e-4))
            self.assertEqual(metadata["ort_fallback"]["checks"][index]["outputs"][spec.name]["passed"], matches)
            passed &= matches
      self.assertEqual(metadata["ort_fallback"]["reference_parity"], "passed" if passed else "failed")
