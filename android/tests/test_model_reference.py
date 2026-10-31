"""Keep original references and expose portable-export numerical failures."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import onnx
import onnxruntime as ort


class ModelReferenceTest(unittest.TestCase):
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
