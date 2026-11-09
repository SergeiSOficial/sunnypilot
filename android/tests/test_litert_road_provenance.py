import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from prepare_litert_road import fold_constant_sqrt, prepare


class RoadProvenanceTest(unittest.TestCase):
  def test_constant_sqrt_folds_without_changing_values(self):
    graph = helper.make_graph([
      helper.make_node('Sqrt', ['constant'], ['folded']),
      helper.make_node('Sqrt', ['input'], ['dynamic']),
      helper.make_node('Add', ['folded', 'dynamic'], ['output']),
    ], 'sqrt', [helper.make_tensor_value_info('input', TensorProto.FLOAT, [2])],
       [helper.make_tensor_value_info('output', TensorProto.FLOAT, [2])],
       [numpy_helper.from_array(np.array([0.63661975, 4.0], np.float32), 'constant')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    inputs = {'input': np.array([1.0, 9.0], np.float32)}
    expected = ort.InferenceSession(model.SerializeToString()).run(None, inputs)[0]
    model, count = fold_constant_sqrt(model)
    actual = ort.InferenceSession(model.SerializeToString()).run(None, inputs)[0]
    self.assertEqual(count, 1)
    self.assertEqual([node.op_type for node in model.graph.node], ['Sqrt', 'Add'])
    np.testing.assert_array_equal(actual, expected)

  def test_untrusted_source_is_rejected_before_rewrite(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      source = root / 'source.onnx'
      source.write_bytes(b'untrusted')
      output = root / 'output'
      with self.assertRaisesRegex(ValueError, 'Source graph hash mismatch'):
        prepare(source, 'incorrect', root / 'receipt.json', root / 'original.onnx', root, output)
      self.assertFalse(output.exists())

  def test_source_must_be_bound_to_original_reference(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      source, original = root / 'source.onnx', root / 'original.onnx'
      source.write_bytes(b'source')
      original.write_bytes(b'original')
      source_hash = hashlib.sha256(b'source').hexdigest()
      original_hash = hashlib.sha256(b'original').hexdigest()
      (root / 'manifest.json').write_text(json.dumps({'models': {'driving_supercombo': {'sha256': original_hash}}}))
      receipt = root / 'receipt.json'
      receipt.write_text(json.dumps({'original_sha256': 'different', 'derived_sha256': source_hash,
                                     'host_reference': {'passed': True, 'comparisons': 636},
                                     'float_image_transport': True}))
      output = root / 'output'
      with self.assertRaisesRegex(ValueError, 'provenance'):
        prepare(source, source_hash, receipt, original, root, output)
      self.assertFalse(output.exists())
