"""Frozen tracing must prove original outputs before diagnosing GPU tensors."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ncnn_frozen', ROOT / 'android/scripts/compare_ncnn_frozen.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FrozenTraceTest(unittest.TestCase):
  def fixture(self, root):
    base = root / 'android/app/build/model-reference'
    for name in ('host-fusion', 'accelerators/ncnn'):
      (base / name).mkdir(parents=True)
    def info(name):
      return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 2])
    model = onnx.helper.make_model(onnx.helper.make_graph([
      onnx.helper.make_node('Relu', ['x'], ['y']), onnx.helper.make_node('Identity', ['y'], ['z'])],
      'frozen', [info('x')], [info('z')], value_info=[info('y')]),
      opset_imports=[onnx.helper.make_opsetid('', 17)], ir_version=9)
    for name in ('host-fusion/dmonitoring_model.onnx', 'accelerators/dmonitoring_model.onnx'):
      onnx.save(model, base / name)
    weights = base / 'accelerators/ncnn/dmonitoring_model.ncnn.bin'
    weights.write_bytes(b'weights')
    expected = np.array([[0., 2.]], np.float32)
    np.savez(base / 'dmonitoring_model-outputs.npz', outputs=expected)
    report = root / 'trace.json'
    report.with_suffix('').mkdir()
    capture = report.with_suffix('') / 'out.bin'
    capture.write_bytes(expected.tobytes())
    report.write_text(json.dumps({'seed': 42, 'sha256': {'model.bin': module.digest(weights)},
      'parity': {'passed': False}, 'tensors': [{'operator': 'relu', 'shape': [1, 1, 1, 1, 2], 'file': 'out.bin', 'sha256': module.digest(capture)}]}))
    report.with_name('trace-comparison.json').write_text(json.dumps({
      'original_graph_sha256': module.digest(base / 'accelerators/dmonitoring_model.onnx'),
      'checks': [{'operator': 'relu', 'original_tensor': 'y'}]}))
    return report, base

  def analyze(self, root, report):
    with patch.object(module, 'ROOT', root), patch.object(module, 'initial_inputs', return_value={'x': np.array([[-1., 2.]], np.float32)}):
      return module.analyze(report)

  def test_tapping_preserves_final_output_without_qualifying_accelerator(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      report, _ = self.fixture(root)
      result = self.analyze(root, report)
      self.assertTrue(result['untapped_original_exact'])
      self.assertTrue(result['checks'][0]['final_output_unchanged'])
      self.assertTrue(result['checks'][0]['parity']['passed'])
      self.assertFalse(result['accelerator_full_model_parity']['passed'])
      self.assertFalse(result['qualified'])

  def test_changed_golden_is_rejected_before_trace_comparison(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      report, base = self.fixture(root)
      np.savez(base / 'dmonitoring_model-outputs.npz', outputs=np.zeros((1, 2), np.float32))
      with self.assertRaisesRegex(ValueError, 'original golden'):
        self.analyze(root, report)

  def test_changed_capture_is_rejected(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      report, _ = self.fixture(root)
      (report.with_suffix('') / 'out.bin').write_bytes(b'changed')
      with self.assertRaisesRegex(ValueError, 'Captured tensor changed'):
        self.analyze(root, report)

  def test_missing_layout_mapping_is_explicit(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      report, _ = self.fixture(root)
      mapping = report.with_name('trace-comparison.json')
      data = json.loads(mapping.read_text())
      data['checks'][0]['original_tensor'] = 'unknown_layout_alias'
      mapping.write_text(json.dumps(data))
      result = self.analyze(root, report)
      self.assertEqual(result['checks'], [])
      self.assertEqual(len(result['unavailable']), 1)
      self.assertFalse(result['qualified'])

  def test_changed_logical_layout_is_rejected(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      report, _ = self.fixture(root)
      data = json.loads(report.read_text())
      data['tensors'][0]['shape'] = [1, 1, 1, 2, 2]
      report.write_text(json.dumps(data))
      with self.assertRaisesRegex(ValueError, 'Captured layout changed'):
        self.analyze(root, report)
