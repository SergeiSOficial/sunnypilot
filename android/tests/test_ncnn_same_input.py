"""Same-input diagnosis must retain original rounding and reject changed captures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ncnn_same_input', ROOT / 'android/scripts/compare_ncnn_same_input.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SameInputTest(unittest.TestCase):
  def fixture(self, root, operation):
    x = np.array([[1.0004, -.125, .5, 2.]], dtype=np.float32)
    kind, attributes, expected = {
      'TanH': ('Tanh', {}, np.tanh(x)),
      'Reduction': ('ReduceMean', {'axes': [1], 'keepdims': 1}, x.mean(axis=1, keepdims=True)),
      'Cast': ('Cast', {'to': onnx.TensorProto.FLOAT16}, x.astype(np.float16)),
    }[operation]
    node = onnx.helper.make_node(kind, ['x'], ['y'], name='original', **attributes)
    model = onnx.helper.make_model(onnx.helper.make_graph([node], 'section',
      [onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, x.shape)],
      [onnx.helper.make_tensor_value_info('y', onnx.helper.np_dtype_to_tensor_dtype(expected.dtype), expected.shape)]),
      opset_imports=[onnx.helper.make_opsetid('', 17)], ir_version=9)
    graph = root / 'android/app/build/model-reference/accelerators/dmonitoring_model.onnx'
    graph.parent.mkdir(parents=True)
    onnx.save(model, graph)
    report = root / 'trace.json'
    folder = report.with_suffix('')
    folder.mkdir()
    tensors = []
    for name, operator, value in [('in', 'input', x), ('out', 'operation', expected)]:
      raw = value.astype(np.float32).tobytes()
      (folder / (name + '.bin')).write_bytes(raw)
      tensors.append({'blob': name, 'operator': operator, 'file': name + '.bin', 'sha256': hashlib.sha256(raw).hexdigest()})
    report.write_text(json.dumps({'tensors': tensors}))
    report.with_name('trace-comparison.json').write_text(json.dumps({
      'original_graph_sha256': hashlib.sha256(graph.read_bytes()).hexdigest(),
      'checks': [{'operator': 'operation', 'original_tensor': 'y'}]}))
    (folder / 'model.param').write_text('7767517\n1 2\n' + operation + ' operation 1 1 in out\n')
    return report

  def test_nonlinear_reduction_and_cast_use_actual_captured_input(self):
    for operation in ('TanH', 'Reduction', 'Cast'):
      with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        report = self.fixture(root, operation)
        with patch.object(module, 'ROOT', root):
          result = module.analyze(report)
        self.assertFalse(result['qualified'])
        self.assertTrue(result['diagnostic_only'])
        self.assertTrue(result['checks'][0]['parity']['passed'])
        if operation == 'Cast':
          self.assertEqual(result['checks'][0]['exact_elements'], 4)

  def test_changed_capture_is_rejected_before_numerical_comparison(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      report = self.fixture(root, 'TanH')
      (report.with_suffix('') / 'in.bin').write_bytes(b'changed')
      with patch.object(module, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Captured tensor changed'):
        module.analyze(report)

  def test_multiple_captured_inputs_keep_local_error_separate_from_upstream_error(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      report = self.fixture(root, 'TanH')
      x = np.array([[1., .5, -.125, 2.]], np.float32)
      z = np.array([[.25, 2., 4., -1.]], np.float32)
      tensor = onnx.helper.make_tensor_value_info
      graph = onnx.helper.make_model(onnx.helper.make_graph(
        [onnx.helper.make_node('Mul', ['x', 'z'], ['y'])], 'gating',
        [tensor(name, onnx.TensorProto.FLOAT, x.shape) for name in ('x', 'z')],
        [tensor('y', onnx.TensorProto.FLOAT, x.shape)]),
        opset_imports=[onnx.helper.make_opsetid('', 17)], ir_version=9)
      model_path = root / 'android/app/build/model-reference/accelerators/dmonitoring_model.onnx'
      onnx.save(graph, model_path)
      manifest = json.loads(report.read_text())
      for item, value in zip(manifest['tensors'], (x, x * z), strict=True):
        raw = value.tobytes()
        (report.with_suffix('') / item['file']).write_bytes(raw)
        item['sha256'] = hashlib.sha256(raw).hexdigest()
      raw = z.tobytes()
      (report.with_suffix('') / 'second.bin').write_bytes(raw)
      manifest['tensors'].append({'blob': 'second', 'operator': 'second', 'file': 'second.bin', 'sha256': hashlib.sha256(raw).hexdigest()})
      report.write_text(json.dumps(manifest))
      report.with_name('trace-comparison.json').write_text(json.dumps({
        'original_graph_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'checks': [{'operator': op, 'original_tensor': name} for op, name in [('input', 'x'), ('second', 'z'), ('operation', 'y')]]}))
      (report.with_suffix('') / 'model.param').write_text('7767517\n1 3\nBinaryOp operation 2 1 in second out 0=2\n')
      with patch.object(module, 'ROOT', root):
        result = module.analyze(report)
      check = result['checks'][0]
      self.assertEqual(len(check['captured_inputs']), 2)
      self.assertTrue(check['parity']['exact'])
      self.assertEqual(check['half_rounding_disagreements'], 0)
      self.assertFalse(result['qualified'])


if __name__ == '__main__':
  unittest.main()
