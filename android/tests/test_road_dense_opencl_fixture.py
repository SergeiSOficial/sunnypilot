import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'android/scripts'))
from prepare_road_dense_opencl_probe import prepare
from prepare_snpe_dense_probe import ordered_half_gemv
from validate_road_dense_opencl import fixture_paths

CAPTURE = ROOT / 'android/build/candidates/candidate74-runtime/follow-up/snpe-dense-isolated-original-capture'


@unittest.skipUnless((CAPTURE / 'fixtures/manifest.json').exists(), 'captured dense fixture unavailable')
class RoadDenseFixtureTest(unittest.TestCase):
  def test_captured_weights_reproduce_original_and_cannot_overwrite(self):
    source = CAPTURE / 'graphs/dense.onnx'
    fixtures = CAPTURE / 'fixtures'
    with tempfile.TemporaryDirectory() as temporary:
      output = Path(temporary) / 'fixture'
      report = prepare(source, fixtures, output)
      self.assertEqual(len(fixture_paths(output)), 4)
      self.assertFalse(report['qualified'])
      self.assertFalse(report['vehicle_tx_enabled'])
      x = np.fromfile(output / 'input.f32', np.float32).reshape(1, 1024)
      graph = onnx.load(source)
      node = next(n for n in graph.graph.node if n.op_type == 'Gemm')
      values = {v.name: numpy_helper.to_array(v) for v in graph.graph.initializer}
      actual = ordered_half_gemv(x, values[node.input[1]], values[node.input[2]]).astype(np.float32)
      np.testing.assert_array_equal(actual.reshape(-1), np.fromfile(output / 'expected.f32', np.float32))
      with self.assertRaisesRegex(ValueError, 'already exists'):
        prepare(source, fixtures, output)
      manifest_path = output / 'manifest.json'
      manifest = json.loads(manifest_path.read_text())
      manifest['files']['input']['file'] = '../input.f32'
      manifest_path.write_text(json.dumps(manifest))
      with self.assertRaisesRegex(ValueError, 'filename'):
        fixture_paths(output)

  def test_rejects_changed_capture_before_output_creation(self):
    with tempfile.TemporaryDirectory() as temporary:
      folder = Path(temporary) / 'fixtures'
      (folder / 'dense').mkdir(parents=True)
      original = json.loads((CAPTURE / 'fixtures/manifest.json').read_text())
      (folder / 'manifest.json').write_text(json.dumps(original))
      (folder / 'dense/input.raw').write_bytes(b'changed')
      (folder / 'dense/reference.npz').write_bytes(b'changed')
      output = Path(temporary) / 'output'
      with self.assertRaisesRegex(ValueError, 'provenance'):
        prepare(CAPTURE / 'graphs/dense.onnx', folder, output)
      self.assertFalse(output.exists())


if __name__ == '__main__':
  unittest.main()
