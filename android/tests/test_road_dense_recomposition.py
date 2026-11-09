"""A copied or changed device result must not be accepted as GPU evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from decompose_road import digest
from validate_road_dense_recomposition import validate_case


class DenseDeviceProvenanceTest(unittest.TestCase):
  def test_tampered_device_output_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      fixtures, result = root/'fixtures', root/'result'
      fixtures.mkdir()
      result.mkdir()
      files = {}
      for name in ('input', 'weights', 'bias', 'expected'):
        path = fixtures/(name+'.f32')
        path.write_bytes(b'fixture')
        files[name] = {'file': path.name, 'sha256': digest(path)}
      (fixtures/'manifest.json').write_text(json.dumps({'files': files}))
      output = result/'actual.f32'
      output.write_bytes(b'original')
      receipt = {'status': 'pass', 'gpu_only': True, 'remote_directory_removed': True,
                 'fixture_manifest_sha256': digest(fixtures/'manifest.json'),
                 'device_output_sha256': digest(output)}
      (result/'results.json').write_text(json.dumps(receipt))
      output.write_bytes(b'modified')
      (root/'manifest.json').write_text('{}')
      with self.assertRaisesRegex(ValueError, 'provenance'):
        validate_case(root, fixtures, result, 42)


if __name__ == '__main__':
  unittest.main()
