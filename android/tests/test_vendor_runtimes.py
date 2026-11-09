"""Reject corrupt distributions and HTTP servers ignoring byte ranges."""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_vendor_runtimes import RemoteZip, checked_digest


class Response(io.BytesIO):
  def __init__(self, data=b'', status=200, **headers):
    super().__init__(data); self.status, self.headers = status, headers


class VendorRuntimeTest(unittest.TestCase):
  def test_pinned_runtime_corruption_rejected(self):
    with self.assertRaisesRegex(ValueError, 'hash mismatch'):
      checked_digest('litert-2.3.0.aar', b'corrupt')

  def test_range_cache_and_exact_seek(self):
    responses = [Response(**{'Content-Length': '6'}),
                 Response(b'abcdef', 206, **{'Content-Range': 'bytes 0-5/6'})]
    with patch('urllib.request.urlopen', side_effect=responses) as call:
      source = RemoteZip('https://example.invalid/sdk.zip')
      self.assertEqual(source.read(2), b'ab')
      source.seek(-2, 2)
      self.assertEqual(source.read(2), b'ef')
      self.assertEqual(call.call_count, 2)
      self.assertEqual(source.read(1), b'')
      with self.assertRaises(ValueError): source.seek(-1)

  def test_full_download_response_rejected(self):
    with patch('urllib.request.urlopen', side_effect=[Response(**{'Content-Length': '6'}), Response(b'abcdef')]):
      with self.assertRaisesRegex(ValueError, 'exact SDK range'): RemoteZip('https://example.invalid').read(2)

  def test_truncated_range_rejected(self):
    with patch('urllib.request.urlopen', side_effect=[Response(**{'Content-Length': '6'}),
        Response(b'abc', 206, **{'Content-Range': 'bytes 0-5/6'})]):
      with self.assertRaisesRegex(ValueError, 'Truncated'): RemoteZip('https://example.invalid').read(2)


if __name__ == '__main__': unittest.main()
