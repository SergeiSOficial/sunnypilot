"""Converter receipts and original-reference probe semantics."""
import hashlib
import shlex
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from convert_snpe_linux import digest
from prepare_snpe_probes import prepare
from setup_snpe_linux import environment
from validate_snpe_sections import contracts, remote_shell
from prepare_snpe_sections import required_inputs
from onnx import helper as h, TensorProto as T
from prepare_snpe_dense_probe import ordered_half_gemv


class SnpeToolsTest(unittest.TestCase):
  def test_adb_shell_preserves_graph_and_output_argument(self):
    args = ['env', 'LD_LIBRARY_PATH=/data/local/tmp/test', 'snpe-net-run',
            '--set_output_tensors', 'vision model/vision_outputs,next_state_img_q', '--runtime_order', 'gpu']
    command = remote_shell(args)
    self.assertEqual(command[0], 'shell')
    self.assertEqual(shlex.split(command[1]), args)

  def test_half_accumulation_is_not_float_accumulation_with_output_cast(self):
    x = np.ones((1,16), np.float16)
    weight = np.zeros((1,16), np.float16); weight[0,0] = 2048; weight[0,8] = 1
    bias = np.ones(1, np.float16)
    self.assertEqual(float(ordered_half_gemv(x, weight, bias)[0,0]), 2048.)
    widened = (x.astype(np.float32) @ weight.astype(np.float32).T + bias).astype(np.float16)
    self.assertEqual(float(widened[0,0]), 2050.)
    with self.assertRaisesRegex(ValueError, 'Unsupported ordered-half'):
      ordered_half_gemv(x[:,:15], weight[:,:15], bias)

  def test_unused_input_is_not_bound_to_pruned_dlc(self):
    model = h.make_model(h.make_graph([
      h.make_node('Identity', ['active'], ['result']),
      h.make_node('Cast', ['unused'], ['dead_cast'], to=T.FLOAT),
    ], 'pruned', [h.make_tensor_value_info(n, T.FLOAT, [1]) for n in ('active', 'unused')],
      [h.make_tensor_value_info('result', T.FLOAT, [1])]))
    self.assertEqual(required_inputs(model), {'active'})

  def test_digest_streaming_python310_compatible(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp)/'artifact'; data = b'original' * 200000; path.write_bytes(data)
      self.assertEqual(digest(path), hashlib.sha256(data).hexdigest())

  def test_explicit_rounding_and_layout_references(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp); manifest = prepare(root)
      self.assertEqual(len(manifest['probes']), 9)
      self.assertFalse(manifest['qualified']); self.assertFalse(manifest['vehicle_tx_enabled'])
      with np.load(root/'fp32-cast.npz') as fixture:
        np.testing.assert_array_equal(fixture['expected'], fixture['x'].astype(np.float16).astype(np.float32))
        self.assertTrue(np.any(fixture['expected'] != fixture['x']))
      with np.load(root/'fp16-conv.npz') as fixture:
        raw = np.fromfile(root/'fp16-conv-nhwc.raw', np.float32).reshape(1, 8, 8, 3)
        np.testing.assert_array_equal(raw, fixture['x'].transpose(0, 2, 3, 1))

  def test_dlc_contract_tables_do_not_use_internal_shapes(self):
    info = """internal tensor [1,8,8,3]
| Input Name | Dimensions | Type | Encoding Info |
| x | 1,6,6,4 | Float_32 | None |
| Output Name | Dimensions | Type | Encoding Info |
| y | 1,8 | Float_32 | None |
Total parameters: 0
"""
    self.assertEqual(contracts(info, 'Input'), {'x': [1,6,6,4]})
    self.assertEqual(contracts(info, 'Output'), {'y': [1,8]})
    with self.assertRaises(IndexError): contracts('no input contract', 'Input')

  def test_private_loader_environment(self):
    with tempfile.TemporaryDirectory() as tmp:
      sdk, env = environment(Path(tmp))
      self.assertTrue(str(sdk).startswith(tmp))
      self.assertIn(str(Path(tmp)/'sysroot'), env['LD_LIBRARY_PATH'])
      self.assertEqual(env['PYTHONPATH'], str(sdk/'lib/python'))


if __name__ == '__main__': unittest.main()
