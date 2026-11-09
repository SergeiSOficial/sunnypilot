import hashlib
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from convert_litert_isolated import convert


class ConversionIsolationTest(unittest.TestCase):
  def test_mutating_converter_cannot_modify_original(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      source = root / 'model.onnx'
      source.write_bytes(b'original')
      converter = root / 'converter'
      converter.write_text('#!/usr/bin/env python3\nimport pathlib,sys\n'
                           'a=sys.argv\np=pathlib.Path(a[a.index("-i")+1])\np.write_bytes(b"changed")\n'
                           'o=pathlib.Path(a[a.index("-o")+1]);o.mkdir();(o/"out.tflite").write_bytes(b"model")\n')
      converter.chmod(0o755)
      result = convert(source, hashlib.sha256(b'original').hexdigest(), converter, root / 'out')
      self.assertEqual(source.read_bytes(), b'original')
      self.assertTrue(result['staged_input_mutated'])
      self.assertEqual(result['status'], 'converted')
      self.assertTrue((root / 'out' / 'receipt.json').exists())

  def test_unexpected_source_hash_is_rejected_before_output_creation(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      source = root / 'model.onnx'
      source.write_bytes(b'original')
      with self.assertRaises(ValueError):
        convert(source, 'wrong', root / 'converter', root / 'out')
      self.assertFalse((root / 'out').exists())

  def test_success_without_a_model_is_a_failure(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      source = root / 'model.onnx'
      source.write_bytes(b'original')
      converter = root / 'converter'
      converter.write_text('#!/bin/sh\nexit 0\n')
      converter.chmod(0o755)
      result = convert(source, hashlib.sha256(b'original').hexdigest(), converter, root / 'out')
      self.assertEqual(result['status'], 'failed')
