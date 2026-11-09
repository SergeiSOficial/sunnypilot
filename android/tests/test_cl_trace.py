"""A stopped GPU launch must leave its exact checkpoint on disk."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app/src/main/python"))
from fold3_cl_trace import checkpoints


class CheckpointTest(unittest.TestCase):
  def check_launch(self, fail):
    class Program:
      def __init__(self, device, obj):
        self.fail = device

      def __call__(self, *buffers, **kwargs):
        self.wait = kwargs["wait"]
        if self.fail:
          raise RuntimeError("GPU event failed")
        return .0012

    original_init, original_call = Program.__init__, Program.__call__
    source = b"kernel void stalled_kernel() {}"
    modules = {"tinygrad.runtime.ops_cl": SimpleNamespace(CLProgram=Program),
               "fold3_models": SimpleNamespace(_thermal_check=lambda: None)}
    with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, modules):
      directory = Path(temporary)
      with checkpoints(directory):
        program = Program(fail, SimpleNamespace(lib=source, name="stalled_kernel"))
        if fail:
          with self.assertRaisesRegex(RuntimeError, "GPU event failed"):
            program(global_size=(128, 1, 1), local_size=(32, 1, 1))
        else:
          self.assertEqual(program(global_size=(128, 1, 1), local_size=(32, 1, 1)), .0012)
        self.assertTrue(program.wait)
      self.assertIs(Program.__init__, original_init)
      self.assertIs(Program.__call__, original_call)
      latest = json.loads((directory / "latest.json").read_text())
      self.assertEqual(latest["sequence"], 1)
      self.assertEqual(latest["kernel"], "stalled_kernel")
      self.assertEqual(latest["global_size"], [128, 1, 1])
      self.assertEqual(latest["local_size"], [32, 1, 1])
      digest = hashlib.sha256(source).hexdigest()
      self.assertEqual(latest["source_sha256"], digest)
      self.assertEqual((directory / f"{digest}.cl").read_bytes(), source)
      if fail:
        self.assertEqual(latest["status"], "waiting")
        self.assertFalse((directory / "completed.jsonl").exists())
      else:
        self.assertEqual(latest["status"], "completed")
        self.assertEqual(latest["serialized_event_ms"], 1.2)
        self.assertEqual(json.loads((directory / "completed.jsonl").read_text()), latest)

  def test_completed_launch(self):
    self.check_launch(False)

  def test_failed_launch_retains_waiting_checkpoint(self):
    self.check_launch(True)


if __name__ == "__main__":
  unittest.main()
