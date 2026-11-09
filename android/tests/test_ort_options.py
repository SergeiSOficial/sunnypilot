"""Provider diagnostics must preserve the requested backend and array ownership."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np


class OrtOptionsTest(unittest.TestCase):
  def test_profile_and_provider_options_reach_native_session_without_changing_inputs(self):
    native = Mock()
    native.run.return_value = {"y": ([1], "float32", np.array([2.], np.float32).tobytes())}
    location = Path(__file__).resolve().parents[1] / "app/src/main/python/fold3_ort.py"
    spec = importlib.util.spec_from_file_location("checked_ort", location)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"fold3_ort_native": native}):
      spec.loader.exec_module(module)
    options = {"profiling_level": "detailed"}
    model = module.OrtModel("model.onnx", "qnn-gpu", library_dir="/libs", no_cpu_fallback=True,
                            provider_options=options, profile_path=Path("/data/profile"))
    native.create.assert_called_once_with("model.onnx", "qnn-gpu", 2, False, False, "/libs", False, True,
                                          options, "/data/profile")
    x = np.arange(8, dtype=np.float16)[::2]
    np.testing.assert_array_equal(model({"x": x})["y"], [2.])
    passed = native.run.call_args.args[1]["x"]
    self.assertTrue(passed.flags.c_contiguous)
    self.assertEqual(passed.dtype, x.dtype)
    np.testing.assert_array_equal(passed, x)
    model.end_profiling()
    native.end_profiling.assert_called_once_with(native.create.return_value)
