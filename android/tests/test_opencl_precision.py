"""The bounded GPU Gemm must preserve the pinned arm64 half rounding order."""
from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tinygrad_repo"))
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_opencl import fused_ops


class OpenCLPrecisionTest(unittest.TestCase):
  def test_trace_distinguishes_reference_attention_masks_from_numerical_faults(self):
    from fold3_model_trace import compare
    expected = np.array([0., -np.inf], dtype=np.float32)
    matched = compare(expected.copy(), expected, allow_masked_infinity=True)
    self.assertTrue(matched["passed"])
    self.assertEqual(matched["unexpected_nonfinite_elements"], 0)
    for actual in ([np.inf, -np.inf], [np.nan, -np.inf], [0., np.inf], [0., 0.]):
      self.assertFalse(compare(np.array(actual, dtype=np.float32), expected, allow_masked_infinity=True)["passed"])

  def test_android_renderer_uses_rte_for_scalar_and_vector_half_casts(self):
    from tinygrad import UOp, dtypes
    from tinygrad.helpers import Target
    from tinygrad.uop.ops import Ops
    with tempfile.TemporaryDirectory() as temporary:
      for relative in ("tinygrad/tensor.py", "tinygrad/runtime/ops_disk.py", "tinygrad/renderer/cstyle.py"):
        target = Path(temporary) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / "tinygrad_repo" / relative).read_bytes())
      subprocess.run(["git", "apply", str(ROOT / "android/patches/tinygrad-android.patch")], cwd=temporary, check=True)
      spec = importlib.util.spec_from_file_location("fold3_cl_renderer", Path(temporary) / "tinygrad/renderer/cstyle.py")
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
      renderer = module.OpenCLRenderer(Target("CL"))
      for width in (1, 2, 4, 8, 16):
        source = tuple(range(width)) if width > 1 else 1.
        value = UOp.const(source, dtypes.float32).cast(dtypes.float16)
        expression = renderer.render_cast(value, "x")
        suffix = str(width) if width > 1 else ""
        self.assertEqual(expression, f"fold3_half_rte{width}(x)")
        kernel = renderer.render_kernel("probe", [], [], [value])
        self.assertIn(f"convert_half{suffix}_rte(x)", kernel)
        self.assertIn(f"as_half{suffix}((ushort{suffix})", kernel)
        self.assertIn(f"convert_ushort{suffix}_rte(fabs(x) * 16777216.0f)", kernel)
        nested = "source_expression"
        for _ in range(8):
          nested = renderer.render_cast(value, nested)
        self.assertEqual(nested.count("source_expression"), 1)
        self.assertLess(len(nested), 200)
      # Custom matrix operations have no address space. Their scalar casts
      # need the same rounding helper as ordinary ALU expressions.
      custom = UOp(Ops.CUSTOM, src=(UOp.const(1., dtypes.float32),), arg=("{0}", dtypes.float32)).cast(dtypes.float16)
      self.assertIsNone(custom.addrspace)
      self.assertEqual(renderer.render_cast(custom, "x"), "fold3_half_rte1(x)")
      self.assertIn("static inline half fold3_half_rte1", renderer.render_kernel("custom_probe", [], [], [custom]))

  @unittest.skipUnless(sys.platform == "darwin", "Host Metal checks the native half fma; device OpenCL is checked separately")
  def test_all_finite_half_values_and_midpoint_neighbors(self):
    from fold3_model_trace import half_conversion_diagnostic
    result = half_conversion_diagnostic("METAL")
    self.assertTrue(result["passed"], result)
    self.assertGreater(result["subnormal_values"], 2046)

  @unittest.skipUnless(sys.platform == "darwin", "Host Metal checks the native half fma; device OpenCL is checked separately")
  def test_bounded_gemm_matches_half_fma_lanes_reduction_tail_and_bias(self):
    from tinygrad import Tensor
    runner = SimpleNamespace(onnx_ops={})
    fused_ops(runner)
    for seed in (42, 7, 123):
      random = np.random.default_rng(seed)
      for k in (8, 19, 64, 512, 1024, 513):
        with self.subTest(seed=seed, k=k):
          a = random.normal(size=(1, k)).astype(np.float16)
          b = random.normal(size=(32, k)).astype(np.float16)
          if k == 513:
            b *= np.float16(0.015625)
          bias = random.normal(size=(32,)).astype(np.float16)
          end = k // 8 * 8
          lanes = np.zeros((32, 8), dtype=np.float16)
          for index in range(0, end, 8):
            lanes = (a[:, index:index + 8].astype(np.float64) * b[:, index:index + 8].astype(np.float64)
                     + lanes.astype(np.float64)).astype(np.float16)
          for width in (4, 2, 1):
            lanes = (lanes[:, ::2] + lanes[:, 1::2]).reshape(32, width)
          expected = lanes.reshape(1, 32)
          for index in range(end, k):
            expected = (expected + (a[0, index] * b[:, index]).astype(np.float16)).astype(np.float16)
          expected = (expected + bias).astype(np.float16)
          values = [Tensor(value, device="METAL") for value in (a, b, bias)]
          actual = runner.onnx_ops["Gemm"](*values, transB=1).numpy()
          np.testing.assert_array_equal(actual, expected)
          fused = runner.onnx_ops["FusedGemm"](*values, transB=1, activation="Relu").numpy()
          np.testing.assert_array_equal(fused, np.maximum(expected, 0))

  def test_other_gemm_shapes_use_the_pinned_operator(self):
    from tinygrad import Tensor
    sentinel = object()
    calls = []
    def original(*args, **kwargs):
      calls.append((args, kwargs))
      return sentinel
    runner = SimpleNamespace(onnx_ops={"Gemm": original})
    fused_ops(runner)
    a = Tensor.zeros(2, 8, dtype="float16", device="CPU")
    b = Tensor.zeros(32, 8, dtype="float16", device="CPU")
    self.assertIs(runner.onnx_ops["Gemm"](a, b, transB=1), sentinel)
    self.assertEqual(len(calls), 1)


if __name__ == "__main__":
  unittest.main()
