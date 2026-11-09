"""Regression checks for reference arithmetic and ONNX shape semantics."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tinygrad_repo"))
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_opencl import fused_ops


@unittest.skipUnless(sys.platform == "darwin", "Metal arithmetic; Android OpenCL is validated on the phone")
class HostMathTest(unittest.TestCase):
  def setUp(self):
    from tinygrad.nn.onnx import onnx_ops
    runner = SimpleNamespace(onnx_ops=onnx_ops)
    fused_ops(runner)
    self.ops = runner.onnx_ops

  @staticmethod
  def tensors(*values):
    from tinygrad import Tensor
    return [Tensor(value, device="METAL") for value in values]

  @staticmethod
  def reference(operator, values, **attributes):
    import onnxruntime as ort
    from onnx import helper
    inputs = [helper.make_tensor_value_info(key, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape)
              for key, value in values.items()]
    node = helper.make_node(operator, list(values), ["output"], **attributes)
    graph = helper.make_graph([node], "arithmetic-regression", inputs, [helper.make_tensor_value_info("output", inputs[0].type.tensor_type.elem_type, None)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(None, values)[0]

  def test_convolution_channel_tiles_preserve_groups_padding_stride_and_dilation(self):
    from tinygrad.nn.onnx import onnx_ops
    runner = SimpleNamespace(onnx_ops=onnx_ops)
    fused_ops(runner, diagnostic_mad=True)
    self.ops = runner.onnx_ops
    random = np.random.default_rng(42)
    for group, channels in ((1, 8), (2, 8), (2, 6)):
      for strides, dilations in (((1, 1), (1, 1)), ((2, 1), (1, 2))):
        with self.subTest(group=group, channels=channels, strides=strides, dilations=dilations):
          values = {"x": random.normal(size=(2, 4, 9, 11)).astype(np.float32),
                    "w": random.normal(size=(channels, 4 // group, 3, 3)).astype(np.float32),
                    "b": random.normal(size=channels).astype(np.float32)}
          attributes = {"group": group, "pads": (1, 2, 0, 1), "strides": strides, "dilations": dilations}
          expected = self.reference("Conv", values, **attributes)
          actual = self.ops["FusedConv"](*self.tensors(*values.values()), **attributes).numpy()
          np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)

  def test_half_gemv_selects_eigen_scalar_coefficient_and_fma_paths(self):
    random = np.random.default_rng(7)
    for width, columns in ((7, 1), (8, 1), (16, 1), (24, 1), (32, 1), (64, 1), (67, 1),
                           (16, 2), (16, 3), (8, 10), (8, 11), (19, 3)):
      with self.subTest(width=width, columns=columns):
        values = {"a": random.normal(size=(1, width)).astype(np.float16),
                  "b": random.normal(size=(columns, width)).astype(np.float16),
                  "c": random.normal(size=columns).astype(np.float16)}
        expected = self.reference("Gemm", values, transB=1)
        actual = self.ops["Gemm"](*self.tensors(*values.values()), transB=1).numpy()
        np.testing.assert_array_equal(actual, expected)

  def test_concat_preserves_subnormal_signed_zero_and_nan_bits(self):
    for dtype, unsigned in ((np.float16, np.uint16), (np.float32, np.uint32)):
      # Unequal branch sizes reproduce the zero-padded arithmetic cat path.
      bits = np.array([0, 1, 2, 511, 1023, 1024], dtype=unsigned)
      sign = np.array([1 << (np.dtype(unsigned).itemsize * 8 - 1)], dtype=unsigned)
      bits = np.concatenate((bits, bits | sign,
                             np.array([np.inf, -np.inf, np.nan], dtype=dtype).view(unsigned)))
      values = [bits[:5].view(dtype).reshape(1, -1), bits[5:].view(dtype).reshape(1, -1)]
      for axis in (0, 1, -1):
        arrays = [value.T if axis == 0 else value for value in values]
        expected = np.concatenate(arrays, axis=axis)
        actual = self.ops["Concat"](*self.tensors(*arrays), axis=axis).numpy()
        np.testing.assert_array_equal(actual.view(unsigned), expected.view(unsigned))

  def test_normalization_sqrt_preserves_fp32_rounding_and_ieee_fallback(self):
    from tinygrad import Tensor, dtypes
    from tinygrad.uop.ops import KernelInfo
    from fold3_host_math import output_range, rounded_sqrt
    random = np.random.default_rng(42)
    values = random.integers(0x00800000, 0x7f800000, 65536, dtype=np.uint32).view(np.float32)
    values[:8] = [0., -0., np.inf, np.nan, -1., np.finfo(np.float32).tiny, np.finfo(np.float32).max, 1.]

    def kernel(dest, source):
      index, axes = output_range(dest.numel())
      value = rounded_sqrt(source.flatten()[index].load(), True)
      return dest.flatten()[index].set(value, end=axes).sink(arg=KernelInfo(name="checked_sqrt", opts_to_apply=()))

    source = Tensor(values, device="METAL").realize()
    actual = Tensor.empty(values.shape, device="METAL", dtype=dtypes.float32).custom_kernel(source, fxn=kernel)[0].numpy()
    with np.errstate(invalid="ignore"):
      expected = np.sqrt(values)
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint32), expected[finite].view(np.uint32))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))

  def test_exponential_matches_mlas_including_attention_range_and_ieee_faults(self):
    random = np.random.default_rng(42)
    x = np.concatenate((random.uniform(-105, 90, 2048).astype(np.float32),
                        random.uniform(-12, 0, 2048).astype(np.float32),
                        np.array([-np.inf, np.inf, np.nan, 0.], dtype=np.float32)))
    expected = self.reference("Exp", {"x": x})
    actual = self.ops["Exp"](*self.tensors(x)).numpy()
    normal = (np.abs(expected) >= np.finfo(np.float32).tiny) | (expected == 0) | ~np.isfinite(expected)
    np.testing.assert_array_equal(actual[normal], expected[normal])
    # The GPU may flush float32 subnormals. They remain far below the original
    # absolute tolerance; normal attention values must match bit for bit.
    np.testing.assert_allclose(actual[~normal], expected[~normal], rtol=1e-3, atol=1e-4)

  def test_division_broadcasts_ranks_and_preserves_ieee_faults(self):
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    for b in (np.array(3., dtype=np.float32), np.array([[2.], [3.], [7.]], dtype=np.float32)):
      actual = self.ops["Div"](*self.tensors(a, b)).numpy()
      np.testing.assert_array_equal(actual, a / b)
    a = np.array([1., 0., np.inf, -1., 1.], dtype=np.float32)
    b = np.array([0., 0., 2., 0., np.inf], dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
      np.testing.assert_array_equal(self.ops["Div"](*self.tensors(a, b)).numpy(), a / b)
    # First differing operation in the Fold 3 driver-camera replay, frame 41:
    # this MLAS tanh quotient lies next to a float32 rounding midpoint.
    a = np.array([0.0033100165892392397], dtype=np.float32)
    b = np.array([0.005847712513059378], dtype=np.float32)
    np.testing.assert_array_equal(self.ops["Div"](*self.tensors(a, b)).numpy(), a / b)
    random = np.random.default_rng(42)
    a, b = [np.ldexp(random.uniform(-1, 1, 4096).astype(np.float32),
                     random.integers(-30, 31, 4096)).astype(np.float32) for _ in range(2)]
    np.testing.assert_array_equal(self.ops["Div"](*self.tensors(a, b)).numpy(), a / b)

  def test_attention_sum_preserves_row_alignment_packets_prefix_and_tail(self):
    random = np.random.default_rng(42)
    for width in (8, 9, 10, 11, 12, 16, 19, 32, 67):
      with self.subTest(width=width):
        x = random.normal(size=(2, 3, width)).astype(np.float32)
        axes = np.array([-1], dtype=np.int64)
        for keepdims in (0, 1):
          expected = self.reference("ReduceSum", {"x": x, "axes": axes}, keepdims=keepdims)
          actual = self.ops["ReduceSum"](*self.tensors(x, axes), keepdims=keepdims).numpy()
          np.testing.assert_array_equal(actual, expected)

  def test_scalar_square_and_cube_match_reference_rounding(self):
    from tinygrad import Tensor
    random = np.random.default_rng(42)
    x = random.normal(size=(4, 128)).astype(np.float32)
    for exponent in (2., 3.):
      with self.subTest(exponent=exponent):
        expected = self.reference("Pow", {"x": x, "y": np.array(exponent, dtype=np.float32)})
        actual = self.ops["Pow"](self.tensors(x)[0], Tensor(exponent, device="METAL")).numpy()
        np.testing.assert_array_equal(actual, expected)
    # A final singleton selects ORT's powf broadcast loop rather than its
    # explicit-multiply loop. Both retain the original numerical tolerances.
    x = x.reshape(1, -1, 1, 1)
    expected = self.reference("Pow", {"x": x, "y": np.array(3., dtype=np.float32)})
    actual = self.ops["Pow"](self.tensors(x)[0], Tensor(3., device="METAL")).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)

  def test_half_attention_matmul_rounds_pairs_before_accumulation(self):
    random = np.random.default_rng(42)
    for width in (1, 9, 64):
      with self.subTest(width=width):
        values = {"a": random.normal(size=(1, 3, 9, width)).astype(np.float16),
                      "b": random.normal(size=(1, 1, width, 64)).astype(np.float16)}
        expected = self.reference("MatMul", values)
        actual = self.ops["MatMul"](*self.tensors(*values.values())).numpy()
        np.testing.assert_array_equal(actual, expected)

  def test_road_frame26_attention_preserves_pair_boundary_and_odd_tail(self):
    # Seed42 recurrent frame26, head4, row8, column27: a single half-ULP
    # attention error amplifies into 74 full-model tolerance violations.
    a = np.array([.30126953125, .2410888671875, .1651611328125, .133056640625,
                  .0926513671875, .0269927978515625, .00543212890625,
                  .00787353515625, .0264739990234375], np.float16).reshape(1, 1, 1, 9)
    b = np.array([-.07586669921875, -.006633758544921875, -.08929443359375,
                  .09002685546875, -.00682830810546875, .00519561767578125,
                  .05828857421875, .198974609375, .30810546875], np.float16).reshape(1, 1, 9, 1)
    expected = self.reference("MatMul", {"a": a, "b": b})
    actual = self.ops["MatMul"](*self.tensors(a, b)).numpy()
    np.testing.assert_array_equal(actual, expected)
    sequential = np.float32(0)
    for x, y in zip(a.flatten(), b.flatten(), strict=True):
      sequential = np.float32(sequential + np.float32(x) * np.float32(y))
    self.assertNotEqual(np.float16(sequential), expected.item())

  def test_half_dot_rounds_each_pair_before_adding_to_accumulator(self):
    random = np.random.default_rng(123)
    for width in (1, 19, 512):
      with self.subTest(width=width):
        a = random.normal(size=(3, width)).astype(np.float16)
        b = random.normal(size=(width, 17)).astype(np.float16)
        bias = random.normal(size=17).astype(np.float16)
        expected = np.broadcast_to(bias, (3, 17)).astype(np.float32).copy()
        for index in range(0, width - 1, 2):
          pair = (a[:, index, None].astype(np.float64) * b[index].astype(np.float64)
                  + a[:, index + 1, None].astype(np.float64) * b[index + 1].astype(np.float64)).astype(np.float32)
          expected = (expected + pair).astype(np.float32)
        if width % 2:
          expected = (a[:, -1, None].astype(np.float64) * b[-1].astype(np.float64) + expected).astype(np.float32)
        actual = self.ops["Gemm"](*self.tensors(a, b, bias)).numpy()
        np.testing.assert_array_equal(actual, expected.astype(np.float16))

  def test_attention_matmul_broadcasts_and_accumulates_in_order(self):
    random = np.random.default_rng(7)
    a = random.normal(size=(2, 3, 19)).astype(np.float16)
    b = random.normal(size=(19, 5)).astype(np.float16)
    expected = np.zeros((2, 3, 5), dtype=np.float32)
    for index in range(19):
      expected = (a[:, :, index, None].astype(np.float64) * b[index].astype(np.float64) + expected).astype(np.float32)
    np.testing.assert_array_equal(self.ops["MatMul"](*self.tensors(a, b)).numpy(), expected.astype(np.float16))

  def test_half_dot_preserves_extreme_finite_products_and_pair_rounding(self):
    bits = np.arange(65536, dtype=np.uint16)
    bits = np.where((bits & 0x7c00) == 0x7c00, (bits & 0x8000) | 0x7bff, bits).astype(np.uint16)
    a = np.stack((bits, np.roll(bits, 1), np.roll(bits, 129)), axis=1).view(np.float16)
    weights = np.ldexp(np.arange(1, 18, dtype=np.float32), -14).astype(np.float16)
    b = np.stack((weights, -weights, weights / np.float16(2)))
    bias = ((np.arange(17) % 5 - 2) / 1024).astype(np.float16)
    pair = (a[:, 0, None].astype(np.float64) * b[0].astype(np.float64)
            + a[:, 1, None].astype(np.float64) * b[1].astype(np.float64)).astype(np.float32)
    for operator, initial in (("MatMul", np.zeros(17, dtype=np.float32)), ("Gemm", bias.astype(np.float32))):
      with self.subTest(operator=operator):
        expected = (pair + initial).astype(np.float32)
        expected = (a[:, 2, None].astype(np.float64) * b[2].astype(np.float64) + expected).astype(np.float32).astype(np.float16)
        values = (a, b) if operator == "MatMul" else (a, b, bias)
        actual = self.ops[operator](*self.tensors(*values)).numpy()
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))

  def test_layernorm_keeps_statistics_and_affine_in_float32(self):
    random = np.random.default_rng(42)
    x = random.normal(size=(2, 9, 512)).astype(np.float16)
    scale = random.normal(size=512).astype(np.float16)
    bias = random.normal(size=512).astype(np.float16)
    mean = np.zeros((2, 9), dtype=np.float32)
    m2 = np.zeros_like(mean)
    for index in range(512):
      delta = (x[:, :, index].astype(np.float32) - mean).astype(np.float32)
      mean = (mean + delta / np.float32(index + 1)).astype(np.float32)
      centered = (x[:, :, index].astype(np.float32) - mean).astype(np.float32)
      m2 = (delta.astype(np.float64) * centered.astype(np.float64) + m2).astype(np.float32)
    std = np.sqrt(m2 / np.float32(512) + np.float32(1e-5))
    normalized = (x.astype(np.float32) - mean[:, :, None]) / std[:, :, None]
    expected = (normalized.astype(np.float64) * scale.astype(np.float64) + bias.astype(np.float64)).astype(np.float32).astype(np.float16)
    actual, actual_mean, inverse = self.ops["LayerNormalization"](*self.tensors(x, scale, bias), epsilon=1e-5)
    np.testing.assert_array_equal(actual.numpy(), expected)
    np.testing.assert_array_equal(actual_mean.numpy(), mean[:, :, None])
    np.testing.assert_array_equal(inverse.numpy(), (1. / std)[:, :, None])


if __name__ == "__main__":
  unittest.main()
