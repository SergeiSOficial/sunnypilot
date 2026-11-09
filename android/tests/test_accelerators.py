"""Accelerator adapters and qualification must fail closed."""
import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
sys.path.insert(0, str(ROOT / "android/scripts"))
from fold3_accel import AccelModel, qualified
from fold3_models import trace_profile
from fold3_sequences import trace_inputs
from prepare_htp_probes import Reader, graph
from reference_precision import portable


class AcceleratorTest(unittest.TestCase):
  def test_pinned_mnn_converter_preserves_constant_division(self):
    converter = ROOT / "android/build/native/accelerators/mnn-patched/host/MNNConvert"
    if not converter.is_file():
      self.skipTest("Build the pinned MNN converter for its native conversion check")
    with tempfile.TemporaryDirectory() as temporary:
      folder = Path(temporary)
      model = h.make_model(h.make_graph([
        h.make_node("Div", ["x", "divisor"], ["y"], name="original_division")], "division",
        [h.make_tensor_value_info("x", T.FLOAT, [1, 4, 1, 1])],
        [h.make_tensor_value_info("y", T.FLOAT, [1, 4, 1, 1])],
        [onnx.numpy_helper.from_array(np.array([63.75, 255., .4, 3.], np.float32).reshape(1, 4, 1, 1), "divisor")]),
        opset_imports=[h.make_opsetid("", 21)])
      model.ir_version = 10
      onnx.save(model, folder / "division.onnx")
      subprocess.run([str(converter), "-f", "ONNX", "--modelFile", str(folder / "division.onnx"),
                      "--MNNModel", str(folder / "division.mnn"), "--bizCode", "fold3"],
                     check=True, capture_output=True, timeout=60)
      subprocess.run([str(converter), "-f", "MNN", "--modelFile", str(folder / "division.mnn"),
                      "--JsonFile", str(folder / "division.json")], check=True, capture_output=True, timeout=60)
      converted = json.loads((folder / "division.json").read_text())
      operations = [op["main"]["opType"] for op in converted["oplists"] if op["type"] == "BinaryOp"]
      self.assertEqual(operations, ["REALDIV"], "Reciprocal multiplication changes FP32 normalization")

  def test_selected_driver_frame_trace_is_explicit_and_bounded(self):
    self.assertEqual(trace_profile("mnn-opencl-buffer"), ("mnn-opencl-buffer", 0, None))
    self.assertEqual(trace_profile("trace-mnn-buffer:96"), ("trace-mnn-buffer", 96, None))
    self.assertEqual(trace_profile("trace-mnn-buffer:544:7"),
                     ("trace-mnn-buffer", 544, {"seed": 7, "frame": None}))
    self.assertEqual(trace_profile("trace-mnn-image:0:123"),
                     ("trace-mnn-image", 0, {"seed": 123, "frame": None}))
    self.assertEqual(trace_profile("trace-mnn-buffer:96:123:27"),
                     ("trace-mnn-buffer", 96, {"seed": 123, "frame": 27}))
    self.assertEqual(trace_profile("trace-mnn-road-buffer:544:42:26"),
                     ("trace-mnn-road-buffer", 544, {"seed": 42, "frame": 26}))
    for profile in ("mnn-opencl-buffer:96:123:27", "trace-mnn-buffer:4097", "trace-mnn-buffer:-1",
                    "trace-mnn-buffer:0:9", "trace-mnn-buffer:0:123:32", "trace-mnn-buffer:0:9:1"):
      with self.subTest(profile=profile), self.assertRaises(ValueError):
        trace_profile(profile)

  def test_selected_road_seed_keeps_original_recurrent_state(self):
    original = {"new_img": np.arange(12, dtype=np.uint8), "state_x": np.arange(6, dtype=np.float32)}
    with patch("fold3_sequences.initial_inputs", return_value=original) as initial:
      actual = trace_inputs("unused", "driving_supercombo", 7)
    initial.assert_called_once_with("unused", "driving_supercombo", 7)
    for key in original:
      np.testing.assert_array_equal(actual[key], original[key])
      self.assertFalse(np.shares_memory(actual[key], original[key]))

  def test_selected_driver_frame_keeps_fixture_inputs_and_rejects_reset_feedback(self):
    original = {"input_img": np.arange(128, dtype=np.uint8).reshape(1, 128),
                "calib": np.array([[.1, .2, .3]], dtype=np.float32)}
    with patch("fold3_sequences.initial_inputs", return_value=original):
      actual = trace_inputs("unused", "dmonitoring_model", 123, 27)
      np.testing.assert_array_equal(actual["input_img"], np.roll(original["input_img"], 27 * 17, axis=-1))
      np.testing.assert_array_equal(actual["calib"], original["calib"])
      self.assertFalse(np.shares_memory(actual["input_img"], original["input_img"]))
      for name, seed, frame in (("driving_supercombo", 123, 27), ("dmonitoring_model", 9, 0),
                                ("dmonitoring_model", True, 0), ("dmonitoring_model", 123, 32)):
        with self.subTest(name=name, seed=seed, frame=frame), self.assertRaises(ValueError):
          trace_inputs("unused", name, seed, frame)
    with patch("fold3_sequences.initial_inputs", return_value={**original, "state_x": np.zeros(1)}):
      with self.assertRaisesRegex(ValueError, "reset recurrent feedback"):
        trace_inputs("unused", "dmonitoring_model", 123, 27)

  def test_road_trace_warms_up_with_actual_backend_feedback(self):
    original = {"new_img": np.arange(128, dtype=np.uint8).reshape(1, 128),
                "state_x": np.zeros(1, np.float32)}
    seen = []
    def actual_backend(inputs):
      seen.append(float(inputs["state_x"][0]))
      return {"next_state_x": inputs["state_x"] + 3}
    progress = Mock()
    with patch("fold3_sequences.initial_inputs", return_value=original):
      actual = trace_inputs("unused", "driving_supercombo", 42, 26, actual_backend, progress)
    self.assertEqual(seen, list(range(0, 78, 3)))
    self.assertEqual(progress.call_count, 26)
    np.testing.assert_array_equal(actual["state_x"], [78])
    np.testing.assert_array_equal(actual["new_img"], np.roll(original["new_img"], 26 * 17, axis=-1))
    np.testing.assert_array_equal(original["state_x"], [0])

  def test_portable_residual_convolution_preserves_bias_rounding_order(self):
    import onnxruntime as ort
    shape = [1, 2, 1, 3]
    x = np.full(shape, 1e8, dtype=np.float32)
    weights = np.eye(2, dtype=np.float32).reshape(2, 2, 1, 1)
    bias = np.array([1., -1.], dtype=np.float32)
    for activation in (None, "Relu", "Tanh"):
      with self.subTest(activation=activation):
        attrs = {} if activation is None else {"activation": activation}
        original = h.make_model(h.make_graph([
          h.make_node("FusedConv", ["x", "w", "b", "residual"], ["y"], domain="com.microsoft", **attrs)],
          "residual_bias_order", [h.make_tensor_value_info(key, T.FLOAT, shape) for key in ("x", "residual")],
          [h.make_tensor_value_info("y", T.FLOAT, shape)],
          [onnx.numpy_helper.from_array(weights, "w"), onnx.numpy_helper.from_array(bias, "b")]),
          opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.microsoft", 1)], ir_version=10)
        inputs = {"x": x, "residual": -x}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        expected = ort.InferenceSession(original.SerializeToString(), options).run(None, inputs)[0]
        for internal_input in (False, True):
          with self.subTest(internal_input=internal_input):
            source = copy.deepcopy(original)
            if internal_input:
              source.graph.node[0].input[0] = "internal_x"
              source.graph.node.insert(0, h.make_node("Identity", ["x"], ["internal_x"]))
              self.assertEqual(len(source.graph.value_info), 0)
            converted = portable(source)
            actual = ort.InferenceSession(converted.SerializeToString(), options).run(None, inputs)[0]
            np.testing.assert_array_equal(actual, expected)
        # Bias-before-residual loses the unit bias to the large dot product.
        self.assertTrue(np.any(expected != 0))

  def test_spatial_mean_tags_preserve_values_and_external_contract(self):
    import onnxruntime as ort
    from prepare_accel_models import retain_host_spatial_means
    for dtype, shape, axes, count in ((T.FLOAT, [1, 3, 8, 12], [2, 3], 1),
        (T.FLOAT, [1, 3, 8, 12], [-2, -1], 1), (T.FLOAT16, [1, 3, 8, 12], [2, 3], 0),
        (T.FLOAT, [1, 3, 3, 3], [2, 3], 0), (T.FLOAT, [1, 3, 8, 12], [1, 2], 0)):
      output_shape = [1 if i in [a % 4 for a in axes] else d for i, d in enumerate(shape)]
      original = h.make_model(h.make_graph([
        h.make_node("ReduceMean", ["x", "axes"], ["mean"], name="spatial_mean"),
        h.make_node("Identity", ["mean"], ["y"])], "mean",
        [h.make_tensor_value_info("x", dtype, shape)], [h.make_tensor_value_info("y", dtype, output_shape)],
        [onnx.numpy_helper.from_array(np.array(axes, np.int64), "axes")]),
        opset_imports=[h.make_opsetid("", 21)])
      original.ir_version = 10
      converted, provenance = retain_host_spatial_means(original)
      self.assertEqual(len(provenance), count)
      self.assertEqual(converted.graph.input, original.graph.input)
      self.assertEqual(converted.graph.output, original.graph.output)
      x = np.random.default_rng(386).normal(size=shape).astype(np.float16 if dtype == T.FLOAT16 else np.float32)
      expected = ort.InferenceSession(original.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})
      actual = ort.InferenceSession(converted.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})
      np.testing.assert_array_equal(actual[0], expected[0])
      if count:
        self.assertTrue(converted.graph.node[0].output[0].startswith("fold3_host_mean8_"))
        self.assertEqual(converted.graph.node[1].input[0], converted.graph.node[0].output[0])
        collision = copy.deepcopy(original)
        collision.graph.node[1].output[0] = "fold3_host_mean8_mean"
        with self.assertRaisesRegex(ValueError, "collision"):
          retain_host_spatial_means(collision)

  def test_scalar_power_rewrite_preserves_host_results_and_broadcast_boundaries(self):
    import onnxruntime as ort
    from prepare_accel_models import retain_host_scalar_powers
    for dtype, shape, exponent, rewritten in ((T.FLOAT, [1, 2, 4], 3., True),
        (T.FLOAT, [2, 1], 3., False), (T.FLOAT, [2, 1], 2., True),
        (T.FLOAT16, [1, 2, 4], 3., False), (T.FLOAT, [None, 4], 3., False)):
      np_type = np.float16 if dtype == T.FLOAT16 else np.float32
      original = h.make_model(h.make_graph([
        h.make_node("Pow", ["x", "p"], ["y"], name="original_power")], "power",
        [h.make_tensor_value_info("x", dtype, shape)], [h.make_tensor_value_info("y", dtype, shape)],
        [onnx.numpy_helper.from_array(np.array(exponent, dtype=np_type), "p")]),
        opset_imports=[h.make_opsetid("", 21)])
      original.ir_version = 10
      converted, provenance = retain_host_scalar_powers(original)
      self.assertEqual(bool(provenance), rewritten)
      self.assertEqual(converted.graph.input, original.graph.input)
      self.assertEqual(converted.graph.output, original.graph.output)
      if rewritten:
        self.assertEqual(provenance[0]["original_operator"], "Pow")
        self.assertEqual(converted.graph.node[-1].name, "original_power")
        self.assertTrue(all(node.op_type == "Mul" for node in converted.graph.node))
      else:
        self.assertEqual(converted.graph.node[0].op_type, "Pow")
      values = np.random.default_rng(921).normal(size=[2 if d is None else d for d in shape]).astype(np_type)
      expected = ort.InferenceSession(original.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": values})
      actual = ort.InferenceSession(converted.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": values})
      np.testing.assert_array_equal(actual[0], expected[0])

  def test_internal_broadcast_cube_retains_pow_semantics_and_tags_gpu_kernel(self):
    import onnxruntime as ort
    from prepare_accel_models import retain_host_scalar_powers
    model = h.make_model(h.make_graph([
      h.make_node("Pow", ["x", "p"], ["cube"], name="original_cube"),
      h.make_node("Identity", ["cube"], ["y"])], "broadcast_cube",
      [h.make_tensor_value_info("x", T.FLOAT, [1, 16, 1, 1])],
      [h.make_tensor_value_info("y", T.FLOAT, [1, 16, 1, 1])],
      [onnx.numpy_helper.from_array(np.array(3., np.float32), "p")]),
      opset_imports=[h.make_opsetid("", 21)])
    model.ir_version = 10
    converted, provenance = retain_host_scalar_powers(model)
    self.assertEqual(provenance[0]["arithmetic"], "compensated_float32_cube")
    self.assertEqual(converted.graph.node[0].op_type, "Pow")
    self.assertEqual(converted.graph.node[0].output[0], "fold3_host_powf_cube_cube")
    self.assertEqual(converted.graph.node[1].input[0], converted.graph.node[0].output[0])
    self.assertEqual(converted.graph.input, model.graph.input)
    self.assertEqual(converted.graph.output, model.graph.output)
    x = np.random.default_rng(921).normal(size=[1, 16, 1, 1]).astype(np.float32)
    expected = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})
    actual = ort.InferenceSession(converted.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})
    np.testing.assert_array_equal(actual[0], expected[0])
    model.graph.node[1].output[0] = "fold3_host_powf_cube_cube"
    with self.assertRaisesRegex(ValueError, "collision"):
      retain_host_scalar_powers(model)

  def test_host_conv_order_tags_only_original_float_ops_without_changing_values(self):
    import onnxruntime as ort
    from prepare_accel_models import retain_host_conv_order
    weights = onnx.numpy_helper.from_array(np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2) / 4, "w")
    original = h.make_model(h.make_graph([
      h.make_node("Conv", ["x", "w"], ["conv"], name="original_conv"),
      h.make_node("Identity", ["conv"], ["y"])], "conv",
      [h.make_tensor_value_info("x", T.FLOAT, [1, 1, 3, 3])],
      [h.make_tensor_value_info("y", T.FLOAT, [1, 1, 2, 2])], [weights]), opset_imports=[h.make_opsetid("", 21)])
    original.ir_version = 10
    values = {"x": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)}
    expected = ort.InferenceSession(original.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, values)
    for fused, prefix in ((False, "fold3_host_conv_bl_"), (True, "fold3_host_conv_bf_")):
      host = copy.deepcopy(original)
      host.graph.node[0].op_type = "NhwcFusedConv" if fused else "Conv"
      converted = retain_host_conv_order(original, host)
      self.assertTrue(converted.graph.node[0].output[0].startswith(prefix))
      self.assertEqual(converted.graph.node[1].input[0], converted.graph.node[0].output[0])
      self.assertEqual(converted.graph.input[0].name, "x")
      self.assertEqual(converted.graph.output[0].name, "y")
      actual = ort.InferenceSession(converted.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, values)
      np.testing.assert_array_equal(actual[0], expected[0])
    native = retain_host_conv_order(original, original, native_convolutions=True)
    self.assertEqual(native.graph.node[0].output[0], "conv")
    self.assertEqual(native.graph.node[1].input[0], "conv")
    np.testing.assert_array_equal(
      ort.InferenceSession(native.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, values)[0], expected[0])
    self.assertEqual(original.graph.node[0].output[0], "conv")
    with self.assertRaisesRegex(ValueError, "external"):
      unsafe = copy.deepcopy(original)
      unsafe.graph.output[0].name = "conv"
      retain_host_conv_order(unsafe, original)

  def test_host_sigmoid_tags_require_original_fp32_provenance_and_full_vectors(self):
    import onnxruntime as ort
    from prepare_accel_models import retain_host_sigmoids
    for dtype, count, proven in ((T.FLOAT, 4, True), (T.FLOAT, 3, True),
                                 (T.FLOAT16, 4, True), (T.FLOAT, 4, False)):
      original = h.make_model(h.make_graph([
        h.make_node("Sigmoid", ["x"], ["sigmoid"], name="original_sigmoid"),
        h.make_node("Identity", ["sigmoid"], ["y"])], "sigmoid",
        [h.make_tensor_value_info("x", dtype, [1, count])],
        [h.make_tensor_value_info("y", dtype, [1, count])]), opset_imports=[h.make_opsetid("", 21)])
      original.ir_version = 10
      host = copy.deepcopy(original)
      if not proven:
        host.graph.node[0].op_type = "Tanh"
      converted, provenance = retain_host_sigmoids(original, host)
      tagged = dtype == T.FLOAT and count % 4 == 0 and proven
      self.assertEqual(converted.graph.node[0].output[0].startswith("fold3_host_sigmoid_"), tagged)
      self.assertEqual(len(provenance), int(tagged))
      self.assertEqual(converted.graph.input, original.graph.input)
      self.assertEqual(converted.graph.output, original.graph.output)
      values = {"x": np.linspace(-3, 3, count).reshape(1, count).astype(np.float32 if dtype == T.FLOAT else np.float16)}
      options = ort.SessionOptions()
      options.intra_op_num_threads = 2
      a = ort.InferenceSession(original.SerializeToString(), options).run(None, values)[0]
      b = ort.InferenceSession(converted.SerializeToString(), options).run(None, values)[0]
      np.testing.assert_array_equal(a, b)
      self.assertEqual(original.graph.node[0].output[0], "sigmoid")

  def test_host_sigmoid_recognizes_original_fused_conv_and_rejects_collision(self):
    from prepare_accel_models import retain_host_sigmoids
    original = h.make_model(h.make_graph([
      h.make_node("Conv", ["x", "w"], ["conv"], name="conv"),
      h.make_node("Sigmoid", ["conv"], ["sigmoid"]),
      h.make_node("Identity", ["sigmoid"], ["y"])], "sigmoid",
      [h.make_tensor_value_info("x", T.FLOAT, [1, 4, 1, 1])],
      [h.make_tensor_value_info("y", T.FLOAT, [1, 4, 1, 1])],
      [onnx.numpy_helper.from_array(np.eye(4, dtype=np.float32).reshape(4, 4, 1, 1), "w")]),
      opset_imports=[h.make_opsetid("", 21)])
    original.ir_version = 10
    host = copy.deepcopy(original)
    del host.graph.node[:]
    host.graph.node.append(h.make_node("FusedConv", ["x", "w"], ["sigmoid"],
                                      name="conv", domain="com.microsoft", activation="Sigmoid"))
    converted, provenance = retain_host_sigmoids(original, host)
    self.assertEqual(provenance[0]["original_operator"], "FusedConv")
    self.assertEqual(converted.graph.node[2].input[0], "fold3_host_sigmoid_sigmoid")
    converter = ROOT / "android/build/native/accelerators/mnn-patched/host/MNNConvert"
    if converter.is_file():
      with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        onnx.save(converted, folder / "sigmoid.onnx")
        for args in (["-f", "ONNX", "--modelFile", str(folder / "sigmoid.onnx"),
                      "--MNNModel", str(folder / "sigmoid.mnn"), "--bizCode", "fold3"],
                     ["-f", "MNN", "--modelFile", str(folder / "sigmoid.mnn"), "--JsonFile", str(folder / "sigmoid.json")]):
          subprocess.run([str(converter), *args], check=True, capture_output=True, timeout=60)
        operations = json.loads((folder / "sigmoid.json").read_text())["oplists"]
        sigmoids = [op for op in operations if op["type"] == "UnaryOp" and op["main"]["opType"] == "SIGMOID"]
        self.assertEqual(len(sigmoids), 1)
        self.assertEqual(sigmoids[0]["name"], "fold3_host_sigmoid_sigmoid")
    collision = copy.deepcopy(original)
    collision.graph.node.append(h.make_node("Identity", ["x"], ["fold3_host_sigmoid_sigmoid"]))
    with self.assertRaisesRegex(ValueError, "collision"):
      retain_host_sigmoids(collision, host)
    external = copy.deepcopy(original)
    external.graph.output[0].name = "sigmoid"
    _, provenance = retain_host_sigmoids(external, host)
    self.assertEqual(provenance, [])

  def test_host_gemv_order_preserves_cast_boundary_and_rejects_unverified_shapes(self):
    import onnxruntime as ort
    from prepare_accel_models import retain_host_conv_order
    for dtype, batch, trans_b, alpha in ((T.FLOAT, 1, 1, 1.), (T.FLOAT, 2, 1, 1.),
        (T.FLOAT, 1, 0, 1.), (T.FLOAT16, 1, 1, 1.), (T.FLOAT, 1, 1, .5)):
      np_type = np.float16 if dtype == T.FLOAT16 else np.float32
      w = np.arange(12, dtype=np_type).reshape(3, 4) / 32
      original = h.make_model(h.make_graph([
        h.make_node("Gemm", ["x", "w", "b"], ["gemv"], name="original_gemv", transB=trans_b, alpha=alpha),
        h.make_node("Relu", ["gemv"], ["relu"]),
        h.make_node("Cast", ["relu"], ["y"], to=T.FLOAT16)], "gemv",
        [h.make_tensor_value_info("x", dtype, [batch, 4])],
        [h.make_tensor_value_info("y", T.FLOAT16, [batch, 3])],
        [onnx.numpy_helper.from_array(w if trans_b else w.T.copy(), "w"),
         onnx.numpy_helper.from_array(np.array([.01, -.02, .03], np_type), "b")]),
        opset_imports=[h.make_opsetid("", 21)])
      original.ir_version = 10
      converted = retain_host_conv_order(original, original)
      tagged = dtype == T.FLOAT and batch == 1 and trans_b == 1 and alpha == 1.
      self.assertEqual(converted.graph.node[0].output[0].startswith("fold3_host_conv_bl_"), tagged)
      self.assertEqual(converted.graph.node[1].input[0], converted.graph.node[0].output[0])
      self.assertEqual(converted.graph.input, original.graph.input)
      self.assertEqual(converted.graph.output, original.graph.output)
      self.assertEqual(converted.graph.node[0].attribute, original.graph.node[0].attribute)
      self.assertEqual(converted.graph.initializer, original.graph.initializer)
      x = np.random.default_rng(426).normal(size=(batch, 4)).astype(np_type)
      options = ort.SessionOptions()
      options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
      expected = ort.InferenceSession(original.SerializeToString(), options).run(None, {"x": x})[0]
      actual = ort.InferenceSession(converted.SerializeToString(), options).run(None, {"x": x})[0]
      np.testing.assert_array_equal(actual, expected)
      self.assertEqual(original.graph.node[0].output[0], "gemv")
      converter = ROOT / "android/build/native/accelerators/mnn-patched/host/MNNConvert"
      if tagged and converter.is_file():
        with tempfile.TemporaryDirectory() as temporary:
          folder = Path(temporary)
          onnx.save(converted, folder / "gemv.onnx")
          for args in (["-f", "ONNX", "--modelFile", str(folder / "gemv.onnx"),
                        "--MNNModel", str(folder / "gemv.mnn"), "--bizCode", "fold3"],
                       ["-f", "MNN", "--modelFile", str(folder / "gemv.mnn"), "--JsonFile", str(folder / "gemv.json")]):
            subprocess.run([str(converter), *args], check=True, capture_output=True, timeout=60)
          operations = json.loads((folder / "gemv.json").read_text())["oplists"]
          convolutions = [op for op in operations if op["type"] == "Convolution"]
          self.assertEqual(len(convolutions), 1)
          self.assertTrue(convolutions[0]["name"].startswith("fold3_host_conv_bl_"))
          self.assertTrue(convolutions[0]["main"]["common"]["relu"])

  def test_gather_nd_negative_constants_wrap_without_changing_shared_indices(self):
    import onnxruntime as ort
    from prepare_accel_models import normalize_constant_gather_nd
    for batch, shape, indices in ((0, [5, 2], [[-2], [-1]]), (1, [2, 5, 2], [[[-2], [-1]], [[0], [-3]]])):
      with self.subTest(batch=batch):
        model = h.make_model(h.make_graph([
          h.make_node("GatherND", ["x", "indices"], ["y"], batch_dims=batch),
          h.make_node("Identity", ["indices"], ["original_indices"])], "gather",
          [h.make_tensor_value_info("x", T.FLOAT, shape)],
          [h.make_tensor_value_info("y", T.FLOAT, [2, 2] if not batch else [2, 2, 2]),
           h.make_tensor_value_info("original_indices", T.INT64, list(np.shape(indices)))],
          [onnx.numpy_helper.from_array(np.array(indices, np.int64), "indices")]),
          opset_imports=[h.make_opsetid("", 21)])
        model.ir_version = 10
        converted = normalize_constant_gather_nd(model)
        self.assertEqual(model.graph.node[0].input[1], "indices")
        self.assertEqual(converted.graph.node[1].input[0], "indices")
        values = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
        expected = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, values)
        actual = ort.InferenceSession(converted.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, values)
        for a, b in zip(actual, expected):
          np.testing.assert_array_equal(a, b)

  def test_gather_nd_rejects_unproven_negative_index_bounds(self):
    from prepare_accel_models import normalize_constant_gather_nd
    for shape, value in ((["unknown", 2], -1), ([5, 2], -6)):
      model = h.make_model(h.make_graph([h.make_node("GatherND", ["x", "i"], ["y"])], "invalid",
        [h.make_tensor_value_info("x", T.FLOAT, shape)], [h.make_tensor_value_info("y", T.FLOAT, [1, 2])],
        [onnx.numpy_helper.from_array(np.array([[value]], np.int64), "i")]))
      with self.assertRaises(ValueError):
        normalize_constant_gather_nd(model)

  def test_road_gather_nd_lowers_only_contiguous_first_axis_selections(self):
    import onnxruntime as ort
    from prepare_accel_models import lower_contiguous_gather_nd
    for indices, expected_op in (([-2, -1], "Slice"), ([-1, -2], "GatherND"), ([0, 0], "GatherND")):
      with self.subTest(indices=indices):
        model = h.make_model(h.make_graph([
          h.make_node("GatherND", ["x", "indices"], ["y"], batch_dims=0)], "gather",
          [h.make_tensor_value_info("x", T.FLOAT, [5, 2])],
          [h.make_tensor_value_info("y", T.FLOAT, [2, 2])],
          [onnx.numpy_helper.from_array(np.array(indices, np.int64).reshape(-1, 1), "indices")]),
          opset_imports=[h.make_opsetid("", 21)])
        model.ir_version = 10
        converted = lower_contiguous_gather_nd(model)
        self.assertEqual(model.graph.node[0].op_type, "GatherND")
        self.assertEqual(converted.graph.node[0].op_type, expected_op)
        inputs = {"x": np.arange(10, dtype=np.float32).reshape(5, 2)}
        actual = ort.InferenceSession(converted.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, inputs)
        np.testing.assert_array_equal(actual[0], inputs["x"][indices])

  def test_road_reduction_repair_checks_axis_layout(self):
    from prepare_accel_models import repair_ncnn_road_reductions
    with tempfile.TemporaryDirectory() as temporary:
      graph = Path(temporary) / "road.ncnn.param"
      typed = Path(temporary) / "road.pnnx.param"
      typed.write_text("7767517\n5 10\n"
                       "ReduceMax first 1 1 a b axes=1 keepdims=0 noop_with_empty_axes=0 #a=(25,4,1,8)f32 #b=(25,1,8)f32\n"
                       "ReduceMax last 1 1 c d axes=-1 keepdims=1 noop_with_empty_axes=0 #c=(1,8,9,9)f32 #d=(1,8,9,1)f32\n"
                       "torch.norm torch.norm_260 1 1 e f dim=-1 keepdim=True p=2 #e=(1,512)f32 #f=(1,1)f32\n"
                       "torch.norm torch.norm_261 1 1 g h dim=-1 keepdim=True p=2 #g=(1,512)f32 #h=(1,1)f32\n"
                       "torch.sum torch.sum_2 1 1 i j dim=-1 keepdim=True #i=(1,8,9,9)f32 #j=(1,8,9,1)f32\n")
      original = ("7767517\n6 11\nReshape view 1 1 input a 0=8 1=1 11=4 12=1 13=233 2=25\n"
                  "ReduceMax first 1 1 a b\nReduceMax last 1 1 c d\n"
                  "Reduction norm_252 1 1 e f 0=8 1=0 -23303=0 4=1 5=1\n"
                  "Reduction norm_253 1 1 g h 0=8 1=0 -23303=0 4=1 5=1\n"
                  "Reduction sum_255 1 1 i j 0=0 1=0 -23303=0 4=1 5=1\n")
      graph.write_text(original)
      self.assertEqual(repair_ncnn_road_reductions(graph, typed), ["first", "last", "norm_252", "norm_253", "sum_255"])
      self.assertIn("Reduction first 1 1 a b 0=4 1=0 -23303=1,1 4=0 5=1", graph.read_text())
      self.assertIn("Reduction last 1 1 c d 0=4 1=0 -23303=1,-1 4=1 5=1", graph.read_text())
      self.assertIn("Reduction sum_255 1 1 i j 0=0 1=0 -23303=1,-1 4=1 5=1", graph.read_text())
      changed = original.replace("11=4", "11=1")
      graph.write_text(changed)
      with self.assertRaisesRegex(ValueError, "layout changed"):
        repair_ncnn_road_reductions(graph, typed)
      self.assertEqual(graph.read_text(), changed)
      graph.write_text(original)
      typed.write_text(typed.read_text().replace("torch.sum_2 1 1 i j dim=-1", "torch.sum_2 1 1 i j dim=1"))
      with self.assertRaisesRegex(ValueError, "scalar reduction contract changed"):
        repair_ncnn_road_reductions(graph, typed)
      self.assertEqual(graph.read_text(), original)

  def test_half_gemv_marker_requires_the_proven_host_reduction_case(self):
    from prepare_accel_models import retain_half_boundaries
    for k, n, trans_b, expected in ((32, 3, 1, True), (8, 2, 1, True), (32, 1, 1, True),
                                  (31, 3, 1, True), (2, 512, 1, True), (32, 3, 0, False)):
      with self.subTest(k=k, n=n, trans_b=trans_b):
        weight_shape = (n, k) if trans_b else (k, n)
        model = h.make_model(h.make_graph([
          h.make_node("Gemm", ["x", "w", "b"], ["y"], name="original_gemm", transB=trans_b)], "gemv",
          [h.make_tensor_value_info("x", T.FLOAT16, [1, k])], [h.make_tensor_value_info("y", T.FLOAT16, [1, n])],
          [onnx.numpy_helper.from_array(np.zeros(weight_shape, np.float16), "w"),
           onnx.numpy_helper.from_array(np.zeros(n, np.float16), "b")]), opset_imports=[h.make_opsetid("", 21)])
        converted = retain_half_boundaries(model)
        self.assertEqual(converted.graph.node[0].output[0].startswith("fold3_half_gemv8_"), expected)
        self.assertEqual(converted.graph.node[0].op_type, "Gemm")
        self.assertEqual(converted.graph.node[-1].output[0], "y")

  def test_half_dot2_marker_preserves_no_transpose_gemm_semantics(self):
    from prepare_accel_models import retain_half_boundaries
    for rows, width, columns, trans_b, alpha, dynamic_bias, dtype, vector_bias, tagged in (
        (9, 512, 1536, 0, 1., False, T.FLOAT16, True, True),
        (3, 5, 7, 0, 1., False, T.FLOAT16, True, True),
        (1, 2, 3, 0, 1., False, T.FLOAT16, True, True),
        (9, 5, 7, 1, 1., False, T.FLOAT16, True, False),
        (3, 5, 7, 0, .5, False, T.FLOAT16, True, False),
        (3, 5, 7, 0, 1., True, T.FLOAT16, True, False),
        (3, 5, 7, 0, 1., False, T.FLOAT, True, False),
        (3, 5, 7, 0, 1., False, T.FLOAT16, False, False)):
      with self.subTest(rows=rows, width=width, trans_b=trans_b, alpha=alpha,
                        dynamic_bias=dynamic_bias, dtype=dtype, vector_bias=vector_bias):
        np_type = np.float16 if dtype == T.FLOAT16 else np.float32
        bias_shape = [columns] if vector_bias else [1, columns]
        weights = onnx.numpy_helper.from_array(np.zeros((columns, width) if trans_b else (width, columns), np_type), "w")
        bias = onnx.numpy_helper.from_array(np.zeros(bias_shape, np_type), "b")
        inputs = [h.make_tensor_value_info("x", dtype, [rows, width])]
        if dynamic_bias:
          inputs.append(h.make_tensor_value_info("b", dtype, bias_shape))
        original = h.make_model(h.make_graph([
          h.make_node("Gemm", ["x", "w", "b"], ["y"], name="original_gemm", transB=trans_b, alpha=alpha)],
          "dot2", inputs, [h.make_tensor_value_info("y", dtype, [rows, columns])],
          [weights] if dynamic_bias else [weights, bias]), opset_imports=[h.make_opsetid("", 21)])
        converted = retain_half_boundaries(original)
        self.assertEqual(converted.graph.node[0].output[0].startswith("fold3_half_dot2_"), tagged)
        self.assertEqual(converted.graph.input, original.graph.input)
        self.assertEqual(converted.graph.output, original.graph.output)
        self.assertEqual(converted.graph.initializer, original.graph.initializer)
        self.assertEqual(converted.graph.node[0].attribute, original.graph.node[0].attribute)
        converter = ROOT / "android/build/native/accelerators/mnn-patched/host/MNNConvert"
        if tagged and rows == 3 and converter.is_file():
          with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            onnx.save(converted, folder / "dot2.onnx")
            for args in (["-f", "ONNX", "--modelFile", str(folder / "dot2.onnx"),
                          "--MNNModel", str(folder / "dot2.mnn"), "--bizCode", "fold3"],
                         ["-f", "MNN", "--modelFile", str(folder / "dot2.mnn"), "--JsonFile", str(folder / "dot2.json")]):
              subprocess.run([str(converter), *args], check=True, capture_output=True, timeout=60)
            operations = json.loads((folder / "dot2.json").read_text())["oplists"]
            convolutions = [op for op in operations if op["type"] == "Convolution"]
            self.assertEqual(len(convolutions), 1)
            self.assertTrue(convolutions[0]["name"].startswith("fold3_half_dot2_"))

  def test_half_batched_matmul_marker_preserves_broadcast_and_rounding(self):
    import onnxruntime as ort
    from prepare_accel_models import retain_half_boundaries
    for left, right, dtype, tagged in (
        ([1, 8, 9, 9], [1, 8, 9, 64], T.FLOAT16, True),
        ([2, 3, 5], [5, 7], T.FLOAT16, True),
        ([3, 5], [2, 5, 7], T.FLOAT16, True),
        ([1, 3, 5], [2, 5, 7], T.FLOAT16, True),
        ([2, 3, 5], [2, 5, 7], T.FLOAT, False),
        ([3, 5], [5, 7], T.FLOAT16, False)):
      with self.subTest(left=left, right=right, dtype=dtype):
        shape = list(np.broadcast_shapes(left[:-2], right[:-2])) + [left[-2], right[-1]]
        original = h.make_model(h.make_graph([
          h.make_node("MatMul", ["a", "b"], ["y"], name="original_attention")], "attention",
          [h.make_tensor_value_info("a", dtype, left), h.make_tensor_value_info("b", dtype, right)],
          [h.make_tensor_value_info("y", dtype, shape)]), opset_imports=[h.make_opsetid("", 21)])
        original.ir_version = 10
        converted = retain_half_boundaries(original)
        self.assertEqual(converted.graph.node[0].output[0].startswith("fold3_half_matmul2_"), tagged)
        self.assertEqual(converted.graph.input, original.graph.input)
        self.assertEqual(converted.graph.output, original.graph.output)
        self.assertEqual(converted.graph.node[0].attribute, original.graph.node[0].attribute)
        rng = np.random.default_rng(5)
        np_type = np.float16 if dtype == T.FLOAT16 else np.float32
        inputs = {"a": rng.normal(size=left).astype(np_type), "b": rng.normal(size=right).astype(np_type)}
        expected = ort.InferenceSession(original.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
        actual = ort.InferenceSession(converted.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
        np.testing.assert_array_equal(actual, expected)
        converter = ROOT / "android/build/native/accelerators/mnn-patched/host/MNNConvert"
        if tagged and left == [1, 3, 5] and converter.is_file():
          with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            onnx.save(converted, folder / "attention.onnx")
            for args in (["-f", "ONNX", "--modelFile", str(folder / "attention.onnx"),
                          "--MNNModel", str(folder / "attention.mnn"), "--bizCode", "fold3"],
                         ["-f", "MNN", "--modelFile", str(folder / "attention.mnn"), "--JsonFile", str(folder / "attention.json")]):
              subprocess.run([str(converter), *args], check=True, capture_output=True, timeout=60)
            operations = json.loads((folder / "attention.json").read_text())["oplists"]
            attention = [op for op in operations if op["type"] == "MatMul"]
            self.assertEqual(len(attention), 1)
            self.assertTrue(attention[0]["name"].startswith("fold3_half_matmul2_"))

  def test_half_layernorm_marker_requires_original_last_axis_and_fp32_stash(self):
    from prepare_accel_models import retain_half_boundaries
    for axis, stash, dtype, dynamic_bias, tagged in (
        (-1, 1, T.FLOAT16, False, True), (2, 1, T.FLOAT16, False, True),
        (1, 1, T.FLOAT16, False, False), (-1, 10, T.FLOAT16, False, False),
        (-1, 1, T.FLOAT, False, False), (-1, 1, T.FLOAT16, True, False)):
      with self.subTest(axis=axis, stash=stash, dtype=dtype, dynamic_bias=dynamic_bias):
        np_type = np.float16 if dtype == T.FLOAT16 else np.float32
        shape = [1, 9, 16]
        inputs = [h.make_tensor_value_info("x", dtype, shape)]
        constants = [onnx.numpy_helper.from_array(np.ones(16, np_type), "gamma")]
        if dynamic_bias:
          inputs.append(h.make_tensor_value_info("beta", dtype, [16]))
        else:
          constants.append(onnx.numpy_helper.from_array(np.zeros(16, np_type), "beta"))
        original = h.make_model(h.make_graph([
          h.make_node("LayerNormalization", ["x", "gamma", "beta"], ["y"],
                      name="original_layernorm", axis=axis, stash_type=stash, epsilon=1e-5)],
          "layernorm", inputs, [h.make_tensor_value_info("y", dtype, shape)], constants),
          opset_imports=[h.make_opsetid("", 21)])
        converted = retain_half_boundaries(original)
        self.assertEqual(converted.graph.node[0].output[0].startswith("fold3_half_layernorm_"), tagged)
        self.assertEqual(converted.graph.input, original.graph.input)
        self.assertEqual(converted.graph.output, original.graph.output)
        self.assertEqual(converted.graph.initializer, original.graph.initializer)
        self.assertEqual(converted.graph.node[0].attribute, original.graph.node[0].attribute)
        if tagged:
          self.assertEqual(converted.graph.node[-1].op_type, "Cast")
          self.assertEqual(converted.graph.node[-1].output[0], "y")

  def test_explicit_half_boundaries_keep_original_arithmetic_and_outputs(self):
    import onnxruntime as ort
    from prepare_accel_models import retain_half_boundaries
    model = h.make_model(h.make_graph([
      h.make_node("Cast", ["x"], ["half"], to=T.FLOAT16),
      h.make_node("Add", ["half", "half"], ["sum"], name="original_add"),
      h.make_node("Cast", ["sum"], ["y"], to=T.FLOAT)], "half",
      [h.make_tensor_value_info("x", T.FLOAT, [5])], [h.make_tensor_value_info("y", T.FLOAT, [5])]),
      opset_imports=[h.make_opsetid("", 21)])
    model.ir_version = 10
    converted = retain_half_boundaries(model)
    self.assertEqual([node.op_type for node in converted.graph.node], ["Cast", "Add", "Cast", "Cast"])
    self.assertEqual(converted.graph.node[1].name, "original_add")
    self.assertEqual(converted.graph.node[2].attribute[0].i, T.FLOAT16)
    values = {"x": np.array([.3333, .00001, 42.125, -2.0009, 65504.], dtype=np.float32)}
    expected = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, values)
    actual = ort.InferenceSession(converted.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, values)
    np.testing.assert_array_equal(actual[0], expected[0])

  def test_calibration_is_separate_and_qdq_modes_are_explicit(self):
    root = ROOT / "android/app/build/model-reference/htp-probes"
    manifest = json.loads((root / "manifest.json").read_text())
    self.assertFalse(set(manifest["calibration_seeds"]) & {7, 42, 123, manifest["validation_seed"]})
    self.assertEqual((manifest["rtol"], manifest["atol"]), (1e-3, 1e-4))
    for mode, weight in (("a16w16", T.UINT16), ("a16w8", T.UINT8)):
      model = onnx.load(root / f"{mode}-conv.onnx")
      nodes = {node.output[0]: node for node in model.graph.node}
      initializers = {tensor.name: tensor for tensor in model.graph.initializer}
      conv = next(node for node in model.graph.node if node.op_type == "Conv")
      activation_dq, weight_dq = nodes[conv.input[0]], nodes[conv.input[1]]
      self.assertEqual(initializers[activation_dq.input[2]].data_type, T.UINT16)
      self.assertEqual(initializers[weight_dq.input[2]].data_type, weight)
    reader = Reader([1, 3, 8, 8])
    self.assertEqual(reader.get_next()["x"].dtype, np.float32)

  def test_portable_export_preserves_fp16_operator_boundary(self):
    model = h.make_model(h.make_graph([h.make_node("FusedGemm", ["x", "w"], ["y"], domain="com.microsoft", transB=1)],
      "half", [h.make_tensor_value_info("x", T.FLOAT16, [1, 8]), h.make_tensor_value_info("w", T.FLOAT16, [4, 8])],
      [h.make_tensor_value_info("y", T.FLOAT16, [1, 4])]), opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.microsoft", 1)])
    converted = portable(model, promote_half_gemm=False)
    self.assertEqual([node.op_type for node in converted.graph.node], ["Gemm"])
    self.assertEqual(converted.graph.output[0].type.tensor_type.elem_type, T.FLOAT16)
    for kind in ("conv", "gemm", "activation", "layout"):
      onnx.checker.check_model(graph(kind)[0])

  def test_adapter_preserves_external_state_and_rejects_fractional_image_feedback(self):
    native = Mock()
    native.describe.return_value = {"inputs": {"x": {"shape": [1, 2], "dtype": "float32"}}}
    metadata = {"inputs": {"x": {"shape": [1, 2], "type": "tensor(uint8)"}},
                "outputs": {"next_state_img_q": {"shape": [1, 2], "type": "tensor(uint8)"}},
                "ncnn": {"input_batch_axes": {"x": None}, "output_batch_axes": {"next_state_img_q": None}}}
    values = np.array([[0, 255]], np.uint8)
    with patch.dict(sys.modules, {"fold3_accel_native": native}):
      for backend, key in (("mnn-opencl", "next_state_img_q"), ("ncnn-vulkan", "out0")):
        model = AccelModel("test", backend, metadata)
        native.run.return_value = {key: ([1, 2], "float32", values.astype(np.float32).tobytes())}
        np.testing.assert_array_equal(model({"x": values})["next_state_img_q"], values)
        passed = native.run.call_args.args[1]["x" if backend == "mnn-opencl" else "in0"]
        if backend == "ncnn-vulkan":
          passed, batched = passed
          self.assertFalse(batched)
        np.testing.assert_array_equal(passed, values)
        self.assertEqual(passed.dtype, np.float32)
        native.run.return_value = {key: ([1, 2], "float32", np.array([[.5, 255]], np.float32).tobytes())}
        with self.assertRaisesRegex(ValueError, "Non-integral"):
          model({"x": values})
        with self.assertRaisesRegex(ValueError, "input type changed"):
          model({"x": values.astype(np.float32)})

  def test_ncnn_converter_batch_contract_and_nonzero_axis(self):
    from prepare_accel_models import ncnn_contract
    from tempfile import TemporaryDirectory
    metadata = {"inputs": {"x": {"shape": [3, 2, 4], "type": "tensor(float)"}},
                "outputs": {"y": {"shape": [3, 2, 4], "type": "tensor(float)"}}}
    with TemporaryDirectory() as folder:
      script = Path(folder) / "generated.py"
      script.write_text('ex.input("in0", ncnn.Mat(in0.numpy(), batch_index=1).clone())\nout0.numpy(batch_index=1)\n')
      metadata["ncnn"] = ncnn_contract(script, metadata["inputs"], metadata["outputs"])
      script.write_text('ex.input("in0", ncnn.Mat(in0.numpy(), batch_index=233).clone())\nout0.numpy(batch_index=233)\n')
      self.assertIsNone(ncnn_contract(script, metadata["inputs"], metadata["outputs"])["input_batch_axes"]["x"])
      script.write_text('raise RuntimeError("never execute converter source")')
      with self.assertRaisesRegex(ValueError, "Missing/ambiguous"):
        ncnn_contract(script, metadata["inputs"], metadata["outputs"])
    values = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    native = Mock()
    native.run.return_value = {"out0": (None, "float32", np.moveaxis(values, 1, 0).copy().tobytes())}
    with patch.dict(sys.modules, {"fold3_accel_native": native}):
      model = AccelModel("test", "ncnn-vulkan", metadata)
      np.testing.assert_array_equal(model({"x": values})["y"], values)
      buffer, batched = native.run.call_args.args[1]["in0"]
      self.assertTrue(batched)
      np.testing.assert_array_equal(buffer, np.moveaxis(values, 1, 0))

  def test_ncnn_road_singleton_edits_preserve_native_width_and_axis(self):
    from prepare_accel_models import repair_ncnn_road_rank_changes
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as folder:
      graph = Path(folder) / "road.ncnn.param"
      typed = Path(folder) / "road.pnnx.param"
      typed.write_text(
        "7767517\n4 8\n"
        "torch.unsqueeze a 1 1 a0 a1 dim=1 $input=a0 #a0=(1,512)f16 #a1=(1,1,512)f16\n"
        "torch.unsqueeze b 1 1 b0 b1 dim=1 $input=b0 #b0=(1,512)f16 #b1=(1,1,512)f16\n"
        "Tensor.select c 1 1 c0 c1 dim=1 index=8 #c0=(1,9,512)f16 #c1=(1,512)f16\n"
        "torch.unsqueeze d 1 1 d0 d1 dim=0 $input=d0 #d0=(1,512)f32 #d1=(1,1,512)f32\n"
      )
      original = "7767517\n4 8\n" + "".join(
        f"{op} {name} 1 1 {name}0 {name}1 -23303=1,0\n"
        for op, name in zip(("ExpandDims", "ExpandDims", "Squeeze", "ExpandDims"), "abcd", strict=True)
      )
      graph.write_text(original)
      repair_ncnn_road_rank_changes(graph, typed)
      self.assertEqual(graph.read_text().splitlines()[2:], [
        "Reshape a 1 1 a0 a1 0=512 1=1",
        "Reshape b 1 1 b0 b1 0=512 1=1",
        "Reshape c 1 1 c0 c1 0=512",
        "Reshape d 1 1 d0 d1 0=512 1=1",
      ])
      invalid = original.replace("-23303=1,0", "-23303=1,1", 1)
      graph.write_text(invalid)
      with self.assertRaisesRegex(ValueError, "singleton axis changed"):
        repair_ncnn_road_rank_changes(graph, typed)
      self.assertEqual(graph.read_text(), invalid)

  def test_ncnn_packed_road_image_history_preserves_external_order(self):
    external = {"shape": [2, 5, 6, 128, 256], "type": "tensor(uint8)"}
    packed = {"shape": [2, 30, 128, 256], "type": "tensor(uint8)"}
    metadata = {
      "inputs": {"state_img_q": external},
      "outputs": {"next_state_img_q": external},
      "ncnn": {
        "engine_inputs": {"state_img_q": packed},
        "engine_outputs": {"next_state_img_q": packed},
        "input_batch_axes": {"state_img_q": None},
        "output_batch_axes": {"next_state_img_q": None},
      },
    }
    values = np.random.default_rng(42).integers(0, 256, size=external["shape"], dtype=np.uint8)
    native = Mock()
    native.run.return_value = {"out0": (None, "float32", values.astype(np.float32).tobytes())}
    with patch.dict(sys.modules, {"fold3_accel_native": native}):
      model = AccelModel("test", "ncnn-vulkan", metadata)
      result = model({"state_img_q": values})
      np.testing.assert_array_equal(result["next_state_img_q"], values)
      buffer, batched = native.run.call_args.args[1]["in0"]
      self.assertFalse(batched)
      np.testing.assert_array_equal(buffer, values.reshape(packed["shape"]))
      metadata["ncnn"]["engine_outputs"]["next_state_img_q"] = dict(packed, shape=[2, 6, 640, 256])
      with self.assertRaisesRegex(ValueError, "Unsupported ncnn tensor adapter"):
        model({"state_img_q": values})

  def test_mnn_dropped_inputs_require_original_reachability_proof(self):
    from prepare_accel_models import unused_inputs
    model = h.make_model(h.make_graph([h.make_node("Identity", ["x"], ["y"])], "unused",
      [h.make_tensor_value_info(key, T.FLOAT, [1]) for key in ("x", "unused")],
      [h.make_tensor_value_info("y", T.FLOAT, [1])]))
    self.assertEqual(unused_inputs(model), ["unused"])
    metadata = {"inputs": {key: {"shape": [1], "type": "tensor(float)"} for key in ("x", "unused")},
                "outputs": {"y": {"shape": [1], "type": "tensor(float)"}}, "unused_inputs": unused_inputs(model)}
    native = Mock()
    native.describe.return_value = {"inputs": {"x": {"shape": [1], "dtype": "float32"}}}
    native.run.return_value = {"y": ([1], "float32", np.ones(1, np.float32).tobytes())}
    with patch.dict(sys.modules, {"fold3_accel_native": native}):
      AccelModel("test", "mnn-opencl", metadata)({"x": np.ones(1, np.float32), "unused": np.zeros(1, np.float32)})
      self.assertEqual(set(native.run.call_args.args[1]), {"x"})
      native.describe.return_value = {"inputs": {}}
      with self.assertRaisesRegex(ValueError, "reachable"):
        AccelModel("test", "mnn-opencl", metadata)

  def test_float_image_transport_preserves_half_and_rejects_byte_arithmetic(self):
    from prepare_accel_models import float_image_transport
    import onnxruntime as ort
    model = h.make_model(h.make_graph([
      h.make_node("Identity", ["x"], ["history"]),
      h.make_node("Cast", ["history"], ["half"], to=T.FLOAT16)], "images",
      [h.make_tensor_value_info("x", T.UINT8, [256])],
      [h.make_tensor_value_info("history", T.UINT8, [256]), h.make_tensor_value_info("half", T.FLOAT16, [256])]),
      opset_imports=[h.make_opsetid("", 21)])
    model.ir_version = 10
    converted = float_image_transport(model)
    self.assertEqual(converted.graph.output[1].type.tensor_type.elem_type, T.FLOAT16)
    self.assertEqual(converted.graph.node[1].attribute[0].i, T.FLOAT16)
    values = np.arange(256, dtype=np.uint8)
    outputs = ort.InferenceSession(converted.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": values.astype(np.float32)})
    np.testing.assert_array_equal(outputs[0], values)
    np.testing.assert_array_equal(outputs[1], values.astype(np.float16))
    converted.graph.node[0].op_type = "Add"
    converted.graph.node[0].input.append("x")
    # Test against the byte graph: arithmetic must never be silently promoted.
    model.graph.node[0].op_type = "Add"
    model.graph.node[0].input.append("x")
    with self.assertRaisesRegex(ValueError, "Non-transport"):
      float_image_transport(model)

  def test_qualification_invalidates_changed_artifacts_and_cpu_fallback(self):
    identity = {"backend": "mnn-opencl", "memory_mode": "buffer", "cpu_fallback": False, "fingerprint": "Fold3",
                "sha256": {"apk": "1" * 64, "model": "2" * 64, "provider": "3" * 64, "implementation": "4" * 64,
                           "opencl_plugin": "5" * 64}}
    entry = {"identity": identity, "diagnostic_only": False,
             "assignment": {"no_cpu_fallback_verified": True, "memory_mode": "buffer", "trace_limit": 0},
             "status": "passed", "performance_status": "passed", "samples": 100, "work_p99_ms": 49,
             "deadline_misses_50ms": 0, "parity": {"out": {"passed": True}},
             "additional_seed_parity": {str(seed): {"out": {"passed": True}} for seed in (7, 123)},
             "sequences": {key: {"passed": True, "frames": count, "checks": [
               {"frame": frame, "outputs": {"out": {"passed": True}}} for frame in range(count)]}
               for key, count in (("seed42", 32), ("seed7", 32), ("seed123", 32), ("camera-replay", 60))}}
    self.assertTrue(qualified(entry, identity))
    for key in ("apk", "model", "provider", "implementation", "opencl_plugin"):
      current = copy.deepcopy(identity)
      current["sha256"][key] = "f" * 64
      self.assertFalse(qualified(entry, current))
    self.assertFalse(qualified(None, identity))
    self.assertFalse(qualified(entry, {**identity, "memory_mode": "image"}))
    self.assertFalse(qualified({**entry, "assignment": {**entry["assignment"], "trace_limit": 1}}, identity))
    for mutation in ({"assignment": {**entry["assignment"], "diagnostic_dispatch_limit": 131072}},
                     {"diagnostic_only": True}, {"assignment": {"no_cpu_fallback_verified": False}},
                     {"samples": 99}, {"deadline_misses_50ms": 1}):
      self.assertFalse(qualified({**entry, **mutation}, identity))
    with self.assertRaisesRegex(RuntimeError, "No accelerator"):
      with patch.dict(sys.modules, {"fold3_accel_native": Mock(create=Mock(side_effect=RuntimeError("No accelerator")))}):
        AccelModel("model", "ncnn-vulkan", {})
