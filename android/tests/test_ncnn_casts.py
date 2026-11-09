"""A missing/ambiguous converter dtype must never silently erase rounding."""
from pathlib import Path
import sys
import struct
import numpy as np
import onnx
from onnx import helper, numpy_helper
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_accel_models import repair_ncnn_casts, mark_ncnn_half_gemv, mark_ncnn_half_heads, restore_ncnn_half_outputs, mark_ncnn_reference_layernorm, mark_ncnn_float_gemm
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app/src/main/python"))
from fold3_accel import AccelModel


class NcnnCastTest(unittest.TestCase):
  def test_float_gemm_marker_is_typed_provenance_bound_and_atomic(self):
    with tempfile.TemporaryDirectory() as directory:
      folder = Path(directory)
      graph, original = folder / "model.param", folder / "original.onnx"
      weight = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
      bias = np.arange(4, dtype=np.float32)
      model = helper.make_model(helper.make_graph([
        helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="plain_linear", transB=1)], "gemm",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(weight, "w"), numpy_helper.from_array(bias, "b")]))
      onnx.save(model, original)
      baseline = "7767517\n2 2\nInput input 0 1 x\nInnerProduct linear 1 1 x y 0=4 1=1 2=16\n"
      graph.write_text(baseline)
      graph.with_suffix(".bin").write_bytes(struct.pack("<I", 0) + weight.tobytes() + bias.tobytes())
      record = mark_ncnn_float_gemm(graph, original, ["plain_linear"])
      self.assertEqual(record["requires_reference_float_gemm"], 1)
      self.assertIn("9=97", graph.read_text())
      selected = graph.read_bytes()
      mark_ncnn_float_gemm(graph, original, ["plain_linear"])
      self.assertEqual(graph.read_bytes(), selected)
      for requested in (["missing"], ["plain_linear", "plain_linear"]):
        with self.assertRaises(ValueError):
          mark_ncnn_float_gemm(graph, original, requested)
        self.assertEqual(graph.read_bytes(), selected)
      # Transported float weights cannot silently qualify an original half Gemm.
      for value in (*model.graph.input, *model.graph.output):
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
      onnx.save(model, original)
      with self.assertRaisesRegex(ValueError, "FP32 Gemm"):
        mark_ncnn_float_gemm(graph, original, ["plain_linear"])
      self.assertEqual(graph.read_bytes(), selected)
      onnx.save(helper.make_model(helper.make_graph([
        helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="plain_linear", transB=1, alpha=2.)], "changed",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(weight, "w"), numpy_helper.from_array(bias, "b")])), original)
      with self.assertRaises(ValueError):
        mark_ncnn_float_gemm(graph, original, ["plain_linear"])
      self.assertEqual(graph.read_bytes(), selected)

  def test_reference_layernorm_requires_unique_original_semantics_and_is_idempotent(self):
    with tempfile.TemporaryDirectory() as directory:
      folder = Path(directory)
      graph, original = folder / "model.param", folder / "original.onnx"
      scale, bias = np.array([1., 2.], np.float16), np.array([.5, -.5], np.float16)
      model = helper.make_model(helper.make_graph([
        helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["y"],
                         name="original_norm", axis=-1, stash_type=1, epsilon=1e-5)], "norm",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT16, [1, 3, 2])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [1, 3, 2])],
        [numpy_helper.from_array(scale, "scale"), numpy_helper.from_array(bias, "bias")]))
      onnx.save(model, original)
      graph.with_suffix(".bin").write_bytes(scale.astype("<f4").tobytes() + bias.astype("<f4").tobytes())
      baseline = "7767517\n2 2\nInput input 0 1 x\nLayerNorm norm 1 1 x y 0=2 1=1.000000e-5 2=1\n"
      graph.write_text(baseline)
      record = mark_ncnn_reference_layernorm(graph, original, ["original_norm"])
      self.assertEqual(record["requires_reference_layernorm"], 1)
      self.assertIn("29=1", graph.read_text())
      self.assertNotIn("30=1", graph.read_text())
      selected = graph.read_bytes()
      mark_ncnn_reference_layernorm(graph, original, ["original_norm"])
      self.assertEqual(graph.read_bytes(), selected)
      for requested in (["absent"], ["original_norm", "original_norm"]):
        with self.assertRaises(ValueError):
          mark_ncnn_reference_layernorm(graph, original, requested)
        self.assertEqual(graph.read_bytes(), selected)
      # Boundary restoration must remain idempotent with the native selector.
      restore_ncnn_half_outputs(graph, original)
      repaired = graph.read_bytes()
      self.assertIn(b"Cast fold3_round_norm", repaired)
      restore_ncnn_half_outputs(graph, original)
      self.assertEqual(graph.read_bytes(), repaired)
      for key, value in (("axis", 1), ("stash_type", 10), ("epsilon", .001)):
        changed = onnx.ModelProto()
        changed.CopyFrom(model)
        for attribute in changed.graph.node[0].attribute:
          if attribute.name == key:
            attribute.CopyFrom(helper.make_attribute(key, value))
        onnx.save(changed, original)
        with self.assertRaises(ValueError):
          mark_ncnn_reference_layernorm(graph, original, ["original_norm"])
        self.assertEqual(graph.read_bytes(), repaired)

  def test_reference_layernorm_rejects_missing_native_capability(self):
    native = Mock()
    metadata = {"ncnn": {"requires_reference_layernorm": 1}}
    with patch.dict(sys.modules, {"fold3_accel_native": native}):
      for capability in (None, 0, 2):
        native.describe.return_value = {"reference_layernorm": capability}
        with self.assertRaisesRegex(ValueError, "reference LayerNorm"):
          AccelModel("diagnostic.param", "ncnn-vulkan", metadata)
      native.describe.return_value = {"reference_layernorm": 1}
      self.assertEqual(AccelModel("diagnostic.param", "ncnn-vulkan", metadata).info["reference_layernorm"], 1)

  def test_float_gemm_rejects_missing_native_capability(self):
    native = Mock()
    metadata = {"ncnn": {"requires_reference_float_gemm": 1}}
    with patch.dict(sys.modules, {"fold3_accel_native": native}):
      for capability in (None, 0, 2):
        native.describe.return_value = {"reference_float_gemm": capability}
        with self.assertRaisesRegex(ValueError, "FP32 Gemm"):
          AccelModel("diagnostic.param", "ncnn-vulkan", metadata)
      native.describe.return_value = {"reference_float_gemm": 1}
      self.assertEqual(AccelModel("diagnostic.param", "ncnn-vulkan", metadata).info["reference_float_gemm"], 1)

  def repair(self, typed, converted):
    with tempfile.TemporaryDirectory() as temporary:
      folder = Path(temporary)
      source, graph = folder / "typed.param", folder / "graph.param"
      source.write_text("7767517\n4 4\n" + typed)
      graph.write_text("7767517\n4 4\n" + converted)
      before = graph.read_bytes()
      try:
        result = repair_ncnn_casts(graph, source)
      except ValueError:
        self.assertEqual(before, graph.read_bytes())
        raise
      return result, graph.read_text()

  def test_half_boundaries_and_exact_external_byte_transport(self):
    typed = "\n".join(("pnnx.Input in0 0 1 0 #0=(1,4)u8",
                       "Cast a 1 1 0 1 #0=(1,4)u8 #1=(1,4)f32",
                       "Cast b 1 1 1 2 #1=(1,4)f32 #2=(1,4)f16",
                       "Cast c 1 1 2 3 #2=(1,4)f16 #3=(1,4)f32")) + "\n"
    result, graph = self.repair(typed, "Cast a 1 1 in0 1\nCast b 1 1 1 2\nCast c 1 1 2 out0\n")
    self.assertEqual([(item["type_from"], item["type_to"]) for item in result], [(1, 1), (1, 2), (2, 1)])
    self.assertIn("Cast b 1 1 1 2 0=1 1=2", graph)

  def test_reject_missing_or_conflicting_types_without_edit(self):
    for typed, graph in (
        ("Cast a 1 1 x y\n", "Cast a 1 1 x y\n"),
        ("Cast a 1 1 x y #x=(1)f32 #y=(1)f16\n", "Cast a 1 1 x y 0=2 1=1\n"),
        ("Cast a 1 1 x y #x=(1)u8 #y=(1)f32\n", "Cast a 1 1 x y\n"),
        ("Cast a 1 1 x y #x=(1)i64 #y=(1)f32\n", "Cast a 1 1 x y\n"),
        ("Cast a 1 1 x y #x=(1)f32 #y=(1)f16\n", "Cast other 1 1 x y\n")):
      with self.subTest(typed=typed, graph=graph), self.assertRaises(ValueError):
        self.repair(typed, graph)

  def test_half_gemv_marker_rejects_changed_dimensions_or_fused_activation(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "graph.param"
      good = "7767517\n1 2\nInnerProduct linear_100 1 1 a b 0=512 1=1 2=655360\n"
      for invalid in (good.replace("0=512", "0=256"), good.replace("2=655360", "2=512"),
                      good.replace("2=655360", "2=655360 9=1"), good.replace("linear_100", "other")):
        path.write_text(invalid)
        with self.assertRaises(ValueError):
          mark_ncnn_half_gemv(path)
        self.assertEqual(path.read_text(), invalid)
      path.write_text(good)
      self.assertTrue(mark_ncnn_half_gemv(path)["diagnostic_only"])
      self.assertIn("9=99", path.read_text())

  def test_half_heads_bind_original_weights_and_types_before_edit(self):
    with tempfile.TemporaryDirectory() as directory:
      folder = Path(directory)
      graph = folder / "graph.param"
      original = folder / "original.onnx"
      w = np.array([[.5, -.25], [.125, .75]], dtype=np.float16)
      b = np.array([.25, .125], dtype=np.float16)
      node = helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="half_head", transB=1)
      model = helper.make_model(helper.make_graph([node], "half", [helper.make_tensor_value_info("x", 10, [1, 2])],
        [helper.make_tensor_value_info("y", 10, [1, 2])], [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")]))
      onnx.save(model, original)
      text = "7767517\n1 2\nInnerProduct head 1 1 a b 0=2 1=1 2=4\n"
      payload = struct.pack("<I", 0) + w.astype("<f4").tobytes() + b.astype("<f4").tobytes()
      graph.with_suffix(".bin").write_bytes(payload)
      graph.write_text(text)
      result = mark_ncnn_half_heads(graph, original)
      self.assertEqual(result["operators"][0]["original_node"], "half_head")
      self.assertIn("9=99", graph.read_text())
      # Constants use ncnn's depth key 11, and LayerNorm serializes both
      # affine arrays before a later half head. Do not read their bytes as
      # the next convolution's weight tag.
      prefix = ("MemoryData history 0 1 h 0=2 1=1 11=3 2=1\n"
                "LayerNorm norm 1 1 h n 0=2 2=1\n")
      graph.write_text(text.replace("1 2\n", "3 4\n").replace("InnerProduct", prefix + "InnerProduct"))
      graph.with_suffix(".bin").write_bytes(np.arange(10, dtype="<f4").tobytes() + payload)
      self.assertEqual(mark_ncnn_half_heads(graph, original)["operators"][0]["original_node"], "half_head")
      for changed_graph, changed_weights in ((text.replace("2=4", "2=4 9=1"), payload),
                                             (text, payload[:-4] + struct.pack("<f", .5))):
        graph.write_text(changed_graph)
        graph.with_suffix(".bin").write_bytes(changed_weights)
        with self.assertRaises(ValueError):
          mark_ncnn_half_heads(graph, original)
        self.assertEqual(graph.read_text(), changed_graph)

      # Conversion duplicates this head after a Split. Both copies must keep
      # the original half accumulation, including aliases of aliases.
      duplicate = ("7767517\n5 7\nSplit split 1 2 a a0 a1\n"
                   "Split nested 1 2 a1 a2 a3\n"
                   "InnerProduct first 1 1 a0 b0 0=2 1=1 2=4\n"
                   "InnerProduct second 1 1 a2 b1 0=2 1=1 2=4\n")
      graph.write_text(duplicate)
      graph.with_suffix(".bin").write_bytes(payload * 2)
      result = mark_ncnn_half_heads(graph, original, original_nodes={"half_head"})
      self.assertEqual([v["operator"] for v in result["operators"]], ["first", "second"])
      self.assertEqual([v["input_identity"] for v in result["operators"]], ["a", "a"])
      self.assertEqual(graph.read_text().count("9=99"), 2)
      # Equal weights cannot justify annotating a different input. Validate
      # the complete payload even after all selected heads have been found.
      for invalid, data in ((duplicate.replace("a2 b1", "other b1"), payload * 2),
                            (duplicate, payload * 2 + b"trailing"),
                            (duplicate, (payload * 2)[:-4])):
        graph.write_text(invalid)
        graph.with_suffix(".bin").write_bytes(data)
        with self.assertRaises(ValueError):
          mark_ncnn_half_heads(graph, original, original_nodes={"half_head"})
        self.assertEqual(graph.read_text(), invalid)

  def test_ordered_half_graph_rejects_runtime_without_that_feature(self):
    native = Mock()
    native.describe.return_value = {"explicit_half_rounding": True}
    with patch.dict(sys.modules, {"fold3_accel_native": native}):
      with self.assertRaisesRegex(ValueError, "ordered half GEMV"):
        AccelModel("diagnostic.param", "ncnn-vulkan", {"ncnn": {"requires_ordered_half_gemv": True}})

  def test_implicit_half_outputs_require_original_weights_types_and_affine_parameters(self):
    with tempfile.TemporaryDirectory() as directory:
      graph, original = (Path(directory) / name for name in ("graph.param", "original.onnx"))
      tensor = helper.make_tensor_value_info
      scale, shift = np.array([1., .5], np.float16), np.array([.25, .125], np.float16)
      w, b = np.array([[.5, -.25], [.125, .75]], np.float16), np.array([.5, .25], np.float16)
      model = helper.make_model(helper.make_graph([
        helper.make_node("LayerNormalization", ["x", "scale", "shift"], ["norm"], name="norm", axis=-1, epsilon=1e-5),
        helper.make_node("Gemm", ["norm", "w", "b"], ["y"], name="head", transB=0)], "half",
        [tensor("x", 10, [1, 2])], [tensor("y", 10, [1, 2])],
        [numpy_helper.from_array(v, k) for k, v in (("scale", scale), ("shift", shift), ("w", w), ("b", b))]))
      onnx.save(model, original)
      text = ("7767517\n3 3\nInput input 0 1 x\nLayerNorm norm 1 1 x n 0=2 1=1.000000e-5 2=1\n"
              "InnerProduct head 1 1 n y 0=2 1=1 2=4\n")
      data = b"".join(v.astype("<f4").tobytes() for v in (scale, shift))
      data += struct.pack("<I", 0) + w.T.astype("<f4").tobytes() + b.astype("<f4").tobytes()
      graph.write_text(text)
      graph.with_suffix(".bin").write_bytes(data)
      result = restore_ncnn_half_outputs(graph, original)
      self.assertFalse(result["qualified"])
      self.assertEqual([v["operator"] for v in result["operators"]], ["norm", "head"])
      self.assertEqual(graph.read_text().splitlines()[1], "5 5")
      self.assertIn("Cast fold3_round_norm 1 1 fold3_unrounded_norm n 0=1 1=2", graph.read_text())
      self.assertIn("Cast fold3_round_head 1 1 fold3_unrounded_head y 0=1 1=2", graph.read_text())
      for invalid, payload in ((text.replace("1.000000e-5", "0.001"), data),
                               (text.replace("2=4", "2=4 9=1"), data),
                               (text.replace("input 0 1 x", "fold3_round_head 0 1 x"), data),
                               (text, data + bytes(4))):
        graph.write_text(invalid)
        graph.with_suffix(".bin").write_bytes(payload)
        with self.assertRaises(ValueError):
          restore_ncnn_half_outputs(graph, original)
        self.assertEqual(graph.read_text(), invalid)
      # A matching FP32 head makes dtype attribution ambiguous; never round it.
      model.graph.node.append(helper.make_node("Gemm", ["other", "w", "b"], ["other_out"], name="other", transB=0))
      model.graph.input.append(tensor("other", 1, [1, 2]))
      model.graph.output.append(tensor("other_out", 1, [1, 2]))
      onnx.save(model, original)
      graph.write_text(text)
      graph.with_suffix(".bin").write_bytes(data)
      with self.assertRaisesRegex(ValueError, "Ambiguous original half output"):
        restore_ncnn_half_outputs(graph, original)
      self.assertEqual(graph.read_text(), text)

  def test_half_dot2_targets_only_verified_notrans_half_matrices(self):
    with tempfile.TemporaryDirectory() as directory:
      graph, original = (Path(directory) / name for name in ("graph.param", "original.onnx"))
      tensor = helper.make_tensor_value_info
      w = np.array([[.5, -.25, .125], [.75, .25, -.5]], np.float16)
      b = np.array([.5, .25, -.125], np.float16)
      def model(input_type=10, transposed=False):
        weights = w.T if transposed else w
        return helper.make_model(helper.make_graph([
          helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="head", transB=int(transposed))],
          "half", [tensor("x", input_type, [4, 2])], [tensor("y", 10, [4, 3])],
          [numpy_helper.from_array(weights, "w"), numpy_helper.from_array(b, "b")]))
      text = "7767517\n2 2\nInput input 0 1 x\nInnerProduct head 1 1 x y 0=3 1=1 2=6\n"
      data = struct.pack("<I", 0) + w.T.astype("<f4").tobytes() + b.astype("<f4").tobytes()
      graph.write_text(text)
      graph.with_suffix(".bin").write_bytes(data)
      onnx.save(model(), original)
      result = restore_ncnn_half_outputs(graph, original, mode="dot2")
      self.assertEqual(graph.read_text().splitlines()[1], "2 2")
      self.assertTrue(graph.read_text().endswith("9=98\n"))
      self.assertEqual(result["operators"][0]["original_nodes"], ["head"])
      self.assertFalse(result["qualified"])
      before = graph.read_text()
      self.assertEqual(restore_ncnn_half_outputs(graph, original, mode="dot2")["operators"], [])
      self.assertEqual(graph.read_text(), before)
      for input_type, transposed, invalid, payload, error in (
          (10, True, text, data, None),
          (1, False, text, data, "Ambiguous original half dot2"),
          (10, False, text.replace("2=6", "2=6 9=4"), data, "Fused activation"),
          (10, False, text, data + bytes(4), "Incomplete serialized")):
        onnx.save(model(input_type, transposed), original)
        graph.write_text(invalid)
        graph.with_suffix(".bin").write_bytes(payload)
        if error:
          with self.assertRaisesRegex(ValueError, error):
            restore_ncnn_half_outputs(graph, original, mode="dot2")
        else:
          self.assertEqual(restore_ncnn_half_outputs(graph, original, mode="dot2")["operators"], [])
        self.assertEqual(graph.read_text(), invalid)
      # Matching bytes with conflicting transpose semantics are ambiguous.
      ambiguous = model()
      ambiguous.graph.node.append(model(transposed=True).graph.node[0])
      ambiguous.graph.node[-1].name = "conflicting"
      ambiguous.graph.node[-1].input[1] = "wt"
      ambiguous.graph.node[-1].output[0] = "other_y"
      ambiguous.graph.initializer.append(numpy_helper.from_array(w.T, "wt"))
      ambiguous.graph.output.append(tensor("other_y", 10, [4, 3]))
      onnx.save(ambiguous, original)
      graph.write_text(text)
      graph.with_suffix(".bin").write_bytes(data)
      with self.assertRaisesRegex(ValueError, "Ambiguous original half dot2"):
        restore_ncnn_half_outputs(graph, original, mode="dot2")
      self.assertEqual(graph.read_text(), text)

  def test_dot2_requires_native_capability(self):
    native = Mock()
    with patch.dict(sys.modules, {"fold3_accel_native": native}):
      native.describe.return_value = {"ordered_half_dot2": False}
      with self.assertRaisesRegex(ValueError, "ordered half dot2"):
        AccelModel("diagnostic.param", "ncnn-vulkan", {"ncnn": {"requires_ordered_half_dot2": True}})

  def test_patched_graph_rejects_an_unpatched_runtime(self):
    native = Mock()
    metadata = {"ncnn": {"requires_explicit_half_rounding": True}}
    with patch.dict(sys.modules, {"fold3_accel_native": native}):
      for flag in (None, False, "true"):
        native.describe.return_value = {"explicit_half_rounding": flag}
        with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "patched Vulkan"):
          AccelModel("diagnostic.param", "ncnn-vulkan", metadata)
      native.describe.return_value = {"explicit_half_rounding": True}
      self.assertEqual(AccelModel("diagnostic.param", "ncnn-vulkan", metadata).backend, "ncnn-vulkan")


if __name__ == "__main__":
  unittest.main()
