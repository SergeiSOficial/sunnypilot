"""Weight provenance and captured-input tracing cannot alias recurrent inputs."""
from pathlib import Path
import struct
import sys
import unittest

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_ncnn_trace import weight_mapping, precision_boundary_comparison
from fold3_model_trace import compare
from probe_ncnn import unfuse_activations


class NcnnTraceTest(unittest.TestCase):
  def test_expected_attention_masks_are_allowed_only_for_diagnostics(self):
    expected = np.array([-np.inf, 1.], np.float32)
    self.assertFalse(compare(expected, expected)["passed"])
    matched = compare(expected, expected, allow_masked_infinity=True)
    self.assertTrue(matched["passed"])
    self.assertEqual(matched["unexpected_nonfinite_elements"], 0)
    for actual in ([-np.inf, -np.inf], [np.inf, 1.], [np.nan, 1.], [0., 1.]):
      self.assertFalse(compare(np.array(actual, np.float32), expected, allow_masked_infinity=True)["passed"])

  def test_rounding_diagnostics_do_not_relax_or_replace_original_tolerance(self):
    expected = np.array([1., 2.], np.float16)
    actual = np.array([1.0006, 2.0004], np.float32)
    self.assertTrue(compare(actual, expected)["passed"])
    self.assertEqual(precision_boundary_comparison(actual, expected), {"half_rounding_disagreements": 1})
    self.assertEqual(precision_boundary_comparison(expected.astype(np.float32), expected),
                     {"half_rounding_disagreements": 0})
    self.assertEqual(precision_boundary_comparison(actual, expected.astype(np.float32)), {})

  def test_unfused_diagnostic_preserves_original_output_and_counts(self):
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
      graph = Path(directory) / 'model.param'
      graph.write_text('7767517\n2 3\nInput input 0 1 x\nConvolution conv 1 1 x y 0=4 6=16 9=4\n')
      records = unfuse_activations(graph, ['conv'])
      self.assertEqual(records[0]['activation'], 'Sigmoid')
      self.assertIn('3 4\n', graph.read_text())
      self.assertIn('Convolution conv 1 1 x fold3_preactivation_conv 0=4 6=16\n', graph.read_text())
      self.assertTrue(graph.read_text().endswith('Sigmoid fold3_activation_conv 1 1 fold3_preactivation_conv y\n'))
      before = graph.read_bytes()
      with self.assertRaisesRegex(ValueError, 'Unsupported fused activation'):
        unfuse_activations(graph, ['conv'])
      self.assertEqual(graph.read_bytes(), before)

  def test_unfusing_rejects_collisions_without_rewriting(self):
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
      graph = Path(directory) / "model.param"
      graph.write_text("7767517\n2 3\nInput input 0 1 fold3_preactivation_conv\nConvolution conv 1 1 fold3_preactivation_conv y 0=4 6=16 9=4\n")
      before = graph.read_bytes()
      with self.assertRaisesRegex(ValueError, "name collision"):
        unfuse_activations(graph, ["conv"])
      self.assertEqual(graph.read_bytes(), before)
      self.assertEqual(unfuse_activations(graph, []), [])
      self.assertEqual(graph.read_bytes(), before)

  def fixture(self, independent=False):
    tensor = onnx.helper.make_tensor_value_info
    w = np.array([[1., 2.], [3., 4.]], np.float16)
    b = np.array([.5, -.5], np.float16)
    inputs = [tensor("x", onnx.TensorProto.FLOAT16, [1, 2])]
    if independent:
      inputs.append(tensor("other", onnx.TensorProto.FLOAT16, [1, 2]))
    nodes = [onnx.helper.make_node("Gemm", [name, weight, "b"], [output], transB=1)
             for name, weight, output in (("x", "w", "a"), ("other" if independent else "x", "shared_w", "z"))]
    graph = onnx.helper.make_graph(nodes, "shared-head", inputs,
        [tensor("a", onnx.TensorProto.FLOAT16, [1, 2]), tensor("z", onnx.TensorProto.FLOAT16, [1, 2])],
        [onnx.numpy_helper.from_array(value, name) for value, name in ((w, "w"), (w, "shared_w"), (b, "b"))])
    model = onnx.helper.make_model(graph)
    info = {v.name: v for v in (*graph.input, *graph.output)}
    weights = struct.pack("<I", 0) + w.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    return model, info, weights

  def test_deduplicated_heads_require_identical_upstream_provenance(self):
    graph, info, weights = self.fixture()
    param = ["InnerProduct head 1 1 in out 0=2 1=1 2=4"]
    self.assertEqual(weight_mapping(param, weights, graph, info)["head"], ("InnerProduct", "a"))
    independent, other_info, _ = self.fixture(independent=True)
    with self.assertRaisesRegex(ValueError, "Ambiguous original weight provenance"):
      weight_mapping(param, weights, independent, other_info)

  def test_rank_four_constants_and_layer_norm_keep_weight_offsets(self):
    graph, info, weights = self.fixture()
    param = ["MemoryData constant 0 1 fill 0=2 1=3 11=4 2=5",
             "LayerNorm norm 1 1 x n 0=2 2=1", "InnerProduct head 1 1 n y 0=2 1=1 2=4"]
    prefix = bytes(2 * 3 * 4 * 5 * 4 + 2 * 2 * 4)
    self.assertIn("head", weight_mapping(param, prefix + weights, graph, info, {"head"}))
    with self.assertRaisesRegex(ValueError, "Unmapped serialized weights"):
      weight_mapping(param, prefix + weights + bytes(4), graph, info, {"head"})

  def test_layernorm_mapping_preserves_affine_epsilon_and_input_provenance(self):
    model, info, _ = self.fixture()
    model.graph.node.clear()
    model.graph.node.append(onnx.helper.make_node("LayerNormalization", ["x", "b", "b"], ["a"], epsilon=1e-5))
    scale = np.array([.5, -.5], "<f4")
    payload = scale.tobytes() * 2
    param = ["LayerNorm norm 1 1 in out 0=2 1=1.000000e-5 2=1"]
    self.assertEqual(weight_mapping(param, payload, model, info)["norm"], ("LayerNorm", "a"))
    with self.assertRaisesRegex(ValueError, "LayerNorm provenance"):
      weight_mapping([param[0].replace("1.000000e-5", "0.001")], payload, model, info)
    model.graph.input.append(onnx.helper.make_tensor_value_info("other", onnx.TensorProto.FLOAT16, [1, 2]))
    model.graph.node.append(onnx.helper.make_node("LayerNormalization", ["other", "b", "b"], ["z"], epsilon=1e-5))
    with self.assertRaisesRegex(ValueError, "LayerNorm provenance"):
      weight_mapping(param, payload, model, info)

  def test_gemm_transpose_is_bound_to_the_actual_weight_transport(self):
    graph, info, weights = self.fixture()
    for value in graph.graph.initializer:
      if value.name in ("w", "shared_w"):
        value.CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(value).T.copy(), value.name))
    for node in graph.graph.node:
      node.attribute[0].i = 0  # original transB=0, converted output-major weights
    param = ["InnerProduct head 1 1 in out 0=2 1=1 2=4"]
    self.assertEqual(weight_mapping(param, weights, graph, info)["head"], ("InnerProduct", "a"))
    marked = [param[0] + " 9=98"]
    self.assertEqual(weight_mapping(marked, weights, graph, info)["head"], ("InnerProduct", "a"))
    for node in graph.graph.node:
      node.attribute[0].i = 1
    with self.assertRaisesRegex(ValueError, "Ambiguous original weight provenance"):
      weight_mapping(param, weights, graph, info)

  def test_fused_sigmoid_mapping_cannot_skip_a_precision_boundary(self):
    model, info, weights = self.fixture()
    # Keep one head so its consumer is uniquely proven by weight provenance.
    del model.graph.node[1:]
    model.graph.node.append(onnx.helper.make_node("Sigmoid", ["a"], ["sigmoid"]))
    param = ["InnerProduct fused 1 1 in out 0=2 1=1 2=4 9=4"]
    self.assertEqual(weight_mapping(param, weights, model, info)["fused"], ("InnerProduct", "sigmoid"))
    model.graph.node.insert(1, onnx.helper.make_node("Cast", ["a"], ["rounded"], to=onnx.TensorProto.FLOAT))
    model.graph.node[-1].input[0] = "rounded"
    with self.assertRaisesRegex(ValueError, "Unsupported fused activation"):
      weight_mapping(param, weights, model, info)


if __name__ == "__main__":
  unittest.main()
