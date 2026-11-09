import importlib.util
from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper
import onnxruntime as ort

spec = importlib.util.spec_from_file_location("model_trace", Path(__file__).resolve().parents[1] / "scripts/model_trace.py")
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


class ImageGraphTest(unittest.TestCase):
  def test_history_rank_lowering_preserves_leading_axes_and_byte_values(self):
    constants = [numpy_helper.from_array(np.array(v, np.int64), k) for k, v in
                 (("axis", [1]), ("start", [1]), ("end", [3]))]
    nodes = [h.make_node("Unsqueeze", ["image", "axis"], ["expanded"]),
             h.make_node("Slice", ["state", "start", "end", "axis"], ["previous"]),
             h.make_node("Concat", ["previous", "expanded"], ["next_state"], axis=1)]
    model = h.make_model(h.make_graph(nodes, "history", [h.make_tensor_value_info("image", T.UINT8, [2, 1, 8, 16]),
                         h.make_tensor_value_info("state", T.UINT8, [2, 3, 1, 8, 16])],
                         [h.make_tensor_value_info("next_state", T.UINT8, [2, 3, 1, 8, 16])], constants),
                         opset_imports=[h.make_opsetid("", 18)], ir_version=10)
    source = model.SerializeToString()
    lifted = trace.float_image_ops(onnx.load_from_string(source))
    onnx.checker.check_model(lifted)
    inputs = {"image": np.arange(256, dtype=np.uint8).reshape(2, 1, 8, 16),
              "state": np.arange(768).astype(np.uint8).reshape(2, 3, 1, 8, 16)}
    before = ort.InferenceSession(source).run(None, inputs)[0]
    after = ort.InferenceSession(lifted.SerializeToString()).run(None, inputs)[0]
    np.testing.assert_array_equal(after, before)
    types = {v.name: v.type.tensor_type for v in lifted.graph.value_info}
    for node in lifted.graph.node:
      if node.op_type in ("Slice", "Concat"):
        self.assertEqual(len(types[node.output[0]].shape.dim), 4)
        self.assertEqual(types[node.output[0]].elem_type, T.FLOAT16)

  def test_byte_rearrangement_is_exact_and_preserves_external_types(self):
    constants = [numpy_helper.from_array(np.array(v, np.int64), k) for k, v in
                 (("axis", [0]), ("first", 0), ("start", [0]), ("end", [256]), ("step", [2]))]
    nodes = [h.make_node("Unsqueeze", ["image", "axis"], ["expanded"]),
             h.make_node("Concat", ["state", "expanded"], ["next_state"], axis=0),
             h.make_node("Gather", ["next_state", "first"], ["selected"], axis=0),
             h.make_node("Slice", ["selected", "start", "end", "axis", "step"], ["sliced"]),
             h.make_node("Cast", ["sliced"], ["features"], to=T.FLOAT16)]
    model = h.make_model(h.make_graph(nodes, "image", [h.make_tensor_value_info("image", T.UINT8, [256]),
                         h.make_tensor_value_info("state", T.UINT8, [2, 256])],
                         [h.make_tensor_value_info("next_state", T.UINT8, [3, 256]),
                          h.make_tensor_value_info("features", T.FLOAT16, [128])], constants),
                         opset_imports=[h.make_opsetid("", 18)], ir_version=10)
    source = model.SerializeToString()
    lifted = trace.float_image_ops(onnx.load_from_string(source))
    onnx.checker.check_model(lifted)
    inputs = {"image": np.arange(256, dtype=np.uint8),
              "state": np.stack((np.arange(255, -1, -1, dtype=np.uint8), np.arange(256, dtype=np.uint8)))}
    before = ort.InferenceSession(source).run(None, inputs)
    after = ort.InferenceSession(lifted.SerializeToString()).run(None, inputs)
    for expected, actual in zip(before, after, strict=True):
      self.assertEqual(expected.dtype, actual.dtype)
      np.testing.assert_array_equal(expected, actual)
    types = {v.name: v.type.tensor_type.elem_type for v in lifted.graph.value_info}
    for node in lifted.graph.node:
      if node.op_type in ("Unsqueeze", "Concat", "Gather", "Slice"):
        self.assertEqual(types[node.output[0]], T.FLOAT16)
    self.assertEqual(next(v for v in lifted.graph.initializer if v.name == "step").data_type, T.INT64)


if __name__ == "__main__":
  unittest.main()
