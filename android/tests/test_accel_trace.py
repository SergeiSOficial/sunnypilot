"""Accelerator trace provenance must not guess matching tensors by size."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_accel_trace import compare_trace, original_mapping


class AcceleratorTraceTest(unittest.TestCase):
  def test_reduction_callback_unit_batch_is_explicit_and_not_a_general_reshape(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      references, folder = root / "references", root / "trace"
      (references / "accelerators").mkdir(parents=True)
      folder.mkdir()
      values = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
      np.savez(references / "dmonitoring_model-inputs.npz", input_img=values)
      model = helper.make_model(helper.make_graph([
        helper.make_node("ReduceMean", ["input_img"], ["y"], name="mean", axes=[2, 3], keepdims=1)], "trace",
        [helper.make_tensor_value_info("input_img", TensorProto.FLOAT, [1, 2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1, 1])]),
        opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
      onnx.save(model, references / "accelerators/dmonitoring_model-mnn-io.onnx")
      values.mean(axis=(2, 3), keepdims=True).tofile(folder / "0000.bin")
      tensor = {"tensor": "y", "operator": "mean", "operator_type": "Reduction", "layout": "NCHW",
                "file": "0000.bin", "dtype": "float32", "shape": [2, 1, 1]}
      manifest = {"model": "dmonitoring_model", "seed": 42, "memory_mode": "buffer", "tensors": [tensor]}
      for layout in ("NCHW", "NHWC"):
        tensor["layout"] = layout
        (folder / "manifest.json").write_text(json.dumps(manifest))
        result = compare_trace(folder, references)
        self.assertEqual(result["first_tolerance_failure"] is None, layout == "NCHW")
        self.assertEqual(result["checks"][0]["shape"], [2, 1, 1])
        self.assertEqual(result["checks"][0].get("trace_shape_adapter"),
                         "restore_unit_reduction_batch" if layout == "NCHW" else None)

  def test_recurrent_archive_requires_hash_schema_and_actual_feedback_provenance(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      references, folder = root / "references", root / "trace"
      (references / "accelerators").mkdir(parents=True)
      folder.mkdir()
      original = {"new_img": np.arange(4, dtype=np.uint8).reshape(1, 4), "state_x": np.zeros((1, 1), np.float32)}
      np.savez(references / "driving_supercombo-inputs.npz", **original)
      model = helper.make_model(helper.make_graph([
        helper.make_node("Identity", ["state_x"], ["y"], name="state_identity")], "trace",
        [helper.make_tensor_value_info("new_img", TensorProto.FLOAT, [1, 4]),
         helper.make_tensor_value_info("state_x", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])]),
        opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
      onnx.save(model, references / "accelerators/driving_supercombo-mnn-io.onnx")
      actual_inputs = {**original, "state_x": np.array([[78]], np.float32)}
      np.savez(folder / "inputs.npz", **actual_inputs)
      (folder / "0000.bin").write_bytes(actual_inputs["state_x"].tobytes())
      manifest = {"model": "driving_supercombo", "seed": 42, "frame": 26, "memory_mode": "buffer",
        "input_archive_sha256": hashlib.sha256((folder / "inputs.npz").read_bytes()).hexdigest(),
        "feedback_source": "actual-untapped-backend", "warmup_frames": 26,
        "inputs_sha256": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in actual_inputs.items()},
        "tensors": [{"tensor": "y", "operator": "state_identity", "file": "0000.bin", "dtype": "float32", "shape": [1, 1]}]}
      (folder / "manifest.json").write_text(json.dumps(manifest))
      self.assertIsNone(compare_trace(folder, references)["first_tolerance_failure"])
      for key, value, error in (("feedback_source", "goldens", "feedback provenance"),
                                ("warmup_frames", 0, "feedback provenance"),
                                ("input_archive_sha256", "wrong", "archive hash")):
        (folder / "manifest.json").write_text(json.dumps({**manifest, key: value}))
        with self.subTest(key=key), self.assertRaisesRegex(ValueError, error):
          compare_trace(folder, references)

  def test_selected_sequence_frame_has_exact_raw_input_provenance(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      references, folder = root / "references", root / "trace"
      (references / "accelerators").mkdir(parents=True)
      folder.mkdir()
      values = np.arange(128, dtype=np.uint8).reshape(1, 128)
      np.savez(references / "dmonitoring_model-case-123-inputs.npz", input_img=values)
      model = helper.make_model(helper.make_graph([
        helper.make_node("Identity", ["input_img"], ["y"], name="original_identity")], "trace",
        [helper.make_tensor_value_info("input_img", TensorProto.FLOAT, [1, 128])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 128])]),
        opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
      onnx.save(model, references / "accelerators/dmonitoring_model-mnn-io.onnx")
      frame = np.roll(values, 27 * 17)
      (folder / "0000.bin").write_bytes(frame.astype(np.float32).tobytes())
      manifest = {"model": "dmonitoring_model", "seed": 123, "frame": 27,
                  "memory_mode": "buffer", "inputs_sha256": {
                    "input_img": hashlib.sha256(frame.tobytes()).hexdigest()},
                  "tensors": [{"tensor": "y", "operator": "original_identity",
                               "file": "0000.bin", "dtype": "float32", "shape": [1, 128]}]}
      (folder / "manifest.json").write_text(json.dumps(manifest))
      result = compare_trace(folder, references)
      self.assertEqual((result["seed"], result["frame"]), (123, 27))
      self.assertIsNone(result["first_tolerance_failure"])
      self.assertEqual(result["checks"][0]["parity"]["max_abs_error"], 0)
      manifest["frame"] = 26
      (folder / "manifest.json").write_text(json.dumps(manifest))
      with self.assertRaisesRegex(ValueError, "Diagnostic input provenance mismatch"):
        compare_trace(folder, references)

  def test_exact_tensor_names_and_converter_prefix_have_original_provenance(self):
    model = helper.make_model(helper.make_graph([
      helper.make_node("Div", ["x", "x"], ["div"], name="original_div")], "test",
      [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 2])],
      [helper.make_tensor_value_info("div", TensorProto.FLOAT16, [1, 2])]))
    expected = {"tensor": "div", "node": "original_div", "operator": "Div", "domain": ""}
    self.assertEqual(original_mapping(model, "div", "fused"), expected)
    self.assertEqual(original_mapping(model, "", "InsertedPrecisionFreeCast_div"), expected)
    self.assertIsNone(original_mapping(model, "raster_0", "binary_9"))


if __name__ == "__main__":
  unittest.main()
