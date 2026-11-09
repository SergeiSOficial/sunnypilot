"""Original numerical references and complete recurrent qualification gates."""
import copy
import hashlib
import importlib.machinery
import json
from pathlib import Path
import py_compile
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_opencl import live_model, qualified
from fold3_model_trace import compare_replay_frame
from fold3_sequences import digest, feedback, sequence_input, validate


class ModelSequencesTest(unittest.TestCase):
  def test_isolated_sequence_keeps_actual_feedback_for_every_frame(self):
    for labels in ((), ("seed42", "seed42"), ("unknown",)):
      with self.assertRaisesRegex(ValueError, "sequence"):
        validate(None, Path("unused"), "model", labels=labels)
    with tempfile.TemporaryDirectory() as temporary:
      folder = Path(temporary) / "sequences"
      folder.mkdir()
      np.savez(folder / "camera-inputs.npz", driver=np.zeros(1))
      expected = np.arange(1, 33, dtype=np.float16).reshape(32, 1)
      np.savez(folder / "model-seed7.npz", outputs=expected, next_state_a=expected)
      assets = {path.name: digest(path) for path in folder.glob("*.npz")}
      (folder / "manifest.json").write_text(json.dumps({"rtol": 1e-3, "atol": 1e-4, "assets": assets,
        "models": {"model": {"seed7": {"seed": 7, "frames": 32, "uint8_outputs": [{} for _ in range(32)]}}}}))
      seen = []
      def model(inputs):
        seen.append(float(inputs["state_a"][0]))
        value = inputs["state_a"] + np.float16(1)
        return {"outputs": value.copy(), "next_state_a": value.copy()}
      initial = {"new_img": np.zeros((1, 4), np.uint8), "state_a": np.zeros(1, np.float16)}
      with patch("fold3_sequences.initial_inputs", return_value=initial):
        result = validate(model, temporary, "model", labels=("seed7",))
      self.assertEqual(set(result), {"seed7"})
      self.assertTrue(result["seed7"]["passed"])
      self.assertEqual(seen, list(range(32)))
      np.testing.assert_array_equal(initial["state_a"], 0)

  def test_replay_trace_checks_hashed_image_history(self):
    from fold3_sequences import output_record
    actual = {"output": np.array([[1., 2.]], dtype=np.float32),
              "next_state_img_q": np.arange(8, dtype=np.uint8)}
    hashes = output_record({"next_state_img_q": actual["next_state_img_q"]})
    with tempfile.TemporaryDirectory() as temporary:
      path = Path(temporary) / "replay.npz"
      np.savez(path, output=np.stack([actual["output"], actual["output"] + 1]))
      with np.load(path) as expected:
        checks = compare_replay_frame(actual, expected, hashes, 0)
        self.assertTrue(all(item["passed"] for item in checks.values()))
        changed = {key: value.copy() for key, value in actual.items()}
        changed["next_state_img_q"][0] ^= 1
        self.assertFalse(compare_replay_frame(changed, expected, hashes, 0)["next_state_img_q"]["passed"])
        self.assertFalse(compare_replay_frame(actual, expected, hashes, 1)["output"]["passed"])
        for invalid in ({"output": actual["output"]}, {**actual, "extra": actual["output"]}):
          with self.assertRaisesRegex(ValueError, "output set changed"):
            compare_replay_frame(invalid, expected, hashes, 0)

  def test_backend_identity_changes_with_operator_renderer_or_compiler(self):
    import fold3_opencl
    root = ROOT / "identity-fixture"
    versions = {}
    java = SimpleNamespace(jclass=lambda _: SimpleNamespace(FINGERPRINT="test-device", applicationApk=lambda: "/test/base.apk"))
    with patch.dict(sys.modules, {"java": java}), \
         patch("fold3_sequences.digest", return_value="reference"), \
         patch("fold3_opencl.code_digest", side_effect=lambda module: versions.get(module, "original")):
      original = fold3_opencl.identity("test", root / "model.onnx", root / "references")
      for module in ("fold3_opencl", "tinygrad.nn.onnx", "tinygrad.uop.ops", "tinygrad.renderer.cstyle", "tinygrad.runtime.ops_cl"):
        with self.subTest(module=module):
          versions[module] = "changed"
          self.assertNotEqual(fold3_opencl.identity("test", root / "model.onnx", root / "references"), original)
          versions.clear()

  def test_code_identity_supports_sourceless_packaged_modules(self):
    from fold3_opencl import code_digest
    with tempfile.TemporaryDirectory() as temporary:
      source = Path(temporary) / "packaged.py"
      bytecode = source.with_suffix(".pyc")
      source.write_text("VALUE = 1\n")
      for sourceless in (False, True):
        if sourceless:
          py_compile.compile(str(source), cfile=str(bytecode), doraise=True)
          source.unlink()
        asset = bytecode if sourceless else source
        loader_type = importlib.machinery.SourcelessFileLoader if sourceless else importlib.machinery.SourceFileLoader
        loader = loader_type("packaged", str(asset))
        with patch("importlib.util.find_spec", return_value=SimpleNamespace(loader=loader)):
          original = code_digest("packaged")
          self.assertEqual(original, hashlib.sha256(asset.read_bytes()).hexdigest())
          asset.write_bytes(asset.read_bytes() + b"changed")
          self.assertNotEqual(code_digest("packaged"), original)
        if not sourceless:
          source.write_text("VALUE = 1\n")

  def test_feedback_rejects_nonfinite_outputs_and_invalid_state_layout(self):
    inputs = {"state_a": np.zeros((1, 2), dtype=np.float16)}
    for value in (np.full((1, 2), np.nan, dtype=np.float16), np.ones((2, 1), dtype=np.float16),
                  np.ones((1, 2), dtype=np.float32)):
      with self.assertRaises(ValueError):
        feedback(inputs, {"next_state_a": value})
      np.testing.assert_array_equal(inputs["state_a"], 0)
    feedback(inputs, {"next_state_a": np.ones((1, 2), dtype=np.float16)})
    initial = {"new_img": np.arange(40).reshape(1, 40), "state_a": np.zeros((1, 2), dtype=np.float16)}
    inputs["new_img"] = initial["new_img"].copy()
    sequence_input(inputs, initial, 1)
    np.testing.assert_array_equal(inputs["new_img"], np.roll(initial["new_img"], 17, axis=-1))
    np.testing.assert_array_equal(inputs["state_a"], 1)

  def test_live_qualification_requires_every_seed_recurrent_frame_and_replay_output(self):
    entry = {"status": "passed", "performance_status": "passed", "samples": 100, "work_p99_ms": 49.0,
             "identity": {"backend": "CL", "cpu_fallback": False, "fingerprint": "test-device", "sha256": {"apk": "1" * 64}},
             "deadline_misses_50ms": 0, "parity": {"output": {"passed": True}},
             "additional_seed_parity": {key: {"output": {"passed": True}} for key in ("7", "123")},
             "sequences": {key: {"passed": True, "frames": count,
                                  "checks": [{"frame": frame, "outputs": {"output": {"passed": True}}}
                                             for frame in range(count)]}
                           for key, count in (("seed42", 32), ("seed7", 32), ("seed123", 32), ("camera-replay", 60))}}
    self.assertTrue(qualified(entry))
    for case in (None, {}, {**entry, "performance_status": "failed"}, {**entry, "samples": 99},
                 {**entry, "deadline_misses_50ms": 1}, {**entry, "work_p99_ms": 50},
                 {**entry, "work_p99_ms": float("nan")},
                 {**entry, "identity": {}}, {**entry, "identity": {**entry["identity"], "cpu_fallback": True}},
                 {**entry, "additional_seed_parity": {"7": entry["additional_seed_parity"]["7"]}}):
      self.assertFalse(qualified(case))
    for key in entry["sequences"]:
      missing = copy.deepcopy(entry)
      missing["sequences"][key]["checks"].pop()
      self.assertFalse(qualified(missing))
      wrong = copy.deepcopy(entry)
      wrong["sequences"][key]["checks"][-1]["outputs"]["output"]["passed"] = False
      self.assertFalse(qualified(wrong))

  def test_host_fused_graph_matches_originals_over_all_recurrent_and_camera_frames(self):
    references = ROOT / "android/app/build/model-reference"
    manifest = json.loads((references / "host-fusion/manifest.json").read_text())
    for name, metadata in manifest["models"].items():
      with self.subTest(model=name):
        self.assertEqual(digest(ROOT / f"openpilot/selfdrive/modeld/models/{name}.onnx"), metadata["original_sha256"])
        path = references / f"host-fusion/{name}.onnx"
        self.assertEqual(digest(path), metadata["sha256"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        model = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        names = [spec.name for spec in model.get_outputs()]
        results = validate(lambda values, names=names, model=model: dict(zip(names, model.run(None, values), strict=True)), references, name)
        self.assertEqual(set(results), {"seed42", "seed7", "seed123", "camera-replay"})
        for label, result in results.items():
          self.assertTrue(result["passed"], (name, label, [frame for frame in result["checks"]
                                                        if any(not item["passed"] for item in frame["outputs"].values())][:1]))

  def test_native_mad_diagnostic_cannot_select_a_live_backend(self):
    with tempfile.TemporaryDirectory() as directory:
      Path(directory, "model-qualification.json").write_text(json.dumps({
        "backend": "CL", "profile": "host-fusion-cl-mad-road", "models": {}}))
      fallback = object()
      model = Mock(return_value=fallback)
      with patch.dict(sys.modules, {"fold3_ort": SimpleNamespace(OrtModel=model)}):
        actual, reason = live_model(directory, "driving_supercombo", Path("original.onnx"))
      self.assertIs(actual, fallback)
      self.assertIn("qualification blocked", reason)
      model.assert_called_once_with(Path("original.onnx"), "cpu", threads=2)


if __name__ == "__main__":
  unittest.main()
