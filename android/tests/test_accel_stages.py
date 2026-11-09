"""Accuracy evidence is reusable only for the same untapped implementation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
sys.path.insert(0, str(ROOT / "android/scripts"))
sys.path.insert(0, str(Path(__file__).parent))
import fold3_accel
from validate_models import validate
from test_validate_models import Device

NAMES = ("dmonitoring_model", "driving_supercombo")


def evidence(name):
  outputs = {"outputs"} if name == NAMES[0] else {"outputs", "next_state_img_q", "next_state_desire_q", "next_state_feat_q"}
  checks = {key: {"passed": True, "unexpected_nonfinite_elements": 0} for key in outputs}
  return {"identity": {"name": name, "backend": "mnn-opencl", "memory_mode": "buffer", "cpu_fallback": False,
                       "fingerprint": "test-device", "sha256": {key: "a" * 64 for key in
                         ("apk", "model", "original_model", "references", "sequences", "conversion", "provider",
                          "opencl_plugin", "driver", "implementation", "native")}},
          "status": "passed", "accuracy_status": "passed", "performance_status": "not_run", "diagnostic_only": True,
          "qualified": False, "rtol": 1e-3, "atol": 1e-4,
          "assignment": {"no_cpu_fallback_verified": True, "memory_mode": "buffer", "trace_limit": 0},
          "parity": copy.deepcopy(checks), "additional_seed_parity": {str(seed): copy.deepcopy(checks) for seed in (7, 123)},
          "sequences": {key: {"passed": True, "frames": count, "checks": [
            {"frame": frame, "outputs": copy.deepcopy(checks)} for frame in range(count)]}
            for key, count in (("seed42", 32), ("seed7", 32), ("seed123", 32), ("camera-replay", 60))}}


class AccuracyStagesTest(unittest.TestCase):
  def test_recurrent_trace_checks_hashed_image_history_without_qualifying(self):
    from fold3_sequences import output_record
    name = NAMES[1]
    outputs = {"outputs": np.zeros(1, np.float32), "next_state_img_q": np.arange(8, dtype=np.uint8)}
    record = evidence(name)
    class Model:
      session = None
      native = SimpleNamespace(trace=lambda session: [])
      def __init__(self, *args): pass
      def describe(self): return record["assignment"]
      def __call__(self, inputs): return outputs
    with tempfile.TemporaryDirectory() as temporary:
      folder = Path(temporary)
      root = folder / "fold3_reference"
      (root / "accelerators").mkdir(parents=True)
      (root / "sequences").mkdir()
      (root / "accelerators/manifest.json").write_text(json.dumps({"models": {name: {
        "original_reference": {"passed": True}, "mnn": {"status": "converted", "sha256": "a"}}}}))
      np.savez(root / f"sequences/{name}-seed42.npz", outputs=np.zeros((1, 1), np.float32))
      (root / "sequences/manifest.json").write_text(json.dumps({"models": {name: {"seed42": {
        "uint8_outputs": [output_record(outputs)]}}}}))
      with patch.object(fold3_accel, "__file__", str(folder / "fold3_accel.py")), \
           patch.object(fold3_accel, "AccelModel", Model), patch.object(fold3_accel, "identity", return_value=record["identity"]), \
           patch("fold3_sequences.digest", return_value="a"), \
           patch("fold3_sequences.initial_inputs", return_value={"new_img": np.zeros((1, 4), np.uint8)}), \
           patch("fold3_models._thermal_check"):
        report = fold3_accel.benchmark(folder, "mnn-opencl", folder, "buffer", trace_limit=1,
          trace_case={"seed": 42, "frame": 0}, selected_model=name)
      entry = report["models"][name]
      self.assertEqual(entry["status"], "diagnostic_only", entry)
      self.assertTrue(all(check["passed"] for check in entry["parity"].values()))
      self.assertFalse(entry["qualified"])
      self.assertFalse(list(folder.glob("accel-accuracy-*.json")))

  def test_numerical_evidence_cannot_qualify_live_inference(self):
    for name, comparisons in zip(NAMES, (159, 636), strict=True):
      entry = evidence(name)
      self.assertTrue(fold3_accel.accuracy_passed(entry, entry["identity"]))
      self.assertFalse(fold3_accel.qualified(entry, entry["identity"]))
      self.assertEqual((3 + sum(v["frames"] for v in entry["sequences"].values())) * len(entry["parity"]), comparisons)

  def test_proofs_invalidate_changed_artifacts_fallback_trace_or_incomplete_outputs(self):
    entry = evidence(NAMES[1])
    for key in entry["identity"]["sha256"]:
      changed = copy.deepcopy(entry["identity"])
      changed["sha256"][key] = "b" * 64
      self.assertFalse(fold3_accel.accuracy_passed(entry, changed), key)
    mutations = [lambda e: e.update(rtol=1e-2), lambda e: e.update(atol=1e-3),
                 lambda e: e["identity"].update(cpu_fallback=True),
                 lambda e: e["assignment"].update(no_cpu_fallback_verified=False),
                 lambda e: e["assignment"].update(trace_limit=32),
                 lambda e: e["assignment"].update(profiling_enabled=True),
                 lambda e: e["assignment"].update(memory_mode="image"),
                 lambda e: e["additional_seed_parity"].pop("7"),
                 lambda e: e["parity"].pop("next_state_img_q"),
                 lambda e: e["sequences"]["camera-replay"]["checks"].pop(),
                 lambda e: e["sequences"]["seed123"]["checks"][27]["outputs"].pop("outputs"),
                 lambda e: e["sequences"]["seed7"]["checks"][2]["outputs"]["outputs"].update(unexpected_nonfinite_elements=1)]
    for index, mutate in enumerate(mutations):
      changed = copy.deepcopy(entry)
      mutate(changed)
      self.assertFalse(fold3_accel.accuracy_passed(changed, changed["identity"]), index)

  def _run_benchmark(self, folder, stage, selected=None, write_proofs=(), altered_identity=False, part=None, missing_conversion=False):
    root = folder / "fold3_reference"
    (root / "accelerators").mkdir(parents=True, exist_ok=True)
    records = {name: evidence(name) for name in NAMES}
    metadata = {name: {"name": name, "original_reference": {"passed": True}, "mnn": {"status": "converted", "sha256": "a"}}
                for name in NAMES}
    if missing_conversion:
      metadata[NAMES[1]].pop("mnn")
    (root / "accelerators/manifest.json").write_text(json.dumps({"models": metadata}))
    for name in NAMES:
      for seed in (42, 7, 123):
        suffix = "" if seed == 42 else f"-case-{seed}"
        np.savez(root / f"{name}{suffix}-outputs.npz", **{key: np.zeros(1, np.float32) for key in records[name]["parity"]})
    for name in write_proofs:
      (folder / f"accel-accuracy-mnn-opencl-buffer-{name}.json").write_text(json.dumps(records[name]))
    calls = []

    class Model:
      def __init__(self, path, backend, meta, *args):
        self.name = meta["name"]
        calls.append(("create", self.name))

      def describe(self):
        return records[self.name]["assignment"]

      def __call__(self, inputs):
        calls.append(("run", self.name))
        return {key: np.zeros(1, np.float32) for key in records[self.name]["parity"]}

    def sequence(model, root, name, thermal, labels=None):
      calls.append(("validate", name))
      return {key: value for key, value in records[name]["sequences"].items() if labels is None or key in labels}

    def identity(name, *args):
      result = copy.deepcopy(records[name]["identity"])
      if altered_identity:
        result["sha256"]["apk"] = "b" * 64
      return result

    with patch.object(fold3_accel, "__file__", str(folder / "fold3_accel.py")), \
         patch.object(fold3_accel, "AccelModel", Model), patch.object(fold3_accel, "identity", identity), \
         patch("fold3_sequences.digest", return_value="a"), patch("fold3_sequences.initial_inputs", return_value={}), \
         patch("fold3_sequences.validate", side_effect=sequence), patch("fold3_models._thermal_check"), \
         patch("fold3_accel.time.sleep"):
      report = fold3_accel.benchmark(folder, "mnn-opencl", folder, "buffer", stage=stage, selected_model=selected, accuracy_part=part)
    return report, calls

  def test_all_795_comparisons_precede_any_timing(self):
    with tempfile.TemporaryDirectory() as temporary:
      report, calls = self._run_benchmark(Path(temporary), "all")
      self.assertEqual(report["timing_accuracy_gate"]["comparisons"], 795)
      self.assertEqual(calls[:10], [("create", NAMES[0]), *[("run", NAMES[0])] * 3, ("validate", NAMES[0]),
                                  ("create", NAMES[1]), *[("run", NAMES[1])] * 3, ("validate", NAMES[1])])
      self.assertTrue(all(entry["qualified"] for entry in report["models"].values()))

  def test_missing_conversion_blocks_execution_and_timing(self):
    with tempfile.TemporaryDirectory() as temporary:
      report, calls = self._run_benchmark(Path(temporary), "all", missing_conversion=True)
      entry = report["models"][NAMES[1]]
      self.assertEqual(entry["status"], "blocked")
      self.assertIn("missing conversion record", entry["error"])
      self.assertNotIn(("create", NAMES[1]), calls)
      self.assertNotIn("timing_accuracy_gate", report)

  def test_separate_accuracy_run_does_not_time_or_qualify(self):
    with tempfile.TemporaryDirectory() as temporary:
      folder = Path(temporary)
      report, calls = self._run_benchmark(folder, "accuracy", NAMES[1])
      self.assertEqual(set(report["models"]), {NAMES[1]})
      self.assertNotIn("timing_accuracy_gate", report)
      self.assertEqual(calls.count(("run", NAMES[1])), 3)
      self.assertFalse(list(folder.glob("accel-qualification-*.json")))
      self.assertTrue(fold3_accel.accuracy_passed(report["models"][NAMES[1]], report["models"][NAMES[1]]["identity"]))

  def test_timing_requires_both_current_model_proofs_before_session_creation(self):
    for proofs, changed in (((), False), ((NAMES[0],), False), (NAMES, True)):
      with self.subTest(proofs=proofs, changed=changed), tempfile.TemporaryDirectory() as temporary:
        report, calls = self._run_benchmark(Path(temporary), "timing", NAMES[0], proofs, changed)
        self.assertEqual(calls, [])
        self.assertIn("timing_blocked_by", report)
        self.assertNotIn("timing_accuracy_gate", report)

  def test_host_acceptance_distinguishes_accuracy_from_performance(self):
    with tempfile.TemporaryDirectory() as temporary:
      entry = evidence(NAMES[0])
      report = {"started_utc": 2, "completed_utc": 3, "vehicle_tx_enabled": False,
                "stage": "accuracy", "models": {NAMES[0]: entry}}
      output = Path(temporary) / "report.json"
      self.assertEqual(validate(Device([{}, report]), "mnn-buffer-accuracy-driver", output)["validation_status"], "passed")
      self.assertEqual(validate(Device([{}, report]), "mnn-buffer-timing-driver", output)["validation_status"], "blocked")
      entry["sequences"]["camera-replay"]["checks"].pop()
      self.assertEqual(validate(Device([{}, report]), "mnn-buffer-accuracy-driver", output)["validation_status"], "blocked")

  def test_isolated_parts_preserve_all_comparisons_without_qualifying_performance(self):
    for name in NAMES:
      with self.subTest(model=name), tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        for part in fold3_accel.ACCURACY_PARTS:
          report, calls = self._run_benchmark(folder, "accuracy", name, part=part)
          entry = report["models"][name]
          self.assertTrue(fold3_accel.accuracy_part_passed(entry, entry["identity"], part), entry)
          self.assertFalse(fold3_accel.accuracy_passed(entry, entry["identity"]))
          self.assertFalse(fold3_accel.qualified(entry, entry["identity"]))
          self.assertEqual(calls.count(("run", name)), 3 if part == "initial" else 0)
          self.assertEqual(calls.count(("validate", name)), 0 if part == "initial" else 1)
          if part != "camera-replay":
            self.assertNotIn("accuracy_aggregate", report)
        combined = report["accuracy_aggregate"][name]
        self.assertTrue(fold3_accel.accuracy_passed(combined, combined["identity"]))
        self.assertEqual(set(combined["accuracy_parts"]), set(fold3_accel.ACCURACY_PARTS))
        self.assertFalse(fold3_accel.qualified(combined, combined["identity"]))

  def test_parts_cannot_mix_artifacts_or_skip_frames_and_failed_retry_invalidates_proof(self):
    with tempfile.TemporaryDirectory() as temporary:
      folder = Path(temporary)
      for part in fold3_accel.ACCURACY_PARTS:
        self._run_benchmark(folder, "accuracy", NAMES[1], part=part)
      parts = {part: json.loads((folder / f"accel-accuracy-part-mnn-opencl-buffer-{NAMES[1]}-{part}.json").read_text())
               for part in fold3_accel.ACCURACY_PARTS}
      current = parts["initial"]["identity"]
      self.assertIsNotNone(fold3_accel.combine_accuracy_parts(parts, current))
      for part in parts:
        altered = copy.deepcopy(parts)
        altered[part]["identity"]["sha256"]["apk"] = "b" * 64
        self.assertIsNone(fold3_accel.combine_accuracy_parts(altered, current), part)
        altered = copy.deepcopy(parts)
        altered.pop(part)
        self.assertIsNone(fold3_accel.combine_accuracy_parts(altered, current), part)
      for mutate in (lambda e: e["sequences"]["seed42"]["checks"].pop(),
                     lambda e: e["sequences"]["seed42"]["checks"][1].update(frame=0),
                     lambda e: e["assignment"].update(trace_limit=1),
                     lambda e: e.update(atol=1e-3)):
        altered = copy.deepcopy(parts)
        mutate(altered["seed42"])
        self.assertIsNone(fold3_accel.combine_accuracy_parts(altered, current))
      self._run_benchmark(folder, "accuracy", NAMES[1], altered_identity=True, part="seed42")
      self.assertFalse((folder / f"accel-accuracy-mnn-opencl-buffer-{NAMES[1]}.json").exists())

  def test_host_accepts_only_requested_part_and_never_timing(self):
    with tempfile.TemporaryDirectory() as temporary:
      folder = Path(temporary)
      report, _ = self._run_benchmark(folder, "accuracy", NAMES[1], part="seed7")
      output = folder / "report.json"
      self.assertEqual(validate(Device([{}, report]), "mnn-buffer-accuracy-road-seed7", output)["validation_status"], "passed")
      self.assertEqual(validate(Device([{}, report]), "mnn-buffer-accuracy-road-seed42", output)["validation_status"], "blocked")
      self.assertEqual(validate(Device([{}, report]), "mnn-buffer-timing-road", output)["validation_status"], "blocked")


if __name__ == "__main__":
  unittest.main()
