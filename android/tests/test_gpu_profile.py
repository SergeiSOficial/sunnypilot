"""GPU diagnostics must reject incomplete data and retain repeated dispatch costs."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
sys.path.insert(0, str(ROOT / "android/scripts"))
from fold3_gpu_profile import summarize, summarize_submissions
from prepare_htp_probes import quantization_options
from onnxruntime.quantization import QuantType


class GpuProfileTest(unittest.TestCase):
  def sample(self):
    return {"version": 1, "enabled": True, "valid": True, "timestamp_valid_bits": 48,
            "gpu_layers": [{"node": "conv", "operator": "Convolution", "gpu_ms": 2.},
                           {"node": "conv", "operator": "Convolution", "gpu_ms": 1.},
                           {"node": "cast", "operator": "Cast", "gpu_ms": .5}]}

  def test_aggregate_repeated_dispatches_without_counting_warmup(self):
    rows = summarize([self.sample(), self.sample()])
    self.assertEqual([(r["node"], r["mean_gpu_ms"]) for r in rows], [("conv", 3.), ("cast", .5)])

  def test_incomplete_or_invalid_queries_rejected(self):
    changes = ({"valid": False}, {"enabled": False}, {"version": 2},
               {"timestamp_valid_bits": 0}, {"timestamp_valid_bits": 65}, {"gpu_layers": []})
    for change in changes:
      with self.subTest(change=change), self.assertRaises(ValueError):
        summarize([dict(self.sample(), **change)])
    with self.assertRaises(ValueError):
      summarize([])
    for duration in (-1., float("nan"), float("inf")):
      sample = copy.deepcopy(self.sample())
      sample["gpu_layers"][0]["gpu_ms"] = duration
      with self.subTest(duration=duration), self.assertRaises(ValueError):
        summarize([sample])

  def test_submission_waits_are_separated_not_subtracted_from_wall_timing(self):
    sample = {"command_record_ms": 10., "internal_submissions": {"version": 1,
      "internal_count": 2, "dispatch_limit": 262144, "rough_score": 9,
      "submitted_groups": 800000, "wait_ms": 8.}}
    result = summarize_submissions([sample, sample])
    self.assertEqual(result["mean_internal_wait_ms"], 8.)
    self.assertEqual(result["mean_record_excluding_internal_wait_ms"], 2.)
    self.assertEqual(sample["command_record_ms"], 10.)
    for field, value in [("version", 2), ("internal_count", -1), ("dispatch_limit", 1.5),
                         ("wait_ms", float("nan")), ("wait_ms", float("inf")), ("wait_ms", 12.)]:
      bad = copy.deepcopy(sample)
      bad["internal_submissions"][field] = value
      with self.subTest(field=field, value=value), self.assertRaises(ValueError):
        summarize_submissions([bad])
    with self.assertRaises(ValueError):
      summarize_submissions([])
    with self.assertRaises(ValueError):
      summarize_submissions([{}])

  def test_htp_precision_and_weight_axis_are_explicit(self):
    self.assertEqual(quantization_options("a16w16")["weight_type"], QuantType.QUInt16)
    self.assertEqual(quantization_options("a16w8")["weight_type"], QuantType.QUInt8)
    for mode, per_channel in (("a16w8-symmetric", False), ("a16w8-per-channel", True)):
      options = quantization_options(mode)
      self.assertEqual(options["weight_type"], QuantType.QInt8)
      self.assertEqual(options["activation_type"], QuantType.QUInt16)
      self.assertTrue(options["weight_symmetric"])
      self.assertEqual(options["per_channel"], per_channel)
    for mode in ("fp16", "unknown"):
      with self.assertRaises(ValueError):
        quantization_options(mode)
