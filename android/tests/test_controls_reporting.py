"""Control timing handoff preserves samples without sorting on its loop or GIL."""
import json
import base64
import struct
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app/src/main/python"))
from fold3_daemons import Report


class ControlsReportingTest(unittest.TestCase):
  def test_controls_transient_phases_keep_heartbeat_frequency_and_immediate_failures(self):
    with tempfile.TemporaryDirectory() as temporary:
      worker = Mock()
      report = Report("controlsd", temporary, worker)
      try:
        # Keep batch I/O out of this focused heartbeat test. Alternating input
        # availability must not turn the heartbeat into a 100 Hz Java call.
        report.last_write = 100.
        with patch("fold3_daemons.time.monotonic") as clock:
          for index in range(100):
            clock.return_value = 100. + index / 100
            report.update("running" if index % 2 else "waiting_for_selfdrive_state")
          self.assertEqual(worker.progress.call_count, 4)
          clock.return_value = 100.991
          report.update("failed", error="injected reporting failure")
          self.assertEqual(worker.progress.call_count, 5)
          self.assertEqual(worker.progress.call_args.args, ("failed",))
      finally:
        report.close()

  def drain(self, report):
    deadline = time.monotonic() + 2
    while report.pending.qsize() and time.monotonic() < deadline:
      time.sleep(.005)

  def test_batches_preserve_every_sample_and_deadline_counter(self):
    with tempfile.TemporaryDirectory() as temporary:
      worker = Mock()
      report = Report("controlsd", temporary, worker)
      timings = {"cycle_before_reporting_ms": 12., "release_to_completion_before_reporting_ms": 2.,
                 "active_cycle_before_reporting_ms": 2., "skipped_slots": 1, "scheduler_delay_ms": .1}
      try:
        with patch("fold3_daemons.sorted", side_effect=AssertionError("Controls process sorted timing samples"), create=True):
          for _ in range(20):
            report.controls_progress(False, 2_000_000, timings)
          self.drain(report)
          report.last_write = 0
          report.update("waiting_for_selfdrive_state")
          report.close()
        batches = [json.loads(path.read_text()) for path in sorted((Path(temporary) / "controlsd-batches").glob("*.json"))]
        self.assertEqual([item["sequence"] for item in batches], [1, 2])
        self.assertEqual(sum(item["row_count"] for item in batches), 20)
        for item in batches:
          self.assertEqual(item["version"], 3)
          raw = base64.b64decode(item["rows_base64"], validate=True)
          count = item["row_count"] * len(item["columns"])
          self.assertEqual(len(raw), count * 8)
          values = struct.unpack("<" + "d" * count, raw)
          if count:
            self.assertEqual(values[0], 2_000_000.)
          self.assertNotIn("rows", item)
        data = batches[-1]["data"]
        self.assertEqual(data["work_samples"], 20)
        self.assertEqual(data["cycle_deadline_misses"], 20)
        self.assertEqual(data["scheduled_deadline_misses"], 0)
        self.assertEqual(data["skipped_slots_total"], 20)
        self.assertEqual(data["updated_input_samples"], 0)
        self.assertEqual(len(report.samples), 0)
        self.assertFalse((Path(temporary) / "controlsd.json").exists())
        self.assertGreaterEqual(worker.progress.call_count, 1)
      finally:
        if report.writer.is_alive():
          report.close()

  def test_binary_handoff_preserves_numerical_fault_for_manager_rejection(self):
    with tempfile.TemporaryDirectory() as temporary:
      report = Report("controlsd", temporary, Mock())
      # Put the fault in an earlier row, not the latest JSON summary. The
      # binary transport must preserve it for ControlsMetrics.Window.add.
      report.last_write = time.monotonic()
      report.controls_progress(True, float("inf"), {"cycle_before_reporting_ms": 10.,
          "release_to_completion_before_reporting_ms": 10., "skipped_slots": 0})
      report.last_write = 0
      report.update("running")
      report.close()
      self.assertIsNone(report.write_error)
      batch = json.loads(next(report.batch_directory.glob("*.json")).read_text())
      values = struct.unpack("<" + "d" * len(batch["columns"]), base64.b64decode(batch["rows_base64"], validate=True))
      self.assertEqual(values[0], float("inf"))
      self.assertGreaterEqual(values[1], 10.)

  def test_writer_owns_buffer_until_serialization_finishes(self):
    with tempfile.TemporaryDirectory() as temporary:
      report = Report("controlsd", temporary, Mock())
      report.last_write = time.monotonic()
      timings = {"cycle_before_reporting_ms": 2., "release_to_completion_before_reporting_ms": 2., "skipped_slots": 0}
      entered, release = threading.Event(), threading.Event()
      write = Path.write_text
      def slow_write(path, *args, **kwargs):
        entered.set()
        if not release.wait(2):
          raise TimeoutError("Test did not release writer")
        return write(path, *args, **kwargs)
      try:
        report.controls_progress(True, 1_000_000, timings)
        owned = report.control_buffer
        with patch.object(Path, "write_text", slow_write):
          report.last_write = 0
          report.update("running")
          self.assertTrue(entered.wait(2))
          report.controls_progress(True, 2_000_000, timings)
          self.assertIsNot(report.control_buffer, owned)
          self.assertEqual(owned[0], 1_000_000.)
          self.assertEqual(report.control_buffer[0], 2_000_000.)
          release.set()
          report.close()
        batch = json.loads(next(report.batch_directory.glob("*.json")).read_text())
        raw = base64.b64decode(batch["rows_base64"], validate=True)
        self.assertEqual(len(raw), len(batch["columns"]) * 8)
        self.assertEqual(struct.unpack_from("<d", raw)[0], 1_000_000.)
      finally:
        release.set()
        if report.writer.is_alive():
          report.close()

  def test_writer_failure_remains_fatal_to_supervision(self):
    with tempfile.TemporaryDirectory() as temporary:
      report = Report("controlsd", temporary, Mock())
      report.write_error = OSError("Storage unavailable")
      try:
        with self.assertRaisesRegex(RuntimeError, "report writer failed"):
          report.update("running")
      finally:
        report.close()

  def test_stalled_consumer_cannot_silently_discard_samples(self):
    with tempfile.TemporaryDirectory() as temporary:
      report = Report("controlsd", temporary, Mock())
      for index in range(8):
        (report.batch_directory / f"old-{index}.json").write_text("{}")
      report.update("initializing_controls")
      report.close()
      self.assertIsNotNone(report.write_error)
      self.assertIn("consumer stalled", str(report.write_error))


if __name__ == "__main__":
  unittest.main()
