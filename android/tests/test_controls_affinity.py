"""CPU placement diagnostics must not affect physical or ordinary sessions."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app/src/main/python"))
from fold3_daemons import controls, controls_affinity, controls_cpu_placement


class ControlsAffinityTest(unittest.TestCase):
  def test_controls_priority_is_scoped_and_observed_at_the_service_entry(self):
    for model, expected, fail in (("SM-F926B", -16, False), ("sdk_gphone64_arm64", -16, False), ("SM-F926B", -16, True)):
      with self.subTest(model=model):
        process = Mock(THREAD_PRIORITY_DISPLAY=-4, THREAD_PRIORITY_AUDIO=-16)
        process.getThreadPriority.return_value = expected
        instance = Mock()
        previous_interval = sys.getswitchinterval()
        def run(fail=fail, previous_interval=previous_interval, **kwargs):
          self.assertEqual(sys.getswitchinterval(), previous_interval)
          if fail:
            raise RuntimeError("injected controls fault")
        instance.run.side_effect = run
        report = Mock(data={})
        modules = {
          "java": SimpleNamespace(jclass=lambda name, process=process, model=model: process if name == "android.os.Process" else SimpleNamespace(MODEL=model)),
          "openpilot.common.params": SimpleNamespace(Params=lambda: Mock()),
          "openpilot.common.realtime": SimpleNamespace(config_realtime_process=Mock(), Priority=SimpleNamespace(CTRL_HIGH=1)),
          "openpilot.selfdrive.controls.controlsd": SimpleNamespace(Controls=Mock(return_value=instance))}
        with patch.dict(sys.modules, modules), patch("fold3_daemons.os.sched_getaffinity", return_value={0, 4, 6}, create=True), \
             patch("fold3_daemons.controls_cpu_placement"), patch("fold3_daemons.controls_affinity"), \
             patch("ctypes.CDLL", side_effect=OSError("host has no Android prctl")):
          if fail:
            with self.assertRaisesRegex(RuntimeError, "injected controls fault"):
              controls(report)
          else:
            controls(report)
        self.assertEqual(sys.getswitchinterval(), previous_interval)
        self.assertEqual(report.data["android_controls_gil_switch_interval_s"], previous_interval)
        process.setThreadPriority.assert_called_once_with(expected)
        self.assertEqual(report.data["android_thread_priority"], expected)
        modules["openpilot.selfdrive.controls.controlsd"].Controls.assert_called_once_with(android=True)
        instance.run.assert_called_once_with(progress=report.controls_progress, android=True)

  def test_measured_fold_pair_is_applied_to_calling_thread(self):
    with tempfile.TemporaryDirectory() as directory:
      report = SimpleNamespace(path=Path(directory) / "controlsd.json", data={"simulation": True})
      with patch("fold3_daemons.os.sched_getaffinity", side_effect=[{0, 4, 5, 7}, {4, 5}], create=True), \
           patch("fold3_daemons.os.sched_setaffinity", create=True) as change, \
           patch("fold3_daemons.Path.read_text", autospec=True,
                 side_effect=lambda path: "325" if "cpu0/" in str(path) else "1024" if "cpu7/" in str(path) else "828"):
        controls_cpu_placement(report, "SM-F926B")
      change.assert_called_once_with(0, [4, 5])
      self.assertEqual(report.data["android_controls_cpu_placement"]["after"], [4, 5])
      self.assertEqual(report.data["android_controls_cpu_placement"]["status"], "applied")

  def test_other_devices_and_explicit_diagnostics_keep_existing_policy(self):
    with tempfile.TemporaryDirectory() as directory, patch("fold3_daemons.os.sched_setaffinity", create=True) as change:
      report = SimpleNamespace(path=Path(directory) / "controlsd.json", data={"simulation": True})
      controls_cpu_placement(report, "sdk_gphone64_arm64")
      (report.path.parent / "controls-affinity-diagnostic.json").write_text("[4, 5]")
      controls_cpu_placement(report, "SM-F926B")
      change.assert_not_called()
      self.assertNotIn("android_controls_cpu_placement", report.data)

  def test_measured_alternative_pair_handles_cpu5_offline(self):
    for allowed in ({0, 1, 4, 6, 7}, {0, 1, 4, 5, 6, 7}):
      with self.subTest(allowed=allowed), tempfile.TemporaryDirectory() as directory:
        report = SimpleNamespace(path=Path(directory) / "controlsd.json", data={"simulation": True})
        with patch("fold3_daemons.os.sched_getaffinity", side_effect=[allowed, {4, 6}], create=True), \
             patch("fold3_daemons.os.sched_setaffinity", create=True) as change, \
             patch("fold3_daemons.Path.read_text", autospec=True,
                   side_effect=lambda path: "325" if any(f"cpu{i}/" in str(path) for i in (0, 1)) else
                                            "1024" if "cpu7/" in str(path) else "828"):
          controls_cpu_placement(report, "SM-F926B")
        change.assert_called_once_with(0, [4, 6])
        policy = report.data["android_controls_cpu_placement"]
        self.assertEqual(policy["policy"], "fold3-performance-pair-v3")
        self.assertEqual(policy["after"], [4, 6])

  def test_cpu4_unavailable_uses_remaining_homogeneous_pair(self):
    for capacity6, expected in ((828, [5, 6]), (1024, None)):
      with self.subTest(capacity6=capacity6), tempfile.TemporaryDirectory() as directory:
        report = SimpleNamespace(path=Path(directory) / "controlsd.json", data={"simulation": True})
        with patch("fold3_daemons.os.sched_getaffinity", side_effect=[{0, 1, 2, 5, 6}, {5, 6}], create=True), \
             patch("fold3_daemons.os.sched_setaffinity", create=True) as change, \
             patch("fold3_daemons.Path.read_text", autospec=True,
                   side_effect=lambda path, capacity6=capacity6: str(capacity6) if "cpu6/" in str(path) else
                                            "828" if "cpu5/" in str(path) else "325"):
          controls_cpu_placement(report, "SM-F926B")
        policy = report.data["android_controls_cpu_placement"]
        if expected is None:
          change.assert_not_called()
          self.assertEqual(policy["status"], "unavailable")
        else:
          change.assert_called_once_with(0, expected)
          self.assertEqual(policy["after"], expected)
          self.assertEqual(policy["status"], "applied")

  def test_unavailable_topology_or_permission_failure_remains_visible(self):
    with tempfile.TemporaryDirectory() as directory:
      report = SimpleNamespace(path=Path(directory) / "controlsd.json", data={})
      with patch("fold3_daemons.os.sched_getaffinity", return_value={0, 4}, create=True), \
           patch("fold3_daemons.os.sched_setaffinity", create=True) as change, \
           patch("fold3_daemons.Path.read_text", return_value="325"):
        controls_cpu_placement(report, "SM-F926B")
      change.assert_not_called()
      self.assertEqual(report.data["android_controls_cpu_placement"]["status"], "unavailable")
      with patch("fold3_daemons.os.sched_getaffinity", return_value={0, 4, 5}, create=True), \
           patch("fold3_daemons.os.sched_setaffinity", side_effect=PermissionError("denied"), create=True), \
           patch("fold3_daemons.Path.read_text", autospec=True,
                 side_effect=lambda path: "325" if "cpu0/" in str(path) else "828"):
        controls_cpu_placement(report, "SM-F926B")
      self.assertIn("denied", report.data["android_controls_cpu_placement"]["error"])

  def test_physical_and_unconfigured_sessions_do_not_change_affinity(self):
    with tempfile.TemporaryDirectory() as directory, patch("fold3_daemons.os.sched_setaffinity", create=True) as change:
      report = SimpleNamespace(path=Path(directory) / "controlsd.json", data={"simulation": True})
      controls_affinity(report)
      (report.path.parent / "controls-affinity-diagnostic.json").write_text("[4]")
      report.data["simulation"] = False
      controls_affinity(report)
      change.assert_not_called()

  def test_applies_and_verifies_only_current_thread(self):
    with tempfile.TemporaryDirectory() as directory:
      report = SimpleNamespace(path=Path(directory) / "controlsd.json", data={"simulation": True})
      (report.path.parent / "controls-affinity-diagnostic.json").write_text("[4, 5]")
      with patch("fold3_daemons.os.sched_getaffinity", side_effect=[{0, 4, 5}, {4, 5}], create=True), \
           patch("fold3_daemons.os.sched_setaffinity", create=True) as change:
        controls_affinity(report)
      change.assert_called_once_with(0, [4, 5])
      self.assertEqual(report.data["android_controls_affinity"],
                       {"diagnostic_only": True, "before": [0, 4, 5], "after": [4, 5]})

  def test_rejects_invalid_or_unavailable_cpus(self):
    with tempfile.TemporaryDirectory() as directory:
      report = SimpleNamespace(path=Path(directory) / "controlsd.json", data={"simulation": True})
      configuration = report.path.parent / "controls-affinity-diagnostic.json"
      with patch("fold3_daemons.os.sched_getaffinity", return_value={4, 5}, create=True), \
           patch("fold3_daemons.os.sched_setaffinity", create=True) as change:
        for value in ([], [0], [True], "4", {"cpu": 4}):
          configuration.write_text(json.dumps(value))
          with self.subTest(value=value), self.assertRaises(ValueError):
            controls_affinity(report)
        change.assert_not_called()

  def test_placement_failure_remains_visible(self):
    with tempfile.TemporaryDirectory() as directory:
      report = SimpleNamespace(path=Path(directory) / "controlsd.json", data={"simulation": True})
      (report.path.parent / "controls-affinity-diagnostic.json").write_text("[4]")
      with patch("fold3_daemons.os.sched_getaffinity", return_value={0, 4}, create=True), \
           patch("fold3_daemons.os.sched_setaffinity", create=True):
        with self.assertRaisesRegex(RuntimeError, "not applied"):
          controls_affinity(report)
