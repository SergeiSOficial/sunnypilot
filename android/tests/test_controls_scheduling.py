"""Exercise the actual controls loop without loading hardware/native bindings."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class StopLoop(Exception):
  pass


class SchedulingTest(unittest.TestCase):
  def exercise(self, android, immediate_input=False, work_ms=None, wake_delay_ms=0, reporting_ms=0):
    source = Path(__file__).resolve().parents[2] / "openpilot/selfdrive/controls/controlsd.py"
    tree = ast.parse(source.read_text())
    controls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Controls")
    run = next(node for node in controls.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    namespace = {"DT_CTRL": .01, "Ratekeeper": lambda *a, **k: SimpleNamespace(monitor_time=lambda: None)}
    exec(compile(ast.Module(body=[run], type_ignores=[]), str(source), "exec"), namespace)
    clock, cpu_clock, waits, reports, computes = [0], [0], [], [], []

    def update(timeout):
      waits.append(timeout)
      if len(waits) > 3:
        raise StopLoop
      if not immediate_input:
        clock[0] += timeout * 1_000_000

    def compute():
      computes.append(clock[0])
      clock[0] += round((work_ms if work_ms is not None else 2 if immediate_input else 12) * 1e6)
      cpu_clock[0] += round((work_ms if work_ms is not None else 2 if immediate_input else 12) * 5e5)
      return None, None

    instance = SimpleNamespace(update=update, state_control=compute, publish=lambda *a: None,
      get_params_sp=lambda *a: None, run_ext=lambda *a: None, pm=None,
      sm=SimpleNamespace(updated={"selfdriveState": False}, logMonoTime={"selfdriveState": 0}))
    def sleep(seconds):
      clock[0] += round(seconds * 1e9 + wake_delay_ms * 1e6)

    def progress(*args):
      reports.append(args)
      clock[0] += round(reporting_ms * 1e6)

    with patch("time.monotonic_ns", side_effect=lambda: clock[0]), \
         patch("time.thread_time_ns", side_effect=lambda: cpu_clock[0]), \
         patch("time.sleep", side_effect=sleep), self.assertRaises(StopLoop):
      namespace["run"](instance, progress=progress, android=android,
        wait_until=lambda deadline: sleep((deadline - clock[0]) / 1e9))
    return waits, reports, computes

  def test_android_reads_latest_inputs_without_blocking_and_counts_overruns(self):
    waits, reports, computes = self.exercise(True)
    self.assertEqual(waits, [0] * 4)
    self.assertEqual(computes, [0, 12_000_000, 24_000_000])
    self.assertEqual(reports[1][2]["skipped_slots"], 0)
    self.assertEqual(reports[1][2]["scheduler_delay_ms"], 2.)
    self.assertGreater(reports[1][2]["release_to_completion_before_reporting_ms"], 10)
    self.assertGreater(reports[0][2]["cycle_before_reporting_ms"], 10)
    self.assertFalse(reports[0][0])  # missing input remains visible to supervision

  def test_other_platforms_keep_the_existing_poll_timeout(self):
    waits, _, _ = self.exercise(False)
    self.assertEqual(waits, [15] * 4)

  def test_immediate_input_cannot_run_controls_faster_than_100_hz(self):
    _, reports, computes = self.exercise(True, immediate_input=True)
    self.assertEqual(computes, [0, 10_000_000, 20_000_000])
    self.assertEqual(reports[1][2]["pacing_wait_ms"], 8.)
    self.assertEqual(reports[1][2]["cycle_before_reporting_ms"], 10.)
    self.assertEqual(reports[1][2]["selfdrive_ipc_age_ms"], -1.)
    self.assertEqual(reports[1][2]["active_cycle_before_reporting_ms"], 2.)
    self.assertEqual(reports[1][2]["release_to_completion_before_reporting_ms"], 2.)

  def test_late_wakeup_remains_visible_without_a_second_pacing_wait(self):
    waits, reports, computes = self.exercise(True, immediate_input=True, wake_delay_ms=3)
    self.assertEqual(waits, [0] * 4)
    self.assertEqual(computes, [0, 13_000_000, 23_000_000])
    self.assertEqual(reports[1][2]["scheduler_delay_ms"], 3.)
    self.assertEqual(reports[1][2]["release_to_completion_before_reporting_ms"], 5.)

  def test_cpu_diagnostics_do_not_subtract_waits_from_wall_deadlines(self):
    _, reports, _ = self.exercise(True, immediate_input=True, work_ms=12)
    metrics = reports[0][2]
    self.assertEqual(metrics["active_thread_cpu_ms"], 6.)
    self.assertEqual(metrics["active_off_cpu_ms"], 6.)
    self.assertEqual(metrics["input_thread_cpu_ms"], 0.)
    self.assertEqual(metrics["active_cycle_before_reporting_ms"], 12.)
    self.assertEqual(metrics["release_to_completion_before_reporting_ms"], 12.)

  def test_reporting_overrun_cannot_create_catchup_bursts(self):
    _, reports, computes = self.exercise(True, immediate_input=True, reporting_ms=13)
    self.assertEqual(computes, [0, 15_000_000, 30_000_000])
    self.assertEqual(reports[1][2]["skipped_slots"], 0)
    self.assertEqual(reports[2][2]["skipped_slots"], 1)
    self.assertEqual(reports[1][2]["scheduler_delay_ms"], 5.)

  def test_long_work_skips_expired_slots_once_without_waiting_another_slot(self):
    _, reports, computes = self.exercise(True, work_ms=35)
    self.assertEqual(computes, [0, 35_000_000, 70_000_000])
    self.assertEqual([r[2]["skipped_slots"] for r in reports], [0, 2, 3])
    self.assertEqual([r[2]["scheduler_delay_ms"] for r in reports], [0., 5., 0.])

  def test_wakeup_crossing_multiple_slots_counts_each_expired_release(self):
    _, reports, computes = self.exercise(True, immediate_input=True, wake_delay_ms=25)
    self.assertEqual(computes, [0, 35_000_000, 65_000_000])
    self.assertEqual(reports[1][2]["skipped_slots"], 2)
    self.assertEqual(reports[1][2]["scheduler_delay_ms"], 5.)
    self.assertEqual(reports[1][2]["release_to_completion_before_reporting_ms"], 7.)

  def test_missing_input_does_not_change_the_control_cadence(self):
    waits, reports, computes = self.exercise(True, work_ms=2)
    self.assertEqual(waits, [0] * 4)
    self.assertEqual(computes, [0, 10_000_000, 20_000_000])
    self.assertTrue(all(not item[0] for item in reports))

  def test_other_platforms_do_not_add_deadline_pacing(self):
    _, reports, computes = self.exercise(False, immediate_input=True)
    self.assertEqual(computes, [0, 2_000_000, 4_000_000])
    self.assertTrue(all(report[2]["pacing_wait_ms"] == 0 for report in reports))


if __name__ == "__main__":
  unittest.main()
