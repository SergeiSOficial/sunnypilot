import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_instrumentation import completed_tests


class InstrumentationTest(unittest.TestCase):
  def test_aborted_or_incomplete_suite_cannot_pass_from_exit_code_alone(self):
    for output in ("", "Time: 1.0\n", "FAILURES!!!\nTests run: 17, Failures: 1", "INSTRUMENTATION_ABORTED: crashed",
                   "OK (17 tests)\nINSTRUMENTATION_FAILED", "shortMsg=crashed\nOK (17 tests)"):
      self.assertEqual(completed_tests(output), 0)
    self.assertEqual(completed_tests("Time: 23.3\n\nOK (17 tests)\n"), 17)
