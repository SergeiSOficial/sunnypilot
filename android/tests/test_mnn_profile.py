import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from profile_mnn import parse_events


class MnnProfileTest(unittest.TestCase):
  def test_excludes_audit_and_keeps_repeated_kernels(self):
    def run(i):
      return "\n".join([f"FOLD3_PROFILE_BEGIN {i}", "kernel time = 1250    us convolution",
                        "kernel time = 750    us convolution", "total kernel time = 2000 us, conv time = 2000 us",
                        f"FOLD3_PROFILE_END {i}", ""])
    parsed = parse_events("kernel time = 999999 us audit\n" + "".join(run(i) for i in range(3)))
    self.assertEqual([r["kernel_total_ms"] for r in parsed], [2, 2, 2])
    self.assertEqual(len(parsed[0]["kernels"]), 2)

  def test_missing_truncated_or_empty_runs_are_rejected(self):
    for log in ("", "FOLD3_PROFILE_BEGIN 0\n", "FOLD3_PROFILE_END 0\n",
                "FOLD3_PROFILE_BEGIN 0\nFOLD3_PROFILE_END 0\n",
                "FOLD3_PROFILE_BEGIN 0\nFOLD3_PROFILE_BEGIN 1\n"):
      with self.subTest(log=log), self.assertRaises(ValueError):
        parse_events(log)

  def test_mismatched_run_is_rejected(self):
    with self.assertRaises(ValueError):
      parse_events("FOLD3_PROFILE_BEGIN 0\nkernel time = 1000 us conv\nFOLD3_PROFILE_END 1\n")

  def test_changed_or_lost_event_coverage_is_rejected(self):
    for last in ("conv", "conv\nkernel time = 100 us cast", "cast"):
      log = "\n".join(f"FOLD3_PROFILE_BEGIN {i}\nkernel time = 1000 us {last if i == 2 else 'conv'}\n"
                       + f"total kernel time = {1100 if i == 2 and '100 us' in last else 1000} us, conv time = 0 us\nFOLD3_PROFILE_END {i}"
                       for i in range(3))
      if last == "conv":
        self.assertEqual(len(parse_events(log)), 3)
      else:
        with self.subTest(last=last), self.assertRaises(ValueError):
          parse_events(log)

  def test_missing_or_inconsistent_native_totals_are_rejected(self):
    for total in ("", "total kernel time = 1001 us, conv time = 0 us\n",
                  "total kernel time = 1000 us, conv time = 0 us\n" * 2):
      log = "".join(f"FOLD3_PROFILE_BEGIN {i}\nkernel time = 1000 us conv\n{total}FOLD3_PROFILE_END {i}\n"
                    for i in range(3))
      with self.subTest(total=total), self.assertRaises(ValueError):
        parse_events(log)


if __name__ == "__main__":
  unittest.main()
