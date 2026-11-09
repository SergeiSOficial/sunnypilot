"""Absolute native wait error handling, without depending on host libc APIs."""
import ctypes
import errno
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'app/src/main/python/fold3_timing.py'
spec = importlib.util.spec_from_file_location('fold3_timing', SOURCE)
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class NativeWaitTest(unittest.TestCase):
  def bind(self, results):
    calls = []
    def native(clock, flags, target, remaining):
      target = ctypes.cast(target, ctypes.POINTER(timing.Timespec)).contents
      calls.append((clock, flags, target.tv_sec, target.tv_nsec, remaining))
      return results.pop(0)
    return timing.make_wait_until(native), calls

  def test_eintr_retries_identical_absolute_deadline(self):
    wait, calls = self.bind([errno.EINTR, errno.EINTR, 0])
    wait(12_345_678_901)
    self.assertEqual(len(calls), 3)
    self.assertEqual(calls, [calls[0]] * 3)
    self.assertEqual(calls[0][1:], (1, 12, 345_678_901, None))

  def test_failures_propagate_without_relative_sleep_fallback(self):
    wait, calls = self.bind([errno.EINVAL])
    with self.assertRaises(OSError) as error:
      wait(0)
    self.assertEqual(error.exception.errno, errno.EINVAL)
    self.assertEqual(len(calls), 1)

  def test_reject_invalid_deadlines_before_native_call(self):
    wait, calls = self.bind([])
    for deadline in (-1, 2**63, 1.5, None):
      with self.assertRaises(ValueError):
        wait(deadline)
    self.assertFalse(calls)
