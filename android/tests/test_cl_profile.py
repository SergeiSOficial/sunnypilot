import ctypes
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app/src/main/python"))
from fold3_cl_profile import events


class ProfileTest(unittest.TestCase):
  def fixture(self):
    calls = []
    cl = SimpleNamespace(cl_event=ctypes.c_void_p, CL_PROFILING_COMMAND_QUEUED=1, CL_PROFILING_COMMAND_SUBMIT=2,
                         CL_PROFILING_COMMAND_START=3, CL_PROFILING_COMMAND_END=4)

    def enqueue(*args):
      calls.append("enqueue")
      if args[-1] is not None:
        args[-1]._obj.value = len(calls)
      return 0

    for name in ("clEnqueueNDRangeKernel", "clEnqueueWriteBuffer", "clEnqueueReadBuffer"):
      setattr(cl, name, enqueue)

    def stamp(event, key, size, target, unused):
      target._obj.value = event.value * 100 + key
      return 0

    cl.clGetEventProfilingInfo = stamp
    cl.clFinish = lambda queue: calls.append("finish") or 0
    cl.clReleaseEvent = lambda event: calls.append("release") or 0

    class Program:
      def __init__(self, device, obj):
        pass

      def __call__(self):
        return cl.clEnqueueNDRangeKernel(1, 2, 3, None, None, None, 0, None, None)

    return cl, Program, calls, enqueue

  def test_submission_is_asynchronous_and_events_are_released(self):
    cl, program, calls, original = self.fixture()
    with events(cl, program) as report:
      kernel = program(None, SimpleNamespace(name="convolution", lib=b"kernel"))
      kernel()
      kernel()
      self.assertEqual(calls, ["enqueue", "enqueue"])
    self.assertEqual(calls, ["enqueue", "enqueue", "finish", "release", "release"])
    self.assertIs(cl.clEnqueueNDRangeKernel, original)
    self.assertEqual(report["totals"]["convolution"]["launches"], 2)
    self.assertFalse(report["serialized_checkpoints"])

  def test_exception_restores_hooks_and_releases_events(self):
    cl, program, calls, original = self.fixture()
    with self.assertRaisesRegex(ValueError, "inference"):
      with events(cl, program):
        program(None, SimpleNamespace(name="kernel", lib=b"source"))()
        raise ValueError("inference")
    self.assertIs(cl.clEnqueueNDRangeKernel, original)
    self.assertEqual(calls.count("release"), 1)

  def test_bounded_profile_keeps_executing_without_claiming_complete_coverage(self):
    cl, program, calls, _ = self.fixture()
    with events(cl, program, limit=1) as report:
      kernel = program(None, SimpleNamespace(name="kernel", lib=b"source"))
      kernel()
      kernel()
    self.assertEqual(calls.count("enqueue"), 2)
    self.assertEqual(report["dropped"], 1)


if __name__ == "__main__":
  unittest.main()
