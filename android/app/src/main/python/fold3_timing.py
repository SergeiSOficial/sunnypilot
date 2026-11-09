"""Android controls pacing against the native monotonic clock."""
import ctypes
import errno
import time


class Timespec(ctypes.Structure):
  _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


def make_wait_until(clock_nanosleep):
  """Bind once; CDLL calls release the GIL. EINTR retains the same deadline."""
  if ctypes.sizeof(ctypes.c_long) != 8:
    raise RuntimeError("Android absolute pacing requires the supported ARM64 ABI")
  clock_nanosleep.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(Timespec), ctypes.c_void_p]
  clock_nanosleep.restype = ctypes.c_int

  def wait_until(deadline_ns):
    if not isinstance(deadline_ns, int) or not 0 <= deadline_ns < 2**63:
      raise ValueError("Invalid monotonic deadline")
    target = Timespec(*divmod(deadline_ns, 1_000_000_000))
    while True:
      result = clock_nanosleep(time.CLOCK_MONOTONIC, 1, ctypes.byref(target), None)  # TIMER_ABSTIME
      if result == 0:
        return
      if result != errno.EINTR:
        raise OSError(result, "clock_nanosleep failed")
  return wait_until


def android_wait_until():
  # Bind lazily so host tools and other platforms do not require Android libc.
  return make_wait_until(ctypes.CDLL("libc.so").clock_nanosleep)
