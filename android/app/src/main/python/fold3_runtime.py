"""Report real Android import/backend failures without substituting fake daemons."""
import importlib
import json
import os
import platform
import time


def configure(data_dir, simulation=False):
  """Set sandbox paths before importing any upstream module with global state."""
  os.chdir(data_dir)
  os.environ["ANDROID_DATA_ROOT"] = data_dir
  os.environ["PARAMS_ROOT"] = os.path.join(data_dir, "sim-params" if simulation else "params")
  os.environ["ANDROID_PARAMS_MEMORY_ROOT"] = os.path.join(data_dir, "sim-memory-params" if simulation else "memory-params")
  os.environ["LOG_ROOT"] = os.path.join(data_dir, "logs")
  os.environ["COMMA_CACHE"] = os.path.join(data_dir, "cache")
  os.environ["XDG_CACHE_HOME"] = os.path.join(data_dir, "cache")
  os.environ["OPENCL_PATH"] = "/vendor/lib64/libOpenCL.so"
  os.environ["CL"] = "1"
  os.environ["DEV"] = "CL"
  # Android has no desktop monitor to probe before NativeActivity attaches.
  os.environ["SCALE"] = "1"
  os.environ["BIG"] = "1"
  # The extended Linux settings depend on NetworkManager/system update tools.
  # Reuse the upstream landscape UI with Android-owned settings instead.
  os.environ["SUNNYPILOT_UI"] = "0"
  # Android services provide process isolation. Python subprocess workers require
  # POSIX named semaphores, which Android does not implement.
  os.environ["PARALLEL"] = "0"
  if simulation:
    os.environ["SIMULATION"] = "1"
  else:
    os.environ.pop("SIMULATION", None)
  ipc_dir = os.path.join(data_dir, "sim-ipc" if simulation else "ipc")
  os.makedirs(ipc_dir, mode=0o700, exist_ok=True)
  os.chmod(ipc_dir, 0o700)
  os.environ["MSGQ_ANDROID_PATH"] = ipc_dir
  return ipc_dir


def probe(data_dir):
  configure(data_dir)
  result = {"started_utc": time.time(), "pid": os.getpid(), "python": platform.python_version(),  # noqa: TID251 -- UTC evidence timestamp
            "machine": platform.machine(), "vehicle_tx_enabled": False, "modules": {}}
  report_path = os.path.join(data_dir, "runtime.json")
  modules = ("numpy", "tinygrad", "capnp", "msgq.ipc_pyx", "msgq.visionipc", "openpilot.cereal", "openpilot.cereal.messaging",
             "opendbc.car.hyundai.values", "openpilot.common.params", "pyray", "zmq", "zstandard", "setproctitle",
             "openpilot.selfdrive.controls.controlsd", "openpilot.selfdrive.monitoring.dmonitoringd", "openpilot.selfdrive.ui.ui")
  for name in modules:
    result["modules"][name] = {"status": "running"}
    with open(report_path, "w") as report:
      json.dump(result, report, indent=2)
    start = time.monotonic_ns()
    try:
      module = importlib.import_module(name)
      result["modules"][name] = {"status": "passed", "version": getattr(module, "__version__", None),
                                "file": getattr(module, "__file__", None), "import_ms": (time.monotonic_ns() - start) / 1e6}
    except Exception as exc:
      import traceback
      result["modules"][name] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
  from fold3_driving import MODULES
  for module in MODULES.values():
    name = "openpilot.selfdrive." + module
    try:
      importlib.import_module(name)
      result["modules"][name] = {"status": "passed"}
    except Exception as exc:
      import traceback
      result["modules"][name] = {"status": "blocked", "error": str(exc), "traceback": traceback.format_exc()}
  try:
    from fold3_driving import native_check
    result["native_driving"] = native_check()
  except Exception as exc:
    import traceback
    result["native_driving"] = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
  try:
    import msgq
    from openpilot.cereal import log
    endpoint = f"android_diag_{os.getpid()}"
    sender = msgq.pub_sock(endpoint)
    receiver = msgq.sub_sock(endpoint, timeout=1000)
    message = log.Event.new_message(valid=False, logMonoTime=time.monotonic_ns())
    message.init("carState")
    message.carState.vEgo = 8.25
    payload = message.to_bytes()
    sender.send(payload)
    received = receiver.receive()
    if received != payload:
      raise ValueError("msgq payload mismatch or timeout")
    with log.Event.from_bytes(received) as decoded:
      if decoded.valid or decoded.carState.vEgo != 8.25 or decoded.logMonoTime != message.logMonoTime:
        raise ValueError("Cereal message metadata mismatch")
    result["cereal_msgq_round_trip"] = {"status": "passed", "bytes": len(payload), "valid": False}
  except Exception as exc:
    result["cereal_msgq_round_trip"] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
  try:
    from tinygrad import Tensor
    a = Tensor([1., 2., 3.], device="CL")
    values = (a * a + 1).numpy().tolist()
    result["tinygrad_cl"] = {"status": "passed" if values == [2., 5., 10.] else "failed", "values": values}
  except Exception as exc:
    result["tinygrad_cl"] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
  result["driving_runtime_ready"] = False
  import numpy as np
  result["numpy_upstream_requirement"] = "passed" if int(np.__version__.split(".")[0]) >= 2 else "blocked: NumPy >=2.0 required"
  with open(report_path, "w") as report:
    json.dump(result, report, indent=2)
  return json.dumps(result, indent=2)


def receive_camera_test(data_dir, ready):
  """Verify Java's NV12 publisher against the actual Python VisionIPC client."""
  configure(data_dir)
  from msgq.visionipc import VisionIpcClient
  from openpilot.cereal.visionipc import VisionStreamType
  client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD, False)
  deadline = time.monotonic() + 10
  while not client.connect(False):
    if time.monotonic() >= deadline:
      raise TimeoutError("Camera publisher connection timed out")
    time.sleep(.01)
  ready.countDown()
  deadline = time.monotonic() + 4
  frames, previous = 0, -1
  while time.monotonic() < deadline and frames < 10:
    buffer = client.recv(100)
    if buffer is None:
      continue
    frame = client.frame_id
    if frame <= previous or client.valid or client.timestamp_sof != 1000000 + frame:
      raise ValueError("Camera frame order, timestamp or calibration validity changed")
    if int(buffer.data[0]) != frame % 128 or int(buffer.data[client.uv_offset]) != frame % 128:
      raise ValueError("Camera NV12 payload or UV offset mismatch")
    previous = frame
    frames += 1
  if frames < 10:
    raise TimeoutError("Insufficient Java/Python camera frames")
  return frames
