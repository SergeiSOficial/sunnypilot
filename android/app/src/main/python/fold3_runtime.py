"""Report real Android import/backend failures without substituting fake daemons."""
import importlib
import json
import os
import platform
import time


def probe(data_dir):
  os.chdir(data_dir)
  os.environ["HOME"] = data_dir
  os.environ["XDG_CACHE_HOME"] = os.path.join(data_dir, "cache")
  os.environ["OPENCL_PATH"] = "/vendor/lib64/libOpenCL.so"
  os.environ["CL"] = "1"
  os.environ["DEV"] = "CL"
  # Android services provide process isolation. Python subprocess workers require
  # POSIX named semaphores, which Android does not implement.
  os.environ["PARALLEL"] = "0"
  ipc_dir = os.path.join(data_dir, "ipc")
  os.makedirs(ipc_dir, mode=0o700, exist_ok=True)
  os.chmod(ipc_dir, 0o700)
  os.environ["MSGQ_ANDROID_PATH"] = ipc_dir
  result = {"started_utc": time.time(), "pid": os.getpid(), "python": platform.python_version(),  # noqa: TID251 -- UTC evidence timestamp
            "machine": platform.machine(), "vehicle_tx_enabled": False, "modules": {}}
  report_path = os.path.join(data_dir, "runtime.json")
  modules = ("numpy", "tinygrad", "capnp", "msgq.ipc_pyx", "msgq.visionipc", "openpilot.cereal", "openpilot.cereal.messaging",
             "opendbc.car.hyundai.values", "openpilot.common.params", "pyray")
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
      result["modules"][name] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
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
  result["numpy_upstream_requirement"] = "blocked: diagnostic NumPy 1.26.2 does not satisfy upstream >=2.0"
  with open(report_path, "w") as report:
    json.dump(result, report, indent=2)
  return json.dumps(result, indent=2)
