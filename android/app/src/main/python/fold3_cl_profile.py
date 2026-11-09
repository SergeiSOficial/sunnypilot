"""Bounded OpenCL events, read after submission; no per-kernel waits or I/O."""
from collections import defaultdict
from contextlib import contextmanager
import ctypes
import hashlib
import json
from pathlib import Path
import time


@contextmanager
def events(cl, program, limit=20000):
  """Leave normal enqueue ordering intact, restore hooks and release every event."""
  records, compiles, queues = [], [], {}
  state = {"kernel": None, "dropped": 0, "externally_owned_events": 0}
  originals = {key: getattr(cl, key) for key in ("clEnqueueNDRangeKernel", "clEnqueueWriteBuffer", "clEnqueueReadBuffer")}
  original_init, original_call = program.__init__, program.__call__

  def initialize(self, device, obj):
    started = time.monotonic_ns()
    original_init(self, device, obj)
    self._fold3_profile = {"kernel": obj.name, "source_sha256": hashlib.sha256(obj.lib).hexdigest()}
    compiles.append({**self._fold3_profile, "host_ms": (time.monotonic_ns() - started) / 1e6})

  def call(self, *args, **kwargs):
    previous = state["kernel"]
    state["kernel"] = getattr(self, "_fold3_profile", {"kernel": "cached-before-profile"})
    try:
      return original_call(self, *args, **kwargs)
    finally:
      state["kernel"] = previous

  def enqueue(name, kind):
    def submit(*args):
      if args[-1] is not None or len(records) >= limit:
        state["externally_owned_events" if args[-1] is not None else "dropped"] += 1
        return originals[name](*args)
      event = cl.cl_event()
      started = time.monotonic_ns()
      result = originals[name](*args[:-1], ctypes.byref(event))
      if result != 0:
        if event:
          cl.clReleaseEvent(event)
        return result
      queues[id(args[0])] = args[0]
      records.append({"event": event, "kind": kind, **(state["kernel"] or {}),
                      "bytes": int(args[4]) if kind != "kernel" else 0,
                      "host_enqueue_ms": (time.monotonic_ns() - started) / 1e6})
      return result
    return submit

  report = {"serialized_checkpoints": False, "events": [], "compilation": compiles}
  program.__init__, program.__call__ = initialize, call
  for name, kind in zip(originals, ("kernel", "copyin", "readback"), strict=True):
    setattr(cl, name, enqueue(name, kind))
  try:
    yield report
  finally:
    program.__init__, program.__call__ = original_init, original_call
    for name, original in originals.items():
      setattr(cl, name, original)
    try:
      # One drain after the measured calls, never a serialized kernel checkpoint.
      started = time.monotonic_ns()
      for queue in queues.values():
        if cl.clFinish(queue) != 0:
          raise RuntimeError("OpenCL profile drain failed")
      report["final_drain_ms"] = (time.monotonic_ns() - started) / 1e6
      for record in records:
        output = {key: value for key, value in record.items() if key != "event"}
        for key in ("QUEUED", "SUBMIT", "START", "END"):
          stamp = ctypes.c_uint64()
          if cl.clGetEventProfilingInfo(record["event"], getattr(cl, "CL_PROFILING_COMMAND_" + key), 8,
                                       ctypes.byref(stamp), None) != 0:
            raise RuntimeError("OpenCL profiling timestamp unavailable")
          output[key.lower() + "_ns"] = stamp.value
        if not output["queued_ns"] <= output["submit_ns"] <= output["start_ns"] <= output["end_ns"]:
          raise RuntimeError("OpenCL profiling timestamps out of order")
        output["execution_ms"] = (output["end_ns"] - output["start_ns"]) / 1e6
        output["queue_ms"] = (output["start_ns"] - output["queued_ns"]) / 1e6
        report["events"].append(output)
    finally:
      for record in records:
        cl.clReleaseEvent(record["event"])
      report.update(dropped=state["dropped"], externally_owned_events=state["externally_owned_events"])
  totals = defaultdict(lambda: {"launches": 0, "execution_ms": 0., "host_enqueue_ms": 0., "bytes": 0})
  for record in report["events"]:
    total = totals[record.get("kernel", record["kind"])]
    total["launches"] += 1
    for key in ("execution_ms", "host_enqueue_ms", "bytes"):
      total[key] += record[key]
  report["totals"] = dict(sorted(totals.items(), key=lambda item: -item[1]["execution_ms"]))


def profile(data_dir, name):
  import numpy as np
  from tinygrad.helpers import Context
  from tinygrad.runtime.ops_cl import cl, CLProgram
  from fold3_models import _thermal_check
  from fold3_model_trace import compare
  from fold3_opencl import OpenCLModel, identity
  references = Path(__file__).parent / "fold3_reference"
  destination = Path(data_dir) / "model-report.json"
  report = {"profile": "profile-cl", "diagnostic_only": True, "vehicle_tx_enabled": False,
            "started_utc": time.time(), "models": {}}  # noqa: TID251 -- UTC audit metadata
  entry = report["models"][name] = {"status": "running"}
  destination.write_text(json.dumps(report, indent=2))
  try:
    path = references / f"host-fusion/{name}.onnx"
    with np.load(references / f"{name}-inputs.npz") as source:
      inputs = dict(source)
    _thermal_check()
    with Context(IMAGE=0, FLOAT16=0, BEAM=0), events(cl, CLProgram) as cold:
      started = time.monotonic_ns()
      model = OpenCLModel(path, host_fusion=True)
      for _ in range(3):
        _thermal_check()
        actual = model(inputs)
      entry["cold_and_warmup_ms"] = (time.monotonic_ns() - started) / 1e6
    entry["cold_events"] = cold
    times = []
    with events(cl, CLProgram) as warm:
      for _ in range(5):
        _thermal_check()
        started = time.monotonic_ns()
        actual = model(inputs)
        times.append((time.monotonic_ns() - started) / 1e6)
    entry.update(warm_events=warm, profiled_end_to_end_ms=times, identity=identity(name, path, references))
    with np.load(references / f"{name}-outputs.npz") as expected:
      entry["parity"] = {key: compare(actual[key], expected[key]) for key in expected.files}
    entry["status"] = "passed" if all(v["passed"] for v in entry["parity"].values()) and not warm["dropped"] else "failed"
  except Exception as exc:
    import traceback
    entry.update(status="blocked", error=str(exc), traceback=traceback.format_exc())
  report["completed_utc"] = time.time()  # noqa: TID251 -- UTC audit metadata
  destination.write_text(json.dumps(report, indent=2))
  return json.dumps(report)
