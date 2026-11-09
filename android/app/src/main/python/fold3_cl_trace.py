"""Debug-only, serialized OpenCL checkpoints for a driver stall.

Each launch is persisted before waiting on its event. This path cannot qualify
live inference, and its serialized durations are not performance measurements.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time


@contextmanager
def checkpoints(directory):
  from tinygrad.runtime.ops_cl import CLProgram

  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  (directory / "completed.jsonl").unlink(missing_ok=True)
  original_init, original_call = CLProgram.__init__, CLProgram.__call__
  sequence = 0

  def initialize(self, device, obj):
    self._fold3_source = obj.lib
    self._fold3_name = obj.name
    original_init(self, device, obj)

  def call(self, *buffers, **kwargs):
    from fold3_models import _thermal_check
    nonlocal sequence
    _thermal_check()
    sequence += 1
    source = self._fold3_source
    digest = hashlib.sha256(source).hexdigest()
    path = directory / f"{digest}.cl"
    if not path.exists():
      path.write_bytes(source)
    entry = {"sequence": sequence, "kernel": self._fold3_name, "source_sha256": digest,
             "global_size": list(kwargs.get("global_size", (1, 1, 1))),
             "local_size": list(kwargs.get("local_size", (1, 1, 1))), "status": "waiting",
             "started_monotonic_ns": time.monotonic_ns()}
    latest = directory / "latest.json"
    latest.write_text(json.dumps(entry, indent=2))
    kwargs["wait"] = True
    result = original_call(self, *buffers, **kwargs)
    entry.update(status="completed", serialized_event_ms=None if result is None else result * 1000,
                 completed_monotonic_ns=time.monotonic_ns())
    latest.write_text(json.dumps(entry, indent=2))
    with (directory / "completed.jsonl").open("a") as log:
      log.write(json.dumps(entry) + "\n")
    return result

  CLProgram.__init__, CLProgram.__call__ = initialize, call
  try:
    yield
  finally:
    CLProgram.__init__, CLProgram.__call__ = original_init, original_call


def trace_kernels(data_dir):
  import numpy as np
  from tinygrad.helpers import Context
  from fold3_model_trace import compare
  from fold3_models import _thermal_check
  from fold3_opencl import OpenCLModel, identity

  references = Path(__file__).parent / "fold3_reference"
  manifest = json.loads((references / "manifest.json").read_text())
  fused = json.loads((references / "host-fusion/manifest.json").read_text())["models"]
  destination = Path(data_dir) / "model-report.json"
  report = {"profile": "trace-kernels", "backend": "CL", "diagnostic_only": True,
            "started_utc": time.time(), "vehicle_tx_enabled": False, "models": {}}
  for name, metadata in manifest["models"].items():
    entry = report["models"][name] = {"status": "running", "performance_status": "blocked"}
    destination.write_text(json.dumps(report, indent=2))
    try:
      path = references / f"host-fusion/{name}.onnx"
      if (fused[name]["original_sha256"] != metadata["sha256"] or
          hashlib.sha256(path.read_bytes()).hexdigest() != fused[name]["sha256"]):
        raise ValueError("Host fusion provenance changed")
      _thermal_check()
      entry["identity"] = identity(name, path, references)
      destination.write_text(json.dumps(report, indent=2))
      with Context(IMAGE=0, FLOAT16=0, BEAM=0), checkpoints(Path(data_dir) / "kernel-trace" / name):
        model = OpenCLModel(path, host_fusion=True)
        with np.load(references / f"{name}-inputs.npz") as source:
          outputs = model(dict(source))
      with np.load(references / f"{name}-outputs.npz") as expected:
        entry["parity"] = {key: compare(outputs[key], expected[key]) for key in expected.files}
      entry.update(status="passed" if all(v["passed"] for v in entry["parity"].values()) else "failed",
                   identity=identity(name, path, references))
    except Exception as exc:
      import traceback
      entry.update(status="blocked", error=str(exc), traceback=traceback.format_exc())
    destination.write_text(json.dumps(report, indent=2))
  report["completed_utc"] = time.time()
  destination.write_text(json.dumps(report, indent=2))
  return json.dumps(report)
