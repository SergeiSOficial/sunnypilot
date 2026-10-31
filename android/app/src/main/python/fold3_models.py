"""Compile the original ONNX models for Android OpenCL and compare CPU references."""
import hashlib
import json
from pathlib import Path
import time


def benchmark(data_dir, profile="baseline"):
  if profile not in ("baseline", "image-fp32", "beam1", "fp32", "fp32-image"):
    raise ValueError("Unknown model profile")
  from fold3_runtime import probe
  probe(data_dir)
  from tinygrad.helpers import Context
  # Image-backed buffers are an independently measured layout experiment.
  # FLOAT16=0 leaves the original ONNX precision policy unchanged.
  with Context(IMAGE=int(profile in ("image-fp32", "fp32-image")), FLOAT16=0, BEAM=int(profile == "beam1")):
    return _benchmark(data_dir, profile)


def _benchmark(data_dir, profile):
  import numpy as np
  from tinygrad import Tensor, Device, TinyJit
  from tinygrad.nn.onnx import OnnxRunner
  models = Path(__file__).parent / "openpilot/selfdrive/modeld/models"
  references = Path(__file__).parent / "fold3_reference"
  promoted = profile in ("fp32", "fp32-image")
  if promoted:
    references /= "fp32"
    models = references
  destination = Path(data_dir) / "model-report.json"
  progress = Path(data_dir) / "model-progress.txt"
  report = {"backend": "CL", "precision": "FP16 promoted to FP32" if promoted else "original ONNX types",
            "profile": profile, "image_buffers": int(profile in ("image-fp32", "fp32-image")), "float16_override": 0,
            "beam_width": int(profile == "beam1"),
            "started_utc": time.time(), "parallel_compiler_workers": 0,  # noqa: TID251 -- UTC evidence timestamp, not a duration
            "vehicle_tx_enabled": False, "replay_validation": "blocked: synthetic cases only", "models": {}}
  manifest = json.loads((references / "manifest.json").read_text())
  report["host_reference"] = manifest
  destination.write_text(json.dumps(report, indent=2))
  for name, metadata in manifest["models"].items():
    entry = report["models"][name] = {"status": "running"}
    destination.write_text(json.dumps(report, indent=2))
    try:
      progress.write_text(f"Compiling {name} for the phone OpenCL driver")
      path = models / f"{name}.onnx"
      with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != metadata["sha256"]:
          raise ValueError("model hash differs from host reference")
      start = time.monotonic_ns()
      # The pinned DISK backend imports optional POSIX shared-memory support which
      # Chaquopy omits. Read the small ONNX assets through the portable Python backend.
      runner = OnnxRunner(Tensor(path.read_bytes(), device="PYTHON")).to("CL")
      with np.load(references / f"{name}-inputs.npz") as source:
        inputs = {key: Tensor(source[key], device="CL").realize() for key in source.files}
      @TinyJit
      def run(_runner=runner, **values):
        outputs = _runner(values)
        Tensor.realize(*outputs.values())
        return outputs
      for _ in range(3):
        outputs = run(**inputs)
        Device["CL"].synchronize()
      entry["compile_ms"] = (time.monotonic_ns() - start) / 1e6
      parity = {}
      with np.load(references / f"{name}-outputs.npz") as expected:
        for key in expected.files:
          actual = outputs[key].numpy()
          error = np.abs(actual.astype(np.float64) - expected[key].astype(np.float64))
          passed = bool(np.all(np.isfinite(actual)) and np.allclose(actual, expected[key], rtol=1e-3, atol=1e-4))
          parity[key] = {"passed": passed, "max_abs_error": float(error.max()) if np.all(np.isfinite(error)) else None,
                         "shape": list(actual.shape)}
      entry["parity"] = parity
      work_ms = []
      for i in range(100):
        started = time.monotonic_ns()
        run(**inputs)
        Device["CL"].synchronize()
        work_ms.append((time.monotonic_ns() - started) / 1e6)
        if i % 10 == 0:
          progress.write_text(f"{name}: {i}/100 inference samples")
        time.sleep(max(0, .05 - work_ms[-1] / 1000))
      entry.update({"status": "passed" if all(p["passed"] for p in parity.values()) else "failed",
                    "samples": len(work_ms), "work_p50_ms": float(np.percentile(work_ms, 50)),
                    "work_p99_ms": float(np.percentile(work_ms, 99)), "work_max_ms": max(work_ms),
                    "deadline_misses_50ms": sum(t >= 50 for t in work_ms)})
      entry["performance_status"] = "failed" if entry["work_p99_ms"] >= 50 else "passed"
    except Exception as exc:
      import traceback
      entry.update({"status": "blocked", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    destination.write_text(json.dumps(report, indent=2))
  report["completed_utc"] = time.time()  # noqa: TID251 -- UTC evidence timestamp
  destination.write_text(json.dumps(report, indent=2))
  progress.write_text("Model validation complete; model-report.json records parity and measured work times.")
  return json.dumps(report)
