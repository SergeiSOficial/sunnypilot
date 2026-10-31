"""Compile the original ONNX models for Android OpenCL and compare CPU references."""
import hashlib
import json
from pathlib import Path
import time


def benchmark(data_dir, profile="baseline"):
  if profile not in ("baseline", "image-fp32", "beam1", "fp32", "fp32-image", "ort-cpu", "ort-cpu-4", "ort-nnapi",
                     "ort-cpu-4-spin", "ort-cpu-strict", "ort-fallback", "ort-fp32"):
    raise ValueError("Unknown model profile")
  from fold3_runtime import configure
  configure(data_dir)
  # Chaquopy extracts package data when that package is imported. Import only
  # the model assets, without starting the unrelated dependency/GPU probe.
  import openpilot.selfdrive.modeld.models  # noqa: F401
  if profile.startswith("ort-"):
    if profile == "ort-fp32":
      return _benchmark_ort(data_dir, "cpu", threads=2, promoted=True)
    if profile == "ort-fallback":
      return _benchmark_ort(data_dir, "cpu", threads=4, strict=True, fallback=True)
    if profile == "ort-cpu-4-spin":
      return _benchmark_ort(data_dir, "cpu", threads=4, spinning=True)
    if profile == "ort-cpu-strict":
      return _benchmark_ort(data_dir, "cpu", threads=4, strict=True)
    if profile == "ort-cpu-4":
      return _benchmark_ort(data_dir, "cpu", threads=4)
    return _benchmark_ort(data_dir, profile.removeprefix("ort-"))
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


def _benchmark_ort(data_dir, backend, threads=2, spinning=False, strict=False, fallback=False, promoted=False):
  import numpy as np
  from fold3_ort import OrtModel
  assets = Path(__file__).parent
  references = assets / "fold3_reference"
  original_references = references
  if promoted:
    references /= "fp32"
  manifest = json.loads((references / "manifest.json").read_text())
  report = {"backend": f"ONNX Runtime 1.31.1 {backend}", "precision": "FP32 diagnostic promotion" if promoted else "original ONNX types",
            "started_utc": time.time(), "vehicle_tx_enabled": False, "threads": threads,  # noqa: TID251
            "spinning": spinning, "graph_optimizations": "disabled" if strict else "default",
            "compiled_host_fallback": fallback,
            "optional_sme_sve_kernels": False,
            "synthetic_only": True, "replay_validation": "blocked: synthetic cases only",
            "host_reference": manifest, "models": {}}
  destination = Path(data_dir) / "model-report.json"
  progress = Path(data_dir) / "model-progress.txt"
  for name, metadata in manifest["models"].items():
    entry = report["models"][name] = {"status": "running"}
    destination.write_text(json.dumps(report, indent=2))
    try:
      path = references / f"{name}.onnx" if promoted else assets / f"openpilot/selfdrive/modeld/models/{name}.onnx"
      if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("model hash differs from host reference")
      if fallback:
        path = references / "ort-fallback" / f"{name}.onnx"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["ort_fallback"]["sha256"]:
          raise ValueError("compiled model hash differs from host reference")
      progress.write_text(f"Loading original {name} using ORT {backend}")
      started = time.monotonic_ns()
      model = OrtModel(path, backend, threads=threads, spinning=spinning, strict=strict)
      with np.load(references / f"{name}-inputs.npz") as source:
        inputs = {key: source[key] for key in source.files}
      for _ in range(3):
        outputs = model(inputs)
      entry["initialization_ms"] = (time.monotonic_ns() - started) / 1e6
      parity = {}
      with np.load(references / f"{name}-outputs.npz") as expected:
        for key in expected.files:
          actual = outputs[key]
          error = np.abs(actual.astype(np.float64) - expected[key].astype(np.float64))
          parity[key] = {"passed": bool(np.isfinite(actual).all() and np.allclose(actual, expected[key], rtol=1e-3, atol=1e-4)),
                         "max_abs_error": float(error.max()) if np.isfinite(error).all() else None}
      cases = {}
      for seed in metadata.get("additional_validation_seeds", []):
        with np.load(references / f"{name}-case-{seed}-inputs.npz") as source:
          case_actual = model({key: source[key] for key in source.files})
        with np.load(references / f"{name}-case-{seed}-outputs.npz") as expected:
          cases[str(seed)] = {key: {"passed": bool(np.isfinite(case_actual[key]).all()
            and np.allclose(case_actual[key], expected[key], rtol=1e-3, atol=1e-4)),
            "max_abs_error": float(np.max(np.abs(case_actual[key].astype(np.float64) - expected[key])))
            if np.isfinite(case_actual[key]).all() else None} for key in expected.files}
      original_parity = {}
      if promoted:
        original_model = assets / f"openpilot/selfdrive/modeld/models/{name}.onnx"
        if hashlib.sha256(original_model.read_bytes()).hexdigest() != metadata["original_sha256"]:
          raise ValueError("original model hash differs from promotion provenance")
        for suffix in ("", *(f"-case-{seed}" for seed in metadata.get("additional_validation_seeds", []))):
          with np.load(original_references / f"{name}{suffix}-inputs.npz") as source:
            candidate = model({key: source[key] for key in source.files})
          with np.load(original_references / f"{name}{suffix}-outputs.npz") as expected:
            original_parity[suffix or "seed42"] = {
              key: {"passed": bool(np.isfinite(candidate[key]).all() and np.allclose(candidate[key], expected[key], rtol=1e-3, atol=1e-4)),
                    "max_abs_error": float(np.max(np.abs(candidate[key].astype(np.float64) - expected[key].astype(np.float64))))
                    if np.isfinite(candidate[key]).all() else None} for key in expected.files}
      samples = []
      cpu_started = time.process_time_ns()
      wall_started = time.monotonic_ns()
      for i in range(100):
        started = time.monotonic_ns()
        model(inputs)
        samples.append((time.monotonic_ns() - started) / 1e6)
        if i % 10 == 0:
          progress.write_text(f"ORT {backend} {name}: {i}/100 samples")
        time.sleep(max(0, .05 - samples[-1] / 1000))
      passed = all(p["passed"] for case in [parity, *cases.values()] for p in case.values())
      equivalent = all(p["passed"] for case in original_parity.values() for p in case.values())
      entry.update(status="passed" if passed and equivalent else "failed", parity=parity, additional_cases=cases,
                   backend_parity_status="passed" if passed else "failed", original_model_parity=original_parity,
                   samples=len(samples), work_p50_ms=float(np.percentile(samples, 50)), work_p99_ms=float(np.percentile(samples, 99)),
                   work_max_ms=max(samples), deadline_misses_50ms=sum(s >= 50 for s in samples),
                   process_cpu_ms=(time.process_time_ns() - cpu_started) / 1e6,
                   timed_wall_ms=(time.monotonic_ns() - wall_started) / 1e6)
      entry["performance_status"] = "passed" if entry["work_p99_ms"] < 50 else "failed"
    except Exception as exc:
      import traceback
      entry.update(status="blocked", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    destination.write_text(json.dumps(report, indent=2))
  report["completed_utc"] = time.time()  # noqa: TID251
  destination.write_text(json.dumps(report, indent=2))
  progress.write_text("ORT validation complete; parity and timing are separate gates.")
  return json.dumps(report)
