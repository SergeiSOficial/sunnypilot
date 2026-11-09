"""Compile the original ONNX models for Android OpenCL and compare CPU references."""
import hashlib
import json
from pathlib import Path
import time


def trace_profile(profile):
  parts = profile.split(":")
  if len(parts) == 1:
    return profile, 0, None
  if (parts[0] not in ("trace-mnn-image", "trace-mnn-buffer", "trace-mnn-road-buffer") or len(parts) not in (2, 3, 4)
      or not all(value.isdecimal() for value in parts[1:]) or not 0 <= int(parts[1]) <= 4096):
    raise ValueError("Invalid MNN diagnostic trace window")
  case = None
  if len(parts) >= 3:
    seed = int(parts[2])
    frame = int(parts[3]) if len(parts) == 4 else None
    if seed not in (42, 7, 123) or frame is not None and not 0 <= frame < 32:
      raise ValueError("Invalid MNN diagnostic seed/frame")
    case = {"seed": seed, "frame": frame}
  return parts[0], int(parts[1]), case


def accel_profile(profile):
  """One complete recurrent sequence per isolated accuracy experiment."""
  if not profile.startswith("mnn-buffer-"):
    return None
  parts = profile.removeprefix("mnn-buffer-").split("-", 2)
  if len(parts) < 2 or parts[0] not in ("accuracy", "timing") or parts[1] not in ("driver", "road"):
    return None
  part = parts[2] if len(parts) == 3 else None
  if part is not None and (parts[0] != "accuracy" or part not in ("initial", "seed42", "seed7", "seed123", "camera-replay")):
    return None
  return parts[0], "dmonitoring_model" if parts[1] == "driver" else "driving_supercombo", part


def benchmark(data_dir, profile="baseline", library_dir=None):
  profile, trace_skip, trace_case = trace_profile(profile)
  staged = accel_profile(profile)
  dispatch_profiles = {f"ncnn-dispatch-{limit}": (limit, False) for limit in (131072, 262144, 524288)}
  dispatch_profiles.update({f"profile-ncnn-dispatch-{limit}": (limit, True) for limit in (131072, 262144, 524288)})
  dispatch = dispatch_profiles.get(profile)
  if profile not in ("baseline", "image-fp32", "beam1", "fp32", "fp32-image", "ort-cpu", "ort-cpu-4", "ort-nnapi",
                     "ort-cpu-4-spin", "ort-cpu-strict", "ort-fallback", "ort-fp32", "ort-host-fusion", "host-fusion-cl",
                     "host-fusion-cl-road", "host-fusion-cl-driver",
                     "host-fusion-cl-mad-road", "host-fusion-cl-mad-driver",
                     "trace", "trace-cpu", "trace-accum", "trace-tail", "trace-fused", "trace-kernels", "trace-camera", "trace-road-camera", "trace-road-seed7",
                     "mixed-accum", "qnn-gpu", "qnn-htp", "qnn-htp-probes", "accel-probes", "mnn-opencl", "mnn-opencl-buffer",
                     "mnn-buffer-accuracy-driver", "mnn-buffer-accuracy-road", "mnn-buffer-timing-driver", "mnn-buffer-timing-road",
                     "trace-mnn-image", "trace-mnn-buffer", "trace-mnn-road-buffer", "ncnn-vulkan", "profile-ncnn-vulkan", "trace-qnn", "profile-cl-road", "profile-cl-driver") and staged is None and dispatch is None:
    raise ValueError("Unknown model profile")
  from fold3_runtime import configure
  configure(data_dir)
  # Chaquopy extracts package data when that package is imported. Import only
  # the model assets, without starting the unrelated dependency/GPU probe.
  import openpilot.selfdrive.modeld.models  # noqa: F401
  if dispatch is not None:
    limit, profile_gpu = dispatch
    if profile_gpu:
      from fold3_gpu_profile import run
      return run(data_dir, library_dir, road_dispatch_limit=limit)
    from fold3_accel import benchmark as accel_benchmark
    return accel_benchmark(data_dir, "ncnn-vulkan", library_dir, road_dispatch_limit=limit)
  if staged is not None:
    from fold3_accel import benchmark as accel_benchmark
    stage, model, part = staged
    return accel_benchmark(data_dir, "mnn-opencl", library_dir, "buffer", stage=stage,
                           selected_model=model, accuracy_part=part)
  if profile == "qnn-htp-probes":
    from fold3_accel import htp_probes
    return htp_probes(data_dir, library_dir)
  if profile == "profile-ncnn-vulkan":
    from fold3_gpu_profile import run
    return run(data_dir, library_dir)
  if profile in ("mnn-opencl", "ncnn-vulkan"):
    from fold3_accel import benchmark as accel_benchmark
    return accel_benchmark(data_dir, profile, library_dir)
  if profile in ("mnn-opencl-buffer", "trace-mnn-image", "trace-mnn-buffer", "trace-mnn-road-buffer"):
    from fold3_accel import benchmark as accel_benchmark
    return accel_benchmark(data_dir, "mnn-opencl", library_dir,
                           "buffer" if profile.endswith("buffer") else "image", 32 if profile.startswith("trace-") else 0,
                           trace_skip, trace_case, selected_model="driving_supercombo" if profile == "trace-mnn-road-buffer" else None)
  if profile == "accel-probes":
    from fold3_accel import gpu_probes
    return gpu_probes(data_dir)
  if profile in ("profile-cl-road", "profile-cl-driver"):
    from fold3_cl_profile import profile as cl_profile
    return cl_profile(data_dir, "driving_supercombo" if profile.endswith("road") else "dmonitoring_model")
  if profile == "trace-qnn":
    from fold3_model_trace import trace
    return trace(data_dir, opencl=False, qnn=True, library_dir=library_dir)
  if profile == "trace-kernels":
    from fold3_cl_trace import trace_kernels
    return trace_kernels(data_dir)
  if profile in ("trace-camera", "trace-road-camera", "trace-road-seed7"):
    from fold3_model_trace import trace_camera
    return trace_camera(data_dir, road=profile.startswith("trace-road"), recurrent=profile == "trace-road-seed7")
  if profile in ("trace", "trace-cpu", "trace-accum", "trace-tail", "trace-fused"):
    from fold3_model_trace import trace
    return trace(data_dir, opencl=profile not in ("trace-cpu", "trace-tail"), accumulation=profile == "trace-accum",
                 directory=profile if profile in ("trace-tail", "trace-fused") else "trace")
  if profile in ("qnn-gpu", "qnn-htp"):
    return _benchmark_ort(data_dir, profile, library_dir=library_dir)
  if profile.startswith("ort-"):
    if profile == "ort-host-fusion":
      return _benchmark_ort(data_dir, "cpu", threads=2, strict=True, host_fusion=True)
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
  from tinygrad import Tensor, Device
  from fold3_opencl import OpenCLModel, identity, qualified
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
            "vehicle_tx_enabled": False, "models": {}}
  manifest = json.loads((references / "manifest.json").read_text())
  report["host_reference"] = manifest
  selected = {"host-fusion-cl-road": "driving_supercombo", "host-fusion-cl-driver": "dmonitoring_model",
              "host-fusion-cl-mad-road": "driving_supercombo", "host-fusion-cl-mad-driver": "dmonitoring_model"}.get(profile)
  diagnostic_mad = profile.startswith("host-fusion-cl-mad-")
  report["arithmetic"] = "native mad convolution/activation diagnostic" if diagnostic_mad else "reference ordered fma"
  if selected:
    report.update(diagnostic_only=True, selected_model=selected,
                  scope="Single model only; cannot write live backend qualification")
  destination.write_text(json.dumps(report, indent=2))
  for name, metadata in manifest["models"].items():
    if selected and name != selected:
      continue
    entry = report["models"][name] = {"status": "running"}
    destination.write_text(json.dumps(report, indent=2))
    try:
      progress.write_text(f"Compiling {name} for the phone OpenCL driver")
      fused = profile.startswith("host-fusion-cl")
      path = references / f"host-fusion/{name}.onnx" if fused else models / f"{name}.onnx"
      expected_hash = metadata["sha256"]
      if fused:
        compiled = json.loads((references / "host-fusion/manifest.json").read_text())["models"][name]
        if compiled["original_sha256"] != metadata["sha256"]:
          raise ValueError("Host fusion provenance changed")
        expected_hash = compiled["sha256"]
      with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != expected_hash:
          raise ValueError("model hash differs from host reference")
      start = time.monotonic_ns()
      # The pinned DISK backend imports optional POSIX shared-memory support which
      # Chaquopy omits. Read the small ONNX assets through the portable Python backend.
      model = OpenCLModel(path, host_fusion=fused, image_buffers=int(profile in ("image-fp32", "fp32-image")),
                          beam=int(profile == "beam1"), diagnostic_mad=diagnostic_mad)
      if profile == "mixed-accum":
        from fold3_precision import accumulation_diagnostic
        accumulation_diagnostic(model.runner)
        report["diagnostic_only"] = True
      with np.load(references / f"{name}-inputs.npz") as source:
        inputs = {key: Tensor(source[key], device="CL").realize() for key in source.files}
      run = model.run_tensors
      for warmup in range(3):
        _thermal_check()
        progress.write_text(f"{name}: warmup {warmup + 1}/3")
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
      destination.write_text(json.dumps(report, indent=2))
      from fold3_model_trace import compare
      cases = {}
      entry["additional_seed_parity"] = cases
      for seed in metadata.get("additional_validation_seeds", []):
        progress.write_text(f"{name}: original-reference seed {seed}")
        with np.load(references / f"{name}-case-{seed}-inputs.npz") as source:
          values = model(dict(source))
        with np.load(references / f"{name}-case-{seed}-outputs.npz") as expected:
          cases[str(seed)] = {key: compare(values[key], expected[key]) for key in expected.files}
        destination.write_text(json.dumps(report, indent=2))
      if not promoted:
        from fold3_sequences import validate
        progress.write_text(f"{name}: checking recurrent sequences and camera replay")
        entry["sequences"] = validate(model, references, name, _thermal_check)
        entry["identity"] = identity(name, path, references)
        destination.write_text(json.dumps(report, indent=2))
      work_ms = []
      for i in range(100):
        _thermal_check()
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
      entry["performance_status"] = "passed" if entry["work_p99_ms"] < 50 and entry["deadline_misses_50ms"] == 0 else "failed"
      if any(not value["passed"] for case in entry.get("additional_seed_parity", {}).values() for value in case.values()):
        entry["status"] = "failed"
      if any(not value["passed"] for value in entry.get("sequences", {}).values()):
        entry["status"] = "failed"
    except Exception as exc:
      import traceback
      entry.update({"status": "blocked", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    destination.write_text(json.dumps(report, indent=2))
  report["completed_utc"] = time.time()  # noqa: TID251 -- UTC evidence timestamp
  if profile in ("baseline", "host-fusion-cl") and len(report["models"]) == 2 and all(qualified(entry) for entry in report["models"].values()):
    (Path(data_dir) / "model-qualification.json").write_text(json.dumps(report, indent=2))
  destination.write_text(json.dumps(report, indent=2))
  progress.write_text("Model validation complete; model-report.json records parity and measured work times.")
  return json.dumps(report)


def _thermal_check():
  from java import jclass
  status = int(jclass("ai.sunnypilot.fold3.AndroidHardware").thermalStatus())
  if status >= 2:
    raise RuntimeError(f"Model qualification stopped at Android thermal status {status}")


def _benchmark_ort(data_dir, backend, threads=2, spinning=False, strict=False, fallback=False, promoted=False, host_fusion=False,
                   library_dir=None):
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
            "synthetic_only": promoted, "host_fusion": host_fusion,
            "host_reference": manifest, "models": {}}
  destination = Path(data_dir) / "model-report.json"
  progress = Path(data_dir) / "model-progress.txt"
  for name, metadata in manifest["models"].items():
    entry = report["models"][name] = {"status": "running"}
    destination.write_text(json.dumps(report, indent=2))
    model = None
    try:
      path = references / f"{name}.onnx" if promoted else assets / f"openpilot/selfdrive/modeld/models/{name}.onnx"
      if host_fusion:
        path = references / f"host-fusion/{name}.onnx"
      if not host_fusion and hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("model hash differs from host reference")
      if fallback:
        path = references / "ort-fallback" / f"{name}.onnx"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["ort_fallback"]["sha256"]:
          raise ValueError("compiled model hash differs from host reference")
      progress.write_text(f"Loading original {name} using ORT {backend}")
      started = time.monotonic_ns()
      if host_fusion:
        compiled = json.loads((references / "host-fusion/manifest.json").read_text())["models"][name]
        if compiled["original_sha256"] != metadata["sha256"] or hashlib.sha256(path.read_bytes()).hexdigest() != compiled["sha256"]:
          raise ValueError("Host fusion provenance changed")
      model = OrtModel(path, backend, threads=threads, spinning=spinning, strict=strict, library_dir=library_dir,
                       assignments=True, no_cpu_fallback=backend.startswith("qnn-") or backend == "nnapi")
      entry["assignment"] = model.describe()
      with np.load(references / f"{name}-inputs.npz") as source:
        inputs = {key: source[key] for key in source.files}
      if backend.startswith("qnn-") or backend == "nnapi":
        smoke = None
        try:
          _thermal_check()
          smoke = OrtModel(path, backend, threads=threads, spinning=spinning, strict=strict, library_dir=library_dir,
                           assignments=True, no_cpu_fallback=True)
          smoke(inputs)
          entry["no_cpu_fallback_smoke"] = {"status": "passed", "assignment": smoke.describe()}
        except Exception as exc:
          entry["no_cpu_fallback_smoke"] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
        finally:
          del smoke
        if backend.startswith("qnn-"):
          cpu_nodes = entry["assignment"]["providers"].get("CPUExecutionProvider", {}).get("nodes", 0)
          entry["cpu_fallback"] = ("present" if cpu_nodes else
                                   "not observed" if entry["no_cpu_fallback_smoke"]["status"] == "passed" else "unverified")
      if backend.startswith("qnn-") and not entry["assignment"]["providers"].get("QNNExecutionProvider", {}).get("nodes"):
        raise RuntimeError("QNN provider assigned no model nodes; accelerator performance was not measured")
      for _ in range(3):
        _thermal_check()
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
      initial_parity = (bool(parity) and set(cases) == {"7", "123"}
                        and all(p["passed"] for case in [parity, *cases.values()] for p in case.values()))
      if not promoted and initial_parity:
        from fold3_sequences import validate
        entry["sequences"] = validate(model, references, name, _thermal_check)
      elif not promoted:
        entry["sequence_status"] = "skipped: original-reference parity failed"
      cpu_started = time.process_time_ns()
      wall_started = time.monotonic_ns()
      for i in range(100):
        _thermal_check()
        started = time.monotonic_ns()
        model(inputs)
        samples.append((time.monotonic_ns() - started) / 1e6)
        if i % 10 == 0:
          progress.write_text(f"ORT {backend} {name}: {i}/100 samples")
        time.sleep(max(0, .05 - samples[-1] / 1000))
      passed = (initial_parity and set(entry.get("sequences", {})) == {"seed42", "seed7", "seed123", "camera-replay"}
                and all(value["passed"] for value in entry["sequences"].values())) if not promoted else initial_parity
      equivalent = all(p["passed"] for case in original_parity.values() for p in case.values())
      entry.update(status="passed" if passed and equivalent else "failed", parity=parity, additional_cases=cases,
                   backend_parity_status="passed" if passed else "failed", original_model_parity=original_parity,
                   samples=len(samples), work_p50_ms=float(np.percentile(samples, 50)), work_p95_ms=float(np.percentile(samples, 95)),
                   work_p99_ms=float(np.percentile(samples, 99)),
                   work_max_ms=max(samples), deadline_misses_50ms=sum(s >= 50 for s in samples),
                   process_cpu_ms=(time.process_time_ns() - cpu_started) / 1e6,
                   timed_wall_ms=(time.monotonic_ns() - wall_started) / 1e6)
      entry["performance_status"] = "passed" if entry["work_p99_ms"] < 50 and entry["deadline_misses_50ms"] == 0 else "failed"
    except Exception as exc:
      import traceback
      entry.update(status="blocked", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
      del model
    destination.write_text(json.dumps(report, indent=2))
  report["completed_utc"] = time.time()  # noqa: TID251
  destination.write_text(json.dumps(report, indent=2))
  progress.write_text("ORT validation complete; parity and timing are separate gates.")
  return json.dumps(report)
