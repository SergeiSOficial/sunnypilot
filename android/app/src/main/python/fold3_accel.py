"""Fail-closed accelerator diagnostics, separate from qualified live perception."""
import hashlib
import json
from pathlib import Path
import time

ACCURACY_PARTS = ("initial", "seed42", "seed7", "seed123", "camera-replay")
MODEL_OUTPUTS = {"dmonitoring_model": {"outputs"}, "driving_supercombo": {
    "outputs", "next_state_img_q", "next_state_desire_q", "next_state_feat_q"}}


class AccelModel:
  """Explicit diagnostic engine with unchanged external tensor contracts."""
  def __init__(self, path, backend, metadata, memory_mode="image", trace_limit=0, trace_skip=0, profiling=False, dispatch_limit=0):
    import fold3_accel_native as native
    self.native, self.metadata, self.backend = native, metadata, backend
    if type(dispatch_limit) is not int or dispatch_limit not in (0, 131072, 262144, 524288):
      raise ValueError("Invalid diagnostic dispatch limit")
    if dispatch_limit and (backend != "ncnn-vulkan" or Path(path).name != "driving_supercombo.ncnn.param" or
                           set(metadata.get("outputs", {})) != MODEL_OUTPUTS["driving_supercombo"]):
      raise ValueError("Diagnostic dispatch override requires the road Vulkan model")
    args = (str(path), backend, memory_mode, trace_limit, trace_skip, profiling)
    self.session = native.create(*args, dispatch_limit) if dispatch_limit else native.create(*args)
    self.info = native.describe(self.session)
    if dispatch_limit and self.info.get("diagnostic_dispatch_limit") != dispatch_limit:
      raise ValueError("Native dispatch configuration mismatch")
    if (backend == "ncnn-vulkan" and metadata.get("ncnn", {}).get("requires_explicit_half_rounding") and
        self.info.get("explicit_half_rounding") is not True):
      raise ValueError("Converted ncnn graph requires patched Vulkan half-rounding shaders")
    if (backend == "ncnn-vulkan" and metadata.get("ncnn", {}).get("requires_ordered_half_gemv") and
        self.info.get("ordered_half_gemv") is not True):
      raise ValueError("Converted ncnn graph requires ordered half GEMV shaders")
    if (backend == "ncnn-vulkan" and metadata.get("ncnn", {}).get("requires_ordered_half_dot2") and
        self.info.get("ordered_half_dot2") is not True):
      raise ValueError("Converted ncnn graph requires ordered half dot2 shaders")
    required_norm = metadata.get("ncnn", {}).get("requires_reference_layernorm", 0)
    if backend == "ncnn-vulkan" and required_norm and self.info.get("reference_layernorm") != required_norm:
      raise ValueError("Converted ncnn graph requires reference LayerNorm shaders")
    required_gemm = metadata.get("ncnn", {}).get("requires_reference_float_gemm", 0)
    if backend == "ncnn-vulkan" and required_gemm and self.info.get("reference_float_gemm") != required_gemm:
      raise ValueError("Converted ncnn graph requires reference FP32 Gemm shaders")
    if backend == "mnn-opencl":
      missing = set(metadata["inputs"]) - set(self.info["inputs"])
      if not missing <= set(metadata.get("unused_inputs", [])) or set(self.info["inputs"]) - set(metadata["inputs"]):
        raise ValueError("MNN changed reachable model inputs")

  def _engine_shape(self, key, contract, direction):
    shape = contract["shape"]
    engine = self.metadata["ncnn"].get(f"engine_{direction}", {}).get(key, contract)
    if engine == contract:
      return shape
    history_key = "state_img_q" if direction == "inputs" else "next_state_img_q"
    if (key != history_key or contract != {"shape": [2, 5, 6, 128, 256], "type": "tensor(uint8)"} or
        engine != {"shape": [2, 30, 128, 256], "type": "tensor(uint8)"}):
      raise ValueError(f"Unsupported ncnn tensor adapter: {key}")
    return engine["shape"]

  def describe(self):
    return self.native.describe(self.session)

  def __call__(self, inputs):
    import numpy as np
    if set(inputs) != set(self.metadata["inputs"]):
      raise ValueError("Accelerator external input set changed")
    buffers = {}
    for index, (key, contract) in enumerate(self.metadata["inputs"].items()):
      value = inputs[key]
      external_type = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(uint8)": np.uint8}[contract["type"]]
      if value.dtype != external_type:
        raise ValueError(f"Accelerator external input type changed: {key}")
      if list(value.shape) != contract["shape"]:
        raise ValueError(f"Accelerator input shape changed: {key}")
      if self.backend == "ncnn-vulkan":
        # ncnn's converter exposes floating boundaries. All uint8 input values
        # are exactly representable; model parity still checks internal casts.
        value = value.reshape(self._engine_shape(key, contract, "inputs"))
        axis = self.metadata["ncnn"]["input_batch_axes"][key]
        if axis is not None:
          value = np.moveaxis(value, axis, 0)
        if not 1 <= value.ndim - (axis is not None) <= 4:
          raise ValueError(f"ncnn cannot represent converted input rank/batch: {key}")
        buffers[f"in{index}"] = (np.ascontiguousarray(value, dtype=np.float32), axis is not None)
      else:
        if key not in self.info["inputs"] and key in self.metadata.get("unused_inputs", []):
          continue
        target = self.info["inputs"][key]
        if list(target["shape"]) != contract["shape"]:
          raise ValueError(f"MNN converted input shape changed: {key}")
        converted = np.ascontiguousarray(value, dtype=target["dtype"])
        if not np.array_equal(converted.astype(value.dtype), value):
          raise ValueError(f"Lossy MNN input adapter: {key}")
        buffers[key] = converted
    raw = self.native.run(self.session, buffers)
    output = {}
    for index, (key, contract) in enumerate(self.metadata["outputs"].items()):
      shape, dtype, data = raw[f"out{index}" if self.backend == "ncnn-vulkan" else key]
      engine_shape = self._engine_shape(key, contract, "outputs") if self.backend == "ncnn-vulkan" else contract["shape"]
      if shape is not None and list(shape) != engine_shape:
        raise ValueError(f"Accelerator output shape changed: {key}")
      value = np.frombuffer(data, dtype=dtype)
      if self.backend == "ncnn-vulkan":
        axis = self.metadata["ncnn"]["output_batch_axes"][key]
        dense_shape = list(engine_shape)
        if axis is not None:
          dense_shape.insert(0, dense_shape.pop(axis))
        value = value.reshape(dense_shape)
        if axis is not None:
          value = np.moveaxis(value, 0, axis)
        value = value.reshape(contract["shape"])
      else:
        value = value.reshape(contract["shape"])
      target = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(uint8)": np.uint8}[contract["type"]]
      if target == np.uint8 and (not np.isfinite(value).all() or not ((value >= 0) & (value <= 255) & (value == np.floor(value))).all()):
        raise ValueError("Non-integral/out-of-range image state from converted graph")
      output[key] = value.astype(target, copy=False)
    if len(raw) != len(output):
      raise ValueError("Accelerator output set changed")
    return output


def identity(name, path, references, backend, library_dir, memory_mode="image", dispatch_limit=0):
  from fold3_sequences import digest
  from fold3_opencl import code_digest
  from java import jclass
  paths = {"model": path, "original_model": Path(__file__).parent / f"openpilot/selfdrive/modeld/models/{name}.onnx",
           "references": references / "manifest.json", "sequences": references / "sequences/manifest.json",
           "conversion": references / "accelerators/manifest.json",
           "apk": Path(str(jclass("ai.sunnypilot.fold3.AndroidHardware").applicationApk()))}
  if backend == "mnn-opencl":
    paths.update(provider=Path(library_dir) / "libMNN.so", opencl_plugin=Path(library_dir) / "libMNN_CL.so",
                 driver=Path("/vendor/lib64/libOpenCL.so"))
  else:
    paths["weights"] = Path(str(path).removesuffix(".param") + ".bin")
    paths["driver"] = Path("/vendor/lib64/hw/vulkan.adreno.so")
    provider = Path(library_dir) / "libncnn.so"
    if provider.is_file():
      paths["provider"] = provider
    else:
      paths["shader_compiler"] = Path(library_dir) / "libglslang.so"
  return {"name": name, "backend": backend, "memory_mode": memory_mode, "cpu_fallback": False,
          **({"diagnostic_dispatch_limit": dispatch_limit} if dispatch_limit else {}),
          "fingerprint": str(jclass("android.os.Build").FINGERPRINT),
          "sha256": {**{key: digest(path) for key, path in paths.items()},
                     "implementation": code_digest(__name__), "native": code_digest("fold3_accel_native")}}


def qualified(entry, current_identity):
  # Reuse the complete existing numerical/timing gate, while binding the
  # diagnostic engine's actual APK, graph, code, provider and device identity.
  from fold3_opencl import qualified as reference_qualified
  if (not isinstance(entry, dict) or not isinstance(current_identity, dict) or
      entry.get("identity") != current_identity or current_identity.get("backend") not in ("mnn-opencl", "ncnn-vulkan")
      or not entry.get("assignment", {}).get("no_cpu_fallback_verified")
      or entry.get("assignment", {}).get("trace_limit", 0) != 0 or entry.get("assignment", {}).get("profiling_enabled", False)
      or entry.get("assignment", {}).get("diagnostic_dispatch_limit", 0) != 0
      or entry.get("diagnostic_only") is not False
      or (current_identity.get("backend") == "mnn-opencl" and
          (current_identity.get("memory_mode") not in ("image", "buffer") or
           entry.get("assignment", {}).get("memory_mode") != current_identity.get("memory_mode")))):
    return False
  return reference_qualified({**entry, "identity": {**current_identity, "backend": "CL"}})


def _accuracy_identity_passed(entry, current_identity):
  if (not isinstance(entry, dict) or not isinstance(current_identity, dict) or
      entry.get("identity") != current_identity or current_identity.get("cpu_fallback") is not False or
      current_identity.get("backend") not in ("mnn-opencl", "ncnn-vulkan") or
      not current_identity.get("fingerprint") or len(current_identity.get("sha256", {}).get("apk", "")) != 64 or
      entry.get("status") != "passed" or
      entry.get("rtol") != 1e-3 or entry.get("atol") != 1e-4 or
      current_identity.get("name") not in MODEL_OUTPUTS or
      entry.get("assignment", {}).get("no_cpu_fallback_verified") is not True or
      entry.get("assignment", {}).get("trace_limit", 0) != 0 or
      entry.get("assignment", {}).get("profiling_enabled", False) or
      entry.get("assignment", {}).get("diagnostic_dispatch_limit", 0) != current_identity.get("diagnostic_dispatch_limit", 0) or
      (current_identity.get("backend") == "mnn-opencl" and
       entry.get("assignment", {}).get("memory_mode") != current_identity.get("memory_mode"))):
    return False
  return True


def accuracy_passed(entry, current_identity):
  """Bind all 159/636 comparisons to the actual untapped GPU implementation."""
  from fold3_opencl import original_reference_passed
  if (not _accuracy_identity_passed(entry, current_identity) or entry.get("accuracy_status") != "passed" or
      set(entry.get("parity", {})) != MODEL_OUTPUTS[current_identity["name"]]):
    return False
  return original_reference_passed(entry)


def accuracy_part_passed(entry, current_identity, part):
  """Partial evidence never authorizes inference or substitutes recurrent state."""
  if (part not in ACCURACY_PARTS or not _accuracy_identity_passed(entry, current_identity) or
      entry.get("accuracy_part") != part or entry.get("accuracy_status") != "partial" or
      entry.get("performance_status") != "not_run" or entry.get("diagnostic_only") is not True or
      entry.get("qualified") is not False):
    return False
  outputs = MODEL_OUTPUTS[current_identity["name"]]
  def passed(case):
    return set(case) == outputs and all(value.get("passed") is True and
      value.get("unexpected_nonfinite_elements", 0) == 0 for value in case.values())
  if part == "initial":
    seeds = entry.get("additional_seed_parity", {})
    return set(seeds) == {"7", "123"} and all(passed(case) for case in (entry.get("parity", {}), *seeds.values()))
  sequences = entry.get("sequences", {})
  if set(sequences) != {part}:
    return False
  sequence = sequences[part]
  count = 60 if part == "camera-replay" else 32
  checks = sequence.get("checks", [])
  return (sequence.get("passed") is True and sequence.get("frames") == count and len(checks) == count and
          [check.get("frame") for check in checks] == list(range(count)) and
          all(passed(check.get("outputs", {})) for check in checks))


def combine_accuracy_parts(parts, current_identity):
  """Accept only five complete, untapped experiments from the same APK/device."""
  if set(parts) != set(ACCURACY_PARTS) or any(not accuracy_part_passed(parts[part], current_identity, part) for part in parts):
    return None
  combined = {**parts["initial"], "accuracy_status": "passed", "sequences": {
      part: parts[part]["sequences"][part] for part in ACCURACY_PARTS[1:]},
      "accuracy_parts": {part: hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
                         for part, record in parts.items()}}
  combined.pop("accuracy_part")
  return combined if accuracy_passed(combined, current_identity) else None


def benchmark(data_dir, backend, library_dir, memory_mode="image", trace_limit=0, trace_skip=0, trace_case=None,
              stage="all", selected_model=None, accuracy_part=None, road_dispatch_limit=0):
  if type(road_dispatch_limit) is not int or road_dispatch_limit not in (0, 131072, 262144, 524288):
    raise ValueError("Invalid road dispatch experiment")
  if road_dispatch_limit and (backend != "ncnn-vulkan" or trace_limit):
    raise ValueError("Dispatch experiment requires untapped Vulkan models")
  import numpy as np
  from fold3_models import _thermal_check
  from fold3_model_trace import compare, compare_replay_frame
  from fold3_sequences import digest, initial_inputs, trace_inputs, validate
  if trace_case is not None and (not trace_limit or backend != "mnn-opencl"):
    raise ValueError("A selected frame is diagnostic only")
  names = ("dmonitoring_model", "driving_supercombo")
  if stage not in ("all", "accuracy", "timing") or selected_model is not None and selected_model not in names:
    raise ValueError("Invalid accelerator validation stage/model")
  if trace_limit and stage != "all":
    raise ValueError("Traces cannot supply staged accuracy or timing evidence")
  if accuracy_part is not None and (accuracy_part not in ACCURACY_PARTS or stage != "accuracy" or selected_model is None or trace_limit):
    raise ValueError("An accuracy part requires one untapped model and accuracy-only execution")
  root = Path(__file__).parent / "fold3_reference"
  manifest = json.loads((root / "accelerators/manifest.json").read_text())
  engine = "mnn" if backend == "mnn-opencl" else "ncnn"
  report = {"started_utc": time.time(), "backend": backend, "memory_mode": memory_mode,
            "road_dispatch_limit": road_dispatch_limit, "stage": stage, "diagnostic_tracing": bool(trace_limit), "vehicle_tx_enabled": False, "models": {}}
  if accuracy_part is not None:
    report["accuracy_part"] = accuracy_part
  destination = Path(data_dir) / "model-report.json"
  def save():
    destination.write_text(json.dumps(report, indent=2) + "\n")
  save()
  selected = names if selected_model is None else (selected_model,)
  phases = ("accuracy",) if trace_limit else ("accuracy", "timing") if stage == "all" else (stage,)
  def model_path(name):
    return root / "accelerators" / engine / (name + (".mnn" if engine == "mnn" else ".ncnn.param"))
  def dispatch(name):
    return road_dispatch_limit if name == "driving_supercombo" else 0
  def proof_suffix(name):
    return f"-dispatch{dispatch(name)}" if dispatch(name) else ""
  def proof_path(name):
    return Path(data_dir) / f"accel-accuracy-{backend}-{memory_mode}-{name}{proof_suffix(name)}.json"
  def part_path(name, part):
    return Path(data_dir) / f"accel-accuracy-part-{backend}-{memory_mode}-{name}{proof_suffix(name)}-{part}.json"
  if "accuracy" in phases:
    for name in selected:
      proof_path(name).unlink(missing_ok=True)
      (Path(data_dir) / f"accel-qualification-{name}.json").unlink(missing_ok=True)
      if accuracy_part is not None:
        part_path(name, accuracy_part).unlink(missing_ok=True)
  for phase in phases:
    proofs = {}
    if phase == "timing":
      try:
        for name in names:
          proof = json.loads(proof_path(name).read_text())
          current = identity(name, model_path(name), root, backend, library_dir, memory_mode, dispatch(name))
          if not accuracy_passed(proof, current):
            raise ValueError(f"Incomplete or changed original-reference accuracy evidence: {name}")
          proofs[name] = proof
        report["timing_accuracy_gate"] = {"passed": True, "comparisons": 795,
            "identities": {name: proof["identity"] for name, proof in proofs.items()}}
      except (OSError, ValueError) as exc:
        report["timing_blocked_by"] = str(exc)
        for name in selected:
          entry = report["models"].setdefault(name, {"status": "blocked", "performance_status": "blocked"})
          entry["timing_error"] = str(exc)
        break
    for name in selected:
      if (trace_case is not None and trace_case["frame"] is not None and name != "dmonitoring_model" and
          selected_model != "driving_supercombo"):
        continue
      entry = report["models"][name] = dict(proofs[name]) if phase == "timing" else {}
      entry.update(status="running", diagnostic_only=True, performance_status="not_run", qualified=False,
                   accuracy_status="pending" if phase == "accuracy" else "passed", rtol=1e-3, atol=1e-4)
      model = None
      diagnostic_inputs = None
      try:
        _thermal_check()
        metadata = manifest["models"][name]
        conversion = metadata.get(engine, {})
        if not metadata["original_reference"]["passed"] or conversion.get("status") != "converted":
          entry["conversion"] = conversion
          raise ValueError(f"{engine} conversion is not verified against original model: " +
                           conversion.get("error", "missing conversion record" if not conversion else "unverified"))
        path = model_path(name)
        if digest(path) != metadata[engine]["sha256"]:
          raise ValueError("Converted model hash mismatch")
        if engine == "ncnn" and digest(path.with_suffix(".bin")) != metadata[engine]["weights_sha256"]:
          raise ValueError("Converted weight hash mismatch")
        entry["identity"] = identity(name, path, root, backend, library_dir, memory_mode, dispatch(name))
        if trace_case is not None and trace_case["frame"] is not None and name == "driving_supercombo":
          # Warm up untapped GPU execution with its own state, never goldens.
          warmup = AccelModel(path, backend, metadata, memory_mode)
          try:
            diagnostic_inputs = trace_inputs(root, name, trace_case["seed"], trace_case["frame"], warmup, _thermal_check)
            entry["warmup_assignment"] = warmup.describe()
            if trace_case["frame"] and not entry["warmup_assignment"].get("no_cpu_fallback_verified"):
              raise ValueError("Recurrent trace warmup used an unverified backend")
          finally:
            del warmup
        started = time.monotonic_ns()
        model = AccelModel(path, backend, metadata, memory_mode, trace_limit, trace_skip, **({"dispatch_limit": dispatch(name)} if dispatch(name) else {}))
        entry["initialization_ms"] = (time.monotonic_ns() - started) / 1e6
        entry["assignment"] = model.describe()
        save()
        if trace_case is not None:
          seed, frame = trace_case["seed"], trace_case["frame"]
          inputs = diagnostic_inputs if diagnostic_inputs is not None else trace_inputs(root, name, seed, frame)
          diagnostic_inputs = inputs
          actual = model(inputs)
          suffix = "" if seed == 42 else f"-case-{seed}"
          expected_path = root / (f"{name}{suffix}-outputs.npz" if frame is None else f"sequences/{name}-seed{seed}.npz")
          with np.load(expected_path) as expected:
            if frame is None:
              entry["parity"] = {key: compare(value, expected[key]) for key, value in actual.items()}
            else:
              sequence = json.loads((root / "sequences/manifest.json").read_text())["models"][name][f"seed{seed}"]
              entry["parity"] = compare_replay_frame(actual, expected, sequence["uint8_outputs"][frame], frame)
          entry.update(status="diagnostic_only", trace_case=trace_case, assignment=model.describe(),
                       error="Selected-input trace cannot qualify untapped model accuracy or performance")
          continue
        if accuracy_part is not None and accuracy_part != "initial":
          entry["sequences"] = validate(model, root, name, _thermal_check, labels=(accuracy_part,))
        else:
          cases = {}
          for seed in (42, 7, 123):
            _thermal_check()
            actual = model(initial_inputs(root, name, seed))
            suffix = "" if seed == 42 else f"-case-{seed}"
            with np.load(root / f"{name}{suffix}-outputs.npz") as expected:
              cases[str(seed)] = {key: compare(value, expected[key]) for key, value in actual.items()}
          entry.update(parity=cases["42"], additional_seed_parity={key: cases[key] for key in ("7", "123")})
        entry["assignment"] = model.describe()
        save()
        if not entry["assignment"]["no_cpu_fallback_verified"]:
          raise ValueError("No-fallback operator audit incomplete")
        if accuracy_part is not None:
          entry.update(status="passed", accuracy_status="partial", accuracy_part=accuracy_part,
                       started_utc=report["started_utc"], completed_utc=time.time())
          if not accuracy_part_passed(entry, entry["identity"], accuracy_part):
            entry.update(status="failed", accuracy_status="failed", error="Original-reference accuracy part failed")
            continue
          part_path(name, accuracy_part).write_text(json.dumps(entry, indent=2) + "\n")
          parts = {}
          for part in ACCURACY_PARTS:
            try:
              parts[part] = json.loads(part_path(name, part).read_text())
            except (OSError, ValueError):
              pass
          combined = combine_accuracy_parts(parts, entry["identity"])
          if combined is not None:
            proof_path(name).write_text(json.dumps(combined, indent=2) + "\n")
            report["accuracy_aggregate"] = {name: combined}
          continue
        if not all(check["passed"] for case in cases.values() for check in case.values()):
          entry.update(status="failed", accuracy_status="failed",
                       error="Original-reference tolerance failed; recurrent/performance stages not advanced")
          proof_path(name).unlink(missing_ok=True)
          continue
        if trace_limit:
          entry.update(status="diagnostic_only", error="Traced execution cannot qualify untapped model performance")
          continue
        if phase == "accuracy":
          entry["sequences"] = validate(model, root, name, _thermal_check)
          entry.update(status="passed", accuracy_status="passed")
          if not accuracy_passed(entry, entry["identity"]):
            entry.update(status="failed", accuracy_status="failed",
                         error="Original recurrent/replay parity failed; timing not advanced")
            continue
          entry.update(status="passed", performance_status="not_run", diagnostic_only=True, qualified=False)
          proof_path(name).write_text(json.dumps(entry, indent=2) + "\n")
          continue
        save()
        samples, stages = [], []
        inputs = initial_inputs(root, name, 42)
        for _ in range(100):
          _thermal_check()
          start = time.monotonic_ns()
          model(inputs)
          samples.append((time.monotonic_ns() - start) / 1e6)
          stages.append({k: v for k, v in model.describe().items() if k.endswith("_ms")})
          time.sleep(max(0, .05 - samples[-1] / 1000))
        entry.update(status="passed", samples=100, work_p50_ms=float(np.percentile(samples, 50)),
                     work_p99_ms=float(np.percentile(samples, 99)), deadline_misses_50ms=sum(x >= 50 for x in samples),
                     samples_ms=samples, stage_timings=stages, diagnostic_only=False)
        entry["performance_status"] = "passed" if max(samples) < 50 else "failed"
        entry["qualified"] = qualified(entry, identity(name, path, root, backend, library_dir, memory_mode, dispatch(name)))
        if entry["qualified"]:
          (Path(data_dir) / f"accel-qualification-{name}.json").write_text(json.dumps(entry, indent=2) + "\n")
      except Exception as exc:
        entry.update(status="blocked", performance_status="blocked", error=f"{type(exc).__name__}: {exc}")
        if model is not None:
          entry["assignment"] = model.describe()
      finally:
        if model is not None and trace_limit:
          traces = model.native.trace(model.session)
          folder = Path(data_dir) / ("accel-trace-" + name + "-" + memory_mode)
          folder.mkdir(exist_ok=True)
          for index, item in enumerate(traces):
            item["file"] = f"{index:04d}.bin"
            (folder / item["file"]).write_bytes(item.pop("data"))
          seed, frame = (trace_case["seed"], trace_case["frame"]) if trace_case is not None else (42, None)
          inputs = diagnostic_inputs if diagnostic_inputs is not None else trace_inputs(root, name, seed, frame)
          np.savez(folder / "inputs.npz", **inputs)
          (folder / "manifest.json").write_text(json.dumps({"seed": seed, "frame": frame, "diagnostic_only": True,
            "input_archive_sha256": digest(folder / "inputs.npz"),
            "feedback_source": "actual-untapped-backend" if name == "driving_supercombo" and frame is not None else "initial-fixture",
            "warmup_frames": frame if name == "driving_supercombo" and frame is not None else 0,
            "inputs_sha256": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in inputs.items()},
            "model": name, "memory_mode": memory_mode, "trace_skip": trace_skip, "tensors": traces}, indent=2) + "\n")
          entry["trace"] = {"path": str(folder), "tensors": len(traces), "seed": seed, "frame": frame}
        del model
        save()
  report["completed_utc"] = time.time()
  save()
  return report


def live_model(data_dir, name, library_dir):
  """Use an independently qualified model only while its complete identity matches."""
  root = Path(__file__).parent / "fold3_reference"
  try:
    entry = json.loads((Path(data_dir) / f"accel-qualification-{name}.json").read_text())
    backend = entry["identity"]["backend"]
    if backend not in ("mnn-opencl", "ncnn-vulkan"):
      return None
    engine = "mnn" if backend == "mnn-opencl" else "ncnn"
    metadata = json.loads((root / "accelerators/manifest.json").read_text())["models"][name]
    path = root / "accelerators" / engine / (name + (".mnn" if engine == "mnn" else ".ncnn.param"))
    memory_mode = entry["identity"].get("memory_mode", "image")
    current = identity(name, path, root, backend, library_dir, memory_mode)
    if not qualified(entry, current):
      return None
    model = AccelModel(path, backend, metadata, memory_mode)
    # The new session must audit actual assignments before its outputs are used.
    from fold3_sequences import initial_inputs
    model(initial_inputs(root, name, 42))
    if not model.describe()["no_cpu_fallback_verified"]:
      return None
    return model, f"{backend}: original reference, recurrent and camera replay qualified"
  except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
    return None


def gpu_probes(data_dir):
  import numpy as np
  from fold3_models import _thermal_check
  from fold3_model_trace import compare
  from fold3_sequences import digest
  root = Path(__file__).parent / "fold3_reference/gpu-probes"
  manifest = json.loads((root / "manifest.json").read_text())
  report = {"started_utc": time.time(), "diagnostic_only": True, "vehicle_tx_enabled": False, "models": {}}
  destination = Path(data_dir) / "model-report.json"
  for backend, engine in (("mnn-opencl", "mnn"), ("ncnn-vulkan", "ncnn")):
    for kind, metadata in manifest["probes"].items():
      entry = report["models"][backend + "-" + kind] = {"status": "running", "performance_status": "diagnostic_only"}
      model = None
      try:
        _thermal_check()
        path = root / engine / (kind + (".mnn" if engine == "mnn" else ".ncnn.param"))
        if digest(path) != metadata[engine]["sha256"] or digest(root / f"{kind}.npz") != metadata["data_sha256"]:
          raise ValueError("GPU probe hash mismatch")
        if engine == "ncnn" and digest(path.with_suffix(".bin")) != metadata[engine]["weights_sha256"]:
          raise ValueError("GPU probe weight hash mismatch")
        entry["model_sha256"] = digest(path)
        with np.load(root / f"{kind}.npz") as data:
          inputs, expected = {"x": data["x"]}, data["expected"]
        started = time.monotonic_ns()
        model = AccelModel(path, backend, metadata)
        entry["initialization_ms"] = (time.monotonic_ns() - started) / 1e6
        actual = model(inputs)["y"]
        entry["parity"] = compare(actual, expected)
        entry["assignment"] = model.describe()
        samples = []
        for _ in range(20):
          _thermal_check()
          started = time.monotonic_ns()
          model(inputs)
          samples.append((time.monotonic_ns() - started) / 1e6)
        entry.update(status="passed" if entry["parity"]["passed"] else "failed", samples_ms=samples,
                     work_p50_ms=float(np.percentile(samples, 50)), work_p99_ms=float(np.percentile(samples, 99)))
      except Exception as exc:
        entry.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        if model is not None:
          entry["assignment"] = model.describe()
      finally:
        del model
      destination.write_text(json.dumps(report, indent=2) + "\n")
  report["completed_utc"] = time.time()
  destination.write_text(json.dumps(report, indent=2) + "\n")
  return report


def htp_probes(data_dir, library_dir):
  import numpy as np
  from fold3_model_trace import compare
  from fold3_models import _thermal_check
  from fold3_ort import OrtModel
  root = Path(__file__).parent / "fold3_reference/htp-probes"
  manifest = json.loads((root / "manifest.json").read_text())
  report = {"started_utc": time.time(), "backend": "qnn-htp", "diagnostic_only": True,
            "vehicle_tx_enabled": False, "models": {}, "manifest": manifest,
            "provider_options": {"htp_arch": "68", "device_id": "0", "profiling_level": "detailed"}}
  from java import jclass
  from fold3_sequences import digest
  apk = Path(str(jclass("ai.sunnypilot.fold3.AndroidHardware").applicationApk()))
  report["identity"] = {"apk_sha256": digest(apk), "fingerprint": str(jclass("android.os.Build").FINGERPRINT),
    "manifest_sha256": digest(root / "manifest.json"),
    "provider_libraries": {p.name: digest(p) for p in sorted(Path(library_dir).glob("libQnn*.so"))}}
  destination = Path(data_dir) / "model-report.json"
  def save():
    destination.write_text(json.dumps(report, indent=2) + "\n")
  save()
  for key, metadata in manifest["probes"].items():
    entry = report["models"][key] = {"status": "running", "precision": metadata["precision"],
                                    "operator": metadata["operator"], "cpu_fallback": "unverified"}
    model = None
    try:
      _thermal_check()
      if metadata["status"] != "prepared":
        raise RuntimeError(metadata.get("error", "Probe conversion failed"))
      path = root / f"{key}.onnx"
      if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Diagnostic model hash differs from probe manifest")
      data = root / f"{key}.npz"
      if hashlib.sha256(data.read_bytes()).hexdigest() != metadata["data_sha256"]:
        raise ValueError("Diagnostic input hash differs from probe manifest")
      with np.load(data) as source:
        inputs, expected, original = {"x": source["x"]}, source["expected"], source["original"]
      started = time.monotonic_ns()
      entry["provider_options"] = {**report["provider_options"],
                                    "profiling_file_path": str(Path(data_dir) / f"htp-{key}-qnn.csv")}
      model = OrtModel(path, "qnn-htp", library_dir=library_dir, assignments=True, no_cpu_fallback=True,
                       provider_options=entry["provider_options"], profile_path=Path(data_dir) / f"htp-{key}")
      entry["initialization_ms"] = (time.monotonic_ns() - started) / 1e6
      entry["assignment"] = model.describe()
      providers = entry["assignment"].get("providers", {})
      if providers.get("CPUExecutionProvider", {}).get("nodes", 0) or not providers.get("QNNExecutionProvider", {}).get("nodes", 0):
        raise RuntimeError("Requested NPU did not own every operation")
      entry["cpu_fallback"] = "disabled"
      output = model(inputs)["y"]
      entry["backend_parity"] = compare(output, expected)
      entry["original_parity"] = compare(output, original)
      entry["quantization_parity"] = compare(expected, original)
      entry["failure_class"] = ("quantization" if not entry["quantization_parity"]["passed"] else
        "htp_execution" if not entry["backend_parity"]["passed"] else
        "original_reference" if not entry["original_parity"]["passed"] else None)
      samples = []
      for _ in range(10):
        _thermal_check()
        started = time.monotonic_ns()
        model(inputs)
        samples.append((time.monotonic_ns() - started) / 1e6)
      entry.update(status="passed" if entry["backend_parity"]["passed"] and entry["original_parity"]["passed"] else "failed",
                   performance_status="diagnostic_only", samples_ms=samples)
    except Exception as exc:
      entry.update(status="blocked", performance_status="blocked", error=f"{type(exc).__name__}: {exc}",
                   failure_class="unsupported_or_initialization")
    finally:
      if model is not None:
        try:
          entry["profile_path"] = model.end_profiling()
        except Exception as exc:
          entry["profiling_error"] = str(exc)
      del model
    save()
  report["completed_utc"] = time.time()
  report["validation_status"] = "diagnostic_only"
  save()
  return report
