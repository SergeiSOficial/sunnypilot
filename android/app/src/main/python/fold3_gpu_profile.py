"""Bounded, diagnostic Vulkan timestamps; never performance qualification."""
import json
from pathlib import Path
import time


def summarize(samples):
  if not samples:
    raise ValueError("Missing Vulkan samples")
  totals = {}
  for sample in samples:
    if sample.get("version") != 1 or not sample.get("enabled") or not sample.get("valid") or not sample.get("gpu_layers"):
      raise ValueError("Incomplete Vulkan timestamps")
    bits = sample["timestamp_valid_bits"]
    if not 1 <= bits <= 64:
      raise ValueError("Invalid Vulkan timestamp width")
    for layer in sample["gpu_layers"]:
      elapsed = layer["gpu_ms"]
      if not 0 <= elapsed < float("inf"):
        raise ValueError("Invalid Vulkan layer duration")
      key = (layer["node"], layer["operator"])
      totals[key] = totals.get(key, 0.) + elapsed
  return [{"node": key[0], "operator": key[1], "mean_gpu_ms": value / len(samples)}
          for key, value in sorted(totals.items(), key=lambda pair: pair[1], reverse=True)]


def summarize_submissions(samples):
  """Separate existing intra-graph fences from the final submission wait."""
  if not samples:
    raise ValueError("Missing submission samples")
  rows = []
  for sample in samples:
    row = sample.get("internal_submissions", {})
    if row.get("version") != 1:
      raise ValueError("Missing submission metrics")
    for name in ("internal_count", "dispatch_limit", "rough_score", "submitted_groups"):
      if type(row.get(name)) is not int or row[name] < 0:
        raise ValueError("Invalid submission count")
    wait = row.get("wait_ms", -1.)
    if not 0 <= wait < float("inf") or wait > sample["command_record_ms"] + 1.:
      raise ValueError("Invalid internal submission wait")
    rows.append(row)
  return {"version": 1, "samples": len(rows),
          "dispatch_limits": sorted({row["dispatch_limit"] for row in rows}),
          "rough_scores": sorted({row["rough_score"] for row in rows}),
          "mean_internal_count": sum(row["internal_count"] for row in rows) / len(rows),
          "mean_internal_wait_ms": sum(row["wait_ms"] for row in rows) / len(rows),
          "mean_record_excluding_internal_wait_ms": sum(
            sample["command_record_ms"] - row["wait_ms"] for sample, row in zip(samples, rows)) / len(rows)}


def run(data_dir, library_dir, road_dispatch_limit=0):
  if type(road_dispatch_limit) is not int or road_dispatch_limit not in (0, 131072, 262144, 524288):
    raise ValueError("Invalid road dispatch experiment")
  from fold3_accel import AccelModel, MODEL_OUTPUTS, accuracy_passed, identity
  from fold3_models import _thermal_check
  from fold3_sequences import initial_inputs
  from fold3_model_trace import compare
  from fold3_accel import __file__ as accelerator_file
  root = Path(accelerator_file).parent / "fold3_reference"
  manifest = json.loads((root / "accelerators/manifest.json").read_text())
  report = {"started_utc": time.time(), "diagnostic_only": True, "vehicle_tx_enabled": False,
            "backend": "ncnn-vulkan", "profiling_version": 1, "road_dispatch_limit": road_dispatch_limit, "models": {}}
  destination = Path(data_dir) / "model-report.json"
  def save():
    destination.write_text(json.dumps(report, indent=2) + "\n")
  save()
  try:
    paths, identities = {}, {}
    for name in MODEL_OUTPUTS:
      paths[name] = root / f"accelerators/ncnn/{name}.ncnn.param"
      limit = road_dispatch_limit if name == "driving_supercombo" else 0
      identities[name] = identity(name, paths[name], root, "ncnn-vulkan", library_dir, dispatch_limit=limit)
      suffix = f"-dispatch{limit}" if limit else ""
      proof = json.loads((Path(data_dir) / f"accel-accuracy-ncnn-vulkan-image-{name}{suffix}.json").read_text())
      if not accuracy_passed(proof, identities[name]):
        raise ValueError(f"Complete current-artifact accuracy required before profiling: {name}")
    report["accuracy_comparisons"] = 795
    for name in MODEL_OUTPUTS:
      model = None
      entry = report["models"][name] = {"status": "running", "identity": identities[name],
        "diagnostic_only": True, "qualified": False, "performance_status": "diagnostic_only"}
      save()
      try:
        _thermal_check()
        started = time.monotonic_ns()
        model = AccelModel(paths[name], "ncnn-vulkan", manifest["models"][name], profiling=True,
                           **({"dispatch_limit": road_dispatch_limit} if name == "driving_supercombo" and road_dispatch_limit else {}))
        entry["initialization_ms"] = (time.monotonic_ns() - started) / 1e6
        entry["assignment"] = model.describe()
        inputs = initial_inputs(root, name, 42)
        import numpy as np
        with np.load(root / f"{name}-outputs.npz") as source:
          expected = {key: source[key] for key in source.files}
        samples, wall = [], []
        for index in range(12):
          _thermal_check()
          started = time.monotonic_ns()
          output = model(inputs)
          duration = (time.monotonic_ns() - started) / 1e6
          if not all(compare(output[key], expected[key])["passed"] for key in expected):
            raise ValueError("Profiled graph differs from original reference")
          if index >= 2:
            samples.append(model.native.profile(model.session))
            wall.append(duration)
        entry.update(status="passed", warmup_runs=2, samples=samples, wall_ms=wall, ranked_layers=summarize(samples),
                     submission_summary=summarize_submissions(samples))
      except Exception as exc:
        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
      finally:
        del model
      save()
  except Exception as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
  report["completed_utc"] = time.time()
  save()
  return report
