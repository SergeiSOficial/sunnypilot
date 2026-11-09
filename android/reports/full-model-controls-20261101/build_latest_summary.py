#!/usr/bin/env python3
"""Bind the final evidence to an immutable APK; keep earlier reports intact."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

from build_summary import controls

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_accel import accuracy_passed, combine_accuracy_parts


def read(path):
  return json.loads(path.read_text())


def write(name, value):
  (REPORT / name).write_text(json.dumps(value, indent=2) + "\n")


def host_result(filename):
  path = REPORT / filename
  if not path.exists():
    return {"status": "blocked", "evidence": filename}
  log = path.read_text()
  runs = list(re.finditer(r"^Ran (\d+) tests? in ([\d.]+)s$", log, re.MULTILINE))
  if not runs:
    return {"status": "failed", "evidence": filename}
  run = runs[-1]
  tail = log[run.end():]
  skipped = re.search(r"^OK \(skipped=(\d+)\)$", tail, re.MULTILINE)
  return {"status": "passed" if re.search(r"^OK(?: \(skipped=\d+\))?$", tail, re.MULTILINE) else "failed",
          "run": int(run[1]), "seconds": float(run[2]), "skipped": int(skipped[1]) if skipped else 0,
          "evidence": filename}


def accelerator_timing(source, apk, identities):
  result = read(source)
  isolation_path = source.with_name(source.stem + "-isolation.json")
  isolation = read(isolation_path)
  assert isolation["artifact_stable"] and isolation["apk_sha256"] == apk
  assert isolation["isolation_status"] == "passed"
  gate = result.get("timing_accuracy_gate", {})
  assert gate.get("passed") is True and gate.get("comparisons") == 795
  assert gate["identities"] == identities
  records = {}
  for name, entry in result["models"].items():
    assert entry["identity"] == identities[name]
    assert entry["assignment"].get("no_cpu_fallback_verified") is True
    stages = entry.get("stage_timings", [])
    records[name] = {key: entry.get(key) for key in (
        "status", "performance_status", "qualified", "samples", "work_p50_ms", "work_p99_ms",
        "deadline_misses_50ms", "initialization_ms", "error")}
    if result.get("validation_status") == "blocked":
      # The host can stop the service between its atomic report writes. Keep
      # the raw report intact, but never summarize its last "running" record
      # as a completed benchmark or derive percentiles from a last-call stage.
      records[name].update(status="blocked", performance_status="blocked", qualified=False,
                           error=result.get("error"), samples=len(entry.get("samples_ms", [])))
    records[name]["last_reported_call_wall_stages_ms"] = {
        key: value for key, value in entry["assignment"].items() if key.endswith("_ms")}
    records[name]["last_reported_call_is_performance_sample"] = False
    records[name]["wall_stage_percentiles_ms"] = {
        key: {"p50": float(np.percentile([row[key] for row in stages], 50)),
              "p99": float(np.percentile([row[key] for row in stages], 99))}
        for key in sorted(set.intersection(*(set(row) for row in stages)))
    } if stages else {}
    records[name].update(evidence=source.name, isolation=isolation_path.name,
                         evidence_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
  return records


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("candidate")
  parser.add_argument("--emulator-inventory", type=Path)
  parser.add_argument("--fold-inventory", type=Path)
  parser.add_argument("--model-evidence", type=Path, action="append", help="Separate isolated accuracy-stage report")
  parser.add_argument("--timing-evidence", type=Path, action="append", default=[], help="Isolated full-model timing report")
  parser.add_argument("--build-evidence", type=Path, help="Successful build log, retaining interrupted attempts separately")
  args = parser.parse_args()
  candidate = args.candidate
  suffix = candidate.removeprefix("candidate")
  artifact = read(ROOT / "android/build/candidates" / candidate / "artifacts.json")
  apk = artifact["files"]["app-debug.apk"]["sha256"]
  build_evidence = args.build_evidence or REPORT / f"{candidate}-build.log"
  model_sources = args.model_evidence or [REPORT / f"fold{suffix}-mnn-buffer.json"]
  models = {"models": {}}
  model_experiments = []
  for source in model_sources:
    if not source.exists():
      model_experiments.append({"expected_evidence": source.name, "validation_status": "blocked",
          "isolation_status": "not_run", "error": "Required device model evidence is missing; no comparisons counted"})
      continue
    result = read(source)
    evidence_sources = result.get("part_evidence") or [{"evidence": source.name,
        "isolation": source.stem + "-isolation.json"}]
    parts = {}
    for record in evidence_sources:
      assert all(Path(record[key]).name == record[key] for key in ("evidence", "isolation"))
      part_source = source.parent / record["evidence"]
      part_result = read(part_source)
      if "sha256" in record:
        assert hashlib.sha256(part_source.read_bytes()).hexdigest() == record["sha256"]
        assert len(part_result["models"]) == 1
        part_entry = next(iter(part_result["models"].values()))
        assert part_entry["accuracy_part"] not in parts
        parts[part_entry["accuracy_part"]] = part_entry
      isolation_path = source.parent / record["isolation"]
      isolation = read(isolation_path)
      assert isolation["artifact_stable"] and isolation["apk_sha256"] == apk
      model_experiments.append({"evidence": part_source.name, "isolation": isolation_path.name,
                                "validation_status": part_result.get("validation_status"),
                                "error": part_result.get("error"), "isolation_status": isolation["isolation_status"]})
    if parts:
      assert len(result["models"]) == 1
      aggregate = next(iter(result["models"].values()))
      assert combine_accuracy_parts(parts, aggregate["identity"]) == aggregate
    for name, entry in result["models"].items():
      assert entry.get("identity", {}).get("sha256", {}).get("apk") == apk
      if name not in models["models"]:
        models["models"][name] = entry
      else:
        previous = models["models"][name]
        assert previous["identity"] == entry["identity"], "Mixed model identities"
        merged = {**previous, **entry}
        for key in ("parity", "additional_seed_parity", "sequences"):
          old, new = previous.get(key, {}), entry.get(key, {})
          assert not set(old).intersection(new), "Duplicate numerical comparisons"
          if old or new:
            merged[key] = {**old, **new}
        models["models"][name] = merged
  checks, identities, assignments = [], {}, {}
  for name, entry in models["models"].items():
    identities[name] = entry.get("identity")
    assignments[name] = entry.get("assignment")
    cases = {"42": entry.get("parity", {}), **entry.get("additional_seed_parity", {})}
    for seed, outputs in cases.items():
      checks.extend({"model": name, "case": seed, "output": key, **value}
                    for key, value in outputs.items())
    for label, sequence in entry.get("sequences", {}).items():
      for frame in sequence["checks"]:
        checks.extend({"model": name, "case": label, "frame": frame["frame"], "output": key, **value}
                      for key, value in frame["outputs"].items())
  failed = sum(not check["passed"] for check in checks)
  no_fallback = bool(models["models"]) and all(entry.get("assignment", {}).get("no_cpu_fallback_verified") is True
                                            for entry in models["models"].values())
  full_accuracy = (len(checks) == 795 and failed == 0 and no_fallback and
                   all(accuracy_passed(entry, entry["identity"]) for entry in models["models"].values()) and
                   all(item["isolation_status"] == "passed" for item in model_experiments))
  converted = read(ROOT / "android/build/candidates" / candidate / "graphs/manifest.json")
  host_checks = {name: {key: entry["original_reference"][key] for key in ("passed", "comparisons")}
                 for name, entry in converted["models"].items()}
  write(f"{candidate}-accuracy.json", {
      "status": "passed" if full_accuracy else "failed" if failed else "blocked",
      "candidate": candidate, "apk_sha256": apk, "rtol": 1e-3, "atol": 1e-4,
      "backend": "mnn-opencl-buffer", "required_original_reference_comparisons": 795,
      "comparisons_run": len(checks), "passed": len(checks) - failed, "failed": failed,
      "comparisons_blocked": 795 - len(checks),
      "unexpected_nonfinite_elements": sum(check.get("unexpected_nonfinite_elements", 0) for check in checks),
      "model_status": {name: {key: value for key, value in entry.items()
                              if key in ("status", "error", "performance_status", "qualified")}
                       for name, entry in models["models"].items()},
      "models_promoted": [name for name, entry in models["models"].items() if entry.get("qualified")],
      "experiments": model_experiments,
      "no_cpu_fallback_verified": no_fallback,
      "identities": identities, "assignments": assignments, "checks": checks,
      "host_derived_graphs": {"models": host_checks, "evidence": f"android/build/candidates/{candidate}/graphs/manifest.json",
                              "qualifies_device_backend": False}})
  baseline = controls("candidate44", ROOT / "android/reports/npu-recovery-20261101/fold-candidate44-chain-device-steady.json")
  current_path = REPORT / f"fold-{candidate}-chain-device-steady.json"
  current = controls(candidate, current_path)
  assert current["fixture_sha256"] == baseline["fixture_sha256"]
  raw_metrics = read(current_path)["runtime_reports"]["controlsd"]
  current["cpu_placement"] = raw_metrics.get("android_controls_cpu_placement")
  current["diagnostic_cpu_affinity"] = raw_metrics.get("android_controls_affinity")
  timings = {}
  for source in args.timing_evidence:
    assert full_accuracy, "Timing cannot qualify incomplete numerical evidence"
    records = accelerator_timing(source, apk, identities)
    assert not set(records).intersection(timings), "Duplicate timing evidence"
    timings.update(records)
  write(f"{candidate}-performance-comparison.json", {
      "status": "failed" if current["legacy_cycle_deadline_misses"] or current["work_deadline_misses"]
                                  or current.get("scheduled_deadline_misses") or current.get("skipped_slots") else "passed",
      "matched_fixture_sha256": current["fixture_sha256"], "runs": [baseline, current],
      "accelerator_timing": {"status": "passed" if full_accuracy and set(timings) == set(identities) and all(
          row["performance_status"] == "passed" and row["qualified"] is True for row in timings.values())
          else "failed" if any(row["performance_status"] == "failed" for row in timings.values()) else "blocked",
          "models": timings,
          "measurement": "Complete inference wall time; upload/compute/readback are host wall stages. "
                         "Asynchronous GPU execution can finish during readback; these are not per-kernel GPU event times."},
      "notes": ["Raw intervals, legacy complete-cycle latency and deadline counters retain their definitions",
                "Release-to-completion is supplemental schema-2 evidence",
                "Force-stop can leave an unflushed journal tail; reported samples are retained",
                "Injected controls fixtures do not qualify camera perception or vehicle operation"]})
  devices = {}
  for device in ("fold", "emulator"):
    inventory = getattr(args, device + "_inventory") or REPORT / f"{device}-{candidate}-validation-driver.json"
    rows = read(inventory)
    installed = read(REPORT / f"{candidate}-installed-{device}.json")
    assert installed["files"]["app-debug.apk"]["sha256"] == apk
    checks = []
    for row in rows:
      record = {key: row.get(key) for key in ("name", "status", "isolation_status", "log")}
      if row["status"] == "pending":
        record["status"] = "blocked"
      else:
        command = row["command"]
        evidence = read(Path(command[command.index("--report") + 1]))
        record["apk_sha256"] = evidence.get("apk_sha256")
        assert record["apk_sha256"] == apk, f"Wrong APK for {device} {row['name']}"
      checks.append(record)
    device_passed = bool(checks) and all(row["status"] == "passed" and row.get("isolation_status") == "passed"
                                         for row in checks)
    device_failed = any(row["status"] == "failed" or row.get("isolation_status") == "failed" for row in checks)
    devices[device] = {"status": "passed" if device_passed else "failed" if device_failed else "blocked",
                       "installed": installed,
                       "inventory": inventory.name, "checks": checks,
                       "retained_attempts": [p.name for p in sorted(REPORT.glob(f"{device}-{candidate}*-validation-driver.json"))
                                             if p.resolve() != inventory.resolve()]}
  write(f"{candidate}-validation.json", {
      "candidate": candidate, "apk_sha256": apk, "branch": "codex/fold3-android-port",
      "qualification_status": "blocked", "vehicle_tx_enabled": False, "physical_engagement_allowed": False,
      "host_tests": host_result(f"{candidate}-all-host.log"),
      "hyundai_safety_simulator_protocol_tests": host_result(f"{candidate}-hyundai-safety-sim.log"),
      "java_unit_build": {"status": "passed" if "BUILD SUCCESSFUL" in build_evidence.read_text()
                         else "failed", "evidence": build_evidence.name},
      "devices": devices, "accuracy_report": f"{candidate}-accuracy.json",
      "performance_report": f"{candidate}-performance-comparison.json",
      "combined_visual_workloads": {"status": "blocked", "stages": ["30 seconds", "2 minutes", "30 minutes"],
          "reason": "Accuracy, fresh camera input and timing must pass before advancement"},
      "physical_gates": {"independent_rear_sensor_identity": "blocked", "mounted_calibration": "blocked",
                         "Panda_firmware_passive_fingerprint_harness_vehicle": "blocked"},
      "cooling_optimization": "deferred", "thermal_safeguards": "retained"})
  retained = []
  for path in sorted(REPORT.glob("*-isolation.json")):
    evidence = read(path)
    retained.append({"experiment": path.name, "apk_sha256": evidence.get("apk_sha256"),
                     "validation_status": evidence.get("validation_status"),
                     "isolation_status": evidence.get("isolation_status"),
                     "artifact_stable": evidence.get("artifact_stable")})
  write(f"{candidate}-retained-experiments.json", retained)
  print(json.dumps({"candidate": candidate, "apk_sha256": apk, "accuracy_passed": full_accuracy,
                    "device_status": {key: value["status"] for key, value in devices.items()}}))


if __name__ == "__main__":
  main()
