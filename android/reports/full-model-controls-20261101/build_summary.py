#!/usr/bin/env python3
"""Summarize retained evidence, without promoting diagnostic results."""
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
APK = "9868e5b9ca7d24feaefd9acfbcdabaf4f5226e85063e03fffec4400aecc1a859"


def read(path):
  return json.loads(path.read_text())


def write(name, data):
  (REPORT / name).write_text(json.dumps(data, indent=2) + "\n")


def controls(candidate, path):
  result = read(path)
  metrics = result["runtime_reports"]["controlsd"]
  artifact = read(ROOT / "android/build/candidates" / candidate / "artifacts.json")
  files = artifact.get("files", artifact.get("artifacts"))
  return {"candidate": candidate, "report": str(path.relative_to(ROOT / "android/reports")),
          "apk_sha256": files["app-debug.apk"]["sha256"],
          "fixture_sha256": result["fixture_sha256"],
          "functional_status": result["status"],
          "timing_status": "failed" if (metrics["deadline_misses"] or metrics["cycle_deadline_misses"] or
                                        metrics.get("scheduled_deadline_misses", 0) or
                                        metrics.get("skipped_slots_total", 0)) else "passed",
          "samples": metrics["work_samples"], "mean_hz": metrics["mean_cycle_hz"],
          "work_p99_ms": metrics["p99_work_ms"],
          "legacy_complete_cycle_p99_ms": metrics["p99_complete_cycle_ms"],
          "stages_p99_ms": metrics.get("p99_stages_ms", {}),
          "work_deadline_misses": metrics["deadline_misses"],
          "legacy_cycle_deadline_misses": metrics["cycle_deadline_misses"],
          "scheduled_deadline_misses": metrics.get("scheduled_deadline_misses"),
          "skipped_slots": metrics.get("skipped_slots_total"),
          "ui_mean_fps": result["runtime_reports"]["ui-report"]["mean_fps"],
          "source_mean_hz": result["source_mean_hz"],
          "source_deadline_misses": result["source_deadline_misses"],
          "timing_schema_version": metrics.get("timing_schema_version", 1),
          "metrics_process": metrics.get("metrics_process", "controls"),
          "synthetic_sources": result["synthetic_sources"], "vehicle_tx_enabled": False}


def main():
  artifact = read(ROOT / "android/build/candidates/candidate45/artifacts.json")
  assert artifact["files"]["app-debug.apk"]["sha256"] == APK
  models = read(REPORT / "fold45h-mnn-buffer-retry.json")
  checks = []
  identities = {}
  for name, record in models["models"].items():
    identities[name] = record["identity"]
    for seed, parity in [(42, record["parity"]), *[(int(seed), value) for seed, value in record["additional_seed_parity"].items()]]:
      for output, result in parity.items():
        checks.append({"model": name, "seed": seed, "output": output, **result})
  failed = sum(not check["passed"] for check in checks)
  accuracy = {"status": "failed", "apk_sha256": APK, "backend": "mnn-opencl-buffer",
              "rtol": 1e-3, "atol": 1e-4, "required_original_reference_comparisons": 795,
              "initial_seed_comparisons_run": len(checks), "passed": len(checks) - failed,
              "failed": failed, "remaining_recurrent_replay_comparisons_blocked": 795 - len(checks),
              "unexpected_nonfinite_elements": sum(check["unexpected_nonfinite_elements"] for check in checks),
              "no_cpu_fallback_verified": all(record["assignment"]["no_cpu_fallback_verified"] for record in models["models"].values()),
              "models_promoted": [], "identities": identities, "checks": checks,
              "host_converted_graph_comparisons": {"status": "passed", "count": 795,
                  "evidence": "candidate45h-conversion.log", "qualifies_android_engine": False},
              "historical_strict_opencl_comparisons": {"status": "passed", "count": 795,
                  "candidates": ["candidate35", "candidate40"], "qualifies_candidate45_mnn": False}}
  write("candidate45-accuracy.json", accuracy)
  write("candidate45-artifacts.json", {**artifact, "model_identities": identities,
      "mnn_build": read(ROOT / "android/build/candidates/candidate45/mnn-build.json")})

  old = controls("candidate44", ROOT / "android/reports/npu-recovery-20261101/fold-candidate44-chain-device-steady.json")
  current = controls("candidate45", REPORT / "fold-candidate45-chain-device-steady.json")
  repeat = controls("candidate45h", REPORT / "fold45h-controls-default.json")
  assert old["fixture_sha256"] == current["fixture_sha256"] == repeat["fixture_sha256"]
  write("candidate45-performance-comparison.json", {
      "status": "failed", "matched_fixture_sha256": current["fixture_sha256"],
      "baseline": old, "candidate45": current, "same_apk_separate_run": repeat,
      "work_p99_change_percent": 100 * (current["work_p99_ms"] / old["work_p99_ms"] - 1),
      "legacy_cycle_p99_change_percent": 100 * (current["legacy_complete_cycle_p99_ms"] / old["legacy_complete_cycle_p99_ms"] - 1),
      "limitations": ["Injected inputs; no qualified camera inference workload",
          "New release metrics do not replace legacy cycle latency or deadline counters",
          "Reported batch coverage does not include an unflushed tail at force-stop",
          "Accelerator timing and combined soaks blocked by full-model numerical failure"]})

  retained = []
  for path in sorted(REPORT.glob("*-isolation.json")):
    data = read(path)
    retained.append({"report": path.name, "apk_sha256": data.get("apk_sha256"),
                     "status": data.get("validation_status", data.get("status")),
                     "isolation_status": data.get("isolation_status"),
                     "artifact_stable": data.get("artifact_stable"),
                     "excluded_from_candidate45_accuracy": path.name == "fold45h-mnn-buffer-isolation.json"})
  write("retained-experiments.json", retained)

  devices = {}
  for device in ("fold", "emulator"):
    inventory = REPORT / f"{device}-candidate45-validation-driver.json"
    rows = read(inventory)
    installed = REPORT / f"candidate45-installed-{device}.json"
    def inventory_status(records):
      if any(row["status"] == "failed" or row.get("isolation_status") == "failed" for row in records):
        return "failed"
      return "passed" if all(row["status"] == "passed" and row.get("isolation_status") == "passed" for row in records) else "blocked"
    devices[device] = {"status": inventory_status(rows),
        "checks": [{key: row.get(key) for key in ("name", "status", "isolation_status", "log")} for row in rows],
        "installed": read(installed) if installed.exists() else {"status": "pending"}}
    if device == "emulator":
      devices[device]["renderer"] = "host"
      retry_path = REPORT / "emulator-candidate45-retry1-validation-driver.json"
      if retry_path.exists():
        retry = read(retry_path)
        devices[device]["initial_host_renderer_status"] = devices[device]["status"]
        devices[device]["retry"] = {"renderer": "swiftshader", "status": inventory_status(retry),
            "inventory": retry_path.name,
            "checks": [{key: row.get(key) for key in ("name", "status", "isolation_status", "log")} for row in retry]}
        devices[device]["status"] = devices[device]["retry"]["status"]
        retry_installed = REPORT / "candidate45-installed-emulator-retry1.json"
        if retry_installed.exists():
          devices[device]["retry"]["installed"] = read(retry_installed)
      fresh_path = REPORT / "emulator-candidate45-retry2-validation-driver.json"
      if fresh_path.exists():
        fresh = read(fresh_path)
        fixtures = [row for row in fresh if row["name"] not in ("thermal", "camera-timing")]
        devices[device]["fresh_post_anr_retry"] = {
            "renderer": "swiftshader", "status": inventory_status(fresh),
            "fixture_status": inventory_status(fixtures), "inventory": fresh_path.name,
            "systemui_anr_dismissed_before_inventory": True,
            "checks": [{key: row.get(key) for key in ("name", "status", "isolation_status", "log")} for row in fresh]}
        devices[device]["status"] = devices[device]["fresh_post_anr_retry"]["status"]
  write("candidate45-validation.json", {"overall_status": "failed", "qualification_status": "blocked", "branch": "codex/fold3-android-port",
      "apk_sha256": APK, "vehicle_tx_enabled": False, "physical_engagement_allowed": False,
      "focused_host_tests": {"status": "passed", "run": 137, "evidence": "candidate45-final-host.log"},
      "hyundai_panda_safety_sim_protocol": {"status": "passed", "run": 1920, "skipped": 200,
          "evidence": "candidate45-hyundai-safety-sim-retry.log", "physical_hardware_verified": False},
      "java_unit_build": {"status": "passed", "evidence": "candidate45h-build.log"},
      "devices": devices, "accelerated_accuracy": "failed", "controls_timing": "failed",
      "combined_30s_2min_30min_visual_workloads": {"status": "blocked",
          "reason": "Full-model accuracy, input freshness and timing gates have not passed; injected fixtures are separate"},
      "physical_gates": {"independent_rear_sensor_identity": "blocked", "mounted_calibration": "blocked",
          "panda_firmware_passive_fingerprint_harness_vehicle": "blocked"},
      "cooling_optimization": "deferred", "thermal_safeguards": "retained"})
  print(json.dumps({"apk_sha256": APK, "accuracy": "failed", "controls_timing": "failed", "devices": {key: value["status"] for key, value in devices.items()}}))


if __name__ == "__main__":
  main()
