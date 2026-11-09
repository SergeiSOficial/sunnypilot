#!/usr/bin/env python3
"""One real Camera2 trial; use experiment.py for stop/cooldown/cleanup isolation."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time

from experiment import Device, PACKAGE


def validate(device, ids, logical, seconds, width, height, output, samsung_multiview=False, camera_profile="record", continuous_readback=False, evidence_directory=None, private_gpu=False):
  previous = read(device)
  launched = time.monotonic()
  args = ["shell", "am", "start", "-n", device.package + "/ai.sunnypilot.fold3.MainActivity",
          "--ez", "run_probe", "true", "--es", "camera_ids", ",".join(ids),
          "--ei", "duration_seconds", str(seconds), "--ei", "width", str(width), "--ei", "height", str(height),
          "--es", "camera_profile", camera_profile, "--ez", "private_gpu", str(private_gpu).lower(),
          "--ez", "record_camera_evidence", str(evidence_directory is not None).lower()]
  if logical:
    args += ["--es", "logical_camera", logical]
  if samsung_multiview:
    args += ["--ez", "samsung_multiview", "true", "--ez", "front_camera", str("1" in ids).lower(),
             "--ez", "continuous_readback", str(continuous_readback).lower()]
  device.assert_thermal_safe()
  device.command(*args)
  deadline = time.monotonic() + seconds + 90
  report = {}
  while time.monotonic() < deadline:
    try:
      device.assert_thermal_safe()
    except RuntimeError as exc:
      report = {"status": "failed", "error": str(exc), "host_launch_monotonic_s": launched}
      break
    candidate = read(device)
    if (candidate.get("started_utc") != previous.get("started_utc") and
        (candidate.get("completed_utc") or candidate.get("fatal_error"))):
      report = candidate
      break
    time.sleep(1)
  if not report:
    report = {"status": "failed", "error": "Fresh camera trial did not finish", "host_launch_monotonic_s": launched}
  elif "status" not in report:
    attempts = report.get("attempts", [])
    report["status"] = attempts[0].get("status", "failed") if len(attempts) == 1 else "failed"
    if report["status"] not in ("passed", "failed", "blocked"):
      report["status"] = "failed"
    if report.get("vehicle_tx_enabled") is not False or report.get("driving_runtime_ready") is not False:
      raise RuntimeError("Camera diagnostic transmission/engagement guard failed")
  if evidence_directory is not None and report.get("attempts"):
    try:
      attempt = report["attempts"][0]
      report["evidence"] = (collect_evidence(device, evidence_directory, attempt.get("evidence_frames", {})) if samsung_multiview
                            else collect_direct_evidence(device, evidence_directory, attempt.get("streams", [])))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
      report["evidence"] = {"status": "failed", "reason": str(exc)}
    finally:
      device.command("shell", "run-as", device.package, "rm", "-f",
                     *(
                       ("files/multiview-preview.clip", "files/multiview-thumbnail.clip") if samsung_multiview
                       else (f"files/direct-{sensor}.clip" for sensor in ids)))
  Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(json.dumps({"status": report["status"], "logical_access": report.get("logical_access"),
                    "attempts": report.get("attempts", []), "error": report.get("error")}, allow_nan=False))
  return report["status"] == "passed"


FRAME_BYTES = 960 * 540 * 3 // 2
RECORD_BYTES = 8 + FRAME_BYTES
MAX_SKEW_NS = 20_000_000


def paired_evidence(clips, destination):
  """Export two equally sized raw clips plus their original capture timestamps."""
  previous = [0, 0]

  def read_frame(stream):
    record = clips[stream].read(RECORD_BYTES)
    if not record:
      return None
    if len(record) != RECORD_BYTES:
      raise ValueError("Incomplete camera evidence frame")
    timestamp, = struct.unpack_from(">q", record)
    if timestamp <= previous[stream]:
      raise ValueError("Nonmonotonic camera evidence timestamp")
    previous[stream] = timestamp
    return timestamp, memoryview(record)[8:]

  pairs = []
  with (destination / "preview.nv12").open("wb") as preview, (destination / "thumbnail.nv12").open("wb") as thumbnail:
    left, right = read_frame(0), read_frame(1)
    while left is not None and right is not None:
      skew = left[0] - right[0]
      if abs(skew) <= MAX_SKEW_NS:
        preview.write(left[1])
        thumbnail.write(right[1])
        pairs.append([left[0], right[0]])
        left, right = read_frame(0), read_frame(1)
      elif skew < 0:
        left = read_frame(0)
      else:
        right = read_frame(1)
  return pairs


def collect_evidence(device, directory, expected):
  directory.mkdir(parents=True, exist_ok=True)
  if not all(expected.get(stream, 0) for stream in ("preview", "thumbnail")):
    return {"status": "failed", "reason": "Both clips must contain frames"}
  paths = []
  for stream in ("preview", "thumbnail"):
    path = directory / f"{stream}.clip"
    with path.open("wb") as output:
      subprocess.run([device.adb, "-s", device.serial, "exec-out", "run-as", device.package,
                      "cat", f"files/multiview-{stream}.clip"], stdout=output, check=True, timeout=120)
    if path.stat().st_size != RECORD_BYTES * expected[stream]:
      raise ValueError(f"Truncated {stream} evidence clip")
    paths.append(path)
  with paths[0].open("rb") as preview, paths[1].open("rb") as thumbnail:
    pairs = paired_evidence((preview, thumbnail), directory)
  return {"status": "passed" if pairs else "failed", "paired_frames": len(pairs),
          "timestamp_columns": ["preview_monotonic_ns", "thumbnail_monotonic_ns"],
          "timestamps_ns": pairs, "max_allowed_skew_ns": MAX_SKEW_NS,
          "max_observed_skew_ns": max((abs(a - b) for a, b in pairs), default=None),
          "distinct_sensor_evidence": False}


def collect_direct_evidence(device, directory, streams):
  directory.mkdir(parents=True, exist_ok=True)
  results = {}
  for stream in streams:
    count = stream["evidence_frames"]
    if not count:
      continue
    sensor = stream["sensor_id"]
    path = directory / f"direct-{sensor}.clip"
    with path.open("wb") as output:
      subprocess.run([device.adb, "-s", device.serial, "exec-out", "run-as", device.package, "cat",
                      f"files/{path.name}"], stdout=output, check=True, timeout=30)
    width, height = stream["output_width"], stream["output_height"]
    if path.stat().st_size != count * (8 + width * height * 3 // 2):
      raise ValueError(f"Incomplete direct evidence for sensor {sensor}")
    results[sensor] = {"file": str(path.resolve()), "width": width, "height": height, "frames": count}
  return {"status": "passed" if results else "blocked", "direct_clips": results}


def read(device):
  try:
    return json.loads(device.command("shell", "run-as", device.package, "cat", "files/camera-report.json"))
  except (subprocess.CalledProcessError, json.JSONDecodeError):
    return {}


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--package", default=PACKAGE)
  parser.add_argument("--ids", required=True, help="Distinct physical IDs, for example 0,2,1")
  parser.add_argument("--logical", help="Logical camera serving the rear physical outputs")
  parser.add_argument("--samsung-multiview", action="store_true", help="Diagnose Samsung's composite transport; never qualifies physical streams")
  parser.add_argument("--camera-profile", choices=("record", "preview", "record-720p", "record-depth2", "high-speed-120", "manual-8ms", "manual-8ms-low-latency", "manual-2ms-low-latency", "low-latency", "low-latency-preview", "low-latency-record"), default="record")
  parser.add_argument("--private-gpu", action="store_true", help="Use Camera2 PRIVATE surfaces with GLES conversion")
  parser.add_argument("--continuous-readback", action="store_true", help="Measure continuous GPU conversion/readback without enabling vehicle inputs")
  parser.add_argument("--seconds", type=int, default=10, choices=range(3, 1801))
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--output", required=True)
  parser.add_argument("--evidence-directory", type=Path, help="Export sampled, timestamp-paired diagnostic clips")
  args = parser.parse_args()
  ids = args.ids.split(",")
  if not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids) or any(not i or i.strip() != i for i in ids):
    parser.error("Use one to three distinct application-visible Camera2 IDs")
  if args.samsung_multiview and (args.logical != "20" or set(ids) not in ({"0", "2"}, {"0", "2", "1"})):
    parser.error("Samsung multiview requires --logical 20 and --ids 0,2 or 0,2,1")
  if args.continuous_readback and not args.samsung_multiview:
    parser.error("Continuous readback requires Samsung multiview")
  if args.camera_profile == "high-speed-120" and (
      args.samsung_multiview or args.logical is not None or len(ids) != 1 or not args.private_gpu or args.evidence_directory is not None):
    parser.error("High-speed capture requires one public camera, --private-gpu, and no evidence recording")
  if args.camera_profile in ("manual-8ms", "manual-8ms-low-latency", "manual-2ms-low-latency") and (args.samsung_multiview or args.logical is not None):
    parser.error("Manual exposure requires public independent rear cameras")
  if args.camera_profile.startswith("low-latency") and (args.samsung_multiview or args.logical is not None):
    parser.error("Low-latency profiles require public independent sessions")
  if not args.samsung_multiview and args.camera_profile not in ("record", "preview", "high-speed-120", "manual-8ms", "manual-8ms-low-latency", "manual-2ms-low-latency", "low-latency", "low-latency-preview", "low-latency-record"):
    parser.error("Public Camera2 supports record and preview profiles")
  if args.private_gpu and args.samsung_multiview:
    parser.error("The Samsung path already defines its own PRIVATE output")
  if args.evidence_directory is not None and (args.private_gpu or args.seconds > 60 or not args.samsung_multiview and len(ids) != 1):
    parser.error("Recording evidence requires one direct YUV camera or Samsung multiview and at most 60 seconds")
  raise SystemExit(0 if validate(Device(args.serial, args.adb, args.package), ids, args.logical,
                               args.seconds, args.width, args.height, args.output, args.samsung_multiview,
                               args.camera_profile, args.continuous_readback, args.evidence_directory, args.private_gpu) else 1)
