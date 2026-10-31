#!/usr/bin/env python3
"""Validate an already-running Raylib UI without restarting driving services."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np
from PIL import Image


def validate(serial, adb, resize, report_path):
  if resize and not serial.startswith("emulator-"):
    raise ValueError("display overrides are restricted to the dedicated emulator")
  def command(*args):
    return subprocess.check_output([adb, "-s", serial, *args], text=True, timeout=30)
  def snapshot():
    def read(name):
      return json.loads(command("shell", "run-as", "ai.sunnypilot.fold3", "cat", f"files/{name}.json"))
    return {"ui": read("ui-report"), "manager": read("manager")}
  def visible_banner():
    # Samsung screencap may prepend a display-selection warning to stdout.
    # Capture to a unique file, then read only its PNG bytes.
    remote = f"/data/local/tmp/fold3-lifecycle-{uuid.uuid4().hex}.png"
    try:
      command("shell", "screencap", "-p", remote)
      png = subprocess.check_output([adb, "-s", serial, "exec-out", "cat", remote], timeout=30)
    finally:
      command("shell", "rm", "-f", remote)
    with Image.open(BytesIO(png)) as screenshot:
      pixels = np.asarray(screenshot.convert("RGB"), dtype=np.int16)
      red = (pixels[:, :, 0] > 80) & (pixels[:, :, 1] < pixels[:, :, 0] * .4) & (pixels[:, :, 2] < pixels[:, :, 0] * .4)
      return {"screenshot_size": list(screenshot.size), "tx_disabled_banner_pixels": int(red.sum()),
              "passed": int(red.sum()) >= 500}
  def wait_for_frames(previous, *, new_renderer=False, resized=False, expected_size=None):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
      current = snapshot()
      new_pid = current["ui"].get("pid") != previous["ui"].get("pid")
      dimensions_changed = current["ui"].get("surface_size") != previous["ui"].get("surface_size")
      frames_advanced = current["ui"].get("frames", 0) > previous["ui"].get("frames", 0)
      ready = new_pid and current["ui"].get("frames", 0) > 0 if new_renderer else frames_advanced
      landscape_size = sorted(expected_size, reverse=True) if expected_size is not None else None
      expected_surface = landscape_size is None or current["ui"].get("surface_size") == landscape_size
      if ready and expected_surface and (not resized or dimensions_changed):
        assert current["ui"]["status"] == "running"
        assert current["ui"]["full_upstream_ui"] is True
        assert new_pid == new_renderer, "unexpected UI process lifetime"
        assert current["manager"]["pid"] == previous["manager"]["pid"], "manager restarted"
        assert current["manager"]["vehicle_tx_enabled"] is False
        for name, process in previous["manager"]["processes"].items():
          assert current["manager"]["processes"][name]["pid"] == process["pid"], f"{name} restarted"
        current["visible_rendering"] = visible_banner()
        expected_pixels = landscape_size is None or current["visible_rendering"]["screenshot_size"] == landscape_size
        if current["visible_rendering"]["passed"] and expected_pixels:
          return current
      time.sleep(.25)
    raise TimeoutError("Raylib did not resume visible frame delivery with its TX-disabled banner")
  report = {"serial": serial, "vehicle_tx_enabled": False, "physical_fold_transition_verified": False,
            "status": "starting", "display_override_test": resize}
  restore = None
  try:
    deadline = time.monotonic() + 45
    while True:
      first = snapshot()
      # Workers change their process title; Android's pidof no longer matches
      # the package name. ps NAME retains the original Android process name.
      live_ui_pids = [line.split()[0] for line in command("shell", "ps", "-A", "-o", "PID,NAME").splitlines()
                     if line.split()[-1] == "ai.sunnypilot.fold3:ui"]
      live_ui_pid = live_ui_pids[0] if live_ui_pids else ""
      if first["ui"].get("status") == "running" and str(first["ui"].get("pid")) == live_ui_pid:
        first["visible_rendering"] = visible_banner()
        if first["visible_rendering"]["passed"]:
          break
      if time.monotonic() >= deadline:
        raise TimeoutError("no current Raylib rendering report")
      time.sleep(.25)
    report["before"] = first
    command("shell", "input", "keyevent", "KEYCODE_HOME")
    time.sleep(2)
    report["background"] = snapshot()
    command("shell", "am", "start", "-n", "ai.sunnypilot.fold3/.MainActivity", "--ez", "raylib", "true")
    report["resumed"] = wait_for_frames(report["background"])
    if resize:
      size = command("shell", "wm", "size")
      restore = size.split("Override size: ")[-1].strip() if "Override size:" in size else "reset"
      restored_size = [int(value) for value in (restore if restore != "reset" else size.split("Physical size: ")[-1].strip()).split("x")]
      report["display_sizes"] = {"requested": [1280, 720], "restore": restored_size}
      command("shell", "wm", "size", "1280x720")
      report["resized"] = wait_for_frames(report["resumed"], resized=True, expected_size=[1280, 720])
      command("shell", "wm", "size", restore)
      restore = None
      report["restored"] = wait_for_frames(report["resized"], resized=True, expected_size=restored_size)
    previous = report.get("restored", report["resumed"])
    command("shell", "input", "keyevent", "KEYCODE_BACK")
    time.sleep(2)
    command("shell", "am", "start", "-n", "ai.sunnypilot.fold3/.MainActivity", "--ez", "raylib", "true",
            "--ez", "simulation", str(previous["manager"]["simulation"]).lower())
    report["renderer_recreated"] = wait_for_frames(previous, new_renderer=True)
    report["status"] = "passed"
  except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
  finally:
    if restore is not None:
      command("shell", "wm", "size", restore)
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--resize", action="store_true")
  parser.add_argument("--report", default="android-lifecycle.json")
  args = parser.parse_args()
  result = validate(args.serial, args.adb, args.resize, args.report)
  print(json.dumps(result, indent=2))
  raise SystemExit(0 if result["status"] == "passed" else 1)
