"""Android service entry points. No fork, native executable launch, or Panda TX."""
from collections import deque
import json
import math
import os
from pathlib import Path
import time
import traceback


class Report:
  def __init__(self, name, directory, worker):
    self.path = Path(directory) / f"{name}.json"
    self.worker = worker
    self.samples = deque(maxlen=10000)
    self.last_write = 0.0
    self.data = {"daemon": name, "pid": os.getpid(), "vehicle_tx_enabled": False,
                 "engagement_allowed": False, "work_samples": 0, "deadline_misses": 0}

  def update(self, phase, **values):
    self.worker.progress(phase)
    self.data.update(phase=phase, **values)
    if time.monotonic() - self.last_write >= 1 or phase == "failed":
      if self.samples:
        ordered = sorted(self.samples)
        self.data["p99_work_ms"] = ordered[math.ceil(len(ordered) * .99) - 1] / 1e6
      temporary = self.path.with_suffix(".tmp")
      temporary.write_text(json.dumps(self.data, indent=2))
      if phase == "failed":
        self.path.with_name(self.path.stem + "-last-failure.json").write_text(json.dumps(self.data, indent=2))
      temporary.replace(self.path)
      self.last_write = time.monotonic()

  def controls_progress(self, updated, work_ns):
    if updated:
      self.samples.append(work_ns)
      self.data["work_samples"] += 1
      self.data["deadline_misses"] += int(work_ns >= 10_000_000)
    self.update("running" if updated else "waiting_for_selfdrive_state",
                timing_excludes_input_wait=True)


def controls(report):
  from openpilot.common.params import Params
  from openpilot.selfdrive.controls.controlsd import Controls
  params = Params()
  # Physical fingerprint/firmware qualification must supply these. Simulator
  # parameters live in a separate Params AND msgq namespace.
  while params.get("CarParams") is None or params.get("CarParamsSP") is None:
    report.update("waiting_for_car_params")
    time.sleep(.5)
  report.update("initializing_controls")
  Controls().run(progress=report.controls_progress)


def monitoring(report):
  from openpilot.common.params import Params
  from openpilot.selfdrive.monitoring.dmonitoringd import dmonitoringd_thread
  # Known vehicle configuration supplied by the user, still subject to the
  # upstream driver's wheel-position learning and all DM validity checks.
  params = Params()
  if params.get("IsRhdDetected") is None:
    params.put_bool("IsRhdDetected", True)
  dmonitoringd_thread(progress=lambda updated: report.update("running" if updated else "waiting_for_driver_model"))


def sensors(report):
  from java import jclass
  import openpilot.cereal.messaging as messaging
  hardware = jclass("ai.sunnypilot.fold3.AndroidHardware")
  clock = jclass("android.os.SystemClock")
  source = hardware()
  pm = messaging.PubMaster(["accelerometer", "gyroscope"])
  count, rejected = 0, 0
  try:
    while True:
      # Android sensor timestamps use elapsedRealtimeNanos (includes suspend).
      # Cereal Python timestamps use CLOCK_MONOTONIC. Recompute the offset after
      # resume rather than pretending that the two clocks are identical.
      offset = int(clock.elapsedRealtimeNanos()) - time.monotonic_ns()
      rows = source.drain()
      for row in rows:
        sensor_type, stamp, x, y, z, accuracy = row
        stamp = int(stamp) - offset
        if not 0 <= time.monotonic_ns() - stamp < 1_000_000_000:
          rejected += 1
          continue
        name, union = ("accelerometer", "acceleration") if sensor_type == 1 else ("gyroscope", "gyroUncalibrated")
        event = messaging.new_message(name, valid=False)
        sensor = getattr(event, name)
        sensor.timestamp, sensor.source = stamp, "android"
        sensor.init(union).v = [x, y, z]
        pm.send(name, event)
        count += 1
      report.update("running" if len(rows) else "waiting_for_imu", published_samples=count,
                    dropped_samples=int(source.droppedSamples()), stale_samples=rejected,
                    calibrated=False, validity_blocker="phone-to-vehicle mounting rotation has not been calibrated",
                    thermal_status=int(hardware.thermalStatus()), power_watts=float(hardware.powerWatts()))
      time.sleep(.01)
  finally:
    source.close()


def run(name, directory, worker, simulation=False):
  from fold3_runtime import configure
  configure(directory, simulation)
  report = Report(name, directory, worker)
  report.update("importing", simulation=simulation)
  try:
    if name == "sim_bridge":
      if not simulation:
        raise PermissionError("ADB simulator transport requires an explicit debug simulation session")
      from fold3_sim import serve
      serve(report)
    else:
      {"controlsd": controls, "dmonitoringd": monitoring, "sensord": sensors}[name](report)
  except Exception:
    report.update("failed", error=traceback.format_exc())
    raise
