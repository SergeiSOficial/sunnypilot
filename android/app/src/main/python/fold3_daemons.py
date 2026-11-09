"""Android service entry points. No fork, native executable launch, or Panda TX."""
from collections import deque
from array import array
import base64
import gc
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback


class Report:
  def __init__(self, name, directory, worker):
    self.path = Path(directory) / f"{name}.json"
    self.worker = worker
    self.samples = deque(maxlen=10000)
    self.cycle_samples = deque(maxlen=10000)
    self.stage_samples = {}
    self.controls_started_ns = None
    self.last_write = 0.0
    self.last_heartbeat = 0.0
    self.data = {"daemon": name, "pid": os.getpid(), "vehicle_tx_enabled": False,
                 "engagement_allowed": False, "work_samples": 0, "deadline_misses": 0}
    self.pending = queue.Queue(maxsize=1)
    self.write_error = None
    self.control_buffer = None
    self.control_row_count = 0
    self.control_buffers = queue.LifoQueue(maxsize=3)
    self.control_columns = None
    self.batch_sequence = 0
    self.batch_directory = Path(directory) / "controlsd-batches" if name == "controlsd" else None
    if self.batch_directory is not None:
      self.batch_directory.mkdir(exist_ok=True)
    self.writer = threading.Thread(target=self._writer, name=name + "-report", daemon=True)
    self.writer.start()

  def _writer(self):
    while True:
      snapshot = self.pending.get()
      if snapshot is None:
        return
      if self.batch_directory is not None:
        data, values, row_count, columns, done = snapshot
        try:
          # The manager process owns percentile calculation and controlsd.json.
          # Only a single second of samples crosses this bounded handoff.
          if len(list(self.batch_directory.glob("*.json"))) >= 8:
            raise RuntimeError("Controls metrics consumer stalled; samples cannot be discarded")
          self.batch_sequence += 1
          output = self.batch_directory / f"{data['pid']}-{self.batch_sequence:010d}.json"
          temporary = output.with_suffix(".tmp")
          # Encoding thousands of floats as JSON holds this interpreter's GIL.
          # Copy the bounded owned batch as IEEE doubles; the manager decodes it.
          used = memoryview(values)[:row_count * len(columns)] if values is not None else b""
          # Preserve every bit, including numerical faults. The manager validates
          # each decoded value before publishing statistics; scanning again here
          # would hold the controls interpreter's GIL for the complete batch.
          if sys.byteorder != "little":
            used = array("d", used)
            used.byteswap()
          temporary.write_text(json.dumps({"version": 3, "sequence": self.batch_sequence,
              "data": data, "columns": columns, "row_count": row_count,
              "rows_base64": base64.b64encode(used).decode("ascii")}, allow_nan=False))
          temporary.replace(output)
        except Exception as exc:
          self.write_error = exc
        finally:
          if values is not None:
            self.control_buffers.put_nowait(values)
          if done is not None:
            done.set()
        continue
      data, samples, cycles, stages, done = snapshot
      try:
        if samples:
          data["p99_work_ms"] = sorted(samples)[math.ceil(len(samples) * .99) - 1] / 1e6
        if cycles:
          data["p99_complete_cycle_ms"] = sorted(cycles)[math.ceil(len(cycles) * .99) - 1]
        data["p99_stages_ms"] = {key: sorted(values)[math.ceil(len(values) * .99) - 1]
                                  for key, values in stages.items() if values}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2))
        if data["phase"] == "failed":
          self.path.with_name(self.path.stem + "-last-failure.json").write_text(json.dumps(data, indent=2))
        temporary.replace(self.path)
      except Exception as exc:
        self.write_error = exc
      finally:
        if done is not None:
          done.set()

  def close(self):
    self.pending.put(None, timeout=5)
    self.writer.join(5)
    if self.writer.is_alive():
      raise TimeoutError("Daemon report writer did not stop")

  def update(self, phase, **values):
    if self.write_error is not None:
      raise RuntimeError("Daemon report writer failed") from self.write_error
    now = time.monotonic()
    # Controls may alternate running/waiting on every nonblocking input poll.
    # Keep its 4 Hz heartbeat without crossing into Java for each transient
    # phase. Failure notification stays immediate; other daemons are unchanged.
    phase_changed = phase != self.data.get("phase")
    if now - self.last_heartbeat >= .25 or phase == "failed" or (phase_changed and self.batch_directory is None):
      self.worker.progress(phase)
      self.last_heartbeat = now
    self.data.update(phase=phase, **values)
    if now - self.last_write >= 1 or phase == "failed":
      # Snapshot on the loop thread; sorting and disk I/O must not stall controls.
      done = threading.Event() if phase == "failed" else None
      if self.batch_directory is not None:
        # Three owned buffers cover the active, queued and writing batches.
        # A buffer returns to the pool only after serialization completes.
        replacement = self.control_buffers.get_nowait() if self.control_buffer is not None else None
        snapshot = (dict(self.data), self.control_buffer, self.control_row_count, self.control_columns or (), done)
        # Never overwrite an unconsumed timing batch, including failure reports.
        self.pending.put_nowait(snapshot)
        self.control_buffer = replacement
        self.control_row_count = 0
      else:
        snapshot = (dict(self.data), tuple(self.samples), tuple(self.cycle_samples),
                    {key: tuple(samples) for key, samples in self.stage_samples.items()}, done)
        try:
          self.pending.get_nowait()
        except queue.Empty:
          pass
        self.pending.put_nowait(snapshot)
      self.last_write = now
      if done is not None:
        if not done.wait(5):
          raise TimeoutError("Failure report was not flushed")
        if self.write_error is not None:
          raise RuntimeError("Daemon report writer failed") from self.write_error

  def controls_progress(self, updated, work_ns, timings):
    started = time.monotonic_ns()
    # Flush completed samples before adding this cycle. The row being measured
    # is never shared with the writer until the following batch is handed off.
    self.update("running" if updated else "waiting_for_selfdrive_state")
    if self.controls_started_ns is None:
      self.controls_started_ns = started
    self.data["work_samples"] += 1
    self.data["deadline_misses"] += int(work_ns >= 10_000_000)
    self.data["updated_input_samples"] = self.data.get("updated_input_samples", 0) + int(updated)
    self.data.update(timings, work_timing_excludes_input_wait=True, timing_schema_version=2)
    if self.control_columns is None:
      self.control_columns = ("work_ns", "complete_cycle_ms", *timings, "reporting_ms", "release_to_completion_ms")
      self.control_buffer = array("d", [0.]) * (1024 * len(self.control_columns))
      for _ in range(2):
        self.control_buffers.put_nowait(array("d", [0.]) * len(self.control_buffer))
    elif tuple(timings) != self.control_columns[2:-2]:
      raise ValueError("Controls timing columns changed during a run")
    if self.control_row_count >= 1024:
      raise RuntimeError("Controls metrics batch overflow")
    if started > self.controls_started_ns:
      self.data["mean_cycle_hz"] = (self.data["work_samples"] - 1) * 1e9 / (started - self.controls_started_ns)
    offset = self.control_row_count * len(self.control_columns)
    self.control_buffer[offset] = work_ns
    for index, value in enumerate(timings.values(), offset + 2):
      self.control_buffer[index] = value
    self.control_row_count += 1
    reporting_ns = time.monotonic_ns() - started
    complete_cycle_ms = timings["cycle_before_reporting_ms"] + reporting_ns / 1e6
    release_ms = timings["release_to_completion_before_reporting_ms"] + reporting_ns / 1e6
    self.control_buffer[offset + 1] = complete_cycle_ms
    self.control_buffer[offset + len(self.control_columns) - 2] = reporting_ns / 1e6
    self.control_buffer[offset + len(self.control_columns) - 1] = release_ms
    self.data["cycle_deadline_misses"] = self.data.get("cycle_deadline_misses", 0) + int(complete_cycle_ms >= 10)
    self.data["scheduled_deadline_misses"] = self.data.get("scheduled_deadline_misses", 0) + int(release_ms >= 10)
    self.data["skipped_slots_total"] = self.data.get("skipped_slots_total", 0) + timings["skipped_slots"]
    self.data.update(reporting_ms=reporting_ns / 1e6,
                     complete_cycle_ms=complete_cycle_ms)


def controls_cpu_placement(report, model):
  """Keep the Fold 3 loop on the measured pair of performance cores.

  The calling controls thread changes its own mask only. Other workers and
  platforms retain their scheduling. Kernel cpusets remain authoritative.
  """
  if model != "SM-F926B":
    return
  if report.data.get("simulation") and (report.path.parent / "controls-affinity-diagnostic.json").exists():
    return
  policy = {"policy": "fold3-performance-pair-v3", "diagnostic_only": False}
  report.data["android_controls_cpu_placement"] = policy
  try:
    allowed = os.sched_getaffinity(0)
    capacities = {cpu: int(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpu_capacity").read_text()) for cpu in allowed}
    policy.update(before=sorted(allowed), capacities=capacities)
    # Android can remove any of the three performance cores from this task's
    # allowed mask. Retain the measured preference, then accept the remaining
    # homogeneous pair when CPU4 is unavailable. Never expand the allowed mask
    # or substitute a little/prime core for a missing performance core.
    selected = next((pair for pair in ([4, 6], [4, 5], [5, 6]) if set(pair).issubset(allowed) and
                     capacities[pair[0]] == capacities[pair[1]] and capacities[pair[0]] > 0 and
                     any(value < capacities[pair[0]] for value in capacities.values())), None)
    if selected is None:
      raise ValueError("Homogeneous Fold 3 performance pair is unavailable")
    os.sched_setaffinity(0, selected)
    actual = os.sched_getaffinity(0)
    if actual != set(selected):
      raise RuntimeError("Controls CPU placement was not applied")
    policy.update(status="applied", after=sorted(actual))
  except (OSError, ValueError) as exc:
    # Preserve the failure and let the ordinary timing gate reject the run.
    policy.update(status="unavailable", error=str(exc))


def controls_affinity(report):
  """Self-only CPU placement probe, restricted to debug simulator sessions."""
  configuration = report.path.parent / "controls-affinity-diagnostic.json"
  if not report.data.get("simulation") or not configuration.exists():
    return
  if configuration.stat().st_size > 128:
    raise ValueError("Oversized diagnostic controls affinity")
  selected = json.loads(configuration.read_text())
  allowed = os.sched_getaffinity(0)
  if (not isinstance(selected, list) or not selected or
      any(type(cpu) is not int or cpu not in allowed for cpu in selected)):
    raise ValueError("Diagnostic controls CPUs must be a nonempty allowed subset")
  os.sched_setaffinity(0, selected)
  actual = os.sched_getaffinity(0)
  if actual != set(selected):
    raise RuntimeError("Diagnostic controls affinity was not applied")
  report.data["android_controls_affinity"] = {"diagnostic_only": True,
      "before": sorted(allowed), "after": sorted(actual)}


def controls(report):
  from openpilot.common.params import Params
  from openpilot.common.realtime import config_realtime_process, Priority
  from openpilot.selfdrive.controls.controlsd import Controls
  params = Params()
  # Physical fingerprint/firmware qualification must supply these. Simulator
  # parameters live in a separate Params AND msgq namespace.
  while params.get("CarParams") is None or params.get("CarParamsSP") is None:
    report.update("waiting_for_car_params")
    time.sleep(.5)
  report.update("initializing_controls")
  # Match upstream main(): disabling cyclic GC was previously omitted by this
  # service entry point. Android's PC hardware path skips privileged schedulers.
  config_realtime_process(4, Priority.CTRL_HIGH)
  report.data["cyclic_gc_enabled"] = gc.isenabled()
  # The Android worker otherwise competes at the default foreground priority;
  # matched runs measured multi-millisecond release delays despite short work.
  # This is the public app thread API, not a privileged realtime scheduler.
  from java import jclass
  process = jclass("android.os.Process")
  model = str(jclass("android.os.Build").MODEL)
  # Give the deadline worker the public low-latency app thread priority.
  # This remains SCHED_OTHER, scoped to this thread; no realtime policy or
  # thermal/cpuset override is requested. Record the actual applied value.
  process.setThreadPriority(process.THREAD_PRIORITY_AUDIO)
  report.data["android_thread_priority"] = process.getThreadPriority(process.myTid())
  report.data["android_worker_tid"] = process.myTid()
  report.data["android_allowed_cpus_at_start"] = sorted(os.sched_getaffinity(0))
  try:
    # This kernel omits the proc timerslack files. PR_GET_TIMERSLACK queries
    # this calling thread directly and does not change its scheduling policy.
    import ctypes
    prctl = ctypes.CDLL("libc.so", use_errno=True).prctl
    prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
    prctl.restype = ctypes.c_long
    slack = prctl(30, 0, 0, 0, 0)  # PR_GET_TIMERSLACK, from the pinned NDK.
    if slack < 0:
      raise OSError(ctypes.get_errno(), "PR_GET_TIMERSLACK")
    report.data["android_timer_slack_ns"] = slack
  except (OSError, ValueError) as exc:
    report.data["android_timer_slack_error"] = str(exc)
  controls_cpu_placement(report, model)
  controls_affinity(report)
  # Preserve the measured policy. The 1 ms GIL experiment regressed in the
  # matched emulator fixture; report the actual quantum without changing it.
  report.data["android_controls_gil_switch_interval_s"] = sys.getswitchinterval()
  Controls(android=True).run(progress=report.controls_progress, android=True)


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


def run(name, directory, worker, simulation=False, full_chain=False):
  from fold3_runtime import configure
  configure(directory, simulation)
  report = Report(name, directory, worker)
  report.update("importing", simulation=simulation)
  try:
    if name == "sim_bridge":
      if not simulation:
        raise PermissionError("ADB simulator transport requires an explicit debug simulation session")
      from fold3_sim import serve
      serve(report, full_chain=full_chain)
    else:
      if name in ("modeld", "dmonitoringmodeld"):
        if simulation:
          raise PermissionError("Physical cameras and diagnostic models are excluded from simulator workers")
        from fold3_vision import run as vision
        vision(report, driver=name == "dmonitoringmodeld")
      else:
        from fold3_driving import MODULES, upstream, vehicle
        if name in MODULES:
          upstream(name, report)
        else:
          {"controlsd": controls, "dmonitoringd": monitoring, "sensord": sensors, "card": vehicle}[name](report)
  except Exception:
    report.update("failed", error=traceback.format_exc())
    raise
  finally:
    report.close()
