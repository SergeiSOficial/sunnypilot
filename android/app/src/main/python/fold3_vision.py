"""Android camera-to-model diagnostics using original models and upstream parsers.

Camera projection/mount calibration, numerical parity and timing are unqualified.
No result from this module is allowed to become a valid driving input.
"""
from collections import deque
from itertools import product
import hashlib
import json
from pathlib import Path
import time


MODEL_DEADLINE_NS = 50_000_000


class FrameGate:
  """Bounded, one-to-one timestamp pairing; duplicate/stale frames never pair."""
  def __init__(self, sensors, max_age_ns=100_000_000, max_skew_ns=20_000_000):
    if not sensors or len(set(sensors)) != len(sensors):
      raise ValueError("Distinct physical sensor IDs required")
    self.sensors = tuple(sensors)
    self.queues = {sensor: deque(maxlen=2) for sensor in sensors}
    self.last = dict.fromkeys(sensors, 0)
    self.staging = {}
    self.age, self.skew = max_age_ns, max_skew_ns
    self.rejected = 0
    self.expired = 0
    self.received = 0
    self.skipped = 0
    self.last_frame_id = {}

  def _fresh(self, sensor, stamp, now):
    if sensor not in self.queues or stamp <= self.last[sensor] or not 0 <= now - stamp <= self.age:
      self.rejected += 1
      self.expired += int(sensor in self.queues and stamp > self.last[sensor] and now - stamp > self.age)
      return False
    return True

  def add(self, sensor, stamp, payload, now):
    if not self._fresh(sensor, stamp, now):
      return False
    self.last[sensor] = stamp
    queue = self.queues[sensor]
    if len(queue) == queue.maxlen:
      self.rejected += 1
    queue.append((stamp, payload))
    return True

  def add_frame(self, sensor, stamp, frame_id, data, geometry, now):
    """Own the shared-ring pixels without allocating a frame-sized object each time."""
    self.received += 1
    if sensor in self.queues:
      previous = self.last_frame_id.get(sensor, frame_id)
      self.skipped += max(0, frame_id - previous - 1)
      self.last_frame_id[sensor] = max(previous, frame_id)
    if not self._fresh(sensor, stamp, now):
      return False
    source = memoryview(data).cast("B")
    buffers = self.staging.get(sensor)
    if buffers is None or len(buffers[0]) != len(source):
      self.rejected += len(self.queues[sensor])
      self.queues[sensor].clear()
      # The queue holds at most two frames. The third slot is free until the
      # current pair has finished processing on this same inference thread.
      buffers = self.staging[sensor] = deque(bytearray(len(source)) for _ in range(3))
    buffers[0][:] = source
    accepted = self.add(sensor, stamp, (frame_id, buffers[0], *geometry), now)
    if accepted:
      buffers.rotate(-1)
    return accepted

  def pair(self, now):
    for queue in self.queues.values():
      while queue and now - queue[0][0] > self.age:
        queue.popleft()
        self.rejected += 1
        self.expired += 1
    queues = [self.queues[sensor] for sensor in self.sensors]
    candidates = []
    for indices in product(*(range(len(queue)) for queue in queues)):
      stamps = [queue[index][0] for queue, index in zip(queues, indices, strict=True)]
      if max(stamps) - min(stamps) <= self.skew:
        candidates.append(((min(stamps), max(stamps)), indices))
    if candidates:
      _, indices = max(candidates)
      pair = {}
      for sensor, queue, index in zip(self.sensors, queues, indices, strict=True):
        for _ in range(index):
          queue.popleft()
          self.rejected += 1
        pair[sensor] = queue.popleft()
      return pair
    while all(self.queues.values()):
      times = {sensor: self.queues[sensor][0][0] for sensor in self.sensors}
      self.queues[min(times, key=times.get)].popleft()
      self.rejected += 1
    return None


def carry_model_state(inputs, outputs):
  """Reject non-finite predictions and keep them out of recurrent inputs."""
  import numpy as np
  finite = all(np.isfinite(value).all() for value in outputs.values())
  for key, value in inputs.items():
    if key.startswith("state_"):
      if finite:
        value[:] = outputs["next_" + key]
      else:
        value.fill(0)
  return finite


class DiagnosticWarp:
  """Reusable nearest-neighbour preview projection, explicitly uncalibrated.

  Channel packing matches upstream compile_warp.frames_to_tensor. This affine
  preview must be replaced by a measured optical/mount transform for driving.
  """
  def __init__(self, width, height, stride, uv_offset, output_width, output_height, luma=False):
    import numpy as np
    try:
      from fold3_ort_native import gather_rows_u8
      self.gather = gather_rows_u8
    except ImportError:
      self.gather = None
    if min(width, height) <= 0 or width % 2 or height % 2 or stride < width or uv_offset < stride * height:
      raise ValueError("Invalid NV12 layout")
    self.width, self.height, self.stride, self.uv_offset = width, height, stride, uv_offset
    self.luma = luma
    ow, oh = output_width, output_height
    if ow % 2 or oh % 2:
      raise ValueError("Even model dimensions required")
    scale = max(width / ow, height / oh)
    x = np.rint((np.arange(ow) - (ow - 1) / 2) * scale + (width - 1) / 2).clip(0, width - 1).astype(np.intp)
    y = np.rint((np.arange(oh) - (oh - 1) / 2) * scale + (height - 1) / 2).clip(0, height - 1).astype(np.intp)
    self.columns, self.rows = x, y * stride
    # The native path retains 2,400 row/column indices instead of 1,382,400
    # full offsets for the driver input; repeated scaled rows become memcpy.
    if self.gather is None:
      self.y_indices = (self.rows[:, None] + x[None, :]).ravel()
    self.y = np.empty((oh, ow), dtype=np.uint8)
    if not luma:
      self.uv_indices = (uv_offset + (y[::2, None] // 2) * stride + (x[None, ::2] // 2) * 2).ravel()
      self.v_indices = self.uv_indices + 1
      self.u, self.v = np.empty((oh // 2, ow // 2), dtype=np.uint8), np.empty((oh // 2, ow // 2), dtype=np.uint8)
      self.output = np.empty((6, oh // 2, ow // 2), dtype=np.uint8)

  def __call__(self, data):
    import numpy as np
    source = np.frombuffer(data, dtype=np.uint8)
    if len(source) < self.uv_offset + self.stride * self.height // 2:
      raise ValueError("Truncated NV12 frame")
    # All indices were bounded above and the full source length was checked.
    # clip avoids NumPy's unconditional temporary for mode="raise".
    if self.gather is None:
      np.take(source, self.y_indices, out=self.y.reshape(-1), mode="clip")
    else:
      self.gather(source, self.rows, self.columns, self.y)
    if self.luma:
      return self.y.reshape(1, -1)
    np.take(source, self.uv_indices, out=self.u.reshape(-1), mode="clip")
    np.take(source, self.v_indices, out=self.v.reshape(-1), mode="clip")
    for target, plane in zip(self.output, (self.y[::2, ::2], self.y[1::2, ::2], self.y[::2, 1::2], self.y[1::2, 1::2], self.u, self.v), strict=True):
      target[:] = plane
    return self.output


class RoadWarp(DiagnosticWarp):
  """Measured pinhole/Brown projection with the pinned model's two distinct FOVs.

  Intrinsics/distortion refer to the delivered, oriented image, not a sensor's
  full active array. Samsung composite/GL outputs require their own measurement.
  """
  def __init__(self, width, height, stride, uv_offset, calibration, wide=False):
    import numpy as np
    from openpilot.common.transformations.model import get_warp_matrix
    intrinsics = np.asarray(calibration["intrinsics"], dtype=np.float64)
    euler = np.asarray(calibration["device_from_calib_euler"], dtype=np.float64)
    distortion = np.asarray(calibration["opencv_distortion"], dtype=np.float64)
    if (calibration["image_size"] != [width, height] or intrinsics.shape != (3, 3) or euler.shape != (3,)
        or distortion.shape != (5,) or not all(np.isfinite(a).all() for a in (intrinsics, euler, distortion))
        or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0 or not np.array_equal(intrinsics[2], [0, 0, 1])
        or intrinsics[0, 1] != 0 or intrinsics[1, 0] != 0):
      raise ValueError("Invalid measured camera geometry")
    super().__init__(width, height, stride, uv_offset, 512, 256)
    self.gather = None  # Perspective indices depend on both coordinates.
    matrix = get_warp_matrix(euler, intrinsics, bigmodel_frame=wide)

    def indices(ow, oh, chroma=False):
      yy, xx = np.indices((oh, ow), dtype=np.float64)
      xy = matrix @ np.stack((xx.ravel() * (2 if chroma else 1), yy.ravel() * (2 if chroma else 1), np.ones(ow * oh)))
      if (xy[2] <= 1e-8).any():
        raise ValueError("Camera projection crosses the horizon")
      x, y = xy[:2] / xy[2]
      x, y = (x - intrinsics[0, 2]) / intrinsics[0, 0], (y - intrinsics[1, 2]) / intrinsics[1, 1]
      k1, k2, p1, p2, k3 = distortion
      r2 = x*x + y*y
      radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
      dx, dy = x * radial + 2*p1*x*y + p2*(r2 + 2*x*x), y * radial + p1*(r2 + 2*y*y) + 2*p2*x*y
      scale = 2 if chroma else 1
      x, y = (dx * intrinsics[0, 0] + intrinsics[0, 2]) / scale, (dy * intrinsics[1, 1] + intrinsics[1, 2]) / scale
      if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite camera projection")
      x = np.rint(x).clip(0, width // scale - 1).astype(np.intp)
      y = np.rint(y).clip(0, height // scale - 1).astype(np.intp)
      return y * stride + x * scale

    self.y_indices = indices(512, 256)
    self.uv_indices = uv_offset + indices(256, 128, True)
    self.v_indices = self.uv_indices + 1


def load_road_calibration(path, origins):
  """A mount or stream change invalidates calibration; never invent defaults."""
  data = json.loads(Path(path).read_text())
  if data.get("schema_version") != 1 or data.get("mount_status") != "measured":
    raise ValueError("Measured final vehicle mount calibration required")
  cameras = data["cameras"]
  if set(cameras) != {"0", "2"}:
    raise ValueError("Both measured rear cameras required")
  identities = []
  for sensor, camera in cameras.items():
    if (not camera.get("measurement_evidence") or not camera.get("physical_sensor_id")
        or camera.get("stream_origin") != origins.get(sensor) or "unverified" in camera["stream_origin"]):
      raise ValueError("Camera identity, measurement or stream origin not established")
    identities.append(camera["physical_sensor_id"])
  if len(set(identities)) != 2:
    raise ValueError("Distinct measured rear sensor identities required")
  return cameras


class ThermalGate:
  """Pause bench model work at Android MODERATE; resume after ten cool seconds."""
  def __init__(self):
    self.paused, self.cool_since = False, None

  def update(self, status, now):
    if status >= 4:
      raise RuntimeError(f"Critical Android thermal status: {status}")
    if status >= 2:
      self.paused, self.cool_since = True, None
    elif self.paused:
      if self.cool_since is None:
        self.cool_since = now
      if now - self.cool_since >= 10_000_000_000:
        self.paused = False
    return self.paused


def model_asset(name):
  root = Path(__file__).parent
  metadata = json.loads((root / "fold3_reference/manifest.json").read_text())["models"][name]
  path = root / f"openpilot/selfdrive/modeld/models/{name}.onnx"
  if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
    raise ValueError("Model asset hash mismatch")
  return path, metadata


def camera_available(client):
  # A successful VisionIPC socket handshake can legitimately return zero FDs
  # for a stream which the server does not produce.
  return client.is_connected() and client.num_buffers > 0


def inference_delay_ns(last_started, now):
  # Bound model work to the upstream 20 Hz cadence without catching up after a stall.
  return max(0, last_started + MODEL_DEADLINE_NS - now)


def publish_with_deadline(pm, messages, deadline_ns, clock=time.monotonic_ns):
  # Serialize first so parsing and encoding consume the same capture budget.
  payloads = tuple((name, msg.to_bytes()) for name, msg in messages)
  for service, payload in payloads:
    if clock() > deadline_ns:
      return False
    pm.send(service, payload)
  # A send already in flight cannot be withdrawn. Its completion still counts
  # as a miss, and all Android diagnostic messages are explicitly invalid.
  return clock() <= deadline_ns


def run(report, driver=False):
  import numpy as np
  from fold3_opencl import live_model
  from msgq.visionipc import VisionIpcClient
  from openpilot.cereal.visionipc import VisionStreamType
  from openpilot.cereal import log, messaging
  from java import jclass
  hardware = jclass("ai.sunnypilot.fold3.AndroidHardware")
  thermal, was_paused = ThermalGate(), False
  calibration, origins, calibration_blocker = {}, {}, "Measured optical and final mount calibration missing"
  sensors = ("1",) if driver else ("0", "2")
  types = (VisionStreamType.VISION_STREAM_CABIN,) if driver else (VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD)
  clients = {sensor: VisionIpcClient("camerad", stream, True) for sensor, stream in zip(sensors, types, strict=True)}
  gate = FrameGate(sensors, max_age_ns=MODEL_DEADLINE_NS)
  model, warps, inputs = None, {}, {}
  count, published, late, resets, nonfinite = 0, 0, 0, 0, 0
  numerical_fault = False
  last_inference, last_retry, last_started = 0, 0, 0
  warmup_discarded, thermal_discarded = 0, 0
  stages = {name: deque(maxlen=4096) for name in ("receive_copy", "input_age", "prepare", "inference", "output_age",
                                                "publication", "capture_to_publication")}
  report.data.update(capture_budget_ns=MODEL_DEADLINE_NS, timing_schema="sensor-capture-v1", work_deadline_misses=0)
  pm = messaging.PubMaster(["driverStateV2"] if driver else ["modelV2", "cameraOdometry"])
  if driver:
    from openpilot.selfdrive.modeld.dmonitoringmodeld import slice_outputs, parse_model_output, get_driverstate_packet
  else:
    from openpilot.selfdrive.modeld.parse_model_outputs import Parser
    from openpilot.selfdrive.modeld.fill_model_msg import PublishState, fill_model_msg, fill_pose_msg
    parser, publish_state = Parser(), PublishState()
  while True:
    status = int(hardware.thermalStatus())
    if thermal.update(status, time.monotonic_ns()):
      if not was_paused:
        for queue in gate.queues.values():
          thermal_discarded += len(queue)
          queue.clear()
        for key, value in inputs.items():
          if key.startswith("state_"):
            value.fill(0)
        resets += 1
        last_inference = 0
      was_paused = True
      report.update("thermal_paused", thermal_status=status, target_hz=20, target_suspended=True,
                    valid=False, calibrated=False, computed_frames=count, recurrent_resets=resets,
                    blocker="Bench inference paused at Android MODERATE; requires ten cool seconds before resuming")
      time.sleep(.25)
      continue
    was_paused = False
    report.data.update(thermal_status=status, target_suspended=False)
    if numerical_fault:
      # A non-finite prediction cannot become a usable input by spending more
      # CPU on the same unqualified backend. Preserve the fault until restart.
      report.update("blocked_numerical_fault", valid=False, calibrated=False, computed_frames=count,
                    published_frames=published, nonfinite_outputs=nonfinite, numerical_fault_latched=True,
                    blocker="Non-finite model output; inference stopped until the worker is restarted")
      time.sleep(.25)
      continue
    delay = inference_delay_ns(last_started, time.monotonic_ns())
    if delay:
      time.sleep(delay / 1e9)
    now = time.monotonic_ns()
    if now - last_retry > 200_000_000:
      for client in clients.values():
        if not camera_available(client):
          client.connect(False)
      last_retry = now
    missing = [sensor for sensor, client in clients.items() if not camera_available(client)]
    if missing:
      report.update("waiting_for_camera", missing_physical_sensors=missing, calibrated=False, valid=False,
                    blocker="Distinct narrow and wide rear streams are required" if not driver else "Driver camera unavailable")
      time.sleep(.02)
      continue
    report.data["missing_physical_sensors"] = []
    receiving = time.monotonic_ns()
    for sensor, client in clients.items():
      frame = client.recv(timeout_ms=0)
      if frame is None:
        continue
      # Copy out of the bounded shared ring before another producer frame can
      # overwrite it during model execution; no Camera2 buffer is retained.
      geometry = (frame.width, frame.height, frame.stride, frame.uv_offset)
      gate.add_frame(sensor, client.timestamp_sof, client.frame_id, frame.data, geometry, time.monotonic_ns())
    stages["receive_copy"].append(time.monotonic_ns() - receiving)
    pair = gate.pair(time.monotonic_ns())
    report.data.update(received_inputs=gate.received, skipped_input_frames=gate.skipped, expired_inputs=gate.expired,
                       rejected_frames=gate.rejected, thermal_discarded_inputs=thermal_discarded,
                       deadline_misses=gate.expired + late)
    if pair is None:
      report.update("waiting_for_synchronized_frames", rejected_frames=gate.rejected, computed_frames=count,
                    published_frames=published, late_outputs=late, valid=False, calibrated=False)
      time.sleep(.005)
      continue
    if model is None:
      warmup_discarded += len(pair)
      report.data["warmup_discarded_inputs"] = warmup_discarded
      report.update("initializing_diagnostic_model")
      try:
        sources = json.loads((report.path.parent / "vision-sources.json").read_text())
        origins = sources["stream_origins"]
        report.data["stream_origins"] = origins
      except (OSError, KeyError, ValueError):
        origins = {}
      if not driver:
        try:
          calibration = load_road_calibration(report.path.parent / "camera-calibration.json", origins)
          calibration_blocker = "Measured geometry loaded; numerical parity, stream identity and timing still unqualified"
        except (OSError, KeyError, ValueError) as exc:
          calibration_blocker = str(exc)
      path, metadata = model_asset("dmonitoring_model" if driver else "driving_supercombo")
      model, backend = live_model(report.path.parent, "dmonitoring_model" if driver else "driving_supercombo", path)
      inputs = {key: np.zeros(shape, dtype=metadata["input_dtypes"][key]) for key, shape in metadata["inputs"].items()}
      slices = {key: slice(*value) for key, value in metadata["output_slices"].items()}
      if not driver:
        inputs["traffic_convention"][:] = [[0., 1.]]  # Known RHD vehicle.
      # Model initialization can invalidate the pair which triggered loading.
      continue
    newest = max(stamp for stamp, _ in pair.values())
    if last_inference and newest - last_inference > 100_000_000:
      for key, value in inputs.items():
        if key.startswith("state_"):
          value.fill(0)
      resets += 1
    started = time.monotonic_ns()
    last_started = started
    oldest = min(stamp for stamp, _ in pair.values())
    deadline = oldest + MODEL_DEADLINE_NS
    stages["input_age"].append(started - oldest)
    for i, sensor in enumerate(sensors):
      stamp, (frame_id, data, width, height, stride, uv_offset) = pair[sensor]
      key = (sensor, width, height, stride, uv_offset)
      if key not in warps:
        warps = {k: v for k, v in warps.items() if k[0] != sensor}
        warps[key] = (RoadWarp(width, height, stride, uv_offset, calibration[sensor], wide=sensor == "2")
                      if calibration and not driver else
                      DiagnosticWarp(width, height, stride, uv_offset, 1440 if driver else 512, 960 if driver else 256, luma=driver))
      prepared = warps[key](data)
      if driver:
        inputs["input_img"][:] = prepared
      else:
        inputs["new_img"][i] = prepared
    # Missing mount calibration is deliberately not replaced by learned/default
    # calibration. Zero calib/action/desire below belong only to this invalid
    # diagnostic output, never the qualified driving model runtime.
    prepared_at = time.monotonic_ns()
    stages["prepare"].append(prepared_at - started)
    if prepared_at > deadline:
      late += 1
      report.update("waiting_for_fresh_frames", late_outputs=late, deadline_misses=gate.expired + late,
                    valid=False, calibrated=False)
      continue
    outputs = model(inputs)
    inferred_at = time.monotonic_ns()
    finite = carry_model_state(inputs, outputs)
    elapsed = time.monotonic_ns() - started
    report.samples.append(elapsed)
    report.data["work_samples"] += 1
    report.data["work_deadline_misses"] += int(elapsed > MODEL_DEADLINE_NS)
    count += 1
    last_inference = newest
    stages["inference"].append(inferred_at - prepared_at)
    ready_at = time.monotonic_ns()
    stages["output_age"].append(ready_at - oldest)
    on_time = ready_at <= deadline
    if not finite:
      nonfinite += 1
      resets += 1
      numerical_fault = True
    elif on_time and driver:
      parsed = parse_model_output(slice_outputs(outputs["outputs"].astype(np.float32).reshape(-1), slices))
      parsed["raw_pred"] = b""
      message = get_driverstate_packet(parsed, frame_id, newest, elapsed / 1e9, 0.)
      message.valid = False
      on_time = publish_with_deadline(pm, (("driverStateV2", message),), deadline)
    elif on_time:
      raw = outputs["outputs"].astype(np.float32)
      parsed = parser.parse_outputs({key: raw[:, value] for key, value in slices.items()})
      message, pose = messaging.new_message("modelV2", valid=False), messaging.new_message("cameraOdometry", valid=False)
      fill_model_msg(message, parsed, log.ModelDataV2.Action(), publish_state, pair["0"][1][0], pair["2"][1][0],
                     pair["0"][1][0], 0., oldest, elapsed / 1e9, False)
      fill_pose_msg(pose, parsed, pair["0"][1][0], 0, oldest, False)
      on_time = publish_with_deadline(pm, (("modelV2", message), ("cameraOdometry", pose)), deadline)
    if finite and ready_at <= deadline:
      completed = time.monotonic_ns()
      stages["publication"].append(completed - ready_at)
      stages["capture_to_publication"].append(completed - oldest)
      on_time &= completed <= deadline
    if not on_time:
      late += 1
      report.data["deadline_misses"] = gate.expired + late
      if finite:
        for key, value in inputs.items():
          if key.startswith("state_"):
            value.fill(0)
        resets += 1
    published += int(finite and on_time)
    if time.monotonic() - report.last_write >= 1:
      percentiles = {key: np.percentile(values, (50, 95, 99)) / 1e6 for key, values in stages.items() if values}
      for index, quantile in enumerate((50, 95, 99)):
        report.data[f"stage_p{quantile}_ms"] = {key: float(values[index]) for key, values in percentiles.items()}
    report.update("running", valid=False, calibrated=False, computed_frames=count, published_frames=published,
                  rejected_frames=gate.rejected, late_outputs=late, nonfinite_outputs=nonfinite, recurrent_resets=resets, work_ms=elapsed / 1e6,
                  projection="measured narrow/large model geometry" if calibration else "uncalibrated preview affine; not suitable for driving",
                  backend=backend,
                  measured_projection_loaded=bool(calibration), calibration_blocker=calibration_blocker,
                  input_frame_ids={sensor: value[1][0] for sensor, value in pair.items()},
                  input_timestamps_ns={sensor: value[0] for sensor, value in pair.items()}, input_skew_ns=newest - oldest,
                  numerical_parity="failed on original synthetic reference; diagnostic only", target_hz=20,
                  stage_sampling="last 4096 computations")
