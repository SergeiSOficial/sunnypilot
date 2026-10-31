"""Android camera-to-model diagnostics using original models and upstream parsers.

Camera projection/mount calibration, numerical parity and timing are unqualified.
No result from this module is allowed to become a valid driving input.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import time


class FrameGate:
  """Bounded, one-to-one timestamp pairing; duplicate/stale frames never pair."""
  def __init__(self, sensors, max_age_ns=100_000_000, max_skew_ns=20_000_000):
    if len(set(sensors)) != len(sensors):
      raise ValueError("Distinct physical sensor IDs required")
    self.sensors = tuple(sensors)
    self.queues = {sensor: deque(maxlen=2) for sensor in sensors}
    self.last = dict.fromkeys(sensors, 0)
    self.age, self.skew = max_age_ns, max_skew_ns
    self.rejected = 0

  def add(self, sensor, stamp, payload, now):
    if sensor not in self.queues or stamp <= self.last[sensor] or not 0 <= now - stamp <= self.age:
      self.rejected += 1
      return False
    self.last[sensor] = stamp
    queue = self.queues[sensor]
    if len(queue) == queue.maxlen:
      self.rejected += 1
    queue.append((stamp, payload))
    return True

  def pair(self, now):
    for queue in self.queues.values():
      while queue and now - queue[0][0] > self.age:
        queue.popleft()
        self.rejected += 1
    while all(self.queues.values()):
      times = {sensor: self.queues[sensor][0][0] for sensor in self.sensors}
      if max(times.values()) - min(times.values()) <= self.skew:
        return {sensor: self.queues[sensor].popleft() for sensor in self.sensors}
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
    if min(width, height) <= 0 or width % 2 or height % 2 or stride < width or uv_offset < stride * height:
      raise ValueError("Invalid NV12 layout")
    self.width, self.height, self.stride, self.uv_offset = width, height, stride, uv_offset
    self.luma = luma
    ow, oh = output_width, output_height
    if ow % 2 or oh % 2:
      raise ValueError("Even model dimensions required")
    scale = max(width / ow, height / oh)
    x = np.rint((np.arange(ow) - (ow - 1) / 2) * scale + (width - 1) / 2).clip(0, width - 1).astype(np.int32)
    y = np.rint((np.arange(oh) - (oh - 1) / 2) * scale + (height - 1) / 2).clip(0, height - 1).astype(np.int32)
    self.y_indices = (y[:, None] * stride + x[None, :]).ravel()
    self.y = np.empty((oh, ow), dtype=np.uint8)
    if not luma:
      self.uv_indices = (uv_offset + (y[::2, None] // 2) * stride + (x[None, ::2] // 2) * 2).ravel()
      self.u, self.v = np.empty((oh // 2, ow // 2), dtype=np.uint8), np.empty((oh // 2, ow // 2), dtype=np.uint8)
      self.output = np.empty((6, oh // 2, ow // 2), dtype=np.uint8)

  def __call__(self, data):
    import numpy as np
    source = np.frombuffer(data, dtype=np.uint8)
    if len(source) < self.uv_offset + self.stride * self.height // 2:
      raise ValueError("Truncated NV12 frame")
    np.take(source, self.y_indices, out=self.y.reshape(-1))
    if self.luma:
      return self.y.reshape(1, -1)
    np.take(source, self.uv_indices, out=self.u.reshape(-1))
    np.take(source, self.uv_indices + 1, out=self.v.reshape(-1))
    for target, plane in zip(self.output, (self.y[::2, ::2], self.y[1::2, ::2], self.y[::2, 1::2], self.y[1::2, 1::2], self.u, self.v), strict=True):
      target[:] = plane
    return self.output


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


def run(report, driver=False):
  import numpy as np
  from fold3_ort import OrtModel
  from msgq.visionipc import VisionIpcClient
  from openpilot.cereal.visionipc import VisionStreamType
  from openpilot.cereal import log, messaging
  sensors = ("1",) if driver else ("0", "2")
  types = (VisionStreamType.VISION_STREAM_CABIN,) if driver else (VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD)
  clients = {sensor: VisionIpcClient("camerad", stream, True) for sensor, stream in zip(sensors, types, strict=True)}
  gate = FrameGate(sensors)
  model, warps, inputs = None, {}, {}
  count, published, late, resets, nonfinite = 0, 0, 0, 0, 0
  last_inference, last_retry = 0, 0
  pm = messaging.PubMaster(["driverStateV2"] if driver else ["modelV2", "cameraOdometry"])
  if driver:
    from openpilot.selfdrive.modeld.dmonitoringmodeld import slice_outputs, parse_model_output, get_driverstate_packet
  else:
    from openpilot.selfdrive.modeld.parse_model_outputs import Parser
    from openpilot.selfdrive.modeld.fill_model_msg import PublishState, fill_model_msg, fill_pose_msg
    parser, publish_state = Parser(), PublishState()
  while True:
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
    for sensor, client in clients.items():
      frame = client.recv(timeout_ms=0)
      if frame is None:
        continue
      # Copy out of the bounded shared ring before another producer frame can
      # overwrite it during model execution; no Camera2 buffer is retained.
      payload = (client.frame_id, bytes(frame.data), frame.width, frame.height, frame.stride, frame.uv_offset)
      gate.add(sensor, client.timestamp_sof, payload, time.monotonic_ns())
    pair = gate.pair(time.monotonic_ns())
    if pair is None:
      report.update("waiting_for_synchronized_frames", rejected_frames=gate.rejected, computed_frames=count,
                    published_frames=published, late_outputs=late, valid=False, calibrated=False)
      time.sleep(.005)
      continue
    if model is None:
      report.update("initializing_diagnostic_model")
      path, metadata = model_asset("dmonitoring_model" if driver else "driving_supercombo")
      model = OrtModel(path, "cpu", threads=4)
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
    for i, sensor in enumerate(sensors):
      stamp, (frame_id, data, width, height, stride, uv_offset) = pair[sensor]
      key = (sensor, width, height, stride, uv_offset)
      if key not in warps:
        warps = {k: v for k, v in warps.items() if k[0] != sensor}
        warps[key] = DiagnosticWarp(width, height, stride, uv_offset, 1440 if driver else 512, 960 if driver else 256, luma=driver)
      prepared = warps[key](data)
      if driver:
        inputs["input_img"][:] = prepared
      else:
        inputs["new_img"][i] = prepared
    # Missing mount calibration is deliberately not replaced by learned/default
    # calibration. Zero calib/action/desire below belong only to this invalid
    # diagnostic output, never the qualified driving model runtime.
    outputs = model(inputs)
    finite = carry_model_state(inputs, outputs)
    elapsed = time.monotonic_ns() - started
    report.samples.append(elapsed)
    report.data["work_samples"] += 1
    report.data["deadline_misses"] += int(elapsed >= 50_000_000)
    count += 1
    last_inference = newest
    stale = time.monotonic_ns() - min(stamp for stamp, _ in pair.values()) > gate.age
    if not finite:
      nonfinite += 1
      resets += 1
    elif stale:
      late += 1
    elif driver:
      parsed = parse_model_output(slice_outputs(outputs["outputs"].astype(np.float32).reshape(-1), slices))
      parsed["raw_pred"] = b""
      message = get_driverstate_packet(parsed, frame_id, newest, elapsed / 1e9, 0.)
      message.valid = False
      pm.send("driverStateV2", message)
      published += 1
    else:
      raw = outputs["outputs"].astype(np.float32)
      parsed = parser.parse_outputs({key: raw[:, value] for key, value in slices.items()})
      message, pose = messaging.new_message("modelV2", valid=False), messaging.new_message("cameraOdometry", valid=False)
      fill_model_msg(message, parsed, log.ModelDataV2.Action(), publish_state, pair["0"][1][0], pair["2"][1][0],
                     pair["0"][1][0], 0., newest, elapsed / 1e9, False)
      fill_pose_msg(pose, parsed, pair["0"][1][0], 0, newest, False)
      pm.send("modelV2", message)
      pm.send("cameraOdometry", pose)
      published += 1
    report.update("running", valid=False, calibrated=False, computed_frames=count, published_frames=published,
                  rejected_frames=gate.rejected, late_outputs=late, nonfinite_outputs=nonfinite, recurrent_resets=resets, work_ms=elapsed / 1e6,
                  projection="uncalibrated preview affine; not suitable for driving", backend="ORT CPU, 4 threads",
                  numerical_parity="failed on original synthetic reference; diagnostic only")
