#!/usr/bin/env python3
"""Pin 60 recorded camera inputs and original-model recurrent CPU references."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
sys.path.insert(0, str(ROOT / "tinygrad_repo"))
from fold3_sequences import SEEDS, STEPS, REPLAY_FRAMES, digest, initial_inputs, sequence_input, feedback, output_record

ROUTE = "98395b7c5b27882e|0000002b--2686b5a2d0"
BASE = "https://commadataci.blob.core.windows.net/openpilotci/98395b7c5b27882e/0000002b--2686b5a2d0/1/"
ASSETS = {"rlog.zst": "bb74414f3ef0d9c1f92713c52baa775ca96602312ff8ff8413491ccb5f238a14",
          "fcamera.hevc": "55bbabc1f481fb86170c4203d629264dca44b0c09cdbe93a577cb5446e00261e",
          "ecamera.hevc": "d42b9ccc13b35ccbcb77787bf311a7a8ee1e60255412df4e6c8dacde65e04e8a",
          "dcamera.hevc": "495e175eda6156ec8f4256696110db7767eddaf636a7975d6ecb919431d78da7"}


def fetch(folder, name):
  path = folder / name
  if not path.exists():
    temporary = path.with_suffix(".download")
    urllib.request.urlretrieve(BASE + name, temporary)
    if digest(temporary) != ASSETS[name]:
      temporary.unlink()
      raise ValueError(f"Replay download hash mismatch: {name}")
    temporary.replace(path)
  if digest(path) != ASSETS[name]:
    raise ValueError(f"Replay asset hash mismatch: {name}")
  return path


def frames(path, width, height, count):
  completed = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(count),
                              "-f", "rawvideo", "-pix_fmt", "nv12", "-threads", "1", "pipe:1"],
                             check=True, stdout=subprocess.PIPE)
  expected = count * width * height * 3 // 2
  if len(completed.stdout) != expected:
    raise ValueError(f"Replay frame count/dimensions differ: {path}")
  return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(count, -1)


def camera_inputs(folder, destination):
  from openpilot.tools.lib.logreader import LogReader
  from openpilot.common.transformations.camera import DEVICE_CAMERAS, _os_fisheye
  from openpilot.common.transformations.model import get_warp_matrix, dmonitoringmodel_intrinsics
  from tinygrad import Tensor
  from examples.openpilot.compile_warp import NV12Frame, make_frame_prepare, make_luma_warp
  log = list(LogReader(str(folder / "rlog.zst")))
  calibration = next(message.extrinsicsCalibration for message in log if message.which() == "extrinsicsCalibration")
  euler = np.array(calibration.rpyCalib, dtype=np.float32)
  device_type = str(next(message.deviceState.deviceType for message in log if message.which() == "deviceState"))
  streams = ("narrowRoadCameraState", "wideRoadCameraState", "cabinCameraState")
  states = {stream: [getattr(message, stream) for message in log if message.which() == stream][:REPLAY_FRAMES] for stream in streams}
  indexes = {}
  for stream in streams:
    kind = stream.replace("CameraState", "EncodeIdx")
    encoded = {getattr(message, kind).frameId: getattr(message, kind) for message in log
               if message.which() == kind and getattr(message, kind).segmentNum == 1}
    indexes[stream] = [encoded[state.frameId] for state in states[stream]]
    if len(indexes[stream]) != REPLAY_FRAMES or any(index.timestampSof != state.timestampSof
        for index, state in zip(indexes[stream], states[stream], strict=True)):
      raise ValueError("Replay image/log correspondence differs")
  cameras = DEVICE_CAMERAS[(device_type, str(states[streams[0]][0].sensor))]
  width, height = cameras.narrow_road.width, cameras.narrow_road.height
  layout = NV12Frame(width, height, width, height, height // 2, width * height * 3 // 2)
  transforms = (get_warp_matrix(euler, cameras.narrow_road.intrinsics, False).astype(np.float32),
                get_warp_matrix(euler, cameras.wide_road.intrinsics, True).astype(np.float32),
                np.linalg.inv(dmonitoringmodel_intrinsics @ np.linalg.inv(_os_fisheye.intrinsics)).astype(np.float32))
  warps = (make_frame_prepare(layout, 512, 256), make_frame_prepare(layout, 512, 256), make_luma_warp(layout, 1440, 960))
  prepared = []
  for stream, name, matrix, warp in zip(streams, ("fcamera.hevc", "ecamera.hevc", "dcamera.hevc"), transforms, warps, strict=True):
    indices = [index.segmentId for index in indexes[stream]]
    raw = frames(folder / name, width, height, max(indices) + 1)[indices]
    transform = Tensor(matrix)
    images = [warp(Tensor(frame), transform).numpy() for frame in raw]
    prepared.append(np.stack(images))
  np.savez_compressed(destination / "camera-inputs.npz", rear=np.stack(prepared[:2], axis=1), driver=prepared[2])
  timestamps = {key: [{"frame_id": state.frameId, "decode_frame_index": index.segmentId,
                       "sensor_sof_ns": state.timestampSof, "sensor_eof_ns": state.timestampEof}
                      for state, index in zip(values, indexes[key], strict=True)] for key, values in states.items()}
  return {"route": ROUTE, "segment": 1, "start_frame": 0, "frames": REPLAY_FRAMES, "source_sha256": ASSETS,
          "device_type": device_type, "image_size": [width, height], "device_from_calib_euler": euler.tolist(),
          "transforms": [matrix.tolist() for matrix in transforms], "timestamps": timestamps,
          "action_t": [[0.2, 0.2]], "traffic_convention": "RHD", "desire": "zero",
          "scope": "Recorded camera numerical replay with fixed RHD/action/desire inputs; not full process or vehicle qualification"}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--reuse-camera-inputs", action="store_true")
  args = parser.parse_args()
  os.environ.setdefault("DEV", "CPU")
  references = ROOT / "android/app/build/model-reference"
  destination = references / "sequences"
  destination.mkdir(exist_ok=True)
  folder = ROOT / "android/build/model-replay"
  folder.mkdir(exist_ok=True)
  with ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(lambda name: fetch(folder, name), ASSETS))
  manifest = {"rtol": 1e-3, "atol": 1e-4, "onnxruntime": ort.__version__, "numpy": np.__version__,
              "vehicle_tx_enabled": False, "models": {}}
  manifest["replay"] = (json.loads((destination / "manifest.json").read_text())["replay"] if args.reuse_camera_inputs else
                         camera_inputs(folder, destination))
  originals = json.loads((references / "manifest.json").read_text())
  for name, metadata in originals["models"].items():
    path = ROOT / f"openpilot/selfdrive/modeld/models/{name}.onnx"
    if digest(path) != metadata["sha256"]:
      raise ValueError("Original model hash changed")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    specs = session.get_outputs()
    manifest["models"][name] = {"sha256": metadata["sha256"]}
    for seed in (*SEEDS, None):
      label = "camera-replay" if seed is None else f"seed{seed}"
      initial = initial_inputs(references, name, seed or 42)
      inputs = {key: value.copy() for key, value in initial.items()}
      images = None
      if seed is None:
        with np.load(destination / "camera-inputs.npz") as source:
          images = source["driver" if name == "dmonitoring_model" else "rear"]
        for key in inputs:
          if key.startswith("state_"):
            inputs[key].fill(0)
        if name == "dmonitoring_model":
          inputs["calib"][:] = manifest["replay"]["device_from_calib_euler"]
        else:
          inputs["desire"].fill(0)
          inputs["action_t"][:] = manifest["replay"]["action_t"]
          inputs["traffic_convention"][:] = [[0., 1.]]
      count = REPLAY_FRAMES if seed is None else STEPS
      expected, hashes = {}, []
      for frame in range(count):
        sequence_input(inputs, initial, frame, images)
        outputs = dict(zip((spec.name for spec in specs), session.run(None, inputs), strict=True))
        if frame == 0 and seed is not None:
          suffix = "" if seed == 42 else f"-case-{seed}"
          with np.load(references / f"{name}{suffix}-outputs.npz") as frozen:
            for key in frozen:
              np.testing.assert_allclose(outputs[key], frozen[key], rtol=1e-3, atol=1e-4)
        hashes.append(output_record(outputs))
        for key, value in outputs.items():
          if value.dtype != np.uint8:
            expected.setdefault(key, []).append(value.copy())
        feedback(inputs, outputs)
      np.savez_compressed(destination / f"{name}-{label}.npz", **{key: np.stack(value) for key, value in expected.items()})
      manifest["models"][name][label] = {"seed": seed or 42, "frames": count, "uint8_outputs": hashes}
      print(name, label, count, flush=True)
  manifest["assets"] = {path.name: digest(path) for path in sorted(destination.glob("*.npz"))}
  (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
  main()
