"""Hash-checked driver replay inputs for native diagnostics, never qualification."""
import hashlib
import json
from pathlib import Path

import numpy as np


def replay_case(references, frame):
  if type(frame) is not int or not 0 <= frame < 60:
    raise ValueError("Invalid driver replay frame")
  base = Path(references)
  folder = base / "sequences"
  manifest = json.loads((folder / "manifest.json").read_text())
  if manifest.get("rtol") != 1e-3 or manifest.get("atol") != 1e-4:
    raise ValueError("Replay tolerances changed")
  for name in ("camera-inputs.npz", "dmonitoring_model-camera-replay.npz"):
    with (folder / name).open("rb") as stream:
      if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["assets"][name]:
        raise ValueError("Replay asset changed: " + name)
  metadata = manifest["models"]["dmonitoring_model"]["camera-replay"]
  if metadata["frames"] != 60:
    raise ValueError("Incomplete driver replay")
  suffix = "" if metadata["seed"] == 42 else "-case-" + str(metadata["seed"])
  with np.load(base / ("dmonitoring_model" + suffix + "-inputs.npz")) as saved:
    inputs = {key: value.copy() for key, value in saved.items()}
  if set(inputs) != {"input_img", "calib"}:
    raise ValueError("Driver replay contract changed")
  with np.load(folder / "camera-inputs.npz") as saved:
    inputs["input_img"][:] = saved["driver"][frame]
  inputs["calib"][:] = manifest["replay"]["device_from_calib_euler"]
  with np.load(folder / "dmonitoring_model-camera-replay.npz") as saved:
    golden = saved["outputs"][frame].copy()
  return inputs, golden
