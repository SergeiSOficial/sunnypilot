"""Deterministic recurrent and recorded-camera numerical checks, never actuation."""
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (42, 7, 123)
STEPS = 32
REPLAY_FRAMES = 60
LABELS = ("seed42", "seed7", "seed123", "camera-replay")


def digest(path):
  with Path(path).open("rb") as source:
    return hashlib.file_digest(source, "sha256").hexdigest()


def initial_inputs(references, name, seed):
  suffix = "" if seed == 42 else f"-case-{seed}"
  with np.load(Path(references) / f"{name}{suffix}-inputs.npz") as source:
    return {key: value.copy() for key, value in source.items()}


def sequence_input(inputs, initial, frame, images=None):
  """Vary images without resetting feedback from the preceding invocation."""
  key = "input_img" if "input_img" in inputs else "new_img"
  inputs[key][:] = np.roll(initial[key], frame * 17, axis=-1) if images is None else images[frame]


def trace_inputs(references, name, seed, frame=None, model=None, progress=lambda: None):
  """Reproduce one diagnostic input without substituting reference feedback."""
  if type(seed) is not int or seed not in SEEDS:
    raise ValueError("Unknown diagnostic seed")
  if frame is not None and (name not in ("dmonitoring_model", "driving_supercombo") or
                            type(frame) is not int or not 0 <= frame < STEPS):
    raise ValueError("Frame tracing requires a valid recurrent-fixture frame")
  initial = initial_inputs(references, name, seed)
  inputs = {key: value.copy() for key, value in initial.items()}
  if frame is not None:
    if name == "driving_supercombo" and model is None:
      raise ValueError("Road frame tracing requires actual backend warmup")
    if any(key.startswith("state_") for key in inputs):
      if name != "driving_supercombo" or model is None:
        raise ValueError("Frame tracing cannot reset recurrent feedback; actual backend warmup required")
      for index in range(frame):
        progress()
        sequence_input(inputs, initial, index)
        feedback(inputs, model(inputs))
    sequence_input(inputs, initial, frame)
  return inputs


def feedback(inputs, outputs):
  if not all(np.isfinite(value).all() for value in outputs.values()):
    raise ValueError("Non-finite sequence output; feedback rejected")
  for key in inputs:
    if key.startswith("state_"):
      value = outputs["next_" + key]
      if value.shape != inputs[key].shape or value.dtype != inputs[key].dtype:
        raise ValueError(f"Invalid recurrent state {key}")
      inputs[key][:] = value


def output_record(outputs):
  # Image history alone is ~2 MB/frame. Exact hashes retain its validation
  # without packaging hundreds of identical shifted image histories.
  return {key: hashlib.sha256(value.tobytes()).hexdigest()
          for key, value in outputs.items() if value.dtype == np.uint8}


def validate(model, references, name, progress=lambda: None, labels=LABELS):
  from fold3_model_trace import compare
  if not labels or len(set(labels)) != len(labels) or any(label not in LABELS for label in labels):
    raise ValueError("Invalid or duplicated recurrent/replay sequence")
  references = Path(references)
  folder = references / "sequences"
  manifest = json.loads((folder / "manifest.json").read_text())
  if manifest.get("rtol") != 1e-3 or manifest.get("atol") != 1e-4:
    raise ValueError("Sequence tolerances differ from original reference")
  for asset, expected in manifest["assets"].items():
    if digest(folder / asset) != expected:
      raise ValueError(f"Sequence asset hash mismatch: {asset}")
  results = {}
  for label in labels:
    metadata = manifest["models"][name][label]
    count = REPLAY_FRAMES if label == "camera-replay" else STEPS
    if metadata["frames"] != count:
      raise ValueError("Incomplete recurrent/replay fixture")
    initial = initial_inputs(references, name, metadata["seed"])
    inputs = {key: value.copy() for key, value in initial.items()}
    with np.load(folder / "camera-inputs.npz") as cameras, np.load(folder / f"{name}-{label}.npz") as expected:
      images = cameras["driver" if name == "dmonitoring_model" else "rear"] if label == "camera-replay" else None
      if images is not None:
        for key in inputs:
          if key.startswith("state_"):
            inputs[key].fill(0)
        if name == "dmonitoring_model":
          inputs["calib"][:] = manifest["replay"]["device_from_calib_euler"]
        else:
          inputs["desire"].fill(0)
          inputs["action_t"][:] = manifest["replay"]["action_t"]
          inputs["traffic_convention"][:] = [[0., 1.]]
      checks = []
      for frame in range(count):
        progress()
        sequence_input(inputs, initial, frame, images)
        actual = model(inputs)
        check = {key: compare(value, expected[key][frame]) for key, value in actual.items() if value.dtype != np.uint8}
        exact = output_record(actual)
        hashes = metadata["uint8_outputs"][frame]
        if set(actual) != set(expected.files) | set(hashes):
          raise ValueError("Sequence output set changed")
        check.update({key: {"passed": exact.get(key) == value, "exact_hash": exact.get(key)} for key, value in hashes.items()})
        checks.append({"frame": frame, "outputs": check})
        feedback(inputs, actual)
      results[label] = {"passed": all(value["passed"] for frame in checks for value in frame["outputs"].values()),
                        "frames": count, "checks": checks}
  return results
