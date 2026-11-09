#!/usr/bin/env python3
"""Compare debug Camera2 lens references with simultaneous Samsung output clips."""
import argparse
import json
from pathlib import Path
import struct

import cv2
import numpy as np

from validate_cameras import MAX_SKEW_NS


def load_clip(path, width, height):
  if min(width, height) < 2 or width % 2 or height % 2:
    raise ValueError("Invalid NV12 dimensions")
  luma_bytes = width * height
  size = luma_bytes * 3 // 2
  frames = []
  with path.open("rb") as stream:
    while header := stream.read(8):
      data = stream.read(size)
      if len(header) != 8 or len(data) != size:
        raise ValueError(f"Truncated evidence clip {path}")
      timestamp, = struct.unpack(">q", header)
      if timestamp <= (frames[-1][0] if frames else 0):
        raise ValueError(f"Nonmonotonic timestamps in {path}")
      frames.append((timestamp, np.frombuffer(data[:luma_bytes], np.uint8).reshape(height, width)))
  if not frames:
    raise ValueError(f"Empty evidence clip {path}")
  return frames


def features(image):
  height, width = image.shape
  scale = 960 / max(width, height)
  if scale != 1:
    image = cv2.resize(image, None, fx=scale, fy=scale)
  # SIFT handles rotation and uniform scale. Anisotropic stretching is tested separately.
  return image.shape, *cv2.SIFT_create(nfeatures=1200).detectAndCompute(image, None)


def quarter_turn_same_aspect(image):
  # Test the vendor preview's rotated raster as it would appear in a landscape surface.
  height, width = image.shape
  return cv2.resize(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), (width, height))


def compare(reference, candidate):
  _, src_keys, src_desc = reference
  (height, width), dst_keys, dst_desc = candidate
  if src_desc is None or dst_desc is None or min(len(src_desc), len(dst_desc)) < 2:
    return {"matches": 0, "inliers": 0}
  matcher = cv2.BFMatcher(cv2.NORM_L2)

  def matches(a, b):
    return {first.queryIdx: first.trainIdx for first, second in matcher.knnMatch(a, b, k=2)
            if first.distance < .72 * second.distance}

  forward, reverse = matches(src_desc, dst_desc), matches(dst_desc, src_desc)
  pairs = [(a, b) for a, b in forward.items() if reverse.get(b) == a]
  result = {"matches": len(pairs), "inliers": 0}
  if len(pairs) < 12:
    return result
  src = np.float32([src_keys[a].pt for a, _ in pairs])
  dst = np.float32([dst_keys[b].pt for _, b in pairs])
  matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3)
  if matrix is None:
    return result
  inliers = mask.ravel().astype(bool)
  projected = cv2.perspectiveTransform(src[None, :, :], matrix)[0]
  residual = np.linalg.norm(projected - dst, axis=1)
  cells = (dst * [4 / width, 4 / height]).astype(int).clip(0, 3)
  # Region counts and residuals expose mixed projections that a global fit can hide.
  grid = []
  for y in range(4):
    for x in range(4):
      selected = (cells == [x, y]).all(axis=1)
      grid.append({"matches": int(selected.sum()), "inliers": int((selected & inliers).sum()),
                   "median_residual_px": float(np.median(residual[selected])) if selected.any() else None})
  return {"matches": len(pairs), "inliers": int(inliers.sum()), "candidate_grid": grid}


def read_reference(report_path, sensor):
  report = json.loads(report_path.read_text())
  clip = report["evidence"]["direct_clips"][sensor]
  if not report["attempts"][0]["streams"][0]["sensor_identity_evidence"]:
    raise ValueError(f"Reference lacks Camera2 sensor identity {report_path}")
  frames = load_clip(Path(clip["file"]), clip["width"], clip["height"])
  return features(frames[len(frames) // 2][1])


def trial(root, index):
  refs = {"wide": read_reference(root / "wide-reference.json", "0"),
          "ultrawide": read_reference(root / "ultrawide-reference.json", "2")}
  preview, thumbnail = [load_clip(root / f"samsung-{index}" / f"{stream}.clip", 960, 540)
                        for stream in ("preview", "thumbnail")]
  receipt = json.loads((root / f"samsung-{index}.json").read_text())
  attempt = receipt["attempts"][0]
  physical_results = sorted(attempt.get("capture_metadata", {}).get("physical_results", []))
  # The samples in the clips use a common clock but do not carry a physical result assignment.
  pairs = receipt["evidence"].get("timestamps_ns", [])
  sample_pairs = pairs
  if not sample_pairs:
    # A stationary scene can still be inspected. This pair never counts for timing.
    sample_pairs = [min(((a, b) for a, _ in preview for b, _ in thumbnail), key=lambda pair: abs(pair[0] - pair[1]))]
  preview, thumbnail = dict(preview), dict(thumbnail)
  comparisons = []
  for i in np.linspace(0, len(sample_pairs) - 1, min(3, len(sample_pairs)), dtype=int):
    a, b = sample_pairs[i]
    if pairs and abs(a - b) > MAX_SKEW_NS:
      raise ValueError("Evidence pair exceeds synchronization limit")
    row = {"capture_times_ns": [a, b], "paired_for_timing": bool(pairs)}
    for name, frame in (("preview", preview[a]), ("thumbnail", thumbnail[b])):
      candidate = features(frame)
      row[name] = {lens: compare(reference, candidate) for lens, reference in refs.items()}
    candidate = features(quarter_turn_same_aspect(preview[a]))
    row["preview_quarter_turn_same_aspect"] = {lens: compare(reference, candidate) for lens, reference in refs.items()}
    comparisons.append(row)
  return {"trial": index, "paired_samples": len(pairs), "physical_result_ids": physical_results,
          "reference_cross_check": compare(refs["wide"], refs["ultrawide"]),
          "comparisons": comparisons, "identity_status": "blocked",
          "reason": "Optical matches are diagnostic; no per-output physical timestamp assignment"}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("trial_directory", type=Path)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()
  rows = []
  for i in (1, 2, 3):
    try:
      rows.append(trial(args.trial_directory, i))
    except (OSError, KeyError, IndexError, ValueError, cv2.error) as exc:
      rows.append({"trial": i, "identity_status": "blocked", "reason": str(exc)})
  args.output.write_text(json.dumps({"status": "blocked", "trials": rows}, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
  main()
