#!/usr/bin/env python3
"""Convert isolated FP32 operators without changing original model assets."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort

from prepare_htp_probes import graph
from prepare_accel_models import ROOT, digest, ncnn_contract


def prepare(destination):
  destination.mkdir(parents=True, exist_ok=True)
  tools = ROOT / "android/build/native/accelerators"
  report = {"diagnostic_only": True, "vehicle_tx_enabled": False, "probes": {}}
  for kind in ("conv", "gemm", "activation", "layout"):
    model, shape = graph(kind)
    path = destination / f"{kind}.onnx"
    onnx.save(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(2029).uniform(-.2, .2, shape).astype(np.float32)
    expected = session.run(None, {"x": x})[0]
    np.savez(destination / f"{kind}.npz", x=x, expected=expected)
    entry = report["probes"][kind] = {"source_sha256": digest(path), "data_sha256": digest(destination / f"{kind}.npz"),
       "inputs": {"x": {"shape": shape, "type": "tensor(float)"}},
       "outputs": {"y": {"shape": list(expected.shape), "type": "tensor(float)"}}}
    for engine in ("mnn", "ncnn"):
      folder = destination / engine
      folder.mkdir(exist_ok=True)
      output = folder / (kind + (".mnn" if engine == "mnn" else ".ncnn.param"))
      if engine == "mnn":
        code = "import sys;sys.path.insert(0,sys.argv[1]);import _tools;_tools.mnnconvert(['MNNConvert','-f','ONNX','--modelFile',sys.argv[2],'--MNNModel',sys.argv[3],'--bizCode','fold3'])"
        command = [sys.executable, "-c", code, str(tools / "mnn-python"), str(path), str(output)]
      else:
        command = [str(tools / "pnnx/pnnx/pnnx"), str(path), "fp16=0",
                   f"inputshape=[{','.join(map(str, shape))}]f32", f"ncnnparam={output}",
                   f"ncnnbin={output.with_suffix('.bin')}", f"pnnxparam={folder / (kind + '.pnnx.param')}",
                   f"pnnxbin={folder / (kind + '.pnnx.bin')}", f"pnnxpy={folder / (kind + '_pnnx.py')}",
                   f"ncnnpy={folder / (kind + '_ncnn.py')}"]
      with (folder / f"{kind}-conversion.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=folder, timeout=60)
      entry[engine] = {"exit_code": result.returncode, "status": "converted" if result.returncode == 0 and output.is_file() else "blocked",
                       "sha256": digest(output) if output.is_file() else None}
      if engine == "ncnn" and output.with_suffix(".bin").is_file():
        entry[engine]["weights_sha256"] = digest(output.with_suffix(".bin"))
        entry[engine].update(ncnn_contract(folder / (kind + "_ncnn.py"), entry["inputs"], entry["outputs"]))
  (destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
  prepare(ROOT / "android/app/build/model-reference/gpu-probes")
