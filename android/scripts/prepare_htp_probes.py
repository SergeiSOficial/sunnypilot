#!/usr/bin/env python3
"""Prepare isolated HTP precision/layout probes, independent of model goldens."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

ROOT = Path(__file__).resolve().parents[2]
MODES = ("fp16", "a16w16", "a16w8", "a16w8-symmetric", "a16w8-per-channel")


def quantization_options(mode):
  if mode not in MODES or mode == "fp16":
    raise ValueError("Unsupported quantized HTP probe precision")
  signed = mode in ("a16w8-symmetric", "a16w8-per-channel")
  return {"activation_type": QuantType.QUInt16,
          "weight_type": QuantType.QUInt16 if mode == "a16w16" else QuantType.QInt8 if signed else QuantType.QUInt8,
          "per_channel": mode == "a16w8-per-channel", "weight_symmetric": True if signed else None}


class Reader(CalibrationDataReader):
  def __init__(self, shape):
    self.values = iter({"x": np.random.default_rng(seed).uniform(-.25, .25, shape).astype(np.float32)}
                       for seed in range(1000, 1016))

  def get_next(self):
    return next(self.values, None)


def graph(kind, half=False):
  rng = np.random.default_rng(1000)
  dtype, scalar = (np.float16, T.FLOAT16) if half else (np.float32, T.FLOAT)
  initializers = []
  if kind == "conv":
    shape, out = [1, 3, 8, 8], [1, 4, 6, 6]
    initializers = [numpy_helper.from_array(rng.uniform(-.25, .25, (4, 3, 3, 3)).astype(dtype), "w"),
                    numpy_helper.from_array(rng.uniform(-.01, .01, 4).astype(dtype), "b")]
    nodes = [h.make_node("Conv", ["x", "w", "b"], ["y"], name="probe_conv", kernel_shape=[3, 3])]
  elif kind == "gemm":
    shape, out = [2, 16], [2, 8]
    initializers = [numpy_helper.from_array(rng.uniform(-.25, .25, (8, 16)).astype(dtype), "w"),
                    numpy_helper.from_array(rng.uniform(-.01, .01, 8).astype(dtype), "b")]
    nodes = [h.make_node("Gemm", ["x", "w", "b"], ["y"], name="probe_gemm", transB=1)]
  elif kind == "activation":
    shape = out = [1, 4, 6, 6]
    nodes = [h.make_node("Sigmoid", ["x"], ["s"], name="probe_sigmoid"),
             h.make_node("Mul", ["x", "s"], ["m"], name="probe_silu"),
             h.make_node("Tanh", ["m"], ["y"], name="probe_tanh")]
  elif kind == "layout":
    shape, out = [1, 4, 6, 6], [1, 6, 6, 4]
    nodes = [h.make_node("Transpose", ["x"], ["y"], name="probe_nchw_nhwc", perm=[0, 2, 3, 1])]
  else:
    raise ValueError(kind)
  model = h.make_model(h.make_graph(nodes, "isolated_htp_probe", [h.make_tensor_value_info("x", scalar, shape)],
                                   [h.make_tensor_value_info("y", scalar, out)], initializers),
                       opset_imports=[h.make_opsetid("", 21)], ir_version=10)
  onnx.checker.check_model(model)
  return model, shape


def prepare(destination):
  destination.mkdir(parents=True, exist_ok=True)
  manifest = {"diagnostic_only": True, "vehicle_tx_enabled": False, "calibration_seeds": list(range(1000, 1016)),
              "validation_seed": 2029, "rtol": 1e-3, "atol": 1e-4, "probes": {}}
  options = ort.SessionOptions()
  options.intra_op_num_threads = 2
  for kind in ("conv", "gemm", "activation", "layout"):
    base, shape = graph(kind)
    base_path = destination / f"{kind}-float.onnx"
    onnx.save(base, base_path)
    x = np.random.default_rng(2029).uniform(-.2, .2, shape).astype(np.float32)
    original = ort.InferenceSession(str(base_path), options, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    for mode in MODES:
      key = f"{mode}-{kind}"
      entry = manifest["probes"][key] = {"precision": mode, "operator": kind, "input_shape": shape}
      try:
        path = destination / f"{key}.onnx"
        if mode == "fp16":
          onnx.save(graph(kind, half=True)[0], path)
          inputs = {"x": x.astype(np.float16)}
        else:
          config = get_qnn_qdq_config(base_path, Reader(shape), **quantization_options(mode))
          quantize(base_path, path, config)
          inputs = {"x": x}
        expected = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(None, inputs)[0]
        np.savez(destination / f"{key}.npz", x=inputs["x"], expected=expected, original=original)
        inferred = onnx.shape_inference.infer_shapes(onnx.load(path))
        quantizers = [{"name": n.name, "op": n.op_type, "inputs": list(n.input),
                       "axes": [int(a.i) for a in n.attribute if a.name == "axis"]}
                      for n in inferred.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
        entry.update(status="prepared", sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     data_sha256=hashlib.sha256((destination / f"{key}.npz").read_bytes()).hexdigest(),
                     original_representation_passed=bool(np.allclose(expected, original, rtol=1e-3, atol=1e-4)),
                     quantizers=quantizers,
                     quantization_parameters={v.name: {"dtype": T.DataType.Name(v.data_type), "shape": list(v.dims),
                       "values": numpy_helper.to_array(v).reshape(-1).tolist()}
                       for v in inferred.graph.initializer if v.name.endswith(("_scale", "_zero_point"))},
                     nodes=[{"name": n.name, "op": n.op_type, "domain": n.domain} for n in inferred.graph.node],
                     tensors={v.name: {"dtype": T.DataType.Name(v.type.tensor_type.elem_type),
                                       "shape": [d.dim_value for d in v.type.tensor_type.shape.dim]}
                              for v in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)})
      except Exception as exc:
        entry.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
  (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  return manifest


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--destination", type=Path, default=ROOT / "android/app/build/model-reference/htp-probes")
  args = parser.parse_args()
  report = prepare(args.destination)
  print(json.dumps({k: {f: v[f] for f in ("status", "error", "original_representation_passed") if f in v}
                    for k, v in report["probes"].items()}, indent=2))
