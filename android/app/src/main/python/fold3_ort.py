"""Bionic ONNX Runtime using the original ONNX types and weights."""
import numpy as np

import fold3_ort_native


class OrtModel:
  def __init__(self, path, backend="cpu", threads=2, spinning=False, strict=False):
    self.session = fold3_ort_native.create(str(path), backend, threads, spinning, strict)

  def __call__(self, inputs):
    buffers = {name: np.ascontiguousarray(value) for name, value in inputs.items()}
    return {name: np.frombuffer(data, dtype=dtype).reshape(shape)
            for name, (shape, dtype, data) in fold3_ort_native.run(self.session, buffers).items()}
