"""Bionic ONNX Runtime using the original ONNX types and weights."""
import numpy as np

import fold3_ort_native


class OrtModel:
  def __init__(self, path, backend="cpu", threads=2, spinning=False, strict=False, library_dir=None,
               assignments=False, no_cpu_fallback=False, provider_options=None, profile_path=None):
    self.session = fold3_ort_native.create(str(path), backend, threads, spinning, strict, library_dir,
                                           assignments, no_cpu_fallback, provider_options or {},
                                           None if profile_path is None else str(profile_path))

  def describe(self):
    return fold3_ort_native.describe(self.session)

  def end_profiling(self):
    return fold3_ort_native.end_profiling(self.session)

  def __call__(self, inputs):
    buffers = {name: np.ascontiguousarray(value) for name, value in inputs.items()}
    return {name: np.frombuffer(data, dtype=dtype).reshape(shape)
            for name, (shape, dtype, data) in fold3_ort_native.run(self.session, buffers).items()}
