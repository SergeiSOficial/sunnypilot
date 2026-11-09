"""Diagnostic per-operator accumulation; never selected for live inference."""


def accumulation_diagnostic(runner):
  """Keep FP16 operator boundaries, adding bias before the final FP16 rounding."""
  from tinygrad import dtypes
  original = dict(runner.onnx_ops)

  def conv(x, w, b=None, **options):
    if x.dtype == dtypes.float16:
      return original["Conv"](x.float(), w.float(), None if b is None else b.float(), **options).cast(x.dtype)
    return original["Conv"](x, w, b, **options)

  def gemm(a, b, c=None, **options):
    if a.dtype == dtypes.float16:
      return original["Gemm"](a.float(), b.float(), None if c is None else c.float(), **options).cast(a.dtype)
    return original["Gemm"](a, b, c, **options)

  runner.onnx_ops = {**original, "Conv": conv, "Gemm": gemm}
  return runner
