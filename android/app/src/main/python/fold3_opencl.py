"""One OpenCL inference path for numerical qualification and live diagnostics."""
import json
from pathlib import Path


def code_digest(module):
  """Hash the actual imported asset, including Chaquopy's source-free bytecode."""
  import hashlib
  from importlib.util import find_spec
  spec = find_spec(module)
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot identify packaged module {module}")
  return hashlib.sha256(spec.loader.get_data(spec.loader.get_filename(module))).hexdigest()


def fused_ops(runner, diagnostic_mad=False):
  """Preserve the host graph's precision boundaries and fused activations."""
  ops = dict(runner.onnx_ops)

  def host_gemm(a, b, c=None, alpha=1., beta=1., transA=0, transB=0, **options):
    from tinygrad import Tensor, dtypes
    if (a.dtype != dtypes.float16 or transA or not transB or a.shape[0] != 1 or alpha != 1. or beta != 1.
        or options.get("broadcast", 0)):
      return ops["Gemm"](a, b, c, alpha=alpha, beta=beta, transA=transA, transB=transB, **options)
    # Match Eigen's coefficient/dot fallback as well as its larger GEMV path.
    # The coefficient fallback rounds products separately; the scalar dot
    # uses four packet accumulators. GEMV uses eight half FMA lanes.
    k = a.shape[1]
    end = k // 8 * 8
    opencl = a.device.split(":")[0] == "CL"

    def half(value):
      # The Android OpenCL renderer emits scalar/vector RTE casts. Keeping a
      # real CAST also lets the renderer choose the correct vector width.
      return value.cast(dtypes.float16)

    if end and k + 1 + b.shape[0] < 20:
      products = half(b.float() * a.float()).realize()
      lanes = products[:, :8]
      if end > 8:
        other = products[:, 8:16]
        for start in range(16, end // 16 * 16, 16):
          lanes = half(lanes.float() + products[:, start:start + 8].float()).realize()
          other = half(other.float() + products[:, start + 8:start + 16].float()).realize()
        lanes = half(lanes.float() + other.float()).realize()
        if end % 16:
          lanes = half(lanes.float() + products[:, end - 8:end].float()).realize()
    elif end and a.device.split(":")[0] in ("CL", "METAL"):
      from tinygrad import UOp
      from tinygrad.dtype import AddrSpace
      from tinygrad.uop.ops import AxisType, KernelInfo, Ops
      from fold3_host_math import output_range

      packets = min(4, end // 8) if b.shape[0] == 1 else 1

      def accumulate(dest, left, right):
        lane, output_axes = output_range(dest.numel())
        block = UOp.range((end // 8 + packets - 1) // packets, 1, AxisType.LOOP)
        register = UOp.placeholder((1,), dest.dtype, 0, addrspace=AddrSpace.REG)
        register = register.after(lane)[0].set(0)
        index = block * packets * 8 + lane % (packets * 8)
        def scalar(expression, *values, dtype=dtypes.float32):
          return UOp(Ops.CUSTOM, src=values, arg=(expression, dtype))

        x, y, z = (value.cast(dtypes.float32) for value in
                   (left[0, index.minimum(end - 1)].load(), right[lane // (packets * 8), index.minimum(end - 1)].load(),
                    register.after(block)[0].load()))
        # A half product is exact in float32. TwoSum retains the error in the
        # following addition so a float32 midpoint cannot double-round the
        # final half FMA. Opaque scalar steps prevent algebraic cancellation.
        product = scalar("{0} * {1}", x, y)
        total = scalar("{0} + {1}", product, z)
        part = scalar("{0} - {1}", total, product)
        error = scalar("({0} - ({1} - {2})) + ({3} - {2})", product, total, part, z)
        bits = scalar("as_uint({0})" if opencl else "as_type<uint>({0})", total, dtype=dtypes.uint32)
        exponent = scalar("({0} >> 23) & 255u", bits, dtype=dtypes.uint32)
        shift = scalar("(uint)clamp(126 - (int){0}, 13, 24)", exponent, dtype=dtypes.uint32)
        adjusted = scalar("({0} >= 102u && {0} <= 142u && {4} != 0.0f && " +
                          "((({1} & 0x7fffffu) | 0x800000u) & ((1u << {2}) - 1u)) == (1u << ({2} - 1u)))" +
                          " ? nextafter({3}, copysign(INFINITY, {4})) : {3}", exponent, bits, shift, total, error)
        value = (index < end).where(adjusted.cast(dtypes.float16), register.after(block)[0].load())
        register = register[0].set(value, end=block)
        return dest.flatten()[lane].set(register[0], end=output_axes).sink(
          arg=KernelInfo(name=f"host_half_gemm_{b.shape[0]}_{k}", opts_to_apply=()))

      partial = Tensor.empty(b.shape[0], packets, 8, device=a.device, dtype=dtypes.float16).custom_kernel(a, b, fxn=accumulate)[0]
      lanes = partial[:, packets - 1]
      for packet in range(packets - 2, -1, -1):
        lanes = half(partial[:, packet].float() + lanes.float()).realize()
    else:
      lanes = Tensor.zeros(b.shape[0], 8, device=a.device, dtype=dtypes.float16)
      for start in range(0, end, 8):
        lanes = (b[:, start:start + 8].float() * a[:, start:start + 8].float() + lanes.float()).cast(dtypes.float16)
    for width in (4, 2, 1):
      lanes = half(lanes[:, ::2].float() + lanes[:, 1::2].float()).reshape(b.shape[0], width)
    value = lanes.reshape(1, b.shape[0])
    for index in range(end, k):
      product = half(b[:, index].float() * a[0, index].float())
      value = half(value.float() + product.float())
    return value if c is None else half(value.float() + c.float())

  def activate(value, activation=None, activation_params=()):
    if activation_params:
      raise ValueError("Unsupported fused activation parameters")
    if isinstance(activation, bytes):
      activation = activation.decode()
    if activation is None:
      return value
    return {"Relu": value.relu, "Sigmoid": value.sigmoid, "Tanh": value.tanh}[activation]()

  def conv(x, w, b=None, residual=None, activation=None, activation_params=(), **options):
    value = ops["Conv"](x, w, b, **options)
    if residual is not None:
      value = value + residual
    return activate(value, activation, activation_params)

  def nhwc(x, w, b=None, residual=None, **options):
    value = conv(x.permute(0, 3, 1, 2), w, b, None if residual is None else residual.permute(0, 3, 1, 2), **options)
    return value.permute(0, 2, 3, 1)

  def gemm(a, b, c=None, activation=None, activation_params=(), **options):
    return activate(host_gemm(a, b, c, **options), activation, activation_params)

  runner.onnx_ops = {**ops, "Gemm": host_gemm, "FusedConv": conv, "NhwcFusedConv": nhwc, "FusedGemm": gemm,
                     "QuickGelu": lambda x, alpha=1.: x * (x * alpha).sigmoid()}
  from fold3_host_math import install
  install(runner, diagnostic_mad=diagnostic_mad)


class OpenCLModel:
  def __init__(self, path, host_fusion=False, image_buffers=0, beam=0, device="CL", diagnostic_mad=False):
    from tinygrad import Tensor, TinyJit
    from tinygrad.nn.onnx import OnnxRunner
    self.device = device
    self.runner = OnnxRunner(Tensor(Path(path).read_bytes(), device="PYTHON")).to(device)
    if host_fusion:
      fused_ops(self.runner, diagnostic_mad=diagnostic_mad)

    @TinyJit
    def run(**values):
      outputs = self.runner(values)
      Tensor.realize(*outputs.values())
      return outputs
    def execute(**values):
      from tinygrad.helpers import Context
      with Context(IMAGE=image_buffers, FLOAT16=0, BEAM=beam):
        return run(**values)
    self.run_tensors = execute

  def __call__(self, inputs):
    from tinygrad import Tensor
    outputs = self.run_tensors(**{key: Tensor(value, device=self.device).realize() for key, value in inputs.items()})
    return {key: value.numpy() for key, value in outputs.items()}


def identity(name, path, references):
  from fold3_sequences import digest
  from java import jclass
  paths = {"model": path, "original_references": references / "manifest.json",
           "sequences": references / "sequences/manifest.json",
           "driver": Path("/vendor/lib64/libOpenCL.so"),
           "apk": Path(str(jclass("ai.sunnypilot.fold3.AndroidHardware").applicationApk()))}
  modules = {"runner": __name__, "host_math": "fold3_host_math", "tinygrad": "tinygrad.nn.onnx", "uops": "tinygrad.uop.ops",
             "renderer": "tinygrad.renderer.cstyle", "compiler": "tinygrad.runtime.ops_cl"}
  return {"name": name, "backend": "CL", "cpu_fallback": False,
          "fingerprint": str(jclass("android.os.Build").FINGERPRINT),
          "sha256": {**{key: digest(value) for key, value in paths.items()},
                     **{key: code_digest(value) for key, value in modules.items()}}}


def qualified(entry):
  """All original seeds, all recurrent frames and the full replay are mandatory."""
  if not isinstance(entry, dict):
    return False
  identity = entry.get("identity", {})
  if (identity.get("backend") != "CL" or identity.get("cpu_fallback") is not False or
      not identity.get("fingerprint") or len(identity.get("sha256", {}).get("apk", "")) != 64):
    return False
  if entry.get("status") != "passed" or entry.get("performance_status") != "passed":
    return False
  if entry.get("samples") != 100 or entry.get("deadline_misses_50ms") != 0:
    return False
  if not isinstance(entry.get("work_p99_ms"), (int, float)) or not 0 <= entry["work_p99_ms"] < 50:
    return False
  return original_reference_passed(entry)


def original_reference_passed(entry):
  """Numerical evidence only; this cannot qualify a live backend."""
  if not isinstance(entry, dict):
    return False
  outputs = set(entry.get("parity", {}))
  if not outputs:
    return False
  seeds = entry.get("additional_seed_parity", {})
  def passed(value):
    return value.get("passed") is True and value.get("unexpected_nonfinite_elements", 0) == 0
  if set(seeds) != {"7", "123"}:
    return False
  for case in (entry.get("parity", {}), *seeds.values()):
    if set(case) != outputs or not all(passed(value) for value in case.values()):
      return False
  sequences = entry.get("sequences", {})
  if set(sequences) != {"seed42", "seed7", "seed123", "camera-replay"}:
    return False
  for key, sequence in sequences.items():
    count = 60 if key == "camera-replay" else 32
    checks = sequence.get("checks", [])
    if (sequence.get("passed") is not True or sequence.get("frames") != count or len(checks) != count
        or [check.get("frame") for check in checks] != list(range(count))
        or any(set(check.get("outputs", {})) != outputs or any(not passed(value) for value in check["outputs"].values()) for check in checks)):
      return False
  return True


def live_model(data_dir, name, original):
  """No stale or incomplete benchmark can enable the OpenCL live path."""
  from fold3_ort import OrtModel
  if (Path(data_dir) / f"accel-qualification-{name}.json").is_file():
    from java import jclass
    from fold3_accel import live_model as accelerator_model
    candidate = accelerator_model(data_dir, name, str(jclass("ai.sunnypilot.fold3.AndroidHardware").nativeLibraries()))
    if candidate is not None:
      return candidate
  references = Path(__file__).parent / "fold3_reference"
  try:
    report = json.loads((Path(data_dir) / "model-qualification.json").read_text())
    if report.get("backend") != "CL" or report.get("profile") not in ("baseline", "host-fusion-cl"):
      raise ValueError("Unqualified OpenCL profile")
    entry = report["models"][name]
    fused = report["profile"] == "host-fusion-cl"
    path = references / f"host-fusion/{name}.onnx" if fused else original
    if not qualified(entry) or entry["identity"] != identity(name, path, references):
      raise ValueError("Numerical qualification or backend identity differs")
    return OpenCLModel(path, host_fusion=fused), "OpenCL: original reference, recurrent and camera replay qualified"
  except (OSError, ValueError, KeyError, TypeError, AttributeError):
    return OrtModel(original, "cpu", threads=2), "ORT CPU, 2 threads; OpenCL qualification blocked"
