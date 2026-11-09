"""Pinned arm64 host arithmetic for the host-fused OpenCL graph.

These operators preserve the reference's accumulation and rounding boundaries.
Qualification still requires the original goldens, recurrent replay and deadlines.
"""
from math import prod


def output_range(size, max_lanes=32):
  """Schedule scalar outputs together; preserve the ordered arithmetic loops."""
  from tinygrad import UOp
  from tinygrad.uop.ops import AxisType
  lanes = next(width for width in (128, 64, 32, 16, 8, 4, 2, 1) if width <= max_lanes and size % width == 0)
  group = UOp.range(size // lanes, 0, AxisType.GLOBAL)
  local = UOp.range(lanes, 100, AxisType.LOCAL)
  return group * lanes + local, (group, local)


def rounded_sqrt(value, metal):
  """Round positive normal FP32 roots using exact integer midpoint comparisons."""
  from tinygrad import UOp, dtypes
  from tinygrad.uop.ops import Ops

  def native(x):
    return UOp(Ops.CUSTOM, src=(x,), arg=("precise::sqrt({0})" if metal else "sqrt({0})", dtypes.float32))

  bits = value.bitcast(dtypes.uint32)
  exponent = (bits >> 23) & 255
  normal = (value > 0) & (exponent > 0) & (exponent < 255)
  # Other IEEE cases retain the platform result. Use 1.0 for their integer
  # calculation so zero, subnormals and NaNs cannot produce invalid shifts.
  bits = normal.where(bits, UOp.const(0x3f800000, dtypes.uint32))
  exponent = bits >> 23
  expected_exponent = (exponent + 127) >> 1
  radicand = ((bits & 0x7fffff) | 0x800000).cast(dtypes.uint64) << (23 + ((exponent - 127) & 1))
  result = native(bits.bitcast(dtypes.float32))
  # The CL sqrt error bound is three ulps. All integer products below fit
  # in 52 bits; no floating multiply or approximate reciprocal decides a step.
  for _ in range(4):
    encoded = result.bitcast(dtypes.uint32)
    significand = ((encoded & 0x7fffff) | 0x800000).cast(dtypes.uint64) << ((encoded >> 23) + 1 - expected_exponent)
    up = 4 * radicand > significand * significand
    neighbor = up.where(encoded + 1, encoded - 1)
    adjacent = ((neighbor & 0x7fffff) | 0x800000).cast(dtypes.uint64) << ((neighbor >> 23) + 1 - expected_exponent)
    midpoint = (significand + adjacent) * (significand + adjacent)
    move = up.where(16 * radicand > midpoint, 16 * radicand < midpoint)
    result = move.where(neighbor, encoded).bitcast(dtypes.float32)
  return normal.where(result, native(value))


def install(runner, diagnostic_mad=False):
  from tinygrad import Tensor, UOp, dtypes
  from tinygrad.dtype import AddrSpace
  from tinygrad.uop.ops import AxisType, KernelInfo, Ops

  ops = dict(runner.onnx_ops)

  def scalar(expression, *values):
    values = tuple(value if isinstance(value, UOp) else UOp.const(value, dtypes.float32) for value in values)
    return UOp(Ops.CUSTOM, src=values, arg=(expression, dtypes.float32))

  def divide(left, right, metal):
    # Adreno does not advertise correctly rounded division. A residual update
    # alone still rounds the replay's midpoint case incorrectly: classify the
    # exact residual against half the adjacent-float gap with FMA instead.
    # Keep IEEE exceptional results;
    # fma(-inf, 0, x) must not turn a valid infinite quotient into a NaN.
    expression = "precise::divide({0}, {1})" if metal else "({0} / {1})"
    quotient = scalar(expression, left, right)
    residual = scalar("fma(-{0}, {1}, {2})", quotient, right, left)
    corrected = scalar("{0} + {1}", quotient, scalar(expression, residual, right))
    magnitude = scalar("fabs({0})", corrected)
    denominator = scalar("fabs({0})", right)
    residual = scalar("fma(-{0}, {1}, fabs({2}))", magnitude, denominator, left)
    neighbor = scalar("nextafter({0}, {1} < 0.0f ? 0.0f : INFINITY)", magnitude, residual)
    half_gap = scalar("fabs({0} - {1}) * 0.5f", neighbor, magnitude)
    distance = scalar("fma(-{0}, {1}, fabs({2}))", half_gap, denominator, residual)
    bits = "as_type<uint>({0})" if metal else "as_uint({0})"
    rounded = scalar("({2} > 0.0f || ({2} == 0.0f && (" + bits + " & 1U))) ? {1} : {0}",
                     magnitude, neighbor, distance)
    corrected = scalar("({0} > 0.0f && isfinite({1})) ? copysign({2}, {3}) : {3}",
                       half_gap, neighbor, rounded, corrected)
    return scalar("(isfinite({0}) && isfinite({1}) && isfinite({2}) && {1} != 0.0f) ? {3} : {2}",
                  left, right, quotient, corrected)

  def division(a, b):
    if a.dtype != b.dtype or a.dtype != dtypes.float32:
      return a / b
    rank = max(a.ndim, b.ndim)
    a = a.reshape((1,) * (rank - a.ndim) + a.shape)
    b = b.reshape((1,) * (rank - b.ndim) + b.shape)
    if any(left != right and left != 1 and right != 1 for left, right in zip(a.shape, b.shape, strict=True)):
      raise ValueError("Incompatible division dimensions")
    shape = tuple(max(left, right) for left, right in zip(a.shape, b.shape, strict=True))
    scalar_a, scalar_b = a.numel() == 1, b.numel() == 1
    a = (a if scalar_a else a.expand(shape)).contiguous().realize()
    b = (b if scalar_b else b.expand(shape)).contiguous().realize()
    # A scalar constant may remain CONST after contiguous(). Custom kernels
    # require owned buffer inputs; staging also avoids host scalar readback.
    if scalar_a:
      a = Tensor.empty(*a.shape, device=a.device, dtype=a.dtype).assign(a).realize()
    if scalar_b:
      b = Tensor.empty(*b.shape, device=b.device, dtype=b.dtype).assign(b).realize()
    metal = a.device.split(":")[0] == "METAL"

    def kernel(dest, left, right):
      index, output_axes = output_range(dest.numel())
      value = divide(left.flatten()[0 if scalar_a else index].load(), right.flatten()[0 if scalar_b else index].load(), metal)
      return dest.flatten()[index].set(value, end=output_axes).sink(arg=KernelInfo(name="host_divide", opts_to_apply=()))
    return Tensor.empty(*shape, device=a.device, dtype=a.dtype).custom_kernel(a, b, fxn=kernel)[0]

  def activation(x, kind):
    source = x.float().contiguous().realize()
    metal = x.device.split(":")[0] == "METAL"
    multiply_add = "mad" if diagnostic_mad and x.device.split(":")[0] == "CL" else "fma"

    def kernel(dest, source):
      index, output_axes = output_range(dest.numel())
      limit = 9. if kind == "Tanh" else 18.
      value = source.flatten()[index].load().maximum(-limit).minimum(limit)
      square = scalar("{0} * {0}", value)
      # ONNX Runtime 1.31.1 MLAS tanh.cpp/logistic.cpp (MIT).
      if kind == "Tanh":
        numerator = (-2.76076847742355e-16, 2.00018790482477e-13, -8.60467152213735e-11,
                     5.12229709037114e-08, 1.48572235717979e-05, 6.37261928875436e-04, 4.89352455891786e-03)
        denominator = (1.19825839466702e-06, 1.18534705686654e-04, 2.26843463243900e-03, 4.89352518554385e-03)
      else:
        numerator = (4.37031012579801e-11, 1.15627324459942e-07, 6.08574864600143e-05,
                     8.51377133304701e-03, 2.48287947061529e-01)
        denominator = (6.10247389755681e-13, 5.76102136993427e-09, 6.29106785017040e-06,
                       1.70198817374094e-03, 1.16817656904453e-01, 9.93151921023180e-01)

      def polynomial(coefficients):
        result = UOp.const(coefficients[0], dtypes.float32)
        for coefficient in coefficients[1:]:
          result = scalar(multiply_add + "({0}, {1}, {2})", result, square, coefficient)
        return result
      result = divide(scalar("{0} * {1}", polynomial(numerator), value), polynomial(denominator), metal)
      if kind != "Tanh":
        result = scalar("{0} + 0.5f", result).maximum(0.).minimum(1.)
      return dest.flatten()[index].set(result, end=output_axes).sink(arg=KernelInfo(name="host_" + kind, opts_to_apply=()))
    return Tensor.empty(*x.shape, device=x.device, dtype=dtypes.float32).custom_kernel(source, fxn=kernel)[0].cast(x.dtype)

  def exponential(x):
    if x.dtype != dtypes.float32:
      return ops["Exp"](x)
    source = x.contiguous().realize()

    def kernel(dest, values):
      index, output_axes = output_range(dest.numel())
      original = values.flatten()[index].load()
      # ONNX Runtime 1.31.1 MLAS compute.cpp (MIT), including range reduction.
      value = scalar("clamp({0}, -103.9720840454f, 88.7762626647950f)", original)
      biased = scalar("fma({0}, 1.44269504088896341f, 12582912.0f)", value)
      exponent = scalar("{0} - 12582912.0f", biased)
      value = scalar("fma({0}, -6.93145752e-1f, {1})", exponent, value)
      value = scalar("fma({0}, -1.42860677e-6f, {1})", exponent, value)
      overflow = biased.bitcast(dtypes.uint32) << 23
      normal = overflow.bitcast(dtypes.int32).maximum(-1056964608).minimum(1065353216).bitcast(dtypes.uint32)
      overflow = (overflow - normal + 1065353216).bitcast(dtypes.float32)
      normal = (normal + 1065353216).bitcast(dtypes.float32)
      coefficients = ("0x1.694000p-10", "0x1.125edcp-7", "0x1.555b5ap-5", "0x1.555450p-3", "0x1.fffff6p-2", "0x1p+0")
      polynomial = UOp.const(float.fromhex(coefficients[0]), dtypes.float32)
      for coefficient in coefficients[1:]:
        polynomial = scalar("fma({0}, {1}, {2})", polynomial, value, float.fromhex(coefficient))
      value = scalar("{0} * {1}", value, overflow)
      result = scalar("{0} * {1}", scalar("fma({0}, {1}, {2})", polynomial, value, overflow), normal)
      result = scalar("isnan({0}) ? {0} : {1}", original, result)
      return dest.flatten()[index].set(result, end=output_axes).sink(arg=KernelInfo(name="host_exp", opts_to_apply=()))

    return Tensor.empty(*x.shape, device=x.device, dtype=x.dtype).custom_kernel(source, fxn=kernel)[0]

  def activate(value, kind=None, parameters=()):
    if parameters:
      raise ValueError("Unsupported fused activation parameters")
    if isinstance(kind, bytes):
      kind = kind.decode()
    if kind is None:
      return value
    if kind == "Relu":
      return value.relu()
    if kind not in ("Tanh", "Sigmoid"):
      raise ValueError(f"Unsupported fused activation {kind}")
    return activation(value, kind)

  def nhwc(x, w, b=None, residual=None, activation=None, activation_params=(), strides=(1, 1), pads=(0, 0, 0, 0),
           dilations=(1, 1), group=1, auto_pad="NOTSET", kernel_shape=None, bias_first=True):
    if x.dtype != dtypes.float32 or w.dtype != dtypes.float32 or auto_pad not in ("NOTSET", b"NOTSET"):
      return ops["NhwcFusedConv"](x, w, b, residual, activation=activation, activation_params=activation_params,
                                  strides=strides, pads=pads, dilations=dilations, group=group, auto_pad=auto_pad,
                                  kernel_shape=kernel_shape)
    n, ih, iw, ic = x.shape
    oc, icg, kh, kw = w.shape
    if ic != icg * group or oc % group:
      raise ValueError("Convolution groups do not match input and weights")
    pt, pl, pb, pr = pads
    sh, sw = strides
    dh, dw = dilations
    oh, ow = (ih + pt + pb - dh * (kh - 1) - 1) // sh + 1, (iw + pl + pr - dw * (kw - 1) - 1) // sw + 1
    padded = x.pad(((0, 0), (pt, pb), (pl, pr), (0, 0))).contiguous()
    bias = Tensor.zeros(oc, device=x.device, dtype=x.dtype) if b is None else b
    # Adjacent lanes compute adjacent output channels. Keep their weight loads
    # adjacent too, without reordering the per-output accumulation.
    weights = w.permute(2, 3, 1, 0).contiguous().realize()
    tile = 4 if diagnostic_mad and oc // group % 4 == 0 else 1
    # Adreno does not advertise correctly rounded FMA. Measure its native
    # multiply-add separately; original-reference/recurrent gates still apply.
    multiply_add = "mad" if diagnostic_mad and x.device.split(":")[0] == "CL" else "fma"

    def accumulate(dest, left, right, bias):
      # Event profiles put ordered convolutions first. More independent lanes
      # amortize workgroup scheduling without changing any output's FMA order.
      index, output_axes = output_range(dest.numel() // tile, max_lanes=128)
      flat = index * tile
      k = UOp.range(icg * kh * kw, 1, AxisType.LOOP)
      c, px, py, batch = flat % oc, flat // oc % ow, flat // oc // ow % oh, flat // oc // ow // oh
      ky, kx, ci = k // icg // kw, k // icg % kw, k % icg
      # Independent accumulators share an input load without reordering FMAs.
      reg = UOp.placeholder((tile,), dtypes.float32, 0, addrspace=AddrSpace.REG)
      reg = reg.after(index).after(*[reg[i].store(bias[c + i].load() if bias_first else 0.) for i in range(tile)])
      a = left[batch, py * sh + ky * dh, px * sw + kx * dw, c // (oc // group) * icg + ci].load()
      old = reg.after(k)
      reg = reg.after(UOp.group(*[reg[i].store(scalar(multiply_add + "({0}, {1}, {2})", a,
                      right[ky, kx, ci, c + i].load(), old[i].load())) for i in range(tile)]).end(k))
      return UOp.group(*[dest.flatten()[flat + i].store(reg[i].load()) for i in range(tile)]).end(*output_axes).sink(
        arg=KernelInfo(name="host_conv", opts_to_apply=()))
    value = Tensor.empty(n, oh, ow, oc, device=x.device, dtype=x.dtype).custom_kernel(padded, weights, bias, fxn=accumulate)[0]
    # MLAS preloads residuals into GEMM output before its bias/activation pass.
    if residual is not None:
      value = (value + residual).realize()
    if not bias_first:
      value = (value + bias.reshape(1, 1, 1, oc)).realize()
    return activate(value, activation, activation_params)

  def conv(x, w, b=None, residual=None, **options):
    if x.dtype != dtypes.float32 or w.dtype != dtypes.float32:
      return ops["FusedConv"](x, w, b, residual, **options)
    return nhwc(x.permute(0, 2, 3, 1), w, b, None if residual is None else residual.permute(0, 2, 3, 1),
                bias_first=False, **options).permute(0, 3, 1, 2)

  def matmul(a, b):
    if a.ndim < 2 or b.ndim < 2:
      return ops["MatMul"](a, b)
    # ONNX batch broadcasting, followed by explicit row-major staging.
    left, right = a.shape[:-2], b.shape[:-2]
    size = max(len(left), len(right))
    left, right = (1,) * (size - len(left)) + left, (1,) * (size - len(right)) + right
    if any(x != y and x != 1 and y != 1 for x, y in zip(left, right, strict=True)):
      raise ValueError("Incompatible matrix batch dimensions")
    batch = tuple(max(x, y) for x, y in zip(left, right, strict=True))
    m, k = a.shape[-2:]
    n = b.shape[-1]
    if b.shape[-2] != k:
      raise ValueError("Mismatched dot-product width")
    a = a.reshape(*left, m, k).expand(*batch, m, k).contiguous().realize()
    b = b.reshape(*right, k, n).expand(*batch, k, n).contiguous().realize()

    def kernel(dest, left, right):
      index, output_axes = output_range(dest.numel())
      column, row, group = index % n, index // n % m, index // n // m
      reg = UOp.placeholder((1,), dtypes.float32, 0, addrspace=AddrSpace.REG)
      reg = reg.after(index)[0].set(0.)

      def product(step):
        x = left.flatten()[group * m * k + row * k + step].load().cast(dtypes.float32)
        y = right.flatten()[group * k * n + step * n + column].load().cast(dtypes.float32)
        return x, y

      if a.dtype == dtypes.float16:
        # Batched half MatMul uses the same SME2 FDOT rounding as NoTrans
        # Gemm, including the odd K tail in the model's attention values.
        # Half products are exact in FP32. Separate multiply/add avoids
        # software FMA while preserving each pair and accumulator rounding.
        if k >= 2:
          step = UOp.range(k // 2, 1, AxisType.LOOP)
          pair = scalar("{0} + {1}", scalar("{0} * {1}", *product(step * 2)), scalar("{0} * {1}", *product(step * 2 + 1)))
          reg = reg[0].set(scalar("{0} + {1}", pair, reg.after(step)[0].load()), end=step)
        value = reg[0].load()
        if k % 2:
          value = scalar("{0} + {1}", scalar("{0} * {1}", *product(k - 1)), value)
      else:
        step = UOp.range(k, 1, AxisType.LOOP)
        reg = reg[0].set(scalar("fma({0}, {1}, {2})", *product(step), reg.after(step)[0].load()), end=step)
        value = reg[0].load()
      return dest.flatten()[index].set(value.cast(dest.dtype), end=output_axes).sink(arg=KernelInfo(name="host_matmul", opts_to_apply=()))
    return Tensor.empty(*batch, m, n, device=a.device, dtype=a.dtype).custom_kernel(a, b, fxn=kernel)[0]

  def dot2(a, b, c):
    rows, width = a.shape
    columns = b.shape[1]
    bias = Tensor.zeros(columns, device=a.device, dtype=dtypes.float32) if c is None else c.float()

    def kernel(dest, left, right, bias):
      index, output_axes = output_range(dest.numel())
      row, column = index // columns, index % columns
      reg = UOp.placeholder((1,), dtypes.float32, 0, addrspace=AddrSpace.REG)
      reg = reg.after(index)[0].set(bias[column].load())

      def load(tensor, k):
        return (tensor[row, k] if tensor is left else tensor[k, column]).load().cast(dtypes.float32)
      # SME2 FDOT: round the exact two-product sum to float32, then add
      # to the float32 accumulator. Half products are exact in FP32, so
      # separate multiply/add retains this order without software FMA.
      if width >= 2:
        block = UOp.range(width // 2, 1, AxisType.LOOP)
        product = scalar("{0} * {1}", load(left, block * 2 + 1), load(right, block * 2 + 1))
        pair = scalar("{0} + {1}", scalar("{0} * {1}", load(left, block * 2), load(right, block * 2)), product)
        reg = reg[0].set(scalar("{0} + {1}", pair, reg.after(block)[0].load()), end=block)
      value = reg[0].load()
      if width % 2:
        value = scalar("{0} + {1}", scalar("{0} * {1}", load(left, width - 1), load(right, width - 1)), value)
      return dest.flatten()[index].set(value.cast(dest.dtype), end=output_axes).sink(arg=KernelInfo(name="host_half_dot2", opts_to_apply=()))
    return Tensor.empty(rows, columns, device=a.device, dtype=a.dtype).custom_kernel(
      a.contiguous(), b.contiguous(), bias.contiguous(), fxn=kernel)[0]

  def gemm(a, b, c=None, alpha=1., beta=1., transA=0, transB=0, **options):
    standard = not transA and alpha == beta == 1. and not options.get("broadcast", 0)
    vector_bias = c is None or c.shape == (b.shape[0] if transB else b.shape[1],)
    if standard and vector_bias and a.dtype == dtypes.float16 and not transB:
      return dot2(a, b, c)
    if standard and vector_bias and a.dtype == dtypes.float32:
      value = matmul(a, b.T if transB else b).realize()
      return value if c is None else (value + c).realize()
    return ops["Gemm"](a, b, c, alpha=alpha, beta=beta, transA=transA, transB=transB, **options)

  def fused_gemm(a, b, c=None, activation=None, activation_params=(), **options):
    return activate(gemm(a, b, c, **options), activation, activation_params)

  def mean(x, axes=None, keepdims=1, noop_with_empty_axes=0):
    selected = tuple(axes.tolist()) if isinstance(axes, Tensor) else tuple(axes or ())
    selected = tuple(sorted(axis % x.ndim for axis in selected))
    if x.dtype != dtypes.float32 or x.ndim != 4 or selected != (2, 3) or x.shape[2] * x.shape[3] % 8:
      return ops["ReduceMean"](x, axes, keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes)
    positions = x.shape[2] * x.shape[3]
    values = x.reshape(-1, positions)
    lanes = values[:, :8].contiguous().realize()
    for start in range(8, positions, 8):
      lanes = (lanes + values[:, start:start + 8]).realize()
    pack = (lanes[:, :4] + lanes[:, 4:]).realize()
    summed = ((pack[:, 0] + pack[:, 1]).realize() + (pack[:, 2] + pack[:, 3]).realize()).realize()
    result = division(summed, Tensor(float(positions), device=x.device, dtype=x.dtype)).realize()
    return result.reshape(x.shape[0], x.shape[1], 1, 1) if keepdims else result.reshape(x.shape[:2])

  def sum_last(x, axes=None, keepdims=1, noop_with_empty_axes=0):
    selected = tuple(axes.tolist()) if isinstance(axes, Tensor) else tuple(axes or ())
    if x.dtype != dtypes.float32 or tuple(axis % x.ndim for axis in selected) != (x.ndim - 1,) or x.shape[-1] < 8:
      return ops["ReduceSum"](x, axes, keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes)
    width = x.shape[-1]
    rows = x.numel() // width
    source = x.contiguous().realize()

    def kernel(dest, values):
      row, output_axes = output_range(rows)
      # Eigen::Map.sum aligns each row to a four-float packet. The prefix is
      # added after packet reduction, so row stride changes the rounding order.
      start = (4 - row * width % 4) % 4
      end = start + (width - start) // 4 * 4
      count = (end - start) // 4

      def load(index):
        return values.flatten()[row * width + index.minimum(width - 1)].load()
      reg = UOp.placeholder((8,), dtypes.float32, 0, addrspace=AddrSpace.REG)
      reg = reg.after(row).after(*[reg[i].store(load(start + i)) for i in range(8)])
      if width >= 16:
        block = UOp.range(width // 8 - 1, 1, AxisType.LOOP)
        first = start + (block + 1) * 8
        old = reg.after(block)
        reg = reg.after(UOp.group(*[reg[i].store((first + 7 < end).where(
          scalar("{0} + {1}", old[i].load(), load(first + i)), old[i].load())) for i in range(8)]).end(block))
      partial = [(count >= 2).where(scalar("{0} + {1}", reg[i].load(), reg[i + 4].load()), reg[i].load()) for i in range(4)]
      if width >= 12:
        partial = [((count > 1) & (count % 2 != 0)).where(
          scalar("{0} + {1}", value, load(end - 4 + i)), value) for i, value in enumerate(partial)]
      result = scalar("{0} + {1}", scalar("{0} + {1}", *partial[:2]), scalar("{0} + {1}", *partial[2:]))
      for i in range(3):
        result = (i < start).where(scalar("{0} + {1}", result, load(UOp.const(i, dtypes.int32))), result)
      for i in range(3):
        result = (end + i < width).where(scalar("{0} + {1}", result, load(end + i)), result)
      return dest.flatten()[row].set(result, end=output_axes).sink(arg=KernelInfo(name="host_aligned_sum", opts_to_apply=()))

    result = Tensor.empty(rows, device=x.device, dtype=x.dtype).custom_kernel(source, fxn=kernel)[0]
    return result.reshape(*x.shape[:-1], 1) if keepdims else result.reshape(x.shape[:-1])

  def layernorm(x, scale, bias=None, axis=-1, epsilon=1e-5, stash_type=1):
    if stash_type != 1:
      raise ValueError("Only float32 normalization statistics are supported")
    axes = tuple(range(axis % x.ndim, x.ndim))
    width = prod(x.shape[i] for i in axes)
    rows = x.numel() // width
    source = x.contiguous().realize()
    metal = x.device.split(":")[0] == "METAL"

    def moments(dest, values):
      row, output_axes = output_range(rows)
      k = UOp.range(width, 1, AxisType.LOOP)
      reg = UOp.placeholder((2,), dtypes.float32, 0, addrspace=AddrSpace.REG)
      reg = reg.after(row).after(reg[0].store(0.), reg[1].store(0.))
      oldmean, oldm2 = reg.after(k)[0].load(), reg.after(k)[1].load()
      value = values.flatten()[row * width + k].load().cast(dtypes.float32)
      delta = scalar("{0} - {1}", value, oldmean)
      average = scalar("{0} + {1}", oldmean, divide(delta, (k + 1).cast(dtypes.float32), metal))
      m2 = scalar("fma({0}, {1}, {2})", delta, scalar("{0} - {1}", value, average), oldm2)
      reg = reg.after(UOp.group(reg[0].store(average), reg[1].store(m2)).end(k))
      variance = scalar("{0} + {1}", divide(reg[1].load(), UOp.const(float(width), dtypes.float32), metal), epsilon)
      std = rounded_sqrt(variance, metal)
      return UOp.group(dest[row, 0].store(reg[0].load()), dest[row, 1].store(std)).end(*output_axes).sink(
        arg=KernelInfo(name="host_welford", opts_to_apply=()))
    stats = Tensor.empty(rows, 2, device=x.device, dtype=dtypes.float32).custom_kernel(source, fxn=moments)[0].realize()
    scales = scale.float().expand(*x.shape).contiguous().realize()
    biases = Tensor.zeros(*x.shape, device=x.device) if bias is None else bias.float().expand(*x.shape).contiguous().realize()

    def output(dest, values, stats, scales, biases):
      index, output_axes = output_range(dest.numel())
      centered = scalar("{0} - {1}", values.flatten()[index].load().cast(dtypes.float32), stats[index // width, 0].load())
      normalized = divide(centered, stats[index // width, 1].load(), metal)
      value = scalar("fma({0}, {1}, {2})", normalized, scales.flatten()[index].load(), biases.flatten()[index].load())
      return dest.flatten()[index].set(value.cast(dest.dtype), end=output_axes).sink(arg=KernelInfo(name="host_layernorm", opts_to_apply=()))
    result = Tensor.empty(*x.shape, device=x.device, dtype=x.dtype).custom_kernel(source, stats, scales, biases, fxn=output)[0]
    shape = (*x.shape[:axis % x.ndim], *((1,) * len(axes)))
    inverse = division(Tensor.ones(rows, device=x.device), stats[:, 1]).reshape(shape)
    return result, stats[:, 0].reshape(shape), inverse

  def materialize(op):
    def execute(*args, **kwargs):
      return op(*args, **kwargs).realize()
    return execute

  def concat(*values, axis):
    # Tensor.cat adds zero-padded inputs of unequal sizes. Adreno half
    # arithmetic flushes subnormals even though these bytes only need moving.
    # Integer concatenation preserves subnormals, signed zero and NaN payloads.
    dtype = values[0].dtype
    if dtype in (dtypes.float16, dtypes.float32):
      bits = dtypes.uint16 if dtype == dtypes.float16 else dtypes.uint32
      return ops["Concat"](*(value.bitcast(bits) for value in values), axis=axis).bitcast(dtype)
    return ops["Concat"](*values, axis=axis)

  def power(x, y):
    # ORT's scalar square/cube path multiplies explicitly. A trailing singleton
    # selects its other broadcast loop, which calls powf instead. Compensating
    # the cube's product rounding avoids GPU POW approximations in that loop.
    exponent = y.uop.ssimplify() if isinstance(y, Tensor) and y.numel() == 1 else y
    if x.dtype == dtypes.float32 and isinstance(exponent, (int, float)) and exponent in (2, 3):
      if exponent == 3 and x.shape and x.shape[-1] == 1:
        def kernel(dest, values):
          index, output_axes = output_range(dest.numel())
          value = values.flatten()[index].load()
          squared = scalar("{0} * {0}", value)
          error = scalar("fma({0}, {0}, -{1})", value, squared)
          cubed = scalar("{0} * {1}", squared, value)
          tail = scalar("fma({0}, {1}, -{2})", squared, value, cubed)
          correction = scalar("fma({0}, {1}, {2})", error, value, tail)
          result = scalar("{0} + {1}", cubed, correction)
          result = scalar("isfinite({0}) ? {1} : {0}", cubed, result)
          return dest.flatten()[index].set(result, end=output_axes).sink(arg=KernelInfo(name="host_scalar_cube", opts_to_apply=()))
        return Tensor.empty(*x.shape, device=x.device, dtype=x.dtype).custom_kernel(x.contiguous(), fxn=kernel)[0]
      squared = (x * x).realize()
      return squared if exponent == 2 else (squared * x).realize()
    return ops["Pow"](x, y).realize()

  runner.onnx_ops.update({key: materialize(ops[key]) for key in ("Mul", "Add", "Sub") if key in ops})
  runner.onnx_ops.update(Conv=conv, FusedConv=conv, NhwcFusedConv=nhwc, Gemm=gemm, FusedGemm=fused_gemm,
                         MatMul=matmul, ReduceMean=mean, ReduceSum=sum_last, LayerNormalization=layernorm, Div=division, Exp=exponential, Pow=power,
                         Concat=concat,
                         Tanh=lambda x: activation(x, "Tanh"), Sigmoid=lambda x: activation(x, "Sigmoid"),
                         QuickGelu=lambda x, alpha=1.: x * activation(x * alpha, "Sigmoid"))
