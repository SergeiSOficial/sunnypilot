// Diagnostic-only eight-lane half-FMA Gemm, with one work item per lane.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF
inline float round_half(float value) {
  half rounded;
  vstore_half_rte(value, 0, &rounded);
  return vload_half(0, &rounded);
}
inline float half_fma(float x, float y, float z) {
  volatile float product = x * y;
  volatile float total = product + z;
  volatile float part = total - product;
  volatile float remainder = total - part;
  volatile float left = product - remainder;
  volatile float right = z - part;
  volatile float error = left + right;
  uint bits = as_uint((float)total);
  uint exponent = (bits >> 23) & 255u;
  uint shift = (uint)clamp(126 - (int)exponent, 13, 24);
  float adjusted = total;
  if (exponent >= 102u && exponent <= 142u && error != 0.0f &&
      ((((bits & 0x7fffffu) | 0x800000u) & ((1u << shift) - 1u)) == (1u << (shift - 1u))))
    adjusted = nextafter(adjusted, copysign(INFINITY, (float)error));
  return round_half(adjusted);
}
__kernel void road_dense(__global const float *input, __global const float *weights,
                         __global const float *bias, __global float *output) {
  int id = get_global_id(0);
  int local_id = get_local_id(0);
  int row = id / 8, lane = id % 8;
  __local float partial[32];
  float accum = 0.0f;
  for (int start = 0; start < 1024; start += 8)
    accum = half_fma(input[start + lane], weights[row * 1024 + start + lane], accum);
  partial[local_id] = accum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0) {
    float lanes[8];
    for (int i = 0; i < 8; ++i) lanes[i] = partial[local_id + i];
    for (int width = 4; width >= 1; width /= 2)
      for (int i = 0; i < width; ++i)
        lanes[i] = round_half(lanes[2 * i] + lanes[2 * i + 1]);
    output[row] = round_half(lanes[0] + bias[row]);
  }
}
