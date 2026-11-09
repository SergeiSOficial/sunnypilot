#include <algorithm>
#include <cstdint>
#include <cstring>
using std::min; using std::max; using std::clamp;
static const float *bottom_blob_data, *weight_data, *bias_data;
static int w, outw, bias_term=1;
#define psc(x) (x)
#define buffer_ld1(buf,i) ((buf)[i])
static uint32_t floatBitsToUint(float v) {uint32_t u; memcpy(&u,&v,4);return u;}
static float uintBitsToFloat(uint32_t u) {float v;memcpy(&v,&u,4);return v;}
static float half_float(uint32_t u) {uint16_t h=u;_Float16 v;memcpy(&v,&h,2);return float(v);}
float fold3_half_rte(float value)
{
    uint32_t bits = floatBitsToUint(value);
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t magnitude = bits & 0x7fffffffu;
    uint32_t result;
    if (magnitude > 0x7f800000u) result = 0x7e00u;
    else if (magnitude >= 0x477ff000u) result = 0x7c00u;
    else if (magnitude >= 0x38800000u)
        result = (magnitude + 0xfffu + ((magnitude >> 13) & 1u) - 0x38000000u) >> 13;
    else if (magnitude <= 0x33000000u) result = 0u;
    else
    {
        uint32_t shift = 126u - (magnitude >> 23);
        uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
        result = (mantissa + ((1u << (shift - 1u)) - 1u) + ((mantissa >> shift) & 1u)) >> shift;
    }
    // Every half is exact here, so unpacking adds no further rounding.
    return half_float(sign | result);
}

// Match the pinned original ARM64 half GEMV, not an FP32 reduction.
float fold3_half_fma(float x, float y, float z)
{
    float product = x * y;
    float total = product + z;
    float part = total - product;
    float remainder = total - part;
    float left = product - remainder;
    float right = z - part;
    float error = left + right;
    uint32_t bits = floatBitsToUint(total);
    uint32_t exponent = (bits >> 23) & 255u;
    uint32_t shift = uint32_t(clamp(126 - int(exponent), 13, 24));
    if (exponent >= 102u && exponent <= 142u && error != 0.0 &&
        ((((bits & 0x7fffffu) | 0x800000u) & ((1u << shift) - 1u)) == (1u << (shift - 1u))))
    {
        // Advance one FP32 ulp toward the exact addition residual.
        bits = ((total > 0.0) == (error > 0.0)) ? bits + 1u : bits - 1u;
        total = uintBitsToFloat(bits);
    }
    return fold3_half_rte(total);
}
float fold3_gemv(int row)
{
    int k = psc(w);
    int n = psc(outw);
    int end = (k / 8) * 8;
    bool coefficient = end > 0 && k + n + 1 < 20;
    int packets = n == 1 && !coefficient ? min(4, max(1, end / 8)) : 1;
    float partial[32];
    for (int i = 0; i < 32; ++i) partial[i] = 0.0;
    for (int start = 0; start < end; start += 8)
    {
        for (int lane = 0; lane < 8; ++lane)
        {
            int j = start + lane;
            int index = ((start / 8) % packets) * 8 + lane;
            float x = float(buffer_ld1(bottom_blob_data, j));
            float y = float(buffer_ld1(weight_data, row * k + j));
            partial[index] = coefficient
                ? fold3_half_rte(partial[index] + fold3_half_rte(x * y))
                : fold3_half_fma(x, y, partial[index]);
        }
    }
    float lanes[8];
    for (int lane = 0; lane < 8; ++lane)
    {
        lanes[lane] = partial[(packets - 1) * 8 + lane];
        for (int packet = packets - 2; packet >= 0; --packet)
            lanes[lane] = fold3_half_rte(partial[packet * 8 + lane] + lanes[lane]);
    }
    for (int width = 4; width >= 1; width /= 2)
        for (int lane = 0; lane < width; ++lane)
            lanes[lane] = fold3_half_rte(lanes[lane * 2] + lanes[lane * 2 + 1]);
    float value = lanes[0];
    for (int i = end; i < k; ++i)
        value = fold3_half_rte(value + fold3_half_rte(float(buffer_ld1(bottom_blob_data, i)) * float(buffer_ld1(weight_data, row * k + i))));
    return bias_term == 1 ? fold3_half_rte(value + float(buffer_ld1(bias_data, row))) : value;
}


extern "C" void run(const float* x, const float* weights, const float* bias, float* output, int k, int n) {
 bottom_blob_data=x; weight_data=weights; bias_data=bias; w=k;outw=n;
 for(int row=0;row<n;row++) output[row]=fold3_gemv(row);
}
