// Pinned host FP32 Conv: ordered ky/kx/ci accumulation with explicit bias order.
// NC4HW4 buffers use channel-block/batch/height/width/lane ordering.
// Four output channels share an input scalar; no CPU partition or readback.
// Retain correctly rounded FMA and the original sequential reduction.
// Vector loads remove redundant input addressing without changing arithmetic.
#pragma OPENCL FP_CONTRACT OFF
__kernel void fold3_host_conv(__global const float* input,
                              __global const float4* weights,
                              __global const float4* bias,
                              __global float4* output,
                              int batch, int inputChannels, int outputChannels,
                              int2 inHW, int2 outHW, int2 kernelHW,
                              int2 stride, int2 dilation, int2 padding) {
    int cb = get_global_id(0) / outHW.y;
    int ox = get_global_id(0) % outHW.y;
    int n = get_global_id(1) / outHW.x;
    int oy = get_global_id(1) % outHW.x;
    if (cb >= (outputChannels + 3) / 4 || n >= batch) return;
    float4 biasValue = bias[cb];
#if defined(FOLD3_VECTOR_FMA_CONV) && !defined(FOLD3_DEPTHWISE)
#define FOLD3_ACCUMULATOR
#else
#define FOLD3_ACCUMULATOR volatile
#endif
#ifdef FOLD3_BIAS_FIRST
    FOLD3_ACCUMULATOR float4 sum = biasValue;
#else
    FOLD3_ACCUMULATOR float4 sum = (float4)0.0f;
#endif
    for (int ky = 0; ky < kernelHW.x; ++ky) {
        int iy = oy * stride.x + ky * dilation.x - padding.x;
        for (int kx = 0; kx < kernelHW.y; ++kx) {
            int ix = ox * stride.y + kx * dilation.y - padding.y;
            int valid = iy >= 0 && iy < inHW.x && ix >= 0 && ix < inHW.y;
#ifdef FOLD3_DEPTHWISE
            float4 value = valid ? vload4(0, input + (((cb * batch + n) * inHW.x + iy) * inHW.y + ix) * 4) : (float4)0.0f;
            // MNN's depthwise pack is [ky,kx,channel-block,4], unlike
            // the regular Conv's private [channel-block,ky,kx,ci,4] pack.
            float4 weight = weights[(ky * kernelHW.y + kx) * ((outputChannels + 3) / 4) + cb];
            sum = fma(value, weight, sum);
#else
#ifdef FOLD3_VECTOR_FMA_CONV
            int ci = 0;
            int weightOffset = ((cb * kernelHW.x + ky) * kernelHW.y + kx) * inputChannels;
            for (; ci + 3 < inputChannels; ci += 4) {
                float4 values = valid ? vload4(0, input + (((ci / 4 * batch + n) * inHW.x + iy) * inHW.y + ix) * 4) : (float4)0.0f;
                // Retain ky/kx/ci order; vectorize loads, never the reduction.
                sum = fma((float4)values.x, weights[weightOffset + ci], sum);
                sum = fma((float4)values.y, weights[weightOffset + ci + 1], sum);
                sum = fma((float4)values.z, weights[weightOffset + ci + 2], sum);
                sum = fma((float4)values.w, weights[weightOffset + ci + 3], sum);
            }
            for (; ci < inputChannels; ++ci) {
                float value = valid ? input[(((ci / 4 * batch + n) * inHW.x + iy) * inHW.y + ix) * 4 + ci % 4] : 0.0f;
                sum = fma((float4)value, weights[weightOffset + ci], sum);
            }
#else
            for (int ci = 0; ci < inputChannels; ++ci) {
                float value = valid ? input[(((ci / 4 * batch + n) * inHW.x + iy) * inHW.y + ix) * 4 + ci % 4] : 0.0f;
                float4 weight = weights[(((cb * kernelHW.x + ky) * kernelHW.y + kx) * inputChannels + ci)];
                sum = fma((float4)value, weight, sum);
            }
#endif
#endif
        }
    }
#ifndef FOLD3_BIAS_FIRST
    sum = sum + biasValue;
#endif
    float4 result = sum;
#ifdef FOLD3_RELU
    result = fmax(result, (float4)0.0f);
#endif
    if (cb * 4 + 1 >= outputChannels) result.y = 0.0f;
    if (cb * 4 + 2 >= outputChannels) result.z = 0.0f;
    if (cb * 4 + 3 >= outputChannels) result.w = 0.0f;
    output[((cb * batch + n) * outHW.x + oy) * outHW.y + ox] = result;
}

// Original FP16 NoTrans Gemm uses SME2 FDOT2: bias first, then a rounded
// FP32 two-product sum followed by a rounded FP32 accumulator update.
// Half products are exact in FP32. Explicit Cast supplies final FP16 rounding.
// Only validated, unactivated 1x1 Gemm conversions reach this entry point.
__kernel void fold3_half_dot2(__global const float* input,
                             __global const float4* weights,
                             __global const float4* bias,
                             __global float4* output,
                             int batch, int inputChannels, int outputChannels,
                             int2 inHW, int2 outHW, int2 kernelHW,
                             int2 stride, int2 dilation, int2 padding) {
    int cb = get_global_id(0) / outHW.y;
    int ox = get_global_id(0) % outHW.y;
    int n = get_global_id(1) / outHW.x;
    int oy = get_global_id(1) % outHW.x;
    if (cb >= (outputChannels + 3) / 4 || n >= batch) return;
    volatile float4 sum = bias[cb];
    for (int ci = 0; ci + 1 < inputChannels; ci += 2) {
        float a = input[(((ci / 4 * batch + n) * inHW.x + oy) * inHW.y + ox) * 4 + ci % 4];
        float b = input[((((ci + 1) / 4 * batch + n) * inHW.x + oy) * inHW.y + ox) * 4 + (ci + 1) % 4];
        volatile float4 first = a * weights[cb * inputChannels + ci];
        volatile float4 second = b * weights[cb * inputChannels + ci + 1];
        volatile float4 pair = first + second;
        sum = pair + sum;
    }
    if (inputChannels % 2) {
        int ci = inputChannels - 1;
        float value = input[(((ci / 4 * batch + n) * inHW.x + oy) * inHW.y + ox) * 4 + ci % 4];
        volatile float4 product = value * weights[cb * inputChannels + ci];
        sum = product + sum;
    }
    float4 result = sum;
    if (cb * 4 + 1 >= outputChannels) result.y = 0.0f;
    if (cb * 4 + 2 >= outputChannels) result.z = 0.0f;
    if (cb * 4 + 3 >= outputChannels) result.w = 0.0f;
    output[((cb * batch + n) * outHW.x + oy) * outHW.y + ox] = result;
}
