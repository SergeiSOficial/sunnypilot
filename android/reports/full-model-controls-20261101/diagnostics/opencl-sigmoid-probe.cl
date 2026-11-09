// ONNX Runtime MLAS tanh.cpp (MIT), matching the pinned host's FP32
// polynomial and final division. Volatile steps preserve arithmetic boundaries
// under MNN's global -cl-mad-enable. This is used only by High buffer Tanh.
inline float fold3_reference_divide(float left, float right) {
    volatile float quotient = left / right;
    if (!isfinite(left) || !isfinite(right) || !isfinite(quotient) || right == 0.0f || left == 0.0f) return quotient;
    volatile float residual = fma(-quotient, right, left);
    volatile float correction = residual / right;
    volatile float corrected = quotient + correction;
    float magnitude = fabs(corrected), denominator = fabs(right);
    residual = fma(-magnitude, denominator, fabs(left));
    float neighbor = nextafter(magnitude, residual < 0.0f ? 0.0f : INFINITY);
    float halfGap = fabs(neighbor - magnitude) * 0.5f;
    float distance = fma(-halfGap, denominator, fabs(residual));
    if (halfGap > 0.0f && isfinite(neighbor) &&
        (distance > 0.0f || (distance == 0.0f && (as_uint(magnitude) & 1U)))) magnitude = neighbor;
    return copysign(magnitude, corrected);
}
inline float4 fold3_host_tanh(float4 input) {
    float4 value = clamp(input, (float4)-9.0f, (float4)9.0f);
    volatile float4 square = value * value;
    float4 numerator = (float4)-2.76076847742355e-16f;
    numerator = fma(numerator, square, (float4)2.00018790482477e-13f);
    numerator = fma(numerator, square, (float4)-8.60467152213735e-11f);
    numerator = fma(numerator, square, (float4)5.12229709037114e-08f);
    numerator = fma(numerator, square, (float4)1.48572235717979e-05f);
    numerator = fma(numerator, square, (float4)6.37261928875436e-04f);
    numerator = fma(numerator, square, (float4)4.89352455891786e-03f);
    float4 denominator = (float4)1.19825839466702e-06f;
    denominator = fma(denominator, square, (float4)1.18534705686654e-04f);
    denominator = fma(denominator, square, (float4)2.26843463243900e-03f);
    denominator = fma(denominator, square, (float4)4.89352518554385e-03f);
    volatile float4 product = numerator * value;
    return (float4)(fold3_reference_divide(product.x, denominator.x),
                    fold3_reference_divide(product.y, denominator.y),
                    fold3_reference_divide(product.z, denominator.z),
                    fold3_reference_divide(product.w, denominator.w));
}

// ONNX Runtime MLAS logistic.cpp, Copyright (c) Microsoft Corporation, MIT.
// The pinned arm64 float4 kernel uses this rational polynomial with explicit
// FMAs and a correctly rounded division. native_exp/native_recip can cross a
// later FP16 rounding boundary even when the sigmoid itself is within tolerance.
inline float4 fold3_host_sigmoid(float4 input) {
    float4 value = clamp(input, (float4)-18.0f, (float4)18.0f);
    volatile float4 square = value * value;
    float4 numerator = fma(square, (float4)4.37031012579801e-11f, (float4)1.15627324459942e-07f);
    numerator = fma(numerator, square, (float4)6.08574864600143e-05f);
    numerator = fma(numerator, square, (float4)8.51377133304701e-03f);
    numerator = fma(numerator, square, (float4)2.48287947061529e-01f);
    volatile float4 product = numerator * value;
    float4 denominator = fma(square, (float4)6.10247389755681e-13f, (float4)5.76102136993427e-09f);
    denominator = fma(denominator, square, (float4)6.29106785017040e-06f);
    denominator = fma(denominator, square, (float4)1.70198817374094e-03f);
    denominator = fma(denominator, square, (float4)1.16817656904453e-01f);
    denominator = fma(denominator, square, (float4)9.93151921023180e-01f);
    volatile float4 quotient = (float4)(fold3_reference_divide(product.x, denominator.x),
                                       fold3_reference_divide(product.y, denominator.y),
                                       fold3_reference_divide(product.z, denominator.z),
                                       fold3_reference_divide(product.w, denominator.w));
    volatile float4 result = quotient + (float4)0.5f;
    return clamp(result, (float4)0.0f, (float4)1.0f);
}


__kernel void bench_sigmoid(__global const float* input, __global const float* unused, __global float* output) {
    size_t id=get_global_id(0);
    float4 in=vload4(id,input);
    float4 native=native_recip((float4)1.0f+native_exp(-in));
    float4 corrected=fold3_host_sigmoid(in);
    float4 exponential=exp(-in);
    float4 ordinary=(float4)1.0f/((float4)1.0f+exponential);
    float4 values[8]={native,corrected,ordinary,in,(float4)0,(float4)0,(float4)0,(float4)0};
    for(size_t stage=0;stage<8;++stage) vstore4(values[stage],id*8+stage,output);
}
