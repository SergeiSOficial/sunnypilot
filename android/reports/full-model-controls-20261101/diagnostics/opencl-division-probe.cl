inline float4 fold3_host_divide(float4 left, float4 right) {
    volatile float4 quotient = left / right;
    volatile float4 residual = fma(-quotient, right, left);
    volatile float4 correction = residual / right;
    volatile float4 corrected = quotient + correction;
    volatile float4 magnitude = fabs(corrected);
    volatile float4 denominator = fabs(right);
    residual = fma(-magnitude, denominator, fabs(left));
    volatile float4 direction = select((float4)INFINITY, (float4)0.0f, residual < (float4)0.0f);
    volatile float4 neighbor = nextafter(magnitude, direction);
    volatile float4 half_gap = fabs(neighbor - magnitude) * 0.5f;
    volatile float4 distance = fma(-half_gap, denominator, fabs(residual));
    int4 round_up = (distance > (float4)0.0f) |
        ((distance == (float4)0.0f) & ((as_uint4(magnitude) & (uint4)1U) != (uint4)0U));
    volatile float4 rounded = select(magnitude, neighbor, round_up);
    corrected = select(corrected, copysign(rounded, corrected),
        (half_gap > (float4)0.0f) & isfinite(neighbor));
    return select(quotient, corrected,
        isfinite(left) & isfinite(right) & isfinite(quotient) & (right != (float4)0.0f));
}

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

__kernel void bench_divide(__global const float4 *left,__global const float4 *right,__global float4 *out) {
 int i=get_global_id(0); float4 a=left[i], b=right[i];
 volatile float4 q=a/b;
 volatile float4 r=fma(-q,b,a);
 volatile float4 c=r/b;
 volatile float4 y=q+c;
 float4 mag=fabs(y);
 r=fma(-mag,fabs(b),fabs(a));
 float4 neighbor=nextafter(mag,select((float4)INFINITY,(float4)0.f,r<(float4)0.f));
 float4 gap=fabs(neighbor-mag)*.5f;
 out[i*8+0]=q;
 out[i*8+1]=fold3_host_divide(a,b);
 out[i*8+2]=(float4)(fold3_reference_divide(a.x,b.x),fold3_reference_divide(a.y,b.y),fold3_reference_divide(a.z,b.z),fold3_reference_divide(a.w,b.w));
 out[i*8+3]=y;out[i*8+4]=r;out[i*8+5]=neighbor;out[i*8+6]=gap;
 out[i*8+7]=fma(-gap,fabs(b),fabs(r));
}
