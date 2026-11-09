"""Extend the pinned patch with captured-input-verified activation and mean arithmetic."""
from pathlib import Path
import difflib
import subprocess
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'android/build/native/accelerators'
source=BASE/'ncnn-patched/ncnn-7e36d25cb153092ba87a60530455c7376fedf452'
pristine=BASE/'ncnn-source/ncnn-7e36d25cb153092ba87a60530455c7376fedf452'
shader=source/'src/layer/vulkan/shader/tanh.comp'
s=shader.read_text()
assert 'fold3_reference_divide' not in s
helpers='''// ONNX Runtime MLAS tanh.cpp, Copyright Microsoft Corporation, MIT.
// Preserve its explicit FP32 FMA and division boundaries in diagnostic FP32 mode.
float fold3_reference_divide(float left, float right)
{
    precise float quotient = left / right;
    if (isnan(left) || isinf(left) || isnan(right) || isinf(right) ||
        isnan(quotient) || isinf(quotient) || right == 0.f || left == 0.f) return quotient;
    precise float residual = fma(-quotient, right, left);
    precise float correction = residual / right;
    precise float corrected = quotient + correction;
    float magnitude = abs(corrected), denominator = abs(right);
    residual = fma(-magnitude, denominator, abs(left));
    uint bits = floatBitsToUint(magnitude);
    float neighbor = uintBitsToFloat(residual < 0.f ? bits - 1u : bits + 1u);
    precise float halfGap = abs(neighbor - magnitude) * 0.5f;
    precise float distance = fma(-halfGap, denominator, abs(residual));
    if (halfGap > 0.f && !isinf(neighbor) &&
        (distance > 0.f || (distance == 0.f && (bits & 1u) != 0u))) magnitude = neighbor;
    return uintBitsToFloat((floatBitsToUint(corrected) & 0x80000000u) | floatBitsToUint(magnitude));
}

float fold3_host_tanh(float input_value)
{
    if (isnan(input_value)) return input_value;
    float value = clamp(input_value, -9.f, 9.f);
    precise float square = value * value;
    float numerator = -2.76076847742355e-16f;
    numerator = fma(numerator, square, 2.00018790482477e-13f);
    numerator = fma(numerator, square, -8.60467152213735e-11f);
    numerator = fma(numerator, square, 5.12229709037114e-08f);
    numerator = fma(numerator, square, 1.48572235717979e-05f);
    numerator = fma(numerator, square, 6.37261928875436e-04f);
    numerator = fma(numerator, square, 4.89352455891786e-03f);
    float denominator = 1.19825839466702e-06f;
    denominator = fma(denominator, square, 1.18534705686654e-04f);
    denominator = fma(denominator, square, 2.26843463243900e-03f);
    denominator = fma(denominator, square, 4.89352518554385e-03f);
    precise float product = numerator * value;
    return fold3_reference_divide(product, denominator);
}

'''
s=s.replace('void main()',helpers+'void main()')
s=s.replace('    // avoid overflow', '''#if !NCNN_fp16_storage && !NCNN_fp16_packed && !NCNN_fp16_arithmetic && !NCNN_bf16_storage && !NCNN_bf16_packed
    v = vec4(fold3_host_tanh(v.x), fold3_host_tanh(v.y), fold3_host_tanh(v.z), fold3_host_tanh(v.w));
#else
    // avoid overflow''')
s=s.replace('    buffer_st4', '#endif\n\n    buffer_st4')
shader.write_text(s)
shader=source/'src/layer/vulkan/shader/reduction.comp';s=shader.read_text()
assert 'fold3_host_mean' not in s
code='''#if !NCNN_fp16_storage && !NCNN_fp16_packed && !NCNN_fp16_arithmetic && !NCNN_bf16_storage && !NCNN_bf16_packed
    // Pinned driver's FP32 [1,960,8,12] spatial mean. Match the host's eight
    // independent lanes, horizontal additions and division (not reciprocal multiply).
    // The uniform branch returns every lane before the generic reduction barriers.
    if (op == 3 && p.in_dims == 3 && in_w == 12 && in_h == 8 && in_c == 960 &&
        p.reduce_w == 1 && p.reduce_h == 1 && p.reduce_d == 0 && p.reduce_c == 0 &&
        p.coeff == 1.f / 96.f)
    {
        if (tid == 0)
        {
            precise float lanes[8];
            for (int lane = 0; lane < 8; lane++)
                lanes[lane] = buffer_ld1(bottom_blob_data, in_index(lane, 0, 0, in_coord[0]));
            for (int offset = 8; offset < 96; offset += 8)
                for (int lane = 0; lane < 8; lane++)
                {
                    int i = offset + lane;
                    lanes[lane] = lanes[lane] + buffer_ld1(bottom_blob_data, in_index(i % 12, i / 12, 0, in_coord[0]));
                }
            precise vec4 packed = vec4(lanes[0], lanes[1], lanes[2], lanes[3]) +
                                  vec4(lanes[4], lanes[5], lanes[6], lanes[7]);
            precise float left = packed.x + packed.y;
            precise float right = packed.z + packed.w;
            precise float summed = left + right;
            precise float quotient = summed / 96.f;
            precise float residual = fma(-quotient, 96.f, summed);
            precise float correction = residual / 96.f;
            precise float fold3_host_mean = quotient + correction;
            buffer_st1(top_blob_data, out_offset(ow, oh, od, oc), fold3_host_mean);
        }
        return;
    }
#endif

'''
s=s.replace('    int reduce_size = 1;',code+'    int reduce_size = 1;');shader.write_text(s)
# Regenerate all existing changes against the pinned pristine tree, retaining them.
patch=ROOT/'android/patches/ncnn-explicit-casts.patch'
old=patch.read_text();paths={line[6:] for line in old.splitlines() if line.startswith('+++ b/')}
paths.add('src/layer/vulkan/shader/tanh.comp')
result=''
for path in sorted(paths):
 result+=''.join(difflib.unified_diff((pristine/path).read_text().splitlines(True),(source/path).read_text().splitlines(True),fromfile='a/'+path,tofile='b/'+path))
patch.write_text(result)
subprocess.run(['patch','--dry-run','-p1','-i',str(patch)],cwd=pristine,check=True)
