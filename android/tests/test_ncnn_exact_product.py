"""Exact short-significand products can bypass software FMA correction."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ExactProductTest(unittest.TestCase):
  def test_zero_weight_tail_preserves_fma_correction(self):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
      self.skipTest("Host C compiler unavailable")
    reference = (ROOT / "android/patches/ncnn-explicit-casts.patch").read_text()
    section = reference.split("+++ b/src/layer/vulkan/shader/convolution_packed.comp", 1)[1].split("--- a/", 1)[0]
    added = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))
    start = added.index("vec4 fold3_spatial_fma(")
    original = added[start:added.index("\n}", start) + 2]
    patch = (ROOT / "android/patches/ncnn-exact-product.patch").read_text()
    blocks = re.findall(r"\+    precise vec4 error;\n(.*?)\n ", patch, re.S)
    self.assertEqual(len(blocks), 3)
    self.assertEqual(len(set(blocks)), 1)
    replacement = "precise vec4 error;\n" + "\n".join(line[1:] for line in blocks[0].splitlines())
    optimized = original.replace("precise vec4 error = ((ah * bh - product) + ah * bl + al * bh) + al * bl;", replacement)
    self.assertNotEqual(original, optimized)

    def scalar(function, name):
      function = function.replace("fold3_spatial_fma", name)
      function = function.replace("uvec4(0xfffff000u)", "0xfffff000u")
      function = function.replace("vec4", "float").replace("lane < 4", "lane < 1")
      function = re.sub(r"\bprecise\s+", "", function).replace("[lane]", "").replace("abs(", "fabsf(")
      return function

    source = r'''
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
typedef uint32_t uint;
static uint floatBitsToUint(float x) { uint b; memcpy(&b,&x,4); return b; }
static float uintBitsToFloat(uint b) { float x; memcpy(&x,&b,4); return x; }
static float from_half(uint16_t bits) { _Float16 h; memcpy(&h,&bits,2); return (float)h; }
static int fold3_zero_weight_tail;
''' + scalar(original, "baseline") + scalar(optimized, "optimized") + r'''
static int check(float a, float b, float c, bool native) {
  fold3_zero_weight_tail=0;
  float prior=baseline(a,b,c), general=optimized(a,b,c);
  fold3_zero_weight_tail=isfinite(b) && (floatBitsToUint(b)&0xfffu)==0u;
  float actual=optimized(a,b,c);
  if(!(isnan(general)&&isnan(prior)) && floatBitsToUint(general)!=floatBitsToUint(prior)) return 3;
  if(!(isnan(actual)&&isnan(prior)) && floatBitsToUint(actual)!=floatBitsToUint(prior)) {
    fprintf(stderr,"Residual regression %a %a %a: %a != %a\n",a,b,c,actual,prior); return 1;
  }
  float expected=fmaf(a,b,c);
  if(native && !(isnan(expected)&&isnan(actual)) && actual!=expected) {
    fprintf(stderr,"FMA mismatch %a %a %a: %a != %a\n",a,b,c,actual,expected); return 2;
  }
  return 0;
}
int main(void) {
  uint state=123;
  for(uint i=0;i<65536;i++) {
    float b=from_half(i); if(!isfinite(b)) continue;
    for(int j=0;j<16;j++) {
      state=1664525u*state+1013904223u;
      float a=((int)(state%2000001u)-1000000)*0.001f;
      state=1664525u*state+1013904223u;
      float c=((int)(state%2000001u)-1000000)*0.001f;
      if(check(a,b,c,true)||check(a,b,-(a*b),true)) return 1;
    }
  }
  /* General FP32 weights, exponent extremes, zeros and non-finite fallback. */
  for(int i=0;i<200000;i++) {
    state=1664525u*state+1013904223u; float a=uintBitsToFloat(state);
    state=1664525u*state+1013904223u; float b=uintBitsToFloat(state);
    state=1664525u*state+1013904223u; float c=uintBitsToFloat(state);
    if(check(a,b,c,false)) return 2;
  }
  return 0;
}
'''
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      (path / "residual.c").write_text(source)
      subprocess.run([compiler, "-std=c11", "-O2", "-ffp-contract=off", str(path / "residual.c"), "-lm", "-o", str(path / "residual")], check=True)
      subprocess.run([str(path / "residual")], check=True)

  def test_weight_tail_is_specialized_from_actual_packed_values(self):
    patch = (ROOT / "android/patches/ncnn-exact-product.patch").read_text()
    self.assertNotIn("+    if (all(equal(bl, vec4(0.f))))", patch)
    self.assertEqual(patch.count("+    if (fold3_zero_weight_tail != 0)"), 3)
    self.assertIn("(bits & 0xfffu) != 0u", patch)
    self.assertIn("(bits & 0x7f800000u) == 0x7f800000u", patch)
    self.assertIn("weight_data_packed.channel(q)", patch)
    self.assertIn("weight_data_packed.w * weight_data_packed.h * weight_data_packed.d", patch)
    for shader, index, slot in (("convolution_packed.comp", 23, "12 + 11"),
                                ("convolution_packed_gemm.comp", 22, "12 + 10"),
                                ("convolution_packed_1x1s1d1.comp", 14, "6 + 8")):
      section = patch.split("+++ b/src/layer/vulkan/shader/" + shader, 1)[1].split("--- a/", 1)[0]
      self.assertIn(f"layout(constant_id = {index}) const int fold3_zero_weight_tail = 0;", section)
      self.assertIn(f"specializations[{slot}].i = fold3_zero_weight_tail;", patch)

  def test_spatial_shortcut_is_scoped_to_the_first_driver_stem(self):
    patch = (ROOT / "android/patches/ncnn-exact-product.patch").read_text()
    for name, channels in (("convolution_packed.comp", "psc(c) * elempack"),
                           ("convolution_packed_gemm.comp", "psc(num_input)")):
      section = patch.split("+++ b/src/layer/vulkan/shader/" + name, 1)[1].split("--- a/", 1)[0]
      self.assertIn(channels + " == 16 && psc(num_output) == 32", section)
      self.assertIn("? fold3_half_stem_fma(", section)
      self.assertIn(": fold3_spatial_fma(", section)
      self.assertIn("return fold3_spatial_fma(a, b, c);", section)

  def test_gradle_checks_every_shader_patch(self):
    gradle = (ROOT / "android/app/build.gradle").read_text()
    for key in ("profile_patch_sha256", "tiling_patch_sha256", "exact_product_patch_sha256"):
      self.assertIn(key, gradle)

  def test_half_products_and_fp32_accumulators_match_fma(self):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
      self.skipTest("Host C compiler unavailable")
    patch = (ROOT / "android/patches/ncnn-exact-product.patch").read_text()
    section = patch.split("+++ b/src/layer/vulkan/shader/convolution_packed.comp", 1)[1].split("--- a/", 1)[0]
    added = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))
    begin = added.index("bool fold3_exact_product(")
    function = added[begin:added.index("\n}", begin) + 2]
    source = r'''
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
typedef uint32_t uint;
static uint floatBitsToUint(float x) { uint b; memcpy(&b,&x,4); return b; }
static float uintBitsToFloat(uint b) { float x; memcpy(&x,&b,4); return x; }
static float from_half(uint16_t bits) { _Float16 h; memcpy(&h,&bits,2); return (float)h; }
''' + function + r'''
static int check(float a, float b, float c) {
  if(!fold3_exact_product(a,b)) return 1;
  float product=a*b;
  if((double)product!=(double)a*(double)b) return 2;
  float actual=product+c, expected=fmaf(a,b,c);
  if(isnan(expected)) return !isnan(actual);
  if(floatBitsToUint(actual)==floatBitsToUint(expected)) return 0;
  fprintf(stderr,"Exact-product mismatch %a %a %a: %a != %a\n",a,b,c,actual,expected);
  return 3;
}
int main(void) {
  uint state=42;
  /* Every finite half, including both zeros and the subnormal boundaries. */
  for(uint i=0;i<65536;i++) {
    float a=from_half(i);
    if(!isfinite(a)) continue;
    for(int j=0;j<8;j++) {
      state=1664525u*state+1013904223u;
      float b=from_half((uint16_t)(state%0x7c00u));
      state=1664525u*state+1013904223u;
      float c=uintBitsToFloat(state);
      if(check(a,b,c)||check(a,-b,c)||check(a,b,-(a*b))) return 1;
    }
  }
  float rejected[]={1.00000011920928955078125f,0x1p-25f,0x1p20f,INFINITY,NAN};
  for(int i=0;i<5;i++)
    if(fold3_exact_product(rejected[i],1.f)||fold3_exact_product(1.f,rejected[i])) return 2;
  return 0;
}
'''
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      (path / "product.c").write_text(source)
      subprocess.run([compiler, "-std=c11", "-O2", "-ffp-contract=off", str(path / "product.c"), "-lm", "-o", str(path / "product")],
                     check=True, env=os.environ)
      subprocess.run([str(path / "product")], check=True)


if __name__ == "__main__":
  unittest.main()
