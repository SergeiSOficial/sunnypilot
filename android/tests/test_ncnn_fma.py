"""Compile the patched scalar GLSL arithmetic and check native FMA rounding."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NcnnFmaTest(unittest.TestCase):
  def test_attention_half_rounding_matches_reference_at_midpoints(self):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
      self.skipTest("Host C compiler unavailable")
    patch = (ROOT / "android/patches/ncnn-explicit-casts.patch").read_text()
    section = patch.rsplit("+++ b/src/layer/vulkan/shader/matmul.comp", 1)[1]
    added = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))
    begin = added.index("float half_rte(")
    function = added[begin:added.index("\n}", begin) + 2]
    self.assertIn("half_rte(total)", added)
    source = r'''
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
typedef uint32_t uint;
typedef struct { float x; } vec2;
static uint floatBitsToUint(float x) { uint b; memcpy(&b,&x,4); return b; }
static float from_half(uint16_t bits) { _Float16 h; memcpy(&h,&bits,2); return (float)h; }
static vec2 unpackHalf2x16(uint bits) { return (vec2){from_half((uint16_t)bits)}; }
''' + function + r'''
static int check(float v) {
  float actual=half_rte(v), expected=(float)(_Float16)v;
  if(isnan(expected)) return !isnan(actual);
  if(floatBitsToUint(actual)==floatBitsToUint(expected)) return 0;
  fprintf(stderr,"Half rounding mismatch %a: %a != %a\n",v,actual,expected);
  return 1;
}
int main(void) {
  for(uint16_t b=0;b<0x7bff;b++) {
    float a=from_half(b), next=from_half(b+1), midpoint=(a+next)*0.5f;
    float values[]={a,midpoint,nextafterf(midpoint,-INFINITY),nextafterf(midpoint,INFINITY)};
    for(int i=0;i<4;i++) if(check(values[i])||check(-values[i])) return 1;
  }
  float edges[]={65504.f,65519.f,65520.f,65536.f,INFINITY,-INFINITY,NAN,-0.f};
  for(int i=0;i<8;i++) if(check(edges[i])) return 2;
  return 0;
}
'''
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      (path / "rounding.c").write_text(source)
      subprocess.run([compiler, "-std=c11", "-O2", "-ffp-contract=off", str(path / "rounding.c"), "-lm", "-o", str(path / "rounding")], check=True)
      subprocess.run([str(path / "rounding")], check=True)

  def test_tanh_fma_retains_midpoint_tail(self):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
      self.skipTest("Host C compiler unavailable")
    patch = (ROOT / "android/patches/ncnn-explicit-casts.patch").read_text()
    section = patch.split("+++ b/src/layer/vulkan/shader/tanh.comp", 1)[1]
    added = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))
    function = added[added.index("float fold3_tanh_fma"):added.index("float fold3_reference_divide")]
    function = re.sub(r"\bprecise\s+", "", function).replace("abs(", "fabsf(")
    source = r'''
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
typedef uint32_t uint;
static uint floatBitsToUint(float x) { uint b; memcpy(&b, &x, 4); return b; }
static float uintBitsToFloat(uint b) { float x; memcpy(&x, &b, 4); return x; }
''' + function + r'''
static int check(float a, float b, float c) {
  float actual = fold3_tanh_fma(a,b,c), expected = fmaf(a,b,c);
  if (actual == expected) return 0;
  fprintf(stderr,"FMA mismatch %a %a %a: %a != %a\n",a,b,c,actual,expected);
  return 1;
}
int main(void) {
  /* First incorrect tanh denominator stage from seed123, plus both signs. */
  float a=0.0001215868687722832f,b=2.5471675395965576f,c=0.0022684347350150347f;
  if(check(a,b,c)||check(-a,b,-c)||check(a,-b,-c)) return 1;
  uint state=123;
  for(int i=0;i<200000;i++) {
    state=1664525u*state+1013904223u; a=((int)(state%200001u)-100000)*0.0001f;
    state=1664525u*state+1013904223u; b=((int)(state%200001u)-100000)*0.0001f;
    state=1664525u*state+1013904223u; c=((int)(state%200001u)-100000)*0.0001f;
    if(check(a,b,c)) return 1;
  }
  return 0;
}
'''
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      (path / "fma.c").write_text(source)
      subprocess.run([compiler, "-std=c11", "-O2", "-ffp-contract=off", str(path / "fma.c"), "-lm", "-o", str(path / "fma")],
                     check=True, env=os.environ)
      subprocess.run([str(path / "fma")], check=True)

  def test_sigmoid_preserves_host_rounding_before_half_cast(self):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
      self.skipTest("Host C compiler unavailable")
    patch = (ROOT / "android/patches/ncnn-explicit-casts.patch").read_text()
    section = patch.split("+++ b/src/layer/vulkan/shader/vulkan_activation.comp", 1)[1].split("--- a/", 1)[0]
    added = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))
    functions = added[added.index("float fold3_activation_fma"):added.index("\n#endif")]
    functions = re.sub(r"\bprecise\s+", "", functions).replace("abs(", "fabsf(")
    reference = functions[functions.index("float fold3_host_sigmoid"):].replace("fold3_host_sigmoid", "native_sigmoid")
    reference = reference.replace("fold3_activation_fma", "fmaf").replace("fold3_activation_divide(product, denominator)", "(product / denominator)")
    source = r'''
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
typedef uint32_t uint;
static uint floatBitsToUint(float x) { uint b; memcpy(&b, &x, 4); return b; }
static float uintBitsToFloat(uint b) { float x; memcpy(&x, &b, 4); return x; }
static float clamp(float x, float lo, float hi) { return fminf(fmaxf(x,lo),hi); }
''' + functions + reference + r'''
int main(void) {
  uint state=42;
  for(int i=0;i<200000;i++) {
    state=1664525u*state+1013904223u;
    float value=((int)(state%4000001u)-2000000)*0.00001f;
    float actual=fold3_host_sigmoid(value), expected=native_sigmoid(value);
    if(floatBitsToUint(actual)!=floatBitsToUint(expected)) {
      fprintf(stderr,"Sigmoid rounding mismatch %a: %a != %a\n",value,actual,expected);
      return 1;
    }
  }
  if(!isnan(fold3_host_sigmoid(NAN))) return 2;
  return 0;
}
'''
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      (path / "sigmoid.c").write_text(source)
      subprocess.run([compiler, "-std=c11", "-O2", "-ffp-contract=off", str(path / "sigmoid.c"), "-lm", "-o", str(path / "sigmoid")], check=True)
      subprocess.run([str(path / "sigmoid")], check=True)
