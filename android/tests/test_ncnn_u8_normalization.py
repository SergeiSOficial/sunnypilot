"""Compile the actual shader helper; exhaust the byte domain and fallback edges."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class U8NormalizationTest(unittest.TestCase):
  def test_exact_bits_and_guarded_fallback(self):
    compiler = shutil.which('clang++')
    if not compiler:
      self.skipTest('C++ compiler unavailable')
    patch = (ROOT/'android/patches/ncnn-u8-normalization.patch').read_text()
    added = '\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    sections = patch.split('--- a/src/layer/vulkan/shader/')[1:]
    self.assertEqual([part.splitlines()[0] for part in sections], ['binaryop.comp', 'binaryop_pack4.comp'])
    helpers = []
    for section in sections:
      lines = '\n'.join(line[1:] for line in section.splitlines() if line.startswith('+') and not line.startswith('+++'))
      helpers.append(lines[lines.index('float fold3_u8_divide_255'):lines.index('#endif')])
      self.assertIn('with_scalar == 1 && const_b == 255.0', lines)
    self.assertEqual(helpers[0], helpers[1])
    for lane in 'xyzw':
      self.assertIn(f'fold3_u8_divide_255(v1.{lane})', sections[1])
    helper = helpers[0]
    program = r'''
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <limits>
using uint = uint32_t;
int fallbacks = 0;
uint bits(float f) { uint n; memcpy(&n,&f,4); return n; }
float uintBitsToFloat(uint n) { float f; memcpy(&f,&n,4); return f; }
int findMSB(uint n) { assert(n); return 31-__builtin_clz(n); }
float fold3_activation_divide(float x,float y) { ++fallbacks; return x/y; }
''' + helper + r'''
int main() {
  for (int x=0;x<256;++x) {
    volatile float input=float(x), denominator=255.f;
    assert(bits(fold3_u8_divide_255(input))==bits(input/denominator));
  }
  assert(fallbacks==0);
  assert(bits(fold3_u8_divide_255(-0.f))==0x80000000u);
  float edges[]={-1.f,255.5f,1.5f,256.f,1.e-38f,
    std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::quiet_NaN()};
  for(float x:edges) {
    int before=fallbacks;
    float actual=fold3_u8_divide_255(x), expected=x/255.f;
    assert(fallbacks==before+1);
    assert(isnan(expected) ? isnan(actual) : bits(expected)==bits(actual));
  }
}
'''
    with tempfile.TemporaryDirectory() as folder:
      source = Path(folder)/'test.cpp'; source.write_text(program)
      target = Path(folder)/'test'
      subprocess.run([compiler,'-std=c++11','-O2','-ffp-contract=off',str(source),'-o',str(target)],check=True)
      subprocess.run([str(target)],check=True)
    self.assertIn('with_scalar == 1 && const_b == 255.0', added)
