"""The final singleton activation must retain the reference cube correction."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class CubeArithmeticTest(unittest.TestCase):
  def test_cube_dispatch_accepts_inplace_output_shape(self):
    patch = (ROOT / "android/patches/ncnn-explicit-casts.patch").read_text()
    for filename, channels in (("binaryop.comp", 1280), ("binaryop_pack4.comp", 320)):
      section = patch.split("+++ b/src/layer/vulkan/shader/" + filename, 1)[1].split("--- a/", 1)[0]
      gate = re.search(r"\+\s+if \((psc\([^\n]+)\)\n\+\s+res = afp", section)[1]
      def selected(output, gate=gate):
        # The pinned forward_inplace leaves input constants zero and supplies
        # only output constants. Evaluate the actual shader predicate with it.
        values = dict.fromkeys(("adims", "aw", "ah", "ac"), 0)
        values.update(zip(("outdims", "outw", "outh", "outc"), output, strict=True))
        terms = re.findall(r"psc\((\w+)\) == (\d+)", gate)
        self.assertEqual(len(terms), 4)
        return all(values[key] == int(expected) for key, expected in terms)
      self.assertTrue(selected((3, 1, 1, channels)))
      self.assertFalse(selected((3, 12, 8, channels)))
      self.assertFalse(selected((1, channels, 1, 1)))

  def test_shader_matches_strict_reference_cube(self):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
      self.skipTest("Host C compiler unavailable")
    patch = (ROOT / "android/patches/ncnn-explicit-casts.patch").read_text()
    section = patch.split("+++ b/src/layer/vulkan/shader/binaryop.comp", 1)[1].split("--- a/", 1)[0]
    added = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))
    functions = added[added.index("float fold3_cube_fma"):added.index("// GLSL pow")]
    functions = re.sub(r"\bprecise\s+", "", functions).replace("abs(", "fabsf(")
    source = r'''
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
typedef uint32_t uint;
static uint floatBitsToUint(float x) { uint b; memcpy(&b,&x,4); return b; }
static float uintBitsToFloat(uint b) { float x; memcpy(&x,&b,4); return x; }
''' + functions + r'''
static int check(float x) {
  float square=x*x, error=fmaf(x,x,-square), cube=square*x;
  float tail=fmaf(square,x,-cube), expected=cube+fmaf(error,x,tail);
  return fold3_reference_cube(x) != expected;
}
int main(void) {
  /* Captured replay21 singleton values differ from ordered square*x. */
  float cases[]={-0.37627893686294556f,0.13494861125946045f,-0.08717626333236694f};
  for(int i=0;i<3;i++) if(check(cases[i])) return 1;
  if(fold3_reference_cube(cases[0]) == (cases[0]*cases[0])*cases[0]) return 2;
  uint state=21;
  for(int i=0;i<100000;i++) {
    state=1664525u*state+1013904223u;
    float x=((int)(state%200001u)-100000)*0.0001f;
    if(check(x)) return 3;
  }
  return 0;
}
'''
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      (path / "cube.c").write_text(source)
      subprocess.run([compiler, "-std=c11", "-O2", "-ffp-contract=off", str(path / "cube.c"), "-lm", "-o", str(path / "cube")],
                     check=True, env=os.environ)
      subprocess.run([str(path / "cube")], check=True)
