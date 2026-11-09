"""Compile the actual marked ncnn shader with its pinned GLSL support macros."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'android/build/native/accelerators/ncnn-patched/ncnn-7e36d25cb153092ba87a60530455c7376fedf452/src'
COMPILER = ROOT / 'android/build/native/glslang-host-check/StandAlone/glslang'
import sys
operation=sys.argv[1]
assert operation in ('convolution_packed_1x1s1d1',)
shader = SOURCE / f'layer/vulkan/shader/{operation}.comp'
original = shader.read_text()
records = []
for half_storage, half_arithmetic in ((0, 0), (1, 0), (1, 1)):
  name = f'{operation}-storage{half_storage}-arithmetic{half_arithmetic}'
  text = original.replace('#version 450', '#version 450\n'
    + '#extension GL_EXT_shader_16bit_storage : require\n'
    + '#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n'
    + f'#define NCNN_fp16_storage {half_storage}\n#define NCNN_fp16_arithmetic {half_arithmetic}\n'
    + (SOURCE / 'ncnn_glsl_ext.comp').read_text()
    + '\nlayout(local_size_x=64, local_size_y=1, local_size_z=1) in;\n')
  diagnostic = HERE / (name + '.comp')
  diagnostic.write_text(text)
  result = subprocess.run([str(COMPILER), '-V', '-S', 'comp',
    '-I' + str(shader.parent), '-o', str(HERE / (name + '.spv')), str(diagnostic)],
    capture_output=True, text=True)
  records.append({'mode': name, 'status': 'passed' if result.returncode == 0 else 'failed',
    'original_shader_sha256': hashlib.sha256(shader.read_bytes()).hexdigest(),
    'diagnostic_shader_sha256': hashlib.sha256(diagnostic.read_bytes()).hexdigest(),
    'output': result.stdout + result.stderr, 'execution_validated': False})
(HERE / f'{operation}-shader-compilation.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps(records, indent=2))
if any(record['status'] != 'passed' for record in records):
  raise SystemExit(1)
