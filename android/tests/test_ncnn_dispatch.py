"""Exercise the actual patched device gate, stock preservation and hard batch cap."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DispatchTest(unittest.TestCase):
  def test_only_exact_fold_gpu_raises_small_stock_limit(self):
    compiler = shutil.which('clang++')
    if not compiler:
      self.skipTest('C++ compiler unavailable')
    patch = (ROOT/'android/patches/ncnn-dispatch-batching.patch').read_text()
    part = patch.split('+++ b/src/fold3_dispatch.h\n',1)[1].split('--- a/',1)[0]
    header = '\n'.join(line[1:] for line in part.splitlines() if line.startswith('+'))
    program = header + r'''
#include <assert.h>
int main() {
  using ncnn::fold3_dispatch_limit;
  assert(fold3_dispatch_limit(0x5143,"Adreno (TM) 660",32768)==262144);
  assert(fold3_dispatch_limit(0x5143,"Adreno (TM) 660",262144)==262144);
  assert(fold3_dispatch_limit(0x5143,"Adreno (TM) 660",1048576)==1048576);
  assert(fold3_dispatch_limit(0x5143,"Adreno (TM) 650",32768)==32768);
  assert(fold3_dispatch_limit(0x5143,"Adreno (TM) 660 extra",32768)==32768);
  assert(fold3_dispatch_limit(0,"Adreno (TM) 660",32768)==32768);
  assert(fold3_dispatch_limit(0x5143,nullptr,32768)==32768);
}
'''
    with tempfile.TemporaryDirectory() as folder:
      source=Path(folder)/'test.cpp';source.write_text(program)
      target=Path(folder)/'test'
      subprocess.run([compiler,'-std=c++11',str(source),'-o',str(target)],check=True)
      subprocess.run([str(target)],check=True)

  def test_session_override_is_bounded_device_specific_and_default_preserving(self):
    compiler = shutil.which('clang++')
    if not compiler:
      self.skipTest('C++ compiler unavailable')
    patch = (ROOT/'android/patches/ncnn-dispatch-batching.patch').read_text()
    part = patch.split('+++ b/src/fold3_dispatch.h\n',1)[1].split('--- a/',1)[0]
    header = '\n'.join(line[1:] for line in part.splitlines() if line.startswith('+'))
    program = header + r'''
#include <assert.h>
int main() {
  using ncnn::fold3_dispatch_limit;
  for (unsigned int limit : {131072u,262144u,524288u}) {
    assert(fold3_dispatch_limit(0x5143,"Adreno (TM) 660",32768,limit)==limit);
    assert(fold3_dispatch_limit(0x5143,"Adreno (TM) 650",32768,limit)==32768);
  }
  assert(fold3_dispatch_limit(0x5143,"Adreno (TM) 660",32768,1)==262144);
  assert(fold3_dispatch_limit(0x5143,"Adreno (TM) 660",1048576,0)==1048576);
}
'''
    with tempfile.TemporaryDirectory() as folder:
      source=Path(folder)/'test.cpp';source.write_text('#include <initializer_list>\n'+program)
      target=Path(folder)/'test'
      subprocess.run([compiler,'-std=c++11',str(source),'-o',str(target)],check=True)
      subprocess.run([str(target)],check=True)

  def test_adapter_rejects_driver_invalid_limits_and_native_mismatch(self):
    import sys
    from unittest.mock import Mock, patch
    sys.path.insert(0, str(ROOT/'android/app/src/main/python'))
    from fold3_accel import AccelModel, MODEL_OUTPUTS, qualified
    native=Mock()
    metadata={'outputs': {key:{} for key in MODEL_OUTPUTS['driving_supercombo']}}
    with patch.dict(sys.modules, {'fold3_accel_native': native}):
      for limit in (-1, True, 1, 1048576):
        with self.assertRaises(ValueError):
          AccelModel('driving_supercombo.ncnn.param','ncnn-vulkan',metadata,dispatch_limit=limit)
      with self.assertRaises(ValueError):
        AccelModel('dmonitoring_model.ncnn.param','ncnn-vulkan',metadata,dispatch_limit=131072)
      native.create.assert_not_called()
      for limit in (131072,262144,524288):
        native.describe.return_value={'diagnostic_dispatch_limit':limit}
        model=AccelModel('driving_supercombo.ncnn.param','ncnn-vulkan',metadata,dispatch_limit=limit)
        self.assertEqual(native.create.call_args.args[-1],limit)
        self.assertEqual(model.info['diagnostic_dispatch_limit'],limit)
      native.describe.return_value={'diagnostic_dispatch_limit':0}
      with self.assertRaisesRegex(ValueError,'configuration mismatch'):
        AccelModel('driving_supercombo.ncnn.param','ncnn-vulkan',metadata,dispatch_limit=131072)
    self.assertFalse(qualified({'assignment':{'diagnostic_dispatch_limit':131072}},{}))

  def test_accuracy_evidence_cannot_cross_dispatch_settings(self):
    import sys
    sys.path.insert(0, str(ROOT/'android/app/src/main/python'))
    from fold3_accel import _accuracy_identity_passed
    identity={'backend':'ncnn-vulkan','cpu_fallback':False,'name':'driving_supercombo',
              'fingerprint':'device','sha256':{'apk':'a'*64},'diagnostic_dispatch_limit':131072}
    entry={'identity':identity,'status':'passed','rtol':1e-3,'atol':1e-4,
           'assignment':{'no_cpu_fallback_verified':True,'diagnostic_dispatch_limit':131072}}
    self.assertTrue(_accuracy_identity_passed(entry,identity))
    self.assertFalse(_accuracy_identity_passed(entry,{**identity,'diagnostic_dispatch_limit':524288}))
    entry['assignment']['diagnostic_dispatch_limit']=0
    self.assertFalse(_accuracy_identity_passed(entry,identity))
