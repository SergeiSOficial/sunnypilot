"""Execute only the marked linear shader on a software Vulkan emulator."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'android/scripts'))
sys.path.insert(0,str(ROOT/'android/app/src/main/python'))
from experiment import Device
from fold3_model_trace import compare

sdk = Path.home()/'Library/Android/sdk'
native = ROOT/'android/build/native/accelerators/ncnn-patched'
stamp = json.loads((native/'build.json').read_text())
assert stamp['patch_sha256'] == hashlib.sha256((ROOT/'android/patches/ncnn-explicit-casts.patch').read_bytes()).hexdigest()
assert stamp['recipe_sha256'] == hashlib.sha256((ROOT/'android/scripts/build_ncnn.py').read_bytes()).hexdigest()
toolchain = sdk/'ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64'
folder = HERE/'vulkan-linear-probe'
folder.mkdir(exist_ok=True)
cpp = HERE/'linear_vulkan_probe.cpp'
binary = folder/'f3linearprobe'
subprocess.run([str(toolchain/'bin/aarch64-linux-android35-clang++'),'-std=c++17','-O2',
 '-I'+str(native/'install/include/ncnn'),str(cpp),'-L'+str(native/'install/lib'),'-lncnn','-o',str(binary)],check=True)
model = onnx.load(ROOT/'android/app/build/model-reference/accelerators/dmonitoring_model-mnn-io.onnx')
node = next(n for n in model.graph.node if n.name=='node_linear')
weights = {v.name:numpy_helper.to_array(v) for v in model.graph.initializer if v.name in node.input}
(folder/'model.param').write_text('7767517\n2 2\nInput input 0 1 in\nInnerProduct linear_100 1 1 in out 0=512 1=1 2=655360 9=99\n')
with (folder/'model.bin').open('wb') as stream:
 stream.write(b'\x00'*4)
 stream.write(np.ascontiguousarray(weights[node.input[1]],dtype=np.float32).tobytes())
 stream.write(np.ascontiguousarray(weights[node.input[2]],dtype=np.float32).tobytes())
x=np.fromfile(ROOT/'android/reports/gpu-runtime-recovery-20261103/ncnn-driver-seed42-reduction/trace-440.bin',np.float32).astype(np.float16).astype(np.float32)
x.tofile(folder/'input.bin')
files=[binary,folder/'model.param',folder/'model.bin',folder/'input.bin',native/'install/lib/libncnn.so',
 toolchain/'sysroot/usr/lib/aarch64-linux-android/libc++_shared.so']
device=Device('emulator-5560',str(sdk/'platform-tools/adb'),'com.samsung.sunnypilot.fold3')
remote='/data/local/tmp/fold3-linear-'+str(time.time_ns())
record={'status':'starting','qualified':False,'diagnostic_only':True,'software_vulkan':True,'vehicle_tx_enabled':False,
 'build':stamp,'probe_sha256':hashlib.sha256(cpp.read_bytes()).hexdigest(),
 'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
created=False
try:
 record['forwards_before']=device.forwards()
 record['before_cleanup']=device.stop()
 record['baseline']=device.cool()
 record['apk_sha256']=device.artifact()
 record['fingerprint']=device.command('shell','getprop','ro.build.fingerprint').strip()
 device.command('shell','mkdir',remote)
 created=True
 for path in files:
  device.command('push',str(path),remote+'/'+path.name)
 device.command('shell','chmod','700',remote+'/f3linearprobe')
 result=subprocess.run([device.adb,'-s',device.serial,'shell','env','LD_LIBRARY_PATH='+remote,
  remote+'/f3linearprobe',remote],capture_output=True,text=True,timeout=60)
 record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
 if result.returncode:
  record['status']='blocked' if 'No Vulkan device' in result.stderr else 'failed'
 else:
  device.command('pull',remote+'/output.bin',str(folder/'output.bin'))
  # Host arithmetic proof's saved library executes the same input as original ONNX.
  import onnxruntime as ort
  import copy
  from onnx import helper
  inferred=onnx.shape_inference.infer_shapes(model)
  info={v.name:v for v in (*inferred.graph.input,*inferred.graph.value_info,*inferred.graph.output)}
  constants=[copy.deepcopy(v) for v in inferred.graph.initializer if v.name in node.input]
  section=helper.make_model(helper.make_graph([copy.deepcopy(node)],'linear-emulator-diagnostic',
   [copy.deepcopy(info[node.input[0]])],[copy.deepcopy(info[node.output[0]])],constants),
   opset_imports=list(model.opset_import),ir_version=model.ir_version)
  options=ort.SessionOptions()
  options.intra_op_num_threads=2
  expected=ort.InferenceSession(section.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,
   {node.input[0]:x.astype(np.float16).reshape(1,1280)})[0]
  record['parity']=compare(np.fromfile(folder/'output.bin',np.float32).reshape(1,512),expected)
  record['status']='passed' if record['parity']['passed'] else 'failed'
  cpu_result=subprocess.run([device.adb,'-s',device.serial,'shell','env','LD_LIBRARY_PATH='+remote,
    remote+'/f3linearprobe',remote,'cpu-rejection'],capture_output=True,text=True,timeout=30)
  record['cpu_rejection']={'passed':cpu_result.returncode!=0,'returncode':cpu_result.returncode,
    'stdout':cpu_result.stdout,'stderr':cpu_result.stderr}
  if cpu_result.returncode==0:
    record['status']='failed'

except Exception as error:
 record.update(status='failed',error=f'{type(error).__name__}: {error}')
finally:
 if created:
  subprocess.run([device.adb,'-s',device.serial,'shell','pkill','-f','^'+remote+'/f3linearprobe'],check=False)
  device.command('shell','rm','-rf',remote)
 record['after_cleanup']=device.stop()
 record['forwards_removed']=device.clean_forwards(record.get('forwards_before',[]))
 (HERE/'linear-software-vulkan.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
