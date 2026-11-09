"""Diagnostic only: measure the controls worker on the phone's big CPU cluster."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from experiment import Device
adb='/Users/sergios/Library/Android/sdk/platform-tools/adb'
serial='RFCR81Y1TBJ'
package='com.samsung.sunnypilot.fold3'
def command(*args):
  return subprocess.check_output([adb,'-s',serial,*args],text=True,timeout=30)
configuration='files/controls-affinity-diagnostic.json'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('report')
parser.add_argument('--cpus',type=int,nargs='+')
args=parser.parse_args()
command('shell','run-as',package,'rm','-f',configuration)
if args.cpus:
  subprocess.run([adb,'-s',serial,'shell','-T','run-as',package,'tee',configuration],
                 input=json.dumps(args.cpus)+'\n',text=True,stdout=subprocess.DEVNULL,check=True,timeout=30)
fixture=None
try:
  Device(serial,adb,package).start_runtime(True,full_chain=True)
  fixture=subprocess.Popen([sys.executable,'-m','openpilot.tools.sim.android_chain_validation','--seconds','30','--device-replay','--steady-inputs','--serial',serial,'--adb',adb,'--report',args.report,'--package',package])
  limit=time.monotonic()+90
  while True:
    r=json.loads(command('exec-out','run-as',package,'cat','files/controlsd.json'))
    manager=json.loads(command('exec-out','run-as',package,'cat','files/manager.json'))
    current=manager.get('processes',{}).get('Controls',{}).get('pid')
    if current and r.get('pid') == current and r.get('android_thread_priority') == -8 and r.get('work_samples',0)>=10:
      break
    if fixture.poll() is not None or time.monotonic()>=limit:
      raise TimeoutError('Fresh running controls worker did not initialize')
    time.sleep(.5)
  affinity=r.get('android_controls_affinity')
  if args.cpus and (not affinity or affinity.get('after')!=sorted(set(args.cpus)) or not affinity.get('diagnostic_only')):
    raise RuntimeError('Self-only diagnostic affinity was not applied')
  pid,tid=str(r['pid']),str(r['android_worker_tid'])
  status=command('shell','run-as',package,'cat',f'/proc/{pid}/task/{tid}/status')
  if ('Tgid:\t'+pid) not in status:
    raise ValueError('Recorded controls TID does not belong to current process')
  Path(args.report+'.affinity.json').write_text(json.dumps({'diagnostic_only':True,
      'pid':r['pid'],'tid':r['android_worker_tid'],'requested':args.cpus,
      'cpu_status':[line for line in status.splitlines() if line.startswith(('Cpus_allowed','Mems_allowed'))],
      'applied':affinity},indent=2)+'\n')
  raise SystemExit(fixture.wait(timeout=180))
finally:
  command('shell','run-as',package,'rm','-f',configuration)
  if fixture is not None and fixture.poll() is None:
    fixture.terminate()
    try: fixture.wait(timeout=15)
    except subprocess.TimeoutExpired:
      fixture.kill();fixture.wait(timeout=10)
