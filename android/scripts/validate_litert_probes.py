#!/usr/bin/env python3
"""Run LiteRT GPU-only probes through the guarded experiment wrapper."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from convert_snpe_linux import digest


def main():
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('adb','serial'):p.add_argument('--'+name,required=True)
  for name in ('probe','models','references','libraries','output'):p.add_argument('--'+name,type=Path,required=True)
  p.add_argument('--precision',choices=('fp32','fp16-fp32-accum'),help='Explicit diagnostic GPU precision')
  args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
  conversion=json.loads((args.models/'conversion.json').read_text())
  remote='/data/local/tmp/fold3-c74-litert-probe';base=[args.adb,'-s',args.serial]
  def adb(*command,check=True):return subprocess.run(base+list(command),capture_output=True,text=True,check=check,timeout=180)
  def upload(source,target):
    adb('push',str(source),target)
    if adb('shell','sha256sum',target).stdout.split()[0]!=digest(source):raise ValueError('Uploaded artifact mismatch')
  report={'qualified':False,'vehicle_tx_enabled':False,'execution_domain':'ADB shell, not app','results':{},'files':{}}
  try:
    adb('shell','mkdir','-p',remote)
    for source in (args.probe,*sorted(args.libraries.glob('*.so'))):
      upload(source,remote+'/'+source.name);report['files'][source.name]=digest(source)
    adb('shell','chmod','700',remote+'/'+args.probe.name)
    for model in sorted(args.models.glob('*.tflite')):
      name=model.stem
      contract=conversion['probes'][name]
      if digest(model)!=contract['tflite_sha256']:raise ValueError('Converted model hash mismatch')
      with np.load(args.references/(name+'.npz')) as reference:
        input_path=args.output/(name+'-input.raw');reference['x'].astype(contract.get('input_dtype',reference['x'].dtype)).tofile(input_path);expected=reference['expected']
      upload(model,remote+'/model.tflite');upload(input_path,remote+'/input.raw')
      adb('shell','rm','-f',remote+'/output.raw')
      command=['shell','env','LD_LIBRARY_PATH='+remote]
      if args.precision: command.append('FOLD3_LITERT_PRECISION='+args.precision)
      command += [remote+'/'+args.probe.name,remote+'/model.tflite',remote+'/input.raw',remote+'/output.raw']
      result=adb(*command,check=False);(args.output/(name+'.log')).write_text(result.stdout+result.stderr)
      entry={'returncode':result.returncode,'status':'execution_failed','tflite_sha256':digest(model),'command':base+command}
      for line in result.stdout.splitlines():
        if line.startswith('{'):entry['runtime']=json.loads(line)
      if result.returncode==3:entry['status']='cpu_partition_rejected'
      if result.returncode==0:
        path=args.output/(name+'-output.raw');adb('pull',remote+'/output.raw',str(path))
        actual=np.fromfile(path,contract.get('output_dtype',expected.dtype))
        if actual.size!=expected.size:entry['status']='invalid_output_size'
        else:
          actual=actual.reshape(expected.shape);mismatch=~np.isclose(actual,expected,rtol=1e-3,atol=1e-4)|~np.isfinite(actual)
          entry.update(status='parity_pass' if not mismatch.any() else 'parity_fail',mismatches=int(mismatch.sum()),max_abs_error=float(np.abs(actual-expected).max()))
      report['results'][name]=entry;print(name,entry['status'],flush=True)
      (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
  finally:
    adb('shell','rm','-rf',remote)
    report['remote_directory_removed']=adb('shell','test','!','-e',remote,check=False).returncode==0
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
  return 0 if report['remote_directory_removed'] and len(report['results'])==3 and all(v['status']=='parity_pass' for v in report['results'].values()) else 1


if __name__=='__main__':raise SystemExit(main())
