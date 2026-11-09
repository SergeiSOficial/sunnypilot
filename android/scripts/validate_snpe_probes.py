#!/usr/bin/env python3
"""GPU-only SNPE primitive diagnostics; invoke through experiment.py."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from convert_snpe_linux import digest
from validate_snpe_sections import contracts


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('adb', 'serial'): parser.add_argument('--'+name, required=True)
  for name in ('probes', 'converted', 'runtime', 'output'): parser.add_argument('--'+name, required=True, type=Path)
  args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
  remote = '/data/local/tmp/fold3-c74-snpe-models'
  base = [args.adb, '-s', args.serial]
  def adb(*command, check=True):
    return subprocess.run(base+list(command), check=check, capture_output=True, text=True, timeout=180)
  def upload(source, target):
    adb('push', str(source), target)
    if adb('shell', 'sha256sum', target).stdout.split()[0] != digest(source): raise ValueError('Uploaded hash mismatch')
  manifest = json.loads((args.probes/'manifest.json').read_text())
  report = {'qualified': False, 'vehicle_tx_enabled': False, 'execution_domain': 'ADB shell, not app',
            'rtol': 1e-3, 'atol': 1e-4, 'cpu_fallback_enabled': False, 'results': {}, 'files': {}}
  try:
    adb('shell', 'mkdir', '-p', remote)
    for source in (args.runtime/'bin/aarch64-android/snpe-net-run', args.runtime/'lib/aarch64-android/libSNPE.so'):
      upload(source, remote+'/'+source.name); report['files'][source.name] = digest(source)
    adb('shell', 'chmod', '700', remote+'/snpe-net-run')
    for name, contract in manifest['probes'].items():
      model = args.converted/(name+'.dlc')
      # DLC-info evidence is mandatory: converter remaps rank-four inputs to NHWC.
      info = (args.converted/(name+'.info.txt')).read_text()
      shape = contract['input_shape']; rank4 = len(shape) == 4
      actual_shape = [shape[0], shape[2], shape[3], shape[1]] if rank4 else shape
      if contracts(info, 'Input') != {'x': actual_shape}: raise ValueError('DLC input shape not evidenced: '+name)
      expected_shape = contract['output_shape']
      if rank4 and contract['operator'] != 'layout': expected_shape = [expected_shape[i] for i in (0,2,3,1)]
      if contracts(info, 'Output') != {'y': expected_shape}: raise ValueError('DLC output shape not evidenced: '+name)
      raw = args.probes/(name+('-nhwc.raw' if rank4 else '-nchw.raw'))
      upload(model, remote+'/model.dlc'); upload(raw, remote+'/input.raw')
      listing = args.output/'input-list.txt'; listing.write_text('x:='+remote+'/input.raw\n')
      upload(listing, remote+'/input-list.txt')
      for mode in ('gpu', 'gpu_float16'):
        output = remote+'/output'; adb('shell', 'rm', '-rf', output)
        command = ['shell', 'env', 'LD_LIBRARY_PATH='+remote, remote+'/snpe-net-run',
                   '--container', remote+'/model.dlc', '--input_list', remote+'/input-list.txt',
                   '--runtime_order', mode, '--profiling_level', 'detailed', '--output_dir', output]
        start = time.monotonic(); result = adb(*command, check=False)
        key = name+'-'+mode; folder = args.output/key; folder.mkdir()
        (folder/'run.log').write_text(result.stdout+result.stderr)
        entry = {'returncode': result.returncode, 'requested_runtime': mode, 'command': base+command,
                 'wall_seconds_including_initialization': time.monotonic()-start,
                 'dlc_sha256': digest(model), 'input_sha256': digest(raw), 'status': 'execution_failed',
                 'actual_assignment': 'pending_diag_decode'}
        adb('pull', output, str(folder), check=False)
        outputs = list(folder.rglob('y.raw'))
        if result.returncode == 0 and len(outputs) == 1:
          actual = np.fromfile(outputs[0], np.float32)
          with np.load(args.probes/(name+'.npz')) as fixture: expected = fixture['expected'].astype(np.float32)
          if actual.size == expected.size:
            if rank4 and contract['operator'] != 'layout':
              n,c,h,w = expected.shape; actual = actual.reshape(n,h,w,c).transpose(0,3,1,2)
            else: actual = actual.reshape(expected.shape)
            finite = bool(np.isfinite(actual).all()); difference = np.abs(actual-expected)
            mismatch = (~np.isclose(actual, expected, rtol=1e-3, atol=1e-4)) | (~np.isfinite(actual))
            entry.update(status='parity_pass' if finite and not mismatch.any() else 'parity_fail',
                         mismatches=int(mismatch.sum()), elements=int(actual.size),
                         max_abs_error=float(difference.max()), finite=finite)
          else: entry.update(status='invalid_output_size', actual_elements=int(actual.size))
        report['results'][key] = entry
        (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(key, entry['status'], entry.get('mismatches', ''), flush=True)
  finally:
    adb('shell', 'rm', '-rf', remote)
    report['remote_directory_removed'] = adb('shell', 'test', '!', '-e', remote, check=False).returncode == 0
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
  return 0 if report['remote_directory_removed'] and len(report['results']) == 18 and all(x['status'] == 'parity_pass' for x in report['results'].values()) else 1


if __name__ == '__main__': raise SystemExit(main())
