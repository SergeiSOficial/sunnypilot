#!/usr/bin/env python3
"""Convert explicitly selected ONNX diagnostics; never alter their precision."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from setup_snpe_linux import environment


def digest(path):
  checksum = hashlib.sha256()
  with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''): checksum.update(chunk)
  return checksum.hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root', required=True, type=Path)
  parser.add_argument('--input', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--float-bitwidth', type=int, choices=(16,32), default=32,
                      help='Diagnostic converter precision; each setting needs original-reference validation')
  args = parser.parse_args(); root = args.root.resolve()
  sdk, env = environment(root)
  args.output.mkdir(parents=True, exist_ok=True)
  report = {'qualified': False, 'vehicle_tx_enabled': False, 'precision_rewrites': [],
            'converter_float_bitwidth': args.float_bitwidth, 'models': {}}
  for model in sorted(args.input.glob('*.onnx')):
    target = args.output/(model.stem+'.dlc')
    if target.exists(): raise ValueError('Refusing to overwrite prior converted artifact: '+str(target))
    command = [str(root/'converter-venv/bin/python'), str(sdk/'bin/x86_64-linux-clang/snpe-onnx-to-dlc'),
               '--input_network', str(model.resolve()), '--output_path', str(target.resolve()),
               '--float_bitwidth', str(args.float_bitwidth)]
    start = time.monotonic()
    with (args.output/(model.stem+'.log')).open('w') as log:
      result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    entry = {'command': command, 'source_sha256': digest(model), 'returncode': result.returncode,
             'conversion_seconds': time.monotonic()-start,
             'status': 'converted_unverified' if result.returncode == 0 and target.is_file() else 'failed'}
    if target.is_file(): entry['dlc_sha256'] = digest(target)
    report['models'][model.stem] = entry
    (args.output/'conversion.json').write_text(json.dumps(report, indent=2)+'\n')
    print(model.stem, entry['status'], flush=True)


if __name__ == '__main__': main()
