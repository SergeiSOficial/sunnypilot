#!/usr/bin/env python3
"""Inspect/download selected files from pinned official runtime distributions."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

QAIRT_VERSION = '2.47.0.260601'
QAIRT_URL = f'https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/{QAIRT_VERSION}/v{QAIRT_VERSION}.zip'
LITERT_VERSION = '2.3.0'
PINNED_HASHES = {
  'snpe/lib/aarch64-android/libSNPE.so': '6d43b43550e737aedda14d1ac3a85632b60525cd4399e518f260ecd02730059c',
  'litert-2.3.0.aar': '9508c6e9142646f0f2bc1fe68b3925fb470ee7ddd29114540339a04a8ab189dc',
  'litert-gpu-2.3.0.aar': '173a4c3813601f7e9542f4e6a49e237775e2f9ce1e0948f35407098bffb1824f',
}


def checked_digest(name, data):
  digest = hashlib.sha256(data).hexdigest()
  if name in PINNED_HASHES and PINNED_HASHES[name] != digest:
    raise ValueError(f'Pinned runtime hash mismatch: {name}')
  return digest


class RemoteZip(io.RawIOBase):
  """Bounded HTTP ranges avoid storing the multi-GB SDK archive."""
  def __init__(self, url):
    self.url, self.position = url, 0
    self.cache_start, self.cache = 0, b''
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as r:
      self.size = int(r.headers['Content-Length'])

  def seekable(self): return True
  def readable(self): return True
  def tell(self): return self.position

  def seek(self, offset, whence=0):
    target = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
    if not 0 <= target <= self.size: raise ValueError('Invalid remote ZIP offset')
    self.position = target
    return target

  def read(self, count=-1):
    count = self.size-self.position if count < 0 else min(count, self.size-self.position)
    if not count: return b''
    if count > 128*1024*1024: raise ValueError('Oversized remote ZIP read')
    start = self.position
    if self.cache_start <= start and start+count <= self.cache_start+len(self.cache):
      self.position += count
      return self.cache[start-self.cache_start:start-self.cache_start+count]
    # ZIP headers and neighboring SDK headers share a range; avoid three HTTP
    # requests per small file without downloading the entire SDK archive.
    fetch_count = min(max(count, 1024*1024), self.size-start)
    request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{start+fetch_count-1}'})
    with urllib.request.urlopen(request, timeout=60) as r:
      if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{start+fetch_count-1}/{self.size}':
        raise ValueError('Server did not honor exact SDK range')
      data = r.read(fetch_count+1)
    if len(data) != fetch_count: raise ValueError('Truncated SDK range')
    self.cache_start, self.cache = start, data
    self.position += count
    return data[:count]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--list-sdk', action='store_true')
  args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
  report = {'qairt_version': QAIRT_VERSION, 'qairt_url': QAIRT_URL, 'litert_version': LITERT_VERSION,
            'diagnostic_only': True, 'qualified': False, 'files': {}}
  with zipfile.ZipFile(RemoteZip(QAIRT_URL)) as archive:
    names = [n for n in archive.namelist() if ('/include/SNPE/' in n or '/include/zdl/' in n or
      '/include/SNPE_c/' in n or '/lib/aarch64-android/' in n and ('libSNPE' in n or 'libSnpe' in n) or
      n.endswith('/bin/x86_64-linux-clang/snpe-onnx-to-dlc')) and not n.endswith('/')]
    if args.list_sdk:
      print(json.dumps(names, indent=2)); return
    for name in names:
      relative = Path(name).relative_to(f'qairt/{QAIRT_VERSION}')
      if '..' in relative.parts: raise ValueError('Unsafe SDK member')
      target = args.output/'snpe'/relative; target.parent.mkdir(parents=True, exist_ok=True)
      data = archive.read(name)
      key = str(target.relative_to(args.output))
      report['files'][key] = checked_digest(key, data)
      target.write_bytes(data)
  for name in ('litert', 'litert-gpu'):
    url = f'https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/{name}/{LITERT_VERSION}/{name}-{LITERT_VERSION}.aar'
    with urllib.request.urlopen(url, timeout=60) as r: data = r.read(128*1024*1024+1)
    if len(data) > 128*1024*1024: raise ValueError('Oversized runtime AAR')
    target = args.output/f'{name}-{LITERT_VERSION}.aar'
    report['files'][target.name] = checked_digest(target.name, data)
    target.write_bytes(data)
  (args.output/'downloads.json').write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'files': len(report['files']), 'output': str(args.output)}), flush=True)


if __name__ == '__main__': main()
