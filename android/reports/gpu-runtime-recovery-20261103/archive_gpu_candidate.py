"""Archive using the established recipe, including the new Vulkan identity."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
name = sys.argv[1]
subprocess.run([sys.executable, str(root / 'android/reports/full-model-controls-20261101/archive_candidate.py'), name], check=True)
folder = root / 'android/build/candidates' / name
shutil.copy2(root / 'android/build/native/accelerators/ncnn-patched/build.json', folder / 'ncnn-build.json')
shutil.copy2(root / 'android/patches/ncnn-explicit-casts.patch', folder / 'ncnn-explicit-casts.patch')
shutil.copytree(root / 'android/app/build/model-reference/accelerators/ncnn', folder / 'graphs/ncnn')
shutil.copy2(root / 'android/build/native/accelerators/ncnn-patched/install/lib/libncnn.so', folder / 'provider-libraries/libncnn.so')
record = json.loads((folder / 'artifacts.json').read_text())
for directory in ('graphs', 'provider-libraries'):
    for path in sorted((folder / directory).rglob('*')):
        if path.is_file():
            record['files'][str(path.relative_to(folder))] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
for filename in ('ncnn-build.json', 'ncnn-explicit-casts.patch', 'mnn-build.json', 'mnn-half-cast.patch'):
    path = folder / filename
    record['files'][filename] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
(folder / 'artifacts.json').write_text(json.dumps(record, indent=2) + '\n')
