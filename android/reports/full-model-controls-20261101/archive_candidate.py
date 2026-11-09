"""Preserve an immutable candidate and converted-graph identity after a build."""
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[3]
name=sys.argv[1]
folder=root/'android/build/candidates'/name
folder.mkdir(exist_ok=False)
paths={'app-debug.apk':root/'android/app/build/outputs/apk/debug/app-debug.apk',
       'app-debug-androidTest.apk':root/'android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk'}
record={'candidate':name,'vehicle_tx_enabled':False,'diagnostic_backends_qualified':False,'files':{}}
for target,source in paths.items():
 shutil.copy2(source,folder/target)
 record['files'][target]={'sha256':hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),'bytes':source.stat().st_size}
# Gradle strips packaged libraries. Qualification uses those installed bytes,
# so retain their identities separately from the native build products below.
with zipfile.ZipFile(folder/'app-debug.apk') as apk:
 record['packaged_libraries']={name:{'sha256':hashlib.sha256(apk.read(name)).hexdigest(),
    'bytes':apk.getinfo(name).file_size} for name in apk.namelist()
    if name.startswith('lib/arm64-v8a/') and name.endswith('.so')}
(folder/'artifacts.json').write_text(json.dumps(record,indent=2)+'\n')
graphs=folder/'graphs';graphs.mkdir()
for path in (root/'android/app/build/model-reference/accelerators').iterdir():
 if path.suffix in ('.onnx','.mnn') or path.name=='manifest.json': shutil.copy2(path,graphs/path.name)
shutil.copy2(root/'android/build/native/accelerators/mnn-patched/build.json',folder/'mnn-build.json')
shutil.copy2(root/'android/patches/mnn-half-cast.patch',folder/'mnn-half-cast.patch')
shutil.copytree(root/'android/app/build/model-reference/accelerators/mnn',graphs/'mnn')
libraries=folder/'provider-libraries';libraries.mkdir()
for filename in ('libMNN.so','libMNN_CL.so'):
 source=root/'android/build/native/accelerators/mnn-patched/android/libs/arm64-v8a'/filename
 shutil.copy2(source,libraries/filename)
 record['files']['provider-libraries/'+filename]={'sha256':hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),'bytes':source.stat().st_size}
(folder/'artifacts.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
