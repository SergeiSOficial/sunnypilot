#!/usr/bin/env python3
"""Install isolated Linux lab tools; never alter services or SDK license consent."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import urllib.request
import zipfile

CLI_URL = 'https://dl.google.com/android/repository/commandlinetools-linux-15859902_latest.zip'
CLI_SHA256 = '4e4c464f145a7512b57d088ac6c278c03c9eea610886b35a5e0804e74eedf583'


def download(url, target, expected):
  if not target.exists():
    temporary = target.with_suffix(target.suffix + '.part')
    with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as output:
      while chunk := source.read(1024 * 1024): output.write(chunk)
    temporary.replace(target)
  digest = hashlib.sha256(target.read_bytes()).hexdigest()
  if digest != expected: raise ValueError('Download hash mismatch: ' + str(target))
  return digest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root', type=Path, required=True)
  args = parser.parse_args(); root = args.root.resolve()
  if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('Requires Linux x86_64')
  downloads = root/'downloads'; downloads.mkdir(parents=True, exist_ok=True)
  tools = root/'tools'; tools.mkdir(exist_ok=True)
  java = tools/'jdk-21'
  if not java.exists():
    # Use the host's authenticated Ubuntu package metadata without installing
    # system packages. A blocked third-party JDK API must not disable TLS checks.
    packages = downloads/'java'; packages.mkdir(exist_ok=True)
    subprocess.run(['apt-get', 'download', 'openjdk-21-jdk-headless', 'openjdk-21-jre-headless'], cwd=packages, check=True)
    java_root = tools/'java-root'; java_root.mkdir(exist_ok=True)
    receipts = {}
    for package in sorted(packages.glob('*.deb')):
      subprocess.run(['dpkg-deb', '-x', str(package), str(java_root)], check=True)
      receipts[package.name] = hashlib.sha256(package.read_bytes()).hexdigest()
    (root/'jdk-source.json').write_text(json.dumps(receipts, indent=2)+'\n')
    java.symlink_to(java_root/'usr/lib/jvm/java-21-openjdk-amd64', target_is_directory=True)
  java_root = tools/'java-root'
  for path in java.resolve().rglob('*'):
    if path.is_symlink():
      target = Path(os.readlink(path))
      private = java_root/str(target).lstrip('/')
      if target.is_absolute() and private.exists():
        path.unlink(); path.symlink_to(private)
  trust = tools/'java-cacerts.jks'
  if not trust.exists():
    pending = trust.with_suffix('.pending')
    if pending.exists(): pending.unlink()
    for certificate in sorted(Path('/etc/ssl/certs').glob('*.pem')):
      subprocess.run([str(java/'bin/keytool'), '-importcert', '-noprompt', '-storepass', 'changeit',
                      '-storetype', 'JKS', '-keystore', str(pending), '-alias', certificate.stem, '-file', str(certificate)], check=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    pending.replace(trust)
  link = java/'lib/security/cacerts'
  if link.is_symlink(): link.unlink()
  if not link.exists(): link.symlink_to(trust)
  sdk = tools/'android-sdk'; sdk.mkdir(exist_ok=True)
  if not (sdk/'licenses/android-sdk-license').exists():
    raise RuntimeError('Stage your existing SDK licenses; this script does not accept licenses')
  cli = sdk/'cmdline-tools/latest'
  archive = downloads/'commandlinetools-linux-15859902_latest.zip'
  download(CLI_URL, archive, CLI_SHA256)
  if not cli.exists():
    with zipfile.ZipFile(archive) as source:
      for member in source.infolist():
        relative = Path(member.filename)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe SDK path')
      source.extractall(sdk)
    # Archive has cmdline-tools as its root; move contents to the versioned root.
    staging = tools/'android-cli-staging'
    (sdk/'cmdline-tools').rename(staging)
    cli.parent.mkdir(); staging.rename(cli)
    for path in (cli/'bin').iterdir(): path.chmod(0o755)
  env = dict(os.environ, JAVA_HOME=str(java), ANDROID_HOME=str(sdk), ANDROID_SDK_ROOT=str(sdk))
  env['PATH'] = str(java/'bin')+':'+env['PATH']
  packages = ['platform-tools', 'platforms;android-35', 'build-tools;35.0.0', 'ndk;28.2.13676358',
              'cmake;3.22.1', 'emulator', 'system-images;android-35;google_apis;x86_64']
  subprocess.run([str(cli/'bin/sdkmanager'), '--sdk_root='+str(sdk), *packages], env=env,
                 stdin=subprocess.DEVNULL, check=True)
  uv = root/'snpe-tools/uv-x86_64-unknown-linux-gnu/uv'
  host = root/'host-venv'
  if not host.exists(): subprocess.run([str(uv), 'venv', '--python', '3.12.16', str(host)], check=True)
  requirements = ['numpy==2.5.3', 'onnx==1.23.2', 'onnxruntime==1.31.1', 'pycapnp==2.1.0',
                  'pyzmq==27.2.0', 'Pillow==12.3.0', 'zstandard==0.25.0', 'cmake==3.31.6',
                  'ninja==1.13.0', 'pytest==8.4.2', 'setuptools==84.2.1', 'Cython==3.2.4',
                  'scons==4.9.1', 'setproctitle==1.3.7', 'pycryptodome==3.23.0',
                  'packaging', 'PyYAML', 'requests', 'tqdm']
  subprocess.run([str(uv), 'pip', 'install', '--python', str(host/'bin/python'), *requirements], check=True)
  freeze = subprocess.check_output([str(uv), 'pip', 'freeze', '--python', str(host/'bin/python')], text=True)
  (root/'host-requirements.txt').write_text(freeze)
  activate = f'''# Source this file; tools remain user-owned and isolated.
export JAVA_HOME='{java}'
export ANDROID_HOME='{sdk}'
export ANDROID_SDK_ROOT='{sdk}'
export PATH='{host}/bin:{java}/bin:{sdk}/platform-tools:{sdk}/emulator:{cli}/bin':"$PATH"
[ ! -f '{root}/activate-native.sh' ] || . '{root}/activate-native.sh'
'''
  (root/'activate-lab.sh').write_text(activate)
  report = {'vehicle_tx_enabled': False, 'qualified': False, 'sdk_packages': packages,
            'java_version': subprocess.check_output([str(java/'bin/java'), '-version'], stderr=subprocess.STDOUT, text=True),
            'kvm_writable': os.access('/dev/kvm', os.W_OK), 'cli_sha256': CLI_SHA256}
  (root/'lab-tools.json').write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
