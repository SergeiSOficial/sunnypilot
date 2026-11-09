"""Hash-bound untapped 795-comparison gate; timing remains conditional."""
import subprocess
from run_devices import HERE, ROOT, PYTHON, ADB, PACKAGE, Device, install, clean

phone = Device('RFCR81Y1TBJ', ADB, PACKAGE)
try:
    install(phone, 'candidate45zw', 'phone-candidate45zw-accuracy')
    command = [PYTHON, 'android/scripts/experiment.py', '--serial', phone.serial, '--adb', ADB,
        '--package', PACKAGE, '--report', str(HERE / 'phone-candidate45zw-models-isolation.json'),
        '--', PYTHON, 'android/scripts/validate_models.py', '--serial', phone.serial, '--adb', ADB,
        '--package', PACKAGE, '--profile', 'ncnn-vulkan', '--timeout', '600',
        '--report', str(HERE / 'phone-candidate45zw-models.json')]
    with (HERE / 'phone-candidate45zw-models.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    print('Model experiment return code:', result.returncode, flush=True)
finally:
    clean(phone, 'phone-candidate45zw-accuracy-final')
