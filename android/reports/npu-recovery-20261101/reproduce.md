# Reproduce the NPU-first diagnostic build

Run from `/Users/sergios/sunnypilot` on `codex/fold3-android-port`.
The existing pinned native, NumPy 2, original-model, host-fusion, sequence and
camera-replay prerequisites in [the Android build guide](../../README.md) are
required. Do not regenerate goldens or change model hashes/tolerances.

```sh
.venv/bin/python android/scripts/prepare_accelerators.py --sources --tools
.venv/bin/python android/scripts/prepare_htp_probes.py
.venv/bin/python android/scripts/prepare_accel_models.py
.venv/bin/python android/scripts/prepare_gpu_probes.py
```

Downloads, sources and converter wheels are SHA-256 pinned in
`prepare_accelerators.py`. MNN is 3.6.2; ncnn/pnnx is 20261020. MNN conversion
uses `_tools.mnnconvert` directly. The separately hashed MNN float-image graph
must pass all 795 original comparisons before conversion. ncnn conversion
records its explicit batch axes and rejects unconverted operations/ranks.

The arithmetic FP16-boundary rewrite is a **failed host diagnostic** and is
excluded from the APK. Its first divergence can be reproduced with:

```sh
.venv/bin/python android/scripts/trace_accelerator_precision.py
```

Build the same Android configuration:

```sh
cd android
env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  FOLD3_BUILD_PYTHON=/Users/sergios/sunnypilot/.venv/bin/python \
  ./gradlew --no-daemon --console plain -Pfold3CameraCompat=true -Pfold3Qnn=true \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
```

Candidate44 is archived in `android/build/candidates/candidate44`, including
both APKs and provider/native library hashes in `artifacts.json`. Install both
with `adb -s SERIAL install -r -t APK`. Verify each installed `base.apk` using
`pm path PACKAGE` and `sha256sum`; compare to the archive before any run.
The package is `com.samsung.sunnypilot.fold3`; its test package adds `.test`.
The phone serial is `RFCR81Y1TBJ`.

Each isolated accelerator experiment uses this template from the repo root:

```sh
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --adb /Users/sergios/Library/Android/sdk/platform-tools/adb \
  --report android/reports/npu-recovery-20261101/NEW-isolation.json -- \
  .venv/bin/python android/scripts/validate_models.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --adb /Users/sergios/Library/Android/sdk/platform-tools/adb \
  --profile qnn-htp-probes --timeout 900 \
  --report android/reports/npu-recovery-20261101/NEW-models.json
```

Use unique report names. Run profiles sequentially: `qnn-htp-probes`,
`accel-probes`, `mnn-opencl`, `ncnn-vulkan`. The wrapper stops app/test
processes, verifies camera release and stable provider PIDs, cleans experiment
forwarding, and requires 30 continuous NORMAL seconds with skin <=32 C.
It verifies the installed APK before/after each experiment. An isolation pass
does not change a failed accuracy or performance verdict.

The phone/emulator `*-candidate44-validation-driver.json` inventories contain
the exact ordered fixture, instrumentation, IPC, lifecycle and fault commands.
Pass each `command` array directly to `subprocess.run` without a shell. Stop
the sequence if isolation fails; retain every raw validation failure. These
debug fixtures cannot qualify perception or vehicle behavior. The included
`run_validation.py` preserves completed rows. To repeat a validation, copy an
inventory under a new name and reset its rows to `status: pending`, keeping the
original evidence. For example, after reviewing its command arrays:

```sh
.venv/bin/python android/reports/npu-recovery-20261101/run_validation.py \
  android/reports/npu-recovery-20261101/emulator-candidate44-validation-driver.json
```

The initial MetaDrive commands in that inventory used a resolved base Python
path and failed to import `capnp` before simulation. The separate
`emulator-candidate44-retry1-validation-driver.json` retains the corrected
commands using `.venv/bin/python`; use those commands for the MetaDrive cases.
Keep the virtual-environment executable path rather than resolving its symlink.

The old `sunnypilot_fold3_api35` AVD has a 6 GB data partition. Candidate44's
update failed for insufficient space, including after removing only the two
port packages. Those failed installations are retained. A new dedicated 12 GB
AVD was created while preserving the old AVD. Create it once:

```sh
env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  /Users/sergios/Library/Android/sdk/cmdline-tools/latest/bin/avdmanager \
  create avd --name sunnypilot_fold3_candidate44_api35 \
  --package 'system-images;android-35;google_apis;arm64-v8a' --device pixel_3_xl
```

Set `disk.dataPartition.size=12G` in that AVD's `config.ini`. Start it after
phone accelerator measurements:

```sh
/Users/sergios/Library/Android/sdk/emulator/emulator \
  -avd sunnypilot_fold3_candidate44_api35 -port 5560 -partition-size 12288 \
  -no-window -no-audio \
  -no-boot-anim -no-snapshot -gpu host -memory 4096 -cores 2 \
  -camera-back emulated -camera-front emulated
```

Wait for `sys.boot_completed=1`, install the same two APKs and verify hashes
on `emulator-5560`. The exact initial camera/location/notification permission
grants and emulator keyguard setup are in `candidate44-emulator-setup.json`;
apply them before camera/runtime tests on a fresh install. Emulator fixtures
test recovery and correctness; measure
accelerator execution only on the physical phone. Stop the app and emulator
after validation. Never transmit to a physical Panda from these fixtures.

Focused host checks:

```sh
env CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ \
  SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  .venv/bin/python -m unittest discover -s android/tests
```

The unchanged Hyundai/Panda safety and simulator checks were also run in this
recovery series (3,940 tests, 356 skipped); see `host-checks.json` for the command
scope and the initial default-compiler failure and successful Xcode retry.
No cooling settings, firmware, original models, goldens or tolerances changed.
