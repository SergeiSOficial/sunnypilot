# Candidate45p reproduction

Run from the repository root on `codex/fold3-android-port`. The immutable
candidate is `android/build/candidates/candidate45p`; earlier candidates must
not be overwritten. This diagnostic APK keeps vehicle TX compiled out, USB
IN-only and physical engagement disabled.

## Build and host checks

```sh
env SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ \
  PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin:$PATH \
  .venv/bin/python -m unittest discover -s android/tests

env SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ \
  PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin:$PATH \
  .venv/bin/python -m unittest opendbc.safety.tests.test_hyundai \
  opendbc.car.hyundai.tests.test_hyundai \
  openpilot.tools.sim.tests.test_android_protocol openpilot.tools.sim.tests.test_metadrive_bridge

.venv/bin/python android/scripts/build_mnn.py
.venv/bin/python android/scripts/prepare_accel_models.py --engine mnn --mnn-patched

cd android
env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  FOLD3_BUILD_PYTHON=/Users/sergios/sunnypilot/.venv/bin/python \
  ./gradlew --no-daemon --console plain -Pfold3CameraCompat=true \
  -Pfold3Qnn=true -Pfold3MnnPatched=true \
  :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest
cd ..
```

The pinned MNN archive, source revision, patch and NDK identity are recorded in
the archived `mnn-build.json`. Rebuild MNN and regenerate converted graphs
explicitly before Gradle when patches or converter inputs change. The conversion checks retain all 795 original
comparisons; these host checks do not qualify MNN execution on Android.

## Install and bind the actual artifacts

The archived application SHA-256 is
`51c3978cff84db0aa1f0dabeadab544c321a339c45d51d60dfada0a4cb4dfbf2`.
The instrumentation SHA-256 is
`b8336ebab92c8bfbe076f45c42c462acb5fc1f35c0072453dc39ef07c7ab3945`.
Use the archived APKs for this result, rather than a subsequent build output.

```sh
ADB=/Users/sergios/Library/Android/sdk/platform-tools/adb
FOLD3_SERIAL=RFCR81Y1TBJ
"$ADB" -s "$FOLD3_SERIAL" install -r -t android/build/candidates/candidate45p/app-debug.apk
"$ADB" -s "$FOLD3_SERIAL" install -r -t android/build/candidates/candidate45p/app-debug-androidTest.apk
.venv/bin/python android/reports/full-model-controls-20261101/verify_installed.py \
  --candidate android/build/candidates/candidate45p --serial "$FOLD3_SERIAL" \
  --output android/reports/full-model-controls-20261101/candidate45p-installed-fold.json
```

Repeat for `emulator-5560`, writing `candidate45p-installed-emulator.json`.
The dedicated AVD is `sunnypilot_fold3_candidate44_api35`, ARM64 Android15.
Shut down the emulator before the matched physical controls timing run.

## Isolated full-model and recovery runs

```sh
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --adb /Users/sergios/Library/Android/sdk/platform-tools/adb \
  --report android/reports/full-model-controls-20261101/fold45p-repeat1-mnn-buffer-isolation.json -- \
  .venv/bin/python android/scripts/validate_models.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --adb /Users/sergios/Library/Android/sdk/platform-tools/adb \
  --profile mnn-opencl-buffer --timeout 900 \
  --report android/reports/full-model-controls-20261101/fold45p-repeat1-mnn-buffer.json

.venv/bin/python android/reports/full-model-controls-20261101/run_validation.py \
  android/reports/full-model-controls-20261101/emulator-candidate45p-validation-driver.json
.venv/bin/python android/reports/full-model-controls-20261101/run_validation.py \
  android/reports/full-model-controls-20261101/fold-candidate45p-validation-driver.json
```

Inventories preserve completed evidence and skip completed rows. For a fresh
repetition, copy the inventory to a new experiment name, set its rows to
`pending`, and change every report/output path; do not overwrite this run.
Each row uses `experiment.py` to stop both app variants and test processes,
verify camera release and unchanged APK identity, remove experiment forwarding,
and require 30 continuous NORMAL seconds with skin temperature at or below
32 degrees C. The same cleanup follows failures.

The controls fixture's functional result is separate from its latency gates.
Compare unchanged raw intervals and legacy complete-cycle counters as well as
schema-2 release-to-completion metrics. Successful injected-input recovery does
not qualify real camera timing, perception or a physical vehicle.

```sh
.venv/bin/python android/reports/full-model-controls-20261101/build_latest_summary.py candidate45p
```

Do not advance to combined visual workloads until full untapped model accuracy,
fresh synchronized independently identified camera inputs, timing and watchdog
gates pass. Cooling optimization remains deferred. Thermal protection stays
active; lens-cover, mounted calibration and Panda/vehicle checks require the
remaining physical work.
