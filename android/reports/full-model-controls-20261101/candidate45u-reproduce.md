# Candidate45t reproduction

Run from the repository root on `codex/fold3-android-port`. The immutable
candidate is `android/build/candidates/candidate45u`; earlier candidates must
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
explicitly before Gradle when patches or converter inputs change. Gradle rejects
a stale native build stamp or graph/provider identity mismatch. The conversion checks retain all 795 original
comparisons; these host checks do not qualify MNN execution on Android.

## Install and bind the actual artifacts

The archived application SHA-256 is
`e34f7d2979bd37b1457b8f83052bdf9436c4b10c491097988078822c87b29319`.
The instrumentation SHA-256 is
`b8336ebab92c8bfbe076f45c42c462acb5fc1f35c0072453dc39ef07c7ab3945`.
Use the archived APKs for this result, rather than a subsequent build output.

```sh
ADB=/Users/sergios/Library/Android/sdk/platform-tools/adb
FOLD3_SERIAL=RFCR81Y1TBJ
"$ADB" -s "$FOLD3_SERIAL" install -r -t -g android/build/candidates/candidate45u/app-debug.apk
"$ADB" -s "$FOLD3_SERIAL" install -r -t -g android/build/candidates/candidate45u/app-debug-androidTest.apk
.venv/bin/python android/reports/full-model-controls-20261101/verify_installed.py \
  --candidate android/build/candidates/candidate45u --serial "$FOLD3_SERIAL" \
  --output android/reports/full-model-controls-20261101/candidate45u-installed-fold.json
```

Repeat for `emulator-5560`, writing `candidate45u-installed-emulator.json`.
The dedicated clean AVD is `sunnypilot_fold3_candidate45_api35`, ARM64 Android15,
720x1480 at density 280, with 4 GB RAM and four cores. The `-g` install option
establishes the granted runtime-permission baseline for these lifecycle fixtures.
The first fresh-install run without it paused beneath a permission dialog; its
two failed lifecycle tests are retained. Explicit grants and before/after state
are in `emulator-candidate45u-permissions-baseline.json`.
Shut down the emulator before the matched physical controls timing run.

## Isolated full-model and recovery runs

Run the road and driver accuracy profiles as separate experiments. Each uses
all applicable initial seeds, recurrent sequences and pinned replay frames.
Every wrapper enforces cleanup, camera release, no experiment forwarding, and
30 continuous NORMAL seconds at skin temperature at or below 32°C before work.
The new profiles do not run accelerator timing or qualify a live model.

```sh
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --adb /Users/sergios/Library/Android/sdk/platform-tools/adb \
  --report android/reports/full-model-controls-20261101/fold45u-mnn-road-accuracy-isolation.json -- \
  .venv/bin/python android/scripts/validate_models.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --adb /Users/sergios/Library/Android/sdk/platform-tools/adb \
  --profile mnn-buffer-accuracy-road --timeout 1200 \
  --report android/reports/full-model-controls-20261101/fold45u-mnn-road-accuracy.json
```

Repeat with `mnn-buffer-accuracy-driver`, replacing `road` with `driver` in
both report paths. Timing profiles `mnn-buffer-timing-driver` and
`mnn-buffer-timing-road` reject absent, partial or changed accuracy evidence
for either model before creating an accelerator session. Use new report paths
for timing or retries. Do not overwrite earlier evidence.

```sh
.venv/bin/python android/reports/full-model-controls-20261101/run_validation.py \
  android/reports/full-model-controls-20261101/fold-candidate45u-validation-driver.json
# Launch the dedicated emulator after physical controls timing completes.
.venv/bin/python android/reports/full-model-controls-20261101/run_validation.py \
  android/reports/full-model-controls-20261101/emulator-candidate45u-granted-validation-driver.json

.venv/bin/python android/reports/full-model-controls-20261101/build_latest_summary.py candidate45u \
  --emulator-inventory android/reports/full-model-controls-20261101/emulator-candidate45u-granted-validation-driver.json \
  --model-evidence android/reports/full-model-controls-20261101/fold45u-mnn-road-accuracy.json \
  --model-evidence android/reports/full-model-controls-20261101/fold45u-mnn-driver-accuracy.json
```

A failed accuracy, freshness, controls timing or watchdog gate blocks combined
visual-load stages. Do not advance to two or 30 minutes on fixture success.
Mounted calibration, independent rear-lens proof and physical Panda/vehicle
checks remain blocked; TX remains disabled regardless of bench results.
