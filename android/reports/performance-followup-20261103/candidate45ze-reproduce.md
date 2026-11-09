# Candidate45ze reproduction

Run in `/Users/sergios/sunnypilot` on `codex/fold3-android-port`. Preserve all candidate archives and prior failures. Vehicle TX remains compiled out and engagement disabled.

## Build and archive

```sh
.venv/bin/python android/scripts/build_mnn.py
.venv/bin/python android/scripts/prepare_accel_models.py --engine mnn --mnn-patched
cd android
env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  FOLD3_BUILD_PYTHON=/Users/sergios/sunnypilot/.venv/bin/python \
  ./gradlew --no-daemon --console plain -Pfold3CameraCompat=true -Pfold3Qnn=true -Pfold3MnnPatched=true \
  :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest
cd ..
.venv/bin/python android/reports/full-model-controls-20261101/archive_candidate.py candidate45ze
```

Use a fresh candidate name for another build; never overwrite the existing archive. The event-profiler build (`--profile-gpu`) is diagnostic and separate from APK libraries.

## Install and verify

```sh
/Users/sergios/Library/Android/sdk/platform-tools/adb -s RFCR81Y1TBJ install -r -t -g android/build/candidates/candidate45ze/app-debug.apk
/Users/sergios/Library/Android/sdk/platform-tools/adb -s RFCR81Y1TBJ install -r -t -g android/build/candidates/candidate45ze/app-debug-androidTest.apk
.venv/bin/python android/reports/full-model-controls-20261101/verify_installed.py \
  --candidate android/build/candidates/candidate45ze --serial RFCR81Y1TBJ \
  --output FRESH-INSTALLED-FOLD.json
```

Repeat on the dedicated ARM64 Android15 `emulator-5560`. Device hash verification is separate from a successful host build. APK SHA-256: `54a613533c10795ab754e2737d5bf2facfd387dfb2bd58ef43017260dd063a3f`.

## Accuracy, then timing

```sh
.venv/bin/python android/reports/full-model-controls-20261101/run_accuracy_parts.py candidate45ze --models driver road
```

The runner refuses to overwrite prior evidence. All 795 actual-device original-reference comparisons must pass, including recurrent sequences and the pinned camera replay, before timing. Host conversion comparisons and synthetic operator checks cannot substitute for these results.

The reviewed inventories `fold-candidate45ze-accelerator-timing-driver.json`, `fold-candidate45ze-validation-driver.json` and `emulator-candidate45ze-validation-driver.json` contain exact experiment commands. They are stateful evidence inventories; the validation driver runs only pending rows. Use fresh reviewed output paths when repeating experiments. A functional fixture PASS does not imply performance acceptance.

```sh
.venv/bin/python android/reports/full-model-controls-20261101/run_validation.py android/reports/performance-followup-20261103/fold-candidate45ze-accelerator-timing-driver.json
.venv/bin/python android/reports/full-model-controls-20261101/run_validation.py android/reports/performance-followup-20261103/fold-candidate45ze-validation-driver.json
.venv/bin/python android/reports/full-model-controls-20261101/run_validation.py android/reports/performance-followup-20261103/emulator-candidate45ze-validation-driver.json
```

Run these sequentially, without overlapping physical-device or emulator experiments. Every command is wrapped by process/camera/forwarding cleanup and at least 30 continuous NORMAL seconds with phone skin <=32C. Thermal pause and abort rules remain active; no cooling optimization is included. Do not advance to combined accelerated-camera workloads while accuracy, inference timing, camera freshness or watchdog gates fail. Emulator results cannot qualify physical GPU performance or vehicle operation.

## Emulator graphics retry

The first SwiftShader/Vulkan attempt passed instrumentation, then failed two chain startups with Raylib stalled in `end_drawing`. Scheduling was interrupted before lifecycle completion. Preserve its inventory and `candidate45ze-emulator-attempt1-interrupted-cleanup.json`; those are not passing lifecycle evidence. The retry uses the same installed APK, freshly hash-verified after reboot, with:

```sh
/Users/sergios/Library/Android/sdk/emulator/emulator \
  -avd sunnypilot_fold3_candidate45_api35 -port 5560 \
  -no-window -no-audio -no-boot-anim -no-snapshot \
  -gpu host -feature -Vulkan -memory 4096 -cores 4 \
  -camera-back emulated -camera-front emulated
```

Use `emulator-candidate45ze-retry1-validation-driver.json` for the fresh retry inventory. Installed-hash evidence is `candidate45ze-installed-emulator-retry1.json`. After all experiments finish, stop app/test processes, verify camera release and forwarding cleanup, establish the cool NORMAL baseline and shut down the emulator. Generate the artifact-bound summary with `.venv/bin/python android/reports/performance-followup-20261103/summarize_results.py`. The script rejects pending rows and mixed APK identities.
