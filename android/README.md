# Fold 3 Android port qualification

This branch contains an installable **qualification app**, not a working driving
port. Vehicle transmission is compiled out. Dual rear capture, sustained camera
operation, model parity and model deadlines failed on the connected Fold 3.
See [VALIDATION.md](VALIDATION.md) for the acceptance matrix and remaining work.

Base: `6f7a2c94c10dffcf9a83b120ab1e6985e4496868`.
Branch: `codex/fold3-android-port`.
Target: stock SM-F926B, Android 15/API 35, ARM64; 2019 Hyundai Ioniq PHEV,
right hand drive, factory ACC and red Panda. No Android replacement, bootloader
changes or Panda flashing are part of these recipes.

## Build on the Mac

Install Git LFS, uv, JDK 17 and the Android SDK. The tested tools are NDK
28.2.13676358, CMake 3.22.1, platform/build tools 35, Gradle 8.9, AGP 8.5.2 and
Chaquopy 17.2.0. Use Python 3.12 to prepare sources. From the repository root:

```sh
git switch codex/fold3-android-port
git lfs install --local
git submodule update --init --recursive
git lfs pull
git submodule foreach --recursive 'git lfs pull'
uv sync --frozen --extra testing --extra tools
uv pip install --python .venv/bin/python numpy==2.5.3 onnx==1.23.2 onnxruntime==1.31.1
.venv/bin/python android/scripts/prepare_native.py
.venv/bin/python android/scripts/model_reference.py

export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME="$HOME/Library/Android/sdk"
export FOLD3_BUILD_PYTHON="$PWD/.venv/bin/python"
"$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" \
  'platform-tools' 'platforms;android-35' 'build-tools;35.0.0' \
  'ndk;28.2.13676358' 'cmake;3.22.1'
cd android
./gradlew --no-daemon --console plain \
  :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest
```

`uv pip install` adds the independent host model reference runtime after the
locked project environment is populated. Running `uv sync` again can remove
these additional packages. It does not change `uv.lock`.

The model assets are the original LFS-managed `dmonitoring_model.onnx` and
`driving_supercombo.onnx`. The reference generator records their SHA-256 hashes,
tensor shapes, seed and host library versions. It uses synthetic inputs with
the upstream RHD traffic convention; it does not establish driving quality.

`prepare_native.py` verifies the pinned msgq checkout and downloads checksum
pinned Python headers, pycapnp and Cap'n Proto, plus a commit pinned libusb
checkout. Patches apply to generated copies under `android/build`, leaving
submodules unchanged. It needs network access on the first run. CMake compiles
ARM64 libusb, msgq/VisionIPC and pycapnp against Android's Python ABI; host Linux
or macOS extension binaries are not packaged. Third party notices are APK assets.

The APK embeds Chaquopy's Python 3.12 and diagnostic NumPy 1.26.2. This NumPy
does **not** satisfy upstream's `>=2.0` requirement. Other native dependencies,
including Params, CAN parsing and Raylib, remain unported. The app starts
diagnostic work in separate `:runtime` and `:models` Android services; this is
not the upstream manager or daemon restart supervision.

Outputs:

- `android/app/build/outputs/apk/debug/app-debug.apk`
- `android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`

These are debug builds signed with the local Android debug key. Clean source
builds are supported; byte-identical APK signatures are not promised.

## Install and validate

From the repository root, select the intended device explicitly:

```sh
export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH"
adb devices -l
export FOLD3_SERIAL=RFCR81Y1TBJ
adb -s "$FOLD3_SERIAL" install -r android/app/build/outputs/apk/debug/app-debug.apk
adb -s "$FOLD3_SERIAL" install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb -s "$FOLD3_SERIAL" shell am instrument -w \
  ai.sunnypilot.fold3.test/androidx.test.runner.AndroidJUnitRunner
adb -s "$FOLD3_SERIAL" shell am start -n ai.sunnypilot.fold3/.MainActivity
```

Accept the camera permission on the phone. The app has buttons for the camera
matrix, a requested 30-minute road/driver camera soak, original-model OpenCL
validation and Panda identity reads. Panda access uses Android's USB permission
dialog. The native USB code only permits identity/version IN requests: CAN
transmission, safety mode changes, reset and flashing are absent. Firmware
compatibility is not yet established, so CAN reception is not started either.

Run one workload at a time, with the activity visible. Stop camera work using
the app button, or stop the whole app with `adb shell am force-stop`. Wait for
installation to finish before launching a run. To start diagnostics through ADB:

```sh
adb -s "$FOLD3_SERIAL" shell am force-stop ai.sunnypilot.fold3
adb -s "$FOLD3_SERIAL" shell am start -n ai.sunnypilot.fold3/.MainActivity --ez run_probe true
# Camera soak only; this is not a combined driving workload:
adb -s "$FOLD3_SERIAL" shell am force-stop ai.sunnypilot.fold3
adb -s "$FOLD3_SERIAL" shell am start -n ai.sunnypilot.fold3/.MainActivity \
  --ez run_probe true --ez soak true --ez soak_only true
```

Model profiles are `baseline`, `image-fp32` (image-buffer experiment), and
`beam1` (kernel search). All retain original ONNX types; the flag `FLOAT16=0`
does not replace FP16 types already present in the model. Compilation can take
several minutes. For example:

```sh
adb -s "$FOLD3_SERIAL" shell am force-stop ai.sunnypilot.fold3
adb -s "$FOLD3_SERIAL" shell am start -n ai.sunnypilot.fold3/.MainActivity \
  --ez run_models true --es model_profile baseline
.venv/bin/python android/scripts/collect_metrics.py --serial "$FOLD3_SERIAL" \
  --process ai.sunnypilot.fold3:models --progress-file model-progress.txt \
  --seconds 1900 --output /tmp/fold3-model-metrics.jsonl
```

The collector records the selected process, rather than aggregate application
memory. Its defaults select the main camera/UI process and `progress.txt`.
The `:models` service needs the explicit options above.

Reports are private app files, accessible with debug `run-as`:

```sh
adb -s "$FOLD3_SERIAL" exec-out run-as ai.sunnypilot.fold3 cat files/camera-report.json > /tmp/fold3-camera.json
adb -s "$FOLD3_SERIAL" exec-out run-as ai.sunnypilot.fold3 cat files/runtime.json > /tmp/fold3-runtime.json
adb -s "$FOLD3_SERIAL" exec-out run-as ai.sunnypilot.fold3 cat files/hardware.json > /tmp/fold3-hardware.json
adb -s "$FOLD3_SERIAL" exec-out run-as ai.sunnypilot.fold3 cat files/model-report.json > /tmp/fold3-model.json
```

Check report start/completion timestamps and status before interpreting results.
The next run replaces the current app report. Committed `reports/` files retain
the actual qualification runs, including failures. There are no raw camera
images, recorded audio or location tracks in these reports.

## Dedicated ARM64 emulator and host tests

```sh
"$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" 'system-images;android-35;google_apis;arm64-v8a'
echo no | "$ANDROID_HOME/cmdline-tools/latest/bin/avdmanager" create avd \
  --name sunnypilot_fold3_api35 --package 'system-images;android-35;google_apis;arm64-v8a'
"$ANDROID_HOME/emulator/emulator" -avd sunnypilot_fold3_api35 -port 5560 \
  -no-window -no-audio -no-boot-anim -no-snapshot -gpu swiftshader_indirect -memory 2048 -cores 2
```

Use `emulator-5560` in place of the physical serial for installation and
instrumentation. The emulator does not expose the phone's OpenCL driver or
prove camera concurrency. No MetaDrive-to-Android bridge is implemented yet.

Focused host checks from the root:

```sh
PATH="$PWD/.venv/bin:$PATH" .venv/bin/scons -j4 --minimal
.venv/bin/python -m unittest opendbc.car.hyundai.tests.test_hyundai \
  msgq.visionipc.tests.test_visionipc msgq.tests.test_messaging
.venv/bin/python -m unittest opendbc.safety.tests.test_hyundai
```

See [CAMERAS.md](CAMERAS.md), [PERFORMANCE.md](PERFORMANCE.md) and
[VALIDATION.md](VALIDATION.md) for measured results and their limits.
