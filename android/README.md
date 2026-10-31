# Fold 3 Android runtime port

This branch runs upstream **controlsd, dmonitoringd and the landscape Raylib UI**
in separate Android processes, with bounded supervision, native Params and
msgq/VisionIPC. The debug ADB bridge runs the phone's steering controller in a
host MetaDrive world. **The complete visual driving stack is not operational.**
Vehicle transmission is compiled out and engagement is unconditionally blocked.
See [VALIDATION.md](VALIDATION.md) for passed, failed and blocked acceptance checks.

Branch: `codex/fold3-android-port`, based on
`6f7a2c94c10dffcf9a83b120ab1e6985e4496868`.
Target: stock SM-F926B, Android 15/API 35, ARM64; right-hand-drive 2019 Hyundai
Ioniq PHEV, factory ACC and red Panda. Installation preserves stock Android.

## Build on the Mac

Tested: JDK 17, NDK 28.2.13676358, CMake 3.31.6, build tools 35.0.0,
Gradle 8.9, AGP 8.5.2, Chaquopy 17.2.0 and host Python 3.12.
Install Git LFS, uv and Cap'n Proto **1.0.1** (`capnp`, `capnpc-c++` on PATH).
Then, from the repository root:

```sh
git switch codex/fold3-android-port
git lfs install --local
git submodule update --init --recursive
git lfs pull
git submodule foreach --recursive 'git lfs pull'
uv sync --frozen --extra testing --extra tools
uv pip install --python .venv/bin/python -r android/requirements-host.txt

export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME="$HOME/Library/Android/sdk"
export FOLD3_BUILD_PYTHON="$PWD/.venv/bin/python"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
"$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" \
  'platform-tools' 'platforms;android-35' 'build-tools;35.0.0' \
  'ndk;28.2.13676358' 'cmake;3.31.6'
.venv/bin/python android/scripts/prepare_native.py
.venv/bin/python android/scripts/model_reference.py
.venv/bin/python android/scripts/model_reference.py --fp32
cd android
# Remove an existing incremental APK before final packaging to avoid ZIP slack.
../.venv/bin/python -c 'from pathlib import Path; Path("app/build/outputs/apk/debug/app-debug.apk").unlink(missing_ok=True)'
./gradlew --no-daemon --console plain \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
```

`uv sync` can remove the supplemental packages. Reinstall the requirements after
syncing. Source preparation verifies archive checksums and Git pins, generates
Cereal C++ headers and Cython/CFFI sources, and patches disposable exports under
`android/build`. Required changes to msgq, tinygrad, Cereal, Cap'n Proto and Raylib
are kept in this repository. Pinned submodule checkouts remain unchanged.
Native modules are cross-compiled for Android Bionic; desktop binaries are not
packaged. The shared Raylib library is used by both NativeActivity and pyray.

The APK contains Python 3.12, NumPy 1.26.2, native Params, pycapnp, msgq/VisionIPC,
libusb, pyzmq/libzmq, zstandard, setproctitle and Raylib/CFFI. NumPy 1.26.2 does
**not** satisfy upstream's NumPy >=2 requirement; only the exercised subset is
validated. With optional FP32 assets this debug APK is about 230 MiB after fresh
packaging. An earlier incremental package contained 214 MiB of unused ZIP space;
repackaging removed it while preserving every ZIP entry's bytes.
The FP32 assets are diagnostic experiments, not accepted driving models.
Third party licenses are packaged as assets.

Outputs (debug key signed):

- `android/app/build/outputs/apk/debug/app-debug.apk`
- `android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`

## Install, run and inspect

Select the device explicitly. These commands also work with `emulator-5560`:

```sh
export FOLD3_SERIAL=RFCR81Y1TBJ
adb -s "$FOLD3_SERIAL" install -r android/app/build/outputs/apk/debug/app-debug.apk
adb -s "$FOLD3_SERIAL" install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb -s "$FOLD3_SERIAL" shell am instrument -w -r \
  ai.sunnypilot.fold3.test/androidx.test.runner.AndroidJUnitRunner
adb -s "$FOLD3_SERIAL" shell am force-stop ai.sunnypilot.fold3
adb -s "$FOLD3_SERIAL" shell am start -n ai.sunnypilot.fold3/.MainActivity \
  --ez run_runtime true --ez raylib true
```

Grant camera/location permissions through Android when needed. The foreground
manager supervises separate controls, monitoring and IMU service processes.
Physical controls wait for real `CarParams` and `CarParamsSP`; there is no fake
fingerprint. DM waits for driver-model output. Raw IMU messages and Camera2
VisionIPC frames are explicitly invalid until mounting/camera calibration is
implemented and verified. The Raylib UI has Android connectivity/application
settings and a permanent transmission-disabled banner. UI backgrounding and
Surface changes do not restart the driving workers. Finishing the UI creates a
fresh renderer on the next launch, preserving the service processes.
EGL falls back when four-sample antialiasing is unavailable and refuses an
invalid window surface. Lifecycle validation checks the visible transmission
banner as well as frame delivery, worker PIDs and requested landscape sizes.

Stop the whole app before switching between physical and simulation sessions.
Diagnostics and model benchmarks are started from the launcher buttons, or with
`--ez run_probe true`, `--ez run_models true --es model_profile baseline`, and
`--ez run_probe true --ez soak true --ez soak_only true`. The soak requests 30
minutes of camera capture; it does not run the full combined driving workload.
Model profiles also include `image-fp32`, `beam1`, `fp32` and `fp32-image`.
The first two retain FP16 types embedded in ONNX; the latter two promote FP16
types and weights to FP32 and compare with their own host reference.

Reports live in private app storage:

```sh
adb -s "$FOLD3_SERIAL" exec-out run-as ai.sunnypilot.fold3 cat files/manager.json
adb -s "$FOLD3_SERIAL" exec-out run-as ai.sunnypilot.fold3 cat files/controlsd.json
adb -s "$FOLD3_SERIAL" exec-out run-as ai.sunnypilot.fold3 cat files/ui-report.json
adb -s "$FOLD3_SERIAL" exec-out run-as ai.sunnypilot.fold3 cat files/camera-report.json
adb -s "$FOLD3_SERIAL" exec-out run-as ai.sunnypilot.fold3 cat files/model-report.json
```

Check timestamps and PIDs: reports persist across installations and can describe
an earlier run. Committed reports retain qualification failures as well as passes.
No camera images, audio recordings or location tracks are retained.
USB permission wraps Android's authorized descriptor in native libusb. Only
identity/version IN reads exist: no CAN OUT, safety mode changes, reset or flash.
Panda firmware compatibility and receive-only CAN integration remain blocked.

## Emulator and MetaDrive

```sh
"$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" 'system-images;android-35;google_apis;arm64-v8a'
echo no | "$ANDROID_HOME/cmdline-tools/latest/bin/avdmanager" create avd \
  --name sunnypilot_fold3_api35 --package 'system-images;android-35;google_apis;arm64-v8a'
"$ANDROID_HOME/emulator/emulator" -avd sunnypilot_fold3_api35 -port 5560 \
  -no-window -no-audio -no-boot-anim -no-snapshot -gpu swiftshader_indirect -memory 4096 -cores 2
uv pip install --python .venv/bin/python -r android/requirements-simulator.txt
adb -s "$FOLD3_SERIAL" shell am force-stop ai.sunnypilot.fold3
adb -s "$FOLD3_SERIAL" shell am start -n ai.sunnypilot.fold3/.MainActivity \
  --ez run_runtime true --ez simulation true --ez raylib true
.venv/bin/python -m openpilot.tools.sim.android_validation --serial "$FOLD3_SERIAL" \
  --metadrive --duration 30 --fault stale --report /tmp/fold3-metadrive.json
.venv/bin/python android/scripts/validate_lifecycle.py --serial "$FOLD3_SERIAL" \
  --report /tmp/fold3-lifecycle.json
# Only on the dedicated emulator: also add --resize to validate display changes.
.venv/bin/python android/scripts/validate_adb_faults.py --serial "$FOLD3_SERIAL" \
  --report /tmp/fold3-adb-faults.json
.venv/bin/python android/scripts/validate_model_reference.py \
  --report /tmp/fold3-host-original-parity.json
# Original parity currently fails (nonzero exit). FP32 uses a different reference:
.venv/bin/python android/scripts/validate_model_reference.py --fp32 \
  --report /tmp/fold3-host-fp32-parity.json
```

The debug-only ADB transport uses an app/ADB-UID restricted socket, a strict
Cereal service allowlist, bounded packets, ordered sequences and input-age checks.
It cannot relay CAN, Panda messages or Params. Simulation uses separate app Params
and IPC directories. Host validation creates temporary Params/IPC namespaces
before constructing MetaDrive. Factory ACC remains host simulated; steering
below the pinned 14.30528 m/s minimum stays inactive. Cancel, stale data and
transport disconnect withdraw both upstream and sunnypilot active states.
Missing model, driver or vehicle state also withdraws activity with a live
connection. The fault script checks these guards and rejects ten malformed or
disallowed session types without restarting the driving workers.

The fixture feeds synthetic perception, driver and calibration messages. Its
steering plant maps torque to angle and is **not** a Hyundai EPS model. A passing
run proves Android controller/transport integration only. For an existing host
simulator/perception stack, `python -m openpilot.tools.sim.android_bridge --help`
describes the Cereal relay. Start every host simulator/perception process with
the same `OPENPILOT_PREFIX=fold3_sim_<name>` and a dedicated `PARAMS_ROOT`, before
importing Cereal or Params. The relay rejects the default host namespace.

See [CAMERAS.md](CAMERAS.md), [PERFORMANCE.md](PERFORMANCE.md) and
[VALIDATION.md](VALIDATION.md) for the physical blockers and remaining work.
