# Fold 3 Android runtime port

This branch hosts upstream **controlsd, dmonitoringd and the landscape Raylib UI**
in separate Android processes, with bounded supervision, native Params and
msgq/VisionIPC. Camera2 delivers road and driver previews through native VisionIPC;
diagnostic model workers use original ONNX assets and upstream message parsers.
The debug ADB bridge runs the phone's steering controller in a host MetaDrive
world. **The complete visual driving stack is not operational.**
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
../.venv/bin/python -c 'from pathlib import Path; [p.unlink(missing_ok=True) for p in (Path("app/build/outputs/apk/debug/app-debug.apk"), Path("app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"))]'
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
libusb, ONNX Runtime 1.31.1, pyzmq/libzmq, zstandard, setproctitle and Raylib/CFFI. NumPy 1.26.2 does
**not** satisfy upstream's NumPy >=2 requirement; only the exercised subset is
validated. Artifact sizes and hashes are recorded in the validation report.
The FP32 assets are diagnostic experiments, not accepted driving models.
Third party licenses are packaged as assets.
ONNX Runtime 1.31.1 is built from pinned source during native preparation, using
`android/scripts/build_ort.py`. Optional KleidiAI/SME and SVE kernels are disabled:
the supplied ARM64 AAR traps an SME instruction on the API 35 emulator. The AAR
remains checksum pinned for its headers and notices; its executable library is
replaced by the source build. `android/build/native/onnxruntime-1.31.1/build.json`
records the source/submodule pins, toolchain, command and library hash. The first
build can take tens of minutes; subsequent builds reuse the CMake cache.
The receive-only Panda decoder is generated from the pinned Panda Python and C
sources, retaining checksum checks even when Python assertions are optimized out.

Outputs (debug key signed):

- `android/app/build/outputs/apk/debug/app-debug.apk`
- `android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`

## Focused host checks

After preparing the host environment and native modules:

```sh
.venv/bin/python -m unittest \
  discover -s android/tests
.venv/bin/python -m unittest \
  openpilot.tools.sim.tests.test_android_protocol
```

These exercise frame pairing, recurrent-state rejection, NV12 layout, pinned
Panda decoding/health gates and the ADB transport contract. Android JNI and
VisionIPC checks run through the device instrumentation command below. The
retained broader host results and their limits are in [VALIDATION.md](VALIDATION.md).

Run device experiments one at a time per device. The wrapper stops both the app
and instrumentation package, verifies no app processes or active camera clients
remain, waits at least 30 seconds with normal thermal status (and phone skin at
most 32°C), then always stops and verifies cleanup after the workload:

```sh
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --adb "$ANDROID_HOME/platform-tools/adb" \
  --report android/reports/experiment-isolation.json -- \
  .venv/bin/python android/scripts/validate_vision.py \
  --serial RFCR81Y1TBJ --adb "$ANDROID_HOME/platform-tools/adb" \
  --seconds 60 --report android/reports/vision-experiment.json
```

Use the same wrapper for instrumentation and model benchmarks. Cleanup preserves
application data, calibration and caches. Do not restart workers between runs or
use thermal overrides on the physical phone. Model collection is available via
`validate_models.py --serial … --profile ort-cpu-4 --report …`; a completed
benchmark can still fail parity or timing. Read each model's acceptance fields.
The diagnostic `ort-fallback` profile contains hash-checked portable graphs and
three synthetic cases. Export disables host-specific NHWC operators; the
manifest records the resulting numerical failures against the unchanged
original reference. Export success is not a parity pass. This profile is
excluded from the live driving runtime.

Live diagnostic inference uses two non-spinning CPU threads and a 20 Hz maximum
start rate. A native reusable row gather reduces preview preprocessing work and
index storage; the exact-output instrumentation comparison covers padded NV12
and the real driver-model input size. Neither optimization changes model
precision, the 100 ms freshness limit or the 20 ms pairing limit. Isolated timing
and CPU savings do not establish sustained performance; see
[PERFORMANCE.md](PERFORMANCE.md) for live and thermal results.

## Install, run and inspect

Select the device explicitly. These commands also work with `emulator-5560`:

```sh
export FOLD3_SERIAL=RFCR81Y1TBJ
adb -s "$FOLD3_SERIAL" install -r android/app/build/outputs/apk/debug/app-debug.apk
adb -s "$FOLD3_SERIAL" install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
.venv/bin/python android/scripts/experiment.py --serial "$FOLD3_SERIAL" \
  --report /tmp/fold3-instrument-isolation.json -- \
  adb -s "$FOLD3_SERIAL" shell am instrument -w -r \
  ai.sunnypilot.fold3.test/androidx.test.runner.AndroidJUnitRunner
```

Require `OK (15 tests)` in the instrumentation output. An `adb` exit code of
zero alone does not establish that Android's test runner passed.

Grant camera/location permissions through Android when needed. The foreground
manager supervises separate controls, monitoring, IMU, road-model and driver-model
service processes in physical mode. Model benchmarks also use a foreground
service; background-service CPU limits distorted the earlier measurements.
Physical controls wait for real `CarParams` and `CarParamsSP`; there is no fake
fingerprint. DM waits for driver-model output. Raw IMU messages and Camera2
VisionIPC frames are explicitly invalid until mounting/camera calibration is
implemented and verified. Diagnostic NV12 projections are explicitly uncalibrated.
The road worker requires distinct rear sensors 0 and 2; it never duplicates a
single camera to satisfy that requirement. Non-finite predictions are discarded
and clear recurrent state. The Raylib UI has Android connectivity/application
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
The original models can also be benchmarked with `ort-cpu`, `ort-cpu-4` and
`ort-nnapi`. CPU fallback remains possible with the NNAPI provider, so its driver
model timing does not prove DSP/GPU execution. All measured original profiles
still fail numerical acceptance.

Request a bounded combined diagnostic from a clean, cooled baseline:

```sh
.venv/bin/python android/scripts/experiment.py --serial "$FOLD3_SERIAL" \
  --report /tmp/fold3-soak-isolation.json -- \
  .venv/bin/python android/scripts/validate_vision.py --serial "$FOLD3_SERIAL" \
  --seconds 1800 --report /tmp/fold3-live-vision-30min.json
```

This runs road/front capture, Raylib, raw IMU and driver inference. Road inference
and physical controls remain blocked; it is not a complete driving workload.
The UI clears previews whose timestamps are future or older than 100 ms.
Capture checks Android's [thermal status](https://developer.android.com/reference/android/os/PowerManager#THERMAL_STATUS_CRITICAL)
before opening cameras and every second. Critical status (4 or higher) closes
capture and prevents retries. The host validator aborts and force-stops its
diagnostic application on critical heat or terminal capture failure.

The dedicated emulator can test startup rejection and an active-session abort.
This command refuses physical device serials and resets the override on exit:

```sh
.venv/bin/python android/scripts/experiment.py --serial emulator-5560 \
  --report /tmp/emulator-thermal-isolation.json -- \
  .venv/bin/python android/scripts/validate_thermal.py --serial emulator-5560 \
  --report android/reports/emulator-thermal-abort.json
```

It also checks host-validator cleanup and verifies that `stop_vision` works
when Android creates a new launcher activity over the driving UI.

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
USB permission wraps Android's authorized descriptor in native libusb. The
launcher offers receive-only Panda start/stop; the debug launch extra is
`--ez panda_rx true`. Native bindings expose whitelisted vendor IN requests and
CAN endpoint 0x81 only. No CAN OUT, safety-mode change, heartbeat, reset or flash
binding exists. The receiver requires red hardware, exact pinned packet versions
and silent health with every engagement flag clear before reading CAN. It checks
health repeatedly, handles fragmented/checksummed packets and publishes invalid
diagnostic CAN with host-arrival timestamps. Signature bytes are recorded without
claiming firmware qualification. Hardware checks are blocked: the user's Panda
is disconnected.

## Emulator and MetaDrive

```sh
"$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" 'system-images;android-35;google_apis;arm64-v8a'
echo no | "$ANDROID_HOME/cmdline-tools/latest/bin/avdmanager" create avd \
  --name sunnypilot_fold3_api35 --package 'system-images;android-35;google_apis;arm64-v8a'
"$ANDROID_HOME/emulator/emulator" -avd sunnypilot_fold3_api35 -port 5560 \
  -no-window -no-audio -no-boot-anim -no-snapshot -gpu swiftshader -memory 4096 -cores 2 \
  -camera-back emulated -camera-front emulated
uv pip install --python .venv/bin/python -r android/requirements-simulator.txt
run_runtime_check() {
  local fold3_report="$1"
  shift
  .venv/bin/python android/scripts/experiment.py --serial "$FOLD3_SERIAL" \
    --runtime simulation --report "$fold3_report" -- "$@"
}
run_runtime_check /tmp/fold3-metadrive-isolation.json \
  .venv/bin/python -m openpilot.tools.sim.android_validation --serial "$FOLD3_SERIAL" \
  --metadrive --duration 30 --fault stale --report /tmp/fold3-metadrive.json
run_runtime_check /tmp/fold3-lifecycle-isolation.json \
  .venv/bin/python android/scripts/validate_lifecycle.py --serial "$FOLD3_SERIAL" \
  --report /tmp/fold3-lifecycle.json
# Only on the dedicated emulator: also add --resize to validate display changes.
run_runtime_check /tmp/fold3-faults-isolation.json \
  .venv/bin/python android/scripts/validate_adb_faults.py --serial "$FOLD3_SERIAL" \
  --report /tmp/fold3-adb-faults.json
.venv/bin/python android/scripts/validate_model_reference.py \
  --report /tmp/fold3-host-original-parity.json
# Original parity currently fails (nonzero exit). FP32 uses a different reference:
.venv/bin/python android/scripts/validate_model_reference.py --fp32 \
  --report /tmp/fold3-host-fp32-parity.json
```

`--runtime simulation` starts a fresh manager and Raylib only after cooldown,
waits up to 60 seconds for new rendered frames, and always stops them afterward.
`--runtime physical` provides the same setup for physical-namespace component
checks; it does not enable engagement or vehicle transmission.
Ensure the dedicated emulator is awake and unlocked after boot. A visible
System UI ANR dialog paused rendering in retained cold-start attempts. Record
any manual recovery; a later lifecycle pass does not establish unattended startup.

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
