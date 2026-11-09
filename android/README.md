# Fold 3 Android runtime port

Latest local artifact: [candidate66 convolution residual recovery](reports/weight-residual-recovery-20261108/README.md). The matching APK is verified on the Fold 3 and ARM64 Android 15 emulator. All 795 original-reference comparisons, 251 host tests and 30 instrumentation tests on each device pass. Inference p99 is 202.939 ms for driver and 383.425 ms for road; all timing samples exceed 50 ms. The shortcut is not accepted as an overall performance improvement. Fold controls work p99 is 2.256 ms; complete-cycle p99 is 12.229 ms and still fails. Three-camera transport works, but freshness and independent rear identity remain unqualified. The host-GPU MetaDrive repeat passes; mounted calibration and physical vehicle checks remain blocked. TX is compiled out, engagement disabled and cooling optimization deferred.

This branch hosts controlsd, driver monitoring, localization, calibration,
parameter/torque/delay learning, radar, planning, vehicle state, selfdrive state
and the landscape Raylib UI in supervised Android processes. Kalman and MPC
bindings are cross-compiled for Bionic. Camera diagnostics deliver candidate
rear and driver images through native VisionIPC; the stock Fold 3 compatibility
path uses Samsung multiview preview/thumbnail transport. The debug ADB bridge supports both
MetaDrive controller tests and a separate sources-only process-chain test.
**The complete visual driving stack is not qualified.** The strict OpenCL
reference in candidates35/40 passes the complete numerical gate on the Fold 3,
with exact outputs across all three seeds, recurrent sequences and camera replay.
The earlier [candidate45zc delivery](reports/full-model-controls-20261101/candidate45zc-delivery.md)
records 795 passing MNN/OpenCL original-reference comparisons on the phone and
separate device and performance results.
Independent rear image identity, freshness and sustained performance still
require qualification.
See the [blocker-fix evidence](reports/blocker-fixes/README.md).
Vehicle transmission is compiled out and engagement is unconditionally blocked.
See [VALIDATION.md](VALIDATION.md) for passed, failed and blocked acceptance checks.

The earlier numerical-validation artifact is
[candidate45zc](build/candidates/candidate45zc/app-debug.apk), SHA-256
`3e3c44ec71339060d81cb8278498df64edf840dd9705dfe4b3b4e04d8ff83361`.
Its [delivery report](reports/full-model-controls-20261101/candidate45zc-delivery.md)
and [commands](reports/full-model-controls-20261101/candidate45zc-reproduce.md)
bind results to the actual APK, model and provider-library hashes. Earlier
artifacts and their failures remain retained.

## Historical candidate45h delivery

The historical diagnostic artifact is [candidate45](build/candidates/candidate45/app-debug.apk),
SHA-256 `9868e5b9ca7d24feaefd9acfbcdabaf4f5226e85063e03fffec4400aecc1a859`.
The [delivery report](reports/full-model-controls-20261101/README.md),
[validation](reports/full-model-controls-20261101/candidate45-validation.json)
and [commands](reports/full-model-controls-20261101/reproduce.md) retain exact
artifacts and earlier failures. It adds actual MNN/OpenCL tensor tracing,
reproducible numerical repairs, Android absolute-slot controls scheduling and
reporting in a separate process. Derived graphs pass all 795 host comparisons;
MNN's untapped phone models still fail six initial output-array comparisons,
with no CPU fallback or unexpected non-finite values. The remaining recurrent
and replay gate is blocked. Controls work p99 improves from 7.30 to 6.21 ms on
the matched fixture; complete-cycle p99 worsens from 16.61 to 17.29 ms, so
100 Hz timing is not qualified. Both installed APK hashes match on the phone
and emulator. Cooling optimization is deferred; thermal protection and vehicle
safeguards remain active. Candidates35,40,44 remain preserved baselines.

Historical [candidate44](build/candidates/candidate44/app-debug.apk), SHA-256
`f18262c93c2132b1e61a1746af6b2c55dcae8820e3d8a9db0dd84bc5472c6360`,
introduced NPU probes, MNN OpenCL and ncnn Vulkan. Its
[NPU recovery report](reports/npu-recovery-20261101/README.md) retains the
previous HTP, operator support and full-model failures under that APK.

The historical measured artifact is [candidate40](build/candidates/candidate40/app-debug.apk),
SHA-256 `bd42b65a3538b8ad73834e90826eb10a98d8afeacbe9d5aae5877d43a25e30ec`.
The [delivery report](reports/performance-fixes-20261101/README.md),
[installed hashes](reports/performance-fixes-20261101/candidate40-installed.json)
and [artifact record](reports/performance-fixes-20261101/candidate40-artifact.json)
bind results to the actual APK and implementation. All
[795 original-reference comparisons](reports/performance-fixes-20261101/candidate40-model-accuracy-gate.json)
pass exactly. Driver-model p99 improves from candidate35's **560.16 ms** to
**381.54 ms**, but still misses every timed 50 ms deadline. Road timing aborts
at MODERATE thermal status after completing its numerical comparisons.
QNN's first failing tensors and CPU partition removal are recorded separately;
QNN remains diagnostic because its numerical accuracy fails. No live inference
backend is qualified. Android controls now share one absolute 10 ms deadline
between input polling and residual pacing; the source fixture runs at 99.83 Hz,
but complete-cycle p99 remains 18.29 ms. Combined preview delivery p99 improves
to 119.37 ms and still fails freshness. Camera acquisition, conversion,
readback and publication remain separately measured with unchanged freshness
and synchronization limits. See the
[changes](reports/performance-fixes-20261101/candidate40-changes.md) and
[reproduction commands](reports/performance-fixes-20261101/candidate40-reproduce.md).
Candidate35 and subsequent rejected experiments remain archived in the reports.

Branch: `codex/fold3-android-port`, based on
`6f7a2c94c10dffcf9a83b120ab1e6985e4496868`.
Target: stock SM-F926B, Android 15/API 35, ARM64; right-hand-drive 2019 Hyundai
Ioniq PHEV, factory ACC and red Panda. Installation preserves stock Android.

## Build on the Mac

Tested: JDK 17 launcher (this Mac's Gradle daemon override uses JDK 21),
NDK 28.2.13676358, CMake 3.31.6, build tools 35.0.0,
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
export PATH="$PWD/.venv/bin:$ANDROID_HOME/platform-tools:$PATH"
"$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" \
  'platform-tools' 'platforms;android-35' 'build-tools;35.0.0' \
  'ndk;28.2.13676358' 'cmake;3.31.6'
.venv/bin/python android/scripts/prepare_native.py
.venv/bin/python android/scripts/model_reference.py
.venv/bin/python android/scripts/model_reference.py --fp32
.venv/bin/python android/scripts/model_trace.py
.venv/bin/python android/scripts/model_trace.py --start 40 --limit 89 \
  --directory trace-tail --model driving_supercombo
.venv/bin/python android/scripts/model_trace.py --optimized --directory trace-fused --limit 80
# Downloads checksum-pinned route assets; requires ffmpeg for camera decoding.
.venv/bin/python android/scripts/prepare_model_sequences.py
cd android
./gradlew --no-daemon --console plain -Pfold3CameraCompat=true \
  :app:cleanPackageDebug :app:assembleDebug :app:assembleDebugAndroidTest \
  :app:testDebugUnitTest
```

`uv sync` can remove the supplemental packages. Reinstall the requirements after
syncing. Source preparation verifies archive checksums and Git pins, generates
Cereal C++ headers and Cython/CFFI sources, and patches disposable exports under
`android/build`. Required changes to msgq, tinygrad, Cereal, Cap'n Proto and Raylib
are kept in this repository. Pinned submodule checkouts remain unchanged.
Native modules are cross-compiled for Android Bionic; desktop binaries are not
packaged. The shared Raylib library is used by both NativeActivity and pyray.

The APK contains Python 3.12, NumPy 2.5.3, native Params, pycapnp, msgq/VisionIPC,
libusb, ONNX Runtime 1.31.1, pyzmq/libzmq, zstandard, setproctitle, Raylib/CFFI
and the native Kalman/MPC bindings. NumPy is built from its checksum-pinned
source with the target Python ABI; affected bindings use its NumPy 2 headers.
FFT, linear algebra, Kalman and MPC checks pass on both devices. Artifact sizes
and hashes are recorded in the validation report.
The FP32 assets are diagnostic experiments, not accepted driving models.
Third party licenses are packaged as assets.
ONNX Runtime 1.31.1 is built from pinned source during native preparation, using
`android/scripts/build_ort.py`. Optional KleidiAI/SME and SVE kernels are disabled:
the supplied ARM64 AAR traps an SME instruction on the API 35 emulator. The AAR
remains checksum pinned for its headers and notices; its executable library is
replaced by the source build. `android/build/native/onnxruntime-1.31.1/build.json`
records the source/submodule pins, toolchain, command and library hash. The first
build can take tens of minutes; subsequent builds reuse the CMake cache.
FP16 is enabled consistently across all native translation units because mixing
Eigen half representations changes their calling convention. The native entry
point rejects devices lacking the required scalar/vector FP16 capabilities.
The reproducible `android/patches/onnxruntime-fp16-groups.patch` fixes byte/element
arithmetic in grouped-weight packing and offsets grouped biases correctly.
Six exact FP16 operator checks pass on each device. Original driving outputs
are now finite with this CPU backend, but original-model parity still fails.
The receive-only Panda decoder is generated from the pinned Panda Python and C
sources, retaining checksum checks even when Python assertions are optimized out.

Outputs (debug key signed):

- `android/app/build/outputs/apk/debug/app-debug.apk`
- `android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`

The build above uses package `com.samsung.sunnypilot.fold3`. Always pass
`--package com.samsung.sunnypilot.fold3` to `experiment.py` **and** its child
validator; regular-package examples below predate this delivery. Fresh commands
and installed hashes are in the [blocker-fix report](reports/blocker-fixes/README.md).

## Samsung multiview diagnostic build

The tested Samsung firmware exposes hidden logical rear groups to a package
prefix allowlist. Build a separate compatibility installation with:

```sh
cd android
# Remove an existing incremental APK before final packaging (as above).
../.venv/bin/python -c 'from pathlib import Path; [p.unlink(missing_ok=True) for p in (Path("app/build/outputs/apk/debug/app-debug.apk"), Path("app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"))]'
./gradlew --no-daemon --console plain -Pfold3CameraCompat=true \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
cd ..
adb -s RFCR81Y1TBJ install -r -g android/app/build/outputs/apk/debug/app-debug.apk
adb -s RFCR81Y1TBJ install -r -g android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --report android/reports/samsung-multiview-isolation.json -- \
  .venv/bin/python android/scripts/validate_cameras.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --ids 0,2,1 --logical 20 --samsung-multiview --seconds 60 \
  --output android/reports/samsung-multiview.json
```

Alternatively, open `com.samsung.sunnypilot.fold3/ai.sunnypilot.fold3.MainActivity`
and select **Test Samsung multiview + front (diagnostic)**. The package is distinct
from the regular installation and every system app. Omit the Gradle property to
build the regular `ai.sunnypilot.fold3` package. Both builds compile out vehicle
transmission. No vendor binaries are redistributed; the phone's installed
Samsung helper is loaded only by the compatibility diagnostic.

The validator exits nonzero while physical-sensor qualification is blocked,
even when `transport_status` passes. Read those fields separately. The diagnostic
saves one thumbnail NV12 image and one opaque-preview PNG in private application
storage for comparison; they are overwritten by the next multiview run and are
never treated as accepted driving inputs. The wrapper stops both package
variants, waits for cooldown and fails any camera-provider restart. See
[CAMERAS.md](CAMERAS.md) for measured results and remaining limitations.

For continuous GL-preview/NV12, thumbnail and front feeds into VisionIPC and the
diagnostic model workers, use the live validator. This mode saves no images:

```sh
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --report android/reports/rear-live-isolation.json -- \
  .venv/bin/python android/scripts/validate_vision.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --samsung-multiview --seconds 60 --report android/reports/rear-live.json
```

The second rear output remains a sensor-unverified candidate, and optical/final
mount calibration is absent. Both model workers and all camera messages remain
invalid. Numerical model faults stop inference until restart; MODERATE thermal
status pauses model work until ten continuous cool seconds have passed. The
host validator returns nonzero for blocked qualification even if capture passes.
See the [current bench report](reports/blocker-fixes/README.md).

Use `validate_instrumentation.py --serial SERIAL --package
com.samsung.sunnypilot.fold3 --report RESULT.json` inside the experiment wrapper
to require all 20 JUnit tests. `am instrument` alone can return zero after a
failed/aborted suite. Secure-keyguard activity tests require manual unlock.
The model validator's `--profile ort-fp32` compares the diagnostic FP32 backend
with its host reference **and** the original goldens; original equivalence
continues to fail and the live runtime retains the original model.

Run each complete original-model OpenCL diagnostic after a separate cooldown:

```sh
for profile in host-fusion-cl-road host-fusion-cl-driver; do
  .venv/bin/python android/scripts/experiment.py --serial "$FOLD3_SERIAL" \
    --package com.samsung.sunnypilot.fold3 \
    --report "/tmp/$profile-isolation.json" -- \
    .venv/bin/python android/scripts/validate_models.py --serial "$FOLD3_SERIAL" \
    --package com.samsung.sunnypilot.fold3 --profile "$profile" --timeout 900 \
    --report "/tmp/$profile.json"
done
```

These profiles check seeds 42/7/123, all 96 recurrent frames, the pinned
60-frame replay and 100 timed calls for one model each. They cannot write live
qualification. A nonzero exit retains failed/blocked results and performs
cleanup. Acceptance still requires both original models, their complete gates,
fresh independent camera inputs and sustained deadlines on the same runtime.

## Focused host checks

After preparing the host environment and native modules:

```sh
CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ \
SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  .venv/bin/python -m unittest discover -s android/tests -v
.venv/bin/python -m unittest \
  openpilot.tools.sim.tests.test_android_protocol
```

These exercise frame pairing, recurrent-state rejection, NV12 layout, pinned
Panda decoding/health gates and the ADB transport contract. Android JNI and
VisionIPC checks run through the device instrumentation command below. The
retained broader host results and their limits are in [VALIDATION.md](VALIDATION.md).

The native msgq checks cover empty registrations and process/thread identity.
Run the actual Android kernel regression on either device through the same
cleanup wrapper (using the compatibility package for the delivered APK):

```sh
.venv/bin/python android/scripts/experiment.py --serial "$FOLD3_SERIAL" \
  --package com.samsung.sunnypilot.fold3 --report /tmp/msgq-isolation.json -- \
  .venv/bin/python android/scripts/validate_msgq_wakeup.py --serial "$FOLD3_SERIAL" \
  --output /tmp/msgq-kernel.json
```

This cross-compiles the pinned implementation with the repository patch,
checks rejection of a live thread in the wrong process, and removes its test
executable afterward. It does not start the driving stack or open a Panda.

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

Qualcomm QNN GPU and HTP are **opt-in diagnostics**, not live driving backends.
The default APK does not bundle them. To prepare the pinned official provider
and runtime (their hashes are checked again by Gradle), then build the separate
camera-compatibility candidate:

```sh
.venv/bin/python android/scripts/prepare_qnn.py
cd android
./gradlew --no-daemon --console plain -Pfold3CameraCompat=true -Pfold3Qnn=true :app:assembleDebug
cd ..
.venv/bin/python android/scripts/experiment.py --serial "$FOLD3_SERIAL" \
  --package com.samsung.sunnypilot.fold3 --report /tmp/qnn-isolation.json -- \
  .venv/bin/python android/scripts/validate_models.py --serial "$FOLD3_SERIAL" \
  --package com.samsung.sunnypilot.fold3 --profile qnn-htp --report /tmp/qnn-models.json
```

Use `qnn-gpu` for the other backend in a separate cooled run. The report lists
provider graph assignments and a no-CPU-fallback smoke test; mere provider
registration or partial assignment is not evidence of full accelerator use.
Original-seed and full recurrent/replay parity, all 100 deadline samples, and
the unchanged physical-camera and vehicle-safety gates must pass independently.
The Qualcomm runtime license and notices accompany the opt-in APK.
Neither QNN accelerator has yet been qualified on the phone. Device discovery,
model assignment, numerical parity, and the performance gates must all pass
before either profile can be considered usable.
On the Fold 3 test run, GPU execution failed original-reference parity for both
models; the driving model also had CPU fallback and missed the 50 ms deadline.
The HTP profiles stopped during ONNX Runtime/QNN graph initialization with an
NHWC layout-assignment error. These diagnostic profiles must not be promoted.

Request a bounded combined diagnostic from a clean, cooled baseline. Use short
correctness runs first; do not repeat the 30-minute workload while parity,
camera identity/freshness or short-run deadlines fail:

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
  -no-window -no-audio -no-boot-anim -no-snapshot -gpu host -memory 4096 -cores 2 \
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
# FP32 is diagnostic and uses a different reference; it cannot qualify live inference:
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
The final Mac validation uses `-gpu host`; the earlier SwiftShader follow-up
hit that System UI dialog. See the [artifact-specific results](reports/blocker-fixes/README.md).

The debug-only ADB transport uses an app/ADB-UID restricted socket, a strict
Cereal service allowlist, bounded packets, ordered sequences and input-age checks.
The controller fixture excludes CAN and Panda input. The separate
`--runtime full-chain` test accepts synthetic passive CAN, IMU and perception
sources; vehicle state, localization, calibration, learning, radar, planning,
selfdrive and controls are produced by Android processes. Neither fixture can
open a physical Panda or transmit to one. It cannot modify arbitrary Params.
The full-chain validator records source rate, exchange work and missed deadlines
alongside complete Android control timing. Add `--device-replay` to run the
bounded source fixture on the device at absolute 100 Hz deadlines; this removes
ADB scheduling from each source cycle. The host-paced mode remains available.
Add `--steady-inputs` to its child command for a timing run without cancellation
or stale-CAN phases; the report marks those fault checks as not run. A fixture
pass establishes connectivity and guards, not valid sustained perception.
An idle ADB connection withdraws inputs every 100 ms; partial/malformed frames
still disconnect. Simulation uses separate app Params
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
