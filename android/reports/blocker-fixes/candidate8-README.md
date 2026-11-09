# Fold 3 blocker-fix evidence — 2026-10-31

> IPC attribution erratum (candidate10): the old fallback/process-group explanation below was an unconfirmed hypothesis. ARM64 disassembly shows `tkill`, and candidate9 still reproduced the SIGUSR2 failure. Candidate10 preserves the subscriber PID and TID and uses `tgkill`; real-kernel isolation checks and subsequent process-chain runs pass. The exact sender of the historical fatal signal was not captured. See the current [blocker-fix report](README.md).

Branch: `codex/fold3-android-port`, HEAD
`7631d5248cc9efd10b962107aadacbb908a9f14d` plus the uncommitted port fixes.
**Vehicle TX is compiled out and physical engagement is unconditionally disabled.**
The complete visual driving runtime is still unqualified. Passing a simulator
test does not qualify cameras, mounted calibration, Panda firmware or a vehicle.

## Delivered artifact

The final candidate is **candidate8**, package `com.samsung.sunnypilot.fold3`:

| Artifact | SHA-256 |
| --- | --- |
| [Debug APK](../../build/candidates/candidate8/app-debug.apk) — 509,216,338 bytes | `98196287effbeaf809b71b256b243eee77fd5f04c92a720ba6244da0a5f93295` |
| [Instrumentation APK](../../build/candidates/candidate8/app-debug-androidTest.apk) — 4,466,867 bytes | `e7a159b944a4e82f3f38a4d370436e9c5dc6197e6f918026247ec1d38e989d58` |

[Artifact metadata](candidate8-artifact.json) records the original model hashes,
native runtime build and compressed/uncompressed payload sizes. Each device
experiment has a matching `-isolation.json`: installed APK hash before cooldown,
at launch and after cleanup; camera-provider PIDs; app process cleanup; camera
ownership; forwarding cleanup; NORMAL thermal status and skin temperature.
An experiment is invalid if its installed artifact changes.

## Implemented fixes

- Built ARM64 NumPy **2.5.3**, matching the host reference, and rebuilt affected
  native extensions. Packaged Bionic Kalman, pose/car EKF and longitudinal MPC
  dependencies from pinned sources. Native entry points run as libraries.
- Added supervised localization, calibration, parameter, torque, delay, radar,
  planning, vehicle-state and selfdrive processes. The process-chain fixture
  injects sensor/CAN/model sources; the required derived services come from
  these processes. The older MetaDrive fixture injects derived inputs and has
  narrower coverage.
- Fixed two ONNX Runtime 1.31.1 FP16 grouped-convolution defects: packed-weight
  byte offsets and per-group bias offsets. The reproducible patch is
  [`onnxruntime-fp16-groups.patch`](../../patches/onnxruntime-fp16-groups.patch).
  Original CPU model results are now finite. This does **not** establish parity.
- Added intermediate-tensor tracing and exact grouped-Conv checks. Original
  model files, reference outputs and `rtol=1e-3, atol=1e-4` are unchanged. Global
  FP32, portable operators and mixed accumulation remain diagnostic experiments.
- Preserved raw camera timestamps with bracketing elapsed-realtime and monotonic
  samples. REALTIME timestamps are converted between clock domains; UNKNOWN
  timestamps remain unqualified. Invalid capture times never become arrival
  times. See the [Android timestamp contract](https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics#SENSOR_INFO_TIMESTAMP_SOURCE).
- Selected the newest synchronized pair, discarded older queued inputs and
  reused owned staging buffers. Freshness stays **100 ms**, pair skew **20 ms**.
- Added feature-checked GLES3 NV12 conversion with two bounded PBO/fence slots;
  retained the fallback. Conversion, padding and orientation equivalence are
  checked on both devices. Disabled supported noise reduction, sharpening and
  stabilization on the Samsung multiview diagnostic request as well as ordinary
  requests, and recorded the actual result modes.
- Added separate input, compute, publishing, extension, reporting and complete
  cycle measurements. Input wait is reported explicitly; work time cannot stand
  in for the complete 100 Hz cycle.
- Connected passive CAN to receive-only vehicle-state processing. The native
  USB path accepts vendor IN reads and bulk endpoint `0x81`; Android vehicle
  state has no `sendcan` publisher and rejects transmission callbacks before
  initialization/application. No active CAN firmware queries or automatic flashing run.
- Fixed idle ADB relay recovery: withdraw stale inputs while retaining a valid
  idle connection; malformed/partial packets still time out. Factory ACC remains
  separate from simulated steering.
- Repaired the legacy simulator's incomplete model/driver fixture. Its missing
  brake-disengagement probabilities caused upstream driver monitoring to raise
  `IndexError`. The host regression executes the actual driver-monitoring policy;
  the final phone fault tests check worker PIDs and restart counters.
- Fixed Raylib Back/reopen re-entry into stale EGL state by terminating its
  dedicated renderer process before another NativeActivity can reuse it.
  Driving services run independently.

## Model findings

No inference backend has passed the original-reference gate. Three independent
seeds are checked: **42, 7, 123**. All final model comparisons must pass before
recurrent and replay acceptance can select a live backend.

The [candidate6 operator trace](fold-operator-trace-candidate6.json) observes finite
values after the grouped-Conv repair. Its first observed divergence is the first
convolution: driver monitoring node 9 (`node_Conv_805`), driving node 26
(`model/node_conv2d`). OpenCL maximum errors there are 0.03125 and 0.125;
CPU errors are 0.234375 and 0.5625. Exposing intermediate outputs changes graph
fusion/rounding: the instrumented graphs themselves do not match the original
graphs. These traces locate diagnostic divergence and cannot qualify the
original model. [Explicit accumulation experiments](convolution-accumulation.json)
also fail to reproduce all sampled driving outputs.

The pinned **60-frame replay is unavailable in this checkout**, and recurrent
sequence acceptance is not implemented. Both remain blocked; single-frame
seed tests and finite outputs do not replace them. No reference or tolerance
was relaxed to obtain a pass.

## Camera and physical gates

Triple-stream **transport** works through logical camera 20: GPU preview,
thumbnail and front camera. Preview/thumbnail names are transport roles, not
proof of distinct physical sensors. The user deferred the three repeated
lens-cover trials. Independent rear images, composite regions and their model
input mapping therefore remain **blocked**. Lens switching or duplicate images
cannot qualify as independent streams under the [multi-camera contract](https://developer.android.com/media/camera/camera2/multi-camera).

The phone is **unmounted**, so camera intrinsics/crop/orientation provenance is
retained as bench geometry, not vehicle calibration. The **Panda is disconnected**:
real firmware compatibility, passive fingerprint, ignition/bus mapping,
Hyundai C harness and physical vehicle behavior are blocked. Tests preserve
factory ACC/AEB, disabled experimental longitudinal control, the pinned
approximately 51 km/h steering minimum and upstream safety limits.

No new 30-minute workload is permitted while identity, numerical correctness or
input freshness fails. Earlier soak failure remains recorded below. Thermal
pause at MODERATE, cooldown and critical-temperature abort are retained.

## Reproduce

Follow the pinned toolchain/native/model preparation in [the build instructions](../../README.md).
Build `-Pfold3CameraCompat=true`, install both APKs, and run every validator
through the experiment wrapper. Example from the repository root:

```sh
export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH"
adb -s RFCR81Y1TBJ install -r -g android/app/build/outputs/apk/debug/app-debug.apk
adb -s RFCR81Y1TBJ install -r -g android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --report /tmp/fold-model-isolation.json -- \
  .venv/bin/python android/scripts/validate_models.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --profile baseline --report /tmp/fold-model.json
```

Use profile `ort-cpu` for the independent CPU diagnostic. For the full process
chain, add `--runtime full-chain` to the wrapper and run
`-m openpilot.tools.sim.android_chain_validation --seconds 30` with the same
serial/package/report arguments. Camera/model combined load uses
`android/scripts/validate_vision.py --seconds 30 --samsung-multiview` and no
wrapper runtime option. Lifecycle uses wrapper `--runtime physical` on the
phone, `--runtime simulation` plus child `--resize` on the dedicated emulator.
The batch JSON files retain every exact final command, including fault and
MetaDrive runs. Install only after the previous experiment has finished cleanup.
The wrapper requires 30 continuous seconds at NORMAL and skin at or below 32°C
before each experiment; it verifies app exit and camera release between runs.

## Earlier evidence retained

Candidate metadata JSON files identify each earlier APK. Later passes do not
erase these failures:

| Earlier artifact/run | Result and scope |
| --- | --- |
| candidate2 | Emulator UNKNOWN-clock camera frames unqualified; slow camera delivery; native FFT/Params/calibration failures subsequently repaired. |
| candidate3 | GLES3 preview readback p99 7.19 ms versus 28.20 ms synchronous RGBA baseline; rear image age still about 147 ms. |
| candidate4 | Non-finite original CPU driving outputs; full-chain DM fixture restarted after malformed feature input. |
| candidate5 | CPU original outputs finite after grouped-Conv patch, all seeds still fail parity; DM work p99 47.03 ms, driving 209.17 ms. Full-chain phone controls work p99 9.25 ms but complete cycle 23.93 ms and mean 90.49 Hz. Camera road freshness failed; receive/copy timing had a unit error, fixed in candidate6. |
| candidate5 emulator | Relay timeout and display-test extent failure; preserved diagnostic attempts. Idle relay recovery and display extent fixed subsequently. |
| candidate6 | Emulator complete process chain passed, 806 packets, no worker restarts; OpenCL finite but all seeds fail parity, DM/driving p99 111.86/87.72 ms. |
| candidate7 | Both devices passed 19 instrumentation tests; emulator background/resize passed but Back/reopen crashed in EGL. Crash/logcat retained; candidate8 addresses renderer disposal. |
| `e91f1190…` isolation | **Invalid experiment**: APK changed during cooldown. It cannot support a pass. |
| `f725600b…` earlier 30-minute soak | Triple transport about 30 fps and UI 29.94 fps, but MODERATE pause after 4:09, SEVERE / 44.6°C, stale DM and non-finite road inference. This is not sustained-stack acceptance. |

Host checks: 57 focused tests passed; 13 Hyundai tests ran with 2 skips;
1,906 Panda NoOutput/Hyundai safety tests ran with 198 skips. The initial wrong
Mac SDK/linker run is retained; the successful safety run used Xcode's 26.5 SDK.
The [quality scan](quality-review.json) is not clean: it reports complexity,
duplication and repository-population findings; review notes distinguish those
from correctness tests. No quality finding was acknowledged to suppress it.

## Final candidate8 device results

Every row below uses the APK hash above. Failures from earlier APKs and earlier
candidate8 attempts remain in their original reports and batch journals.

| Check | Fold 3 | Android 15 ARM64 emulator |
| --- | --- | --- |
| Expanded instrumentation | **Passed**, 19 tests | **Passed**, 19 tests |
| Raylib lifecycle, Back/reopen and independent services | **Passed** | **Passed**, including display resize |
| Complete process-chain fixture | **Passed**, 2,682 packets | Earlier **passed**, 767 packets; host-GPU rerun **failed**, 2,660 packets |
| Required derived outputs and restart budget | **Passed** in injected-input fixture | Outputs present; host-GPU rerun has two worker deaths/restarts |
| ADB malformed packets, stale input and disconnect recovery | **Passed**, corrected legacy fixture | **Passed**, corrected legacy fixture |
| MetaDrive cancellation and stale inputs | **Passed**, synthetic perception/plant | **Passed**, synthetic perception/plant |
| Thermal pause, resume and critical abort | Host guards checked; no physical heating fault injected | **Passed**, emulator thermal overrides |
| Original model parity, three seeds | **Failed**, both OpenCL and CPU | **Blocked** for OpenCL: no phone-equivalent GPU backend |
| Combined camera/model freshness | **Failed**, no fresh road preview samples and no timely model outputs | Camera timing is tested separately from injected inputs |
| Physical rear identity / measured geometry | **Blocked**, cover trials deferred; unmounted | **Blocked**, software cameras cannot qualify phone optics |
| Panda / vehicle | **Blocked**, disconnected | **Blocked**, physical integration cannot be verified |
| Sustained 30-minute driving stack | **Blocked** by correctness/freshness gates | **Blocked**, emulator throughput cannot qualify phone deadlines |

The process-chain reports are [Fold 3](fold-fullchain-candidate8.json) and
[emulator](emulator-fullchain-candidate8.json). They inject source CAN, model,
driver, odometry and sensor data; required localization, calibration, parameter,
radar, planning and selfdrive outputs come from the real supervised processes.
They establish process connectivity and fault handling. Reported invalid inputs
and timing misses prevent a claim of a healthy sustained driving stack.
The later [host-GPU chain run](emulator-host-fullchain-candidate8.json) has no
unsafe outputs and all required outputs, but Radar and Torque each die with
SIGUSR2 and restart once. Logcat identifies the signal. The Android ARM64 msgq
fallback can signal process group zero during subscriber registration; its
zero-target guard is a subsequent candidate9 change, not a candidate8 pass.

The repaired ADB fixture [passes on the phone](fold-adb-faults-fixed-candidate8.json):
ten malformed-session cases are rejected; dropping model inputs stops simulated
control; restoring inputs recovers; disconnect stops control. Controls,
monitoring and bridge PIDs stay unchanged with zero restarts. Earlier
`fold-adb-faults-candidate8` and `emulator-adb-faults-candidate8` failures retain the
incomplete-fixture driver-monitoring exception.

The final phone MetaDrive [cancellation](fold-metadrive-cancel-fixed-candidate8.json)
and [stale-input](fold-metadrive-stale-fixed-candidate8.json) runs pass, with
1,824 / 1,817 packets, 651 / 648 inactive samples after the fault, zero
longitudinal-active samples and maximum lateral errors 0.630 / 0.658 m. These
use synthetic perception, driver monitoring and calibration and an explicitly
synthetic damped torque-to-angle plant. They do not qualify visual closed-loop
driving or Hyundai dynamics.

The final emulator runs use the Mac's host GPU, two emulated cameras and two
virtual CPU cores; the exact setup is in
[configuration evidence](emulator-host-configuration-candidate8.json). The
earlier SwiftShader follow-up encountered a System UI ANR dialog, leaving the
Raylib activity paused after one frame. Those startup failures are retained in
their isolation reports; they do not show a driving-worker crash. A clean
host-GPU launch restored foreground rendering without changing the APK.

The corrected emulator [ADB fault test](emulator-host-adb-faults-fixed-candidate8.json)
passes ten malformed-session cases, stale-input withdrawal, recovery and
disconnect, with unchanged worker PIDs and zero restarts. MetaDrive
[cancellation](emulator-host-metadrive-cancel-fixed-candidate8.json) and
[stale-input](emulator-host-metadrive-stale-fixed-candidate8.json) pass with
1,733 / 1,731 packets, maximum lateral error 0.423 / 0.584 m and zero
longitudinal-active or below-minimum-speed active samples. These retain the
same synthetic-input and plant limitations as the phone tests.

The emulator [thermal test](emulator-host-thermal-dual-candidate8.json) passes:
both model workers pause at MODERATE, remain paused after two cool seconds and
resume after ten, with unchanged PIDs. A critical override prevents camera
opening and stops active capture within 1.53 seconds; the live validator also
aborts and releases resources. The override is reset afterward.

Emulated [Camera2 timing](emulator-host-camera-dual-candidate8.json) remains
**failed**, separately from deterministic injected inputs. Cameras 0/1 deliver
434 / 435 frames at about 29.1 fps, but 434 timestamps per stream are ahead of
the sampled elapsed-realtime clock. Image-age p99 is -24.37 / -21.30 ms. Raw
sensor timestamps and clock samples are retained; arrival timestamps are not
substituted to manufacture valid capture times. These software camera results
cannot qualify the Fold 3's sensor identity or timing.

## Final measurements and comparisons

| Workload | Candidate7 | Candidate8 | Acceptance |
| --- | --- | --- | --- |
| Original OpenCL DM work p99 | 110.43 ms | 112.61 ms | **Failed**, target <50 ms and parity fails |
| Original OpenCL driving work p99 | 86.08 ms | 89.72 ms | **Failed**, target <50 ms and parity fails |
| CPU DM work p99, isolated | — | 43.15 ms | Timing passes alone; parity **fails** |
| CPU driving work p99, isolated | — | 209.30 ms | **Failed** |
| Full-chain controls work p99 | 9.22 ms | 9.31 ms | Work percentile <10 ms; 17 deadline misses remain |
| Full-chain complete cycle p99 / mean rate | 23.87 ms / 91.98 Hz | 24.71 ms / 91.35 Hz | **Failed**, 100 Hz target; 1,313 cycle misses |
| Combined preview GPU readback p99 | 7.30 ms | 10.22 ms | Transport measurement; no improvement claim |
| Thumbnail / wide-preview image age p99 | 126.91 / 155.87 ms | 82.01 / 128.46 ms | Wide stream still **fails** 100 ms freshness |
| Driver image age p99 | 104.20 ms | 108.19 ms | **Failed**, 100 ms limit |
| Combined UI mean | 29.76 fps | 29.54 fps | Approximately 30 fps; does not establish model deadlines |

These are separate short experiments with different contention, not controlled
energy or sustained-throughput comparisons. In candidate8 both OpenCL models
missed all 100 measured 50 ms deadlines. CPU DM met the isolated timing target
but still differed from original outputs for seeds 42/7/123 by up to
0.078125 / 0.337891 / 0.132813. CPU driving differences were
1.296875 / 1.65625 / 1.0625; recurrent feature outputs also failed.
See [OpenCL](fold-opencl-candidate8.json) and [CPU](fold-ort-candidate8.json).

The [30-second combined run](fold-gpu-vision-candidate8.json) captured
900 thumbnail/composite, 899 wide-preview and 899 front frames, about 30 fps,
with zero camera recoveries and no timestamp or packing errors. Physical
capture results remained empty; the vendor `runningPhysicalId=[48,0]` is not
independent sensor proof. All reported noise-reduction, sharpening and
stabilization modes were OFF. The preview and thumbnail changed over time and
overlapped for 29.98 seconds; identity and composite-region provenance remain
unverified until the deferred cover experiment.

Road inference computed 66 finite outputs, all late; driver inference computed
414, all late. Neither worker published a prediction. Road/driver work p99 was
392.92 / 53.12 ms, including 66 / 17 deadline misses. Road stage p99 was
2.83 ms receive/copy, 12.76 ms preparation and 368.81 ms inference; driver was
1.05 / 2.64 / 51.64 ms. Output age p99 was 480.48 / 147.67 ms. Raw timestamps,
clock samples, stage distributions and rejection counts remain in the reports.
No valid output was substituted for a stale or uncalibrated prediction.

Sampled PSS grew from 144,027 to 1,312,876 KiB, including runtime/model startup;
sampled skin went from 29.1°C to 36.9°C and cleanup measured 37.6°C, still NORMAL.
That short run cannot establish sustained thermal stability. The phone then
exited, released cameras and cooled before the next experiment. The
[final phone cleanup](fold-final-cleanup-candidate8.json) confirms no app
processes, no owned cameras, no ADB forwards, NORMAL status, no thermal override,
27.9°C skin and over 30 continuous cool seconds.

The delivery quality scan remains **failed**, with 36 major regressions against
the Git HEAD baseline. Its report includes complexity and duplication findings
alongside repository-population noise. No findings were suppressed. Correctness
test passes do not clear that separate review gate.
