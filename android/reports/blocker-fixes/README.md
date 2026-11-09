# Fold 3 blocker-fix evidence — 2026-11-01

Branch: `codex/fold3-android-port`, HEAD
`7631d5248cc9efd10b962107aadacbb908a9f14d` plus the uncommitted port fixes.
**Vehicle TX is compiled out and physical engagement is unconditionally disabled.**
The complete visual driving runtime is still unqualified. Passing a simulator
test does not qualify cameras, mounted calibration, Panda firmware or a vehicle.

## Current candidate35

The [validation report](candidate35-validation.md) records passed, failed and
blocked results for the exact installed artifact; the
[structured summary](candidate35-validation-summary.json) keeps child exit
statuses separate from [experiment isolation](candidate35-isolation-audit.json).

| Artifact | SHA-256 |
| --- | --- |
| [Debug APK](../../build/candidates/candidate35/app-debug.apk), 815,252,690 bytes | `38645e91f806076517de1497cdb6b5c0cbf4873cfecf5efabb980297a014ad51` |
| [Instrumentation APK](../../build/candidates/candidate35/app-debug-androidTest.apk), 4,492,519 bytes | `98d31adfd4ef7679e70a5fb51083bff55380a0c1080336e9a63278e531efa9fe` |

See [installed hashes](candidate35-installed.json),
[source/build identity](candidate35-artifact.json),
[implementation changes](candidate35-changes.md) and
[reproduction commands](candidate35-reproduce.md).
All **795 original-reference comparisons pass exactly**. Accurate OpenCL work
p99 is **881.57 ms road / 560.16 ms driver** and fails 20 Hz. The faster native-MAD
trial fails parity and cannot qualify live inference. Original models, goldens
and tolerances remain unchanged.

The [performance comparison](candidate35-performance-comparison.json) retains
candidate32, the reverted candidate33 tiling regression and candidate35 results
with their own APK hashes. The [camera profile comparison](candidate35-camera-comparison.json)
records continuous readback; preview freshness still fails. The full process
chain works with synthetic sources, while complete 100 Hz controls timing and
live visual inference remain unqualified. The 30-minute acceptance workload is
blocked. Independent rear identity needs the deferred lens-cover experiment;
the phone is unmounted and Panda disconnected. TX is compiled out and physical
engagement remains unconditionally disabled.

[Earlier failures](candidate35-retained-failures.json) are retained. Candidate34
was built and host-tested but superseded before installation. The
[delivery audit](candidate35-delivery-audit.json) binds application sources and both
installed APKs; [final cleanup](candidate35-final-cleanup.json) records released
cameras, stopped processes, forwards and thermal baselines.
Three [host-only tool corrections](candidate35-host-tooling-update.json) retain
their build-time sources and the initial thermal failure; neither APK changed.

## Historical candidate32 artifact and numerical results

Package: `com.samsung.sunnypilot.fold3`. The same app and test APKs were
[installed and hash-verified](candidate32-installed.json) on the Fold 3 and the
dedicated Android 15 ARM64 emulator.

| Artifact | SHA-256 |
| --- | --- |
| [Debug APK](../../build/candidates/candidate32/app-debug.apk) — 815,233,150 bytes | `a62b55187b2a60a15c9b7d34cb5e6e462cc00c353971d8d126940960861a2810` |
| [Instrumentation APK](../../build/candidates/candidate32/app-debug-androidTest.apk) — 4,490,455 bytes | `bdf8e6c3f977c5afb521c68a7719e700fec2a1d93209c5ff984aec7821883014` |

The [artifact record](candidate32-artifact.json) retains build inputs and
source hashes. [Reproduction commands](candidate32-reproduce.md) cover the
build, host tests and isolated device validators. Original models, goldens and
`rtol=1e-3, atol=1e-4` remain unchanged.

The original-model numerical blocker is resolved in the recorded full gate.
Integer-bit concatenation preserves half subnormals; integer midpoint
correction fixes the normalization square root responsible for the remaining
recurrent divergence. Candidate32 then removes software FMA only from exact
half-input products, preserving the pair and accumulator rounding boundaries.
The [arithmetic investigation](host-numerical-diagnostics.md) retains each
failure, first-divergence trace, regression and device probe.

| Complete Fold 3 gate | Numerical result | Work p50 / p99 | 50 ms misses |
| --- | --- | --- | --- |
| Original road model | **Passed**, 636 exact output comparisons | 850.81 / 1,021.53 ms | **100/100** |
| Original driver monitoring | **Passed**, 159 exact output comparisons | 496.05 / 559.41 ms | **100/100** |

Both models pass seeds 42/7/123, 32 recurrent frames per seed and the pinned
60-frame camera replay. The [full accuracy gate](candidate32-model-accuracy-gate.json)
has zero differing outputs. **Inference performance fails; no live backend is
accepted.** Serialized road half-dot cost falls from 80.19 to 4.22 ms, but FP32
convolution remains dominant. These checkpoint costs are diagnostic;
separately cooled full-model runs provide the timing results above.

The [host checks](candidate32-host-results.json) pass **92 focused tests**,
**3,938 Hyundai/Panda/protocol checks (356 skipped)**, complete original-model
reference sequences and focused Ruff checks. The initial Mac SDK/linker
failure is [retained](candidate32-host-toolchain-failure.json); selecting the
working Xcode compiler/SDK passes the rerun. **20 instrumentation tests per
device** pass, including numerical/thermal faults and USB rejection paths.

The phone and emulator both pass fault and steady sources-only runtime-chain
fixtures: required derived outputs are present, cancellation/stale CAN pass,
and no workers fail or restart. The phone steady test, with the emulator
stopped, averages **100.87 Hz**. Work p99 is **9.683 ms**, complete-cycle p99
**24.365 ms**, with **24 work / 1,213 cycle misses in 2,701 samples**.
The [recorded fixture](fold-full-chain-steady-candidate32.json) includes input
wait and ADB jitter rather than suppressing those misses. Behavioral acceptance
passes; sustained controls timing remains **failed**.

The [30-second combined camera run](fold-vision-candidate32.json) captures
**893 composite / 897 preview / 894 front frames**, with zero recoveries,
timestamp/packing errors or dropped readbacks. Delivery-age p99 is
**98.86/124.60/106.47 ms**. All 58 road predictions and 410 of 411 driver
predictions are late; the remaining driver message is invalid. These use the
unqualified CPU diagnostic backend, separately from the accurate OpenCL gate.
Raylib averages **29.51 fps**. Sampled PSS reaches **1,316,202 KiB**, skin
**37.0°C**; immediate cleanup is **37.3°C/NORMAL**. Camera freshness and
perception performance **fail**. The emulator camera also **fails**: all
289 raw sensor timestamps are in the future and remain rejected.
The [performance comparison](candidate32-performance-comparison.json) identifies
every APK and distinguishes serialized kernel costs from full-model timings.

Independent rear sensor identity still requires the deferred three-repetition
lens-cover test. The phone remains unmounted and the Panda disconnected.
Mounted optical/vehicle calibration, real firmware compatibility, passive
fingerprint, bus mapping, harness and vehicle behavior remain **blocked**.
The 30-minute workload remains blocked by short inference/freshness/deadline
gates. TX stays compiled out regardless of bench results.

## Historical candidate32 acceptance and cleanup

The [device summary](candidate32-validation-summary.json) records **18 isolated
experiments: 13 passed, four failed, one blocked**. The model experiments fail
their performance gate despite exact outputs. The diagnostic serialized-kernel
run is blocked for qualification by design. Raw nonzero wrapper exits are
preserved; the separate [isolation audit](candidate32-isolation-audit.json)
passes all 18 experiments, with stable APKs/providers, released cameras and
stopped app/test processes. Minimum continuous NORMAL baseline is **30.67 s**;
maximum baseline skin temperature is **31.4°C**.

| Acceptance gate | Result | Scope |
| --- | --- | --- |
| Build, host reference, focused host and instrumentation | **Passed** | Actual hashes and commands linked above; 20 instrumentation checks per device |
| Original-model seeds, recurrence and camera replay | **Passed** | 795 exact comparisons; original models, goldens and tolerances |
| Runtime chain, lifecycle, IPC and emulator MetaDrive faults | **Passed components** | Injected sources; cancellation/stale inputs, zero worker failures/restarts |
| Camera transport and approximately 30 fps Raylib | **Passed components** | Triple-stream transport; independent rear identity remains unproven |
| Accurate 20 Hz inference and 100 Hz complete controls timing | **Failed** | Both accurate models exceed 50 ms; runtime fixture retains cycle misses and ADB jitter |
| Phone camera freshness / emulator camera timing | **Failed** | Preview/front exceed age budget; future emulator sensor timestamps rejected |
| Independent rear identity, mounted calibration and physical Panda/vehicle | **Blocked** | Deferred covers; unmounted phone and disconnected Panda |
| 30-minute combined acceptance workload | **Blocked** | Short freshness and timing gates fail |

[Final cleanup](candidate32-final-cleanup.json) passes on both devices: no
app/test processes, cameras released, no ADB forwards and stable camera
providers. The phone finishes at **30.0°C after 32.33 continuous NORMAL seconds**;
the emulator was also cooled and stopped. Build-time source hashes are retained
unchanged; the [delivery audit](candidate32-delivery-audit.json) distinguishes
post-build documentation updates from packaged runtime sources.

Sections below retain historical artifacts and their original failures.

## Historical candidate21 artifact

Package: `com.samsung.sunnypilot.fold3`. Both devices have the same
[hash-verified installation](candidate21-installed.json).

| Artifact | SHA-256 |
| --- | --- |
| [Debug APK](../../build/candidates/candidate21/app-debug.apk) — 815,202,378 bytes | `4c48351cf6d6166ad32d0ab309072e6900438decb1751f86ab30fcdb5109bbfe` |
| [Instrumentation APK](../../build/candidates/candidate21/app-debug-androidTest.apk) — 4,490,427 bytes | `374460c0d2b383b8f057af2bf403f17424df616201b1529736dc5ba882a27738` |

[Artifact metadata](candidate21-artifact.json) records the dirty source snapshot,
original model hashes and native runtime. Python staging was checked
byte-for-byte before archiving. Candidate21 fixes a device-only qualification
error: Chaquopy packages these modules as source-free bytecode, so hashing their
synthetic `__file__` source paths failed. Fingerprints now read the actual code
asset through its import loader. A source-free host regression and the packaged
device instrumentation check cover this path.

The new OpenCL renderer preserves half subnormal results and explicit
round-to-nearest-even boundaries for both ordinary ALU and custom-operation
casts. Small scalar/vector helper functions evaluate their input once, fixing
the exponential shader expansion that stalled candidate17 compilation.
Candidate19's [phone trace](candidate19-model-diagnostic-summary.json) passes
all 63,512 half conversions, including 2,056 subnormal values, and all 18 matrix
microchecks exactly. The same arithmetic is packaged in candidate21. Original
full-model outputs still fail in that trace; this arithmetic repair alone does
not establish model parity. No model or tolerance was changed.

Candidate21 binds live qualification to the ONNX operator, UOp,
renderer and OpenCL compiler implementation hashes. A renderer change cannot
reuse a qualification record produced by different arithmetic.
[Host checks](candidate21-host-results.json) pass 77 focused tests and 1,927
Hyundai/Panda/protocol tests (200 skipped), including all 312 host recurrent
and camera-replay frame comparisons. Earlier toolchain, import, permission,
package-selection and installation-race failures remain in the evidence tree.

Superseded candidate20 (`af767d70…24746`) passed its 19 instrumentation tests
on both devices, but its model benchmark stopped at the code-asset hashing
error after the recurrent/replay checks. Its [disposition](candidate20-disposition.json)
retains that failure and the camera experiment interrupted before capture.

All device runs have separate isolation records. A successful trace collection
or transport check is distinct from numerical, timing and physical acceptance.
Independent lens identity remains blocked by the deferred cover experiment;
the phone remains unmounted and the Panda remains disconnected.

## Candidate21 validation

The [validation summary](candidate21-validation-summary.json) records **17 isolated
device experiments: 13 passed and 4 failed**, all with the candidate21 installed
APK hash unchanged. The experiment status is distinct from driving acceptance:
the full-chain fixtures pass behavior checks while still missing deadlines.

| Check | Result and evidence |
| --- | --- |
| Host | **Passed:** 77 focused tests; 1,927 Hyundai/Panda/protocol tests, including 200 skips; 312 frozen-host recurrent/replay comparisons |
| Instrumentation | **Passed:** 20 tests on [Fold 3](fold-instrumentation-candidate21.json) and [emulator](emulator-instrumentation-candidate21.json), including packaged-code identity, numerical fault, thermal and USB rejection checks |
| Process chain, cancellation and stale CAN | **Passed behavior:** [Fold 3](fold-full-chain-candidate21.json), [emulator](emulator-full-chain-candidate21.json); eleven required derived services, no worker failures or unsafe outputs |
| MetaDrive | **Passed fixture checks:** [cancellation](emulator-metadrive-cancel-candidate21.json) and [stale input](emulator-metadrive-stale-candidate21.json); simulated sources do not qualify camera inference or a vehicle |
| Lifecycle, display, ADB recovery and msgq | **Passed:** device-specific records and commands are indexed in the validation summary |
| Original OpenCL models | **Failed:** both models fail all three seeds, their recurrent sequences and the pinned camera replay; all outputs finite; p99 and deadline targets also fail |
| Fold 3 triple-stream transport | **Passed for 30 seconds:** 894 composite, 897 preview and 895 front frames, zero recoveries or timestamp/packing errors |
| Fold 3 camera freshness | **Failed:** delivery-age p99 composite/preview/front **98.49/124.68/104.60 ms**; insufficient fresh model inputs |
| Actual emulator Camera2 timestamps | **Failed:** all 288 frames have future capture timestamps; p99 delivery age **−26.41 ms**; invalid frames remain rejected |
| Complete controls timing | **Failed:** phone work p99 **9.542 ms**, but 20 work misses and 1,218 complete-cycle misses in 2,688 samples; cycle p99 **23.992 ms** |
| Independent rear identity, mounted calibration and physical Panda | **Blocked:** deferred lens covers, unmounted phone and disconnected Panda |
| Sustained driving and new 30-minute workload | **Blocked:** numerical, identity, freshness and timing gates have not passed |

The [performance comparison](candidate21-performance-comparison.json) retains
the actual profile and APK for each observation. OpenCL DM/driving p99 is
**75.40/105.58 ms**, with **39/100 and 100/100** timed deadline misses. These
results use the frozen host-fusion profile and are not a controlled comparison
with the historical candidate10 baseline profile.

During the combined camera diagnostic, OpenCL remains unqualified, so the
reported live workers use diagnostic ORT CPU inference. All **54 road outputs
are late and withheld**; **442/443 driver outputs are late**. The one published
driver message remains invalid and cannot enable engagement. Road/driver
inference-only p99 is **55.86/41.57 ms**, complete work p99 **67.44/47.07 ms**,
and output age p99 **156.14/139.77 ms**. UI averages **29.55 fps**.

Sampled PSS grows from **250,381 to 1,307,012 KiB**, including startup; sampled
skin temperature reaches **36.5°C**, with immediate cleanup at 36.8°C/NORMAL.
This short run does not establish thermal endurance. The [final cleanup](candidate21-final-cleanup.json)
confirms no app/test processes, cameras released, no ADB forwards, the emulator
stopped, and **29.0°C after more than 30 continuous NORMAL seconds**.

The [structural quality delta](candidate21-final-quality-delta.json) reports 489
regressions, including 45 gating findings, across the entire dirty Android port
and concurrent QNN work against HEAD. This is a separate unresolved check;
functional passes do not make it clean and no finding was automatically acknowledged.

## Historical candidate10 artifact

The earlier **candidate10** used package `com.samsung.sunnypilot.fold3`:

| Artifact | SHA-256 |
| --- | --- |
| [Debug APK](../../build/candidates/candidate10/app-debug.apk) — 512,615,641 bytes | `566e4c253c05aae9a6d8fdce82c68b67f3558612ad97560a5d12b2ca2ce83a9a` |
| [Instrumentation APK](../../build/candidates/candidate10/app-debug-androidTest.apk) — 4,466,867 bytes | `e7a159b944a4e82f3f38a4d370436e9c5dc6197e6f918026247ec1d38e989d58` |

[Artifact metadata](candidate10-artifact.json) records the original model hashes,
native runtime build and compressed/uncompressed payload sizes. Each device
experiment has a matching `-isolation.json`: installed APK hash before cooldown,
at launch and after cleanup; camera-provider PIDs; app process cleanup; camera
ownership; forwarding cleanup; NORMAL thermal status and skin temperature.
An experiment is invalid if its installed artifact changes.

## Implemented fixes

- Fixed Android msgq wakeup identity: subscriber IDs retain both process and
  thread IDs, and `tgkill` targets that exact pair. Zero targets are rejected.
  Two host regressions and real-kernel diagnostics on both devices verify
  registration, mismatched-target rejection and correct delivery. Queue/Cereal
  layouts and non-Android behavior remain unchanged. Eight short full-chain
  checks on this APK complete without worker failures or restarts. The exact
  sender of the earlier SIGUSR2 failures was not captured.
- Corrected the host CAN/sensor fixture to use absolute 100 Hz deadlines instead
  of accumulating relative-sleep overshoot. Source cadence and exchange latency
  are now reported separately. Steady-input and cancellation/stale-input runs
  use the same fixture; this changes test pacing, not controls behavior.
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

The pinned **60-frame camera replay is now prepared**, with verified route assets,
frame-to-encode mapping and the pinned image preprocessing. Recurrent acceptance
now checks 32 frames for each of seeds 42, 7 and 123, feeding each backend's own
state into the next frame. Both frozen host graphs agree with the unchanged
original reference across all 312 model/frame comparisons. Device acceptance is
still required; host agreement and single-frame seed checks cannot select a live
backend. No reference or tolerance was relaxed to obtain a pass.

Further **host-only diagnostics** export all 437 DM and 531 driving tensor taps
(436/527 intermediate taps plus original outputs) after freezing host optimization. Their final outputs still match the unchanged
original references. The [isolated export check](candidate21-isolated-trace-export.json)
also verifies that no packaged reference or host-fusion asset changes.

[Accumulation probes](fused-convolution-accumulation.json) reproduce all 256
sampled outputs of each first NHWC convolution using ordered float32 FMA with
bias first. The [whole-convolution prototype](ordered-convolution-prototype.json)
matches four observed NHWC outputs exactly. Regular NCHW convolutions require
different bias placement; this is not a general replacement kernel.

The [complete prototype trace](complete-trace-ordered-mlas-all-prototype.json)
also tries pinned MLAS activation polynomials and precise Metal division. DM's
single-seed final output meets the tolerance, but intermediate values still
fail, and driving has **213 failing final elements**. The first tolerance failure
is DM FusedGemm node 334 and driving Gemm node 281. Standalone first-activation
[checks](full-first-activation-precise-diagnostic.json) match all 691,200 DM Tanh
values exactly, while driving normalization still differs at float32 rounding
boundaries. These are diagnostic Metal results on the host, not Android OpenCL
qualification. No production kernel, reference, model hash or tolerance was
changed on their strength. The source archive and reproduction steps are linked
from [the diagnostic notes](host-numerical-diagnostics.md).

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
shasum -a 256 android/build/candidates/candidate21/app-debug.apk
adb -s RFCR81Y1TBJ install -r -g android/build/candidates/candidate21/app-debug.apk
adb -s RFCR81Y1TBJ install -r -g android/build/candidates/candidate21/app-debug-androidTest.apk
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --report /tmp/fold-model-isolation.json -- \
  .venv/bin/python android/scripts/validate_models.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --profile host-fusion-cl --report /tmp/fold-model.json
```

Use `validate_instrumentation.py --expected-tests 20` with the same serial and
package through the wrapper for instrumentation. Use profile `ort-cpu` for the
independent CPU diagnostic. For the full process
chain, add `--runtime full-chain` to the wrapper and run
`-m openpilot.tools.sim.android_chain_validation --seconds 30` with the same
serial/package/report arguments. Add `--steady-inputs` for uninterrupted cruise/CAN
input; that mode explicitly skips cancellation and stale-input checks, which
are run separately. Camera/model combined load uses
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
| candidate8 | Phone chain passed, but host-GPU emulator chain killed radar/torque workers with SIGUSR2. The original report's process-group-zero attribution was unproven and is corrected in its erratum. Camera/model freshness and original parity failed. |
| candidate9 | Zero-target guard alone did not repair wakeups: the second emulator chain still lost planning to SIGUSR2. The ARM64 binary used `tkill`; the fallback explanation did not apply. Phone tests and the initial emulator chain passed. |
| `e91f1190…` isolation | **Invalid experiment**: APK changed during cooldown. It cannot support a pass. |
| `f725600b…` earlier 30-minute soak | Triple transport about 30 fps and UI 29.94 fps, but MODERATE pause after 4:09, SEVERE / 44.6°C, stale DM and non-finite road inference. This is not sustained-stack acceptance. |

Host checks: 59 focused tests passed; 13 Hyundai tests ran with 2 skips;
1,906 Panda NoOutput/Hyundai safety tests ran with 198 skips. The initial wrong
Mac SDK/linker run is retained; the successful safety run used Xcode's 26.5 SDK.
The [quality scan](quality-review.json), field `candidate10_final_scan`, is not
clean: 36 major gates remain, including complexity, duplication and
repository-population findings. No finding was acknowledged to suppress gating.
The native submodule checkouts remain at their pinned commits.

## Historical candidate10 results

The [validation summary](candidate10-validation-summary.json) contains **26
isolated experiments: 21 passed and 5 failed**. Each has this APK hash and
verified cleanup. Build/unit checks and 59 focused host checks pass separately.
Historical Hyundai and Panda safety results above were not rerun after the
final IPC/fixture changes; their original logs and scope are retained.

| Check | Result | Evidence and scope |
| --- | --- | --- |
| Installed artifacts | Passed | [Both devices](candidate10-installed.json) match the main and instrumentation APK hashes above. |
| Instrumentation | Passed | 19 tests on [Fold 3](fold-instrumentation-candidate10.json) and 19 on [emulator](emulator-host-instrumentation-candidate10.json), including conversion, numerical rejection and USB rejection paths. |
| Kernel IPC wakeup | Passed | Actual pinned/patched C++ on [phone](fold-msgq-kernel-candidate10.json) and [emulator](emulator-msgq-kernel-candidate10.json); zero targets, wrong process/thread pairs and correct delivery. |
| Supervised process chain | Passed, short fixture scope | Four phone and four emulator runs; all required derived outputs, no worker failures/restarts, no unsafe outputs. This does not establish valid live perception or watchdog qualification. |
| Raylib lifecycle | Passed, tested actions | [Phone](fold-lifecycle-candidate10.json) Back/reopen and independent workers; [emulator](emulator-host-lifecycle-candidate10.json) background, Back/reopen and resize. Physical fold/unfold remains unqualified. |
| ADB fault recovery | Passed | [Phone](fold-adb-faults-fixed-candidate10.json) and [emulator](emulator-host-adb-faults-fixed-candidate10.json): ten malformed sessions, stale-input withdrawal, recovery, disconnect and stable worker PIDs. |
| MetaDrive cancellation / stale inputs | Passed, synthetic | Phone [cancel](fold-metadrive-cancel-fixed-candidate10.json) / [stale](fold-metadrive-stale-fixed-candidate10.json), emulator [cancel](emulator-host-metadrive-cancel-fixed-candidate10.json) / [stale](emulator-host-metadrive-stale-fixed-candidate10.json). Synthetic perception, calibration and damped torque plant; not Hyundai dynamics or visual closed-loop qualification. |
| Thermal guards | Passed, emulator fault injection | [MODERATE pause, ten cool seconds to resume, critical abort/rejection](emulator-host-thermal-dual-candidate10.json); overrides reset. Real sustained thermal performance remains unqualified. |
| Original-model parity | Failed | [Phone OpenCL](fold-opencl-candidate10.json) and [CPU](fold-ort-candidate10.json) fail all three seeds against unchanged references; [emulator CPU](emulator-host-ort-candidate10.json) also fails. |
| Camera/model freshness | Failed | [Combined Fold run](fold-gpu-vision-candidate10.json): zero accepted road/driver publications; every computed prediction is late. |
| Emulator camera timestamps | Failed | [Camera2 run](emulator-host-camera-dual-candidate10.json): 437/436 frames, every timestamp ahead of the sampled clock; rejected without substituting arrival time. |
| Rear identity / calibration / physical Panda | Blocked | Cover trials deferred, phone unmounted, Panda disconnected. |
| Recurrent / pinned replay / 30-minute workload | Blocked on candidate10 | That APK predates sequence/replay fixtures. No accepted original-model backend; identity and freshness prerequisites unmet. |

The phone's final isolated steady-chain fixture runs after the emulator is
stopped. Source rate is 99.996 Hz; controls average **98.97 Hz**, work p99
**9.245 ms** (18 misses in 2,702 samples), and complete-cycle p99 **24.527 ms**
(1,267 misses). Input handling, compute, publishing, extensions and reporting
p99 are 18.920 / 4.751 / 5.491 / 4.468 / 4.234 ms; these stage quantiles are
not additive. The source exchange itself has 16.631 ms p99 work, so host/ADB
jitter is also present. Steady mode omits faults; the separate paced fault run
passes factory-ACC cancellation and stale-CAN handling. Required services exist,
but upstream validity flags and accepted live perception remain unmet.
[Steady timing](fold-isolated-steady-chain-candidate10.json),
[fault timing](fold-paced-fault-chain-candidate10.json).

## Performance comparison

Each number below comes from its named APK. These are short bench runs with
different fixture pacing/load, **not a controlled speedup claim**. Full
precision measurements, parity errors and source files are in the
[comparison JSON](candidate10-performance-comparison.json).

| Fold 3 metric | Candidate8 `98196287…f93295` | Candidate10 `566e4c25…e83a9a` |
| --- | ---: | ---: |
| OpenCL DM / driving work p99 | 112.61 / 89.72 ms | 111.57 / 88.41 ms |
| CPU DM / driving work p99 | 43.15 / 209.30 ms | 39.47 / 209.92 ms |
| Controls work / complete-cycle p99 | 9.31 / 24.71 ms | 9.24 / 24.53 ms |
| Mean controls rate | 91.35 Hz, relative pacing with faults | 98.97 Hz, absolute pacing, steady |
| Combined GPU readback p99 | 10.22 ms | 10.65 ms |
| Thumbnail / preview / front age p99 | 82.01 / 128.46 / 108.19 ms | 126.12 / 144.84 / 102.11 ms |
| Combined road / driver work p99 | 392.92 / 53.12 ms | 247.39 / 73.61 ms |
| Combined road / driver published frames | 0 / 0 | 0 / 0 |
| Combined mean UI rate | 29.54 fps | 29.68 fps |

Isolated CPU DM meets the 50 ms work target, but its parity fails. Neither
original OpenCL model meets that timing target. Every OpenCL model has 100/100
timed deadline misses. Candidate10 CPU driving also misses 100/100. Emulator
CPU DM/driving p99 is 63.42/290.82 ms; numerical parity fails there too.
No backend is selected for accepted live inference.

The final 30-second camera/model diagnostic records **898 thumbnail/composite,
898 preview and 899 front frames**, zero recoveries, no packing/timestamp errors
and no readback drops. Android thermal status stays NORMAL. Actual NR, edge,
EIS and OIS modes are OFF. Empty physical capture-result maps and shared-parent
timestamp skew of zero do not prove independent rear images. There are 53 road
and 469 driver computations, all late and withheld; non-finite counts are zero.
Road receive/copy, prepare and inference p99 are 3.34 / 9.12 / 232.99 ms; driver
values are 1.40 / 2.84 / 65.21 ms. Output-age p99 is 332.65 / 163.82 ms, beyond
the unchanged 100 ms freshness bound. Near-30 fps UI does not qualify the
complete driving stack.

Sampled PSS rises from 355,245 to 1,314,341 KiB including startup. Skin rises
from 28.6°C to 36.6°C in resource samples, reaching 36.8°C immediately after
cleanup. This short run cannot establish sustained thermal or memory behavior.
No new 30-minute attempt was run.

## Historical candidate10 cleanup

[Installed and cleanup evidence](candidate10-installed.json) verifies identical
APKs on both devices. All app/test processes exit, camera ownership is released
and ADB forwards are empty. Camera-server/provider PIDs remain unchanged.
The emulator resets display/thermal overrides and its dedicated AVD stops.
The Fold 3 cools to **29.6°C**, with more than 30 continuous seconds at NORMAL
thermal status and no thermal override. No test is left running.

The branch remains uncommitted local work. **Vehicle TX and physical engagement
remain disabled regardless of these bench results.**
