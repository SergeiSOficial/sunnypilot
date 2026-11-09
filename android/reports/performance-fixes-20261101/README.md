# Fold 3 performance fixes: candidate40

Cooling and temperature optimization are deferred at the user's request; existing
Android thermal protection remains active. **Overall acceptance remains failed**;
verified bench improvements and blocked physical gates are separated below.

The branch is `codex/fold3-android-port`. The installable bench APK is
[app-debug.apk](../../build/candidates/candidate40/app-debug.apk), SHA-256
`bd42b65a3538b8ad73834e90826eb10a98d8afeacbe9d5aae5877d43a25e30ec`. The [test APK](../../build/candidates/candidate40/app-debug-androidTest.apk)
has SHA-256 `98d31adfd4ef7679e70a5fb51083bff55380a0c1080336e9a63278e531efa9fe`. [Installed hashes](candidate40-installed.json) verify both
packages on the Fold 3 and ARM64 Android 15 emulator. Vehicle TX remains compiled
out and physical engagement disabled. This delivery does not qualify driving.

[Implementation changes](candidate40-changes.md), [reproducible commands](candidate40-reproduce.md),
[artifact/source/provider identities](candidate40-artifact.json),
[machine-readable acceptance](candidate40-validation.json) and
[retained earlier failures](candidate40-retained-failures.json) are preserved.
Candidate35 remains the measured baseline, APK
`38645e91f806076517de1497cdb6b5c0cbf4873cfecf5efabb980297a014ad51`.
Separate cooled runs use the same requested workloads and phone firmware;
these are not randomized repeated A/B measurements.

## Accuracy and acceleration

**Passed:** all **795 original-reference comparisons exactly**, unchanged models,
goldens and `rtol=1e-3, atol=1e-4`: three seeds, 32 recurrent frames per seed and
the pinned 60-frame camera replay. Driver contributes 159 comparisons and road
636. No unexpected non-finite output occurs. The [accuracy report](candidate40-model-accuracy-gate.json)
is separate from timing and live qualification.

**Failed:** QNN GPU original-output accuracy. The [final tapped and untapped graph checks](candidate40-qnn-diagnostic-summary.json)
record actual APK/graph hashes, operator, type, layout and provider assignment.
Graph changes remove the tested CPU partitions, but final numerical agreement
still fails. The untapped road graph additionally produces **998 unexpected
non-finite output elements**, while the tapped graph does not. This confirms that
intermediate-only results cannot establish full-graph accuracy. No QNN backend is
promoted; its three-seed, recurrent and replay qualification remains blocked by
this failure. HTP initialization remains deferred.

| Diagnostic graph | Model | QNN nodes / partitions | CPU nodes | Original final outputs |
| --- | --- | --- | --- | --- |
| Tapped | dmonitoring_model | 452 / 1 | 0 | failed (0 unexpected non-finite) |
| Tapped | driving_supercombo | 560 / 1 | 0 | failed (0 unexpected non-finite) |
| Untapped | dmonitoring_model | 435 / 1 | 0 | failed (0 unexpected non-finite) |
| Untapped | driving_supercombo | 559 / 1 | 0 | failed (998 unexpected non-finite) |

First observed tapped divergences (a tap can alter fusion):

- `dmonitoring_model`: Conv `android_precision_8_result`, float32 [1, 32, 120, 180]; maximum absolute error 0.026052474975585938, 166037 elements outside tolerance.
- `driving_supercombo`: Mul `InsertedPrecisionFreeCast_model/p_silu_2`, float32 [1, 512]; maximum absolute error 0.0009162425994873047, 1 elements outside tolerance.

## Performance

Asynchronous OpenCL event profiles separate compilation, queue delay, GPU work,
transfers and readback. They drain events after submitted work and do not insert
serialized per-kernel checkpoints. Ordered convolution was the largest measured
cost. Scheduling 128 independent output lanes preserves each output's original
accumulation and rounding. Candidate39's larger workgroups regressed and were
reverted; [profiles and comparison](candidate38-gpu-profile-comparison.json)
retain the actual APKs and five-sample scope. Those short profiles are not p99
qualification results.

| Complete-model timing | Candidate35 p50 / p99 | Candidate40 p50 / p99 | Result |
| --- | --- | --- | --- |
| Driver, 100 samples | 522.93 / 560.16 ms | 319.90 / 381.54 ms | Failed: 100/100 exceed 50 ms |
| Road | 845.98 / 881.57 ms | Unavailable | Blocked: MODERATE thermal abort after all 636 numerical checks, before completing 100 timings |

Driver p99 improves approximately **31.9%**. The accurate path still cannot sustain
20 Hz. There is no completed candidate40 road p99, and none is inferred from
partial timings or isolated five-sample profiles. Live perception is unqualified.

The Android controls input poll and residual pacing share one absolute 10 ms
deadline. Other platforms retain their 15 ms poll. Candidate38's 114.10 Hz
early-return overspeed was rejected and fixed in candidate40. The final 30-second
device-source fixture retains work and complete-cycle measurements separately:

| Controls metric | Candidate35 | Candidate40 | Result |
| --- | --- | --- | --- |
| Source / controls mean rate | 100.00 / 98.79 Hz | 99.99 / 99.83 Hz | Mean rate corrected; not a sustained qualification |
| Work p99 | 9.01 ms | 8.83 ms | passed against 10 ms |
| Complete-cycle p99 | 23.49 ms | 18.29 ms | failed |
| Complete-cycle misses | 1,303 | 1368 | Retained; every miss is reported |

Stage p99 values (not additive):

- `input_wait_budget_ms`: 9.000 ms.
- `cycle_interval_ms`: 18.364 ms.
- `scheduler_delay_ms`: 6.872 ms.
- `selfdrive_ipc_age_ms`: 21.105 ms.
- `input_handling_ms`: 11.539 ms.
- `pacing_wait_ms`: 10.817 ms.
- `computation_ms`: 6.498 ms.
- `publishing_ms`: 4.021 ms.
- `parameters_extensions_ms`: 2.397 ms.
- `cycle_before_reporting_ms`: 17.934 ms.
- `reporting_ms`: 1.286 ms.

Control computations, cancellation, stale-input checks and upstream watchdogs
are unchanged. Scheduler and input-handling jitter remain failures to resolve.

## Camera acquisition and combined inputs

Four 12-second camera-only profiles use the same continuous GLES readback
workload, with clean cooled starts. Raw sensor/elapsed-realtime/monotonic clocks,
capture-result delivery, texture acquisition, conversion, readback and publication
are separate measurements. Invalid timestamps are rejected; arrival times never
replace capture timestamps. Newest-frame selection, owned staging buffers and
100 ms freshness / 20 ms synchronization limits remain unchanged.

| Profile | Preview delivery p99 | Preview acquisition p99 | Preview late deliveries | Result |
| --- | --- | --- | --- | --- |
| record | 106.01 ms | 102.48 ms | 83 | Transport passed; sensor qualification blocked |
| preview | 90.29 ms | 83.68 ms | 1 | Transport passed; sensor qualification blocked |
| record-720p | 90.52 ms | 85.80 ms | 1 | Transport passed; sensor qualification blocked |
| record-depth2 | 91.77 ms | 85.12 ms | 2 | Transport passed; sensor qualification blocked |

Each profile still has late deliveries. These single camera-only runs cannot
select a sustained combined-workload winner. The existing record profile remains
selected pending that proof. Lens-cover trials remain deferred: no independently
verified rear sensor pair, optical geometry or mounted calibration is claimed.
Samsung multiview preview/thumbnail outputs can contain composite regions.

The final 30-second camera/UI/model-diagnostic run is **failed**: AssertionError: Insufficient fresh preview samples: {'road': 0, 'driver': 5}. Preview delivery p99 is **119.37 ms**. Fresh sampled outputs: `{'road': 0, 'driver': 5}`. Raylib report: `29.75407301833389` fps. See the [combined report](fold-candidate40-vision.json) and [sampled resources](fold-candidate40-vision.jsonl).

App-process PSS was sampled at 1,328,863 KiB and skin at 38.9°C after 32.3 seconds
including startup. These short samples do not establish stable memory or thermals.

The diagnostic CPU model path in that workload has failed original-model parity;
it cannot qualify perception or prove accurate OpenCL combined performance. The
standalone OpenCL accuracy result does not qualify that CPU path. Two-minute and
30-minute workloads are **blocked** by failed short accuracy, freshness and timing
gates. MODERATE pause, cooldown and critical-temperature abort remain active.

## Host, Fold 3, emulator and simulator validation

Build/Java unit checks and **112 focused host tests pass**. The unchanged Hyundai,
Panda and simulator protocol sources retain candidate38's **3,938 tests: 3,582
executed and 356 skipped**; they were not rerun for pacing-only candidate40.
The host-reference 795 check result is reused for identical model/native sources;
all 795 on-device comparisons were rerun on candidate40. See [host results](candidate40-host-results.json).

The command inventories retain exact requests, raw validator outcomes and separate
isolation results. A failed model or camera qualification is never relabeled as
a passed driving test because cleanup or frame transport passed.

| Surface | Validation | Validator result | Isolation |
| --- | --- | --- | --- |
| fold | opencl-reference-driver | failed | passed |
| fold | opencl-reference-road | blocked | passed |
| fold | camera-record | blocked | passed |
| fold | camera-preview | blocked | passed |
| fold | camera-record-720p | blocked | passed |
| fold | camera-record-depth2 | blocked | passed |
| fold | chain-device-steady | passed | passed |
| fold | chain-device-faults | passed | passed |
| fold | instrumentation | passed | passed |
| fold | lifecycle | passed | passed |
| fold | adb-faults | passed | passed |
| fold | vision | failed | passed |
| fold | ipc | passed | passed |
| emulator | instrumentation | passed | passed |
| emulator | camera-timing | failed | passed |
| emulator | chain-device-steady | passed | passed |
| emulator | chain-device-faults | passed | passed |
| emulator | lifecycle | passed | passed |
| emulator | adb-faults | passed | passed |
| emulator | thermal | passed | passed |
| emulator | metadrive-cancel | passed | passed |
| emulator | metadrive-stale | passed | passed |
| emulator | ipc | passed | passed |

[Fold 3 commands](fold-candidate40-validation-driver.json),
[final QNN commands](fold-candidate40-qnn-validation-driver.json) and
[emulator commands](emulator-candidate40-validation-driver.json) identify every
actual APK and isolation report. Emulator camera timing is tested separately from
deterministic injected camera/sensor/CAN inputs. The emulator camera delivers 293 frames at 29.38 fps but all 293 REALTIME capture
timestamps are in the future relative to elapsed realtime (delivery-age p99
**-28.70 ms**). They are rejected and `visionipc_valid` remains false; arrival
timestamps are not substituted. Deterministic fixtures are a separate check.
Emulator and MetaDrive establish correctness/recovery only, not phone acceleration
or physical vehicle behavior.

Every experiment stops app/test processes, verifies camera release and stable
camera services, removes experiment-created forwarding, verifies the actual APK
before/after and requires at least 30 continuous NORMAL seconds with phone skin
at or below 32°C. The emulator is stopped during phone measurements.
[Final cleanup](candidate40-final-cleanup.json) records stopped apps, released
cameras, forwarding and cooldown; the emulator is shut down after testing.

## Remaining gates

- **Failed:** QNN original-output numerical accuracy; 20 Hz accurate inference;
  fresh combined visual inputs; complete control-cycle deadlines; emulator HAL
  capture timestamps.
- **Blocked:** complete road timing after the thermal abort; two-minute and
  30-minute combined acceptance until short gates pass.
- **Blocked:** independent rear sensor proof (manual covers deferred), mounted
  optical calibration (phone unmounted), and firmware/CAN fingerprint/harness/
  vehicle behavior (Panda disconnected).
- **Retained:** TX compiled out, USB receive-only restrictions, no automatic
  flashing, factory ACC/AEB and pinned Hyundai steering/safety limits. Physical
  engagement remains disabled regardless of bench results.
