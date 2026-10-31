# Fold 3 rear pipeline bench validation, 2026-10-31

Vehicle transmission and engagement remain disabled. The phone is not in its
vehicle mount, and no Panda is connected. Camera identity, optical/mount
calibration, original model parity and complete sustained driving performance
remain qualification blockers.

## Implementation and artifact scope

The Samsung compatibility APK now publishes the selected rear GL preview,
multiview YUV thumbnail and front YUV stream through native VisionIPC. The GL
path reuses RGBA/NV12 buffers, applies the SurfaceTexture transform and converts
bottom-up RGBA to top-down BT.601 limited-range NV12. Timestamp-paired rear
candidate inputs reach the original road model. All streams and messages remain
invalid. The thumbnail's physical sensor identity has not been established;
different stream buffers or matching timestamps do not prove separate sensors.

Measured calibration loading uses the pinned narrow/large model projections,
delivered-image intrinsics and Brown distortion. It rejects changed origins,
unverified sensor origins, duplicate sensor identities and an unmeasured mount.
No calibration file was manufactured for this unmounted phone.

Bench inference pauses at Android MODERATE and requires ten continuous cool
seconds before resuming. A non-finite prediction latches a worker fault until
restart, rather than repeatedly computing failed outputs. Neither safeguard
qualifies the models or changes Android thermal protection.

[artifacts.json](artifacts.json) identifies the final APK and unchanged 17-test
instrumentation APK. The final emulator suite, thermal, lifecycle, live capture,
simulator reruns and Fold 3 30-minute capture used this APK. The initial 60-second live run, phone
instrumentation and FP32 benchmarks used
the preceding [artifact](artifacts-before-fault-latch.json); the only subsequent
app change stops repeated non-finite inference and clears stale missing-camera
report fields. Final packaging removes incremental APK ZIP slack.

## Results

| Check | Result | Evidence |
|---|---|---|
| Host focused checks | Passed, 35 tests | [host-tests.txt](host-tests.txt) |
| ADB simulator protocol | Passed, 8 tests | [host-simulator-tests.txt](host-simulator-tests.txt) |
| Gradle APK / instrumentation / JVM build | Passed | [build.txt](build.txt) |
| Fold 3 instrumentation | Passed, 17 tests | [unlocked rerun](fold3-instrumentation-unlocked.json) |
| Emulator instrumentation | Passed, 17 tests on final APK | [final suite result](emulator-final-instrumentation.json), [raw output](emulator-final-instrumentation.txt), [isolation](emulator-final-instrumentation-isolation.json); earlier suite also passed |
| Fold 3 initial live triple-stream transport | Passed capture; qualification blocked | [60-second result](fold3-rear-live.json) |
| Fold 3 FP32 diagnostic backend parity | Passed all three synthetic cases | [model report](fold3-fp32-models.json) |
| Emulator FP32 diagnostic backend parity | Passed all three synthetic cases | [model report](emulator-fp32-models.json) |
| FP32 equivalence to original goldens | Failed on both devices | Same reports, `original_model_parity`; original references/tolerances retained |
| Emulator initial live visual workload | Failed | [result](emulator-live.json), virtual camera latency/future timestamps; final rerun below |
| MODERATE pause/cooldown and critical camera abort | Passed emulator fault injection | [thermal result](emulator-thermal-verified.json), [isolation](emulator-thermal-verified-isolation.json) |
| Raylib background/resume, display resize and renderer recreation | Passed emulator component checks | [lifecycle result](emulator-lifecycle.json), [isolation](emulator-lifecycle-isolation.json); physical fold transition untested |
| Emulator final 30-second camera transport | Passed capture only; failed performance | [result](emulator-live-verified.json), [isolation](emulator-live-verified-isolation.json); virtual timestamps and model latency remain unsuitable |
| Emulator MetaDrive stale-input cancellation / factory ACC fixture | Passed | [result](emulator-metadrive-stale.json); synthetic perception/driver/calibration only |
| Emulator MetaDrive cancellation / factory ACC fixture | Passed | [result](emulator-metadrive-cancel.json); no longitudinal or below-minimum steering commands |
| Emulator controller work deadline | Failed | [runtime snapshot](emulator-metadrive-runtime.json): p99 22.87 ms, 68 misses / 1,761 work samples; synthetic fixture only |
| Fold 3 final 30-minute triple-stream transport | Passed capture; qualification blocked | [summary](fold3-final-soak-summary.json), [result](fold3-final-soak.json), [isolation](fold3-final-soak-isolation.json); no recoveries or worker restarts |
| Sustained 20 Hz inference | Failed | DM p99 64.15 ms, all 3,313 computed predictions late; thermal pause from 4:09 through completion |
| Mounted calibration / real Panda / vehicle | Blocked | Hardware/setup absent |

The initial locked-phone instrumentation attempt failed despite `am instrument`
returning zero. [Parsed acceptance](fold3-instrumentation-acceptance.json) retains
that failure; [raw output](fold3-instrumentation.txt) and wrapper evidence were
not rewritten. `validate_instrumentation.py` now requires a completed JUnit suite
with all 17 tests. The first MODERATE emulator test also failed because model
imports had not finished; the validator now waits for current worker reports
before injecting heat. Its [failure evidence](emulator-thermal.json) is retained.
The following [capture-start race](emulator-thermal-startup-race.json) and
[cold-start timing failure](emulator-final-thermal.json) are also retained.
The final thermal runner waits for actual camera frames before injecting heat.
It distinguishes app startup from the camera worker's rejection when heat is
already critical: rejection took 4 ms without opening a camera. Critical heat
injected into active capture stopped it in 1.94 seconds. The live validator
also rejected critical heat, and all overrides were reset afterward. Both
current model workers paused at MODERATE and resumed only after the ten-second
cool interval, with their PIDs unchanged.

## Initial live results

In 60 seconds, the Fold 3 delivered 1,798 GL preview/NV12 rear candidate frames,
1,798 rear thumbnails and 1,800 front frames. There were zero recoveries,
timestamp or packing errors, and camera-provider/server PIDs stayed unchanged.
Rear timestamp nearest-skew p99 was zero over 59.93 seconds of overlap. The
reported physical result map was empty, so independent rear sensor delivery
remains blocked. No images were saved by this live mode.

GL readback/conversion p99 was 25.44 ms. Rear thumbnail packing p99 was 4.36 ms;
front packing was 3.86 ms. Delivery-age p99 was 110.93 ms for GL preview,
80.29 ms for the thumbnail and 104.84 ms for the front. The original road model
computed 293 non-finite outputs and published none; DM computed 869 late outputs
and published none. Those failures motivated the numerical fault latch.

## Final 30-minute Fold 3 capture

The final APK completed 1,801.68 seconds of capture, observed for 1,806.14
seconds by the host. It delivered 54,006 GL rear-preview frames, 53,824 rear
thumbnails and 53,738 front frames, at approximately 30 fps. There were zero
recoveries, timestamp errors, packing errors, future timestamps or supervised
worker restarts. The UI averaged 29.94 fps. Both app variants and their camera clients
exited afterward; camera service/provider PIDs were unchanged and no thermal
override was active. See the [summary](fold3-final-soak-summary.json) and
[raw samples](fold3-final-soak.jsonl).

The rear physical-result map remained empty. Rear timestamp skew p99 was zero
over 1,800.01 seconds of overlap, but this does not prove independent physical
rear sensors. Optical and final-mount calibration remain missing. All streams
and model outputs remained invalid, with engagement and vehicle TX disabled.

GL readback/conversion p99 was 28.20 ms. Rear-thumbnail/front packing p99 was
5.32/5.73 ms, while delivery-age p99 was 152.11/121.56/116.16 ms for GL rear,
thumbnail and front. Deliveries over 100 ms numbered 49,039/33,424/7,844.
Stable frame counts therefore do not establish fresh perception inputs.

The road model consumed one timestamp-paired rear input, produced a non-finite
output and latched further inference until restart. DM computed 3,313 outputs,
all late, published none, and missed its 50 ms work deadline 851 times; work
p99 was 64.15 ms. Android first reached MODERATE after 249.23 seconds and both
workers remained thermally paused through completion. SEVERE first appeared
at 899.92 seconds; sampled skin temperature reached 44.6°C. Post-startup PSS
ranged from 1,346.6 to 1,409.8 MiB, with model allocations retained while paused.
Thermal safeguards behaved as designed, but sustained 20 Hz inference failed.
This workload differs from the previous regular-package road/front soak and
cannot establish an energy or memory improvement against that run.

## Isolated FP32 diagnostic comparison

All rows use two non-spinning CPU threads, 100 paced timed calls and identical
synthetic inputs for seeds 42, 7 and 123, including nonzero recurrent features.
These are separate model runs, not combined sustained or replay qualification.

| Device / model | FP32 p99 | Original CPU-2 p99 (previous run) | FP32 backend maximum absolute error | Maximum difference from original goldens |
|---|---:|---:|---:|---:|
| Fold 3 DM | 38.01 ms | 35.27 ms | 0.00001717 | 0.00943518 |
| Fold 3 driving | 46.98 ms | 190.92 ms | 0.00037575 | 0.29612827 |
| Emulator DM | 33.58 ms | — | 0.00001681 | 0.00943851 |
| Emulator driving | 42.01 ms | — | 0.00037384 | 0.29607296 |

FP32 backend parity passes the unchanged `rtol=1e-3`, `atol=1e-4` tolerance for
all cases and recurrent outputs. Equivalence to the original goldens fails;
the overall validator correctly returns failure. FP32 is **not** selected in
the driving runtime, and no original reference was rebased. Recorded replay
remains blocked. The faster driving timing is an investigation result, not an
accepted performance optimization.

## Emulator visual and simulator reruns

The final APK's 30-second emulator capture completed without recovery, with
738 rear and 739 front frames at 24.51/24.75 fps. Its transport check passed;
performance did not. The streams recorded 703/699 future delivery timestamps,
which the unchanged frame gate rejects. The UI averaged 19.27 fps. DM computed
30 frames, rejected 307 inputs, discarded 28 late outputs and published two
**invalid** diagnostic outputs; work p99 was 280.39 ms. No ultrawide camera
was provided by this virtual two-camera test, so road inference correctly waited.

The 30-second MetaDrive stale-input fixture produced 904 active steering samples
and 545 inactive samples after fault injection. It produced no longitudinal
commands and no active steering below the pinned 14.3053 m/s minimum. The
steering plant, driver monitoring, calibration and perception are synthetic;
this is a controller fixture, not visual or vehicle qualification.
The cancellation fixture separately produced 979 active samples and 568 inactive
samples after cancellation, also with no longitudinal or below-minimum steering
commands. Its controller work snapshot missed the 10 ms target: p99 22.87 ms,
with 68 deadline misses among 1,761 samples. These simulator successes do not
establish the 100 Hz performance acceptance criterion.

Every experiment has separate isolation evidence recording the installed APK
hash, clean/cool baseline, final process/camera exits, thermal override state and
camera service/provider PIDs. Wrapper failure for a blocked/failed model check
does not mean cleanup failed; inspect `after_cleanup` and the actual result.
The [emulator cleanup](emulator-final-cleanup.json) records no application
processes, no camera clients, no thermal override and no ADB forwarding; the
dedicated emulator was then powered off.
The [final delivery check](final-delivery-check.json) confirms the local and
installed APK hashes match, phone app/camera clients remain stopped and ADB
forwarding is empty. The phone had cooled to 37.5°C / LIGHT by that check;
the cleanup measurement immediately after capture was 44.6°C / SEVERE.
