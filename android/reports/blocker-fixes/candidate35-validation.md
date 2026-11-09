# Candidate35 Fold 3 bench validation — 2026-11-01

**Overall: BLOCKED.** Vehicle TX remains compiled out and physical engagement
unconditionally disabled. No live model backend is accepted. The phone is
unmounted, the Panda disconnected and the manual rear lens-cover trials deferred.
This report distinguishes implemented code, bench component passes and unmet
acceptance gates. No 30-minute acceptance soak or vehicle qualification is claimed.

Branch: `codex/fold3-android-port`; build HEAD
`7631d5248cc9efd10b962107aadacbb908a9f14d` with the existing dirty source preserved.
The [artifact record](candidate35-artifact.json) binds packaged sources and tests.
The [delivery audit](candidate35-delivery-audit.json) verifies both archived and
installed APKs and confirms application and instrumentation sources are unchanged.
Three host-only validation scripts were corrected after a thermal harness failure;
[their original build sources and updated hashes](candidate35-host-tooling-update.json)
are retained. No application or APK changed for the repeat.

| Artifact | SHA-256 |
| --- | --- |
| [App APK](../../build/candidates/candidate35/app-debug.apk), 815,252,690 bytes | `38645e91f806076517de1497cdb6b5c0cbf4873cfecf5efabb980297a014ad51` |
| [Test APK](../../build/candidates/candidate35/app-debug-androidTest.apk), 4,492,519 bytes | `98d31adfd4ef7679e70a5fb51083bff55380a0c1080336e9a63278e531efa9fe` |

Both hashes match the [Fold 3 and API 35 emulator installations](candidate35-installed.json).
Build and experiment commands are in [candidate35-reproduce.md](candidate35-reproduce.md);
individual command inventories are in
[the phone batch](fold-candidate35-validation-driver.json) and
[emulator batch](emulator-candidate35-validation-driver.json).

## Implemented changes

Camera callbacks, GL work and publication use separate bounded workers. Three
owned NV12 buffers prevent overwritten consumer inputs; newer pending frames
replace older pending frames. Sensor timestamp outputs explicitly disable preview
re-spacing/readout timing, while raw times and elapsed/monotonic samples remain
visible. GPU conversion, bounded PBO/fence readback and the checked fallback remain.
Slow-consumer ownership and teardown checks run on both devices.

Control report sorting and file I/O run on a bounded background writer. Heartbeats,
work, stage and complete-cycle misses remain visible; writer failures propagate.
A debug-only on-device 100 Hz source fixture removes per-packet ADB scheduling
from the complete process-chain test and preserves namespace, message and TX guards.
The native-MAD performance trial is diagnostic only and cannot write qualification.
Host validation now checks thermal state before launching a workload, retries
force-stop for late-starting workers and checks the actual thermal rejection message.
See [the change details](candidate35-changes.md).

## Acceptance results

| Gate | Result | Evidence and limit |
| --- | --- | --- |
| Build, Java unit checks and focused host tests | **Passed** | 100 host tests after the tooling corrections (98 at build); Gradle build/unit tasks; [host results](candidate35-host-results.json) |
| Hyundai/Panda/protocol suite | **Passed run** | 3,938 tests run, 356 skipped; no physical Panda |
| Host original-model reference | **Passed** | Original hashes, goldens and tolerances unchanged |
| Installed instrumentation | **Passed** | 22 tests on each device, including camera teardown, GPU conversion, numerical faults, IPC and USB rejection paths |
| Fold 3 original-model numerical gate | **Passed** | 795 exact comparisons; seeds 42/7/123, recurrent sequences and pinned 60-frame camera replay |
| Accurate 20 Hz inference | **Failed** | Road/driver work p99 881.57/560.16 ms; both miss all 100 timed 50 ms deadlines |
| Diagnostic native MAD | **Failed; rejected** | 216/636 road and 137/159 driver comparisons fail; never eligible for live qualification |
| Triple-stream transport | **Passed component** | Four camera profiles plus combined load; zero disconnect recoveries and timestamp errors |
| Camera freshness and usable live road predictions | **Failed** | Camera-only preview p99 105.66–108.83 ms; combined preview p99 160.99 ms; zero road predictions |
| Complete process-chain behavior | **Passed component** | All required derived outputs; factory-ACC cancellation/stale CAN; zero unsafe outputs or worker failures on both devices |
| Sustained complete 100 Hz controls | **Failed** | Phone work p99 9.01 ms, complete-cycle p99 23.49 ms; 1,303 cycle misses |
| Approximately 30 fps Raylib | **Passed short component** | Phone 29.72 fps with synthetic sources; 29.50 fps with cameras/models/UI; sustained acceptance unproven |
| Lifecycle and display tests | **Passed component** | Both lifecycle runs; emulator 1280×720 resize/restore and renderer recreation; physical fold transition unverified |
| ADB malformed/stale/partial/disconnect faults | **Passed** | Both devices; physical namespace remains unavailable to fixtures |
| Native IPC/wakeup probe | **Passed** | Both devices; separate probe binary hashes retained in raw reports; APK VisionIPC separately instrumented |
| Emulator thermal pause/recovery/critical aborts | **Passed component** | MODERATE pause, cooldown and critical camera abort; overrides restricted to emulator and cleared |
| Emulator MetaDrive cancellation/stale faults | **Passed component** | Debug controller fixture, synthetic steering plant; no physical Panda or vehicle qualification |
| Actual emulator Camera2 timing | **Failed** | 289/289 future timestamps rejected; deterministic inputs tested separately |
| Rear sensor identity, optical/mounted calibration | **Blocked** | Deferred three-repeat lens-cover experiment and final mount |
| Real Panda firmware/passive fingerprint/harness/vehicle | **Blocked** | Panda disconnected; no active firmware queries, flashing or TX |
| 30-minute combined acceptance workload | **Blocked; not run** | Short identity, freshness, numerical-backend and complete timing gates unmet |

## Model and performance evidence

The [original-reference accuracy gate](candidate35-model-accuracy-gate.json)
contains 636 road and 159 driver comparisons with zero differing outputs and
zero unexpected non-finite values at `rtol=1e-3, atol=1e-4`. This is a strict
standalone OpenCL numerical pass, not a live performance qualification. The live
CPU diagnostic backend retains its separate failed original-reference parity;
standalone OpenCL agreement does not qualify that backend.

The [performance comparison](candidate35-performance-comparison.json) retains
candidate32, candidate33 and candidate35 with their own APKs. Candidate33's
four-channel FP32 tile passed accuracy but regressed work p99 to
2,109.00/1,203.29 ms and was reverted. Candidate35 native MAD reaches
138.39/102.51 ms p99 but fails the original outputs and remains rejected.
These are separate cooled runs, not a controlled A/B benchmark. No tolerance,
model, replay, thermal limit or freshness limit was relaxed.

With on-device sources the phone control fixture has source rate 100.00 Hz and
controls mean 98.79 Hz. Source work p99 is 13.89 ms with 183
source misses. Control input handling p99 is 19.23 ms, computation 5.83 ms,
publishing 4.12 ms, parameters/extensions 4.44 ms and reporting 1.34 ms. Stage
percentiles are not additive. Controls have 17 work misses and 1,303
complete-cycle misses. The emulator has work/complete-cycle p99
2.98/18.17 ms and mean
100.49 Hz. Scheduling and deadline failures remain explicit.

## Cameras, memory and thermal scope

The [four 12-second camera-only runs](candidate35-camera-comparison.json)
perform continuous conversion/readback and retain raw clocks and sensor metadata.
All four remain blocked on physical rear identity and fail preview freshness.
The [combined 30-second run](fold-candidate35-vision.json) records 893 composite,
895 routed rear-preview readbacks and 894 front frames. The aggregate vendor
`preview_frames` field sums preview and extra record-reader counts; it cannot
establish another sensor or 60 fps. Composite/preview/front delivery p99 is
136.49/160.99/107.97 ms. Road inputs are rejected; 432 driver predictions yield
430 late outputs and two invalid publications. No valid driving output results.

The [resource log](fold-candidate35-vision.jsonl) records total app-process PSS
1,223,296 KiB (1.17 GiB) at 30.6 s and skin temperature 36.1°C with normal Android
thermal state and no override. This short startup/load measurement cannot prove
steady memory, sustained cooling or absence of watchdog violations over 30 minutes.
The actual [emulator camera run](emulator-candidate35-camera-timing.json) has
median delivery age -30.93 ms and p99
-27.97 ms. Its future sensor times remain invalid;
arrival timestamps are not substituted. Injected inputs do not qualify this test.

## Retained failures and final cleanup

The [history index](candidate35-retained-failures.json) retains earlier raw
failures and actual APKs. Candidate33's physical-phone lifecycle command wrongly
included display resize and failed before the test; candidate35 omits that flag
on the phone and passes. Candidate34 was built and host-tested but never installed.
The first candidate35 thermal test retained a stale error-message assertion and
a shutdown race after starting an already-hot workload. Its failed raw report is
retained. The corrected host tooling passes [the repeat](emulator-candidate35-thermal-retry.json)
against the same APK, including MODERATE cooldown, both critical camera cases,
preflight rejection/cleanup and new-activity stop. The original candidate32 wrapper counts are retained separately from its actual
13 passed / four failed / one blocked validation results.

Candidate35 has **26 isolated experiments**. Raw report statuses:
**15 passed, 4 blocked, 7 failed**. Child exit statuses: **15 passed, 11 failed**. Camera transport can pass
while identity is blocked; its nonzero child exit remains preserved. Reference
model commands fail their overall performance gate despite exact numerical outputs.
No failed or blocked validation is relabeled as a pass by a clean isolation result.

The [isolation audit](candidate35-isolation-audit.json) records **26 passed**,
minimum **30.75 continuous NORMAL seconds** and
maximum baseline skin temperature **31.3°C**.
It verifies stopped app/test processes, released cameras, stable providers,
unchanged APKs and forwarding cleanup between experiments.
[Final cleanup](candidate35-final-cleanup.json) is **passed**: no app/test
processes, camera ownership, forwards or thermal overrides on either device.
The phone finishes at **30.4°C after
32.20 continuous NORMAL seconds**. The emulator
is stopped after its own cleanup. Physical folds, final mounting, lens covers and
real Panda/vehicle checks remain unverified; TX stays disabled regardless of bench results.

A [later read-only phone snapshot](candidate35-phone-final-snapshot.json) confirms
25.1°C, NORMAL thermal state, the same installed APK, and no app/test processes,
camera ownership, forwarding rules or thermal overrides. ADB lists only the phone;
the dedicated emulator has exited.
