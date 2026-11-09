# Fold 3 port validation

Latest local artifact: [candidate58 road and camera repair report](reports/road-camera-repair-20261107/README.md). Host tests pass 237/237; both devices pass 30 instrumentation tests on the same verified APK. Of 795 original-reference comparisons, 168 pass, 3 road cases fail and 624 remain blocked. Camera freshness, independent rear identity and the legacy controls complete-cycle gate remain failed or blocked. Longer combined workloads were not advanced. TX remains compiled out and physical engagement disabled.


The earlier [candidate45zd report](reports/full-model-controls-20261101/candidate45zd-delivery.md) records six passing Fold functional/recovery groups and eight of ten passing emulator groups. Controls work p99 is 2.449 ms; complete-cycle p99 remains 14.001 ms and fails timing. All 795 original-reference model comparisons pass on this APK; accelerator timing remains separate. Qualification and vehicle TX remain disabled.

**Overall: BLOCKED.** Vehicle transmission is compiled out and engagement remains
unconditionally disabled. The complete visual driving stack is not qualified.
Both original models pass the complete numerical gate with exact outputs on the
strict OpenCL reference measured in candidate35/40. The earlier
[candidate45zc delivery](reports/full-model-controls-20261101/candidate45zc-delivery.md)
records subsequent MNN repairs and device checks under the actual artifact hash.
Independent rear image identity, timely inference and driver monitoring,
mounted calibration and physical vehicle qualification remain unmet.
Samsung multiview transport with both rear pipelines and the front camera is
operational in the compatibility diagnostic mode. Its rear preview and
thumbnail now reach the road-model worker as explicitly unqualified candidate
streams; independent physical identities remain unproven.

Target: stock Android 15 SM-F926B, Snapdragon 888 / Adreno 660; right-hand-drive
2019 Hyundai Ioniq PHEV, factory ACC and red Panda. Branch
`codex/fold3-android-port` starts at
`6f7a2c94c10dffcf9a83b120ab1e6985e4496868`. Pinned submodules, stock Android,
factory ACC/AEB configuration and Panda firmware are preserved. The Panda is
**disconnected**; no physical CAN, harness or firmware qualification is claimed.

## Candidate45zc numerical recovery

The [candidate45zc APK](build/candidates/candidate45zc/app-debug.apk) SHA-256 is
`3e3c44ec71339060d81cb8278498df64edf840dd9705dfe4b3b4e04d8ff83361`.
Use its [delivery report](reports/full-model-controls-20261101/candidate45zc-delivery.md)
and [reproducible commands](reports/full-model-controls-20261101/candidate45zc-reproduce.md).
All 795 original-reference MNN/OpenCL comparisons pass on the Fold 3 without
CPU fallback. The exact vector FMA convolution optimization retains arithmetic
boundaries; the faster, inaccurate candidate45zb experiment remains rejected.
Accuracy, accelerator timing, controls timing and physical gates are separate
acceptance results; earlier failures below do not describe this newer APK.

## Historical candidate45h accuracy and controls delivery

Diagnostic APK SHA-256:
`9868e5b9ca7d24feaefd9acfbcdabaf4f5226e85063e03fffec4400aecc1a859`.
Use the [delivery report](reports/full-model-controls-20261101/README.md),
[machine-readable result](reports/full-model-controls-20261101/candidate45-validation.json)
and [commands](reports/full-model-controls-20261101/reproduce.md).
The final artifact passes 137 focused host tests, all 795 host conversion
comparisons, native/Java build checks and 1920 Hyundai/Panda safety/simulation/
protocol tests run (200 skipped). Fold 3's 28 instrumentation tests and all six
fixture/recovery groups pass. The emulator's initial host-renderer UI stalls
and separate renderer retry are retained in the current validation record.
Both installed app and test hashes match the archived artifact on both devices.

MNN executes without CPU fallback or unexpected non-finite values, but six of
15 initial seed output-array comparisons fail. The remaining 780 recurrent/
replay comparisons and accelerator workloads are blocked. Controls work p99
is 6.208 ms; complete-cycle p99 is 17.290 ms with deadline misses, so overall
timing fails. New metric definitions do not clear old failures. Bench fixture
passes cannot establish visual inference freshness, independent physical
rear sensors, mounted calibration or physical Panda/vehicle qualification.
TX remains compiled out. Cooling optimization is deferred; experiment cleanup,
NORMAL thermal entry, MODERATE pause and critical abort remain enforced.

## Candidate44 NPU recovery delivery

Diagnostic APK SHA-256:
`f18262c93c2132b1e61a1746af6b2c55dcae8820e3d8a9db0dd84bc5472c6360`.
Use the [recovery report](reports/npu-recovery-20261101/README.md),
[acceptance record](reports/npu-recovery-20261101/candidate44-acceptance.json)
and [commands](reports/npu-recovery-20261101/reproduce.md). Device recovery
fixtures and small accelerator probes cannot qualify full-model accuracy,
sustained timing, camera identity or physical vehicle behavior. Both alternative
GPU engines and HTP remain diagnostic. The original 795 comparisons are retained;
portable host exports pass them without changing original models or goldens.
Cooling changes remain deferred and all existing thermal gates remain enabled.

## Historical candidate40 delivery and acceptance

The final debug APK SHA-256 is
`bd42b65a3538b8ad73834e90826eb10a98d8afeacbe9d5aae5877d43a25e30ec`.
Use the [delivery report](reports/performance-fixes-20261101/README.md),
[machine-readable validation](reports/performance-fixes-20261101/candidate40-validation.json),
[artifact record](reports/performance-fixes-20261101/candidate40-artifact.json),
[installed hashes](reports/performance-fixes-20261101/candidate40-installed.json)
and [commands](reports/performance-fixes-20261101/candidate40-reproduce.md).
Earlier failures retain their own APKs and are not replaced by later successes.

The build, Java unit checks and **112 focused host tests** pass. The unchanged
Hyundai/Panda/protocol sources retain the candidate38 **3,938-test result
(356 skipped)**. The final Fold 3 APK passes all **795 original-reference
comparisons exactly**, with the original models, goldens and tolerances.
Driver inference still misses 20 Hz; road timing stops at MODERATE thermal
status. QNN's numerical gate remains failed. Emulator fixtures and MetaDrive
establish correctness and recovery only. They cannot qualify phone acceleration,
independent rear sensors, mounted calibration or physical vehicle integration.

All **22 instrumentation tests on each device**, lifecycle recovery, ADB fault
rejection, IPC wakeup and steady/faulted device-source process-chain checks pass.
Emulator display changes, thermal protection, MetaDrive cancellation and stale
input rejection also pass. Its actual camera timing fails: all 293 capture
timestamps are in the future; those frames remain invalid. These
behavioral passes do not clear performance: controls average 99.83 Hz with
18.29 ms complete-cycle p99 and 1,368 cycle misses; the combined camera/UI/model
diagnostic fails with 119.37 ms preview delivery p99 and zero fresh sampled road
outputs. Raylib averages 29.75 fps. The final report retains separate emulator
camera timing and deterministic fixture results.

Each experiment records cleanup, camera release, forwarding, thermal baseline
and actual APK hash. Entry requires 30 continuous NORMAL seconds with phone
skin at or below 32°C. MODERATE pause and critical abort are retained. Longer
combined workloads remain blocked until the short correctness, freshness and
timing gates pass. TX remains compiled out regardless of bench results.
Cooling and temperature optimization are deferred at the user's request.

## Historical candidate35 delivery and acceptance

Archived candidate35 debug APK SHA-256:
`38645e91f806076517de1497cdb6b5c0cbf4873cfecf5efabb980297a014ad51`.
The [artifact record](reports/blocker-fixes/candidate35-artifact.json),
[installed hashes](reports/blocker-fixes/candidate35-installed.json),
[changes](reports/blocker-fixes/candidate35-changes.md) and
[reproduction commands](reports/blocker-fixes/candidate35-reproduce.md) identify
the exact build. The complete
[validation report](reports/blocker-fixes/candidate35-validation.md) and
[machine-readable summary](reports/blocker-fixes/candidate35-validation-summary.json)
separate validator results, isolation and acceptance. Earlier failures retain
their [own artifact identities](reports/blocker-fixes/candidate35-retained-failures.json).

Host build/unit checks, **100 focused tests (98 at build)**, **3,938 Hyundai/Panda/protocol tests
(356 skipped)** and the complete host original-model reference gate pass. The
same archived app and test APKs pass **22 instrumentation tests on each device**.
Both installed hashes match the artifact record. The
Fold 3 [original-model accuracy gate](reports/blocker-fixes/candidate35-model-accuracy-gate.json)
passes **795 exact comparisons** at unchanged `rtol=1e-3, atol=1e-4`.
Strict OpenCL road/driver work p99 is **881.57/560.16 ms**; the faster native-MAD
diagnostic fails parity and is rejected. No live backend is qualified.

The on-device 100 Hz source fixture produces every required derived output,
with no unsafe outputs or worker failures. Factory-ACC cancellation and stale
CAN checks pass. Fold 3 controls work p99 is **9.01 ms**, but complete-cycle p99
is **23.49 ms** with 1,303 cycle misses. Camera-only continuous preview delivery
p99 is **105.66–108.83 ms**; the combined visual run reaches **160.99 ms** and
produces no road predictions. Both freshness and complete timing gates fail.
Raylib averages 29.72 fps with synthetic sources and 29.50 fps in the short
combined visual run; neither is 30-minute acceptance evidence.

The deferred lens-cover experiment, optical/mounted calibration, physical
cover/inner-display transitions, actual Panda firmware/CAN/harness behavior and
the 30-minute combined workload remain **blocked or unverified**. Emulator
display changes and deterministic inputs cannot establish those physical gates.
TX remains compiled out regardless of bench results.

## Historical candidate32 delivery identity and test scope

The historical candidate32 artifact, commands and passed/failed/blocked results are in the
[blocker-fix report](reports/blocker-fixes/README.md). Both devices were tested with the same
Samsung compatibility debug package (`com.samsung.sunnypilot.fold3`). The new
runtime includes NumPy 2 and native localization/MPC dependencies. The Fold 3
sources-only process-chain test produces all eleven required derived outputs
without worker restarts; cancellation and stale CAN pass, with TX disabled.
This does not qualify camera inference, control deadlines or a physical vehicle.

The candidate32 app SHA-256 is
`a62b55187b2a60a15c9b7d34cb5e6e462cc00c353971d8d126940960861a2810`.
The [installation record](reports/blocker-fixes/candidate32-installed.json)
verifies the same app and test APKs on both devices. The
[host results](reports/blocker-fixes/candidate32-host-results.json) record
**92 focused checks**, **3,938 Hyundai/Panda/protocol checks (356 skipped)**,
the complete original-model reference gate and focused Ruff checks passing.
Both devices pass **20 instrumentation tests**.

The [complete Fold 3 numerical gate](reports/blocker-fixes/candidate32-model-accuracy-gate.json)
passes all **795 output comparisons exactly**: seeds 42/7/123, 32 recurrent
frames per seed and 60 camera-replay frames per model. Integer-bit concatenation
preserves half subnormals; integer midpoint correction fixes the normalization
square root that caused the remaining recurrent divergence. The half-dot
optimization preserves pair/accumulator rounding and passes the same gate.
Original models, goldens and `rtol=1e-3, atol=1e-4` remain unchanged.

Accuracy does not establish live inference readiness. Road/driver p99 is
**1,021.53/559.41 ms**, with **100/100** timed 50 ms misses per model.
The accurate OpenCL backend remains diagnostic. The
[reproduction commands](reports/blocker-fixes/candidate32-reproduce.md)
and [artifact record](reports/blocker-fixes/candidate32-artifact.json) retain
the exact build and test scope.

Both devices pass the fault and steady sources-only process-chain fixtures.
All required derived outputs are present, with zero worker failures/restarts;
factory-ACC cancellation and stale CAN pass. The phone fixture runs after the
emulator has stopped. It averages **100.87 Hz**, but records **24 work misses
and 1,213 complete-cycle misses in 2,701 samples**. Work p99 is **9.683 ms**;
complete-cycle p99 is **24.365 ms**. Input wait remains included in complete
cycle timing. ADB exchange p99 is **15.328 ms**, so fixture transport jitter
limits the timing conclusion. The behavioral pass does not establish sustained
100 Hz operation or physical vehicle readiness.

The [final device summary](reports/blocker-fixes/candidate32-validation-summary.json)
records **18 isolated experiments: 13 passed, four failed, one blocked**.
The two model validators fail performance despite exact numerical results;
the other failures are phone camera freshness and emulator camera timing.
The serialized kernel profile is diagnostic-only and remains blocked as a
qualification run. Lifecycle recovery, emulator resizing, IPC wakeups,
numerical/thermal/USB rejection components, process fault handling and emulator
MetaDrive cancellation/stale-input fixtures pass within their recorded scope.

The [isolation audit](reports/blocker-fixes/candidate32-isolation-audit.json)
passes for all 18 experiments: unchanged installed APK, released cameras,
stopped app/test processes and stable camera providers. Every baseline has at
least **30.67 continuous NORMAL seconds**, with skin at most **31.4°C**.
Validator failures retain their original nonzero wrapper exits.
[Final cleanup](reports/blocker-fixes/candidate32-final-cleanup.json) verifies
no app/test processes or ADB forwards, cameras released, the emulator stopped,
and the phone at **30.0°C after 32.33 continuous NORMAL seconds**.
No 30-minute run proceeds while the short freshness/timing gates fail.

## Historical candidate21 validation

Historical **candidate21** has app SHA-256
`4c48351cf6d6166ad32d0ab309072e6900438decb1751f86ab30fcdb5109bbfe`;
the [installation record](reports/blocker-fixes/candidate21-installed.json)
verifies identical app and instrumentation APKs on both devices. The
[candidate21 summary](reports/blocker-fixes/candidate21-validation-summary.json)
records **77 focused host tests**, **1,927 Hyundai/Panda/protocol tests (200
skipped)** and **20 instrumentation tests per device**. Seventeen isolated device
experiments complete: **13 passed, 4 failed**. Lifecycle/display recovery,
numerical/thermal/USB rejection components, msgq wakeups, process-chain fault
handling and MetaDrive fixtures pass within their recorded scopes.

The failed experiments include original OpenCL model qualification, the
Fold 3 combined camera/model freshness run, and actual emulator Camera2 timing.
Both originals fail seeds 42/7/123, recurrent sequences and the pinned replay.
DM/driving inference p99 is **75.40/105.58 ms**. Triple-stream capture has no
disconnects in 30 seconds, but preview/front delivery-age p99 is
**124.68/104.60 ms**. Emulator capture timestamps are about 30 ms ahead of the
bracketing elapsed-realtime samples and remain rejected.

The steady phone process-chain fixture averages **100.45 Hz**, with work p99
**9.54 ms**, but records **20 work and 1,218 complete-cycle deadline misses**.
Its behavioral pass does not qualify sustained timing. Separate fault runs
verify cancellation and stale CAN. Independent rear sensor identity, correct
model mapping, mounted calibration and physical Panda/vehicle checks remain
blocked. A new 30-minute workload is blocked by these unmet gates. TX remains
compiled out and engagement disabled. [Final cleanup](reports/blocker-fixes/candidate21-final-cleanup.json)
confirms released cameras, no app/test processes or ADB forwards, the emulator
stopped and 29.0°C after more than 30 continuous NORMAL seconds.

The full dirty Android port's [structural quality report](reports/blocker-fixes/candidate21-final-quality-delta.json)
also remains failed; it includes concurrent work and is not a result specific
to the small candidate21 code-identity repair.

The following candidate10 results are **historical**.

Candidate10 (`566e4c25…e83a9a`) adds PID/thread-specific msgq wakeups, validated
with the actual kernel on both devices. Eight short full-chain checks complete
without worker failures/restarts; both devices pass 19 instrumentation tests.
The source fixture now reports absolute 100 Hz pacing and transport jitter.
The historical [summary](reports/blocker-fixes/candidate10-validation-summary.json)
records 21 passed and 5 failed isolated experiments, with identity/calibration,
recurrent/replay and physical Panda/vehicle gates still blocked. Full original
model parity, camera/model freshness and sustained performance remain failed.

## Previous rear-pipeline delivery

The following earlier Samsung compatibility APK has SHA-256:
`f725600b0b9bcd90208429f5ac2dc85c38541175e4fa30eda85c0aabe7d574f5`.
The [rear-pipeline bench report](reports/rear-pipeline/README.md) records the
continuous GL-to-NV12/VisionIPC bridge, measured-projection loader, numerical
fault latch, MODERATE thermal pause and fresh emulator/Fold 3 results. Host
checks passed 35 tests; the Fold 3 completed all 17 instrumentation tests on
the preceding build before the numerical latch was added, and the emulator
completed all 17 again on the final APK. FP32 diagnostic
backend parity and isolated <50 ms p99 timings pass on both devices, but
equivalence to the original model reference still fails. FP32 is not adopted.
That APK also completed 30-minute triple-stream capture without recovery
or worker restarts; DM work p99 was 64.15 ms and thermal pause began at 4:09,
so sustained perception still fails. Mounted calibration and physical
Panda/vehicle tests remain blocked.

The following previous results retain their original APK identities; current
reruns and their limits are listed in the report above.

Previous Samsung compatibility APK SHA-256:
`0e05c9927a2a12755876ba539ac3eec7d2bcee1f32e89228d2e03c5498cc850a`.
The camera changes, fresh build, host/device checks and each installed hash are
recorded in the [camera validation report](reports/camera-compat/README.md). This build adds Samsung's
mode-37 multiview layout, opaque-preview readback and front capture. It retains
the unconditional engagement and vehicle-transmission blocks.

Previous regular-package debug APK SHA-256:
`e5347db807f6551886d7de706d6ebb939c51dcd25c5c0ade1cd57b7fb24941ee`.
[Installed hashes](reports/installed-artifacts.json) match on the Fold 3 and the
API 35 ARM64 emulator. Build commands are in [README.md](README.md); artifact
sizes, source pins and hashes are in [artifacts.json](reports/artifacts.json).

That APK uses two non-spinning model threads. Instrumentation, phone
60-second live capture, the phone soak attempt, emulator lifecycle/resize,
emulator live capture and thermal-abort tests exercise this APK. The immediately
preceding `31c9028b…` APK used four live threads. Its unchanged camera APIs,
individual model benchmark profiles, phone lifecycle, fault tests, MetaDrive and
Panda-absence checks are retained under [four-thread-validation](reports/four-thread-validation/installed-artifacts.json).
Each experiment's `*-isolation.json` records its actual installed APK hash.
These results are not silently attributed to the newer APK.

## Previous acceptance matrix

These rows record the preceding delivery. The
[blocker-fix report](reports/blocker-fixes/README.md) supersedes them for the
current APK and retains the intervening failures with their actual hashes.

| Area | Result | Evidence and limit |
|---|---|---|
| ARM64 APK / installation | Passed | Gradle build, native Bionic modules, signing and installed hashes verified |
| Focused host checks | Passed | 35 Android tests and 8 simulator protocol tests; [current results](reports/rear-pipeline/host-tests.txt), [protocol results](reports/rear-pipeline/host-simulator-tests.txt) |
| JVM tests | Passed | Two unchanged FrameStats tests in the compatibility build |
| Fold 3 / emulator instrumentation | Passed | 17 each, including continuous GL/NV12 delivery and opaque preview pixels/orientation; [phone](reports/rear-pipeline/fold3-instrumentation-unlocked.json) before the numerical latch, [emulator](reports/rear-pipeline/emulator-final-instrumentation.json) on final APK |
| Exact preprocessing / IPC | Passed component checks | Padded/rotated NV12, native gather equivalence, malformed buffers, VisionIPC and parser checks |
| Standard Camera2 matrix (prior build) | Partial | Phone 16/26 and emulator 6/12 attempts passed; no disconnects, all readers/devices closed; ordinary dual rear combinations failed |
| Samsung rear multiview + front transport | Passed | Mode 37, logical 20 + front 1; 60-second capture, provider remains stable; [camera details](CAMERAS.md) |
| Independent wide / ultrawide driving inputs | Blocked | Preview and thumbnail candidates now reach VisionIPC/modeld; physical sensor identity and optical/mount calibration remain unproven |
| Standard rear/front capture regression | Passed capture only | [Phone 60 seconds](reports/camera-compat/fold3-final-road-front.json), [emulator 10 seconds](reports/camera-compat/emulator-final-road-front.json); virtual-camera timing does not qualify model inputs |
| Phone 60-second live diagnostic (previous regular APK) | Passed capture/preview only | [Previous run](reports/fold3-final-live.json): 1,800 rear / 1,794 front frames, no recovery, all DM predictions late |
| Phone final 30-minute Samsung triple-stream capture | Passed capture; failed perception | [Current summary](reports/rear-pipeline/fold3-final-soak-summary.json): 54,006/53,824 rear candidate frames and 53,738 front, zero recoveries; sensor identity/calibration unqualified |
| Phone previous regular-APK 30-minute diagnostic | Passed capture; failed perception | [Previous soak result](reports/fold3-final-soak.json); no road inference or full planning/control workload |
| Raylib lifecycle / emulator resize | Passed component checks | [Previous phone test](reports/four-thread-validation/fold3-final-lifecycle.json), [current emulator](reports/rear-pipeline/emulator-lifecycle.json); physical fold transition untested |
| Process fault recovery | Passed component checks | [Phone](reports/four-thread-validation/fold3-final-faults.json), [emulator](reports/four-thread-validation/emulator-final-faults.json) |
| MetaDrive / factory ACC / cancellation | Passed fixture checks | [Previous phone](reports/four-thread-validation/fold3-final-metadrive.json), current emulator [stale input](reports/rear-pipeline/emulator-metadrive-stale.json) and [cancel](reports/rear-pipeline/emulator-metadrive-cancel.json); synthetic perception/DM/calibration, no physical TX |
| Thermal pause/cooldown and critical abort | Passed emulator component checks | [Current APK](reports/rear-pipeline/emulator-thermal-verified.json): MODERATE pauses both model workers, ten-second cooldown, critical camera rejection/abort; override reset afterward |
| Emulator live workload | Passed capture; failed performance | [Current 30-second run](reports/rear-pipeline/emulator-live-verified.json): ~24.5 fps cameras / 19.3 fps UI; future sensor timestamps rejected, DM p99 280.39 ms |
| Original model parity | Failed | Three synthetic cases per model, original tolerance unchanged; finite fallback still differs |
| Sustained 20 Hz / p99 <50 ms perception | Failed | DM live work misses deadlines and all outputs are late; driving CPU exceeds 50 ms and native FP16 outputs are non-finite |
| Complete 100 Hz controls | Blocked; emulator fixture deadline failed | Current synthetic control work p99 22.87 ms / 68 misses in 1,761 samples; complete physical input and planning chain unqualified |
| Panda receive-only transport | Passed software / blocked physical | IN-only bindings, pinned decoder/health checks and negative descriptor tests; no Panda connected |
| Vehicle fingerprint / ignition / harness / firmware | Blocked | Real Panda and vehicle checks still required |
| Engagement / vehicle transmission | Disabled | Unconditional engagement block and no USB OUT transport exposed |

The current Samsung triple-stream capture completed 30 minutes with zero
recoveries or worker restarts, a 29.94 fps UI, and verified process/camera exits.
The road worker latched one non-finite result and DM published none of its
3,313 computed predictions because all were late; DM work p99 was 64.15 ms.
Thermal pause began after 249.23 seconds and persisted through completion;
Android reached SEVERE and sampled skin temperature reached 44.6°C. All inputs
remained uncalibrated and invalid. See the [current bench report](reports/rear-pipeline/README.md).

The **previous regular-package** 30-minute road/front capture and preview test **passed** with zero recoveries. It delivered 53,964 rear and 54,003 driver frames at 29.98/30.01 fps; both image-handler drains and camera-close callbacks completed. Final rolling nearest-skew p99 was 16.50 ms, and delivery-age p99 was 84.43/94.31 ms. The host observed 1,817.18 seconds including post-capture sampling.

**Sustained perception failed.** Android reached severe thermal status (3), with skin temperature sampled up to 44.6°C. DM computed 25,909 frames, all late, and published zero; rolling work p99 reached 82.57 ms with 19,786 cumulative 50 ms deadline misses. The UI averaged 29.96 fps. Memory after 60 seconds ranged from 916.1 to 947.4 MiB PSS. See the [soak summary](reports/fold3-final-soak-summary.json) and [raw samples](reports/fold3-final-soak.jsonl). This workload excludes road inference, localization, planning, active physical controls and Panda, so it cannot qualify the complete driving stack.

## Changes and safeguards

Camera teardown stops/aborts requests, marks streams closing, drains image
callbacks, waits for device-close callbacks and only then releases the native
publisher and ownership lock. Every matrix attempt reports both drain and close
completion. The activity waits for its actual landscape surface before starting
capture, fixing the observed initial portrait-surface race. Rotation changes
fail the session rather than silently applying the wrong transform. Supported
noise reduction, edge enhancement and stabilization are disabled for the record
request template. Vendor/provider disconnects remain qualification failures.

The runtime is rebuilt from pinned ONNX Runtime source without optional
KleidiAI/SME/SVE kernels. This removes the emulator's `RDSVL` illegal-instruction
crash. Portable exports omit host-specific NHWC operators; their manifests
retain failed numerical checks against the original goldens. Neither successful
loading nor finite output establishes parity. The original precision and
`rtol=1e-3`, `atol=1e-4` remain unchanged. No precision experiment is accepted.

Native row gathering reuses buffers and reduces preprocessing work. The model
loop caps starts at 20 Hz without catch-up bursts and uses two non-spinning CPU
threads after measured isolated CPU savings. Queues remain bounded; missing,
future, duplicate, >100 ms old or >20 ms skewed frames are rejected. Late and
non-finite predictions are discarded. Every visual diagnostic message remains
invalid; calibration is not replaced with invented transforms. Bench road
inference now receives both Samsung candidate streams. Their physical identity
and geometry remain unqualified, and the numerical fault latch stops repeated
non-finite inference until the worker restarts.

The separate Android services supervise upstream controlsd/dmonitoringd,
Android IMU/GNSS/battery/thermal inputs and Raylib. The full localization,
calibration, planning and vehicle stack remains incomplete. APK NumPy 1.26.2
does not satisfy upstream NumPy >=2; only the exercised subset is validated.
Updates are APK installs. AGNOS/system-service/firmware-flashing paths are excluded.

Panda bindings only expose whitelisted vendor IN reads and bulk IN endpoint
0x81. They verify red hardware and pinned protocol versions, require silent
safety mode, clear controls flags and no faults, and reject malformed packets.
There is no USB OUT, heartbeat, reset, safety-mode change or flashing API.
RX CAN remains invalid and uses host USB arrival timestamps. RHD behavior,
factory ACC/AEB and the approximately 51 km/h steering minimum remain intact.
All upstream torque/disengagement/watchdog safeguards are retained.

## Previous isolation and retained failures

Every new experiment stops the application and instrumentation package, checks
that processes and camera clients are gone, waits at least 30 seconds with
normal thermal state (phone skin <=32°C), and verifies cleanup afterward.
Application data/calibration and caches are preserved. Workers are not restored
between experiments. Thermal overrides are never used on the physical phone.
The emulator was left clean with its override reset and then powered off.
The final phone experiment also left both app variants, instrumentation and
camera clients stopped; the phone was still thermally hot (44.6°C, SEVERE) at
cleanup, with no override. No subsequent workload was started.

The earlier emulator lifecycle precondition failed while a System UI ANR dialog
paused rendering; [failure evidence](reports/emulator-systemui-anr.txt) and the
[failed isolation report](reports/four-thread-validation/emulator-final-lifecycle-isolation.json)
are retained. After closing that dialog, the previous regular APK lifecycle/resize test
passed. A fresh cold-boot check again required closing a System UI ANR dialog;
[checked-in launcher validation](reports/emulator-final-wrapper-validation.json)
then passed lifecycle/resize, but unattended startup remains failed. Its live throughput still failed. Camera providers were not restarted
to manufacture a clean result.

Earlier phone camera stalls, the `e2ee0027…` startup disconnect/recovery and the
20.8-minute critical-thermal attempt remain in their original reports. The
landscape startup fix's intermediate result is retained under
[pre-power-fix](reports/pre-power-fix/fold3-final-live.json). Earlier 72 runtime
checks, Hyundai/IPC 20 tests run / 2 skipped and Panda safety 1,701 run / 198
skipped are historical unchanged-pin evidence, not new runs in this iteration.
Emulator, fixture and diagnostic successes cannot qualify a real vehicle.

### Candidate45zl continuation, 2026-11-03

[Artifact-bound results and commands](reports/full-model-timing-repair-20261103/README.md).
Final APK SHA-256: `0e035bbab912cc5607bde2a1a934a433d452cde8e46d6a69a6e09ae2ff79aa2a`.
Installed emulator hash verified. 189 host tests and 28 instrumentation tests pass;
nine of ten emulator experiment groups pass, with isolation passing in all ten.
The native camera timing test still fails: 284 future sensor timestamps, despite
successful capture/teardown. Synthetic fixture success does not clear this gate.
Full-model ncnn parity still fails 8/553 outputs on seed42; additional seeds,
recurrent/replay qualification and longer performance stages were not advanced.
Fold 3/Panda/vehicle validation is blocked by disconnected hardware. TX stays
compiled out, engagement disabled, and cooling optimization deferred.

### Candidate65 physical and emulator validation, 2026-11-07

[Full report, reproducible commands and retained failures](reports/gpu-npu-recovery-20261107/README.md).
Final APK SHA-256: `8170e642d4c002f84fe2c1cb95ae79505a11be7e4016c707aa0cc3e6cc84b7ec`.
The installed hash matches on the Fold 3 and dedicated ARM64 Android15 emulator.
All 795 original-reference comparisons, 250 host tests and 30 instrumentation
tests on each device pass. Lifecycle, supervision, IPC, numerical/thermal faults,
USB rejection, stale-input handling and factory-ACC cancellation pass; emulator
display recovery and MetaDrive fixtures pass.

Inference deadlines, three-camera freshness and controls complete-cycle timing
still fail. The emulator camera produces 343 invalid future capture timestamps;
all are rejected and no frames are published. Independent rear proof, mounted
calibration and physical Panda/vehicle checks remain blocked. No sustained
perception or vehicle qualification is claimed. All 15 final-candidate experiment
isolation receipts pass, including the established cooldown. Final cleanup
verifies stopped app/test processes, released cameras, removed forwarding and
matching installed hashes; the owned emulator is shut down. TX and engagement
remain disabled; no cooling optimization was performed.


### Candidate66 physical and emulator validation, 2026-11-08

[Full report and artifact-bound summary](reports/weight-residual-recovery-20261108/README.md).
APK SHA-256: `ab383423a396c3903b18a39938542ec0999cd28da1b36f459fdb8ed495701ecd`.
The final installed hash matches on the Fold 3 and ARM64 Android15 emulator,
including the separate host-GPU repeat. All 795 original-reference comparisons,
251 host tests and 30 instrumentation tests on each device pass. Lifecycle,
supervision, IPC, numerical/thermal faults, USB rejection, cancellation and
stale-input handling pass. The 30-second host-GPU MetaDrive repeat passes with
0.574 m maximum lateral error; the initial software-renderer out-of-road failure
is retained. Synthetic simulation cannot qualify perception or a real vehicle.

Inference deadlines, three-camera freshness, independent rear identity and the
legacy controls complete-cycle gate remain failed or blocked. The emulator
camera rejects 316 future timestamps from 325 acquisitions, with 9 publications;
its timestamp contract remains failed. No combined sustained perception stage
is advanced. Real mounted calibration and disconnected Panda/vehicle checks
remain blocked. Candidate66 is diagnostic, not promoted as a performance gain.

All 17 experiment isolation receipts pass, with process/camera/forwarding cleanup
and the established cooldown. Both devices' final cleanup/hash checks pass; both
owned emulator instances are stopped. A transient offline ADB shutdown receipt
is retained alongside the subsequent successful deregistration/process check.
All 163 archived product-source hashes still match. TX and engagement remain
disabled; no cooling optimization was performed.

### Candidate67 validation (2026-11-08)

[Full report and reproducible commands](reports/pipeline-specialization-20261108/README.md)
and [compact receipts](build/candidates/candidate67-specialized-tail/validation-summary.json).
The same main/test APK hashes are verified on Fold3 and ARM64 Android15 emulator.
Host suite: 255 tests pass; instrumentation: 30/30 each; Hyundai behavior: 13 run,
2 skipped; safety: 1899 run, 198 skipped. All 795 original model comparisons pass
at unchanged tolerances, with no CPU fallback. Lifecycle recovery and emulator
display override pass; physical hinge movement was not exercised. Emulator
MetaDrive stale-input/cancellation passes using synthetic perception and plant.

Inference, 50 ms camera freshness and legacy controls complete-cycle timing fail;
independent rear identity remains blocked. Emulator camera returns 341 future
capture timestamps, all rejected, zero publications. Combined qualified
30-second/two-minute/30-minute progression is not run because preceding gates
fail. No simulated result qualifies physical perception or vehicle behavior.

All 15 experiment isolation receipts pass, including camera/process release,
forwarding cleanup and 30 continuous cool NORMAL seconds. Both final installed
hash/cleanup checks pass; owned emulator is stopped. All 83 immutable artifact
hashes and 165 archived/current source hashes match. Initial disk-limited emulator
launch and pre-install archive-check failure remain retained with the successful
retry. Repository-wide Ripwire quality delta did not return and remains blocked;
`git diff --check` passes. TX stays compiled out and engagement disabled.

### Candidate68 validation (2026-11-08)

[Artifact-bound report](reports/dispatch-batching-20261108/README.md).
APK `925b4b0ab7b40e50c6b19b3ca8f8f4fba2ef5eefd74fcc8b5d0588e2dff8a048`.

Same main/test APK hashes verified on Fold3 and ARM64 Android15 emulator. Host 257 tests and instrumentation 30/30 per device pass; all 795 original-reference comparisons pass. Hyundai behavior: 13 run / 2 skipped; safety: 1899 run / 198 skipped. Lifecycle, synthetic controls and emulator MetaDrive faults pass; inference speed, camera freshness and complete-cycle timing fail. All 16 isolation receipts and final cleanup checks pass; owned emulator stopped. All 83 immutable artifacts and 167 source files match recorded hashes. Ripwire quality delta remains blocked after no response. No qualified combined-load progression is run while prerequisite gates fail.

TX and engagement remain disabled; cooling optimization is deferred and thermal safeguards remain active. Prior candidates and failures are preserved.

### Candidate69 validation (2026-11-08)

[Artifact-bound report](reports/exact-normalization-20261108/README.md).
APK `16d07d62d1dd70fc2c9beacbf0c9fa2308f26686cbc1d32a04e1dc35ac7b168a`.

All 795 original-reference comparisons pass at unchanged tolerances, no CPU fallback. Host 258 tests, emulator instrumentation 30/30, lifecycle, synthetic controls and emulator MetaDrive pass. Fold instrumentation fails one of 30 tests: the front-manual rejection is correct but a case-sensitive diagnostic-text assertion is stale. The original failure is retained. Inference has no meaningful improvement; road median regresses. The 2 ms diagnostic exposure reduces isolated ultrawide publication p99 to 55.873 ms but 298/358 publications still exceed 50 ms. Three-stream publication p99 is 107.362 ms; independent rear identity remains blocked. Controls complete-cycle timing and emulator capture timestamps still fail. No qualified combined perception progression is run.

Both APK installed hashes match; all 17 isolation receipts and final cleanup checks pass. Owned emulator stopped. All 83 immutable artifacts and 169 source files were audited before the next candidate's shader change; frozen source independently matches. Disk/SDK/archive failures and bounded Ripwire timeout remain retained. TX and engagement stay disabled; thermal protections stay active and cooling optimization remains deferred.

### Candidate70 validation (2026-11-08)

[Artifact-bound report](reports/packed-normalization-20261108/README.md).
APK `9ef37ce35eabdafdba80c0039b8bbfc4425068152c923c461af0f0f5bbe61db8`.

The exact byte-normalization shortcut now covers ncnn's actual packed Vulkan upload path as well as scalar execution. Driver normalization GPU time falls from 7.698 to 2.472 ms in separate diagnostic profiling. All 795 original-reference comparisons pass at unchanged tolerances without CPU fallback. Unprofiled driver/road inference p99 is 149.742/356.939 ms: both still fail the unchanged 50 ms complete-pipeline budget. Road median improves against candidate69 but remains worse than candidate68; no road performance qualification is claimed.

The supported manual-2ms-low-latency ultrawide diagnostic publishes 357 frames, 305 over 50 ms, publication p99 55.674 ms. Three-camera transport passes but rear publication p99 is 105.184 ms; independent rear identity remains blocked. Emulator invalid capture timestamps remain rejected. Fold synthetic controls work p99 is 2.177 ms at 99.994 Hz, but complete-cycle p99 12.301 ms and 1312 raw cycle misses still fail. Fault fixtures also fail complete-cycle timing; metrics and counters are unchanged. No production controls timing fix is claimed.

Host 258 tests pass. Final instrumentation passes 30/30 on each device after correcting only the case-sensitive diagnostic-text assertion. The initial Fold 1/30 failure, original test APK and manifest remain archived; the production APK stays byte-identical. Lifecycle, synthetic controls behavior and emulator MetaDrive fault checks pass. Main and revised test APK installed hashes match both devices. All 16 experiment isolation receipts, final cleanup checks and both owned emulator shutdowns pass. All 84 immutable artifacts and 169 source hashes match; `git diff --check` passes. Bounded Ripwire quality delta remains blocked. No qualified combined perception progression is run while preceding gates fail.

TX and engagement remain disabled. Processes/cameras/forwarding are clean and final cooldowns pass. Cooling optimization is deferred, thermal safeguards remain active, and mounted calibration, independent rear-lens proof and disconnected Panda/vehicle checks remain blocked.

### Candidate71 validation (2026-11-08)

[Artifact-bound report](reports/spatial-reuse-20261108/README.md).
APK `8d2359a5a0b6ccdf61a73e4c7d04bdb2ef2b0e52f09799c4cc99bd032901c762`.

Four-position convolution weight reuse passes all 795 original-reference comparisons at unchanged tolerances without CPU fallback, but is rejected as a performance optimization. Road conv_88 GPU mean regresses from 17.067 to 24.085 ms. Unprofiled driver/road inference p99 remains 150.972/364.663 ms; every timed sample exceeds 50 ms. Candidate70 remains the prior comparison baseline. The new host test compiles the actual shader body and checks scalar/packed layouts and partial tiles against independent scalar convolution.

Ultrawide direct YUV publication p99 is 55.882 ms; PRIVATE/GLES is slower at 60.403 ms. Both fail the unchanged 50 ms deadline before inference. Three-camera transport passes, rear acquisition/publication p99 is 92.106/97.519 ms, and front delivery p99 is 96.420 ms. Independent rear identity remains blocked. Emulator capture timestamps are invalid and all 340 frames are rejected. Fold controls work p99 is 2.291 ms at 99.996 Hz, with zero scheduled misses/skips; unchanged complete-cycle p99 is 12.294 ms with 1277 raw misses, so that gate still fails. No production controls fix is claimed.

Host 260 tests, instrumentation 30/30 per device, lifecycle, synthetic controls behavior and emulator MetaDrive fault checks pass. Main/test APK hashes match both devices. All 15 isolation receipts, final cleanup and owned emulator shutdown pass. All 83 immutable artifacts and 171 source hashes were audited before candidate72's subsequent shader change. Completed source is retained. Ripwire quality delta remains blocked after no result; `git diff --check` passes. Qualified combined-load progression is not run while preceding gates fail. TX and engagement stay disabled; cooling optimization remains deferred with thermal safeguards active. Physical identity, mounting and disconnected Panda/vehicle gates remain blocked.

### Candidate72 validation (2026-11-08)

[Artifact-bound report and reproducible commands](reports/pair-reuse-20261108/README.md).
APK `dd05f2f5236b1bed3a414ea17a6e1fabdec5dc7af04688fdb12253cac8f169a7`.

The spatial convolution patch now shares weights between two live output accumulators, processing four positions as two bounded pairs. Original accumulation order, correction helpers and rounding boundaries are preserved. All 795 original-reference comparisons pass at unchanged tolerances without CPU fallback. The targeted road conv_88 GPU mean improves from candidate70's 17.067 to 15.040 ms. Unprofiled road median improves 7.0 percent, from 335.224 to 311.609 ms; p99 regresses from 356.939 to 363.188 ms. Driver median/p99 is 137.223/148.841 ms. Every timed sample still exceeds 50 ms, so this is limited median/kernel progress, not complete-pipeline qualification. Candidate71's failed four-position optimization remains archived.

Ultrawide direct YUV publication p99 is 56.001 ms versus PRIVATE/GLES 60.890 ms; neither meets 50 ms before inference. Three-stream transport passes, with rear acquisition/publication p99 99.460/104.056 ms and front delivery 97.442 ms. Rear identity and mounted geometry remain unqualified. All 342 emulator camera frames with invalid timestamps are rejected. Deterministic injected inputs remain a separate correctness check.

Matched Fold controls fixtures average 100 Hz with zero scheduled completion misses/skipped slots. Steady work p99 is 2.141 ms, but complete-cycle p99 remains 12.142 ms with 1231 raw misses. Fault-fixture complete-cycle p99 is 12.136 ms with 1271 raw misses. No controls algorithm, pacing or timing definition changed; no production timing fix is claimed. Raw CPU/off-CPU, wakeup, input, publishing and reporting measurements remain archived.

Host 260 tests, instrumentation 30/30 per device, lifecycle, cancellation/stale-input behavior and emulator MetaDrive checks pass. Final main/test APK hashes match both devices. All 15 isolation receipts and final cleanup/cooldown checks pass; the owned emulator is stopped. All 83 immutable artifact hashes and 171 source hashes match. The first packaging attempt was rejected before installation because Chaquopy build.json was missing; its APK, hash and reason are retained. Regenerating the standard metadata task restores all 20 unchanged Python assets in the tested final APK. Ripwire quality delta remains blocked after no result; `git diff --check` passes.

Inference, three-camera freshness and legacy controls complete-cycle gates still fail. Qualified combined 30-second/two-minute/30-minute progression is not run. TX remains compiled out and physical engagement disabled. Final processes, cameras and forwarding are clean; continuous NORMAL cooldowns exceed 30 seconds at skin temperatures below 32 C. Cooling optimization remains deferred. Independent rear identity, mounted calibration and disconnected Panda/vehicle checks remain blocked.

### Candidate73 road dispatch validation (2026-11-08)

[Artifact-bound report and commands](reports/dispatch-recovery-20261108/README.md).
APK `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`.

Added explicit road-only Adreno 660 Vulkan dispatch diagnostics for 131072, 262144 and 524288. Default execution and driver kernels are unchanged. Accuracy records bind the setting to the actual artifacts and device; diagnostic overrides cannot qualify live perception. Each setting and the default passes all 795 original-reference comparisons at unchanged tolerances without CPU fallback. Road median/p99: default 339.251/363.897 ms; 131072 353.897/367.452 ms; 262144 342.880/362.271 ms; 524288 309.712/352.664 ms. All 100 samples per model/configuration exceed 50 ms. The best setting is diagnostic, not promoted. GPU profiles show 3/2/1 internal submissions respectively; internal waits include GPU execution. The one-versus-two accumulator follow-up remains incomplete.

Host 263 tests, additional qualification tests, native build and APK/Java checks pass. Fold 3 and emulator each pass 30 instrumentation tests, lifecycle and controls behavior fixtures; emulator MetaDrive passes. Fold steady work/legacy cycle p99 is 2.107/12.201 ms at 100 Hz, with zero scheduled misses but 1270 raw cycle misses. Emulator steady work/cycle p99 is 3.180/14.606 ms, with 7 scheduled misses and 17 skipped slots. Timing gates remain failed; definitions/counters are unchanged. No new camera timing or identity qualification is claimed.

All 16 isolation receipts pass. Final main/test installed APK hashes match on both devices; 83 immutable artifacts and 171 source files match. Final cleanup/cooldown passes, no owned processes/cameras/forwarding remain, and the emulator is shut down. Initial build and low-disk emulator launch failures remain archived alongside the successful retries. Qualified combined-load progression is not run while isolated inference fails. TX remains compiled out and engagement disabled. Cooling optimization is deferred with thermal safeguards active. Rear identity, mounted calibration and disconnected Panda/vehicle checks remain blocked.
