# Candidate45zd Fold 3 recovery

Vehicle TX remains compiled out, USB is receive-only and physical engagement
is disabled. Cooling optimization is deferred; thermal protections remain
active. The complete visual driving stack is not qualified.

The [installable APK](../../build/candidates/candidate45zd/app-debug.apk) SHA-256 is
`e7ae5461257aae2abef31e1631c4c79e7a32bc1513523f5d38db021ab9a2334e`.
The test APK SHA-256 is
`b8336ebab92c8bfbe076f45c42c462acb5fc1f35c0072453dc39ef07c7ab3945`.
The [immutable archive](../../build/candidates/candidate45zd/artifacts.json),
[implementation](candidate45zd-implementation.md) and
[reproducible commands](candidate45zd-reproduce.md) retain the actual artifact,
model and provider identities. The [package proof](candidate45zd-package-proof.json)
verifies policy v3 inside the APK and identical packaged native libraries and
converted graphs relative to candidate45zc.

All 177 host checks and the Java/build checks pass. The Hyundai/Panda/simulator
suite runs 1,921 tests, with 1,719 passes and 202 skips. The build succeeds in
3 minutes 48 seconds. Earlier candidate toolchain/build failures remain
retained under their original names.

## Numerical accuracy

All 795 original-reference comparisons pass on the Fold 3 under this APK: 636 road and 159 driver comparisons, all three recurrent seeds and the pinned 60-frame replay. Original tolerances remain `rtol=1e-3`, `atol=1e-4`, with no CPU fallback or unexpected non-finite outputs. All ten isolated accuracy experiments pass cleanup and artifact checks. See [accuracy evidence](candidate45zd-accuracy.json).

## Controls and device recovery

The [Fold 3 installed hashes](candidate45zd-installed-fold.json) match both
archives. Policy v3 repairs the observed core4-unavailable selection defect
without changing control calculations, pacing, stale-input handling,
cancellation or watchdogs. The matched 30-second fixture passes functionally:

| Metric | Candidate44 | Candidate45zc | Candidate45zd |
| --- | ---: | ---: | ---: |
| Work p99 | 7.304 ms | 6.545 ms | 2.449 ms |
| Complete-cycle p99 | 16.613 ms | 15.493 ms | 14.001 ms |
| Work misses | 8 | 0 | 0 |
| Complete-cycle misses | 1,350 | 1,364 | 1,296 |
| Scheduled-completion misses | Not recorded | 12 | 0 |
| Skipped slots | Not recorded | 0 | 0 |

Candidate45zd records 2,661 samples, 99.997 Hz and 29.671 fps Raylib. Schema2
release-to-completion p99 is 6.088 ms. Reporting p99 is 0.367 ms and scheduler
delay p99 is 2.949 ms. Raw cycle intervals and existing counters are preserved;
the changed metric cannot clear the complete-cycle failure. Overall controls
timing remains **failed**.

The [four-run comparison](candidate45zd-controls-comparison.json) also retains
candidate45za's faster earlier complete-cycle measurement. Candidate45zd
actually applies `[4, 6]`; candidate45zc had rejected `[5, 6]` because core4
was absent. Available masks differ, so this comparison does not isolate a
causal speedup from the new fallback. Its exact core4-unavailable selection
and rejection of mismatched capacities are covered by the host regression
test. All six Fold functional/recovery groups now pass, including the resumed IPC check. Each completed experiment verifies cleanup and the actual APK.

The emulator passes eight of ten groups. MetaDrive cancellation and native camera timing fail; their raw reports and isolation records remain retained. Both installed emulator APK hashes match the archive. The phone hashes were verified again after reconnection. The full accuracy gate passes; isolated accelerator timing fails for both models. Combined workloads remain blocked by their other prerequisites.

## Accelerator performance

Driver timing completes 100 samples: p50 **1,240.544 ms**, p99 **1,276.617 ms**, with all 100 missing the 50 ms target. Performance is **failed** despite numerical acceptance. Road timing also completes 100 samples: p50 **2,237.387 ms**, p99 **2,261.544 ms**, with all 100 missing the 50 ms target. Both isolated runs pass cleanup and artifact checks. These are complete inference wall times, not individual GPU event timings.

Stage wall-time p99 is 4.290 / 747.151 / 511.884 ms for driver upload / execution / readback, and 3.103 / 1,627.248 / 571.958 ms for road. These boundaries may include deferred GPU synchronization; they are not asynchronous per-kernel measurements and their percentiles must not be added. Small upload times do not explain the slow total. Kernel execution and output materialization require further profiling before another optimization can be accepted. Neither model is promoted for live perception.

## Experiment cleanup

Both timing experiments verify stopped app/test processes, released cameras, empty experiment forwards and unchanged installed APK hashes. The [final cleanup](fold-candidate45zd-final-cleanup.json) passes: no app/test processes, no camera ownership, no forwards, NORMAL thermal status and 31.6°C after 32.3 continuous eligible idle seconds.

## Remaining gates

Combined visual workloads cannot advance while inference performance, controls timing and input freshness remain unqualified. Independent rear-sensor identity remains unproven while the
manual lens-cover trials are deferred. The phone is unmounted and Panda
disconnected: mounted calibration, firmware compatibility, passive vehicle
fingerprint, harness and physical vehicle behavior remain blocked. Emulator
fixtures cannot qualify those gates.

Candidates35,40,44,45za, rejected45zb and candidate45zc retain their original
artifacts and results. Candidate45zc's emulator camera timestamp and MetaDrive
cancellation failures are retained, rather than transferred into a pass here.
