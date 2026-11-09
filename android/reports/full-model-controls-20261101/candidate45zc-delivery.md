# Candidate45zc Fold 3 recovery

Vehicle TX is compiled out, USB is receive-only and physical engagement is
disabled. Cooling optimization is deferred; thermal safeguards remain active.
The complete visual driving stack is not qualified.

## Artifact and reproduction

The [installable APK](../../build/candidates/candidate45zc/app-debug.apk) SHA-256 is
`3e3c44ec71339060d81cb8278498df64edf840dd9705dfe4b3b4e04d8ff83361`.
The test APK SHA-256 is
`b8336ebab92c8bfbe076f45c42c462acb5fc1f35c0072453dc39ef07c7ab3945`.
Installed hashes match on the [Fold 3](candidate45zc-installed-fold.json) and
[dedicated ARM64 Android15 emulator](candidate45zc-installed-emulator.json).
The [immutable archive](../../build/candidates/candidate45zc/artifacts.json)
retains models, converted graphs, implementation and actual stripped provider
libraries. The packaged MNN OpenCL SHA-256 is
`1f1edda01c03d8d18dc1d7a91dba6336fbb6cc3e2ab53e37ab77c16fa45508d0`.

Use the [reproducible commands](candidate45zc-reproduce.md) and
[implementation details](candidate45zc-implementation.md). The
[validation record](candidate45zc-validation.json),
[accuracy record](candidate45zc-accuracy.json) and
[matched performance comparison](candidate45zc-performance-comparison.json)
bind results to the artifact. The successful
build log is `candidate45zc-build-retry1.log`; the interrupted build is retained.
All 176 host checks pass with the matched Xcode toolchain. The earlier default
toolchain run failed two native linking checks; those failures and the focused
retry remain retained. The Hyundai/Panda/simulator suite runs 1,921 tests:
1,719 pass and 202 are skipped. All 795 host graph-conversion comparisons pass.

## Full-model accuracy: passed

The complete untapped MNN/OpenCL buffer graphs pass **795/795 original-reference
comparisons on the Fold 3**: [636 road](fold45zc-mnn-road-accuracy-complete.json)
and [159 driver](fold45zc-mnn-driver-accuracy-complete.json), including all three
seeds, recurrent sequences and the pinned 60-frame replay. There are no
unexpected non-finite outputs or CPU partitions. Original models, goldens and
`rtol=1e-3`, `atol=1e-4` are unchanged. Every part passes isolation and binds to
this actual APK, models, provider libraries, implementation and device.

The retained candidate45za repair corrects the first divergent road attention
reduction. Candidate45zc then vectorizes convolution input loads while retaining
the exact FMA order. The [standalone operator comparison](candidate45zc-operator-comparison.json)
is bit-for-bit equal in 36 cases; the three large geometries improve by
8.02–9.21 percent in GPU event medians. These separately hashed operator
diagnostics ran with candidate45za installed and do not establish APK inference
p99 or sustained throughput. Candidate45zb's much faster MAD arithmetic fails
full-model tolerances and remains [rejected](candidate45zb-rejection.json).

## Fold 3 performance and recovery

The [driver benchmark](fold45zc-mnn-driver-timing.json) completes 100 untapped
full-model samples after the 795-comparison accuracy gate: p50 **1,234.22 ms**,
p99 **1,255.43 ms**, with all 100 samples missing 50 ms. Performance **fails**.
The [road benchmark](fold45zc-mnn-road-timing.json) is **blocked** by Android
MODERATE thermal status before its complete sample set. No p99 or partial-sample
throughput is inferred. Both experiments pass isolation and retain this APK.
Upload/compute/readback wall stages remain separate from GPU event timings;
queued GPU execution can complete during readback.

The matched 30-second controls fixture passes functionally and measures
99.989 Hz, with Raylib at 29.689 fps. Work p99 is **6.545 ms**, compared with
candidate44's **7.304 ms**; complete-cycle p99 is **15.493 ms**, compared with
**16.613 ms**. It records zero work misses, **1,364 complete-cycle misses**,
**12 scheduled-completion misses** and zero skipped slots. Timing **fails**.
Schema2 release-to-completion p99 is 9.263 ms; that metric does not clear the
misses or replace the preserved complete-cycle definition.

The report exposes a CPU-selection defect: the allowed mask is
`[0, 1, 2, 5, 6]`, while policy v2 requires core4 and rejects the available
equal-capacity pair `[5, 6]`. That failed policy and this measured run remain
retained. The earlier candidate45za run was faster, but does not qualify this
artifact. All six Fold 3 functional/recovery groups pass: steady process-chain
operation, chain faults, all 28 instrumentation tests, lifecycle recovery,
ADB faults and IPC. Every group passes isolation under this same APK. The
[final cleanup](fold-candidate45zc-final-cleanup.json) records stopped processes,
released cameras, empty forwarding and the 30-second NORMAL cooldown.
Numerical accuracy alone cannot qualify perception or clear performance gates.
Cooling optimization remains deferred; thermal protections were not changed
to finish a benchmark.

## Emulator: 8 of 10 groups passed

All 28 instrumentation tests, steady process-chain operation, chain fault
injection, lifecycle, ADB faults, IPC, MetaDrive stale inputs and thermal
recovery pass. All ten experiments pass isolation and retain the same APK hash.

The 30-second MetaDrive cancellation workload **fails** when the synthetic car
leaves the road after cancellation. It records 125 inactive post-fault samples
and zero longitudinal-active samples, then the world exits. That response does
not turn a terminated workload into a pass. The
[failure analysis](candidate45zc-metadrive-failure-analysis.json) retains the
actual APK, source evidence hashes and `out_of_road` reason. Acceptance is
unchanged; synthetic steering/perception cannot qualify a vehicle.

The separate [native camera timing test](emulator-candidate45zc-camera-timing.json)
**fails**: 291 frames at 29.26 fps carry capture timestamps ahead of validated
elapsed realtime (median delivery age -30.90 ms), with 289 capture-result
timestamp errors. All 291 future timestamps remain rejected; arrival times
are not substituted. Deterministic injected-input passes do not clear this
HAL timing failure. The emulator was stopped after
[cleanup and 30 seconds at NORMAL status](emulator-candidate45zc-final-cleanup.json).

## Remaining gates

The 30-second, two-minute and 30-minute combined visual workloads remain
blocked until accuracy, freshness, timing and watchdog prerequisites pass.
Independent rear sensors remain unproven while manual lens-cover trials are
deferred. The phone is unmounted and Panda disconnected: mounted calibration,
firmware compatibility, passive fingerprint, harness and vehicle behavior
remain blocked. Bench or simulator results cannot clear those gates.

Candidates35,40,44, the accurate candidate45za baseline, candidate45zb's
rejection and earlier candidate45 failures remain preserved under their actual
artifact identities.
