# Candidate45za Fold 3 recovery

Vehicle TX is compiled out, USB transport is receive-only and physical
engagement remains disabled. Cooling optimization is deferred; thermal
safeguards and experiment isolation remain active.

## Artifact

[Installable APK](../../build/candidates/candidate45za/app-debug.apk), SHA-256:
`17a87c7909af196941a6659a68c905ba41d5590dc361b8a309e2c4af53e95822`.
The test APK SHA-256 is
`b8336ebab92c8bfbe076f45c42c462acb5fc1f35c0072453dc39ef07c7ab3945`.
Installed hashes match on the [Fold 3](candidate45za-installed-fold.json) and
[dedicated ARM64 Android15 emulator](candidate45za-installed-emulator.json).
The [immutable artifact record](../../build/candidates/candidate45za/artifacts.json)
records models, converted graphs, implementation and packaged provider libraries.

Use the [reproducible commands](candidate45za-reproduce.md) and
[implementation details](candidate45za-implementation.md). Candidates35,40,44,
the earlier candidate45 delivery below, and subsequent intermediate experiments
retain their own artifacts and failures. Results cannot be transferred between
APKs.

## Full-model accuracy: passed

The build and Java unit checks pass. All **176 focused host tests** pass;
**1,921 Hyundai/Panda/simulator tests run, 202 skipped**, pass. Converted ONNX
graphs pass **795 original-reference comparisons** on the host. Phone execution
is checked separately at unchanged `rtol=1e-3`, `atol=1e-4`.

The complete untapped MNN/OpenCL buffer graphs pass **795/795 comparisons on
the Fold 3**: [636 road checks](fold45za-mnn-road-accuracy-complete.json) and
[159 driver checks](fold45za-mnn-driver-accuracy-complete.json), including all
three seeds, recurrent sequences and the pinned 60-frame camera replay. There
are no unexpected non-finite outputs or CPU partitions. Original models,
goldens and tolerances remain unchanged. The formerly failing road seed42
frame26 passes; the largest road absolute error across the full suite is
`3.814697265625e-06`, within tolerance.

All six Fold 3 validation groups pass, including 28 instrumentation tests,
process-chain operation, faults, lifecycle and IPC. The matched controls
fixture runs at 100.003 Hz, with work p99 **2.554 ms** versus candidate44's
**7.304 ms**. Complete-cycle p99 remains **13.787 ms**, with 1,269 retained
legacy deadline misses; release-to-completion p99 is 6.117 ms, with zero
scheduled deadline misses and zero skipped slots. The timing gate remains
failed. UI averages 29.57 fps. These injected-input results do not qualify
visual perception. See the [matched comparison](candidate45za-performance-comparison.json).

Both isolated inference timing attempts were stopped by the MODERATE thermal
safeguard before their 100-sample sets completed. Performance remains blocked;
last-call wall-stage snapshots are diagnostic only and are not p99 results.
The [validation report](candidate45za-validation.json) retains these outcomes.
The 30-second, two-minute and 30-minute visual workloads have not advanced.
Numerical accuracy does not clear freshness, timing or hardware gates.

## Emulator

All 28 instrumentation tests and the complete process-chain, fault, lifecycle,
IPC, MetaDrive cancellation/stale-input and thermal recovery groups pass on
this installed APK. The separate native virtual-camera timing check fails:
282 frames carry capture timestamps about 31 ms ahead of elapsed realtime.
Those frames remain rejected. The [camera evidence](emulator-candidate45za-camera-timing.json)
retains raw timestamps, clock samples and capture-result errors. Deterministic
injected-input passes do not clear this HAL timing failure. The emulator was
stopped after [cleanup and idle cooldown](emulator-candidate45za-final-cleanup.json).

## Remaining physical gates

Rear sensor independence has not been established by the deferred manual
lens-cover trials. Bench crop/calibration metadata cannot qualify the final
vehicle mount. Panda is disconnected, so firmware compatibility, passive
vehicle fingerprint, harness and real vehicle behavior remain blocked. Stock
camera freshness and the sustained visual workload require separate acceptance.
Simulator and emulator passes cannot clear physical gates.
