# Fold 3 accuracy and controls recovery

Current work: [candidate45zd delivery](candidate45zd-delivery.md), [validation](candidate45zd-validation.json) and [commands](candidate45zd-reproduce.md). Both installed APK hashes match; Fold functional checks pass, emulator has two failures, and all 795 original-reference accuracy comparisons now pass. Isolated accelerator timing remains separate.

The earlier numerical-validation artifact and results are in the
[candidate45zc delivery](candidate45zc-delivery.md), with
[reproducible commands](candidate45zc-reproduce.md) and
[implementation details](candidate45zc-implementation.md).
Installed APK hashes bind the recorded Fold 3 and emulator experiments to
candidate45zc. All 795 phone original-reference accuracy comparisons pass;
performance and device recovery retain their separate acceptance results.
Candidate45za, rejected candidate45zb and earlier failures remain retained below.

## Historical candidate45h delivery

**Acceptance: FAILED; qualification: BLOCKED.** Candidate45 implements targeted MNN numerical repairs,
actual OpenCL tensor tracing, absolute-slot Android controls scheduling and
reporting outside the controls Python process. Full-model accelerator accuracy
and complete-cycle timing still fail. The diagnostic backend is not promoted.
Vehicle TX is compiled out, USB is IN-only and physical engagement is disabled.
Cooling optimization remains deferred; thermal safeguards remain active.

Use the [commands](reproduce.md), [validation record](candidate45-validation.json),
[accuracy record](candidate45-accuracy.json),
[matched controls comparison](candidate45-performance-comparison.json) and
[retained experiment index](retained-experiments.json). Earlier failures and
candidates35,40,44 remain available under their own artifact identities.

## Artifacts

The [installable diagnostic APK](../../build/candidates/candidate45/app-debug.apk)
is an immutable copy of candidate45h. Its SHA-256 is
`9868e5b9ca7d24feaefd9acfbcdabaf4f5226e85063e03fffec4400aecc1a859`.
The instrumentation APK SHA-256 is
`b8336ebab92c8bfbe076f45c42c462acb5fc1f35c0072453dc39ef07c7ab3945`.
Both installed APKs were hashed on the
[Fold 3](candidate45-installed-fold.json) and
[ARM64 Android15 emulator](candidate45-installed-emulator.json).
The [artifact record](candidate45-artifacts.json) binds engine source/patch,
derived graphs, provider libraries, original models and device identity.
Java classes retain `ai.sunnypilot.fold3`; this camera-compatible APK uses
application ID `com.samsung.sunnypilot.fold3`. The experiment wrapper rejects a
model validator targeting a different installed variant.

## Accuracy: failed

Original model hashes, goldens and tolerances `rtol=1e-3`, `atol=1e-4` remain
unchanged. All **795 host comparisons** pass for the derived ONNX graphs.
That verifies graph conversion, not execution by MNN on the phone. Historical
strict OpenCL candidates35/40 retain their 795 exact physical comparisons.

Candidate45 MNN/OpenCL buffer execution completes all operations without CPU
fallback or unexpected non-finite outputs. Full untapped models still fail:

| Model / seed | Output elements outside tolerance | Maximum absolute error |
| --- | ---: | ---: |
| Driver / 42 | 7 / 553 | 0.00390625 |
| Driver / 7 | 3 / 553 | 0.00390625 |
| Driver / 123 | 3 / 553 | 0.00390625 |
| Road / 42 | 436 / 2576 | 0.078125 |
| Road / 7 | 543 / 2576 | 0.1875 |
| Road / 123 | 234 / 2576 | 0.09375 |

Of the initial **15 output-array comparisons, nine pass and six fail**.
The remaining **780 recurrent/replay comparisons are blocked** by this gate;
they have not passed on candidate45. The original 795-comparison contract is
preserved. Accelerator timing, combined visual workloads and qualification
remain blocked. `fold45h-mnn-buffer-retry.json` is the valid full-model result;
the earlier `fold45h-mnn-buffer.json` targeted the old app variant and is
explicitly excluded from candidate45 accuracy evidence.

Reproducible MNN patches retain FP16 casts, target half-precision GEMV
accumulation, normalize proven constant negative GatherND indices and repair
the buffer-mode Tanh implementation. The actual-input Tanh replay is exact
across 691,200 and 172,800 elements; see
[the local result](driver45h-tanh-actual-input.json). Its previous 40 FP16
rounding differences are eliminated. These local repairs do not clear full-model
accuracy. Road seed42 has more failed elements than candidate45g (436 versus
364); the regression is retained and this backend is not selected for driving.

Bounded traces record actual OpenCL assignment, dtype, NCHW layout, shapes,
original-node mapping and hashes. Diagnostic readback serializes execution and
is excluded from performance results. In the sampled windows, the first 128
driver tensors pass; the later dense/ReLU region fails. The first observed road
violation is `model/node_conv2d_10`, one element outside tolerance. Replaying
that convolution with its actual GPU input passes locally
([evidence](road45-conv10-actual-input.json)). The captured driver Gemm+ReLU
also passes locally ([evidence](driver45-dense335-actual-input.json)).
These results indicate upstream error propagation; they do not identify the
first divergent operation across unsampled graph regions. There is no basis
to claim a broad convolution or dense-kernel replacement has fixed the models.

The [first driver Conv replay](driver45-stem-actual-input.json) also passes the
original tolerance. Its 12 sampled largest-error outputs match host ordered
FMA accumulation (`ky`, `kx`, input channel, bias first); they differ from MNN.
This establishes a rounding diagnostic without requiring bit identity as the
acceptance rule. [Window64](driver45-window64.json) and
[window96](driver45-window96.json) retain the passing comparisons separately.

[Window288](driver45-window288.json) narrows a later driver violation to the
classifier half-GEMV (`node_linear`, three elements outside tolerance).
[Replaying that layer with its captured GPU input](driver45-classifier-actual-input.json)
matches the host exactly across all 512 outputs. The upstream backbone still
produces slightly different inputs; replacing this GEMV is not supported by
that evidence. These samples do not fill the unsampled 128–287 tensor interval.

## Controls: work improved, complete-cycle timing failed

Android controls pace once at the start of an absolute 10 ms slot and read the
latest inputs without blocking. Expired slots are counted and skipped without
catch-up bursts. Other platforms retain their existing input wait and pacing.
Cancellation, stale-input checks, control calculations and watchdogs remain.

Owned bounded sample batches replace repeated stage-buffer snapshots. A separate
Android manager process computes percentiles, writes raw sample journals and
reports metrics. Heartbeats, bounded backlog and reporting-failure detection
remain. Schema2 release/completion metrics supplement raw cycle intervals and
legacy deadline counters; they do not replace them.

The same fixture hash is
`b5ea99327f0ffc121e5556d29b32d4ce8e6e855d14518f46a3f060ea38721f99`.
Both measurements use injected inputs on the physical Fold 3 and are separate
from qualified visual inference.

| Measurement | Candidate44 | Candidate45 |
| --- | ---: | ---: |
| Samples | 2705 | 2641 |
| Mean controls rate | 99.910 Hz | 97.438 Hz |
| Controls work p99 | 7.304 ms | 6.208 ms |
| Legacy complete-cycle p99 | 16.613 ms | 17.290 ms |
| Work deadline misses | 8 | 0 |
| Legacy cycle deadline misses | 1350 | 1380 |
| Raylib mean rate | 29.68 fps | 29.65 fps |

Work p99 improves **15.0%**; legacy complete-cycle p99 worsens **4.1%**.
Candidate45 additionally reports 67 scheduled completion misses, 69 skipped
slots and 11.045 ms p99 release-to-completion latency. Its scheduler delay p99
is 6.467 ms; reporting p99 is 0.600 ms. Thus the overall 100 Hz deadline gate
still fails. A separate same-APK run measured 5.977 ms work / 16.776 ms legacy
cycle p99 and also failed timing. No fastest-run selection clears the failure.

[Delivered-journal diagnostics](candidate45-controls-deadline-diagnostics.json)
retain the same nearest-rank percentile convention. Missed-release rows have
5.595 ms median scheduler delay and 0.063 ms median reporting time. This
correlation narrows the remaining investigation; it does not prove an OS cause.

[Raw journals](raw-controls-index.json) preserve every delivered reporting
batch. Coverage means reported batches were consumed; a force-stop can discard
an unflushed tail. Debug self-affinity requests were rejected at worker startup
and retain their failure reports. No affinity speedup is claimed.

## Validation

| Check | Result | Evidence / limit |
| --- | --- | --- |
| Focused host checks | PASSED | 137 tests; converter, scheduling, IPC, supervision, tracing, qualification and target provenance |
| Hyundai/Panda safety, simulation, protocol | PASSED | 1920 tests run, 200 skipped; physical Panda disconnected |
| Native build and Java unit/build checks | PASSED | Pinned patched MNN, OpenCL syntax and Gradle logs |
| Fold 3 instrumentation | PASSED | 28 tests on final APK |
| Fold 3 fixture/lifecycle/fault/IPC groups | PASSED | All six final validation groups; timing acceptance remains separate |
| Emulator instrumentation and deterministic fixtures | PASSED | Fresh SwiftShader retry: 28 instrumented tests, complete process chain, lifecycle, cancellation, stale inputs, IPC and thermal handling |
| Emulator actual-camera timing | FAILED | 279/280 timestamps in the future; invalid capture times remain rejected |
| Full accelerated model accuracy | FAILED | Six initial output-array failures; 780 recurrent/replay comparisons blocked |
| Controls complete-cycle timing | FAILED | Raw cycle latency and deadline counters retained |
| 30 s / 2 min / 30 min combined visual workloads | BLOCKED | Accuracy, freshness and timing prerequisites unmet |
| Independent rear sensors / mounted calibration | BLOCKED | Lens-cover trials deferred, phone unmounted |
| Firmware / passive CAN / harness / vehicle behavior | BLOCKED | Panda disconnected; no physical vehicle qualification |

Instrumentation and deterministic fixtures establish recovery behavior. They
cannot establish camera freshness, independent rear sensor identity, physical
fold transitions, full-model accuracy, accelerator speed or road readiness.
Standalone IPC validation uses a separately cross-compiled pinned-msgq kernel
binary; APK identity provides test context, not proof that binary is packaged.

Every experiment stops both app variants/test processes, checks camera release
and provider stability, removes newly created forwarding and records the actual
installed APK. Entry requires 30 continuous NORMAL thermal seconds with phone
skin at or below 32 C. MODERATE pause and critical abort remain enabled.
Final cleanup and any emulator renderer retry are recorded separately.

The final candidate45 emulator run passed all eight instrumentation/fixture
groups and the thermal test. Its real camera HAL advertised REALTIME timestamps
but delivered them about 30 ms ahead of elapsed realtime. The
[camera timing report](candidate45-emulator-camera-timing-summary.json) retains
that failure separately from deterministic injected inputs. No arrival-time
substitution clears it. Both installed APK hashes were verified again before
the emulator was stopped; the final phone and emulator cleanup records show
no app processes, owned cameras, thermal override or experiment forwarding.
