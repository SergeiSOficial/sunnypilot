# Fold 3 performance measurements

Latest local artifact: [candidate58 road and camera repair report](reports/road-camera-repair-20261107/README.md). Both devices pass 30 instrumentation tests on the same verified APK. Driver accuracy remains 159/159 exact; road accuracy fails, so accelerator performance qualification is blocked. Public ultrawide acquisition p99 remains 84–85 ms, already over the 50 ms maximum sensor-to-model-output budget. Fold controls work p99 is 2.131 ms, but complete-cycle p99 is 12.334 ms and still fails. TX and physical engagement remain disabled. No cooling optimization was applied.


The earlier [candidate45ze report](reports/performance-followup-20261103/candidate45ze-delivery.md) records all 795 original-reference model comparisons passing, six passing Fold functional/recovery groups and nine of ten passing emulator groups. Driver inference p99 improved from 1276.617 to 1058.942 ms but still fails the 50 ms target; road timing stopped at the MODERATE thermal safeguard. Controls work p99 is 2.669 ms; complete-cycle p99 remains 13.550 ms and fails timing. Native emulator camera timestamps remain invalid. Qualification and vehicle TX remain disabled. The [candidate45zd baseline](reports/full-model-controls-20261101/candidate45zd-delivery.md) is preserved.

**Freshness and sustained deadlines remain unqualified.** The earlier
[candidate45zc delivery](reports/full-model-controls-20261101/candidate45zc-delivery.md)
records complete MNN numerical acceptance and separate performance results under
its actual APK identity. Earlier candidate measurements below retain their original failures.
Vehicle TX and physical engagement remain disabled.

Cooling and temperature optimization are deferred at the user's request.
Existing Android thermal protection remains active during correctness and timing
experiments; thermal observations below are retained evidence.

## Candidate45 full-model and controls recovery

APK SHA-256:
`9868e5b9ca7d24feaefd9acfbcdabaf4f5226e85063e03fffec4400aecc1a859`.
See the [report](reports/full-model-controls-20261101/README.md) and
[matched comparison](reports/full-model-controls-20261101/candidate45-performance-comparison.json).
Targeted MNN Cast/GEMV/GatherND/Tanh repairs execute without CPU fallback.
The repaired Tanh agrees exactly on captured inputs, but full-model output
accuracy still fails at original tolerances. All 795 host graph-conversion
checks pass; that does not qualify Android MNN execution. Phone accelerator
timing and combined visual workloads remain blocked.

The matched injected-input controls run measures work p99 **6.208 ms** versus
candidate44 **7.304 ms** (15.0% lower), but complete-cycle p99 **17.290 ms**
versus **16.613 ms** (4.1% higher). Candidate45 averages 97.438 Hz, with zero
work misses, 1380 legacy cycle misses, 67 scheduled completion misses and 69
skipped slots. Raylib averages 29.65 fps. Absolute-slot pacing and nonblocking
input handling preserve raw intervals and old counters. New schema2
release-to-completion p99 is 11.045 ms; it cannot replace the failed legacy
measurement. Percentiles now run in the separate manager process; reporting
p99 is 0.600 ms and scheduler delay p99 remains 6.467 ms. Timing is **failed**.
The retained raw journal covers delivered batches, with an unflushed-tail
limitation at force-stop. Earlier candidate45 experiments and regressions
remain in the report's experiment index.

## Candidate44 NPU and alternative GPU diagnostics

APK SHA-256:
`f18262c93c2132b1e61a1746af6b2c55dcae8820e3d8a9db0dd84bc5472c6360`.
The [NPU recovery report](reports/npu-recovery-20261101/README.md) retains QNN
native validation logs, assignment records, separately hashed graphs and
full-model MNN/ncnn results. Quantized HTP execution works, but FP16/a16w16
operator support and a16w8 accuracy fail the prerequisite gates. All eight
small MNN OpenCL/ncnn Vulkan probes pass without CPU computation. Full models
remain unqualified: MNN selects CPU operations; ncnn driver outputs fail all
three seeds and its road conversion leaves unsupported operations/ranks.

No full-model latency improvement is inferred from small probes. Full-model
timing and combined 30-second, two-minute and 30-minute accelerator workloads
remain blocked by accuracy and operator support. The unchanged accurate
OpenCL implementations below remain the measured baselines. A host diagnostic
which rounds FP32 results to FP16 first diverges at Gemm; it is not packaged
or promoted. Original goldens and tolerances are unchanged.

Candidate44's same-hash device-source fixture measures controls work p99
7.30 ms and complete-cycle p99 16.61 ms with 1,350 cycle misses, 99.91 Hz mean
controls and 29.68 fps Raylib. Complete-cycle timing still fails. These separate
cooled injected-input runs cannot establish an inference speedup or qualified
visual-stack operation.

## Historical candidate40 measurements

The final bench APK is
`bd42b65a3538b8ad73834e90826eb10a98d8afeacbe9d5aae5877d43a25e30ec`.
The [delivery report](reports/performance-fixes-20261101/README.md) and
[comparison](reports/performance-fixes-20261101/candidate40-performance-comparison.json)
retain workload, sample counts, APK identity and earlier failures.

All **795 original-reference comparisons pass exactly**. Ordered convolution
workgroups now use up to 128 independent output lanes while preserving each
accumulation and rounding boundary. Asynchronous GPU event profiling identified
convolution as the largest cost. Candidate39's larger workgroups regressed and
were reverted. Short kernel profiles remain separate from full-model p99.

Driver-model p50/p99 is **319.90/381.54 ms** over 100 timed samples, versus
candidate35's **522.93/560.16 ms**. The p99 reduction is approximately **32%**;
all 100 samples still miss 50 ms. The road model completes all 636 numerical
comparisons, then stops at MODERATE thermal status before completing timing.
No completed road p99 is inferred from partial work or five-sample profiles.

QNN diagnostic graph changes eliminate the observed CPU partitions, but original
output accuracy still fails. CPU partition removal is not qualification. The
first failing tensors, provider assignments, raw profiles and full-output
checks remain in the report directory. HTP remains deferred.

Controls use the remaining 10 ms deadline budget for polling and wait only its
unused remainder when inputs return early. Candidate38's 114 Hz overspeed is
retained as a failed experiment. The final device-source fixture runs controls
at **99.83 Hz**, with work p99 **8.83 ms** and complete-cycle p99 **18.29 ms**,
versus candidate35's **23.49 ms** complete-cycle p99. Its **1,368 cycle misses**
remain failures. Scheduler delay p99 is **6.87 ms** and input handling p99
**11.54 ms**; stage percentiles are not additive. IPC age, stage work and every
deadline miss remain visible. Camera-only profile comparisons do not establish
combined inference/UI freshness or sensor identity.

The final 30-second camera/UI/model-diagnostic workload measures preview
delivery p99 **119.37 ms**, against candidate35's **160.99 ms** in a separate
cooled run. It still fails the **100 ms** limit and records zero fresh sampled
road outputs; Raylib averages **29.75 fps**. Readback p99 is **13.07 ms** and
texture acquisition p99 **114.43 ms**, so readback alone cannot resolve freshness.
This workload uses unqualified diagnostic CPU models, not qualified OpenCL
perception. Sampled app-process PSS reaches **1,328,863 KiB** and skin **38.9°C**
at 32.3 seconds, including startup. This does not establish stable memory or
thermal behavior.

Two-minute and 30-minute combined runs require the short input and timing gates.

## Candidate35 measurements

APK SHA-256: `38645e91f806076517de1497cdb6b5c0cbf4873cfecf5efabb980297a014ad51`.
All **795 original-reference comparisons pass exactly**, including three seeds,
recurrence and the pinned 60-frame replay. Models, goldens and tolerances remain
unchanged. The [comparison](reports/blocker-fixes/candidate35-performance-comparison.json)
records each actual APK; these are separate cooled runs, not a controlled A/B.

| Full-model work | Road p50 / p99 | Driver p50 / p99 | Decision |
| --- | --- | --- | --- |
| Candidate32, ordered FMA | 850.81 / 1,021.53 ms | 496.05 / 559.41 ms | Exact, misses 20 Hz |
| Candidate33, four-channel FP32 tile | 2,063.08 / 2,109.00 ms | 1,140.50 / 1,203.29 ms | Exact; tiling reverted after regression |
| Candidate35, ordered FMA | 845.98 / 881.57 ms | 522.93 / 560.16 ms | Exact, misses all 100 timed 50 ms deadlines |
| Candidate35, diagnostic native MAD | 134.39 / 138.39 ms | 89.60 / 102.51 ms | Rejected: 216/636 road and 137/159 driver comparisons fail |

Native MAD cannot write live qualification. The accurate pinned OpenCL backend
remains too slow; transformed CPU/QNN alternatives retain their separate failed
numerical gates. Global FP32 and reduced-precision variants remain diagnostic.

Camera GL acquisition and publication now run on separate bounded workers with
three owned NV12 buffers. Sorting and report-file writes run outside the control
loop. The [on-device 100 Hz source fixture](reports/blocker-fixes/fold-candidate35-chain-device-steady.json)
produces every required derived output with zero unsafe outputs or worker
failures. Controls work p99 is **9.01 ms**, but complete-cycle p99 is **23.49 ms**
with **1,303 cycle misses**, so complete 100 Hz timing fails. Input handling p99
is 19.23 ms and reporting p99 is 1.34 ms; these costs remain visible. Mean
controls rate is 98.79 Hz; source rate is 100.00 Hz. Raylib is **29.72 fps** in
this synthetic-source fixture, which does not qualify sustained visual driving.

The four [camera-only continuous-readback diagnostics](reports/blocker-fixes/candidate35-camera-comparison.json)
retain preview delivery p99 **105.66–108.83 ms** against the 100 ms limit.
Their workload differs from the historical combined camera/model/UI run;
those numbers cannot establish an isolated optimization gain.
The [30-second combined run](reports/blocker-fixes/fold-candidate35-vision.json)
has composite/preview/front delivery p99 **136.49/160.99/107.97 ms**, with zero
camera recoveries or timestamp errors. It rejects all road inputs and computes
432 driver predictions: 430 late, two published with invalid status. Driver
work p99 is 57.87 ms on the unqualified CPU diagnostic backend. The standalone
strict OpenCL accuracy pass does not qualify that CPU backend. Raylib averages
29.50 fps. Timestamp policy and workload differ from older camera experiments;
no controlled camera latency improvement is claimed.

The [resource samples](reports/blocker-fixes/fold-candidate35-vision.jsonl)
record total app-process PSS of 1,223,296 KiB (1.17 GiB) at 30.6 seconds and skin
temperature 36.1°C, with normal Android thermal status and no override. This
short startup/load sample does not establish stable memory or thermal behavior.
The [validation report](reports/blocker-fixes/candidate35-validation.md) records
combined-load, emulator, thermal and cleanup evidence. The 30-minute acceptance
workload remains blocked by short-run correctness and timing gates.

## Historical candidate32 measurements

Candidate32 has APK SHA-256
`a62b55187b2a60a15c9b7d34cb5e6e462cc00c353971d8d126940960861a2810`.
The [complete original-model gate](reports/blocker-fixes/candidate32-model-accuracy-gate.json)
passes all 795 output comparisons exactly, including every recurrent and
camera-replay frame. Original models, goldens and `rtol=1e-3, atol=1e-4` remain
unchanged. Both models still miss all 100 timed 50 ms deadlines.

| Cooled Fold 3 run | Road p50 / p99 | Driver p50 / p99 | Numerical result |
| --- | --- | --- | --- |
| Candidate29 (`04cf8905…e6b39`) | 942.72 / 1,121.55 ms | 548.19 / 605.06 ms | Road recurrence fails; driver passes |
| Candidate31 (`fb108753…302`) | 941.23 / 1,133.89 ms | 497.37 / 560.54 ms | Both exact |
| Candidate32 (`a62b5518…2810`) | 850.81 / 1,021.53 ms | 496.05 / 559.41 ms | Both exact |

These are separate runs, not a controlled A/B benchmark. Candidate32 removes
software FMA only from products of two half inputs, whose products are exact
in FP32, preserving the two rounding boundaries. A bounded device arithmetic
probe and complete model gates verify the change. Serialized checkpoint cost
for road `host_half_dot2` falls from **80.19 to 4.22 ms**; FP32 convolution
still costs **658.86 ms road / 362.30 ms driver** in those diagnostics.
Checkpoint serialization perturbs timing and does not establish model
throughput. See the [arithmetic evidence](reports/blocker-fixes/host-numerical-diagnostics.md)
and [recorded commands](reports/blocker-fixes/candidate32-reproduce.md).

## Historical candidate32 runtime fixtures

Both devices pass sources-only fault and steady process-chain behavior, with
all required derived outputs and zero worker failures/restarts. These tests
inject camera/model/sensor/CAN sources and do not qualify the visual stack.

| Steady 30-second fixture | Mean controls rate | Work p99 | Complete-cycle p99 | Work / cycle misses | Samples |
| --- | --- | --- | --- | --- | --- |
| Fold 3 | 100.87 Hz | 9.683 ms | 24.365 ms | 24 / 1,213 | 2,701 |
| Emulator | 99.73 Hz | 2.111 ms | 16.380 ms | 0 / 1,353 | 2,763 |

The phone test runs with the emulator stopped. Its input handling, computation,
publishing, parameter/extensions and reporting p99 is
**19.182/4.637/5.298/4.223/2.471 ms**. Stage percentiles do not sum to the cycle
percentile. Source rates are **99.996/99.995 Hz** on phone/emulator; ADB exchange
p99 is **15.328/5.871 ms**. Host transport jitter remains part of this fixture.
Work timing excludes input wait; complete-cycle timing includes it. No deadline
misses are suppressed. These results fail complete sustained timing acceptance.
See the [phone](reports/blocker-fixes/fold-full-chain-steady-candidate32.json)
and [emulator](reports/blocker-fixes/emulator-full-chain-steady-candidate32.json)
records for full stage distributions and source timing.

## Historical candidate32 combined camera workload

The [30-second camera run](reports/blocker-fixes/fold-vision-candidate32.json)
has zero recoveries, timestamp/packing errors or dropped PBO readbacks.
Composite/preview/front delivery-age p99 is **98.86/124.60/106.47 ms**;
acquisition-age p99 is **98.52/119.80/105.37 ms**. Preview render/readback/submit
p99 is **3.53/11.40/7.13 ms**. Late inputs and outputs remain rejected.

The unqualified diagnostic CPU road/driver workers compute **58/411 frames**.
Inference-only p99 is **62.75/37.38 ms**, complete work p99 **67.89/40.12 ms**,
and output-age p99 **163.29/134.73 ms**. All road and 410 driver outputs are
late; the remaining driver message is invalid. This CPU diagnostic is distinct
from the accurate OpenCL model gate above and does not establish model parity.
Raylib averages **29.51 fps**, with frame p99 **33.74 ms**.

Sampled process PSS grows **402,054→1,316,202 KiB**, including startup.
Sampled skin temperature reaches **37.0°C** and immediate cleanup **37.3°C**;
sampled Android thermal status stays NORMAL, without an override. These short
measurements cannot establish memory stability or sustained thermal behavior.
The 30-minute workload remains **blocked** by freshness and timing failures.
The [comparison](reports/blocker-fixes/candidate32-performance-comparison.json)
retains each earlier APK, scope and failure.

[Final cleanup](reports/blocker-fixes/candidate32-final-cleanup.json) passes:
the phone is **30.0°C after 32.33 continuous NORMAL seconds**, all app/test
processes stopped, cameras released and ADB forwards cleared. The emulator is
stopped. Every experiment's [baseline audit](reports/blocker-fixes/candidate32-isolation-audit.json)
passes the 30-second NORMAL/32°C cooldown requirement.

## Historical candidate21 measurements

The historical [blocker-fix measurements](reports/blocker-fixes/README.md) and
[machine-readable comparison](reports/blocker-fixes/candidate21-performance-comparison.json)
identify every result's APK. Candidate21 (`4c48351c…09bbfe`) was installed
and hash-verified on both devices. Original model files, references and
`rtol=1e-3, atol=1e-4` remain unchanged. Earlier sections below retain their
original APKs and failures.

Candidate21's OpenCL host-fusion diagnostic completes qualification, but both
original models fail all seeds, recurrent sequences and camera replay. DM/road
work p99 is **75.40/105.58 ms**, with **39/100 and 100/100** 50 ms misses.
All outputs are finite. No live inference backend is accepted.

In the 30-second combined run, composite/preview/front delivery-age p99 is
**98.49/124.68/104.60 ms**. GPU readback/render/submit p99 is
**11.08/2.78/6.85 ms**, with zero dropped readbacks, timestamp errors or recoveries.
Diagnostic ORT CPU road/driver inference-only p99 is **55.86/41.57 ms**;
complete work p99 **67.44/47.07 ms** and output-age p99 **156.14/139.77 ms**.
All 54 road outputs and 442 of 443 driver outputs are late; the remaining driver
message is invalid. UI averages **29.55 fps**. This fails perception acceptance.

The phone's steady full-chain fixture averages **100.45 Hz**. Work p99 is
**9.542 ms**, with **20 misses in 2,688 samples**; complete-cycle p99 is
**23.992 ms**, with **1,218 misses**. Input handling/compute/publish/extensions/report
p99 is **19.203/4.985/5.479/4.302/3.425 ms**. Source cadence is 99.995 Hz,
but ADB exchange p99 is **16.141 ms**, so host transport jitter remains in these
measurements. The emulator's work p99 is **1.905 ms** with zero work misses;
its complete-cycle p99 is **16.441 ms**, with 1,353 misses. Both are synthetic
source tests; neither establishes complete sustained driving deadlines.

Sampled phone PSS grows **250,381→1,307,012 KiB**, including startup. Sampled skin
reaches **36.5°C**, immediate cleanup 36.8°C/NORMAL. Final cleanup verifies
**29.0°C after more than 30 continuous NORMAL seconds**, cameras released,
all app/test processes stopped, no ADB forwards and emulator stopped. No new
30-minute run proceeds while correctness and freshness fail.

## Historical candidate10 measurements

Candidate10 uses a different model profile and native runtime build. The values
below are retained observations, not a controlled speed comparison with candidate21.

On the Fold 3, original OpenCL DM/driving work p99 is **111.57/88.41 ms**;
all three reference seeds fail parity and both models miss 100/100 timed
50 ms deadlines. CPU DM alone reaches **39.47 ms** with no timing misses,
but CPU driving takes **209.92 ms** and both fail parity. No live inference
backend is accepted. The FP16 grouped-convolution repair removes non-finite
CPU driving outputs without establishing original-reference agreement.

The host fixture now uses absolute 100 Hz deadlines. With steady inputs and
the emulator stopped, source rate is 99.996 Hz and controls average **98.97 Hz**.
Work p99 is **9.245 ms**, complete-cycle p99 **24.527 ms**, with 18 work and
1,267 complete-cycle misses in 2,702 samples. Reports separate input handling,
computation, publishing, extensions and reporting. Source/ADB exchange p99 is
16.631 ms, so these measurements include host transport jitter. Separate
cancellation/stale-CAN runs pass; the steady run explicitly omits those faults.
Invalid upstream inputs and cycle jitter prevent sustained 100 Hz qualification.

The 30-second combined diagnostic has thumbnail/preview/front age p99 of
**126.12/144.84/102.11 ms**. GPU readback p99 is **10.65 ms**, with zero readback
drops; bounded staging/PBO pools and newest-pair selection remain enabled.
All 53 road and 469 driver predictions are late and withheld. Work p99 is
**247.39/73.61 ms**, with output age p99 **332.65/163.82 ms**. UI averages
**29.68 fps**, a short rendering result only.

Sampled PSS rises from 355,245 to 1,314,341 KiB including startup. Sampled skin
rises from 28.6°C to 36.6°C; immediate cleanup reaches 36.8°C at NORMAL thermal
status. All processes and camera clients exit, forwards clear, and the phone
cools to 29.6°C with more than 30 continuous NORMAL seconds. The dedicated
emulator stops. A new 30-minute workload remains blocked by identity, parity
and freshness failures; the earlier severe-thermal failure remains recorded.

## Previous rear-pipeline measurements

The previous [rear-pipeline bench results](reports/rear-pipeline/README.md) cover the
continuous rear GL/NV12 and thumbnail transport, original live road-model
failure, thermal/numerical work suppression and a new FP32 CPU diagnostic.
FP32 host-backend parity passes three cases on both devices, with Fold 3
isolated p99 of 38.01 ms for DM and 46.98 ms for driving. Equivalence to the
original goldens fails (maximum differences 0.00944 and 0.29613), so the overall
model gate still fails and FP32 is not adopted. Separate isolated model timings
do not establish concurrent sustained 20 Hz. The results below retain their
original APK identities. Three active pipelines alone do not establish
timely, calibrated perception.
No precision change is accepted. All live model messages remain invalid and
vehicle transmission stays disabled. See [validation scope](VALIDATION.md) for
which APK produced each result.

The earlier `f725600b` APK completed 30-minute triple-stream capture without
recovery, at approximately 30 fps and 29.94 fps UI. This was not sustained
inference: the road worker latched its first non-finite output, DM discarded
all 3,313 computed outputs as late (work p99 64.15 ms, 851 deadline misses),
and thermal pause persisted from 4:09 through completion. Sampled skin reached
44.6°C with SEVERE thermal status; post-startup PSS was 1,346.6–1,409.8 MiB.
The [historical soak summary](reports/rear-pipeline/fold3-final-soak-summary.json)
records the distinct workload and clean exits. Its additional rear pipeline
and retained road model prevent a direct memory/energy comparison with the
previous regular APK.

## Preprocessing on the previous regular APK

The 15-test instrumentation suite compares exact padded NV12/channel packing,
random row maps, malformed inputs and the real 1280x720 to 1440x960 driver input.
Timing uses three warmups and 100 alternating measurements, outside live capture.

| Fold 3 method | p50 | p99 | Index storage |
|---|---:|---:|---:|
| Reusable NumPy gather | 2.967 ms | 2.984 ms | 11,059,200 bytes for flat intp indices |
| Native row gather | 0.784 ms | 0.825 ms | 19,200 bytes for row/column offsets |

Outputs match exactly. The native path avoids per-pixel index storage, releases
the GIL and copies repeated rows. This is a preprocessing microbenchmark, not
an end-to-end speedup claim. [Phone report](reports/fold3-final-preprocess.json),
[emulator report](reports/emulator-final-preprocess.json).

## Pinned portable inference runtime

ORT 1.31.1 is compiled from commit
`6af1a932e2286d5c58166085d24c68783033edee` using the pinned Android NDK,
without KleidiAI/SME/SVE. The source-build manifest and hash are checked by
CMake. The supplied AAR's SME `RDSVL` traps in this API 35 emulator despite
advertised host features; the portable build completes both models without
that crash. It does not solve model accuracy or driving throughput.

These individual profiles were freshly tested on the preceding `31c9028b…`
APK. Model benchmark APIs and profiles are unchanged in the final APK. Each
row has three warmups, 100 paced timed calls and three original synthetic
parity cases (seeds 42, 7, 123, including nonzero recurrent features). Original
model hashes, precision and tolerances (`rtol=1e-3`, `atol=1e-4`) are retained.

| Device / profile | DM p99 | Driving p99 | Numerical result |
|---|---:|---:|---|
| Fold 3 original CPU, 4 threads | 39.06 ms | 191.57 ms | Failed; DM max abs 0.078125, driving non-finite |
| Fold 3 original CPU, 2 threads | 35.27 ms | 190.92 ms | Failed; DM max abs 0.078125, driving non-finite |
| Fold 3 portable fallback, 4 threads | 59.32 ms | 144.60 ms | Finite, failed; DM 0.017578125, driving 0.3125 |
| Emulator original CPU, 4 threads | 42.51 ms | 186.10 ms | Failed; DM differs, driving non-finite |
| Emulator portable fallback, 4 threads | 45.47 ms | 159.82 ms | Finite, failed original-reference parity |

Raw reports: [phone CPU 4](reports/four-thread-validation/fold3-final-model-ort-cpu-4.json),
[phone CPU 2](reports/four-thread-validation/fold3-final-model-ort-cpu.json),
[phone fallback](reports/four-thread-validation/fold3-final-model-ort-fallback.json),
[emulator CPU](reports/four-thread-validation/emulator-final-model-ort-cpu-4.json),
[emulator fallback](reports/four-thread-validation/emulator-final-model-ort-fallback.json).
Portable exports use ORT extended optimizations, avoiding `NhwcFusedConv`
operators unavailable in the portable build. The export itself already fails
host parity (DM max abs 0.00390625; driving 0.03125–0.09375 across three cases).
Its manifest records that failure. [Host export checks](reports/host-portable-export-parity.json).
It is diagnostic only and excluded from the live runtime. Rebaselining to the
fallback, increasing tolerances or accepting higher precision would hide this
failure; none was done. Synthetic checks also do not replace recorded replay.

Two original CPU threads used 4,388 ms process CPU versus 7,644 ms with four
threads for 100 paced DM calls, about 43% less in these isolated runs; wall times
were both about 5.09 seconds. Driving CPU work was 23.29 versus 32.71 seconds.
These are separate measured runs, not a controlled energy comparison. The final
live runtime uses two non-spinning threads, a 50 ms minimum start interval and
no catch-up bursts. The corrected battery current/voltage calculation avoids
integer overflow, but current units and net battery power are not qualified as
a measurement of whole-phone energy savings.

## Previous regular APK live and sustained diagnostics

The [60-second phone run](reports/fold3-final-live.json) passed capture and
preview with one session, no recovery, 1,800 rear / 1,794 front frames near
30 fps and 16.53 ms timestamp-skew p99. Rear/front delivery-age p99 was
82.71/92.72 ms; packing p99 6.59/5.49 ms. Driver-model work p99 was 58.57 ms;
976 predictions were computed, zero published, and all 976 were late.
There were 109 work durations >=50 ms. Stage p99 was 4.07 ms preparation,
55.26 ms inference and 148.29 ms sensor-to-output age. Raylib averaged 29.83
fps with CPU p99 7.02 ms. Near-30-fps rendering does not establish timely
perception. The initial landscape race is fixed in this run; the intermediate
[pre-power-fix result](reports/pre-power-fix/fold3-final-live.json) also captured
without recovery but failed DM freshness.

The 30-minute road/front capture and preview test **passed** with zero recoveries. It delivered 53,964 rear and 54,003 driver frames at 29.98/30.01 fps; both image-handler drains and camera-close callbacks completed. Final rolling nearest-skew p99 was 16.50 ms, and delivery-age p99 was 84.43/94.31 ms. The host observed 1,817.18 seconds including post-capture sampling.

**Sustained perception failed.** Android reached severe thermal status (3), with skin temperature sampled up to 44.6°C. DM computed 25,909 frames, all late, and published zero; rolling work p99 reached 82.57 ms with 19,786 cumulative 50 ms deadline misses. The UI averaged 29.96 fps. Memory after 60 seconds ranged from 916.1 to 947.4 MiB PSS. See the [soak summary](reports/fold3-final-soak-summary.json) and [raw samples](reports/fold3-final-soak.jsonl). This workload excludes road inference, localization, planning, active physical controls and Panda, so it cannot qualify the complete driving stack.

Camera percentiles use the last 4,096 frames, model work the last 10,000 calls,
model stages the last 4,096 computations and UI the last 3,000 frames.
Cumulative frame/deadline/rejection counts cover the run. A final rolling p99
is not a whole-run latency distribution. PSS and skin temperature are sampled
every 30 seconds; startup model loading is separate from the later memory
range. The raw JSONL preserves progression. Thermal override is prohibited on
the phone; native camera and host validation abort at critical status 4.

The [final emulator run](reports/emulator-final-live.json) delivered 812 rear /
829 front frames at 13.39/13.86 fps, UI about 11.97 fps, and failed capture
throughput. There were 618/623 future-delivery timestamps, rejected by the
unchanged freshness guards. Software cameras and SwiftShader do not reproduce
Samsung throughput, thermals or concurrency. After final thermal-guard checks,
the dedicated emulator was cleaned and powered off.

## Retained measurements and practical limits

Earlier foreground-service original OpenCL p99 was 82.84 ms DM and 55.69 ms
driving, both failing parity; max abs errors were 0.048828 and 2.90625. Earlier
NNAPI DM timing passed in isolation but parity failed; driving could not run
on the exposed QTI devices. These remain failed backend experiments:
[OpenCL](reports/fold3-model-opencl-foreground.json),
[NNAPI](reports/fold3-model-ort-nnapi-foreground.json). No reduced-precision or
promoted-FP32 experiment is adopted.

An earlier combined workload reached critical thermal status at 1,247.94
seconds (~20.8 minutes); its incomplete camera report cannot certify the
requested duration. [Earlier summary](reports/fold3-live-vision-30min-summary.json).
Earlier camera-only soaks failed after 180, 437 and 663 seconds. Their retained
failures are not overwritten by shorter successful diagnostics.

Earlier isolated/simulator controls and UI work is recorded in
[controls](reports/fold3-controls-work-final.json) and
[UI](reports/fold3-ui-work-final.json). MetaDrive uses synthetic perception,
calibration and a damped torque-to-angle plant. Those component measurements
exclude real road inference/localization/planning and cannot qualify complete
100 Hz controls or watchdog behavior under a physical driving workload.

### Candidate45zl: disconnected-device continuation, 2026-11-03

[Full report and reproduction](reports/full-model-timing-repair-20261103/README.md).
The matched emulator fixture improved from candidate45zj to candidate45zk: controls work
p99 2.023625→1.303709 ms, release-to-completion p99 4.133958→2.576799 ms,
and scheduled misses 2→0. Source timing also varied, so this is one paired
observation, not an isolated causal or physical-device speedup. The final candidate45zl
repeat averaged 100.0005 Hz, work p99 1.299625 ms, complete-cycle p99 12.094458 ms,
zero scheduled misses/skipped slots, and 1332 raw cycle-deadline counts.
Complete-cycle timing remains unresolved; its definition/counters are unchanged.

All 27 original FP16 driver heads now use the verified ordered accumulation;
81 isolated host comparisons pass. Untapped software-Vulkan driver seed42
improves from 14/553 to 8/553 mismatches but still fails original tolerance.
The diagnostic Tanh/mean rounding repair does not clear that failure.
No backend promotion, sustained accelerator timing or longer combined workload
is qualified. Fold 3 is disconnected; no cooling optimization or TX enablement.

### Candidate65: physical GPU/NPU recovery, 2026-11-07

[Artifact-bound results, comparisons and commands](reports/gpu-npu-recovery-20261107/README.md).
The connected Fold 3 now passes all 795 original-reference comparisons on ncnn
Vulkan without CPU fallback. Output batching, reusable buffers, GPU timestamp
profiling and arithmetic-preserving convolution patches are implemented. The
final candidate remains diagnostic: road inference p50/p99 is 303.183/430.381 ms
and driver inference is 187.602/198.759 ms, each missing 50 ms on all 100 samples.
The road median improves 4.97% versus candidate62, but driver performance regresses;
the change is not accepted as an overall performance improvement. Earlier accurate
candidates and failed trials remain available. HTP probes pass 8 cases, fail
accuracy in 6 and block 6; no NPU model is promoted.

The matched Fold 3 controls fixture averages 99.994 Hz with work p99 2.282 ms,
zero scheduled completion misses/skipped slots, and complete-cycle p99 12.266 ms
with 1155 raw cycle misses. The complete-cycle gate remains failed. These fixtures
do not qualify concurrent perception operation. Public ultrawide direct YUV
delivery p99 is 83.615 ms; the three-stream preview sink reaches 116.886 ms before
inference. Independent rear identity remains unproven. Longer combined workloads
are blocked by these prerequisite failures. TX stays compiled out; cooling
optimization is deferred and thermal protection is unchanged.


### Candidate66: weight-residual shortcut and renderer isolation, 2026-11-08

[Artifact-bound results and reproduction](reports/weight-residual-recovery-20261108/README.md).
The guarded zero-weight-tail Vulkan shortcut preserves all 795 original-reference
comparisons and passes 251 host tests, but is not accepted as an overall speedup.
Driver p50/p99 is 188.473/202.939 ms; road is 320.131/383.425 ms. All 100 samples
per model exceed 50 ms. Candidate65 and earlier accurate baselines are retained.
The dominant profiled road convolution still costs 43.430 ms; profiles do not
qualify performance.

The matched Fold controls fixture has work p99 2.256 ms, 99.996 Hz, no scheduled
misses/skipped slots, and complete-cycle p99 12.229 ms with 1227 raw misses.
Raw stage batches retain all counters and show work/wakeup variation; the legacy
gate remains failed. The same-APK emulator host-GPU repeat improves steady work
p99 from 33.153 to 4.100 ms and frequency from 77.676 to 99.732 Hz, but retains
scheduled and complete-cycle failures. MetaDrive passes on the host-GPU repeat;
the initial out-of-road failure is preserved. This is an environment comparison,
not a physical accelerator or controls code speedup.

Ultrawide delivery p99 is 87.333 ms; three-stream preview/front delivery is
111.789/100.982 ms. Freshness and independent rear identity remain unqualified.
Longer combined perception stages remain blocked. TX stays compiled out, cooling
optimization deferred, and thermal safeguards unchanged.

### Candidate67: pipeline weight specialization and absolute controls wait (2026-11-08)

[Artifact-bound report](reports/pipeline-specialization-20261108/README.md).
APK `6463d0d0603776f8ac7ea48f39618bc795b0292f1717a5d8e0459f29ad90991a`
passes all 795 original-reference comparisons without CPU fallback. A verified
pipeline specialization removes unnecessary weight-residual work, but does not
clear performance: driver p99 195.139 ms, road p99 472.634 ms, all 100 samples
per model over 50 ms. Road p99 regresses from candidate66; candidate67 remains
diagnostic and is not promoted as an overall improvement.

Native absolute controls pacing retains the original counters and single pacing
loop. Fold steady work/cycle p99 is 2.141/12.137 ms, cancellation/stale fixture
2.219/12.441 ms; both average approximately 100 Hz with zero scheduled misses or
skipped slots. The unchanged complete-cycle gate still fails. UI averages
29.71/29.61 fps. These are synthetic fixtures, not combined camera inference.

Matched ultrawide direct YUV delivery p99 is 83.265 ms versus PRIVATE/GLES
89.323 ms. Three-stream rear-preview/front delivery p99 is 110.632/105.025 ms.
The 50 ms sensor-to-output requirement and rear identity remain blockers.
TX and engagement remain disabled; thermal safeguards stay active and cooling
optimization remains deferred.

### Candidate68: bounded Vulkan dispatch batching (2026-11-08)

[Artifact-bound report](reports/dispatch-batching-20261108/README.md).
APK `925b4b0ab7b40e50c6b19b3ca8f8f4fba2ef5eefd74fcc8b5d0588e2dff8a048`.

All 795 accuracy comparisons pass at unchanged tolerances without CPU fallback. Driver p99 improves from 195.139 to 154.106 ms; road p99 from 472.634 to 362.394 ms, but road median regresses from 303.182 to 328.765 ms. All 100 timing samples per model exceed 50 ms; no overall road promotion. Internal GPU submissions are now measured separately and remain included in complete latency.

Fold controls work/cycle p99 is 2.205/12.261 ms steady and 2.137/12.338 ms with faults, approximately 100 Hz, zero scheduled misses/skipped slots. Raw cycle timing remains failed. Camera manual/low-latency publication p99 is 61.741 ms; three-camera rear/front delivery is 102.725/98.047 ms. Freshness and independent rear identity remain blockers.

TX and engagement remain disabled; cooling optimization is deferred and thermal safeguards remain active. Prior candidates and failures are preserved.
