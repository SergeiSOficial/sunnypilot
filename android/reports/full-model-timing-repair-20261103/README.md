# Fold 3 full-model and controls repair, 2026-11-03

Work continues on `codex/fold3-android-port`. TX is compiled out and engagement remains disabled. Fold 3 is disconnected; no physical accelerator, camera, Panda, mounted-calibration or vehicle qualification was performed. Cooling optimization is deferred and thermal safeguards are unchanged.

## Changes

- Extend the proven ordered half Gemm implementation from `linear_100` to all 27 original FP16 driver heads. Identify heads by original ONNX type, attributes and exact serialized weight/bias bytes; validate the complete mapping before mutating the graph. Retain original FP16 rounding, two-product accumulation and bias order. Reject CPU execution, unsupported activation/quantization and changed provenance. Original models and tolerances (`rtol=1e-3`, `atol=1e-4`) are unchanged.
- Reuse the reference's explicit FP32 Tanh polynomial and division in diagnostic ncnn FP32 execution. Use the host's eight-lane accumulation and division for the pinned driver's 960-channel 8x12 spatial mean; retain generic reduction outside that verified configuration. These improve local agreement but did not clear full-model parity.
- Give the emulator controls worker the same public Android urgent-display thread priority already used on the Fold 3. This changes neither privileged scheduling nor other workers. Absolute slot pacing, nonblocking inputs, reporting, raw cycle metrics, deadline counters, cancellation, stale-input checks and watchdog behavior are preserved. The default 5 ms Python GIL quantum is retained; the earlier 1 ms experiment remains rejected.

All ncnn changes are reproducible through `android/patches/ncnn-explicit-casts.patch` against the pinned source. Diagnostic and converted graph hashes, provider hashes and actual APK identity remain separate from qualification. No backend was promoted.

## Artifacts

| Candidate | APK SHA-256 | Purpose |
| --- | --- | --- |
| candidate45zj | 84c907b4957de484bcd4e72f0bd45436d8964085f46d35d3b3a005efa5d3eb58 | Previous measured baseline |
| candidate45zk | 440e6feaa946f1f7541d1af9dc32cb19a6a80954941669c8499025f9e8b8fdaf | 27 half heads and controls priority comparison |
| candidate45zl | 0e035bbab912cc5607bde2a1a934a433d452cde8e46d6a69a6e09ae2ff79aa2a | Final candidate, adds Tanh/mean arithmetic |

Installable archives: `android/build/candidates/<candidate>/app-debug.apk`. `candidate45zl-package-check.json` binds the final APK, unchanged marked graph, packaged provider and native recipe/patch identities. Candidates35,40,44 and earlier failures remain intact. Candidate45zh remains excluded as documented in the previous report.

## Model checks

- **Passed:** all 81 isolated original FP16 Gemm comparisons: 27 heads × three seeds. Arithmetic was extracted from the actual shader and compiled with contraction disabled. This establishes local arithmetic correctness, not complete-model qualification.
- **Passed:** eight Tanh/mean host checks using captured GPU inputs and three seeds. All captured-input and seeded values matched the original reference exactly. Earlier GPU implementations passed local tolerance but differed by up to 2.98e-7 (Tanh) and 5.96e-8 (mean), which can affect later half rounding.
- **Passed:** final candidate45zl actual software-Vulkan Tanh and mean outputs match the original host exactly using the same captured GPU inputs (92160 and 960 elements). The separately hashed trace still reproduces all eight final mismatches; the first mapped violation remains `linear_100` (3/512), not a cleared full-model gate. See `rounding-activation-mean-gpu.json` and `rounding-heads-trace-seed42-comparison.json`.
- **Passed:** actual Tanh and reduction GLSL compilation in FP32, half-storage/float-arithmetic and half-storage/half-arithmetic modes. Native build applies the complete patch to fresh pinned source.
- **Passed:** 159 converted-graph external-I/O original-reference comparisons before packaging candidate45zk. Graph conversion is separate from accelerator execution.
- **Failed:** untapped complete driver, software Vulkan, seed42. Candidate45zj: 14/553 mismatches, max error 0.0046749115. Candidate45zk: 8/553, max error 0.00390625. Candidate45zl: 8/553, max error 0.00390625. Outputs are finite; CPU fallback is rejected. The Vulkan probes are separately hashed standalone executables using the matching native provider build; the installed APK hash records artifact identity, not an in-app inference run. Software-Vulkan timing is not accelerator-performance evidence. Tapped candidate45zk reproduces eight final mismatches and first mapped violation `linear_100` (3/512, max 0.0009765625); its input already differs from the host. Seeds7/123 and later performance stages were not advanced after the full-model failure.
- **Blocked:** all 795 original-reference accelerator comparisons, recurrent/replay qualification and physical performance. Isolated checks, conversion checks and software Vulkan do not substitute for this gate. Road-model accelerator qualification remains open.

## Matched controls fixture

`controls-comparison.json` records identical fixture SHA and the actual installed APKs, with host builds/tests paused for both 30-second runs. These are emulator scheduling results, not Fold 3 measurements.

| Metric | candidate45zj | candidate45zk |
| --- | ---: | ---: |
| Controls work p99 ms | 2.023625 | 1.303709 |
| Complete-cycle p99 ms, unchanged definition | 12.127792 | 11.513292 |
| Release-to-completion p99 ms | 4.133958 | 2.576799 |
| Scheduler delay p99 ms | 2.392294 | 1.217178 |
| Mean controls Hz | 100.0148 | 100.0023 |
| Scheduled deadline misses | 2 | 0 |
| Skipped slots | 0 | 0 |
| Source exchange p99 ms | 2.875333 | 1.788042 |

The candidate improved in this paired run; source timing also varied, so it does not isolate a causal speedup or establish sustained qualification. Complete-cycle p99 still exceeds 10 ms and all raw deadline counters remain reported. The Fold 3 already used this priority; no physical timing improvement is claimed. The final candidate45zl repeat of the same fixture averaged 100.0005 Hz, work p99 1.299625 ms, complete-cycle p99 12.094458 ms and release-to-completion p99 3.174674 ms, with zero scheduled misses/skipped slots. Its raw cycle-deadline counter is 1332; these intervals include pacing and are retained without redefining the metric. This is a synthetic-input functional fixture, not accelerated perception or combined camera/inference qualification.

## Host and emulator validation

- **Passed:** final host suite, 189 tests, using the pinned Xcode SDK. An initial rerun omitted SDKROOT and failed two native compile checks against an incompatible CommandLineTools SDK; that log is retained as `host-suite-sdk-mismatch.log`, followed by `host-suite-final.log` with all checks passing.
- **Passed:** Gradle JVM-test/build tasks and APK/instrumentation assembly. Up-to-date Gradle tests are cache reuse, not a fresh test execution.
- **Passed:** final candidate45zl installed hash; all 28 instrumentation tests; nine of ten emulator groups covering the full synthetic process chain, cancellation/stale inputs, lifecycle, ADB recovery, IPC, MetaDrive and thermal safeguards. Every group passed isolation and APK-stability checks.
- **Failed:** final native emulator camera timing reports future capture timestamps, preserving the previous failure. Deterministic injected-input successes do not clear native camera timing or physical rear-sensor identity.

Every completed experiment so far passed process/camera/forward cleanup and the established 30-second NORMAL cooldown; emulator skin reports 30.8°C. No experiment used a thermal override. Native emulator camera timestamp failures remain preserved; no arrival-time substitution or freshness relaxation was introduced.

Longer two-minute/30-minute combined workloads are blocked by full-model accuracy and physical-device availability. Sustained 20 Hz inference, 100 Hz controls, 30 fps UI, fresh synchronized physical inputs and no watchdog violations are not qualified.

Reproduction commands are in `reproduce.sh`. Runners refuse a second emulator, verify installed APK hashes, isolate experiments, record failure status, and shut the emulator down in cleanup. Use new report names for repeats; never overwrite earlier evidence.

Final state: all ten functional experiment isolation checks and the final software-Vulkan trace cleanup passed; no app/probe processes, camera ownership or forwards remain. The dedicated emulator exited successfully and the final ADB inventory is empty. Final installed APK hash remained unchanged throughout.
