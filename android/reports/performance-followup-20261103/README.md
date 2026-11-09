# Fold 3 GPU event diagnosis

Continue the work in this chat on `codex/fold3-android-port`. Candidate45zd remains the archived, accepted numerical baseline. Candidate45ze is now installed on the Fold 3 for full-model qualification; its app and test APK hashes match the immutable archive. Vehicle permissions remain unchanged. Cooling optimization remains deferred, with thermal safeguards enforced.

## Implementation

`android/scripts/build_mnn.py --profile-gpu` builds event-profiling libraries in `android-profile`, separately from the APK libraries in `android`. `android/scripts/profile_mnn.py` cross-compiles a shell-only probe, rejects CPU fallback, uses the full converted graph without intermediate tensor readbacks, and captures GPU events through the process's standard output. It never clears device logs. The parser rejects incomplete runs and changes in kernel count or order. All-zero inputs and three measured diagnostic runs cannot qualify model accuracy, recurrent behavior, performance p99 or live perception. MNN's profiler changes execution/recording behavior, so these costs cannot replace ordinary complete-inference timing.

Native libraries, model, probe, recipe and installed APK hashes are recorded. The experiment wrapper enforces stopped app/test processes, camera release, forwarding cleanup and 30 continuous NORMAL seconds at skin temperature <=32C before execution. The runner monitors thermal safety, kills only its own remaining probe process and removes all its remote files on exit.

## Reproduce

```sh
.venv/bin/python android/scripts/build_mnn.py --profile-gpu
.venv/bin/python -m unittest android.tests.test_mnn_profile android.tests.test_controls_scheduling -v
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --adb /Users/sergios/Library/Android/sdk/platform-tools/adb \
  --package com.samsung.sunnypilot.fold3 --report ISOLATION.json -- \
  .venv/bin/python android/scripts/profile_mnn.py \
  --serial RFCR81Y1TBJ --adb /Users/sergios/Library/Android/sdk/platform-tools/adb \
  --model dmonitoring_model --report PROFILE.json
```

Use a new output path per attempt. Repeat with `driving_supercombo` only after cleaning and re-establishing the thermal baseline. Vehicle TX remains compiled out and engagement disabled.

## Results

Both direct-output profiles passed event completeness, no-fallback and experiment-isolation checks. All three driver runs contain 579 events; all three road runs contain 736. Event sums match the native profiler totals. The installed candidate45zd APK stayed unchanged (`e7ae5461257aae2abef31e1631c4c79e7a32bc1513523f5d38db021ab9a2334e`). See `profile-summary.json` for source hashes and complete rankings.

| Diagnostic cost | Driver median | Road median |
|---|---:|---:|
| Summed kernel time | 1153.071 ms | 2172.862 ms |
| First image convolution | 211.288 ms | 244.726 ms |
| All UnaryOp0 kernels | 199.737 ms | 211.998 ms |

Road pointwise convolution geometries `768→256` and `256→768` at 8×16 together cost approximately 635 ms. Repeated occurrences of a geometry are aggregated, not treated as one kernel. This prioritizes ordered convolution arithmetic and unary operations. It does not establish a thermal bottleneck or an optimized inference result.

The initial path-error attempt and earlier logcat evidence are retained. In particular, `road-profile.json` had 736/736/702 events; its old parser's PASS is rejected for complete profiling. Direct output corrected the capture problem. No prior failure is overwritten.

The existing candidate45zd full-model acceptance remains 795/795, whereas ordinary inference p99 remains 1276.617 ms (driver) and 2261.544 ms (road). Controls work p99 is 2.449 ms, but complete-cycle p99 is 14.001 ms and still fails timing. These ordinary measurements, emulator failures, camera identity and physical vehicle blockers remain in `../full-model-controls-20261101/candidate45zd-delivery.md`. Candidate45ze accuracy passes, but performance qualification remains open; no new combined-workload qualification is claimed here.

## Guarded exact-product experiment

`conv_guarded_probe.cpp` and `conv_guarded_boundary_probe.cpp` compare a guarded convolution implementation directly against the archived strict kernel. Both device experiments pass 72/72 synthetic cases with zero bit differences, no unexpected non-finite outputs and verified cleanup. Cases cover three seeds, six geometries, both bias orders, batch/tail/dilation/ReLU behavior, FP32 operands requiring strict FMA, and FP16-representable operands. The boundary probe uses production `-cl-mad-enable -w` flags and samples the finite binary16 range, including subnormals.

For seed42, bias-last and half-representable operands, the boundary probe measures 230.976→40.702 ms (road stem), 201.799→39.695 ms (driver stem) and 55.632→8.391 ms (late pointwise). These are synthetic operator medians, not full-model speedups or p99 acceptance.

The reproducible MNN patch now uses a separate rounded multiply/add only when both operands are finite and exactly representable as binary16. Their product is exact in binary32. All other operands retain strict FMA. The production guard additionally excludes subnormal/non-finite accumulators to avoid relying on unadvertised FP32 denormal support. Only the regular vector convolution path changes; depthwise, scalar reference and FDOT2 reductions retain their prior arithmetic. Candidate45ze now passes all 795 original-reference device comparisons with no CPU fallback and no unexpected non-finite outputs; see `candidate45ze-accuracy.json`. Candidate45zd remains the preserved comparison baseline. Accuracy does not establish performance qualification. All 182 host tests pass.

Build the diagnostic APK with the commands in `../full-model-controls-20261101/candidate45zd-reproduce.md`, using `candidate45ze` for the fresh archive and evidence paths. The original models and converted graph semantics stay unchanged. Complete all 795 device comparisons again before ordinary accelerator timing. Operator success cannot qualify the new APK.

## Candidate45ze artifact

The APK SHA-256 is `54a613533c10795ab754e2737d5bf2facfd387dfb2bd58ef43017260dd063a3f`. `candidate45ze-installed-fold.json` verifies both installed packages. Native and Gradle builds passed, as did 182 host checks and all 795 host conversion comparisons. Host conversion does not qualify device execution. The device accuracy run writes immutable per-part evidence under `../full-model-controls-20261101/fold45ze-mnn-*`; Timing is recorded separately: driver p99 is 1058.942 ms over 100 samples (all miss 50 ms), compared with candidate45zd 1276.617 ms. Road timing was stopped at MODERATE by the existing safeguard, so it has no completed p99; the last-call stages are not a completed benchmark. Both timing experiments passed process/artifact isolation. Functional inventories are updated only when their actual experiments run. TX remains compiled out.

Focused Hyundai safety and simulator-protocol host checks also pass: 1921 tests run, 202 skipped (`candidate45ze-hyundai-safety-sim.log`). These do not establish physical Panda or vehicle compatibility.

## Optimized profile and controls rerun

`candidate45ze-profile-summary.json` retains both new diagnostic profiles and their hashes. All three driver runs and all three road runs passed event coverage and total checks. Driver stem convolution median is 94.146 ms versus 211.288 ms in candidate45zd. However, road pointwise geometries remain 322.544 ms (768→256) and 317.767 ms (256→768), versus 320.744 and 314.450 ms. Unary aggregates remain substantial (driver 186.828 ms, road 197.395 ms). These all-zero, three-run event profiles identify remaining costs; they cannot supply ordinary p99 or qualify recurrent performance.

The new matched controls fixture passes functional behavior at 100.003 Hz with UI 29.693 fps. Controls work p99 is 2.668698 ms, complete-cycle p99 13.549531 ms, with 1214 legacy cycle misses, zero work misses, zero scheduled misses and zero skipped slots over 2661 reported samples. Release-to-completion p99 is 5.914315 ms and reporting p99 0.356927 ms. The schema-2 metrics supplement the unchanged legacy counters; complete-cycle timing still fails. Candidate45zd measured 2.448750/14.000989 ms for work/complete-cycle, so this repeat does not establish a controls root-cause fix.

All six Fold 3 functional inventory groups pass, including 28 instrumentation tests, lifecycle, full-chain steady/fault fixtures, ADB recovery and IPC. `candidate45ze-final-fold-cleanup.json` records stopped app processes, released camera ownership, no forwards and the final cool NORMAL baseline. Emulator APK and test hashes also match the archive.

The emulator retry passes nine of ten groups with host OpenGL and guest Vulkan disabled. The earlier SwiftShader/Vulkan attempt stalled in Raylib `end_drawing`; its failures and interrupted inventory remain preserved. Native camera timing still fails: all 289 delivered frames report capture timestamps approximately 30 ms in the future despite declaring REALTIME. Capture times remain rejected, without arrival-time substitution. Final emulator cleanup verifies released cameras, no forwards and the NORMAL baseline; the emulator was then shut down. See [delivery results](candidate45ze-delivery.md) and [reproducible commands](candidate45ze-reproduce.md).
