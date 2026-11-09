# Road-model precision repair, 7 November 2026

Branch: `codex/fold3-android-port`, HEAD `7631d5248cc9efd10b962107aadacbb908a9f14d` plus the preserved working tree. Candidates 35,40,44,54–59 and failed diagnostics remain available. Vehicle TX remains compiled out; physical engagement is disabled. Cooling optimization was not changed.

## Implemented repairs

The two selected approaches are implemented in the [pinned ncnn patch](../../patches/ncnn-explicit-casts.patch): targeted reference LayerNorm arithmetic and explicit preservation of FP16 rounding boundaries. Original models, goldens and tolerances (`rtol=1e-3`, `atol=1e-4`) are unchanged.

- Targeted LayerNorm uses the proven reference accumulation and intermediate rounding rules. Converter annotations are tied to original nodes, epsilon, axes, tensor shapes and dtypes. Other LayerNorm operations keep their existing behavior.
- The selected FP32 linear layers accumulate from zero with the reference FMA rule and add bias last. On captured inputs, bias-last reproduced 512/512 original values; bias-first reproduced only 122/512. Both packed output layouts are handled; unsupported execution is rejected instead of falling back to CPU.
- Attention MatMul now uses explicit integer round-to-nearest-even conversion at the demonstrated FP16 boundary. On identical captured inputs, the old Adreno `packHalf2x16` results matched truncation for all 648 QK values, while nearest-even matched all 648 original-reference values. The AV section also matched all 4,608 original values with nearest-even. This removed the remaining full-model seed mismatches.
- Conversion order preserves explicit half-output casts before selecting half dot-product repairs. Required native capabilities are checked before loading an annotated graph. Model, conversion, implementation, provider, APK and device identities continue to invalidate old qualification records.

The [same-input rounding evidence](../../build/c60-matmul-boundary.json), [FP32 bias evidence](../../build/c59-fp32-gemm-bias-boundary.json), and [mapped attention trace](../../build/road-matmul-rte-c60-seed42-comparison.json) are diagnostic evidence. Final acceptance uses the complete untapped APK graph, not exposed intermediate outputs. Matching negative-infinity attention masks are allowed only in intermediate diagnostic comparisons; final model outputs still require finite values.

## Candidate60 artifact

[Installable APK](../../build/candidates/candidate60-road-rounding/app-debug.apk), [instrumentation APK](../../build/candidates/candidate60-road-rounding/app-debug-androidTest.apk), [full hashes and source snapshot](../../build/candidates/candidate60-road-rounding/artifacts.json).

| Artifact | SHA256 |
|---|---|
| Application APK, 1,147,200,789 bytes | `87f78c3d77df0c93337777d3f5c0b948dc66a3e77029a50b4acc2bf5709cb6bc` |
| Instrumentation APK | `bcc20fc1b227cb03bde6cf0c6fbc153dce82eb718ffb50cf6fd6fad45b6e999c` |
| Packaged ncnn library | `9d25ebf2e9fe70c8d2d28efcd2bb82ea0b7121a6887ebda77e3ea1acce1fc864` |
| Selected road parameter graph | `40a6f4d8f866e59be26114c6fb85269ab463972947eb9a239261ea244feb318a` |
| Selected road weights | `79f56975377749b2d32c79a1f4500ee1cda17b18d7a36a30c939b37d7a6520aa` |
| Original road model | `65a08adc31d5c456219687d99b7bf5e44d61dae2d49ea67850e76105c7248cce` |
| Original driver model | `dee5a294e8afaacc9295ac5d100e00733ecac278e79264b96331e40a3ede1b04` |

Application and test APK hashes matched on the [Fold3](../../build/candidates/candidate60-road-rounding/fold-installed.json) and [Android 15 ARM64 emulator](../../build/candidates/candidate60-road-rounding/emulator-installed.json). The initial emulator installation failed for insufficient storage; its [failed receipt](../../build/candidates/candidate60-road-rounding/emulator-installation.json) is retained. Replacing the old emulator APK with `cmd package uninstall -k` preserved application data and allowed the [successful retry](../../build/candidates/candidate60-road-rounding/emulator-installation-retained-data.json). No device data wipe or security-setting change was used.

## Validation

The installed Fold3 APK passed **all 795 original-reference comparisons**: road 636/636 and driver 159/159. Coverage includes all three initial seeds, three 32-frame recurrent sequences and the pinned 60-frame replay, with no unexpected non-finite outputs or CPU fallback. The [acceptance summary](../../build/candidates/candidate60-road-rounding/model-acceptance-summary.json) verifies original-reference identity and records 663 road and 476 driver Vulkan assignments. Native untapped seed42/7/123 probes also passed independently; those earlier probes are bound to their separately hashed diagnostic libraries and are not presented as candidate60 APK results.

Host validation passed **244 tests**, including extracted-shader FMA/half-rounding checks, converter provenance and rejection paths, trace mapping, qualification invalidation, IPC, scheduling, supervision, Hyundai and Panda restrictions. The final converter-order change additionally passed its focused 13-test suite and reproduced the probed road graph byte for byte; original and converted graph validation each passed 636 comparisons. Gradle application/instrumentation builds and Java unit tests passed.

Focused edit checks and `git diff --check` passed. The repository-wide static quality gate **failed** with 111 major findings across the pre-existing uncommitted port; this is retained in `android/build/c60-quality-delta.xml` and is not reported as a clean repository-wide check.

The installed [Fold3 model report](../../build/candidates/candidate60-road-rounding/fold-models.json) retains separate accuracy and performance gates:

| Model | Original-reference comparisons | Inference p50 / p99 | 50ms misses | Qualification |
|---|---|---|---|---|
| Road | **PASSED**, 636/636 |319.175 / 355.129 ms | 100/100 | **FAILED** timing |
| Driver | **PASSED**, 159/159 |173.366 / 181.884 ms | 100/100 | **FAILED** timing |

These are isolated complete-inference measurements after all 795 comparisons passed, without trace checkpoints. They exclude camera acquisition and therefore cannot establish sensor-to-publication performance. The report retains individual samples and native upload/compute/readback fields. Those coarse native fields are not independent asynchronous per-kernel GPU timings; no kernel-level performance claim is made. Neither model was promoted for live perception.

Both devices passed all 30 instrumentation tests and lifecycle recovery. The emulator additionally passed its display-size/recreation fixture; the Fold3 test did not exercise a manual physical fold transition. These are functional bench results, not camera freshness or vehicle qualification.

The matched Fold3 controls fixture uses the same input hash as candidate58, `b5ea99327f0ffc121e5556d29b32d4ce8e6e855d14518f46a3f060ea38721f99`:

| Fold3 controls metric | Candidate58 | Candidate60 |
|---|---:|---:|
| Mean rate |99.996 Hz |99.995 Hz |
| Work p99 |2.131 ms |2.248 ms |
| Complete-cycle p99 |12.334 ms |12.387 ms |
| Legacy complete-cycle misses |1,208 |1,235 |
| Scheduled completion misses / skipped slots |0 /0 |0 /0 |

The functional fixture passed, but the unchanged complete-cycle timing gate **FAILED** again. This model-accuracy patch did not improve controls timing. Candidate60 release-to-completion p99 was 4.104 ms and UI mean was 29.699 fps; these metrics do not replace the failing complete-cycle gate. See the [Fold3 controls receipt](../../build/candidates/candidate60-road-rounding/fold-controls-steady.json).

Fold3 and emulator cancellation/stale-CAN fixtures passed with zero unsafe outputs and no worker failures. The emulator MetaDrive stale-input fixture passed with the simulated world started; no physical vehicle behavior was exercised. Initial emulator fixtures used SwiftShader: the functional tests passed, but steady controls timing **FAILED** badly (47.153 Hz, work p99 128.834 ms, complete-cycle p99 201.716 ms, 359 scheduled misses and 1,421 skipped slots). These [original emulator results](../../build/candidates/candidate60-road-rounding/emulator-controls-steady.json) are retained. A separate host-GPU campaign uses the supported `-gpu host` configuration, with actual Apple M4/Metal rendering recorded in its launch log; it does not overwrite the software-rendered evidence.

Host GPU rendering passed instrumentation 30/30 and lifecycle/display recovery, and improved emulator steady controls to 99.820 Hz with 4.511 ms work p99. Its complete-cycle p99 remained 14.780 ms, with five scheduled misses and five skipped slots, so timing still **FAILED**. UI mean was 14.931 fps. These are [separate emulator host-GPU results](../../build/candidates/candidate60-road-rounding/emulator-host/emulator-controls-steady.json), not Fold3 accelerator measurements or a cleared controls gate.

The host-GPU [cancellation/stale-CAN fixture](../../build/candidates/candidate60-road-rounding/emulator-host/emulator-controls-faults.json) and [MetaDrive stale-input fixture](../../build/candidates/candidate60-road-rounding/emulator-host/emulator-metadrive.json) also **PASSED**. Both emulator campaigns are retained separately.

Final installed application and instrumentation hashes matched on [Fold3](../../build/candidates/candidate60-road-rounding/fold-installed-final.json) and [emulator](../../build/candidates/candidate60-road-rounding/emulator-installed-final.json). Final cleanup passed on [Fold3](../../build/candidates/candidate60-road-rounding/fold-final-cleanup.json) and [emulator](../../build/candidates/candidate60-road-rounding/emulator-final-cleanup.json): no app/test processes, camera ownership or forwarding remained. Continuous NORMAL cooldown was 33.1 seconds at 26.8°C on Fold3 and 30.7 seconds on the emulator. The owned [emulator was shut down](../../build/candidates/candidate60-road-rounding/emulator-shutdown.json).

The [machine-readable validation summary](../../build/candidates/candidate60-road-rounding/validation-summary.json) binds every experiment to the actual APK hash and retains separate functional, numerical, performance and physical gates. All selected source files still matched the archived APK-build snapshot after validation.

## Reproduction

From the repository root:

```sh
.venv/bin/python android/scripts/build_ncnn.py
.venv/bin/python -c 'import json,subprocess; subprocess.run(json.load(open("android/build/c60-prepare-absolute-command.json")),check=True)'
cmp android/build/c60-accelerators/ncnn/driving_supercombo.ncnn.param android/build/candidates/candidate60-road-rounding/graphs/ncnn/driving_supercombo.ncnn.param
cmp android/build/c60-accelerators/ncnn/driving_supercombo.ncnn.bin android/build/candidates/candidate60-road-rounding/graphs/ncnn/driving_supercombo.ncnn.bin
cp android/build/candidates/candidate60-road-rounding/graphs/ncnn/driving_supercombo.ncnn.param android/app/build/model-reference/accelerators/ncnn/
cp android/build/candidates/candidate60-road-rounding/graphs/ncnn/driving_supercombo.ncnn.bin android/app/build/model-reference/accelerators/ncnn/
cp android/build/candidates/candidate60-road-rounding/model-accelerators-manifest.json android/app/build/model-reference/accelerators/manifest.json
env SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk .venv/bin/python -m unittest discover -s android/tests
env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home ANDROID_HOME=/Users/sergios/Library/Android/sdk FOLD3_BUILD_PYTHON=/Users/sergios/sunnypilot/.venv/bin/python .venv/bin/python -c 'import json,subprocess; subprocess.run(json.load(open("android/build/c60-gradle-command.json")),check=True)'
```

The graph command retains all explicit selected-node arguments. Its generated road parameter graph must match the archived SHA256 above before staging; the copy commands stage the verified identical graph, weights and merged conversion manifest while preserving the existing driver and MNN assets. Gradle retains the actual build flags and the environment is explicit. The archived [source, graphs, patches and build stamps](../../build/candidates/candidate60-road-rounding/) bind the commands to this artifact. `archive_fold3_c60.py` records the staged assets and verifies the graph inside the APK. Rebuilding from a later working tree is a new artifact and must receive a new hash and qualification records.

After installing and verifying the APK, experiments can be repeated independently:

```sh
.venv/bin/python android/build/candidates/candidate60-road-rounding/run_validation.py fold models instrumentation lifecycle controls-steady controls-faults
.venv/bin/python android/build/candidates/candidate60-road-rounding/run_validation.py emulator instrumentation controls-faults lifecycle controls-steady metadrive
```

The first emulator campaign used SwiftShader. To reproduce the host-GPU campaign, launch the dedicated AVD in a separate terminal with `/Users/sergios/Library/Android/sdk/emulator/emulator -avd sunnypilot_fold3_api35 -port 5554 -no-window -no-audio -no-snapshot -gpu host`, then use `android/build/candidates/candidate60-road-rounding/emulator-host/run_validation.py` instead. Its [launch command](../../build/candidates/candidate60-road-rounding/emulator-host/launch-command.json), [renderer log](../../build/candidates/candidate60-road-rounding/emulator-host/emulator.log) and independent experiment receipts retain the actual configuration.

Each check records its exact command and uses the experiment wrapper for process exit, camera release and forwarding cleanup. Before each experiment, the wrapper requires at least 30 continuous seconds at NORMAL thermal status with skin temperature at or below 32°C; final campaign cleanup repeats that cooldown. A validation failure is retained; an isolation failure prevents the next experiment.

## Remaining qualification gates

Accuracy does not establish the 50 ms sensor-to-final-output deadline. The previous candidate58 ultrawide capture-to-consumer p99 was 87.041 ms and the three-stream sink p99 was 112.477 ms, before inference. Those are [earlier APK camera results](../road-camera-repair-20261107/README.md), not new candidate60 measurements. No camera code or capture configuration was changed in this repair.

Independent rear sensor identity, mounted optical/vehicle calibration and real Panda/vehicle checks remain **BLOCKED**: optical covers are deferred, the phone is unmounted and the Panda is disconnected. Earlier complete-cycle controls failures remain failures until the matched fixture passes the unchanged gate. Combined 30-second, two-minute and 30-minute perception qualification cannot advance while preceding timing, freshness or identity gates fail. Simulator and emulator fixtures cannot clear physical vehicle gates. TX remains disabled regardless of bench results.
