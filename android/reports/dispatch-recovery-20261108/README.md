# Candidate73 road dispatch experiments

Overall: **FAILED / unqualified** until all accuracy, sensor-to-output, controls and physical gates pass. Vehicle TX remains compiled out; engagement disabled. Cooling optimization is deferred and thermal safeguards remain active.

APK SHA256: `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`. Installed hash verification: PASS. Native library SHA256: `6bdb98a0c6704e1030d72e3aef7a57fe409d9dc8be365a19bfddcabb37da6f7b`.

## Change

Added an explicit per-session road Vulkan dispatch setting for 131072, 262144 and 524288 workgroups. The default and passing driver kernels are unchanged. Overrides reject other models/backends/devices and unsupported values. Accuracy records bind the setting to the APK, implementation, converted graphs, providers and device. Diagnostic configurations cannot qualify live perception. Full-model checks precede unprofiled timing and separately instrumented GPU profiling.

## Complete isolated inference

These measurements exclude camera acquisition and therefore cannot qualify the 50 ms sensor-to-publication budget. Every dropped/rejected input and end-to-end gate remains applicable. Internal wait includes GPU execution and must not be subtracted as removable overhead.

| Configuration / model | Accuracy gate | p50 ms | p99 ms | Max ms | >50 ms |
|---|---|---:|---:|---:|---:|
| Candidate72 / dmonitoring_model | 795 PASS | 137.223 | 148.841 | 167.380 | 100/100 |
| Candidate72 / driving_supercombo | 795 PASS | 311.609 | 363.188 | 363.726 | 100/100 |
| Candidate73 models / dmonitoring_model | 795 PASS | 138.252 | 150.770 | 169.599 | 100/100 |
| Candidate73 models / driving_supercombo | 795 PASS | 339.251 | 363.897 | 375.508 | 100/100 |
| Candidate73 dispatch131072 / dmonitoring_model | 795 PASS | 137.886 | 149.499 | 170.522 | 100/100 |
| Candidate73 dispatch131072 / driving_supercombo | 795 PASS | 353.897 | 367.452 | 384.042 | 100/100 |
| Candidate73 dispatch262144 / dmonitoring_model | 795 PASS | 137.486 | 148.286 | 172.911 | 100/100 |
| Candidate73 dispatch262144 / driving_supercombo | 795 PASS | 342.880 | 362.271 | 362.437 | 100/100 |
| Candidate73 dispatch524288 / dmonitoring_model | 795 PASS | 138.743 | 148.646 | 166.671 | 100/100 |
| Candidate73 dispatch524288 / driving_supercombo | 795 PASS | 309.712 | 352.664 | 353.043 | 100/100 |

## Diagnostic GPU profiles

Dispatch 131072: status `passed`; submission measurements `{'version': 1, 'samples': 10, 'dispatch_limits': [131072], 'rough_scores': [9], 'mean_internal_count': 3.0, 'mean_internal_wait_ms': 170.1199000120163, 'mean_record_excluding_internal_wait_ms': 10.6298029879837}`.

- `conv_88` / `Convolution`: 31.074 ms mean GPU execution.
- `conv_114` / `Convolution`: 8.974 ms mean GPU execution.
- `conv_90` / `Convolution`: 6.242 ms mean GPU execution.
- `conv_109` / `Convolution`: 5.415 ms mean GPU execution.
- `conv_107` / `Convolution`: 5.162 ms mean GPU execution.
Dispatch 262144: status `passed`; submission measurements `{'version': 1, 'samples': 10, 'dispatch_limits': [262144], 'rough_scores': [9], 'mean_internal_count': 2.0, 'mean_internal_wait_ms': 160.47670001983641, 'mean_record_excluding_internal_wait_ms': 6.947638580163575}`.

- `conv_88` / `Convolution`: 21.110 ms mean GPU execution.
- `conv_114` / `Convolution`: 8.978 ms mean GPU execution.
- `conv_109` / `Convolution`: 5.414 ms mean GPU execution.
- `conv_107` / `Convolution`: 5.156 ms mean GPU execution.
- `conv_111` / `Convolution`: 4.902 ms mean GPU execution.
Dispatch 524288: status `passed`; submission measurements `{'version': 1, 'samples': 10, 'dispatch_limits': [524288], 'rough_scores': [9], 'mean_internal_count': 1.0, 'mean_internal_wait_ms': 132.94650000333786, 'mean_record_excluding_internal_wait_ms': 5.6433383966621395}`.

- `conv_88` / `Convolution`: 14.623 ms mean GPU execution.
- `conv_114` / `Convolution`: 8.974 ms mean GPU execution.
- `conv_109` / `Convolution`: 5.417 ms mean GPU execution.
- `conv_107` / `Convolution`: 5.211 ms mean GPU execution.
- `conv_111` / `Convolution`: 4.904 ms mean GPU execution.

## Controls and recovery

| Device / fixture | Work p99 ms | Legacy complete-cycle p99 ms | Hz | Scheduled misses | Raw cycle misses |
|---|---:|---:|---:|---:|---:|
| fold-steady | 2.107 | 12.201 | 99.999 | 0 | 1270 |
| fold-faults | 2.174 | 12.460 | 99.994 | 0 | 1330 |
| emulator-steady | 3.180 | 14.606 | 99.376 | 7 | 1279 |
| emulator-faults | 1.851 | 11.746 | 99.999 | 0 | 1311 |

Controls calculations, pacing, watchdogs, raw timing definitions and cancellation behavior were preserved. No controls timing fix is claimed from a dispatch-only change. Emulator fixtures establish functional recovery, not physical accelerator speed or road readiness.

fold: instrumentation `{'serial': 'RFCR81Y1TBJ', 'package': 'com.samsung.sunnypilot.fold3', 'completed_tests': 30, 'expected_tests': 30, 'returncode': 0, 'status': 'passed'}`; lifecycle `passed`.

emulator: instrumentation `{'serial': 'emulator-5554', 'package': 'com.samsung.sunnypilot.fold3', 'completed_tests': 30, 'expected_tests': 30, 'returncode': 0, 'status': 'passed'}`; lifecycle `passed`.

Host suite: 263 tests PASS; additional qualification checks PASS. Native and APK builds PASS. Experiment isolation: PASS. Source and immutable artifact audit is retained in artifact-audit.json. Build retries are preserved, including stale-native rejection and the regenerated asset archive.

## Remaining gates

The 524288 setting improves road median/p99 from this APK's default 339.251/363.897 ms to 309.712/352.664 ms (8.7/3.1 percent). Against candidate72, median improves only 0.6 percent and p99 2.9 percent. These isolated measurements do not establish a sustained improvement; device frequency and thermal variation remain possible. No diagnostic override is promoted to live perception.

The initial emulator launch failed because host storage was insufficient. Its failed log and terminated wait are preserved; after reclaiming completed build caches, the separate emulator retry passed instrumentation, recovery and final teardown. The original campaign process therefore retains a nonzero exit, while the per-device final receipts distinguish the successful retry. Controls timing failures remain failures on that retry.

The 50 ms deadline is unchanged. Do not advance to combined 30-second, two-minute or 30-minute qualification while isolated inference fails. Three-camera identity/freshness and the existing controls complete-cycle failure remain separate gates. No new camera timing evidence is claimed by these road-only experiments. The phone is unmounted, optical rear-lens proof deferred and Panda disconnected. Mounted calibration and physical vehicle validation remain BLOCKED.

The one-versus-two live spatial accumulator follow-up has not been completed by this dispatch experiment. Preserve candidate70/71/72 results and do not infer an improvement from a changed metric or isolated kernel time.

## Reproduction

Standard build:

```sh
.venv/bin/python android/scripts/build_ncnn.py
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk .venv/bin/python -m unittest discover -s android/tests
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home ANDROID_HOME=/Users/sergios/Library/Android/sdk FOLD3_BUILD_PYTHON=$PWD/.venv/bin/python android/gradlew -p android --offline --console plain -Pfold3CameraCompat=true -Pfold3Qnn=true -Pfold3MnnPatched=true -Pfold3NcnnPatched=true :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
```

Candidate-local resume scripts preserve the verified current source archive while reclaiming completed, regenerable staging. Ordinary builds use the standard recipe. Each device command is recorded in candidate73-dispatch/*-command.json and wrapped by experiment.py for ownership, process exit, camera release, forwarding cleanup, artifact stability and cooldown. Raw reports and evidence remain under android/build/candidates/candidate73-dispatch.
