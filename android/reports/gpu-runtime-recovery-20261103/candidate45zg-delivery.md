# Fold 3 GPU recovery: candidate45zg

Branch: `codex/fold3-android-port`. This is a diagnostic build, not a qualified driving runtime. Vehicle TX remains compiled out and physical engagement disabled. Cooling optimization is deferred; thermal safeguards remain active.

APK: [candidate45zg/app-debug.apk](../../build/candidates/candidate45zg/app-debug.apk).
SHA-256: `28c7d37923f7b2537c56181a85bc406f65011030a37205f2bbc1524a0e10a3f7`.
The archived test APK and engine/model/patch identities are in [artifacts.json](../../build/candidates/candidate45zg/artifacts.json). Original model hashes remain unchanged. Original tolerances remain `rtol=1e-3`, `atol=1e-4`.

## Implemented and measured

- Reproducible ncnn Vulkan patches restore explicitly typed FP16 casts with integer round-to-nearest-even, replace the invalid negative-base scalar cube operation, and use the existing portable spatial-reduction tree where the Adreno subgroup path is inaccurate.
- The initial driver Cast now has zero mismatches across 1,382,400 values. The cube fix removes the traced unexpected NaNs. The reduction error against a host mean of the same captured GPU input drops from approximately 0.14072 to 5.96e-8. These are operator diagnostics, not full-model qualification.
- Full driver Vulkan shell-probe maximum error drops from approximately 7.27825 before repairs to 0.0056603 after repairs. The remaining 293 output mismatches still fail tolerance. The first mapped remaining tolerance failure is the final linear projection, `linear_100`.
- MNN native OpenCL convolution was tested while retaining the strict GEMV and other required precision boundaries. Small convolution differences cross a later FP16 rounding boundary. The strict GEMV matches the original host operation bit-for-bit when given the same captured GPU input, isolating this failure upstream of GEMV.
- Trace reports preserve original-node mapping, raw shapes, layout, dtype, backend assignment and separately hashed diagnostic graphs. The narrowly checked MNN reduction callback adapter only restores a known omitted leading unit batch dimension; it changes no model execution.
- Runtime/build identities bind the actual APK, converted graphs, provider libraries and patches. Unpatched Cast execution and CPU fallback cannot qualify perception.

## Passed / failed / blocked

| Check | Result | Evidence |
| --- | --- | --- |
| Focused host suite | PASSED: 186 tests | [host-tests-final2.log](host-tests-final2.log) |
| Android build and JVM tests | PASSED | [build log](../../build/candidate45zg-build.log) |
| Fold functional validation | PASSED: six experiments, including 28/28 instrumentation tests | [inventory](fold-candidate45zg-validation-driver.json) |
| Installed APK identity | PASSED on Fold and dedicated ARM64 Android 15 emulator | [Fold](candidate45zg-installed-fold.json), [emulator](candidate45zg-installed-emulator.json) |
| Emulator recovery and injected-input validation | PASSED: nine experiments, including 28/28 instrumentation tests, process supervision, lifecycle, ADB faults, IPC, MetaDrive cancellation/stale inputs and thermal faults | [inventory](emulator-candidate45zg-validation-driver.json) |
| Emulator native camera timing | FAILED: 289 future delivery timestamps and 288 capture-result timestamp errors; capture times were not replaced | [camera report](emulator-candidate45zg-camera-timing.json) |
| Final-APK MNN native driver accuracy | FAILED: all three seeds; 4, 6 and 7 output mismatches; max error 0.00390625 | [accuracy](fold-candidate45zg-mnn-native-initial.json) |
| Final-APK ncnn Vulkan driver accuracy | FAILED: all three seeds; 293, 312 and 297 output mismatches | [accuracy](fold-candidate45zg-ncnn-vulkan-models.json) |
| ncnn road model | BLOCKED: no converted ncnn road entry in this APK; diagnostic reports `KeyError: 'ncnn'` | Same accuracy report |
| Controls work | PASSED isolated work target: p99 2.867604 ms | [matched fixture](fold-candidate45zg-chain-device-steady.json) |
| Controls complete-cycle timing | FAILED: legacy interval p99 14.122188 ms | Same fixture; metric definition retained |
| Full accelerated recurrent/replay suite and timing | BLOCKED by initial accuracy failures | All 795 comparisons remain required |
| 30-second, two-minute and 30-minute combined qualification | BLOCKED by accuracy, inference timing, camera freshness and complete-cycle timing gates | No sustained qualification claimed |
| Rear sensor proof / mounted calibration / Panda vehicle validation | BLOCKED | Lens-cover trials deferred, phone unmounted, Panda disconnected |

Controls measurements use the same fixture as candidate44 and candidate45ze. Candidate44 work/cycle p99 was 7.30/16.61 ms; candidate45ze was 2.6687/13.5495 ms. Candidate45zg is not a measured improvement over candidate45ze. UI mean in the final isolated fixture was 29.55 fps. The report retains 1,273 controls cycle-deadline misses and 238 source-deadline misses. Zero skipped slots or scheduled-release misses does not erase these failures. Scheduler-delay p99 is 2.793229 ms, reporting p99 is 0.362396 ms, and release-to-completion before reporting p99 is 6.18026 ms; these separate metrics do not replace the 14.122188 ms legacy complete-cycle measurement.

The historical 795 passing comparisons belong to candidate45ze's strict accurate MNN/OpenCL implementation. They do not qualify this APK's native-convolution MNN or ncnn backends. No diagnostic checkpoint timing is presented as accelerator performance.

Final cleanup PASSED: both devices have no app/test processes, no camera ownership and no experiment forwarding, with more than 30 continuous NORMAL seconds at skin temperature <=32°C. The emulator exited and is no longer listed by ADB. The final cleanup report retains the installed APK identity and actual thermal samples. No thermal overrides remain.

See the [artifact-bound validation report](candidate45zg-validation.json), [cleanup evidence](candidate45zg-final-cleanup.json), [reproducible commands](candidate45zg-reproduce.md), original-model identities in [original-model-hashes.json](original-model-hashes.json), the [same-input GEMV proof](mnn-native-same-input-gemv.json), and the [Vulkan trace comparison](ncnn-driver-seed42-reduction-comparison.json).
