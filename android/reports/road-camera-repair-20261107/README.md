# Road accuracy and three-camera freshness, 7 November 2026

Branch: `codex/fold3-android-port`, based on `7631d5248cc9efd10b962107aadacbb908a9f14d` plus the existing working tree. Candidates 54 through 57 and failed experiments are retained. Vehicle TX remains compiled out and physical engagement disabled. No cooling changes were made.

Candidate58 is a diagnostic build, not a qualified driving release. A passing transport or synthetic fixture does not qualify road perception, camera identity, camera freshness, or physical vehicle operation.

## Final artifact

[Candidate58 APK](../../build/candidates/candidate58-road-camera-diagnostics/app-debug.apk), [artifact manifest](../../build/candidates/candidate58-road-camera-diagnostics/artifacts.json), [converted graphs, source snapshot and provider libraries](../../build/candidates/candidate58-road-camera-diagnostics/).

| Artifact | SHA256 |
|---|---|
| Application APK, 1,147,195,613 bytes | `c4e196a126a998a1b1678043d48ed61ed2946232997786d929f853cde2c6395b` |
| Instrumentation APK | `bcc20fc1b227cb03bde6cf0c6fbc153dce82eb718ffb50cf6fd6fad45b6e999c` |
| Installed/stripped ncnn library | `691798b24da3dd114d83b245ef625b4fe69df1f9a2236af3b4d36d4beab9cd54` |
| Original road model | `65a08adc31d5c456219687d99b7bf5e44d61dae2d49ea67850e76105c7248cce` |
| Original driver model | `dee5a294e8afaacc9295ac5d100e00733ecac278e79264b96331e40a3ede1b04` |

Candidate58 retains candidate57's selected graphs. The new pairwise half GEMM rewrite remains opt-in diagnostic code and is disabled in these graphs after failing the complete model. The native bridge rejects graphs requiring the new arithmetic when the provider capability is absent. Native/model/APK hashes remain part of qualification invalidation.

The first candidate58 packaging output contained about 1 GiB of unreferenced incremental ZIP data. It was never installed; its [failed receipt and original bytes](../../build/candidates/candidate58-packaging-incremental/) are retained. Clearing only Gradle's disposable package output/state produced the final APK above in a successful 10-second rebuild. This packaging cleanup does not change model selection.

Candidate58's installed application and instrumentation hashes match on both [Fold3](../../build/candidates/candidate58-road-camera-diagnostics/fold-installed.json) and [emulator](../../build/candidates/candidate58-road-camera-diagnostics/emulator-installed.json). Candidate57 remains a separately preserved baseline, APK `5e9bfa1fd599093c961f485e41f248034c1611ecd003d2b6288e46a8d942267e`; both [Fold3](../../build/candidates/candidate57-road-temporal-heads/fold-installed-after-validation.json) and [emulator](../../build/candidates/candidate57-road-temporal-heads/emulator-installed-after-validation.json) hashes matched after its validation campaign.

## Candidate58 model results

The complete untapped APK reproduces candidate57: driver accuracy **PASSED**, all 159 comparisons exact; road accuracy **FAILED**, with 106/254/173 out-of-tolerance main-output elements for seeds42/7/123. All nine initial road state comparisons pass. No unexpected non-finite outputs or CPU fallback were observed (476 driver and 654 road Vulkan assignments). See the [actual installed APK model receipt](../../build/candidates/candidate58-road-camera-diagnostics/fold-models.json).

Of 795 required comparisons, **168 passed, 3 failed and 624 remain blocked** by the failed road seeds. Road recurrent/replay and accelerator timing were not advanced. The selected graphs do not use the rejected pairwise GEMM diagnostic.

Matched original-reference initial cases, using actual APK receipts rather than tensor probes:

| Road case | Candidate54 failing main-output elements | Candidate58 failing main-output elements | Candidate54 / candidate58 failing feature-state elements |
|---|---:|---:|---:|
| seed42 | 867 | 106 | 50 / 0 |
| seed7 | 825 | 254 | 21 / 0 |
| seed123 | 664 | 173 | 11 / 0 |

The mismatch counts improve but do not pass. Maximum absolute error changes from 0.162676/0.224926/0.073627 for candidate54 to 0.046875/0.09375/0.09375 for candidate58; seed123's maximum error worsens despite fewer failed elements. The original tolerances are unchanged. [Candidate54 APK model receipt](../../build/candidates/candidate54-road-mean/fold-models.json).

## Candidate58 camera results

All public ultrawide profiles below use camera ID2 at 640x360 for 12 seconds, the same workload as candidate57. The device advertises no stream-use-case overrides on its public cameras; non-default use cases are therefore rejected before session creation rather than forced.

| Profile | Frames acquired / counted as published | Acquisition p99 | Delivery p99 | Maximum recorded delivery age | Delivery samples over 50 ms |
|---|---:|---:|---:|---:|---:|
| Record, direct YUV | 355 / 355 | 84.963 ms | 87.041 ms | 89.217 ms | 355 |
| Public low latency, direct YUV | 356 / 356 | 85.464 ms | 86.411 ms | 88.661 ms | 356 |
| Public low latency, PRIVATE/GLES | 358 / 356 | 84.037 ms | 88.014 ms | 88.376 ms | 358 |

All three transports **PASSED**, with valid capture timestamps, but every recorded delivery sample misses the **camera-only 50 ms gate**. The GLES run retains two acquired frames not counted as published at shutdown; they are not silently removed from the timing count. These are camera diagnostics, not successful model publications. These profiles do not demonstrate an acquisition improvement over candidate57. [Record receipt](../../build/candidates/candidate58-road-camera-diagnostics/fold-ultrawide-record-yuv.json), [low-latency YUV](../../build/candidates/candidate58-road-camera-diagnostics/fold-ultrawide-low-latency-yuv.json), [PRIVATE/GLES](../../build/candidates/candidate58-road-camera-diagnostics/fold-ultrawide-low-latency-gles.json).

Samsung mode37 again **PASSES transport** but remains **BLOCKED for independent rear identity**: 359 composite, 712 preview callbacks and 359 front frames. Preview acquisition p99 is **108.630 ms**, readback **5.754 ms**, and sink delivery **112.477 ms**. Thumbnail acquisition/delivery is 91.554/92.443 ms; front acquisition/delivery is 102.253/102.739 ms. There are zero timestamp errors, coalesced callbacks, duplicate callbacks, dropped readbacks or superseded publications. This measurement does not identify avoidable application queueing, and it cannot qualify a 50 ms sensor-to-model deadline. [Three-stream receipt](../../build/candidates/candidate58-road-camera-diagnostics/fold-three-camera.json).

## Candidate58 controls fixture

The same full-chain steady fixture (`b5ea99327f0ffc121e5556d29b32d4ce8e6e855d14518f46a3f060ea38721f99`) passes functional checks on both devices. These injected fixtures do not use qualified camera perception.

| Metric | Fold3 | Emulator |
|---|---:|---:|
| Mean controls rate | 99.996 Hz | 99.996 Hz |
| Controls work p99 | 2.131 ms | 1.120 ms |
| Legacy complete-cycle p99 | **12.334 ms, FAILED** | **11.999 ms, FAILED** |
| Legacy cycle deadline misses | 1208 | 1276 |
| Work / scheduled-completion misses / skipped slots | 0 / 0 / 0 | 0 / 0 / 0 |
| Release-to-completion p99 | 3.882 ms | 3.195 ms |
| Source fixture deadline misses | 261 | 0 |
| Mean Raylib UI rate | 29.688 fps | 29.621 fps |

Complete-cycle latency and raw counters are retained. Relative to candidate57's 12.3975 ms Fold3 complete-cycle p99, this is not evidence of a repaired timing gate. Reporting coverage is complete for delivered batches; no claim is made that force-stop captured every final loop iteration. [Fold3 fixture](../../build/candidates/candidate58-road-camera-diagnostics/fold-controls-steady.json), [emulator fixture](../../build/candidates/candidate58-road-camera-diagnostics/emulator-controls-steady.json).

## Candidate58 functional validation

| Check | Result |
|---|---|
| Host suite | **PASSED**, 237 tests |
| Android application, instrumentation and JVM test build | **PASSED**, clean final APK packaging |
| Fold3 instrumentation | **PASSED**, 30 tests |
| ARM64 Android15 emulator instrumentation | **PASSED**, 30 tests |
| Both devices: full-chain stale-input and factory-ACC cancellation faults | **PASSED**, zero unsafe outputs or failed workers; synthetic inputs, TX disabled |
| Both devices: background/resume and renderer recreation | **PASSED**; actual physical fold transition remains unverified |
| Emulator resize, restore and renderer recreation | **PASSED** |
| Emulator MetaDrive stale-input fixture | **PASSED**, 2616 packets, 1663 active and 918 inactive-after-fault samples; zero longitudinal or below-minimum-speed activation |
| Emulator actual camera | **FAILED**, all 228 acquired frames rejected for future capture timestamps, zero published; no arrival-time substitution |
| Experiment isolation | **PASSED**, all 15 campaign wrappers; camera release, per-device exclusion, stable APK/provider and cooldown receipts retained |
| Real-camera/model combined 30-second, two-minute and 30-minute qualification | **BLOCKED**, preceding accuracy, freshness, independent rear identity and legacy controls timing gates have not passed |

The synthetic MetaDrive plant is not a Hyundai model and does not exercise camera inference. Its maximum lateral error is 0.464 m in that fixture. Final [machine-readable measurements and per-experiment outcomes](../../build/candidates/candidate58-road-camera-diagnostics/measured-summary.json) retain numerical failures, all raw timing counters and the distinction between functional and qualification outcomes. Exact [Fold3 commands](../../build/candidates/candidate58-road-camera-diagnostics/fold-commands.json) and [emulator commands](../../build/candidates/candidate58-road-camera-diagnostics/emulator-commands.json) are archived with their actual APK.

Final installed application and instrumentation hashes **PASS** again on [Fold3](../../build/candidates/candidate58-road-camera-diagnostics/fold-installed-after-validation.json) and [emulator](../../build/candidates/candidate58-road-camera-diagnostics/emulator-installed-after-validation.json). Final cleanup **PASSES**: no app/test processes, no owned cameras, no experiment forwards and unchanged APK. Fold3 is NORMAL at 26.1 degrees C after 32.38 continuous seconds NORMAL; emulator is NORMAL at 30.8 degrees C after 30.73 seconds. The emulator was shut down after verification; the Fold3 app remains stopped. [Fold3 cleanup](../../build/candidates/candidate58-road-camera-diagnostics/fold-final-cleanup.json), [emulator cleanup](../../build/candidates/candidate58-road-camera-diagnostics/emulator-final-cleanup.json).

## Candidate57 results

| Check | Result | Evidence |
|---|---|---|
| Host focused suite | **PASSED**, 234 tests, 19.324 seconds | [Log](../../build/candidate57-delivery-host-tests.log) |
| Android build and JVM unit tests | **PASSED**, Gradle success, 83 tasks | [Log](../../build/candidate57-gradle.log) |
| Fold3 driver original-reference suite | **PASSED**, 159/159 exact; 476 Vulkan assignments, no CPU fallback | [Model receipt](../../build/candidates/candidate57-road-temporal-heads/fold-models.json) |
| Fold3 road initial full-model seeds | **FAILED**, seed42/7/123 outputs have 106/254/173 out-of-tolerance elements; max errors 0.046875/0.09375/0.09375. All nine recurrent-state comparisons pass; 654 Vulkan assignments, no CPU fallback | [Model receipt](../../build/candidates/candidate57-road-temporal-heads/fold-models.json) |
| Full 795 native comparisons | **BLOCKED**, 171 executed: 168 pass, 3 fail, remaining 624 road recurrent/replay comparisons gated by failed initial seeds. No unexpected non-finite outputs | Same model receipt |
| Accelerator performance qualification | **BLOCKED** by failed road accuracy; no accelerator timing or combined workload qualification claimed | Same model receipt |

These are trace-free complete-model APK results, not local tensor-section acceptance. The app did not write a successful road accuracy record; the resulting missing-record timing gate is expected and does not permit promotion. The seed42 diagnostic improvement is reproduced in the actual APK.

## Changes

- The pinned ncnn patch now reproduces the original ARM64 arithmetic in the road stem and grouped convolution, uses correctly rounded FP32 division, and retains the original sigmoid approximation. Fused pointwise Conv/Sigmoid now takes the same verified accumulation path as its unfused diagnostic section. FP16 and BF16 execution are not globally changed.
- Trace tools bind original Conv/Gemm nodes by exact weights, bias, layout and graph structure, including transposed ONNX Gemm weights. Fused activations retain both original-node mappings. Weighted and all-tensor traces can follow the road heads and recurrent inputs. Local comparisons use captured Vulkan inputs, not replacement host-prefix inputs. Serialized tracing is excluded from accelerator timing qualification.
- Diagnostic ordered-half annotations are selected by original node and original FP16 contract. Duplicated converted heads are accepted only with the same resolved input through Split aliases; ambiguous inputs or serialized weights are rejected. Road annotation selection is reproducible through `prepare_accel_models.py --ncnn-ordered-half-node`.
- A separately hashed diagnostic rewrite restores omitted half-output boundaries after Gemm and LayerNorm. It worsened the complete seed-42 result (209 failing output elements versus 193 without that rewrite), so **it is not enabled in candidate57**. Local tolerance agreement can still change a downstream half rounding decision; this rewrite is not a qualified repair.
- Public Camera2 `low-latency`, `low-latency-preview` and `low-latency-record` profiles use supported processing controls, size-specific minimum frame durations and advertised regular-session FPS ranges. Advertised stream use cases require support checks. Unsupported vendor modes and recorded provider-crash routes remain excluded. Samsung mode37 remains the three-stream baseline.
- Host experiment runners now hold a per-device lock before stopping processes or touching forwarding, including final cleanup/cooldown. A second experiment on the same device is rejected before it can disturb the first.

## Preserved gates

Original model/golden bytes and `rtol=1e-3`, `atol=1e-4` are unchanged. All 795 comparisons remain required; road recurrent/replay stages remain gated when initial full-model seeds fail. Diagnostic graphs cannot create qualified perception. No CPU fallback is permitted for qualified accelerator results.

The sensor-to-final-model-publication limit remains **50 ms maximum**, using the earlier rear capture for road and current front capture for driver monitoring. The rear synchronization bound remains 20 ms. Transport success and p99 measurements cannot establish a maximum deadline guarantee. Dropped, invalid and unpublished inputs remain failures/counters, not a way to hide misses.

Controls retain the existing complete-cycle timing gate and raw misses. MODERATE thermal pause, cooldown and critical-temperature aborts are unchanged. Every device experiment uses verified process/camera exit, forwarding cleanup and 30 continuous seconds of NORMAL thermal status with skin temperature at or below 32 degrees C.

## Earlier evidence preserved

| Artifact | Result | Evidence |
|---|---|---|
| candidate54 | Driver 159/159; road initial seeds failed; complete-cycle p99 about 12.366 ms | [Baseline report](../accelerator-controls-20261107/README.md) |
| candidate55 | Public ultrawide YUV at 640x360: record acquisition p99 82.912 ms; low-latency p99 83.598 ms. Neither meets 50 ms before inference. | [Record](../../build/candidates/candidate55-road-sigmoid-low-latency/fold-ultrawide-record-yuv.json), [low latency](../../build/candidates/candidate55-road-sigmoid-low-latency/fold-ultrawide-low-latency-yuv.json) |
| candidate55 | Mode37 triple capture transport passes; preview delivery p99 114.763 ms, thumbnail 92.837 ms, front 96.882 ms. Rear identity unproved. | [Three-camera receipt](../../build/candidates/candidate55-road-sigmoid-low-latency/fold-three-camera.json) |
| candidate55 | Matched full-chain fixture work p99 2.287 ms; raw complete-cycle p99 12.324 ms with 1,253 legacy deadline misses. Functional fixture passed; timing failed. | [Controls receipt](../../build/candidates/candidate55-road-sigmoid-low-latency/fold-controls-steady.json) |
| candidate56 | Public ultrawide PRIVATE/GLES at 640x360: 357 frames, 29.849 fps, acquisition p99 88.001 ms, delivery p99 91.572 ms, zero invalid capture timestamps. It does not improve the acquisition floor. | [Receipt](../../build/candidates/candidate56-road-stem-divide/fold-ultrawide-low-latency-gles.json) |
| candidate56 emulator | Host-graphics instrumentation 30/30, controls/faults, lifecycle/display recovery and MetaDrive stale-input fixture pass. Advertised camera ID1 produces future capture timestamps: 228/228 frames rejected, none published. No arrival-time substitution. | [Emulator evidence](../../build/candidates/candidate56-road-stem-divide/emulator-host/) |

Candidate55 APK: `b7501ea24a5933f816abb4b3c2479682ee40df6669110de27a67abc71d7ae9d1`.
Candidate56 APK: `54888874a47672e1432d363157e95cd8f5a2d5008edbe72461cc300acce14c0a`.

Earlier invalid emulator configuration (IDs0,1 although only ID1 was advertised), SwiftShader startup failure, installation storage failure, and the overlapping lifecycle/instrumentation attempt are preserved in the candidate56 directory. The overlap is explicitly invalidated; later clean runs do not erase it.

## Candidate57 device validation

Every row below uses installed APK `5e9bfa1fd599093c961f485e41f248034c1611ecd003d2b6288e46a8d942267e`. Exact commands and cleanup/cooldown receipts are retained in the candidate directory. These tests ran in separate per-device leases; no experiment overlap was accepted.

| Check | Result |
|---|---|
| Fold3 instrumentation | **PASSED**, 30 tests |
| Emulator instrumentation | **PASSED**, 30 tests |
| Both devices: full process chain, stale inputs, cancellation, lifecycle | **PASSED** functional fixtures; perception remains unqualified |
| Emulator display recreation and MetaDrive stale fault | **PASSED** synthetic recovery; no vehicle qualification |
| Fold3 ultrawide record YUV, 640x360 | Transport **PASSED**; acquisition p99 **83.279 ms**, delivery **84.071 ms**; camera-only 50 ms gate **FAILED** |
| Fold3 ultrawide low-latency YUV, 640x360 | Transport **PASSED**; acquisition p99 **82.904 ms**, delivery **84.834 ms**; camera-only gate **FAILED** |
| Fold3 ultrawide low-latency PRIVATE/GLES, 640x360 | Transport **PASSED**; acquisition p99 **84.945 ms**, delivery **88.613 ms**; camera-only gate **FAILED** |
| Fold3 mode37 three streams, 960x540 | Transport **PASSED**; preview acquisition p99 **109.648 ms**, readback **6.455 ms**, sink delivery **114.875 ms**; rear identity **BLOCKED**, freshness **FAILED** |
| Emulator real camera timestamps | **FAILED**: all 228 future-timestamp frames rejected, zero published; arrival time was not substituted |
| Fold3 steady controls | Functional **PASSED**, 100.002 Hz, work p99 **2.269 ms**, complete-cycle p99 **12.3975 ms** (**FAILED**), 1313 legacy cycle misses; no scheduled-completion misses or skipped slots |
| Emulator steady controls | Functional **PASSED**, 99.999 Hz, work p99 **1.474 ms**, complete-cycle p99 **12.432 ms** (**FAILED**), 1293 legacy cycle misses; no scheduled-completion misses or skipped slots |

The matched controls fixture hash is `b5ea99327f0ffc121e5556d29b32d4ce8e6e855d14518f46a3f060ea38721f99`. Fold3 source delivery itself missed 233 fixture deadlines; emulator source missed none. Both retain complete reporting coverage. Scheduled release-to-completion and the legacy complete-cycle metric are distinct; a passing newer metric cannot clear the legacy failure. See [machine-readable measurements](../../build/candidates/candidate57-road-temporal-heads/measured-summary.json).

Mode37 had zero coalesced callbacks, duplicate texture callbacks, dropped readbacks or superseded publications in this run. The measured delay is already present at texture acquisition. No application queue adjustment was promoted without measured evidence of a benefit. The ultrawide low-latency controls did not materially reduce its sensor-to-application floor. These results establish limitations of the tested supported routes, not proof that every stock-camera route is impossible.

No 2-minute or 30-minute combined camera/model workload was started because preceding accuracy and freshness gates fail. The 30-second controls fixtures inject deterministic perception and do not establish combined camera/accelerator readiness.

## Road diagnosis

The native57 library is `d8e88bddb3549d114a08a98479a3041f60006239504265b1b0b7aa247362bef3`. Native probes run with this separately pushed library while candidate56 remains installed; they are not candidate56 APK accelerator qualification.

- The fixed captured road stem and gated pointwise sections match their original sections exactly, including the formerly divergent fused sigmoid boundary.
- Seed42 with four early ordered-half heads still failed 533 output elements. Fourteen selected original heads, including a duplicated converted head, reduced this to 193. Adding the two temporal output heads reduced it to **106**, but maximum absolute output error was still 0.046875 and full-model parity failed.
- With those sixteen original heads, image/desire history are exact and feature history passes tolerance (maximum absolute error 0.000030517578125). This is one seed, not recurrent qualification.
- The captured tail first fails the full-prefix tolerance at `linear_196`, original `model/p_node_MatMul_60/MatMulAddFusion`, FP16 shape 9x512, Vulkan NCDHW. Its section on the captured input passes tolerance but differs in 12 half rounding decisions. Later proven ordered heads match on identical captured inputs; their global error is inherited. The remaining temporal prefix must be corrected before promotion.

Evidence: [diagnostic full-output check and trace](../../build/road-temporal-heads-c57-seed42.json), [original-node mapping](../../build/road-temporal-heads-c57-seed42-comparison.json), [captured-input comparisons](../../build/road-temporal-heads-c57-seed42-same-input.json). The mapping binds the archived candidate56 conversion manifest; using the regenerated candidate57 manifest for old traces is correctly rejected. Complete trace-free APK validation remains the acceptance check.

## Further temporal diagnosis: rejected pairwise rewrite

The native58 diagnostic library SHA256 is `2013f46d0629751acd679c0206f0f1a86eafa20a6d8554365a0f8855827018cb`. These probes ran with candidate57 installed and a separately pushed library; they are not candidate57 APK results.

A private Vulkan-only marker selects the original NoTrans half GEMM's paired products, ordered FP32 accumulation with bias first, and final half rounding. Conversion requires exact original weight/bias bytes, half input/output contracts and NoTrans semantics; conflicting transpose/dtype/fusion provenance is rejected before graph writes. Matrix shape and FP32 transport are checked, CPU execution is rejected, and the reference Cast/GEMV paths remain unchanged.

The complete untapped seed42 graph still failed **133** output elements, maximum error **0.03125**, compared with candidate57's **106**. The rewrite is therefore **REJECTED for model selection**, regardless of its lower maximum error. All three state outputs still pass tolerance. The traced graph reproduces the same full-output failure count.

On identical captured Vulkan inputs, `linear_191`, `linear_192` and `linear_193` now match the original sections **exactly**: 4608, 4608 and 13824 elements respectively, with zero half-rounding disagreements. The trace now binds LayerNorm by affine bytes, epsilon, axis, statistics dtype and upstream provenance. The earliest mapped global tolerance failure is **`ln_119`**, original `model/p_node_layer_norm`, FP16 original shape `[1,9,512]`, actual Vulkan NCDHW FP32 transport. Its identical-input section passes the requested tolerance but disagrees in **13 half rounding decisions out of 4608** (maximum raw difference 0.000118494). These differences can propagate beyond tolerance through subsequent attention; local tolerance passes are insufficient for model promotion. The normalization arithmetic and inherited prefix rounding remain unresolved.

Evidence: [untapped diagnostic](../../build/road-dot2-c58-seed42.json), [captured trace](../../build/road-dot2-prefix-c58-seed42.json), [original-node mapping](../../build/road-dot2-prefix-c58-seed42-comparison.json), [identical-input sections](../../build/road-dot2-prefix-c58-seed42-same-input.json). Earlier linear-only mappings are preserved alongside the expanded analysis.

After the final candidate58 campaign, two further trace-free native diagnostics tested those erased rounding boundaries on the current selected graph. Restoring ten verified half-output boundaries reduces seed42's failing main-output elements from 106 to **68**; combining those boundaries with six ordered NoTrans GEMMs reduces them to **40**. Both have maximum error **0.0078125**, all state outputs within tolerance, and no unexpected non-finite values. Both still **FAIL** original-reference tolerance, so neither is selected in the APK or promoted as accurate perception. These are separately hashed native graphs running beside installed candidate58, not candidate58 model-validation results. The other seeds, recurrent sequences and replay were not advanced for these failed diagnostics. Cleanup, provider stability, installed APK stability and more than 32 continuous seconds NORMAL pass for both.

[Half-boundary diagnostic](../../build/road-half-boundaries-c58-seed42.json), [combined boundary/matrix diagnostic](../../build/road-half-boundaries-dot2-c58-seed42.json). To reproduce, use `probe_ncnn.py --round-half-outputs --trace-count 0` and then add `--ordered-half-dot2`, with separate report paths. The remaining normalization/prefix error needs an arithmetic repair; a reduced mismatch count cannot clear qualification.

The final host suite passes **237 tests** ([log](../../build/candidate58-complete-host-tests.log)). Focused edit checks found no incompatible callers. Whole-working-tree static quality gating still fails against HEAD with 111 preexisting-worse major findings across the accumulated uncommitted port, including the native ORT session's complexity; there is no pinned clean quality baseline. The [final static receipt](../../build/candidate58-post-delivery-quality-delta.xml) retains this limitation; it is not reported as a clean static-analysis pass. Focused `git diff --check` passes.

## Reproduction

Use the pinned Python environment, NDK and model assets already documented in [Android README](../../README.md). Source/build hashes are in each candidate's `artifacts.json`; the dirty working tree is not represented by HEAD alone.

```sh
SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk \
  .venv/bin/python -m unittest discover -s android/tests
.venv/bin/python android/scripts/build_ncnn.py
```

The selected graph's exact road conversion argument list is [candidate57-prepare-command.json](../../build/candidate57-prepare-command.json). Driver conversion and its 27 ordered-half heads are preserved. The rejected global half-output diagnostic is not in that argument list.

```sh
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
ANDROID_HOME=/Users/sergios/Library/Android/sdk \
FOLD3_BUILD_PYTHON=/Users/sergios/sunnypilot/.venv/bin/python \
android/gradlew -p android --offline --console plain \
  -Pfold3CameraCompat=true -Pfold3Qnn=true \
  -Pfold3MnnPatched=true -Pfold3NcnnPatched=true \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
```

Before rebuilding an archived APK that shares hard links with Gradle output, remove the output link, preserving the archived file. Use a new candidate directory; never overwrite earlier receipts.

To reproduce diagnostic-only tracing, use `probe_ncnn.py --ordered-half-dot2 --trace-count 64 --trace-skip 600` with the original tolerances; run `compare_ncnn_trace.py` and `compare_ncnn_same_input.py` on its report. Do not enable `--ncnn-half-dot2` in the selected graph after this failed result.

For each connected device, install the final application and instrumentation with `adb -s SERIAL install -r -g APK`, then use [verify_installed.py](../full-model-controls-20261101/verify_installed.py) with `--serial`, `--candidate` and `--output`. Run [candidate58's campaign](../../build/candidates/candidate58-road-camera-diagnostics/run_validation.py) with `fold` or `emulator`; the runner records exact commands and wraps each experiment in cleanup/cooldown isolation. The dedicated ARM64 Android15 AVD is `sunnypilot_fold3_api35`, serial `emulator-5554`. It advertises camera ID1 at 640x480; emulator camera timing and deterministic injected fixtures are separate checks.

## Physical gates

The Fold3 is unmounted; Panda is disconnected; lens-cover trials remain deferred. Camera timestamp overlap, unequal pixels or multiple active pipelines do not prove independent rear sensors. Independent rear identity, mounted geometry/calibration, actual Panda firmware/fingerprint/harness and vehicle behavior remain **BLOCKED**. Synthetic factory-ACC/cancellation checks cannot clear these gates. TX remains disabled regardless of bench results.
