# Fold 3 road Vulkan follow-up

Four singleton-rank operations in the converted road graph now use Vulkan Reshape. The ncnn patch adds Vulkan MatMul for the two pinned attention shapes and follows the strict OpenCL reference's paired accumulation and half rounding. The build recipe uses a clean source export and disables automatic patch reversal.

## Reproduce

From the repository root, with the pinned Android SDK, NDK, Gradle, host environment and model assets already provisioned:

```sh
.venv/bin/python android/scripts/prepare_accel_models.py --engine ncnn \
  --model driving_supercombo --ncnn-explicit-casts \
  --destination "$PWD/android/build/road-rank-repro"
.venv/bin/python android/scripts/build_ncnn.py
rg 'MatMul_vulkan_layer_creator' android/build/native/accelerators/ncnn-patched/android/src/layer_registry.h
```

Stage the generated road ncnn param and weights under `android/app/build/model-reference/accelerators/ncnn`. Replace the `models.driving_supercombo.ncnn` entry in the staged accelerator manifest with the generated entry. Preserve the driver artifacts and remaining manifest entries.

```sh
export FOLD3_BUILD_PYTHON="$PWD/.venv/bin/python3.12"
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME="$HOME/Library/Android/sdk"
android/gradlew -p android --offline --no-daemon --console plain \
  -Pfold3CameraCompat=true -Pfold3NcnnPatched=true \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
```

The local run harness and raw receipts are under `android/build/road-rank-M2ZsUl`. The harness checks installed hashes, runs model validation on the Fold 3 and uses synthetic full-chain fixtures on both devices. It stops the app and test processes, checks camera release and forwarding, and verifies cooldown. It shuts down the dedicated emulator when done.

## First artifact

APK SHA-256 `cff4fb31c3b2419e6c2a27e985780c79668f7f2d0c01a8cc05c1440e194523ae`. Test APK SHA-256 `96876c828001aafe773f8d00589528522dce16709372ab77e134117186de6a6f`. Both installed hashes matched on both devices.

- Host rewritten road ONNX passed 636 original-reference comparisons. The focused host test suites passed 71 tests.
- Fold 3 driver-model comparisons passed. The road graph was rejected before inference because the first build lacked Vulkan MatMul registration. This APK failed accelerator qualification.
- Each device passed 28 instrumentation tests, the steady and fault fixtures, and the isolation checks.
- Fold 3 steady controls averaged 99.995 Hz. Work p99 was 2.15 ms versus 2.27 ms on candidate45zy. Complete-cycle p99 was 12.31 ms versus 12.26 ms. This run recorded 1,217 raw cycle misses and zero scheduled deadline misses. The controls complete-cycle timing gate remains open.

Vehicle TX and physical engagement are disabled. Lens identity, mounted calibration and physical Panda and vehicle integration require separate evidence.

## Clean ncnn export artifact

APK SHA-256 `0165eb1ffda191362f632437da483d803c5bbae16fc8667563128df4837a21c2`. The ARM64 ncnn library registered both road attention operations on Vulkan. The Fold 3 driver check passed again. Road inference reached output readback, where ncnn's inferred terminal blob list included unused input `in3`. The exact output-set check rejected it. The bridge now excludes terminal blobs declared as inputs and keeps the exact output check for all model outputs.

The Fold 3 steady fixture averaged 99.998 Hz with 2.11 ms work p99, 12.20 ms complete-cycle p99 and 3.91 ms release-to-completion p99. It recorded 1,238 raw cycle misses, zero scheduled misses and no skipped slots. Complete-cycle timing remains open. The run used synthetic inputs.

Both installed hashes matched on the Fold 3 and Android 15 emulator. Each passed 28 instrumentation tests, steady and fault fixtures, and experiment isolation. The model report verified Vulkan assignment for both MatMul layers and no CPU fallback. The readback failure prevented numerical qualification.

## Output bridge artifact

APK SHA-256 `5cc93ea5e10de4d91f09af0dfd00817be8bb273e91cf2e464db3f2f16d592143`. The Fold 3 returned all four road outputs without CPU fallback. Image and desire histories matched exactly for all three seeds. Road outputs and feature history failed tolerance for all three seeds. For seed 42, 1,636 of 2,576 driving output values and 50 feature-history values exceeded tolerance. Driver accuracy passed. The validator stopped before road recurrent replay and timing.

Fold 3 steady controls averaged 99.998 Hz with work p99 of 2.21 ms, complete-cycle p99 of 12.30 ms, 1,267 raw cycle misses, zero scheduled misses and no skipped slots.

The exported road graph contained empty axis lists for two feature L2 norms and the attention softmax denominator. The converter now encodes the original last-axis reductions and checks the pinned operators and tensor layouts before rewriting them.

The bridge APK passed 28 instrumentation tests and both fixtures on each device. Both installed hashes matched and both devices passed cleanup.

## Reduction-axis artifact

APK SHA-256 `b820cad4c36221cdac90a8a63662c25f258370185387fc50fb7517593f24f093`. The rewritten road ONNX passed 636 host original-reference comparisons. The Fold 3 ncnn road run reported no CPU fallback. The driver passed. Image and desire histories were exact again. The road graph failed tolerance in feature history and driving outputs on each seed.

| Seed | Driving values outside tolerance, before → after | Max driving absolute error, before → after | Feature values outside tolerance |
| --- | ---: | ---: | ---: |
| 42 | 1,636 → 867 | 174.15625 → 0.16268 | 50 |
| 7 | 1,606 → 825 | 140.68750 → 0.22493 | 21 |
| 123 | 1,364 → 664 | 149.50000 → 0.07363 | 11 |

The existing tolerance and no-fallback requirements remain in force. Road recurrent/replay, accelerator timing and combined sustained tests did not advance past the seed gate.

The Fold 3 passed 28 instrumentation tests and both 30-second full-chain fixtures. Its matched steady run averaged 100.004 Hz with 2.25 ms work p99, 12.39 ms complete-cycle p99 and 4.21 ms release-to-completion p99. It recorded 1,238 raw cycle misses, zero scheduled misses and zero skipped slots. Candidate45zy measured 12.26 ms complete-cycle p99. Controls code and metric definitions were unchanged in these candidates, and the complete-cycle timing failure remains open.

The Android 15 ARM64 emulator also passed 28 instrumentation tests and both 30-second fixtures. Both devices verified the installed app and test APK hashes and passed experiment isolation. Final cleanup receipts show stopped app/test processes, released cameras, no ADB forwarding and at least 30 consecutive seconds of NORMAL thermal status within the skin-temperature limit. The dedicated emulator shut down.
