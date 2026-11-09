# Fold 3 continuation on the Chromebox

Workspace: `/home/dey/sunnypilot-fold3`, repository `repo`, branch
`codex/fold3-android-port`, inspected HEAD `7631d5248cc9efd10b962107aadacbb908a9f14d`.
The dirty workspace is intentional and includes the existing port. Do not reset it.

TX and engagement remain disabled. Keep the original models, goldens,
`rtol=1e-3`, `atol=1e-4`, all 795 comparisons and 50 ms sensor-to-output gate.
Host/emulator results cannot qualify physical accelerators or vehicle readiness.

## Tools and environment

```sh
ssh chromebox
cd /home/dey/sunnypilot-fold3
. ./activate-lab.sh
. ./activate-native.sh
cd repo
python -m unittest discover -s android/tests
cd android
./gradlew --version
```

The full Linux suite currently reports 280 tests, 4 failures, 2 errors and 27
skips. These remaining failures concern x86 versus original ARM FP16 reference
behavior and ARM-specific ORT NhwcFusedConv artifacts. Do not regenerate goldens
or relax tolerances. The Mac suite passes 280 tests with 3 Linux-only skips.

Installed user-owned tools:

- `converter-venv`: Python 3.10, QAIRT 2.47 converter dependencies;
  SDK `sdk/qairt/2.47.0.260601`, private loader libraries `sysroot`.
- `litert-venv`: Python 3.12, LiteRT 2.3.0 conversion dependencies and ORT.
- `host-venv`: Python 3.12, NumPy 2.5.3, ONNX 1.23.2, ORT 1.31.1,
  testing/build dependencies and editable pinned local packages.
- `tools/jdk-21`, `tools/android-sdk`: Android 35 platform/build-tools,
  platform-tools, NDK 28.2.13676358, CMake 3.22.1, emulator and Android 35
  Google APIs x86_64 image. SDK licenses copied from the already accepted Mac set.
- `tools/bin`: private GCC/G++, Clang, Make, ripgrep and binutils;
  no globally exported private libc or system package/service changes.

Gradle 8.9 starts, Android ARM64 cross-compilation succeeds and native Linux
IPC/parameter bindings import. `/dev/kvm` is currently not writable; Linux
hardware-accelerated emulator operation remains blocked. No VM was started.
The Android 15 ARM64 Mac emulator was cleaned and stopped after validation.

Receipts: `lab-tools.json`, `host-bindings.json`, `native-toolchain.json`,
`converter-host.json`, dependency freezes and install/build logs at workspace root.
Reproducible setup scripts are in `repo/android/scripts/setup_chromebox_tools.py`,
`setup_chromebox_native.py`, `build_chromebox_host.py` and `setup_snpe_linux.py`.
They use isolated environments and private package extraction.

## SNPE diagnosis and reproduction

First road dense layer: ordered eight-lane FP16 accumulation reproduces all
2,048 captured original outputs exactly. SNPE's widened FP32 calculation has
872 tolerance mismatches. Casting its final result to half still leaves 901.
A final cast therefore does not repair the demonstrated arithmetic difference.
This needs an accurate GPU operator or validated split execution; no SNPE
backend is qualified. CPU execution here is explicitly diagnostic only.

```sh
cd /home/dey/sunnypilot-fold3
converter-venv/bin/python repo/android/scripts/setup_snpe_linux.py \
  --root "$PWD" --check-only
converter-venv/bin/python repo/android/scripts/validate_snpe_host.py \
  --root "$PWD" --models work/snpe-dense-isolated-dlc-r2 \
  --fixtures work/snpe-dense-isolated-original-capture/fixtures \
  --output work/snpe-dense-isolated-results-NEW
```

Use fresh output names; preserve previous failures. `work/snpe-vision-trace*`
and `work/snpe-vision-dense-trace*` contain original prefixes, converted DLCs,
raw outputs, original-node mappings and trace results. Original reference
captures also live under
`repo/android/build/candidates/candidate74-runtime/follow-up/`.

To recreate the isolated probe on the original Mac reference environment:

```sh
.venv/bin/python android/scripts/prepare_snpe_dense_probe.py \
  --source android/build/candidates/candidate74-runtime/snpe-road-float-transport/vision.onnx \
  --mean-reference android/build/candidates/candidate74-runtime/follow-up/snpe-vision-trace/fixtures/trace_3/reference.npz \
  --linear-reference android/build/candidates/candidate74-runtime/follow-up/snpe-vision-dense-trace/fixtures/trace_0/reference.npz \
  --output android/build/candidates/candidate74-runtime/follow-up/snpe-dense-isolated-NEW
```

## LiteRT and next gates

The explicit-half convolution with FP32 external transport passes emulator GPU
parity exactly; all three primitive probes pass with GPU-only assignment.
`prepare_litert_probes.py --explicit-half-conv --float-transport` selects the
repaired primitive. This does not prove full-road conversion or intermediate
accumulation. Reject unsupported geometry and silent CPU fallback.

Before device experiments, use `experiment.py` for process/camera/forwarding
cleanup and the established 30-second NORMAL/skin<=32 C gate. Keep thermal
protection; cooling optimization remains deferred. Fold 3 tests are deferred by
request. No candidate74 APK or new physical timing result exists.

The installed emulator candidate73 APK was unchanged:
`d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`.
Its NativeBridge instrumentation passes 8 tests. Standalone runtime probes do
not qualify that APK's inference backend.

Full road parity, 50 ms freshness, controls cycle timing, rear identity, mounted
calibration and disconnected Panda/vehicle qualification remain open. AI
Superstack lab receives a documentation pointer only; no services, workers or
production configuration are activated.
