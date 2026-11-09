# Candidate68: bounded Vulkan batching and camera latency controls

Candidate68 preserves all 795 original-reference comparisons, improves driver
inference, and reduces road p99 in this short comparison. Road median regresses.
Inference, camera freshness and controls complete-cycle timing still fail. This
is a diagnostic APK, not a qualified driving port. TX remains compiled out and
physical engagement disabled. Cooling optimization remains deferred; no thermal
protection was changed.

## Artifact and implementation

Main APK SHA256: `925b4b0ab7b40e50c6b19b3ca8f8f4fba2ef5eefd74fcc8b5d0588e2dff8a048` (1,147,201,452 bytes).

Test APK SHA256: `bcc20fc1b227cb03bde6cf0c6fbc153dce82eb718ffb50cf6fd6fad45b6e999c`.

Packaged libncnn SHA256: `3f7174f099617926eb56f15a9e994abdf3c22ab46c61c989c385d9479a94b271`.

Both APK hashes match their installed copies on the SM-F926B and dedicated
ARM64 Android15 emulator. Branch remains `codex/fold3-android-port`, based on
`7631d5248cc9efd10b962107aadacbb908a9f14d`, with the existing port work preserved.
Previous candidates and failures remain archived.

The pinned ncnn build now applies `ncnn-dispatch-batching.patch` reproducibly.
Its exact Qualcomm/Adreno660 gate increases the small stock pending-dispatch
limit from 32K to 256K groups, preserving larger stock limits and bounded
submit/wait handling. Other devices retain stock behavior. Shader arithmetic,
model graphs, original hashes, tolerances, precision boundaries and fallback
rejection are unchanged. Qualification remains tied to native library hashes.

New profiling separates internal submission counts/waits from final submission
wait and command-record CPU/other time. Internal waits remain in total latency;
profiling is diagnostic and excluded from the timing comparison below. The
host test compiles the actual patched gate and checks exact-device matching,
null/other names, other vendors and preservation of larger stock limits.

Camera2 adds diagnostic `manual-8ms-low-latency`: supported short manual exposure
plus advertised processing controls set before session creation. It preserves
session support, frame-duration/FPS, sensor capability and provider-crash guards.
No unadvertised vendor mode or timestamp substitution was introduced. Focused
checks cover profile rejection and invalid submission metrics.

## Accuracy and inference

**PASS:** all 795 comparisons, including three seeds, recurrent sequences and
pinned replay, at `rtol=1e-3`, `atol=1e-4`, no CPU fallback. Complete untapped
models pass before accelerator timing. The same graphs are used as candidate67.

Each timing row uses 100 unprofiled device samples. Values are milliseconds;
this is a matched short comparison, not a sustained performance qualification.

| Model / candidate | p50 | p99 | Maximum | Samples over 50 ms |
|---|---:|---:|---:|---:|
| Driver / 67 | 183.502 | 195.139 | 314.008 | 100/100 |
| Driver / 68 | 144.506 | 154.106 | 181.566 | 100/100 |
| Road / 67 | 303.182 | 472.634 | 479.241 | 100/100 |
| Road / 68 | 328.765 | 362.394 | 362.635 | 100/100 |

**FAIL:** both models exceed the complete 50 ms budget before camera latency.
Candidate68 is not promoted as an overall road improvement: its road median is
worse despite a lower tail. No relaxed tolerance, dropped-sample success or
profile timing is used to clear this gate.

The profiler confirms rough GPU score 9, applied dispatch limit 262144 and two
internal submissions per inference. Mean internal waits are 48.610 ms driver and
151.233 ms road; final waits 55.421/57.324 ms. Command recording excluding those
internal waits is 17.233/6.586 ms. Those components come from separate profiled
runs and are not substitutes for unprofiled wall latency. Largest measured
kernels remain driver conv70 (8.038 ms), div0 (7.758 ms), conv72 (6.767 ms), and road
conv88 (17.051 ms), conv114 (8.982 ms). GPU arithmetic remains a substantial cost.

## Controls and UI

No controls calculations, pacing policy, metric definitions or counters were
changed in this candidate. All runs use the same full-chain synthetic fixture
hash as candidate67, including cancellation and stale-input handling. They do
not use real camera perception or a physical vehicle.

| Device / fixture | Work p99 ms | Complete-cycle p99 ms | Mean Hz | Scheduled misses / skipped | Raw cycle misses | UI fps |
|---|---:|---:|---:|---:|---:|---:|
| fold-steady | 2.205 | 12.261 | 99.993 | 0 / 0 | 1271 | 29.66 |
| fold-faults | 2.137 | 12.338 | 100.000 | 0 / 0 | 1200 | 29.61 |
| emulator-steady | 1.533 | 11.735 | 100.012 | 0 / 0 | 1343 | 29.49 |
| emulator-faults | 1.260 | 11.256 | 99.999 | 0 / 0 | 1343 | 29.58 |

**PASS:** functional controls fixtures, work-time target and approximately 100 Hz;
UI near 30 fps. **FAIL:** unchanged complete-cycle budget. Fold steady/fault cycle
p99 changes from 12.137/12.441 ms in candidate67 to 12.261/12.338 ms in candidate68; no consistent controls
speedup is established. Raw misses remain visible.

Raw Fold timing analysis records scheduler-delay p99 1.269/1.818 ms, reporting
p99 0.372/0.357 ms and release-to-completion maxima 7.436/9.120 ms. Complete-cycle
latency correlates with adjacent work-time variation (0.886/0.833). Percentiles
are separate distributions and must not be added or subtracted to clear a gate.
Native absolute pacing, nonblocking latest inputs, heartbeat, skipped slots,
watchdogs and reporting-failure detection remain intact.

## Cameras

The isolated ultrawide trials use camera 2, 640x360, 12 seconds, direct YUV, clean
baselines and unchanged 50 ms accounting. These are camera-only timings before
inference and do not qualify three independent streams.

| Profile | Acquisition p99 ms | Publication p99 ms | Frames / publications | Publications over 50 ms |
|---|---:|---:|---:|---:|
| low-latency-yuv | 73.883 | 75.331 | 357 / 357 | 357 |
| manual-yuv | 60.617 | 62.102 | 357 / 357 | 357 |
| manual-low-latency-yuv | 60.145 | 61.741 | 357 / 357 | 357 |

**FAIL:** all three profiles exceed 50 ms on every publication. The combined
profile's 61.741 ms p99 is close to manual exposure alone 62.102 ms; one short run
does not establish a meaningful improvement over that control. The new profile
remains diagnostic.

Mode 37 three-stream baseline: 359 composite frames, 351 rear-preview frames and
358 front frames. Rear-preview acquisition p99 is 98.348 ms; publication 102.725 ms.
Front delivery p99 is 98.047 ms. Transport passes; **freshness fails**. **BLOCKED:**
independent rear identity, since composite transport is not physical-output or
repeatable optical evidence. No overlapping timestamp or image-difference test
is substituted for lens identity.

Emulator camera timing is separately failed: 342 frames, 340 rejected capture
timestamps, 338 capture-result timestamp errors, 2 published frames. Negative
reported ages are invalid-clock diagnostics, not fast camera performance.
Arrival timestamps do not replace bad sensor timestamps. Deterministic injected
fixture success does not qualify this camera route.

## Validation, cleanup and remaining gates

- **PASS:** host 257 tests; Java unit tests; both APK builds; instrumentation 30/30
  on each device; lifecycle recovery and emulator display changes.
- **PASS:** Hyundai behavior 13 run/2 skipped; Hyundai safety 1899 run/198 skipped.
- **PASS:** emulator MetaDrive fault test, 2611 packets, 920 inactive-after-fault,
  no longitudinal-active or below-minimum-active samples, maximum lateral
  error 0.488 m. Perception and plant are synthetic; no road qualification.
- **PASS:** all 16 experiment isolation receipts, stable installed APKs/provider
  processes, camera release, process stop and forwarding cleanup. Final Fold
  NORMAL baseline 32.249 continuous seconds at 27.4 C; emulator 30.776 seconds at 30.8 C.
  Owned emulator stopped and disappeared from ADB; the other attached phone
  was not used.
- **PASS:** 83 immutable artifact hashes and 167 source hashes; `git diff --check`.
- **BLOCKED:** repository-wide Ripwire quality delta did not respond after
  several minutes; waiting cell terminated. No clean quality-delta claim.
- **NOT RUN:** qualified combined 30-second, 2-minute and 30-minute perception
  progression because accuracy passes but speed, freshness and complete-cycle
  gates fail. The 30-second synthetic controls fixtures are separate checks.
- **BLOCKED:** independent rear-lens proof, mounted calibration, passive Panda
  firmware/fingerprint/harness and physical vehicle behavior. Phone is unmounted;
  Panda disconnected. TX and engagement remain disabled regardless of bench
  results. Thermal pause, cooldown and abort safeguards remain active.

## Reproduce and inspect

Use the pinned local toolchains and assets from repository root:

```sh
.venv/bin/python android/scripts/build_ncnn.py
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  .venv/bin/python -m unittest discover -s android/tests
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
ANDROID_HOME=/Users/sergios/Library/Android/sdk \
FOLD3_BUILD_PYTHON=/Users/sergios/sunnypilot/.venv/bin/python \
  android/gradlew -p android --offline --console plain \
  -Pfold3CameraCompat=true -Pfold3Qnn=true \
  -Pfold3MnnPatched=true -Pfold3NcnnPatched=true \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
```

The configured Gradle daemon uses its own Java toolchain; the launcher variable
alone does not prove daemon Java version. APKs are archived before removal of
completed regenerable Python/asset copies to restore emulator disk space. The
build succeeds; original assets, archived APKs and older results remain intact.

[Installable APK](../../build/candidates/candidate68-dispatch-batching/app-debug.apk),
[artifact identities](../../build/candidates/candidate68-dispatch-batching/artifacts.json),
[validation summary](../../build/candidates/candidate68-dispatch-batching/validation-summary.json),
[original-reference and timing report](../../build/candidates/candidate68-dispatch-batching/fold-models.json),
[GPU profile](../../build/candidates/candidate68-dispatch-batching/fold-profile.json),
[raw controls analysis](../../build/candidates/candidate68-dispatch-batching/controls-timing-analysis.json).

Exact argument arrays for every experiment, archive/install/hash receipts,
source snapshots, native build identities and raw reports remain in the
[candidate directory](../../build/candidates/candidate68-dispatch-batching).
Run device commands through `android/scripts/experiment.py` sequentially per serial;
never reuse receipts for a rebuilt APK. The campaign owns only its dedicated AVD.
