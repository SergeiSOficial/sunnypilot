# Candidate69: exact normalization and short-exposure diagnostics

All 795 original-reference comparisons pass at rtol=1e-3, atol=1e-4, with no CPU fallback. Inference speed, camera freshness and controls complete-cycle timing still fail. TX is compiled out and engagement disabled. This APK is diagnostic and is not promoted as a performance improvement.

The reproducible byte-normalization shader preserves exact FP32 division for integral values 0..255 and strict division elsewhere. The host check compiles the actual helper and exhausts all 256 values, signed zero and fallback edges. Candidate69 applies this only to the scalar BinaryOp shader. Subsequent inspection shows Vulkan upload packs divisible input dimensions into pack4 independently of the CPU packing option; candidate70 addresses that missing coverage. Driver div0 remains 7.698 ms in the separate diagnostic GPU profile, versus 7.758 ms in candidate68. Profiling is excluded from timing acceptance.

Camera2 adds the supported diagnostic manual-2ms-low-latency profile, retaining exposure/sensitivity/frame-duration/capability/session checks, vendor exclusions and capture timestamp validation. It changes no live camera default and does not qualify bench image quality or mounting geometry.

## Artifact identities

APK SHA256 `16d07d62d1dd70fc2c9beacbf0c9fa2308f26686cbc1d32a04e1dc35ac7b168a`. Test APK `bcc20fc1b227cb03bde6cf0c6fbc153dce82eb718ffb50cf6fd6fad45b6e999c`. Packaged libncnn `d70ff440b5546fc3d5b2a135801e0477b377451161acb7c130b6accac0168e83`. Both APK hashes are verified on the Fold3 and dedicated ARM64 Android15 emulator. Original models and converted graphs are unchanged from candidate68.

## Matched short inference comparison

100 unprofiled samples per model; all 100 exceed 50 ms in every row. Values are milliseconds. These are isolated work times, already exceeding the sensor-to-output budget before camera delay.

| Candidate / model | p50 | p99 | Maximum |
|---|---:|---:|---:|
| 68 driver | 144.506 | 154.106 | 181.566 |
| 69 driver | 144.227 | 153.467 | 176.957 |
| 68 road | 328.765 | 362.394 | 362.635 |
| 69 road | 343.210 | 362.294 | 362.994 |

No material driver improvement is established; road median regresses. No accuracy tolerance or timing definition is relaxed.

## Controls fixtures

Controls production code and metrics are unchanged. The matched synthetic full-chain fixture has SHA256 b5ea99327f0ffc121e5556d29b32d4ce8e6e855d14518f46a3f060ea38721f99. It is not camera inference or a physical vehicle.

| Device / fixture | Work p99 ms | Complete-cycle p99 ms | Mean Hz | Scheduled misses / skipped | Raw cycle misses | UI fps |
|---|---:|---:|---:|---:|---:|---:|
| fold-steady | 2.183 | 12.556 | 100.001 | 0 / 0 | 1317 | 29.70 |
| fold-faults | 2.232 | 11.794 | 99.997 | 0 / 0 | 1288 | 29.60 |
| emulator-steady | 1.461 | 11.548 | 100.007 | 0 / 0 | 1289 | 29.54 |
| emulator-faults | 1.301 | 11.723 | 99.999 | 0 / 0 | 1306 | 29.58 |

Functional checks pass with zero unsafe outputs. Work and scheduled completion meet their budgets; the unchanged raw complete-cycle gate fails. Fold steady scheduler-delay p99 is 1.559 ms, active CPU 3.085 ms and active off-CPU 0.356 ms; reporting p99 0.386 ms. Tail analysis remains diagnostic and preserves raw intervals and counters. No controls performance gain is claimed.

## Cameras

Isolated camera2, direct YUV 640x360, 12 seconds per profile. These are camera-only results, before inference.

| Profile | Acquisition p99 ms | Publication p99 ms | Publications | Over 50 ms |
|---|---:|---:|---:|---:|
| low-latency-yuv | 72.355 | 73.903 | 357 | 357 |
| manual-yuv | 60.049 | 61.894 | 357 | 357 |
| manual-low-latency-yuv | 60.170 | 61.675 | 357 | 357 |
| manual-2ms-low-latency-yuv | 53.784 | 55.873 | 358 | 298 |

The 2 ms exposure improves the isolated ultrawide result but fails the unchanged 50 ms maximum. It is not proof of simultaneous independent cameras. No invalid capture times were accepted in these physical trials.

Three-stream Samsung mode37 baseline transports 359 composite, 350 rear-preview and 359 front frames. Rear acquisition p99 is 101.562 ms, publication 107.362 ms and front delivery 100.441 ms. Freshness fails; independent rear identity remains blocked. Composite transport is not physical-output assignment or repeatable optical proof.

Emulator camera timing fails separately: all 343 capture timestamps are rejected as invalid/future, zero frames published. Negative diagnostic ages are not performance results. Arrival timestamps do not replace invalid sensor timestamps.

## Validation and cleanup

- PASS: 258 host tests; Gradle/Java checks; emulator instrumentation 30/30; lifecycle recovery and emulator display changes. Fold instrumentation FAIL: 30 run / 1 failure, a case-sensitive diagnostic-text assertion after correct front-profile rejection. This failed receipt is retained and is not counted as passed.
- PASS: Hyundai behavior 13 run / 2 skipped; safety 1899 run / 198 skipped in the clean final log. The earlier wrong-SDK and interrupted overlapping safety log is retained and not counted as a passing run.
- PASS: emulator MetaDrive stale/cancel handling: 2629 packets, 919 inactive-after-fault, zero longitudinal/below-minimum active samples, maximum lateral error 0.5073 m. Synthetic perception and plant cannot qualify a road vehicle.
- PASS: all experiment isolation and final cleanup receipts; stopped processes, released cameras, removed forwards, stable APK/provider identities. Final Fold cool NORMAL period 32.301 s at 27.9 C; emulator 30.776 s at 30.8 C; owned emulator shut down. The other attached phone was untouched.
- PASS: 83 immutable artifacts and 169 source files audited before the subsequent candidate70 source change; the frozen 169-file candidate69 source snapshot also matches independently.
- BLOCKED: Ripwire quality delta did not respond within the bounded attempt; no clean result is claimed.
- NOT RUN: qualified combined 30-second, two-minute and 30-minute perception progression because isolated speed/freshness and complete-cycle gates fail. The synthetic 30-second controls fixtures are separate checks.
- BLOCKED: independent rear proof, mounted calibration and disconnected Panda/vehicle firmware, fingerprint, harness and behavior gates. Thermal protections remain active; cooling optimization is deferred.

The first APK build ran out of disk; a subsequent task-exclusion attempt failed Gradle provider resolution. Both logs remain. The successful build resumed verified completed Python asset tasks with the candidate-local init script. An initial archive check incorrectly compared stripped and unstripped native libraries; the corrected audit verifies both stages. Source/model assets and all frozen candidates were preserved; only reproducible completed staging copies were reclaimed.

## Evidence and reproduction

[Compact receipts](../../build/candidates/candidate69-exact-normalization/validation-summary.json), [artifact manifest](../../build/candidates/candidate69-exact-normalization/artifacts.json), [full model results](../../build/candidates/candidate69-exact-normalization/fold-models.json), [controls analysis](../../build/candidates/candidate69-exact-normalization/controls-timing-analysis.json), [frozen source](../../build/candidates/candidate69-exact-normalization/source).

Use the standard pinned build commands documented in the [candidate68 report](../dispatch-batching-20261108/README.md#reproduce-and-inspect), with sufficient free disk for Python staging. The exact device commands and isolation receipts are in this candidate directory. To reproduce candidate69 after later changes, restore its frozen source in an isolated checkout; current working source may target a later candidate.
