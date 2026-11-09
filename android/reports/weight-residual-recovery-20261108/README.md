# Fold 3 convolution residual recovery — 8 November 2026

Branch: `codex/fold3-android-port`. Candidate65 and earlier evidence are preserved.
Vehicle TX remains compiled out and physical engagement disabled. Cooling
optimization is deferred; the existing thermal safeguards remain active.

## Change and numerical checks

Candidate66 changes the three profiled packed-convolution Vulkan shaders. When
all four weight lanes have a zero low significand split, the software FMA product
residual omits two exactly zero products and their additions. Other weights keep
the original calculation. Product rounding, accumulation order, correction tails,
midpoint handling, FP16 boundaries and graph conversion remain unchanged. The
runtime predicate validates the actual loaded weights; it does not assume every
converted weight is half precision.

The original-model audit finds 46 driver and 60 road convolutions whose constant
weights are exactly representable as FP16. The host regression compiles the actual
shader correction into C with contraction disabled. It checks every finite FP16
weight with varied FP32 operands and cancellation cases against native `fmaf`,
and 200,000 general FP32 triples against the preserved implementation. All 251
host tests pass. Host checks alone do not qualify Vulkan execution.

Original models, converted graphs, goldens and tolerances remain unchanged:
`rtol=1e-3`, `atol=1e-4`, 795 terminal comparisons across three seeds, recurrent
sequences and the pinned replay. Complete untapped model accuracy precedes timing;
profiling cannot create qualification records. The 50 ms maximum capture-to-model
publication budget, 20 ms rear skew limit and existing controls cycle counters
remain authoritative.

## Reproduction and artifacts

[APK](../../build/candidates/candidate66-weight-tail/app-debug.apk) ·
[artifact and source identities](../../build/candidates/candidate66-weight-tail/artifacts.json) ·
[original weight audit](../../build/candidates/candidate66-weight-tail/weight-audit.json) ·
[commands and raw evidence](../../build/candidates/candidate66-weight-tail/) ·
[machine-readable validation summary](../../build/candidates/candidate66-weight-tail/validation-summary.json)

APK SHA-256: `ab383423a396c3903b18a39938542ec0999cd28da1b36f459fdb8ed495701ecd`.
The installed hash is verified on the physical Fold 3 and ARM64 Android 15 emulator.

Use the pinned build commands in the
[previous report](../gpu-npu-recovery-20261107/README.md#reproduce).
After building and installing the matching APK, run the isolated campaigns:

```sh
.venv/bin/python android/build/candidates/candidate66-weight-tail/run_validation.py \
  fold models profile instrumentation controls-steady controls-faults lifecycle
.venv/bin/python android/build/candidates/candidate66-weight-tail/run_validation.py \
  emulator instrumentation controls-faults lifecycle controls-steady metadrive
.venv/bin/python android/build/candidates/candidate66-weight-tail/run_cameras.py fold
.venv/bin/python android/build/candidates/candidate66-weight-tail/run_cameras.py emulator
```

Each experiment uses the existing device lease, process/camera/forwarding cleanup,
and at least 30 continuous seconds of NORMAL thermal status with skin at or below
32°C. The final cleanup scripts verify the installed hash and released resources.
Controls fixtures additionally retain raw timing batches for analysis; no control
calculations, scheduling policy or timing definitions changed in this candidate.

## Results

### Models and GPU profile

**PASS — accuracy:** all 795 original-reference comparisons pass on the physical
Fold 3 (159 driver, 636 road), including all three seeds, recurrent sequences and
pinned replay. Both complete models run through Vulkan with no CPU fallback or
unexpected non-finite outputs. Graph and original-reference identities are
unchanged from candidate65.

**FAIL — inference timing:** all 100 measured samples of each model exceed 50 ms.
All samples count, including the first. These isolated inference failures already
preclude the sensor-to-published-output deadline, even before camera acquisition.

| Model | Candidate65 p50 / p99 / max ms | Candidate66 p50 / p99 / max ms |
|---|---:|---:|
| Driver | 187.602 / 198.759 / 290.332 | 188.473 / 202.939 / 243.463 |
| Road | 303.183 / 430.381 / 458.114 | 320.131 / 383.425 / 430.005 |

The graph, original references, device, backend and 100-sample workload match.
These are individual device runs with runtime variability, not an isolated causal
speedup. Driver median/p99 worsen by 0.46%/2.10%; road median worsens by 5.59%
while p99 improves by 10.91%. The shortcut is **not accepted as an overall
performance improvement**, and candidate66 remains diagnostic and unqualified.
Candidate65 and earlier accurate baselines remain preserved.

Separate diagnostic GPU-event profiling places road `conv_88` at 43.430 ms;
driver `conv_70`, `conv_72` and `div_0` take 11.847, 10.564 and 10.272 ms.
The new predicate did not remove the dominant convolution cost. Profiling results
are excluded from acceptance timing. No NPU probes were rerun for this shader-only
candidate; candidate65's HTP failures remain historical evidence for that artifact,
not candidate66 qualification.

### Controls

Same full-chain fixture and timing schema 2; no controls code, calculations,
counters, thresholds or pacing definitions changed in candidate66.

| Device/workload | Work p99 ms | Complete cycle p99 ms | Mean Hz | Scheduled misses / skipped slots | Raw cycle misses |
|---|---:|---:|---:|---:|---:|
| Candidate65 Fold steady | 2.282 | 12.266 | 99.994 | 0 / 0 | 1155 |
| Candidate66 Fold steady | 2.256 | 12.229 | 99.996 | 0 / 0 | 1227 |
| Candidate66 Fold cancellation/stale | 2.173 | 12.608 | 100.000 | 0 / 0 | 1300 |
| Candidate66 emulator software GPU steady | 33.153 | 52.628 | 77.676 | 327 / 606 | 1060 |
| Candidate66 emulator software GPU faults | 29.851 | 52.555 | 79.283 | 314 / 554 | 1079 |

**PASS — Fold fixture work and scheduled completion. FAIL — legacy complete-cycle
gate.** Factory-ACC cancellation and stale-CAN rejection pass without unsafe
outputs. Raylib averages 29.68/29.57 fps on the Fold fixtures. These are synthetic
full-chain inputs, not concurrent qualified camera inference.

[Raw timing analysis](../../build/candidates/candidate66-weight-tail/controls-timing-analysis.json)
retains every available timing row through teardown (2707 steady / 2761 fault
rows, slightly later than the report snapshots). Fold scheduled completion maxima
are 7.591/8.084 ms, while raw complete-cycle maxima are 15.322/15.700 ms. Consecutive
active-work changes correlate with complete-cycle duration at 0.890/0.841. Steady
p99 input/computation/publishing/extension work is 1.412/0.985/0.821/0.587 ms;
reporting is 0.389 ms, and scheduler delay is 1.218 ms. This supports variation in
work and wakeup timing as contributors; it does not establish a new fix or permit
redefining the failed legacy metric. Raw misses are preserved.

### Cameras

- **PASS — ultrawide transport; FAIL — 50 ms freshness:** public Camera2 640×360
  direct YUV delivers 355 frames, no rejected capture timestamps; acquisition p99
  86.121 ms, delivery p99 87.333 ms. All 355 publications exceed 50 ms before any
  model execution. Candidate65 delivery p99 was 83.615 ms.
- **PASS — three-stream transport; BLOCKED — independent rear identity; FAIL —
  freshness:** Samsung mode37 delivers 354 rear-preview, 359 composite and 359
  front frames. Preview acquisition p99 is 107.598 ms, readback 6.211 ms, sink
  delivery 111.789 ms; front delivery is 100.982 ms. There are 243 preview sink
  deliveries beyond the existing 100 ms stale threshold. No stream is qualified
  for perception and no independent rear proof is inferred from different images
  or overlapping timestamps. Candidate65 preview/front delivery p99 was
  116.886/97.035 ms.
- **FAIL — emulator camera timestamp contract:** 325 acquired frames, 316 rejected
  future capture timestamps and only 9 publications. Capture-result timestamp
  errors number 310. Invalid timestamps are rejected, not replaced by arrival
  time. Deterministic injected input success cannot clear this provider fault.

All Fold camera experiments complete their requested duration, drain image
callbacks and confirm teardown. The provider remains alive with the same PID.
The road deadline still starts at the earlier current rear capture and rear
synchronization remains limited to 20 ms. Transport does not establish acceptable
capture latency, independent sensor geometry, mounted calibration or perception.

### Host, Android and simulator validation

**PASS:** all 251 host tests; native build, Gradle debug/test APK build and Java
unit tests; 30 instrumentation tests on each device. Instrumentation and fixtures
exercise lifecycle/surface recovery, emulator display changes, supervision, IPC,
numerical/thermal faults, USB rejection, cancellation and stale inputs. All 163
archived product-source hashes match the tested worktree. Original/converted
models and goldens are unchanged.

**FAIL — initial emulator MetaDrive run:** the world terminates `out_of_road`
(maximum lateral error 6.912 m), despite cancellation producing inactive outputs.
The original failure, its packets and cleanup receipt remain intact. This test
uses synthetic ground-truth perception, driver state and calibration, and a
damped steering plant; it cannot establish visual or physical Hyundai behavior.

**PASS — same-APK host-GPU MetaDrive repeat:** 30.008 seconds, 2593 packets,
maximum lateral error 0.574 m, 896 inactive-after-fault samples, zero longitudinal
activity and no steering below the Hyundai minimum. The initial failure remains
preserved. This is a synthetic simulator result, not perception or vehicle proof.

The initial emulator records SwiftShader; the repeat uses `-gpu host` and records
the Apple M4 renderer. Its steady fixture improves from 77.676 to 99.732 Hz,
work p99 from 33.153 to 4.100 ms, and complete-cycle p99 from 52.628 to 14.054 ms.
The repeat still has 3 scheduled misses, 8 skipped slots and 1319 raw cycle misses.
The fault repeat has work/cycle p99 2.628/12.976 ms, 99.773 Hz, 3 scheduled misses,
6 skipped slots and 1294 raw cycle misses. **Timing remains failed.** Raylib mean
FPS is 12.81/25.21 in the host-GPU fixtures, so the emulator UI target is not met.
This is one renderer/environment comparison, not a controlled Fold accelerator
speedup or a controls code fix. The repeat's commands, raw timings, renderer
identity and reports are retained under
[emulator-host-gpu-repeat](../../build/candidates/candidate66-weight-tail/emulator-host-gpu-repeat/).

### Cleanup and qualification

All 17 experiment-isolation receipts (14 main, 3 renderer-repeat) pass, including at least 30
continuous seconds of NORMAL thermal status with skin ≤32°C before each run.
Both installed hashes match the final APK. Main-campaign cleanup verifies stopped
app/test processes, released cameras, no experiment forwarding/reverse mappings,
unchanged provider PIDs and a fresh cool baseline. Both owned emulator instances
are shut down. The host-GPU repeat's immediate shutdown receipt initially observed
ADB's transient `offline` registration; that failure is preserved. Follow-up
`shutdown-verified.json` confirms the serial has disappeared and no owned emulator
process remains. The helper now waits for deregistration before judging shutdown.
Only regenerable host build-asset copies were removed; source, original models,
APKs, prior candidates and failed evidence remain preserved.

The broad quality scan compares the existing uncommitted Android port with HEAD;
its retained complexity/size findings do not constitute a clean quality-gate
claim. This iteration does not refactor unrelated existing port code.

**BLOCKED:** 30-second, two-minute and 30-minute combined perception qualification
cannot advance past failed 50 ms inference/camera and complete-cycle gates.
Independent rear proof, mounted optical/vehicle calibration, real Panda firmware,
passive fingerprint/harness checks and vehicle behavior remain separate physical
gates. Cooling optimization remains deferred. TX stays compiled out and physical
engagement disabled regardless of diagnostic or simulator results.
