# Candidate67: GPU weight specialization and absolute controls pacing

Candidate67 is **diagnostic and unqualified**. Accuracy passes; inference speed,
camera freshness and the legacy controls complete-cycle gate still fail. Vehicle
TX remains compiled out and physical engagement disabled. Cooling optimization
is deferred; thermal safeguards and experiment cooldown remain active.

## Artifact and implementation

[Installable debug APK](../../build/candidates/candidate67-specialized-tail/app-debug.apk)
SHA-256: `6463d0d0603776f8ac7ea48f39618bc795b0292f1717a5d8e0459f29ad90991a`.
This exact main APK and its test APK were installed and hash-verified on both the
SM-F926B and dedicated ARM64 Android15 emulator. Candidate65, candidate66 and
prior evidence remain preserved.

[Artifact manifest](../../build/candidates/candidate67-specialized-tail/artifacts.json)
binds original/converted models, provider libraries, native builds and 165 archived
product source files. Its 83 immutable artifact hashes and all archived/current
source hashes match in the [audit](../../build/candidates/candidate67-specialized-tail/artifact-audit.json).
Mutable experiment logs are deliberately outside that immutable artifact manifest.

- The reproducible ncnn patch scans actual packed FP32 weights at pipeline
  creation. A specialization constant removes the low weight-residual arithmetic
  only when every logical packed value is finite and its low 12 bits are zero.
  Uninitialized channel-stride alignment padding is excluded; real packed lane
  padding is included. Unsupported or unmatched weights retain the accurate
  general arithmetic. Three convolution shaders share the checked condition.
- Android controls now wait directly against the absolute monotonic release with
  native `clock_nanosleep`. Interrupted waits retry the same deadline; failures
  propagate. The call releases the GIL. One start-of-slot pacing loop, skipped
  slots, cancellation, stale-input checks, watchdogs and existing raw counters
  remain. Other platforms retain their previous pacing.
- Tests exercise both arithmetic branches against the original exact-product
  reference, shader specialization mapping, absolute-wait interruption/errors,
  and early/missing inputs, delayed wakeups, overruns and reporting contention.

## Accuracy and accelerator timing

**PASS:** all **795** original-reference comparisons (159 driver / 636 road),
three seeds, recurrent sequences and pinned 60-frame replay at unchanged
`rtol=1e-3`, `atol=1e-4`. Both complete untapped models run through Vulkan without
CPU fallback or unexpected non-finite outputs.

**FAIL:** all 100 isolated timing samples for each model exceed 50 ms. These
failures already prevent the sensor-to-published-output deadline before capture
and preprocessing are added. First samples are retained.

| Model | Candidate66 p50 / p99 / max ms | Candidate67 p50 / p99 / max ms |
|---|---:|---:|
| Driver | 188.473 / 202.939 / 243.463 | 183.502 / 195.139 / 314.008 |
| Road | 320.131 / 383.425 / 430.005 | 303.182 / 472.634 / 479.241 |

The device, graphs, references, backend and 100-sample workloads match. These are
individual runs with variability, not proof of an isolated causal speedup. Driver
p99 improves, but road p99 regresses and both maxima worsen. **Do not promote this
candidate as an overall performance improvement.**

Separate diagnostic GPU-event profiling reports road `conv_88` at 35.943 ms,
driver `conv_70` at 9.141 ms and driver `div_0` at 8.620 ms. The command-recording
phase averages 323.282 ms road / 110.477 ms driver; final submission/wait averages
52.484 / 37.986 ms. Recording can include internal synchronization and must not
be equated with pure CPU computation. These profiled runs are excluded from
acceptance timing. HTP was not rerun; previous primitive failures are retained
historical evidence, not current-artifact qualification.

## Controls, UI and simulator

Same full-chain fixtures and timing schema 2. Functional fixture success does
not clear the legacy complete-cycle gate.

| Device / workload | Work p99 ms | Complete cycle p99 ms | Mean Hz | Scheduled misses / skipped | Raw cycle misses | UI fps |
|---|---:|---:|---:|---:|---:|---:|
| Candidate66 Fold steady | 2.256 | 12.229 | 99.996 | 0 / 0 | 1227 | 29.68 |
| Candidate67 Fold steady | 2.141 | 12.137 | 99.999 | 0 / 0 | 1220 | 29.71 |
| Candidate67 Fold faults | 2.219 | 12.441 | 99.999 | 0 / 0 | 1269 | 29.61 |
| Candidate67 emulator steady | 1.113 | 11.969 | 100.002 | 0 / 0 | 1285 | 29.61 |
| Candidate67 emulator faults | 1.871 | 11.962 | 99.994 | 0 / 0 | 1348 | 29.53 |

**PASS:** work and scheduled completion in these synthetic fixtures, factory-ACC
cancellation and stale-input rejection. **FAIL:** unchanged 10 ms complete-cycle
gate. These are not concurrent qualified camera/model workloads.

[Raw timing analysis](../../build/candidates/candidate67-specialized-tail/controls-timing-analysis.json)
retains teardown rows beyond report snapshots. Fold scheduled completion maxima
are 7.938 / 7.385 ms, while complete-cycle maxima are 16.367 / 14.823 ms.
Scheduler-delay p99 is 1.355 / 1.282 ms and reporting p99 0.341 / 0.374 ms.
Consecutive work changes correlate with complete-cycle duration at 0.858 / 0.881.
Native absolute waiting did not eliminate the remaining work/wakeup variation.
No metric was renamed or redefined to clear this failure.

**PASS:** emulator MetaDrive stale-input/cancellation fixture: 2625 packets,
920 inactive samples after fault, zero longitudinal-active samples, zero active
samples below steering minimum, maximum lateral error 0.460 m. This uses synthetic
perception and a synthetic steering plant; it cannot qualify camera inference or
physical Hyundai behavior. The emulator uses host GPU rendering; its timing is
not Fold3 accelerator evidence.

## Camera comparison

Same isolated 640×360 ultrawide, advertised low-latency regular-session profile,
12-second runs; mode37 remains the three-stream transport baseline.

| Route | Frames acquired | Acquisition p99 ms | Delivery p99 ms | Result |
|---|---:|---:|---:|---|
| Ultrawide direct YUV | 357 | 81.909 | 83.265 | FAIL 50 ms |
| Ultrawide PRIVATE/GLES | 357 | 85.332 | 89.323 | FAIL 50 ms |
| Mode37 rear preview | 353 | 104.454 | 110.632 | FAIL 50 ms; identity BLOCKED |
| Mode37 front | 359 | See raw report | 105.025 | FAIL 50 ms |

Direct YUV is faster in this matched isolated comparison. All 357 YUV
publications exceed 50 ms. PRIVATE publishes 356 frames and records 357 late
publication attempts; raw counters are retained without concealing the difference.
Mode37 also delivers 359 composite frames, with preview readback p99 6.663 ms and
150 preview sink deliveries exceeding 100 ms. Three-stream transport passes;
**independent rear identity and geometry remain unqualified**. Shared timestamps,
different pixels or successful transport do not establish distinct rear sensors.

**FAIL:** emulator camera provider timestamp contract: all 341 acquired frames
have future capture timestamps and are rejected; zero publications. Capture-result
timestamp errors also total 341. Arrival time is not substituted. Deterministic
fixture success cannot clear this provider fault.

## Checks, isolation and remaining gates

- **PASS:** host suite 255 tests; Gradle Java tests and both APK builds; Hyundai
  behavior 13 tests run / 2 skipped; Hyundai safety 1899 run / 198 skipped.
- **PASS:** instrumentation 30/30 on each device, including lifecycle,
  supervision, numerical/stale-input/thermal and USB rejection fixtures.
  Lifecycle recovery passes on both; display override is exercised on emulator.
  A physical hinge transition was not performed.
- **PASS:** final installed APK hashes, process/camera release, no experiment
  forwarding/reverses, stable camera providers and 30 continuous seconds of
  NORMAL thermal status at skin temperature ≤32°C. Owned emulator stopped.
- **BLOCKED:** repository-wide Ripwire quality delta returned no result after
  several minutes; no clean static-quality claim is made. `git diff --check`
  passes. Existing dirty port work is preserved.
- **BLOCKED:** combined qualified 30-second / two-minute / 30-minute progression:
  preceding accuracy passes, but inference, camera freshness, identity and legacy
  controls timing gates do not. The 30-second synthetic controls fixtures above
  are not combined perception qualification.
- **BLOCKED:** optical lens-cover proof, mounted calibration, connected Panda
  firmware/passive fingerprint/harness and physical vehicle checks. No TX enabled.

An initial emulator launch failed for host disk capacity; its logs are retained.
Only regenerable completed build asset copies were removed before a successful
host-GPU retry. Original models, frozen APKs and prior failures remain. The first
orchestrator's eventual nonzero exit retains that launch failure; the independent
retry has its own installation, test, cleanup and shutdown receipts. An initial
archive check expecting Python source instead of Chaquopy bytecode also failed
before installation; the corrected check verifies normalized Python3.12 code
objects against source. Neither failure is presented as a passing test.

## Reproduction and evidence

From repository root, using the existing pinned toolchains and model assets:

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

The existing Gradle daemon uses its configured Java toolchain; the launcher
variable above does not establish the daemon's Java version. Device experiments
must be sequential per serial and use `android/scripts/experiment.py`.
Exact argument arrays for each test are archived as `*-command.json`; orchestration
scripts, all failed receipts and complete raw reports are retained in the
[candidate67 directory](../../build/candidates/candidate67-specialized-tail).
Use its [validation summary](../../build/candidates/candidate67-specialized-tail/validation-summary.json),
[Fold model report](../../build/candidates/candidate67-specialized-tail/fold-models.json)
and [GPU profile](../../build/candidates/candidate67-specialized-tail/fold-profile.json)
for artifact-bound measurements. Rebuilding creates a new artifact and requires
new installed-hash and device validation; do not reuse these receipts for it.
