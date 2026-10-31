# Fold 3 performance measurements

**Perception accuracy, freshness and sustained deadlines remain failed.**
No precision change is accepted. All live model messages remain invalid and
vehicle transmission stays disabled. See [validation scope](VALIDATION.md) for
which APK produced each result.

## Preprocessing on the final APK

The 15-test instrumentation suite compares exact padded NV12/channel packing,
random row maps, malformed inputs and the real 1280x720 to 1440x960 driver input.
Timing uses three warmups and 100 alternating measurements, outside live capture.

| Fold 3 method | p50 | p99 | Index storage |
|---|---:|---:|---:|
| Reusable NumPy gather | 2.967 ms | 2.984 ms | 11,059,200 bytes for flat intp indices |
| Native row gather | 0.784 ms | 0.825 ms | 19,200 bytes for row/column offsets |

Outputs match exactly. The native path avoids per-pixel index storage, releases
the GIL and copies repeated rows. This is a preprocessing microbenchmark, not
an end-to-end speedup claim. [Phone report](reports/fold3-final-preprocess.json),
[emulator report](reports/emulator-final-preprocess.json).

## Pinned portable inference runtime

ORT 1.31.1 is compiled from commit
`6af1a932e2286d5c58166085d24c68783033edee` using the pinned Android NDK,
without KleidiAI/SME/SVE. The source-build manifest and hash are checked by
CMake. The supplied AAR's SME `RDSVL` traps in this API 35 emulator despite
advertised host features; the portable build completes both models without
that crash. It does not solve model accuracy or driving throughput.

These individual profiles were freshly tested on the preceding `31c9028b…`
APK. Model benchmark APIs and profiles are unchanged in the final APK. Each
row has three warmups, 100 paced timed calls and three original synthetic
parity cases (seeds 42, 7, 123, including nonzero recurrent features). Original
model hashes, precision and tolerances (`rtol=1e-3`, `atol=1e-4`) are retained.

| Device / profile | DM p99 | Driving p99 | Numerical result |
|---|---:|---:|---|
| Fold 3 original CPU, 4 threads | 39.06 ms | 191.57 ms | Failed; DM max abs 0.078125, driving non-finite |
| Fold 3 original CPU, 2 threads | 35.27 ms | 190.92 ms | Failed; DM max abs 0.078125, driving non-finite |
| Fold 3 portable fallback, 4 threads | 59.32 ms | 144.60 ms | Finite, failed; DM 0.017578125, driving 0.3125 |
| Emulator original CPU, 4 threads | 42.51 ms | 186.10 ms | Failed; DM differs, driving non-finite |
| Emulator portable fallback, 4 threads | 45.47 ms | 159.82 ms | Finite, failed original-reference parity |

Raw reports: [phone CPU 4](reports/four-thread-validation/fold3-final-model-ort-cpu-4.json),
[phone CPU 2](reports/four-thread-validation/fold3-final-model-ort-cpu.json),
[phone fallback](reports/four-thread-validation/fold3-final-model-ort-fallback.json),
[emulator CPU](reports/four-thread-validation/emulator-final-model-ort-cpu-4.json),
[emulator fallback](reports/four-thread-validation/emulator-final-model-ort-fallback.json).
Portable exports use ORT extended optimizations, avoiding `NhwcFusedConv`
operators unavailable in the portable build. The export itself already fails
host parity (DM max abs 0.00390625; driving 0.03125–0.09375 across three cases).
Its manifest records that failure. [Host export checks](reports/host-portable-export-parity.json).
It is diagnostic only and excluded from the live runtime. Rebaselining to the
fallback, increasing tolerances or accepting higher precision would hide this
failure; none was done. Synthetic checks also do not replace recorded replay.

Two original CPU threads used 4,388 ms process CPU versus 7,644 ms with four
threads for 100 paced DM calls, about 43% less in these isolated runs; wall times
were both about 5.09 seconds. Driving CPU work was 23.29 versus 32.71 seconds.
These are separate measured runs, not a controlled energy comparison. The final
live runtime uses two non-spinning threads, a 50 ms minimum start interval and
no catch-up bursts. The corrected battery current/voltage calculation avoids
integer overflow, but current units and net battery power are not qualified as
a measurement of whole-phone energy savings.

## Final APK live and sustained diagnostics

The [60-second phone run](reports/fold3-final-live.json) passed capture and
preview with one session, no recovery, 1,800 rear / 1,794 front frames near
30 fps and 16.53 ms timestamp-skew p99. Rear/front delivery-age p99 was
82.71/92.72 ms; packing p99 6.59/5.49 ms. Driver-model work p99 was 58.57 ms;
976 predictions were computed, zero published, and all 976 were late.
There were 109 work durations >=50 ms. Stage p99 was 4.07 ms preparation,
55.26 ms inference and 148.29 ms sensor-to-output age. Raylib averaged 29.83
fps with CPU p99 7.02 ms. Near-30-fps rendering does not establish timely
perception. The initial landscape race is fixed in this run; the intermediate
[pre-power-fix result](reports/pre-power-fix/fold3-final-live.json) also captured
without recovery but failed DM freshness.

The 30-minute road/front capture and preview test **passed** with zero recoveries. It delivered 53,964 rear and 54,003 driver frames at 29.98/30.01 fps; both image-handler drains and camera-close callbacks completed. Final rolling nearest-skew p99 was 16.50 ms, and delivery-age p99 was 84.43/94.31 ms. The host observed 1,817.18 seconds including post-capture sampling.

**Sustained perception failed.** Android reached severe thermal status (3), with skin temperature sampled up to 44.6°C. DM computed 25,909 frames, all late, and published zero; rolling work p99 reached 82.57 ms with 19,786 cumulative 50 ms deadline misses. The UI averaged 29.96 fps. Memory after 60 seconds ranged from 916.1 to 947.4 MiB PSS. See the [soak summary](reports/fold3-final-soak-summary.json) and [raw samples](reports/fold3-final-soak.jsonl). This workload excludes road inference, localization, planning, active physical controls and Panda, so it cannot qualify the complete driving stack.

Camera percentiles use the last 4,096 frames, model work the last 10,000 calls,
model stages the last 4,096 computations and UI the last 3,000 frames.
Cumulative frame/deadline/rejection counts cover the run. A final rolling p99
is not a whole-run latency distribution. PSS and skin temperature are sampled
every 30 seconds; startup model loading is separate from the later memory
range. The raw JSONL preserves progression. Thermal override is prohibited on
the phone; native camera and host validation abort at critical status 4.

The [final emulator run](reports/emulator-final-live.json) delivered 812 rear /
829 front frames at 13.39/13.86 fps, UI about 11.97 fps, and failed capture
throughput. There were 618/623 future-delivery timestamps, rejected by the
unchanged freshness guards. Software cameras and SwiftShader do not reproduce
Samsung throughput, thermals or concurrency. After final thermal-guard checks,
the dedicated emulator was cleaned and powered off.

## Retained measurements and practical limits

Earlier foreground-service original OpenCL p99 was 82.84 ms DM and 55.69 ms
driving, both failing parity; max abs errors were 0.048828 and 2.90625. Earlier
NNAPI DM timing passed in isolation but parity failed; driving could not run
on the exposed QTI devices. These remain failed backend experiments:
[OpenCL](reports/fold3-model-opencl-foreground.json),
[NNAPI](reports/fold3-model-ort-nnapi-foreground.json). No reduced-precision or
promoted-FP32 experiment is adopted.

An earlier combined workload reached critical thermal status at 1,247.94
seconds (~20.8 minutes); its incomplete camera report cannot certify the
requested duration. [Earlier summary](reports/fold3-live-vision-30min-summary.json).
Earlier camera-only soaks failed after 180, 437 and 663 seconds. Their retained
failures are not overwritten by shorter successful diagnostics.

Earlier isolated/simulator controls and UI work is recorded in
[controls](reports/fold3-controls-work-final.json) and
[UI](reports/fold3-ui-work-final.json). MetaDrive uses synthetic perception,
calibration and a damped torque-to-angle plant. Those component measurements
exclude real road inference/localization/planning and cannot qualify complete
100 Hz controls or watchdog behavior under a physical driving workload.
