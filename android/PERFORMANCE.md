# Fold 3 performance measurements

**The performance and numerical acceptance targets were not met.** No control
loop, full UI or combined workload is running in this APK. No precision
reduction has been accepted.

## Original models on the exposed OpenCL driver

The phone's `/vendor/lib64/libOpenCL.so` loads from an ordinary Android app.
A simple arithmetic probe passed. Both original ONNX models compiled using the
pinned tinygrad OpenCL backend with serial compilation and app-local caches.
The reference uses host ONNX Runtime 1.31.1 CPU, NumPy 2.5.3, seed 42, zero float
inputs and seeded uint8 images, with RHD traffic convention. Android uses
diagnostic NumPy 1.26.2. This is one synthetic case per model, not replay.

Measured baseline: [reports/fold3-model-baseline.json](reports/fold3-model-baseline.json).
Each model ran alone after compilation and three warmups. Durations include
realization/device completion. There are 100 measured calls per model;
the 20 Hz pacing waits only when work finishes within 50 ms.

| Model | Compile | Work p50 | Work p99 | ≥50 ms / 100 | Numerical result |
|---|---:|---:|---:|---:|---|
| Driver monitoring | 111.009 s | 95.459 ms | 98.319 ms | 100 | Failed |
| Driving supercombo | 179.908 s | 71.921 ms | 74.858 ms | 100 | Failed |

The parity check uses `rtol=1e-3`, `atol=1e-4`, without weakening tolerances to
accept this device. Driver-monitoring maximum absolute error was 0.048828125.
Driving `outputs` maximum absolute error was 2.90625 and
`next_state_feat_q` was 0.0116272; the other two outputs matched in this case.
See raw results for each tensor. Neither model qualifies for driving use.

The image-buffer experiment (`IMAGE=1`, `FLOAT16=0`, `BEAM=0`) failed tinygrad
UOp type verification during compilation for both models:
[reports/fold3-model-image-fp32.json](reports/fold3-model-image-fp32.json).
It produced no usable timing comparison. `FLOAT16=0` leaves FP16 tensors
already encoded in ONNX unchanged. The optional `beam1` profile searches kernels
with the original model types. It was operator-interrupted after 1,092.266 s
while still compiling driver monitoring; driving-model search had not started.
The [partial report](reports/fold3-model-beam1-partial.json) and
[interruption record](reports/fold3-model-beam1-interruption.json) retain that
state. This is an incomplete experiment, with no timing, parity or speedup
result. No compile-time acceptance limit is inferred from stopping it.

During that cold search, 44 samples of the `:models` service recorded PSS
221,753–333,664 KiB and RSS 326,020–413,416 KiB. These are compilation-only
process measurements, not inference or whole-application memory. The separate
`fold3-beam1-main-metrics.jsonl` file sampled the main UI process and its stale
camera progress text; it must not be interpreted as model-service usage.

Earlier compile blockers and their fixes are retained in the diagnostic
failure reports: optional io_uring annotation evaluation, Android's missing
named semaphore worker support, and CPU constant copies invoking an unavailable
external compiler. The portable fixes apply to generated Android sources only.
These changes make compilation possible; they are not evidence of a speedup.

## Camera conversion and bounded memory

Camera input is converted from Android YUV planes into reusable NV12 buffers.
The native path checks dimensions, capacities, row padding and chroma pixel
stride; tests include interleaved chroma. Readers and timing windows are bounded.

| Camera-only road 0 + driver 1 | Rear packing p99 | Front packing p99 |
|---|---:|---:|
| Short probe, 1280×720 | 2.31 ms | 2.43 ms |
| Short probe, 640×480 | 1.51 ms | 2.69 ms |
| Longer rerun, 1280×720 | 4.326 ms | 4.607 ms |

The short resolution comparison does not prove a consistent improvement:
front-camera time increased and sessions were short. Packing timing excludes
capture-to-inference latency, UI rendering and actual VisionIPC camera delivery.

During the failed 436.729 s camera rerun, 28 ADB samples inside its actual
start/completion window recorded total PSS 84,052–176,500 KiB and total RSS
209,956–301,940 KiB. Battery temperature was 27.1–29.2°C; reported skin readings
were 29.1–34.3°C. Android thermal status remained 0 in these samples. The
committed metrics include samples before and after capture: filter by the
soak report's UTC window when reproducing this comparison. These observations
cannot establish 30-minute thermal stability or the cause of capture failures.

## Targets still blocked

| Acceptance target | Result |
|---|---|
| Sustained 20 Hz model inference, p99 work <50 ms | Failed baseline; combined models untested |
| 100 Hz controls, p99 work <10 ms | Blocked: Android driving runtime absent |
| 30 fps Raylib UI on cover/inner display | Blocked: Raylib backend absent |
| No upstream watchdog violations | Blocked: upstream supervised stack absent |
| 30-minute combined workload | Blocked; camera-only attempts failed early |
| Reduced precision with replay validation | Blocked; no reduced-precision option adopted |

Measured failures take precedence over successful APK installation or emulator
tests. No simulator or emulator result qualifies physical road operation.
