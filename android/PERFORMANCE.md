# Fold 3 performance measurements

**The model deadline and full visual-driving acceptance targets were not met.**
Android controls and Raylib now run, but passing their isolated checks does not
qualify simultaneous camera, perception, localization, planning and controls.
No precision change is accepted for driving.

## Current foreground-service reruns

Model benchmarks previously used an ordinary background service. They now run
in an Android foreground service, matching the runtime's process importance.
The original models are hash checked and compared with the same independent
host reference (`rtol=1e-3`, `atol=1e-4`). Each row has three warmups and 100
timed calls, measured separately from live capture. Thermal state and execution
order vary, so these are observations rather than controlled speedup claims.

| Backend / model | Work p50 | Work p99 | ≥50 ms / 100 | Acceptance |
|---|---:|---:|---:|---|
| OpenCL, original DM | 82.39 ms | 82.84 ms | 100 | Failed timing and parity (max abs 0.048828) |
| OpenCL, original driving | 54.35 ms | 55.69 ms | 100 | Failed timing and parity (outputs 2.90625, features 0.011627) |
| ORT CPU, 2 threads, DM | 30.53 ms | 36.82 ms | 0 | Timing passed; parity failed (0.078125) |
| ORT CPU, 2 threads, driving | 155.22 ms | 158.68 ms | 100 | Failed; non-finite outputs/features |
| ORT CPU, 4 threads, DM | 28.10 ms | 33.03 ms | 0 | Timing passed; parity failed (0.078125) |
| ORT CPU, 4 threads, driving | 151.57 ms | 154.54 ms | 100 | Failed; non-finite outputs/features |
| ORT NNAPI provider, DM | 30.49 ms | 39.79 ms | 0 | Parity failed (0.078125); provider may fall back to CPU |
| ORT NNAPI provider, driving | — | — | — | Blocked: exposed QTI devices cannot run model |

Raw reports: [OpenCL](reports/fold3-model-opencl-foreground.json),
[CPU two threads](reports/fold3-model-ort-cpu-foreground.json),
[CPU four threads](reports/fold3-model-ort-cpu-4.json),
[NNAPI](reports/fold3-model-ort-nnapi-foreground.json).
The earlier [background CPU four-thread run](reports/fold3-model-ort-cpu-4-background.json)
measured DM p99 328.95 ms; it has different scheduler/thermal conditions and
must not be presented as a controlled thread-count comparison. NNAPI's CPU
device is excluded, but ORT's unsupported-node CPU fallback is still possible.

The 60-second live Fold 3 diagnostic additionally includes camera delivery,
NV12 projection, Raylib and IMU. DM work p99 was **57.22 ms**, with 46 work
durations ≥50 ms across 1,489 calls. All 1,489 outputs exceeded the 100 ms input
age budget and were discarded; zero valid driver predictions were published.
The road worker waited for sensor 2 throughout. Isolated DM benchmark throughput
therefore does not establish sustained 20 Hz perception. Every visual diagnostic
message remains invalid. Non-finite predictions are discarded and reset the
model's recurrent state.

The later requested 30-minute combined diagnostic failed on heat: Android
reported critical status 4 at 1,247.94 seconds. The operator stopped the workload;
the persisted camera report did not finalize. DM computed 20,083 frames and
discarded all as late, with 14,275 work durations ≥50 ms. Its final rolling work
p99 was **102.06 ms**. Raylib's last active report averaged 29.91 fps across
43,050 frames. Active UI samples after 120 seconds measured aggregate PSS from
966,953 to 1,034,027 KiB. These figures describe the partial diagnostic, not a
complete driving workload. See the [sustained-run summary](reports/fold3-live-vision-30min-summary.json).
The delivered APK adds a critical-temperature capture abort; this development
run does not establish sustained qualification for that APK.

The delivered APK's [60-second rerun](reports/fold3-live-vision-final.json)
failed camera reliability after three recoveries. DM computed 1,115 frames,
published zero and discarded all as late; work p99 was 57.74 ms, with 46 work
durations ≥50 ms. Raylib averaged 29.29 fps. This does not improve the sustained
or original-model acceptance result.

The live UI averaged 29.31 fps; recent frame-interval p99 was 34.14 ms and CPU
work p99 8.36 ms. Camera packing p99 was 6.62 ms rear; stream skew p99 was
16.59 ms. See [the live report](reports/fold3-live-vision.json) for per-stream
delivery age, frame intervals, drops and model work. No calibrated road model,
planning or physical control work is included.

Live statistics are bounded: camera timing/skew percentiles use the last 4,096
frames per stream, UI percentiles the last 3,000 frames, and model-work p99 the
last 10,000 calls. Cumulative frame/deadline/rejection counters cover the run;
JSONL samples preserve the progression. A final rolling-window p99 is not a
whole-run latency distribution. PSS is sampled across package processes and
includes shared-page proportional accounting; startup model loading must be
separated from the later memory plateau.

## Earlier OpenCL and precision experiments

The ordinary Android app loads `/vendor/lib64/libOpenCL.so`. The pinned tinygrad
backend compiles both ONNX models with serial compilation and app-local caches.
Measurements include realization/device completion after three warmups, with
100 calls per model. Models are measured separately. Host reference: ONNX Runtime
1.31.1 CPU, NumPy 2.5.3, seed 42, zero float inputs and seeded uint8 images with
RHD traffic convention. Android NumPy is 1.26.2. This is one synthetic case per
model, not replay or a measured live perception pipeline.

| Model / experiment | Compile | Work p50 | Work p99 | ≥50 ms / 100 | Numerical result |
|---|---:|---:|---:|---:|---|
| Original DM, baseline | 111.009 s | 95.459 ms | 98.319 ms | 100 | Failed |
| Original driving, baseline | 179.908 s | 71.921 ms | 74.858 ms | 100 | Failed |
| FP32 DM, image buffers | 321.217 s | 148.691 ms | 158.417 ms | 100 | Passed own FP32 reference only |
| FP32 driving, image buffers | 404.734 s | 263.415 ms | 276.665 ms | 100 | Passed own FP32 reference only |

Raw [baseline](reports/fold3-model-baseline.json) and
[FP32 image-buffer experiment](reports/fold3-model-fp32-image.json).
The latter is slower; no speedup is claimed.

Parity uses unchanged `rtol=1e-3`, `atol=1e-4`. Original phone maximum absolute
errors: DM 0.048828125, driving `outputs` 2.90625, next features 0.0116272.
The [host tinygrad CPU check](reports/host-model-parity-original.json) also fails
on original FP16 types (DM 0.015625, driving 0.21875), so the discrepancy cannot
be attributed solely to the phone's GPU driver.

Promoting FP16 weights, casts and intermediate types to FP32 makes host tinygrad
and Android OpenCL agree with their respective FP32 ONNX Runtime reference.
Phone errors are 0.00000715 DM and 0.00014305 driving. However, FP32 ONNX Runtime
itself differs from the original FP16 reference: driving output max 0.296173,
next features 0.00048198. Passing the promoted reference does **not** establish
original-model semantics or replay quality. The FP32 profile remains diagnostic.
[Host FP32 comparison](reports/host-model-parity-fp32.json).

`IMAGE=1` with the original FP16 types failed UOp verification and produced no
usable timing. `FLOAT16=0` does not replace types encoded in ONNX. `BEAM=1` was
interrupted after 1,092.266 s while compiling its first model; no parity or
performance result was obtained. The original failure, partial and interruption
reports are retained. No tolerance or deadline was relaxed to accept a profile.

## Controls, UI and camera work

The phone's real upstream controlsd runs in an isolated service process. A short
synthetic state/DM/calibration fixture with Raylib visible measured **p99 work
6.399 ms**, 2,223 updated iterations, zero work durations ≥10 ms:
[report](reports/fold3-controls-work.json). It times `state_control`, publication
and extension work; it excludes `update` (message reception and pose calibration)
and the report callback. It is not complete loop work, end-to-end latency or
100 Hz scheduling qualification. The fixed control rate and upstream controller
limits remain.

A later mixed simulator/fault run on the final APK collected 2,991 iterations:
p99 6.501 ms, one ≥10 ms in the same measured subset
([raw report](reports/fold3-controls-work-final.json)). These runs are not a
controlled before/after optimization comparison.

The native Raylib backend runs the upstream landscape UI with a 30 fps target.
UI reports record actual frame intervals, CPU work, bounded recent p99 and
surface dimensions. Background time is included in mean fps, so mean fps during
lifecycle tests is not a steady-state throughput estimate. Emulator software
rendering and physical fold transitions require separate interpretation.
See the committed lifecycle reports for the final measured runs.
The emulator initially counted frames on a black surface with an invalid EGL
configuration. The Android backend now retries without unsupported MSAA and
checks EGL failures. Final lifecycle checks verify visible pixels; the older
emulator frame counters alone were insufficient evidence of rendering.
The final phone UI snapshot recorded 13,200 frames, mean 29.85 fps including
startup, recent frame-interval p99 33.71 ms and CPU-work p99 8.50 ms
([raw report](reports/fold3-ui-work-final.json)). It includes simulated state,
without concurrent camera/model inference. This measures offroad UI rendering;
onroad camera overlays are not qualified. The loaded emulator fault run hit its
100 ms transport timeout; isolated controller validation passes with software
rendering backgrounded. Neither run establishes combined real-time scheduling.

Fresh APK packaging removed 224,024,254 bytes of unused incremental ZIP space
(48.14% of that package). Every ZIP entry had identical size and SHA-256 before
and after: [comparison](reports/apk-packaging-comparison.json). This reduces
installation size/transfer, not model execution time.

NV12 packing reuses buffers and validates row/pixel strides and capacities.
The latest camera-only VisionIPC soak measured packing p99 **4.519 ms rear /
4.745 ms front**, with zero packing errors. It failed at 663.268 s on a stale
front stream; nearest-stream skew p99 was 806.163 ms including stalls/drops.
Packing time excludes capture wait, IPC delivery, transforms and inference.

During the earlier 436.729 s camera rerun, 28 in-window samples recorded total
PSS 84,052–176,500 KiB, RSS 209,956–301,940 KiB, battery 27.1–29.2°C and Android
thermal status 0. The newer camera/VisionIPC report records thermal status 0→1
(light throttling), native heap 20–68 MiB. These observations do not establish
30-minute combined stability or the cause of capture failure.

## Acceptance

| Target | Result |
|---|---|
| 20 Hz models, p99 work <50 ms | **Failed**, original and FP32 experiments |
| Original-model parity and replay | **Failed** synthetic original parity; replay **blocked** |
| 100 Hz controls, p99 work <10 ms | Short controller/publication subset **passed**; complete work/scheduling qualification **blocked** |
| 30 fps Raylib | Rendering **passed**; sustained combined/folded display qualification **blocked** |
| No upstream watchdog violations | Service restart check **passed**; complete stack qualification **blocked** |
| 30-minute combined workload | Partial camera/DM/UI/IMU attempt **failed** on critical heat; full driving workload **blocked** |

The default backend and precision policy are unchanged. Cached compilation,
bounded queues and reusable frame buffers are implemented, but model accuracy,
GPU throughput and camera reliability remain unresolved physical-device blockers.
