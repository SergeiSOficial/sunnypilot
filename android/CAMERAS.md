# SM-F926B camera qualification

Latest APK-specific results, including public low-latency YUV/PRIVATE profiles and
the three-stream rerun, are in the [candidate58 repair report](reports/road-camera-repair-20261107/README.md).
Ultrawide acquisition p99 remains 84–85 ms; mode37 preview acquisition is 108.630 ms.
These fail the 50 ms deadline before inference; independent rear identity remains blocked.
Earlier manual-exposure trials remain in the [candidate54 report](reports/accelerator-controls-20261107/README.md).
The preceding constrained-high-speed diagnostic remains in the
[capture-method comparison](reports/capture-methods-20261107/README.md).
The preceding results for direct YUV, GPU-backed PRIVATE outputs and Samsung
mode 37 remain in the [camera-path comparison](reports/camera-paths-20261106/README.md).
The [three-camera recovery report](reports/three-camera-recovery-20261106/README.md)
records the preceding camera, model and controls work.
The diagnostic model path applies a 50 ms capture-to-publication budget and a
20 ms rear-pair skew limit. The earlier [blocker-fix report](reports/blocker-fixes/README.md)
uses a 100 ms freshness limit. Raw camera timestamps are retained with
elapsed-realtime and monotonic samples. REALTIME timestamps use the measured
clock offset. UNKNOWN timestamps remain unqualified, and invalid sensor times
are rejected.

GLES3 performs verified NV12 conversion with bounded PBO/fence readback and a
checked fallback. Preview/thumbnail/front transport works, but independent rear
identity, composite-region geometry and freshness remain gates. The user has
deferred the three-repetition lens-cover experiment. The phone is unmounted;
no mounted calibration is claimed. Old results below retain their own APKs.

## Candidate35 exposure timestamps and continuous readback

APK SHA-256: `38645e91f806076517de1497cdb6b5c0cbf4873cfecf5efabb980297a014ad51`.
API 33+ outputs explicitly request `TIMESTAMP_BASE_SENSOR`; API 34+ outputs
disable readout timestamps. Android documents that the default SurfaceTexture
timestamp policy can re-space frames and add delivery latency; explicitly
requesting sensor time avoids that policy. Raw capture times and clock samples
are retained without arrival-time substitution. See the
[Android timestamp-base reference](https://developer.android.com/reference/android/hardware/camera2/params/OutputConfiguration#TIMESTAMP_BASE_DEFAULT).

Acquisition, GL and publication have separate bounded workers. Three owned NV12
buffers protect consumers from overwritten data; a newer pending publication
replaces an older pending one. Instrumentation checks slow-consumer ownership,
acquisition progress and camera teardown, plus GLES3/fallback conversion.

Four isolated 12-second runs perform continuous GPU conversion/readback,
hashing and publication timing. They retain zero timestamp/packing errors and
no disconnects. Independent rear sensor identity remains **blocked** in all four.

| Diagnostic profile | Composite delivery p99 | Rear preview delivery p99 | Front delivery p99 |
| --- | --- | --- | --- |
| Record, 1080p, depth 3 | 84.54 ms | 105.66 ms | 97.36 ms |
| Preview template | 82.82 ms | 108.83 ms | 101.92 ms |
| Record, 720p | 80.08 ms | 107.83 ms | 94.31 ms |
| Record, reader depth 2 | 85.02 ms | 106.97 ms | 96.75 ms |

All profiles still fail the 100 ms rear-preview delivery target; none is selected
as qualified. Default capture remains record/1080p/depth 3. The
[profile comparison](reports/blocker-fixes/candidate35-camera-comparison.json)
retains readback, conversion, callback, queue and clock evidence. These are
camera-only measurements, unlike historical combined-load results below.
The [candidate35 validation report](reports/blocker-fixes/candidate35-validation.md)
separately records combined-load and actual emulator-camera timing. Deterministic
injected inputs cannot qualify either camera timing or physical rear identity.

The [combined 30-second run](reports/blocker-fixes/fold-candidate35-vision.json)
records **893 composite, 895 rear-preview readbacks and 894 front frames**, with
zero recoveries and timestamp errors. Preview delivery p99 reaches 160.99 ms;
preview acquisition p99 is 158.06 ms and readback p99 is 11.25 ms. These stage
percentiles are separate measurements, not additive costs. All road inputs are
rejected. The vendor diagnostic's aggregate `preview_frames` field sums GL
preview and the additional record-reader counters; it cannot establish 60 fps
or another independent physical sensor. Use `rear_preview_frames` and
`preview_readback_frames` for the routed preview stream.

The [actual emulator camera test](reports/blocker-fixes/emulator-candidate35-camera-timing.json)
still fails: **289/289 frames have future capture timestamps**, median delivery
age **-30.93 ms**, p99 **-27.97 ms**. Images change, raw timestamps are monotonic
and callbacks close cleanly, but those facts cannot make future capture times
valid. Explicit sensor timing does not resolve this emulator result. Invalid
frames remain rejected; arrival times are not substituted. Deterministic debug
fixtures are tested separately and do not qualify emulator camera timing.

## Historical candidate32 transport and freshness

APK `a62b5518…2810` produces **893 composite, 897 preview and 894 front
frames** in the [30-second combined run](reports/blocker-fixes/fold-vision-candidate32.json).
Composite/front capture averages **29.78/29.82 fps**, with zero recoveries,
timestamp/packing errors or dropped PBO readbacks. Preview/thumbnail frames
change, but shared parent timestamps and changing pixels
do not establish distinct physical sensors. Lens-cover evidence remains pending.

Composite/preview/front delivery-age p99 is **98.86/124.60/106.47 ms**;
acquisition-age p99 is **98.52/119.80/105.37 ms**. Preview render, readback and
publication p99 is **3.53/11.40/3.11 ms**. The recorded acquisition latency already
exceeds the 100 ms budget for preview/front, so readback tuning alone cannot
qualify those inputs. No arrival-time substitution or freshness relaxation is
applied. All **58 road** predictions and **410 of 411 driver** predictions are
late; the remaining driver message is explicitly invalid. These use the
unqualified diagnostic CPU backend, separately from the accurate but slower
OpenCL gate. Raylib averages **29.51 fps**. Camera/model acceptance **fails**.

The final [emulator Camera2 timing run](reports/blocker-fixes/emulator-camera-timing-candidate32.json)
records **289 future timestamps in 289 frames**, with delivery-age p99
**−21.71 ms**. Raw timestamps remain in the evidence and are rejected. Passing
deterministic injected-input tests does not qualify this virtual camera.

Historical candidate21 (`4c48351c…09bbfe`) delivers **894 composite, 897 preview
and 895 front frames** in the [30-second combined run](reports/blocker-fixes/fold-vision-candidate21.json),
with zero recoveries, packing/timestamp errors or dropped PBO readbacks.
The rear candidates overlap for **29.94 seconds**; 894 preview and 893 thumbnail
frames change. This supports working transport, not independent sensor identity.
Delivery-age p99 composite/preview/front is **98.49/124.68/104.60 ms**; acquisition
age p99 is **98.18/119.41/104.24 ms**. Much of the latency precedes application
delivery. GPU readback p99 is **11.08 ms**, render p99 **2.78 ms**; conversion
continues on GLES3 with no synchronous CPU conversion in this run.

Freshness fails. All 54 road predictions and 442 of 443 driver predictions are
late; the remaining driver message is invalid and cannot enable engagement.
Independent rear identity and model-input mapping remain blocked by the deferred
lens-cover experiment. Both rear streams sharing a timestamp from their parent
capture is not evidence of separate physical sensors.

The [candidate21 emulator Camera2 test](reports/blocker-fixes/emulator-camera-timing-candidate21.json)
delivers 288 frames at 28.96 fps, but every capture timestamp is ahead of its
bracketing elapsed-realtime sample. Delivery-age p99 is **−26.41 ms**. These
frames fail validation; raw timestamps are retained and arrival time is never
substituted. Passing deterministic fixtures cannot qualify this camera timing.

The historical candidate10 APK (`566e4c25…e83a9a`) records **898 thumbnail/composite,
898 selected-wide preview and 899 front frames** in 30 seconds, with zero
recoveries and no packing/timestamp errors. Actual noise reduction, sharpening,
OIS and EIS modes are OFF. Image-age p99 is **126.12/144.84/102.11 ms**
respectively, beyond the 100 ms freshness target. GPU readback p99 is 10.65 ms
with no dropped readbacks. Physical capture results remain empty; vendor
physical-ID metadata and shared-parent timestamps cannot qualify image identity.
All 53 road and 469 driver predictions in the combined run are late and withheld.
See the [APK-specific camera/model report](reports/blocker-fixes/fold-gpu-vision-candidate10.json)
for raw clocks, stream origins, overlap and stage timings.

The separate [candidate10 emulator camera test](reports/blocker-fixes/emulator-host-camera-dual-candidate10.json)
delivers 437/436 frames, but every raw sensor timestamp is ahead of the
bracketing elapsed-realtime clock. Age p99 is -27.43/-26.37 ms. The validator
rejects those frames; arrival timestamps do not replace invalid sensor times.
Deterministic injected-input passes do not qualify emulator camera timing.

**Samsung multiview plus front capture now runs concurrently on stock Android 15.**
The opt-in compatibility APK accesses logical rear camera 20 and uses the
installed Samsung helper and public output-configuration extension to reproduce
the stock Director's View session. The [60-second trial](reports/camera-compat/fold3-final-samsung-front.json)
records frame counts, delivery ages and cleanup against its installed APK hash.
The [HAL evidence](reports/camera-compat/fold3-three-camera-hal.txt) confirms
ultrawide IMX258 and wide IMX555 pipelines active together while the front
IMX374 delivers frames. This fixes access and concurrent hardware operation.
Independent wide/ultrawide images suitable for the driving models remain
unqualified; HAL activation and composite thumbnails alone cannot qualify
separate calibrated driving inputs. Transmission remains disabled.

For a lens-cover trial, keep the Fold 3 still and point the rear cameras at a
well-lit, textured scene. Use a 45-second run. Start with both lenses clear,
then cover and uncover each lens separately three times, leaving several seconds
per step. Use the compatibility APK and run from the repository root.

```sh
adb_path="$HOME/Library/Android/sdk/platform-tools/adb"
trial=$(mktemp -d android/build/lens-cover.XXXXXX)
.venv/bin/python android/scripts/experiment.py --adb "$adb_path" --serial "$serial" \
  --package com.samsung.sunnypilot.fold3 --report "$trial/isolation.json" -- \
  .venv/bin/python android/scripts/validate_cameras.py --adb "$adb_path" --serial "$serial" \
  --package com.samsung.sunnypilot.fold3 --ids 0,2 --logical 20 --samsung-multiview \
  --seconds 45 --output "$trial/camera.json" --evidence-directory "$trial/clips"
```

The diagnostic saves up to four 960×540 NV12 frames per second per output. It
exports preview and thumbnail frames in pairs within 20 ms and preserves their
capture timestamps in `camera.json`. An exit status of 1 is expected while
physical sensor identity remains blocked. Use the timestamps to locate any gaps
and inspect the full thumbnail for composite regions. A paired recording alone
does not establish optical identity.

Use `preview_acquisition_age_p99_ns` to locate delay before GL conversion.
`preview_delivery_age_p99_ns` measures age immediately before the output sink;
`preview_sink_delivery_age_p99_ns` includes the sink callback. In evidence
mode that callback can include a sampled disk write. Keep both metrics when
comparing timing across APKs.


The [earlier rear-pipeline bench run](reports/rear-pipeline/README.md) continuously
bridges the selected GL preview, multiview thumbnail and front YUV into
VisionIPC. In the initial 60-second trial, the rear candidates each delivered
1,798 frames and the front delivered 1,800, with zero recoveries or provider
restarts. The original road model received synchronized rear pairs, but its
outputs were non-finite and remain invalid. The rear physical-result map was
empty; timestamp overlap and changing buffers do not establish separate sensors.
This is a transport implementation, not independent-camera qualification.

The earlier `f725600b` APK's [30-minute capture](reports/rear-pipeline/fold3-final-soak-summary.json)
also passed transport with zero recoveries: 54,006 rear-preview, 53,824 thumbnail
and 53,738 front frames, with a 29.94 fps UI. Provider PIDs stayed unchanged and
all app/camera clients exited. Physical rear identities remained unproven, and
sustained inference failed: Android reached SEVERE (44.6°C sampled skin), with
model workers paused after 4:09. No calibrated driving inputs are claimed.

`fold3_vision.RoadWarp` supports measured delivered-image intrinsics, five
OpenCV Brown distortion coefficients and per-camera device-from-calibration
Euler angles, using the pinned narrow/large model projections. An optional
private `files/camera-calibration.json` must have schema version 1, mount status
`measured`, cameras `0`/`2`, distinct `physical_sensor_id`, exact `stream_origin`,
nonempty `measurement_evidence`, `image_size`, a 3×3 `intrinsics` matrix,
`opencv_distortion` and `device_from_calib_euler`. Unverified stream origins
cannot load measured calibration. The phone is not mounted, so no such measured
file has been created; live preprocessing remains an explicitly invalid bench
affine projection. Driver calibration is also unqualified.

Build with `-Pfold3CameraCompat=true`; see [README.md](README.md). The separate
package `com.samsung.sunnypilot.fold3` uses this firmware's camera package-prefix
allowlist without replacing a Samsung application, rooting the phone, changing
system settings or bypassing Android's hidden-API enforcement. The regular
`ai.sunnypilot.fold3` package still cannot read these hidden logical groups.
Compatibility is specific to the tested Samsung firmware.

The diagnostic configures 1920x1080 PRIVATE preview, an untargeted recording
output and a 960x540 YUV callback, with Samsung shooting mode 37 and output
options 1 / 0x41 / 1. It opens rear and front before configuring either session,
drains every output, closes devices and callbacks before releasing the camera
ownership lock, and never publishes the composite as calibrated VisionIPC.
The rear opaque preview is drained on its own EGL thread and uses its
SurfaceTexture transform; instrumentation checks actual pixels, orientation
and continuous NV12 output. Standalone diagnostics can save one preview PNG
and one thumbnail NV12 with timestamps. Live VisionIPC mode instead uses
reusable buffers and saves no images. [Current results and limits](reports/rear-pipeline/README.md)
separate continuous transport from physical-image qualification; the earlier
[one-sample diagnostic](reports/camera-compat/README.md) retains its own evidence.
Mode 32 or ordinary physical-camera outputs can crash this phone's provider;
retained targeted failures are under [camera-compat](reports/camera-compat/).
Hidden cameras are therefore excluded from the automatic session matrix.

## Standard Camera2 path (prior build)

Opening rear IDs 0 and 2 as separate Camera2 devices still fails. The earlier
conclusion that the hardware cannot run multiview was too broad: the stock
Samsung session uses a different layout and vendor configuration.

The fresh [26-attempt matrix](reports/four-thread-validation/fold3-final-camera-matrix.json)
ran on APK `31c9028b…` using the same Camera2 implementation as the final
`e5347db8…` APK (only live model thread count changed afterward). It covers
1280x720 and 640x480. Sixteen attempts passed, ten failed. No attempt reported
a disconnect; all attempts drained image callbacks and received device-close
callbacks. Provider/server PIDs remained stable. The implementation follows
Android's [multi-camera interfaces](https://developer.android.com/media/camera/camera2/multi-camera).

| Camera / combination | Result | Evidence |
|---|---|---|
| Rear 0, 5.00 mm wide | Passed individually | Changing timestamped frames |
| Rear 2, 1.74 mm ultrawide | Passed individually | Changing timestamped frames |
| Front 1 / 3 | Passed individually | Separate application-visible camera IDs |
| Rear 0 + rear 2 | Failed at both sizes | `camera_error:2:2`, maximum cameras in use |
| Rear 0 or 2 + front 1 or 3 | Passed short capture | Overlapping timestamps, distinct IDs/content |
| Two rear + front | Failed | Second rear rejected |
| Logical rear 20 / 23, physical 0 + 2 | Blocked in regular package | Compatibility package can read these groups; ordinary physical-output sessions still fail |

The public camera list and concurrent-camera set remain unchanged by the
compatibility package; logical groups are accessible by their known IDs.
Lens switching and duplicate streams are never counted as dual-sensor capture.
Changing resolution alone did not enable ordinary physical-output sessions.

## Disconnect handling, orientation and frame delivery

Teardown now stops/aborts repeating requests, marks streams closing, detaches
image listeners, drains handler callbacks, closes readers and waits for camera
onClosed callbacks before releasing the native VisionIPC publisher/ownership
lock. Callback threads exit with a bounded join. Incomplete cleanup fails the
session. Publisher send/close are synchronized. The record request disables
noise reduction, edge enhancement and stabilization only when advertised.

The prior startup failure correlated with Camera2 starting before the actual
landscape surface: road rotation 90 / driver 270, then stalled front delivery.
The activity now starts capture once the landscape layout exists. The final
60-second run started with display rotation 90, applied rotation 0 for both
streams and required no recovery. A later rotation change fails capture rather
than silently retaining the old transform. Cover/inner physical fold transitions
and vehicle mounting calibration remain unqualified.

YUV planes are packed into reusable, stride-validated NV12 buffers and delivered
through the real bounded VisionIPC ring in Android's application sandbox.
Instrumentation covers all four rotations and UV packing. BOOTTIME sensor
metadata is converted to Cereal's MONOTONIC clock domain. Sensor orientation,
intrinsics, distortion and applied rotation remain in reports. Every frame has
`valid=false`: mounting transforms and exposure/calibration are not qualified.

Model input queues stay bounded at two frames per sensor, with one-to-one
pairing, a 20 ms skew limit and 100 ms age limit. Future, duplicate and stale
inputs are rejected; late predictions are discarded. The reusable preview warp
preserves upstream channel packing but is an uncalibrated affine projection.
Road inference requires distinct sensors 0 and 2 and waits when 2 is missing.
A stream stall triggers bounded recovery; any recovery fails qualification.
Critical thermal status rejects startup or closes active capture without retry.

## Previous regular-APK live and sustained results

The [previous 60-second phone run](reports/fold3-final-live.json) passed capture and
preview: 1,800 rear / 1,794 driver frames near 30 fps, no recovery, no metadata,
packing or timestamp-order errors, and 16.53 ms nearest-skew p99. Image callbacks
drained and both devices closed. Delivery-age p99 was 82.71 ms rear / 92.72 ms
front, leaving insufficient budget for inference. Every computed DM output was
late and none published. This is a diagnostic capture pass only.

The 30-minute road/front capture and preview test **passed** with zero recoveries. It delivered 53,964 rear and 54,003 driver frames at 29.98/30.01 fps; both image-handler drains and camera-close callbacks completed. Final rolling nearest-skew p99 was 16.50 ms, and delivery-age p99 was 84.43/94.31 ms. The host observed 1,817.18 seconds including post-capture sampling.

**Sustained perception failed.** Android reached severe thermal status (3), with skin temperature sampled up to 44.6°C. DM computed 25,909 frames, all late, and published zero; rolling work p99 reached 82.57 ms with 19,786 cumulative 50 ms deadline misses. The UI averaged 29.96 fps. Memory after 60 seconds ranged from 916.1 to 947.4 MiB PSS. See the [soak summary](reports/fold3-final-soak-summary.json) and [raw samples](reports/fold3-final-soak.jsonl). This workload excludes road inference, localization, planning, active physical controls and Panda, so it cannot qualify the complete driving stack.

The [previous emulator matrix](reports/four-thread-validation/emulator-final-camera-matrix.json)
passed 6/12 attempts with no disconnects and complete cleanup. It cannot
establish Samsung concurrency. Its [previous live run](reports/emulator-final-live.json)
failed throughput (~13–14 fps cameras), with many future sensor timestamps;
freshness guards were retained. A System UI ANR initially paused rendering;
closing its observed dialog allowed lifecycle/resize tests to pass, while live
throughput remained failed.

## Retained failures

The previous `e2ee0027…` [60-second run](reports/fold3-live-vision-final.json)
failed on `camera_disconnected:0`; two retries found camera 0 unavailable and
only the third recovered. Camera-only 30-minute attempts failed after 180.123,
436.729 and 663.268 seconds, with capture failures, metadata loss and front
staleness respectively. The 663-second attempt had 806 ms pairing-skew p99.
The earlier combined diagnostic reached critical thermal status after 20.8
minutes and never certified 30 minutes of capture. Original reports are retained:
[initial](reports/soak-initial.json), [second](reports/soak-rerun.json),
[VisionIPC soak](reports/camera-visionipc-and-soak.json),
[thermal summary](reports/fold3-live-vision-30min-summary.json).

Short clean sessions do not erase those failures or qualify required dual rear
capture, timely driver monitoring or real vehicle use.

### Candidate68 manual exposure and low-latency comparison (2026-11-08)

[Artifact-bound report](reports/dispatch-batching-20261108/README.md).
APK `925b4b0ab7b40e50c6b19b3ca8f8f4fba2ef5eefd74fcc8b5d0588e2dff8a048`.

Diagnostic public Camera2 profile `manual-8ms-low-latency` combines supported short manual exposure with advertised processing controls in session parameters. Isolated ultrawide 640x360 direct YUV publication p99: low-latency 75.331 ms, manual 8 ms exposure 62.102 ms, combined 61.741 ms. Each publishes 357 frames and all exceed 50 ms. This does not qualify the new profile. Mode37 three-stream rear-preview/front delivery p99 is 102.725/98.047 ms; independent rear identity remains unproven. Emulator camera timing rejects 340 of 342 capture timestamps, with 338 result timestamp errors and 2 publications; it remains failed. No arrival-time substitution.

TX and engagement remain disabled; cooling optimization is deferred and thermal safeguards remain active. Prior candidates and failures are preserved.
