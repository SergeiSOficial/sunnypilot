# SM-F926B camera qualification

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

The [current rear-pipeline bench run](reports/rear-pipeline/README.md) continuously
bridges the selected GL preview, multiview thumbnail and front YUV into
VisionIPC. In the initial 60-second trial, the rear candidates each delivered
1,798 frames and the front delivered 1,800, with zero recoveries or provider
restarts. The original road model received synchronized rear pairs, but its
outputs were non-finite and remain invalid. The rear physical-result map was
empty; timestamp overlap and changing buffers do not establish separate sensors.
This is a transport implementation, not independent-camera qualification.

The final compatibility APK's [30-minute capture](reports/rear-pipeline/fold3-final-soak-summary.json)
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
