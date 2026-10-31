# SM-F926B camera qualification

**Required simultaneous dual rear capture failed on stock Android 15.**
Camera2 rejects opening rear IDs 0 and 2 together on this phone. No dual rear
success or complete visual-driving qualification is claimed.

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
| Logical rear 20 / 23, physical 0 + 2 | Blocked | Not available to this application |

The application-visible concurrent-camera set and logical physical groups are
empty. Actual session failures establish the tested limitation. Lens switching
and duplicate streams are never counted as dual-sensor capture. Changing
resolution did not enable the missing combination. Samsung's privileged camera
application is not evidence of Camera2 access for an ordinary application.

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

## Fresh live and sustained results

The [final 60-second phone run](reports/fold3-final-live.json) passed capture and
preview: 1,800 rear / 1,794 driver frames near 30 fps, no recovery, no metadata,
packing or timestamp-order errors, and 16.53 ms nearest-skew p99. Image callbacks
drained and both devices closed. Delivery-age p99 was 82.71 ms rear / 92.72 ms
front, leaving insufficient budget for inference. Every computed DM output was
late and none published. This is a diagnostic capture pass only.

The 30-minute road/front capture and preview test **passed** with zero recoveries. It delivered 53,964 rear and 54,003 driver frames at 29.98/30.01 fps; both image-handler drains and camera-close callbacks completed. Final rolling nearest-skew p99 was 16.50 ms, and delivery-age p99 was 84.43/94.31 ms. The host observed 1,817.18 seconds including post-capture sampling.

**Sustained perception failed.** Android reached severe thermal status (3), with skin temperature sampled up to 44.6°C. DM computed 25,909 frames, all late, and published zero; rolling work p99 reached 82.57 ms with 19,786 cumulative 50 ms deadline misses. The UI averaged 29.96 fps. Memory after 60 seconds ranged from 916.1 to 947.4 MiB PSS. See the [soak summary](reports/fold3-final-soak-summary.json) and [raw samples](reports/fold3-final-soak.jsonl). This workload excludes road inference, localization, planning, active physical controls and Panda, so it cannot qualify the complete driving stack.

The [fresh emulator matrix](reports/four-thread-validation/emulator-final-camera-matrix.json)
passed 6/12 attempts with no disconnects and complete cleanup. It cannot
establish Samsung concurrency. Its [final live run](reports/emulator-final-live.json)
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
