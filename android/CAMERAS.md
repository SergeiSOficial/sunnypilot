# SM-F926B camera qualification

**Required dual rear capture failed on stock Android 15.** Public Camera2 cannot
open rear IDs 0 and 2 together on this phone. No dual rear success is claimed.

The [26-attempt matrix](reports/camera-matrix.json) records application-visible
camera groups, streams and actual sessions at 1280×720 and 640×480. The newer
[VisionIPC/soak report](reports/camera-visionipc-and-soak.json) retains a repeated
matrix and camera metadata. The implementation follows Android's
[multi-camera interfaces](https://developer.android.com/media/camera/camera2/multi-camera).

| Camera / combination | Result | Evidence |
|---|---|---|
| Rear 0, 5.00 mm wide | Passed individually | Changing timestamped frames |
| Rear 2, 1.74 mm ultrawide | Passed individually | Changing timestamped frames |
| Front 1 / 3 | Passed individually | Separate application camera IDs |
| Rear 0 + rear 2 | **Failed** at both sizes | `camera_error:2:2`, maximum cameras in use |
| Rear 0 or 2 + front 1 or 3 | Passed short capture only | Overlapping timestamps, distinct IDs/content |
| Two rear + front | **Failed** | Second rear rejected |
| Logical rear 20 / 23, physical 0 + 2 | **Blocked** | Not available to this application |

The app-visible concurrent-camera set is empty. The actual session failures,
rather than that capability listing alone, establish the tested limitation.
Lens switching and duplicate frames are never counted as dual sensor capture.
Passing road/front capture does not establish driver monitoring quality.

## Native frame delivery and sustained capture

Camera2 YUV planes are packed into reusable NV12 buffers with validated row and
pixel strides, then copied into the real four-buffer-per-stream VisionIPC ring.
The Android Bionic backend uses app-private mmap/socket paths and memfd. Java
publisher-to-Python VisionIPC instrumentation checks ten changing frames,
IDs, timestamps, stride offsets and invalid calibration status. Sensor orientation,
intrinsics and timestamp source are retained in the diagnostic report.
BOOTTIME sensor timestamps are converted to Cereal's MONOTONIC clock domain.

**Every camera frame has `valid=false`.** Mounting transforms, exposure timing,
calibrated intrinsics/extrinsics and usable driver images remain unqualified.
There is no accepted synchronized model-input pipeline. The pairing statistic is
descriptive; it does not grant validity to unsynchronized frames. A missing stream
stops the diagnostic after two seconds and engagement stays blocked.

Three requested 30-minute camera-only road 0 + front 1 soaks failed:

| Run | Actual duration | Rear / front frames | Failure |
|---|---:|---:|---|
| [Initial](reports/soak-initial.json) | 180.123 s | 5345 / 5254 | Capture failure |
| [Second](reports/soak-rerun.json) | 436.729 s | 13046 / 12942 | Front capture metadata failure |
| [VisionIPC rerun](reports/camera-visionipc-and-soak.json) | 663.268 s | 19853 / 19704 | `stale_camera:1` |

The latest run continued through metadata-only failures when an image was still
captured. It retained strict qualification failure for any metadata/lost-buffer
error and stopped on actual frame staleness. Rear timestamps and packing had no
errors; front capture recorded eight metadata failures, five uncaptured requests
and three lost buffers. Packing p99 was 4.519 ms rear / 4.745 ms front; nearest
stream skew p99 was **806.163 ms**, including stalls/drops in the bounded window.
Overlap was 661.565 s. Sustained synchronized capture is not qualified. Android
thermal status went from 0 to 1 (light throttling); the capture-stall cause is
not established. No 30-minute combined visual-driving workload passed.

Readers, native rings and statistic windows are bounded. `acquireLatestImage()`
releases every image, callbacks drain before publisher shutdown, and failures
close devices/readers. No images are saved. No hidden camera allowlisting,
root access or stock camera modification was attempted. Camera concurrency,
synchronization/calibration and sustained DM capture remain engagement blockers.
