# SM-F926B camera qualification — 2026-10-30

**Required dual rear capture failed.** Camera IDs 0 and 2 could not be opened
together through this app on stock Android 15. No dual rear result is counted
as a success. Camera failure keeps vehicle transmission disabled.

The complete 26-attempt matrix is [reports/camera-matrix.json](reports/camera-matrix.json).
All attempts used Camera2 `YUV_420_888` ImageReaders, not Samsung's camera app.
The app records camera characteristics, physical groups, output sizes,
orientation, timestamp source, session failures and delivered frame statistics.
It tries both independent devices and the known HAL logical IDs 20/23.
The implementation follows Android's [multi-camera interfaces](https://developer.android.com/media/camera/camera2/multi-camera).

| Application camera | Facing | Focal length | Application access |
|---|---|---:|---|
| 0 | Rear wide | 5.00 mm | Passed individually |
| 2 | Rear ultrawide | 1.74 mm | Passed individually |
| 1 | Front | 3.30 mm | Passed individually |
| 3 | Front | See raw capabilities | Passed individually |
| 20 / 23 | HAL logical rear groups | Physical 0 + 2 requested | Blocked: IDs not available to this app |

The app-visible concurrent-camera set was empty. That does not alone establish
that every combination fails; actual session attempts below determine results.

| Requested combination | 1280×720 | 640×480 | Evidence |
|---|---|---|---|
| Each camera individually | Passed | Passed | About 29.6–29.7 fps |
| Rear 0 + rear 2 | Failed | Failed | `camera_error:2:2` (maximum cameras in use) |
| Rear 0 + front 1 or 3 | Passed short probe | Passed short probe | Overlapping timestamps and changing frames |
| Rear 2 + front 1 or 3 | Passed short probe | Passed short probe | Overlapping timestamps and changing frames |
| Rear 0 + rear 2 + front 1 or 3 | Failed | Failed | Second rear camera rejected |
| Logical 20/23 with physical 0 + 2 | Blocked | Blocked | Unknown application camera ID |

Passing short road/front probes is not driver monitoring validation. They prove
capture from distinct application camera IDs, with separate metadata and
overlapping timestamps. They do not establish calibrated sensor transforms,
usable face images, independent inference or simultaneous dual rear delivery.
No lens switching or duplicate stream is used as a substitute for dual rear
capture. No frames are saved to disk; a sampled luma hash checks changing input.

Two requested 30-minute camera-only road 0 + front 1 soaks failed:

| Run | Actual duration | Delivered rear / front frames | Failure |
|---|---:|---:|---|
| [Initial](reports/soak-initial.json) | 180.123 s | 5345 / 5254 | `capture_failed:0` |
| [Rerun](reports/soak-rerun.json) | 436.729 s | 13046 / 12942 | Front camera capture failure, reason 0, frame 12942, image captured |

The rerun had monotonic sensor timestamps, no NV12 packing errors and 432.733 s
of overlap. Its last bounded timestamp window had nearest-stream skew p99
16.532 ms. This nearest-neighbour measurement is descriptive; there is no
calibrated synchronization budget or camera-to-VisionIPC publication yet.
The initial matrix's skew calculation included startup samples; the rerun
uses only shared overlap in the last 4096 timestamps per stream.

The app closes devices and readers on failures, uses `acquireLatestImage()`
with bounded reader capacity, reuses NV12 buffers, rejects nonmonotonic
timestamps and stops on missing input. These diagnostics do not replace the
upstream watchdogs. An actual driving camera bridge still needs stream
orientation transforms, calibrated intrinsics/extrinsics, frame IDs, start/end
exposure timestamps, synchronization rejection and valid metadata.

No root access, hidden camera allowlisting or stock-camera modification was
attempted. Further work must resolve the stock Camera2 combination/reliability
limits and validate driver monitoring before any engagement path can exist.
