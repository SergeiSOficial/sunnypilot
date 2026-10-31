package ai.sunnypilot.fold3;

import android.content.Context;
import android.graphics.ImageFormat;
import android.hardware.camera2.*;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.PowerManager;
import android.util.Range;
import android.util.Size;
import org.json.JSONArray;
import org.json.JSONObject;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/** Measures real Camera2 sessions. All camera ownership is scoped to one attempt. */
final class CameraProbe implements AutoCloseable {
    interface Progress { void update(String message); }
    private final CameraManager manager;
    private final HandlerThread thread = new HandlerThread("camera-frames");
    private final Handler handler;
    private final Progress progress;
    private final String ipcDirectory;
    private final Context context;
    private final int displayDegrees;

    private void checkThermalStatus() {
        int status = context.getSystemService(PowerManager.class).getCurrentThermalStatus();
        if (status >= PowerManager.THERMAL_STATUS_CRITICAL)
            throw new IllegalStateException("critical_thermal_status:" + status);
    }

    CameraProbe(Context context, Progress progress) {
        this.context = context;
        displayDegrees = context.getSystemService(android.hardware.display.DisplayManager.class)
            .getDisplay(android.view.Display.DEFAULT_DISPLAY).getRotation() * 90;
        manager = context.getSystemService(CameraManager.class);
        ipcDirectory = new java.io.File(context.getFilesDir(), "ipc").getAbsolutePath();
        this.progress = progress;
        thread.start();
        handler = new Handler(thread.getLooper());
    }
    JSONObject capabilities() throws Exception {
        JSONObject out = new JSONObject();
        JSONArray cameras = new JSONArray();
        for (String id : manager.getCameraIdList()) {
            CameraCharacteristics c = manager.getCameraCharacteristics(id);
            JSONObject camera = new JSONObject().put("id", id)
                .put("facing", c.get(CameraCharacteristics.LENS_FACING))
                .put("orientation", c.get(CameraCharacteristics.SENSOR_ORIENTATION))
                .put("timestamp_source", c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE))
                .put("hardware_level", c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL))
                .put("physical_ids", new JSONArray(c.getPhysicalCameraIds()))
                .put("focal_lengths_mm", array(c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)))
                .put("capabilities", array(c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)));
            StreamConfigurationMap map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            JSONArray sizes = new JSONArray();
            if (map != null && map.getOutputSizes(ImageFormat.YUV_420_888) != null) {
                for (Size size : map.getOutputSizes(ImageFormat.YUV_420_888)) {
                    sizes.put(new JSONObject().put("width", size.getWidth()).put("height", size.getHeight())
                        .put("min_frame_duration_ns", map.getOutputMinFrameDuration(ImageFormat.YUV_420_888, size)));
                }
            }
            cameras.put(camera.put("yuv_sizes", sizes));
        }
        JSONArray concurrent = new JSONArray();
        for (Set<String> ids : manager.getConcurrentCameraIds()) concurrent.put(new JSONArray(ids));
        return out.put("cameras", cameras).put("concurrent_id_sets", concurrent);
    }
    private static JSONArray array(Object value) throws org.json.JSONException {
        return value == null ? new JSONArray() : new JSONArray(value);
    }
    List<String[]> combinations() throws Exception {
        List<String[]> modes = new ArrayList<>();
        List<String> rear = new ArrayList<>(), front = new ArrayList<>();
        for (String id : manager.getCameraIdList()) {
            Integer facing = manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING);
            modes.add(new String[]{id});
            if (Objects.equals(facing, CameraCharacteristics.LENS_FACING_BACK)) rear.add(id);
            if (Objects.equals(facing, CameraCharacteristics.LENS_FACING_FRONT)) front.add(id);
        }
        for (int i = 0; i < rear.size(); i++) {
            for (int j = i + 1; j < rear.size(); j++) {
                modes.add(new String[]{rear.get(i), rear.get(j)});
                for (String driver : front) modes.add(new String[]{rear.get(i), rear.get(j), driver});
            }
            for (String driver : front) modes.add(new String[]{rear.get(i), driver});
        }
        return modes;
    }
    Set<String> logicalCandidates() throws Exception {
        Set<String> ids = new LinkedHashSet<>();
        for (String id : manager.getCameraIdList()) {
            if (manager.getCameraCharacteristics(id).getPhysicalCameraIds().size() >= 2) ids.add(id);
        }
        // Present in the Fold 3 HAL dump. Ordinary app access may be denied: record that result.
        ids.add("20"); ids.add("23");
        return ids;
    }
    private final class Stream implements AutoCloseable {
        final String label;
        final FrameStats stats = new FrameStats();
        final ImageReader reader;
        final ByteBuffer nv12;
        final ByteBuffer packed;
        final int rotation, outputWidth, outputHeight;
        final JSONObject metadata;
        volatile long lastArrivalMs;
        volatile boolean closing;
        long metadataFailures, uncapturedRequests, lostBuffers;
        Stream(String id, int width, int height, VisionPublisher publisher) throws Exception {
            label = id;
            CameraCharacteristics c = manager.getCameraCharacteristics(id);
            rotation = Math.floorMod(c.get(CameraCharacteristics.SENSOR_ORIENTATION) +
                (Objects.equals(c.get(CameraCharacteristics.LENS_FACING), CameraCharacteristics.LENS_FACING_FRONT) ? displayDegrees : -displayDegrees), 360);
            outputWidth = rotation % 180 == 0 ? width : height;
            outputHeight = rotation % 180 == 0 ? height : width;
            metadata = new JSONObject().put("sensor_id", id).put("width", width).put("height", height)
                .put("image_rotation_applied_degrees", rotation).put("output_width", outputWidth).put("output_height", outputHeight)
                .put("display_degrees_at_session_start", displayDegrees).put("mirrored", false)
                .put("sensor_orientation", c.get(CameraCharacteristics.SENSOR_ORIENTATION))
                .put("timestamp_source", c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE))
                .put("intrinsic_calibration", array(c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)))
                .put("lens_distortion", array(c.get(CameraCharacteristics.LENS_DISTORTION)))
                .put("visionipc_valid", false).put("visionipc_stream", VisionPublisher.stream(id))
                .put("focal_lengths_mm", array(c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)));
            reader = ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 3);
            nv12 = ByteBuffer.allocateDirect(width * height * 3 / 2);
            packed = rotation == 0 ? nv12 : ByteBuffer.allocateDirect(width * height * 3 / 2);
            java.nio.file.Path destination = new java.io.File(context.getFilesDir(), "camera-" + id + ".json").toPath();
            java.nio.file.Path temporary = destination.resolveSibling("camera-" + id + ".tmp");
            java.nio.file.Files.write(temporary, metadata.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
            java.nio.file.Files.move(temporary, destination, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            reader.setOnImageAvailableListener(source -> {
                try (Image image = source.acquireLatestImage()) {
                    if (image == null || closing) return;
                    Image.Plane[] p = image.getPlanes();
                    long start = System.nanoTime();
                    long hash = NativeBridge.packNv12(p[0].getBuffer().slice(), p[0].getRowStride(), p[0].getPixelStride(),
                        p[1].getBuffer().slice(), p[1].getRowStride(), p[1].getPixelStride(),
                        p[2].getBuffer().slice(), p[2].getRowStride(), p[2].getPixelStride(), width, height, packed);
                    if (rotation != 0) NativeBridge.rotateNv12(packed, width, height, rotation, nv12);
                    stats.add(image.getTimestamp(), hash, System.nanoTime() - start);
                    long timestamp = image.getTimestamp();
                    if (Objects.equals(c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE), CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME))
                        timestamp -= android.os.SystemClock.elapsedRealtimeNanos() - System.nanoTime();
                    stats.deliveryAge(System.nanoTime() - timestamp);
                    publisher.send(id, nv12, outputWidth, outputHeight, stats.frames, timestamp);
                    lastArrivalMs = android.os.SystemClock.elapsedRealtime();
                } catch (Exception e) { stats.packingErrors++; }
            }, handler);
        }
        JSONObject result() throws Exception {
            return new JSONObject(metadata.toString()).put("frames", stats.frames).put("fps", stats.fps())
                .put("first_timestamp_ns", stats.firstTimestamp).put("last_timestamp_ns", stats.lastTimestamp)
                .put("timestamp_errors", stats.timestampErrors).put("changed_frames", stats.changedFrames)
                .put("metadata_failures", metadataFailures).put("uncaptured_requests", uncapturedRequests).put("lost_buffers", lostBuffers)
                .put("packing_errors", stats.packingErrors).put("packing_p99_ns", stats.percentile(true, .99))
                .put("delivery_age_p99_ns", stats.deliveryAgeP99()).put("delivery_over_100ms", stats.lateDeliveries)
                .put("delivery_future_timestamps", stats.futureDeliveries)
                .put("frame_interval_p99_ns", stats.percentile(false, .99));
        }
        public void close() { reader.setOnImageAvailableListener(null, null); reader.close(); }
    }
    private CameraDevice open(String id, AtomicReference<String> failure, List<CountDownLatch> closures) throws Exception {
        CountDownLatch done = new CountDownLatch(1);
        CountDownLatch closed = new CountDownLatch(1);
        AtomicBoolean abandoned = new AtomicBoolean();
        AtomicReference<CameraDevice> result = new AtomicReference<>();
        manager.openCamera(id, new CameraDevice.StateCallback() {
            public void onOpened(CameraDevice device) {
                result.set(device);
                if (abandoned.get()) device.close();
                done.countDown();
            }
            public void onDisconnected(CameraDevice device) { failure.set("camera_disconnected:" + id); device.close(); done.countDown(); }
            public void onError(CameraDevice device, int error) { failure.set("camera_error:" + id + ":" + error); device.close(); done.countDown(); }
            public void onClosed(CameraDevice device) { closed.countDown(); }
        }, handler);
        closures.add(closed);
        if (!done.await(8, TimeUnit.SECONDS)) {
            abandoned.set(true);
            if (result.get() != null) result.get().close();
            throw new IllegalStateException("open timeout: " + id);
        }
        CameraDevice device = result.get();
        if (device == null || failure.get() != null) throw new IllegalStateException(failure.get());
        return device;
    }
    private CameraCaptureSession session(CameraDevice camera, List<Stream> streams, boolean physical,
                                         AtomicReference<String> failure) throws Exception {
        CountDownLatch done = new CountDownLatch(1);
        AtomicBoolean abandoned = new AtomicBoolean();
        AtomicReference<CameraCaptureSession> result = new AtomicReference<>();
        List<OutputConfiguration> outputs = new ArrayList<>();
        for (Stream stream : streams) {
            OutputConfiguration output = new OutputConfiguration(stream.reader.getSurface());
            if (physical) output.setPhysicalCameraId(stream.label);
            outputs.add(output);
        }
        camera.createCaptureSessionByOutputConfigurations(outputs, new CameraCaptureSession.StateCallback() {
            public void onConfigured(CameraCaptureSession s) {
                result.set(s);
                if (abandoned.get()) s.close();
                done.countDown();
            }
            public void onConfigureFailed(CameraCaptureSession s) { failure.set("session_configuration_failed:" + camera.getId()); s.close(); done.countDown(); }
        }, handler);
        if (!done.await(8, TimeUnit.SECONDS)) {
            abandoned.set(true);
            if (result.get() != null) result.get().close();
            throw new IllegalStateException("session timeout: " + camera.getId());
        }
        if (result.get() == null) throw new IllegalStateException(failure.get());
        CaptureRequest.Builder request = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
        // Use only advertised low-latency modes; Samsung's preview processing can
        // otherwise consume most of the 100 ms sensor-to-output budget.
        CameraCharacteristics characteristics = manager.getCameraCharacteristics(camera.getId());
        setIfSupported(request, CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF,
            characteristics.get(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES));
        setIfSupported(request, CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF,
            characteristics.get(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES));
        setIfSupported(request, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
            characteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES));
        setIfSupported(request, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF,
            characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION));
        for (Stream stream : streams) request.addTarget(stream.reader.getSurface());
        Range<Integer>[] ranges = manager.getCameraCharacteristics(camera.getId()).get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        if (ranges != null && Arrays.asList(ranges).contains(new Range<>(30, 30)))
            request.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(30, 30));
        result.get().setRepeatingRequest(request.build(), new CameraCaptureSession.CaptureCallback() {
            public void onCaptureFailed(CameraCaptureSession s, CaptureRequest r, CaptureFailure f) {
                // A request failure is not a device/session death. Keep measuring
                // recovery and actual frame freshness, but never hide lost metadata.
                for (Stream stream : streams) {
                    if (stream.closing) continue;
                    stream.metadataFailures++;
                    if (!f.wasImageCaptured()) stream.uncapturedRequests++;
                }
            }
            public void onCaptureBufferLost(CameraCaptureSession s, CaptureRequest r, android.view.Surface target, long frame) {
                for (Stream stream : streams) if (!stream.closing && stream.reader.getSurface().equals(target)) stream.lostBuffers++;
            }
        }, handler);
        return result.get();
    }
    private static void setIfSupported(CaptureRequest.Builder request, CaptureRequest.Key<Integer> key, int value, int[] modes) {
        if (modes != null && Arrays.stream(modes).anyMatch(mode -> mode == value)) request.set(key, value);
    }
    JSONObject attempt(String[] ids, String logical, int width, int height, long durationMs) throws Exception {
        JSONObject out = new JSONObject().put("camera_ids", new JSONArray(ids)).put("width", width).put("height", height)
            .put("logical_camera", logical == null ? JSONObject.NULL : logical).put("requested_duration_ms", durationMs);
        List<CameraDevice> devices = new ArrayList<>();
        List<CameraCaptureSession> sessions = new ArrayList<>();
        List<Stream> streams = new ArrayList<>();
        List<CountDownLatch> closures = new ArrayList<>();
        AtomicReference<String> failure = new AtomicReference<>();
        long started = android.os.SystemClock.elapsedRealtime();
        VisionPublisher publisher = null;
        java.nio.channels.FileChannel owner = null;
        java.nio.channels.FileLock ownership = null;
        try {
            checkThermalStatus();
            owner = java.nio.channels.FileChannel.open(new java.io.File(context.getFilesDir(), "camera-owner.lock").toPath(),
                java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.WRITE);
            ownership = owner.tryLock();
            if (ownership == null) throw new IllegalStateException("Another camera diagnostic is still closing; retry shortly");
            if (logical != null) {
                Set<String> physicalIds = manager.getCameraCharacteristics(logical).getPhysicalCameraIds();
                if (!physicalIds.containsAll(Arrays.asList(ids))) throw new IllegalStateException("requested physical IDs not exposed by logical camera");
            }
            CameraCharacteristics first = manager.getCameraCharacteristics(ids[0]);
            int rotation = Math.floorMod(first.get(CameraCharacteristics.SENSOR_ORIENTATION) +
                (Objects.equals(first.get(CameraCharacteristics.LENS_FACING), CameraCharacteristics.LENS_FACING_FRONT) ? displayDegrees : -displayDegrees), 360);
            publisher = new VisionPublisher(ipcDirectory, ids, rotation % 180 == 0 ? width : height, rotation % 180 == 0 ? height : width);
            for (String id : ids) streams.add(new Stream(id, width, height, publisher));
            if (logical != null) {
                CameraDevice device = open(logical, failure, closures); devices.add(device);
                sessions.add(session(device, streams, true, failure));
            } else {
                for (Stream stream : streams) {
                    CameraDevice device = open(stream.label, failure, closures); devices.add(device);
                    sessions.add(session(device, Collections.singletonList(stream), false, failure));
                }
            }
            long captureStarted = android.os.SystemClock.elapsedRealtime();
            long end = captureStarted + durationMs;
            while (android.os.SystemClock.elapsedRealtime() < end && failure.get() == null) {
                Thread.sleep(Math.max(1, Math.min(1000, end - android.os.SystemClock.elapsedRealtime())));
                checkThermalStatus();
                if (context.getSystemService(android.hardware.display.DisplayManager.class)
                    .getDisplay(android.view.Display.DEFAULT_DISPLAY).getRotation() * 90 != displayDegrees)
                    failure.compareAndSet(null, "display_rotation_changed");
                long now = android.os.SystemClock.elapsedRealtime();
                for (Stream stream : streams) {
                    long arrival = stream.lastArrivalMs;
                    if (now - (arrival == 0 ? captureStarted : arrival) > 2000)
                        failure.compareAndSet(null, "stale_camera:" + stream.label);
                }
                progress.update("Capturing " + Arrays.toString(ids) + " via " + (logical == null ? "independent sessions" : logical)
                    + " · " + width + "×" + height + " · " + (end - android.os.SystemClock.elapsedRealtime()) / 1000 + " s remaining");
            }
        } catch (Exception e) { failure.set(e.getClass().getSimpleName() + ": " + e.getMessage()); }
        finally {
            for (Stream stream : streams) stream.closing = true;
            for (CameraCaptureSession s : sessions) {
                try { s.stopRepeating(); s.abortCaptures(); }
                catch (CameraAccessException | IllegalStateException e) { failure.compareAndSet(null, "capture_stop_failed:" + e); }
                s.close();
            }
            for (CameraDevice d : devices) d.close();
            CountDownLatch drained = new CountDownLatch(1);
            handler.post(() -> {
                try { for (Stream stream : streams) stream.close(); }
                finally { drained.countDown(); }
            });
            boolean callbacksDrained = drained.await(3, TimeUnit.SECONDS);
            boolean camerasClosed = true;
            for (CountDownLatch closed : closures) camerasClosed &= closed.await(3, TimeUnit.SECONDS);
            out.put("image_callbacks_drained", callbacksDrained).put("camera_close_callbacks_complete", camerasClosed)
                .put("request_template", "record").put("stabilization", "disabled when advertised");
            if (!callbacksDrained || !camerasClosed) failure.compareAndSet(null, "camera_cleanup_timeout");
            if (publisher != null) publisher.close();
            if (ownership != null) ownership.release();
            if (owner != null) owner.close();
        }
        JSONArray results = new JSONArray();
        boolean healthy = streams.size() == ids.length;
        for (Stream stream : streams) {
            results.put(stream.result());
            healthy &= stream.stats.frames >= Math.max(10, durationMs * 15 / 1000)
                && stream.stats.timestampErrors == 0 && stream.stats.packingErrors == 0
                && stream.metadataFailures == 0 && stream.lostBuffers == 0;
        }
        if (streams.size() >= 2) {
            FrameStats a = streams.get(0).stats, b = streams.get(1).stats;
            long overlap = Math.min(a.lastTimestamp, b.lastTimestamp) - Math.max(a.firstTimestamp, b.firstTimestamp);
            out.put("overlap_ns", Math.max(0, overlap)).put("nearest_skew_p99_ns", pairingSkew(a.timestamps(), b.timestamps()));
            out.put("skew_sampling", "last 4096 timestamps per stream, shared overlap only");
            boolean distinct = !streams.get(0).label.equals(streams.get(1).label) && a.changedFrames > 0 && b.changedFrames > 0 && a.firstHash != b.firstHash;
            out.put("distinct_sensor_evidence", distinct);
            healthy &= overlap > 1_000_000_000L && distinct;
        }
        out.put("streams", results).put("elapsed_ms", android.os.SystemClock.elapsedRealtime() - started)
            .put("completed_requested_duration", failure.get() == null)
            .put("status", failure.get() == null && healthy ? "passed" : "failed");
        if (failure.get() != null) out.put("error", failure.get());
        else if (!healthy) out.put("error", "frame loss/metadata failure or insufficient independent, overlapping, monotonic frames");
        return out;
    }
    static long pairingSkew(long[] a, long[] b) {
        if (a.length == 0 || b.length == 0) return -1;
        long[] skew = new long[a.length];
        int j = 0, count = 0;
        for (int i = 0; i < a.length; i++) {
            // Frames captured before the second session starts cannot be paired.
            if (a[i] < b[0] || a[i] > b[b.length - 1]) continue;
            while (j + 1 < b.length && Math.abs(b[j + 1] - a[i]) < Math.abs(b[j] - a[i])) j++;
            skew[count++] = Math.abs(b[j] - a[i]);
        }
        if (count == 0) return -1;
        Arrays.sort(skew, 0, count);
        return skew[(int)Math.ceil((count - 1) * .99)];
    }
    public void close() throws InterruptedException {
        thread.quitSafely();
        thread.join(3000);
        if (thread.isAlive()) throw new IllegalStateException("Camera callback thread did not exit");
    }
}
