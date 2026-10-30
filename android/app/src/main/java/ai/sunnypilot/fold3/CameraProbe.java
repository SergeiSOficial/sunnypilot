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

    CameraProbe(Context context, Progress progress) {
        manager = context.getSystemService(CameraManager.class);
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
        final JSONObject metadata;
        volatile long lastArrivalMs;
        Stream(String id, int width, int height) throws Exception {
            label = id;
            CameraCharacteristics c = manager.getCameraCharacteristics(id);
            metadata = new JSONObject().put("sensor_id", id).put("width", width).put("height", height)
                .put("sensor_orientation", c.get(CameraCharacteristics.SENSOR_ORIENTATION))
                .put("timestamp_source", c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE))
                .put("focal_lengths_mm", array(c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)));
            reader = ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 3);
            nv12 = ByteBuffer.allocateDirect(width * height * 3 / 2);
            reader.setOnImageAvailableListener(source -> {
                try (Image image = source.acquireLatestImage()) {
                    if (image == null) return;
                    Image.Plane[] p = image.getPlanes();
                    long start = System.nanoTime();
                    long hash = NativeBridge.packNv12(p[0].getBuffer().slice(), p[0].getRowStride(), p[0].getPixelStride(),
                        p[1].getBuffer().slice(), p[1].getRowStride(), p[1].getPixelStride(),
                        p[2].getBuffer().slice(), p[2].getRowStride(), p[2].getPixelStride(), width, height, nv12);
                    stats.add(image.getTimestamp(), hash, System.nanoTime() - start);
                    lastArrivalMs = android.os.SystemClock.elapsedRealtime();
                } catch (Exception e) { stats.packingErrors++; }
            }, handler);
        }
        JSONObject result() throws Exception {
            return new JSONObject(metadata.toString()).put("frames", stats.frames).put("fps", stats.fps())
                .put("first_timestamp_ns", stats.firstTimestamp).put("last_timestamp_ns", stats.lastTimestamp)
                .put("timestamp_errors", stats.timestampErrors).put("changed_frames", stats.changedFrames)
                .put("packing_errors", stats.packingErrors).put("packing_p99_ns", stats.percentile(true, .99))
                .put("frame_interval_p99_ns", stats.percentile(false, .99));
        }
        public void close() { reader.close(); }
    }
    private CameraDevice open(String id, AtomicReference<String> failure) throws Exception {
        CountDownLatch done = new CountDownLatch(1);
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
        }, handler);
        if (!done.await(8, TimeUnit.SECONDS)) {
            abandoned.set(true);
            if (result.get() != null) result.get().close();
            throw new IllegalStateException("open timeout: " + id);
        }
        CameraDevice device = result.get();
        if (device == null) throw new IllegalStateException(failure.get());
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
        CaptureRequest.Builder request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
        for (Stream stream : streams) request.addTarget(stream.reader.getSurface());
        Range<Integer>[] ranges = manager.getCameraCharacteristics(camera.getId()).get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        if (ranges != null && Arrays.asList(ranges).contains(new Range<>(30, 30)))
            request.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(30, 30));
        result.get().setRepeatingRequest(request.build(), new CameraCaptureSession.CaptureCallback() {
            public void onCaptureFailed(CameraCaptureSession s, CaptureRequest r, CaptureFailure f) {
                failure.compareAndSet(null, "capture_failed:camera=" + camera.getId() + ":reason=" + f.getReason()
                    + ":frame=" + f.getFrameNumber() + ":image_captured=" + f.wasImageCaptured());
            }
            public void onCaptureBufferLost(CameraCaptureSession s, CaptureRequest r, android.view.Surface target, long frame) {
                failure.compareAndSet(null, "capture_buffer_lost:camera=" + camera.getId() + ":frame=" + frame);
            }
        }, handler);
        return result.get();
    }
    JSONObject attempt(String[] ids, String logical, int width, int height, long durationMs) throws Exception {
        JSONObject out = new JSONObject().put("camera_ids", new JSONArray(ids)).put("width", width).put("height", height)
            .put("logical_camera", logical == null ? JSONObject.NULL : logical).put("requested_duration_ms", durationMs);
        List<CameraDevice> devices = new ArrayList<>();
        List<CameraCaptureSession> sessions = new ArrayList<>();
        List<Stream> streams = new ArrayList<>();
        AtomicReference<String> failure = new AtomicReference<>();
        long started = android.os.SystemClock.elapsedRealtime();
        try {
            if (logical != null) {
                Set<String> physicalIds = manager.getCameraCharacteristics(logical).getPhysicalCameraIds();
                if (!physicalIds.containsAll(Arrays.asList(ids))) throw new IllegalStateException("requested physical IDs not exposed by logical camera");
            }
            for (String id : ids) streams.add(new Stream(id, width, height));
            if (logical != null) {
                CameraDevice device = open(logical, failure); devices.add(device);
                sessions.add(session(device, streams, true, failure));
            } else {
                for (Stream stream : streams) {
                    CameraDevice device = open(stream.label, failure); devices.add(device);
                    sessions.add(session(device, Collections.singletonList(stream), false, failure));
                }
            }
            long end = android.os.SystemClock.elapsedRealtime() + durationMs;
            while (android.os.SystemClock.elapsedRealtime() < end && failure.get() == null) {
                Thread.sleep(Math.max(1, Math.min(1000, end - android.os.SystemClock.elapsedRealtime())));
                long now = android.os.SystemClock.elapsedRealtime();
                for (Stream stream : streams) {
                    long arrival = stream.lastArrivalMs;
                    if (now - (arrival == 0 ? started : arrival) > 2000)
                        failure.compareAndSet(null, "stale_camera:" + stream.label);
                }
                progress.update("Capturing " + Arrays.toString(ids) + " via " + (logical == null ? "independent sessions" : logical)
                    + " · " + width + "×" + height + " · " + (end - android.os.SystemClock.elapsedRealtime()) / 1000 + " s remaining");
            }
        } catch (Exception e) { failure.set(e.getClass().getSimpleName() + ": " + e.getMessage()); }
        finally {
            for (CameraCaptureSession s : sessions) s.close();
            for (CameraDevice d : devices) d.close();
            // Drain already dispatched image callbacks before reading their final statistics.
            CountDownLatch drained = new CountDownLatch(1); handler.post(drained::countDown); drained.await(3, TimeUnit.SECONDS);
        }
        JSONArray results = new JSONArray();
        boolean healthy = streams.size() == ids.length;
        for (Stream stream : streams) {
            results.put(stream.result());
            healthy &= stream.stats.frames >= Math.max(10, durationMs * 15 / 1000)
                && stream.stats.timestampErrors == 0 && stream.stats.packingErrors == 0;
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
        for (Stream stream : streams) stream.close();
        out.put("streams", results).put("elapsed_ms", android.os.SystemClock.elapsedRealtime() - started)
            .put("status", failure.get() == null && healthy ? "passed" : "failed");
        if (failure.get() != null) out.put("error", failure.get());
        else if (!healthy) out.put("error", "insufficient independent, overlapping, monotonic frames");
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
    public void close() { thread.quitSafely(); }
}
