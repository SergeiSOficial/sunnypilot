package ai.sunnypilot.fold3;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.view.Surface;
import android.hardware.camera2.*;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.SessionConfiguration;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
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

    static OutputConfiguration sensorOutput(OutputConfiguration output) {
        // DEFAULT smooths SurfaceTexture delivery; driving needs the latest sensor frame.
        if (Build.VERSION.SDK_INT >= 33)
            output.setTimestampBase(OutputConfiguration.TIMESTAMP_BASE_SENSOR);
        if (Build.VERSION.SDK_INT >= 34) output.setReadoutTimestampEnabled(false);
        return output;
    }

    private void checkThermalStatus() {
        int status = context.getSystemService(PowerManager.class).getCurrentThermalStatus();
        if (status >= PowerManager.THERMAL_STATUS_CRITICAL)
            throw new IllegalStateException("critical_thermal_status:" + status);
    }

    private void checkBenchThermalStatus() {
        checkThermalStatus();
        int status = context.getSystemService(PowerManager.class).getCurrentThermalStatus();
        if (status >= PowerManager.THERMAL_STATUS_MODERATE)
            throw new IllegalStateException("thermal_pause:" + status);
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
                .put("output_timestamp_base", Build.VERSION.SDK_INT >= 33 ? "sensor" : "default")
                .put("readout_timestamp_enabled", false)
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
            JSONArray privateSizes = new JSONArray(), fpsRanges = new JSONArray();
            Size[] advertisedPrivate = map == null ? null : map.getOutputSizes(SurfaceTexture.class);
            if (advertisedPrivate != null) for (Size size : advertisedPrivate)
                privateSizes.put(new JSONObject().put("width", size.getWidth()).put("height", size.getHeight())
                    .put("min_frame_duration_ns", map.getOutputMinFrameDuration(SurfaceTexture.class, size)));
            Range<Integer>[] advertisedFps = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
            if (advertisedFps != null) for (Range<Integer> range : advertisedFps)
                fpsRanges.put(new JSONArray().put(range.getLower()).put(range.getUpper()));
            cameras.put(camera.put("yuv_sizes", sizes).put("private_sizes", privateSizes).put("target_fps_ranges", fpsRanges));
            captureCapabilities(camera, c);
        }
        JSONArray concurrent = new JSONArray();
        for (Set<String> ids : manager.getConcurrentCameraIds()) concurrent.put(new JSONArray(ids));
        return out.put("cameras", cameras).put("concurrent_id_sets", concurrent);
    }
    JSONObject logicalAccess() throws Exception {
        JSONObject out = new JSONObject();
        // IDs observed in this phone's HAL, including its telephoto physical sensor.
        for (String id : new String[]{"20", "21", "23", "52"}) {
            JSONObject result = new JSONObject();
            try {
                CameraCharacteristics c = manager.getCameraCharacteristics(id);
                result.put("accessible", true).put("physical_ids", new JSONArray(c.getPhysicalCameraIds()))
                    .put("focal_lengths_mm", array(c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)));
                captureCapabilities(result, c);
                List<String> sessionKeys = new ArrayList<>();
                List<CaptureRequest.Key<?>> keys = c.getAvailableSessionKeys();
                if (keys != null) for (CaptureRequest.Key<?> key : keys) sessionKeys.add(key.getName());
                result.put("session_keys", new JSONArray(sessionKeys));
            } catch (CameraAccessException | IllegalArgumentException | SecurityException e) {
                result.put("accessible", false).put("error", e.toString());
            }
            out.put(id, result);
        }
        return out;
    }
    private static JSONArray array(Object value) throws org.json.JSONException {
        return value == null ? new JSONArray() : new JSONArray(value);
    }
    private static void captureCapabilities(JSONObject out, CameraCharacteristics c) throws org.json.JSONException {
        out.put("pipeline_max_depth", c.get(CameraCharacteristics.REQUEST_PIPELINE_MAX_DEPTH));
        if (Build.VERSION.SDK_INT >= 33)
            out.put("stream_use_cases", array(c.get(CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES)));
        if (Build.VERSION.SDK_INT >= 34)
            out.put("sensor_readout_timestamp_support", c.get(CameraCharacteristics.SENSOR_READOUT_TIMESTAMP));
        JSONArray highSpeedModes = new JSONArray();
        StreamConfigurationMap map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map != null) for (Size size : map.getHighSpeedVideoSizes()) {
            JSONArray ranges = new JSONArray();
            for (Range<Integer> range : map.getHighSpeedVideoFpsRangesFor(size))
                ranges.put(new JSONArray().put(range.getLower()).put(range.getUpper()));
            highSpeedModes.put(new JSONObject().put("width", size.getWidth()).put("height", size.getHeight()).put("fps_ranges", ranges));
        }
        out.put("constrained_high_speed_modes", highSpeedModes);
    }
    private static void recordCapture(FrameStats stats, CaptureResult result, long frame) {
        long before = android.os.SystemClock.elapsedRealtimeNanos();
        long monotonic = System.nanoTime();
        long after = android.os.SystemClock.elapsedRealtimeNanos();
        Long sensor = result.get(CaptureResult.SENSOR_TIMESTAMP);
        Long exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
        Long duration = result.get(CaptureResult.SENSOR_FRAME_DURATION);
        Long skew = result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW);
        Byte depth = result.get(CaptureResult.REQUEST_PIPELINE_DEPTH);
        stats.capture(sensor == null ? -1 : sensor, frame, exposure == null ? -1 : exposure,
            duration == null ? -1 : duration, before, monotonic, after, skew == null ? -1 : skew, depth == null ? -1 : depth);
    }
    static long captureTimestamp(FrameStats stats, long raw) {
        long elapsedBefore = android.os.SystemClock.elapsedRealtimeNanos(), monotonic = System.nanoTime();
        long elapsedAfter = android.os.SystemClock.elapsedRealtimeNanos();
        return stats.sensorToMonotonic(raw, elapsedBefore, monotonic, elapsedAfter);
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
        // Hidden vendor IDs need an explicit targeted trial. Some accepted
        // configurations crash the provider, so never open them in the default matrix.
        return ids;
    }
    private final class Stream implements AutoCloseable {
        final String label;
        final FrameStats stats;
        final ImageReader reader;
        final CameraPreviewSample gpuPreview;
        final CameraEvidence evidence;
        final ByteBuffer nv12;
        final ByteBuffer packed;
        final int rotation, outputWidth, outputHeight;
        final JSONObject metadata;
        final boolean directSensor;
        boolean physicalOutputAssigned;
        volatile long lastArrivalMs;
        volatile boolean closing;
        long metadataFailures, uncapturedRequests, lostBuffers, physicalMetadataFrames, publishedFrames, rejectedTimestamps;
        Float measuredFocalLength;
        Long measuredExposure, measuredFrameDuration;
        Integer measuredSensitivity;
        android.graphics.Rect captureCrop;
        Stream(String id, int width, int height, VisionPublisher publisher, boolean gpu, boolean recordEvidence, Size renderedSize) throws Exception {
            label = id;
            int renderedWidth = renderedSize.getWidth(), renderedHeight = renderedSize.getHeight();
            CameraCharacteristics c = manager.getCameraCharacteristics(id);
            directSensor = c.getPhysicalCameraIds().isEmpty();
            rotation = Math.floorMod(c.get(CameraCharacteristics.SENSOR_ORIENTATION) +
                (Objects.equals(c.get(CameraCharacteristics.LENS_FACING), CameraCharacteristics.LENS_FACING_FRONT) ? displayDegrees : -displayDegrees), 360);
            outputWidth = rotation % 180 == 0 ? renderedWidth : renderedHeight;
            outputHeight = rotation % 180 == 0 ? renderedHeight : renderedWidth;
            metadata = new JSONObject().put("sensor_id", id).put("width", width).put("height", height)
                .put("rendered_width", renderedWidth).put("rendered_height", renderedHeight)
                .put("resampling", width == renderedWidth && height == renderedHeight ? "none" : "full_frame_bilinear")
                .put("process_pid", android.os.Process.myPid())
                .put("session_started_elapsed_ns", android.os.SystemClock.elapsedRealtimeNanos())
                .put("raw_acquired_frames", 0)
                .put("image_rotation_applied_degrees", rotation).put("output_width", outputWidth).put("output_height", outputHeight)
                .put("display_degrees_at_session_start", displayDegrees).put("mirrored", false)
                .put("sensor_orientation", c.get(CameraCharacteristics.SENSOR_ORIENTATION))
                .put("timestamp_source", c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE))
                .put("intrinsic_calibration", array(c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)))
                .put("lens_distortion", array(c.get(CameraCharacteristics.LENS_DISTORTION)))
                .put("visionipc_valid", false).put("visionipc_stream", VisionPublisher.stream(id))
                .put("focal_lengths_mm", array(c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)));
            boolean realtime = Objects.equals(c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE),
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME);
            metadata.put("active_array", String.valueOf(c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)))
                .put("transport", gpu ? "private_gpu" : "direct_yuv");
            if (gpu) {
                evidence = null;
                reader = null;
                packed = null;
                nv12 = rotation == 0 ? null : ByteBuffer.allocateDirect(renderedWidth * renderedHeight * 3 / 2);
                gpuPreview = new CameraPreviewSample(null, (data, frame, timestamp) -> {
                    if (closing) return;
                    if (!realtime || timestamp <= 0 || timestamp > System.nanoTime()) {
                        rejectedTimestamps++;
                        return;
                    }
                    if (rotation != 0) NativeBridge.rotateNv12(data, renderedWidth, renderedHeight, rotation, nv12);
                    publisher.send(id, rotation == 0 ? data : nv12, outputWidth, outputHeight, frame, timestamp);
                    publishedFrames++;
                    lastArrivalMs = android.os.SystemClock.elapsedRealtime();
                }, width, height, renderedWidth, renderedHeight);
                stats = gpuPreview.readback;
                return;
            }
            gpuPreview = null;
            stats = new FrameStats();
            reader = ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 3);
            try {
                evidence = recordEvidence ? new CameraEvidence(context.getFilesDir(), "direct-" + id, outputWidth, outputHeight) : null;
            } catch (Exception e) {
                reader.close();
                throw e;
            }
            nv12 = ByteBuffer.allocateDirect(width * height * 3 / 2);
            packed = rotation == 0 ? nv12 : ByteBuffer.allocateDirect(width * height * 3 / 2);
            writeMetadata();
            reader.setOnImageAvailableListener(source -> {
                try (Image image = source.acquireLatestImage()) {
                    if (image == null || closing) return;
                    stats.acquired(image.getTimestamp(), android.os.SystemClock.elapsedRealtimeNanos());
                    Image.Plane[] p = image.getPlanes();
                    long start = System.nanoTime();
                    long hash = NativeBridge.packNv12(p[0].getBuffer().slice(), p[0].getRowStride(), p[0].getPixelStride(),
                        p[1].getBuffer().slice(), p[1].getRowStride(), p[1].getPixelStride(),
                        p[2].getBuffer().slice(), p[2].getRowStride(), p[2].getPixelStride(), width, height, packed);
                    if (rotation != 0) NativeBridge.rotateNv12(packed, width, height, rotation, nv12);
                    stats.add(image.getTimestamp(), hash, System.nanoTime() - start);
                    if (stats.frames == 1) {
                        // Raw acquisition can succeed while an invalid sensor clock
                        // correctly prevents VisionIPC publication. Keep this evidence
                        // separate from calibrated/fresh perception validity.
                        metadata.put("raw_acquired_frames", stats.frames)
                            .put("first_raw_sensor_timestamp_ns", image.getTimestamp());
                        writeMetadata();
                    }
                    long timestamp = image.getTimestamp();
                    if (realtime) timestamp = captureTimestamp(stats, timestamp);
                    if (realtime && timestamp > 0 && timestamp <= System.nanoTime()) {
                        publisher.send(id, nv12, outputWidth, outputHeight, stats.frames, timestamp);
                        publishedFrames++;
                        stats.deliveryAge(System.nanoTime() - timestamp);
                        if (evidence != null) evidence.write(nv12, timestamp);
                    } else {
                        rejectedTimestamps++;
                        if (realtime) stats.deliveryAge(System.nanoTime() - timestamp);
                    }
                    lastArrivalMs = android.os.SystemClock.elapsedRealtime();
                } catch (Exception e) { stats.packingErrors++; }
            }, handler);
        }
        void writeMetadata() throws Exception {
            java.nio.file.Path destination = new java.io.File(context.getFilesDir(), "camera-" + label + ".json").toPath();
            java.nio.file.Path temporary = destination.resolveSibling("camera-" + label + ".tmp");
            java.nio.file.Files.write(temporary, metadata.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
            java.nio.file.Files.move(temporary, destination, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
        boolean hasIdentityEvidence() {
            return (physicalOutputAssigned || directSensor) && stats.matchingCaptureTimestamps() >= 10;
        }
        Surface surface() { return gpuPreview == null ? reader.getSurface() : gpuPreview.surface(); }

        JSONObject result() throws Exception {
            FrameStats acquisition = gpuPreview == null ? stats : gpuPreview.acquisition;
            FrameStats delivery = gpuPreview == null ? stats : gpuPreview.sinkDelivery;
            JSONObject out = new JSONObject(metadata.toString())
                .put("capture_crop", captureCrop == null ? JSONObject.NULL : captureCrop.flattenToString())
                .put("raw_acquired_frames", gpuPreview == null ? stats.frames : gpuPreview.frames)
                .put("published_frames", publishedFrames).put("rejected_capture_timestamps", rejectedTimestamps)
                .put("capture_results_without_published_frame_min", Math.max(0, physicalMetadataFrames - publishedFrames))
                .put("evidence_frames", evidence == null ? 0 : evidence.frames()).put("frames", stats.frames).put("fps", stats.fps())
                .put("first_timestamp_ns", stats.firstTimestamp).put("last_timestamp_ns", stats.lastTimestamp)
                .put("timestamp_errors", stats.timestampErrors).put("changed_frames", stats.changedFrames)
                .put("metadata_failures", metadataFailures).put("uncaptured_requests", uncapturedRequests).put("lost_buffers", lostBuffers)
                .put("physical_metadata_frames", physicalMetadataFrames).put("measured_focal_length_mm", measuredFocalLength)
                .put("physical_output_assigned", physicalOutputAssigned).put("direct_sensor", directSensor)
                .put("image_capture_timestamp_matches", stats.matchingCaptureTimestamps())
                .put("sensor_identity_evidence", hasIdentityEvidence())
                .put("packing_errors", stats.packingErrors).put("packing_p99_ns", stats.percentile(true, .99))
                .put("delivery_age_available", Objects.equals(metadata.opt("timestamp_source"), CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME))
                .put("delivery_age_p50_ns", delivery.deliveryAgePercentile(.50)).put("delivery_age_p95_ns", delivery.deliveryAgePercentile(.95))
                .put("delivery_age_p99_ns", delivery.deliveryAgeP99()).put("delivery_over_100ms", delivery.lateDeliveries)
                .put("visionipc_publication_max_age_ns", delivery.maxDeliveryAge).put("visionipc_publication_over_50ms", delivery.deliveryOver50ms)
                .put("acquisition_age_p50_ns", acquisition.acquisitionAgePercentile(.50)).put("acquisition_age_p95_ns", acquisition.acquisitionAgePercentile(.95))
                .put("acquisition_age_p99_ns", acquisition.acquisitionAgeP99())
                .put("capture_result_age_p99_ns", stats.captureResultAgePercentile(.99))
                .put("capture_result_timestamp_errors", stats.captureResultTimestampErrors)
                .put("delivery_future_timestamps", stats.futureDeliveries)
                .put("clock_sample_columns", "sensor_raw,elapsed_before,monotonic,elapsed_after")
                .put("clock_samples_ns", array(stats.clockSamples()))
                .put("applied_exposure_ns", measuredExposure).put("applied_frame_duration_ns", measuredFrameDuration)
                .put("applied_sensitivity_iso", measuredSensitivity)
                .put("capture_sample_columns", FrameStats.CAPTURE_SAMPLE_COLUMNS)
                .put("capture_samples_ns", array(stats.captureSamples()))
                .put("frame_interval_p99_ns", stats.percentile(false, .99));
            if (gpuPreview != null) out.put("texture_transform", array(gpuPreview.sampleTextureTransform))
                .put("texture_frames", gpuPreview.frames).put("texture_duplicate_callbacks", gpuPreview.duplicateTextureCallbacks)
                .put("texture_coalesced_callbacks", gpuPreview.coalescedCallbacks)
                .put("gles_version", gpuPreview.glesVersion).put("gpu_conversion_active", gpuPreview.gpuConversionActive())
                .put("gpu_dropped_readbacks", gpuPreview.droppedReadbacks)
                .put("superseded_publications", gpuPreview.supersededPublications)
                .put("render_p99_ns", gpuPreview.renderWork.percentile(true, .99))
                .put("conversion_submit_p99_ns", gpuPreview.submitWork.percentile(true, .99))
                .put("cpu_conversion_p99_ns", gpuPreview.conversionWork.percentile(true, .99))
                .put("publication_p99_ns", gpuPreview.publicationWork.percentile(true, .99))
                .put("gpu_error", gpuPreview.error);
            return out;
        }
        public void close() {
            if (gpuPreview != null) {
                try { gpuPreview.close(); }
                catch (Exception e) { stats.packingErrors++; }
            }
            if (reader != null) { reader.setOnImageAvailableListener(null, null); reader.close(); }
            if (evidence != null) {
                try { evidence.close(); }
                catch (java.io.IOException e) { stats.packingErrors++; }
            }
        }
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
    private List<OutputConfiguration> outputs(List<Stream> streams, boolean physical, ImageReader preview) {
        List<OutputConfiguration> outputs = new ArrayList<>();
        for (Stream stream : streams) {
            OutputConfiguration output = sensorOutput(new OutputConfiguration(stream.surface()));
            if (physical) output.setPhysicalCameraId(stream.label);
            outputs.add(output);
        }
        if (preview != null) outputs.add(sensorOutput(new OutputConfiguration(preview.getSurface())));
        return outputs;
    }

    static Range<Integer> regularFps(Range<Integer>[] ranges, long minimumDuration) {
        Range<Integer> selected = null;
        if (ranges == null || minimumDuration < 0) return null;
        for (Range<Integer> range : ranges) {
            int maximum = range.getUpper();
            if (maximum <= 0 || minimumDuration > (1_000_000_000L + maximum - 1) / maximum) continue;
            if (selected == null || maximum > selected.getUpper() ||
                maximum == selected.getUpper() && range.getLower() > selected.getLower()) selected = range;
        }
        return selected;
    }

    private CameraCaptureSession session(CameraDevice camera, List<Stream> streams, boolean physical, ImageReader preview,
                                         AtomicReference<String> failure, int template, boolean highSpeed, long manualExposureNs,
                                         boolean lowLatency, long streamUseCase) throws Exception {
        CountDownLatch done = new CountDownLatch(1);
        AtomicBoolean abandoned = new AtomicBoolean();
        AtomicReference<CameraCaptureSession> result = new AtomicReference<>();
        List<OutputConfiguration> outputs = outputs(streams, physical, preview);
        if (lowLatency) for (OutputConfiguration output : outputs) output.setStreamUseCase(streamUseCase);
        for (Stream stream : streams) stream.physicalOutputAssigned = physical;
        Set<String> physicalIds = new HashSet<>();
        if (physical) for (Stream stream : streams) physicalIds.add(stream.label);
        CaptureRequest.Builder request = physical ? camera.createCaptureRequest(template, physicalIds) :
            camera.createCaptureRequest(template);
        CameraCharacteristics characteristics = manager.getCameraCharacteristics(camera.getId());
        Range<Integer>[] ranges = characteristics.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        if (highSpeed) request.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(120, 120));
        else if (lowLatency) {
            Stream stream = streams.get(0);
            Size size = new Size(stream.metadata.getInt("width"), stream.metadata.getInt("height"));
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            long minimumDuration = stream.gpuPreview != null ? map.getOutputMinFrameDuration(SurfaceTexture.class, size) :
                map.getOutputMinFrameDuration(ImageFormat.YUV_420_888, size);
            Range<Integer> selected = regularFps(ranges, minimumDuration);
            if (selected == null) throw new UnsupportedOperationException("No regular FPS range supports the selected stream duration");
            configureLowLatency(request, characteristics);
            request.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, selected);
            stream.metadata.put("minimum_frame_duration_ns", minimumDuration).put("requested_fps_min", selected.getLower())
                .put("requested_fps_max", selected.getUpper()).put("stream_use_case", streamUseCase)
                .put("low_latency_controls_in_session_parameters", true);
        }
        else if (ranges != null && Arrays.asList(ranges).contains(new Range<>(30, 30)))
            request.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(30, 30));
        if (manualExposureNs > 0) {
            Range<Integer> sensitivity = manager.getCameraCharacteristics(camera.getId()).get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE);
            request.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF);
            request.set(CaptureRequest.SENSOR_EXPOSURE_TIME, manualExposureNs);
            request.set(CaptureRequest.SENSOR_FRAME_DURATION, 33_333_333L);
            request.set(CaptureRequest.SENSOR_SENSITIVITY, sensitivity.clamp(100));
        }
        if (physical) for (String id : physicalIds) {
            request.setPhysicalCameraKey(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO, id);
            request.setPhysicalCameraKey(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON, id);
        }
        SessionConfiguration configuration = new SessionConfiguration(highSpeed ? SessionConfiguration.SESSION_HIGH_SPEED : SessionConfiguration.SESSION_REGULAR, outputs,
            command -> handler.post(command), new CameraCaptureSession.StateCallback() {
            public void onConfigured(CameraCaptureSession s) {
                result.set(s);
                if (abandoned.get()) s.close();
                done.countDown();
            }
            public void onConfigureFailed(CameraCaptureSession s) { failure.set("session_configuration_failed:" + camera.getId()); s.close(); done.countDown(); }
        });
        configuration.setSessionParameters(request.build());
        if ((physical || highSpeed || lowLatency) && !camera.isSessionConfigurationSupported(configuration))
            throw new UnsupportedOperationException("session configuration was rejected");
        camera.createCaptureSession(configuration);
        if (!done.await(8, TimeUnit.SECONDS)) {
            abandoned.set(true);
            if (result.get() != null) result.get().close();
            throw new IllegalStateException("session timeout: " + camera.getId());
        }
        if (result.get() == null) throw new IllegalStateException(failure.get());
        // Use only advertised low-latency modes; Samsung's preview processing can
        // otherwise consume the sensor-to-output budget before inference.
        configureLowLatency(request, manager.getCameraCharacteristics(camera.getId()));
        for (Stream stream : streams) request.addTarget(stream.surface());
        if (preview != null) request.addTarget(preview.getSurface());
        CameraCaptureSession.CaptureCallback callback = new CameraCaptureSession.CaptureCallback() {
            public void onCaptureStarted(CameraCaptureSession s, CaptureRequest r, long timestamp, long frame) {
                if (!physical) for (Stream stream : streams)
                    stream.stats.captureEvent(frame, timestamp, android.os.SystemClock.elapsedRealtimeNanos(), false);
            }
            public void onReadoutStarted(CameraCaptureSession s, CaptureRequest r, long timestamp, long frame) {
                if (!physical) for (Stream stream : streams)
                    stream.stats.captureEvent(frame, timestamp, android.os.SystemClock.elapsedRealtimeNanos(), true);
            }
            public void onCaptureCompleted(CameraCaptureSession s, CaptureRequest r, TotalCaptureResult result) {
                for (Stream stream : streams) {
                    CaptureResult sensor = physical ? result.getPhysicalCameraResults().get(stream.label) : result;
                    if (sensor != null && !stream.closing) {
                        stream.physicalMetadataFrames++;
                        stream.measuredFocalLength = sensor.get(CaptureResult.LENS_FOCAL_LENGTH);
                        stream.captureCrop = sensor.get(CaptureResult.SCALER_CROP_REGION);
                        stream.measuredExposure = sensor.get(CaptureResult.SENSOR_EXPOSURE_TIME);
                        stream.measuredFrameDuration = sensor.get(CaptureResult.SENSOR_FRAME_DURATION);
                        stream.measuredSensitivity = sensor.get(CaptureResult.SENSOR_SENSITIVITY);
                        recordCapture(stream.stats, sensor, result.getFrameNumber());
                    }
                }
            }
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
                for (Stream stream : streams) if (!stream.closing && stream.surface().equals(target)) stream.lostBuffers++;
            }
        };
        if (highSpeed) {
            CameraConstrainedHighSpeedCaptureSession session = (CameraConstrainedHighSpeedCaptureSession)result.get();
            session.setRepeatingBurst(session.createHighSpeedRequestList(request.build()), callback, handler);
        } else result.get().setRepeatingRequest(request.build(), callback, handler);
        return result.get();
    }

    private boolean samsungMultiView(CameraDevice camera) {
        return "20".equals(camera.getId()) && "com.samsung.sunnypilot.fold3".equals(context.getPackageName());
    }

    private OutputConfiguration samsungOutput(android.view.Surface surface, int option) throws Exception {
        // Samsung's public extension marks preview callbacks with option 1.
        return sensorOutput((OutputConfiguration) OutputConfiguration.class.getMethod("semCreateOutputConfiguration",
            int.class, android.view.Surface.class, int.class, int.class)
            .invoke(null, OutputConfiguration.SURFACE_GROUP_ID_NONE, surface, 0, option));
    }

    private void prepareSamsungMultiView(CameraDevice camera, JSONObject report) throws Exception {
        if (!(samsungMultiView(camera) || "1".equals(camera.getId()) &&
            "com.samsung.sunnypilot.fold3".equals(context.getPackageName())))
            throw new IllegalArgumentException("Samsung multiview needs the compatibility APK and rear 20 or front 1");
        String parameters = "samsungcamera=true;shootingmode=37;recording-fps=30;";
        report.put("samsung_parameters_requested", parameters).put("samsung_shooting_mode", 37);
        // Use Samsung's preinstalled helper, as its camera app does. Do not ship
        // vendor binaries or relax Android's hidden-API enforcement.
        ClassLoader loader = new dalvik.system.PathClassLoader("/system/framework/scamera_sdk_util.jar", getClass().getClassLoader());
        Class<?> helper = Class.forName("com.samsung.android.sdk.camera.impl.internal.CustomInterfaceHelper", true, loader);
        try {
            helper.getMethod("setSamsungParameter", CameraDevice.class, String.class).invoke(null, camera, parameters);
        } catch (java.lang.reflect.InvocationTargetException e) {
            throw new IllegalStateException("Samsung multiview parameters failed: " + e.getCause(), e);
        }
        report.put("samsung_parameters_applied", true);
    }
    private static void setIfSupported(CaptureRequest.Builder request, CaptureRequest.Key<Integer> key, int value, int[] modes) {
        if (modes != null && Arrays.stream(modes).anyMatch(mode -> mode == value)) request.set(key, value);
    }

    private static void configureLowLatency(CaptureRequest.Builder request, CameraCharacteristics characteristics) {
        setIfSupported(request, CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF,
            characteristics.get(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES));
        setIfSupported(request, CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF,
            characteristics.get(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES));
        setIfSupported(request, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
            characteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES));
        setIfSupported(request, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF,
            characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION));
    }

    /** Samsung's Director's View transport. Composite thumbnails are not physical streams. */
    JSONObject samsungMultiview(boolean front, long durationMs) throws Exception {
        return samsungMultiview(front, durationMs, false);
    }

    JSONObject samsungMultiview(boolean front, long durationMs, boolean publish) throws Exception {
        return samsungMultiview(front, durationMs, publish, "record");
    }

    JSONObject samsungMultiview(boolean front, long durationMs, boolean publish, String profile) throws Exception {
        return samsungMultiview(front, durationMs, publish, profile, false);
    }

    JSONObject samsungMultiview(boolean front, long durationMs, boolean publish, String profile, boolean continuousReadback) throws Exception {
        return samsungMultiview(front, durationMs, publish, profile, continuousReadback, false);
    }

    JSONObject samsungMultiview(boolean front, long durationMs, boolean publish, String profile,
                                boolean continuousReadback, boolean recordEvidence) throws Exception {
        if (recordEvidence && (publish || durationMs > 60_000))
            throw new IllegalArgumentException("Evidence recording is limited to 60 seconds in diagnostic mode");
        if (!Arrays.asList("record", "preview", "record-720p", "record-depth2").contains(profile))
            throw new IllegalArgumentException("Unsupported diagnostic camera profile");
        int previewWidth = profile.equals("record-720p") ? 1280 : 1920;
        int previewHeight = profile.equals("record-720p") ? 720 : 1080;
        int depth = profile.equals("record-depth2") ? 2 : 3;
        int template = profile.equals("preview") ? CameraDevice.TEMPLATE_PREVIEW : CameraDevice.TEMPLATE_RECORD;
        JSONObject out = new JSONObject().put("camera_ids", new JSONArray(front ? new String[]{"0", "2", "1"} : new String[]{"0", "2"}))
            .put("logical_camera", "20").put("backend", "samsung_multiview").put("requested_duration_ms", durationMs)
            .put("diagnostic_profile", profile).put("preview_width", previewWidth).put("preview_height", previewHeight)
            .put("image_reader_depth", depth).put("request_template", template)
            .put("output_timestamp_base", Build.VERSION.SDK_INT >= 33 ? "sensor" : "default")
            .put("readout_timestamp_enabled", false)
            .put("visionipc_valid", false).put("distinct_sensor_evidence", false);
        out.put("visionipc_publishing", publish).put("width", 960).put("height", 540)
            .put("stream_origins", new JSONObject().put("0", "logical20:selected-wide-preview-rgba-bt601")
                .put("2", "logical20:multiview-thumbnail-sensor-unverified").put("1", "front1:yuv"))
            .put("optical_calibration", "missing").put("mount_calibration", "missing")
            .put("continuous_readback", publish || continuousReadback || recordEvidence);
        List<CameraDevice> devices = new ArrayList<>();
        List<CameraCaptureSession> sessions = new ArrayList<>();
        List<ImageReader> readers = new ArrayList<>();
        List<CountDownLatch> closures = new ArrayList<>();
        List<FrameStats> stats = new ArrayList<>();
        List<CameraPreviewSample> previewSamples = new ArrayList<>();
        AtomicReference<String> failure = new AtomicReference<>();
        AtomicReference<int[]> roi = new AtomicReference<>();
        AtomicReference<JSONObject> captureMetadata = new AtomicReference<>();
        AtomicReference<VisionPublisher> publishing = new AtomicReference<>();
        long[] metadataFrames = {0}, previewFrames = {0}, thumbnailSampleTimestamp = {0};
        long started = android.os.SystemClock.elapsedRealtime();
        String error = null;
        boolean completed = false;
        java.nio.channels.FileChannel owner = null;
        java.nio.channels.FileLock ownership = null;
        CameraEvidence previewEvidence = null, thumbnailEvidence = null;
        try {
            owner = java.nio.channels.FileChannel.open(new java.io.File(context.getFilesDir(), "camera-owner.lock").toPath(),
                java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.WRITE);
            ownership = owner.tryLock();
            if (ownership == null) throw new IllegalStateException("Another camera diagnostic still owns the cameras");
            if (publish) {
                publishing.set(new VisionPublisher(ipcDirectory, front ? new String[]{"0", "2", "1"} : new String[]{"0", "2"}, 960, 540));
                java.nio.file.Path destination = new java.io.File(context.getFilesDir(), "vision-sources.json").toPath();
                java.nio.file.Path temporary = destination.resolveSibling("vision-sources.tmp");
                java.nio.file.Files.write(temporary, out.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
                java.nio.file.Files.move(temporary, destination, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            java.nio.file.Files.deleteIfExists(new java.io.File(context.getFilesDir(), "multiview-960x540.nv12").toPath());
            java.nio.file.Files.deleteIfExists(new java.io.File(context.getFilesDir(), "multiview-preview.png").toPath());
            if (recordEvidence) {
                previewEvidence = new CameraEvidence(context.getFilesDir(), "preview");
                thumbnailEvidence = new CameraEvidence(context.getFilesDir(), "thumbnail");
            }
            final CameraEvidence previewClip = previewEvidence, thumbnailClip = thumbnailEvidence;
            checkThermalStatus();
            CameraDevice rear = open("20", failure, closures); devices.add(rear);
            prepareSamsungMultiView(rear, out);
            if (front) {
                CameraDevice driver = open("1", failure, closures); devices.add(driver);
                prepareSamsungMultiView(driver, new JSONObject());
            }
            for (CameraDevice device : devices) {
                boolean logical = "20".equals(device.getId());
                Integer clock = manager.getCameraCharacteristics(device.getId()).get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
                if ((publish || recordEvidence) && !Integer.valueOf(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME).equals(clock))
                    throw new IllegalStateException("Unqualified multiview timestamp clock: " + device.getId() + ":" + clock);
                FrameStats frames = new FrameStats(); stats.add(frames);
                ImageReader yuv = ImageReader.newInstance(960, 540, ImageFormat.YUV_420_888, depth); readers.add(yuv);
                ByteBuffer packed = ByteBuffer.allocateDirect(960 * 540 * 3 / 2);
                yuv.setOnImageAvailableListener(source -> {
                    try (Image image = source.acquireLatestImage()) {
                        if (image == null) return;
                        frames.acquired(image.getTimestamp(), android.os.SystemClock.elapsedRealtimeNanos());
                        long before = System.nanoTime();
                        Image.Plane[] p = image.getPlanes();
                        long hash = NativeBridge.packNv12(p[0].getBuffer().slice(), p[0].getRowStride(), p[0].getPixelStride(),
                            p[1].getBuffer().slice(), p[1].getRowStride(), p[1].getPixelStride(),
                            p[2].getBuffer().slice(), p[2].getRowStride(), p[2].getPixelStride(), 960, 540, packed);
                        frames.add(image.getTimestamp(), hash, System.nanoTime() - before);
                        frames.deliveryAge(android.os.SystemClock.elapsedRealtimeNanos() - image.getTimestamp());
                        if (publishing.get() != null) {
                            long timestamp = captureTimestamp(frames, image.getTimestamp());
                            publishing.get().send(logical ? "2" : "1", packed, 960, 540, frames.frames, timestamp);
                        }
                        if (logical && thumbnailClip != null)
                            thumbnailClip.write(packed, captureTimestamp(frames, image.getTimestamp()));
                        if (!publish && !recordEvidence && logical && frames.frames == 30) {
                            // One local diagnostic sample; never publish a thumbnail as an independent sensor.
                            ByteBuffer copy = packed.duplicate(); copy.clear();
                            try (java.nio.channels.FileChannel file = java.nio.channels.FileChannel.open(
                                new java.io.File(context.getFilesDir(), "multiview-960x540.nv12").toPath(),
                                java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.TRUNCATE_EXISTING,
                                java.nio.file.StandardOpenOption.WRITE)) {
                                while (copy.hasRemaining()) file.write(copy);
                            }
                            thumbnailSampleTimestamp[0] = image.getTimestamp();
                        }
                    } catch (Exception e) { frames.packingErrors++; failure.compareAndSet(null, e.toString()); }
                }, handler);
                android.view.Surface previewSurface;
                if (logical) {
                    CameraPreviewSample sample = new CameraPreviewSample(publish || recordEvidence ? null : new java.io.File(context.getFilesDir(), "multiview-preview.png"),
                        publish ? (nv12, frame, timestamp) -> publishing.get().send("0", nv12, 960, 540, frame, timestamp)
                            : previewClip != null ? (nv12, frame, timestamp) -> previewClip.write(nv12, timestamp)
                            : continuousReadback ? (nv12, frame, timestamp) -> { } : null, previewWidth, previewHeight);
                    previewSamples.add(sample); previewSurface = sample.surface();
                } else {
                    ImageReader preview = ImageReader.newInstance(previewWidth, previewHeight, ImageFormat.PRIVATE, depth,
                        android.hardware.HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE); readers.add(preview);
                    preview.setOnImageAvailableListener(source -> {
                        try (Image image = source.acquireLatestImage()) { if (image != null) previewFrames[0]++; }
                    }, handler);
                    previewSurface = preview.getSurface();
                }
                // Samsung configures an untargeted recording output even for preview.
                ImageReader recording = ImageReader.newInstance(1920, 1080, ImageFormat.PRIVATE, 3); readers.add(recording);
                List<OutputConfiguration> outputs = Arrays.asList(samsungOutput(previewSurface, 1),
                    samsungOutput(recording.getSurface(), 0x41), samsungOutput(yuv.getSurface(), 1));
                CaptureRequest.Builder request = device.createCaptureRequest(template);
                request.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(30, 30));
                configureLowLatency(request, manager.getCameraCharacteristics(device.getId()));
                request.set(new CaptureRequest.Key<>("samsung.android.control.shootingMode", Integer.class), 37);
                request.set(new CaptureRequest.Key<>("samsung.android.control.cameraClient", Integer.class), 0);
                if (logical) {
                    // The installed Director's View app selects wide as its main view.
                    int[] selectedView = new int[45]; selectedView[9] = 1;
                    request.set(new CaptureRequest.Key<>("samsung.android.control.multiviewCropRoi", int[].class), selectedView);
                }
                request.addTarget(previewSurface);
                request.addTarget(yuv.getSurface());
                CountDownLatch configured = new CountDownLatch(1);
                AtomicReference<CameraCaptureSession> configuredSession = new AtomicReference<>();
                AtomicBoolean abandoned = new AtomicBoolean();
                SessionConfiguration configuration = new SessionConfiguration(SessionConfiguration.SESSION_REGULAR, outputs,
                    context.getMainExecutor(), new CameraCaptureSession.StateCallback() {
                        public void onConfigured(CameraCaptureSession session) {
                            configuredSession.set(session);
                            if (abandoned.get()) session.close();
                            configured.countDown();
                        }
                        public void onConfigureFailed(CameraCaptureSession session) {
                            failure.set("session_configuration_failed:" + device.getId()); session.close(); configured.countDown();
                        }
                    });
                configuration.setSessionParameters(request.build());
                device.createCaptureSession(configuration);
                if (!configured.await(8, TimeUnit.SECONDS) || configuredSession.get() == null) {
                    abandoned.set(true);
                    if (configuredSession.get() != null) configuredSession.get().close();
                    throw new IllegalStateException("Samsung multiview session failed: " + failure.get());
                }
                sessions.add(configuredSession.get());
                configuredSession.get().setRepeatingRequest(request.build(), new CameraCaptureSession.CaptureCallback() {
                    public void onCaptureStarted(CameraCaptureSession s, CaptureRequest r, long timestamp, long frame) {
                        frames.captureEvent(frame, timestamp, android.os.SystemClock.elapsedRealtimeNanos(), false);
                    }
                    public void onReadoutStarted(CameraCaptureSession s, CaptureRequest r, long timestamp, long frame) {
                        frames.captureEvent(frame, timestamp, android.os.SystemClock.elapsedRealtimeNanos(), true);
                    }
                    public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest r, TotalCaptureResult result) {
                        recordCapture(frames, result, result.getFrameNumber());
                        if (logical) {
                            metadataFrames[0]++;
                            int[] value = result.get(new CaptureResult.Key<>("samsung.android.control.multiviewCropRoi", int[].class));
                            if (value != null) roi.set(value.clone());
                            if (metadataFrames[0] % 30 == 1) {
                                JSONObject metadata = new JSONObject();
                                try {
                                    metadata.put("active_physical_id", result.get(CaptureResult.LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID))
                                        .put("physical_results", new JSONArray(result.getPhysicalCameraResults().keySet()))
                                        .put("noise_reduction_mode", result.get(CaptureResult.NOISE_REDUCTION_MODE))
                                        .put("edge_mode", result.get(CaptureResult.EDGE_MODE))
                                        .put("video_stabilization_mode", result.get(CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE))
                                        .put("optical_stabilization_mode", result.get(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE));
                                    for (CaptureResult.Key<?> key : result.getKeys()) {
                                        String name = key.getName();
                                        if (name.equals("samsung.android.control.runningPhysicalId") ||
                                            name.equals("samsung.android.control.multiSensorSensitivityList") ||
                                            name.equals("samsung.android.control.multiExposureTimeList") ||
                                            name.equals("samsung.android.lens.info.currentInfo")) metadata.put(name, array(result.get(key)));
                                    }
                                } catch (Exception e) { try { metadata.put("error", e.toString()); } catch (Exception ignored) { } }
                                captureMetadata.set(metadata);
                            }
                        }
                    }
                    public void onCaptureFailed(CameraCaptureSession session, CaptureRequest r, CaptureFailure f) {
                        failure.compareAndSet(null, "capture_failed:" + device.getId() + ":" + f.getReason());
                    }
                }, handler);
            }
            long captureStartedNs = android.os.SystemClock.elapsedRealtimeNanos();
            long end = android.os.SystemClock.elapsedRealtime() + durationMs;
            while (android.os.SystemClock.elapsedRealtime() < end) {
                if (failure.get() != null) throw new IllegalStateException(failure.get());
                long nowNs = android.os.SystemClock.elapsedRealtimeNanos();
                for (int i = 0; i < stats.size(); i++) synchronized (stats.get(i)) {
                    long timestamp = stats.get(i).lastTimestamp;
                    if (nowNs - (timestamp == 0 ? captureStartedNs : timestamp) > 2_000_000_000L)
                        throw new IllegalStateException("stale_multiview_yuv:" + devices.get(i).getId());
                }
                for (CameraPreviewSample sample : previewSamples) {
                    if (sample.error != null) throw new IllegalStateException(sample.error);
                    if (nowNs - (sample.lastArrivalNs == 0 ? captureStartedNs : sample.lastArrivalNs) > 2_000_000_000L)
                        throw new IllegalStateException("stale_multiview_preview");
                }
                checkThermalStatus();
                progress.update("Samsung multiview transport: " + stats.get(0).frames + " composite frames; transmission disabled");
                Thread.sleep(100);
            }
            completed = true;
        } catch (Exception e) { error = e.toString(); }
        finally {
            for (CameraCaptureSession session : sessions) {
                try { session.stopRepeating(); session.abortCaptures(); } catch (CameraAccessException | IllegalStateException ignored) { }
                session.close();
            }
            for (CameraDevice device : devices) device.close();
            for (CountDownLatch closed : closures) if (!closed.await(5, TimeUnit.SECONDS)) error = "Camera close callback timed out";
            for (ImageReader reader : readers) reader.setOnImageAvailableListener(null, null);
            CountDownLatch drained = new CountDownLatch(1); handler.post(drained::countDown);
            if (!drained.await(5, TimeUnit.SECONDS)) error = "Image callback drain timed out";
            for (ImageReader reader : readers) reader.close();
            for (CameraPreviewSample sample : previewSamples) {
                try { sample.close(); } catch (Exception e) { error = e.toString(); }
            }
            for (CameraEvidence clip : new CameraEvidence[]{previewEvidence, thumbnailEvidence}) {
                if (clip == null) continue;
                try { clip.close(); } catch (Exception e) { error = e.toString(); }
            }
            if (publishing.get() != null) publishing.get().close();
            if (ownership != null) ownership.release();
            if (owner != null) owner.close();
        }
        if (recordEvidence)
            out.put("evidence_frames", new JSONObject()
                .put("preview", previewEvidence == null ? 0 : previewEvidence.frames())
                .put("thumbnail", thumbnailEvidence == null ? 0 : thumbnailEvidence.frames()));
        FrameStats rear = stats.isEmpty() ? new FrameStats() : stats.get(0);
        out.put("composite_frames", rear.frames).put("composite_fps", rear.fps()).put("preview_frames", previewFrames[0])
            .put("metadata_frames", metadataFrames[0]).put("multiview_crop_roi", array(roi.get()))
            .put("capture_metadata", captureMetadata.get())
            .put("timestamp_errors", rear.timestampErrors).put("packing_errors", rear.packingErrors)
            .put("packing_p99_ns", rear.percentile(true, .99)).put("late_deliveries", rear.lateDeliveries)
            .put("future_deliveries", rear.futureDeliveries)
            .put("delivery_age_p50_ns", rear.deliveryAgePercentile(.50)).put("delivery_age_p95_ns", rear.deliveryAgePercentile(.95))
            .put("delivery_age_p99_ns", rear.deliveryAgeP99()).put("completed_requested_duration", completed)
            .put("acquisition_age_p50_ns", rear.acquisitionAgePercentile(.50)).put("acquisition_age_p95_ns", rear.acquisitionAgePercentile(.95))
            .put("acquisition_age_p99_ns", rear.acquisitionAgeP99())
            .put("capture_result_age_p50_ns", rear.captureResultAgePercentile(.50))
            .put("capture_result_age_p95_ns", rear.captureResultAgePercentile(.95))
            .put("capture_result_age_p99_ns", rear.captureResultAgePercentile(.99))
            .put("capture_result_timestamp_errors", rear.captureResultTimestampErrors)
            .put("capture_sample_columns", FrameStats.CAPTURE_SAMPLE_COLUMNS)
            .put("capture_samples_ns", array(rear.captureSamples()))
            .put("elapsed_ms", android.os.SystemClock.elapsedRealtime() - started)
            .put("transport_status", completed && error == null && rear.frames >= durationMs * 15 / 1000 &&
                rear.timestampErrors == 0 && rear.packingErrors == 0 && metadataFrames[0] >= 10 && roi.get() != null ? "passed" : "failed")
            .put("status", "blocked").put("qualification_blocker", "Composite transport does not establish independent physical sensor delivery");
        if (front && stats.size() == 2) {
            FrameStats driver = stats.get(1);
            out.put("front_frames", driver.frames).put("front_fps", driver.fps())
                .put("front_packing_p99_ns", driver.percentile(true, .99)).put("front_late_deliveries", driver.lateDeliveries)
                .put("front_future_deliveries", driver.futureDeliveries)
                .put("front_timestamp_errors", driver.timestampErrors).put("front_packing_errors", driver.packingErrors)
                .put("front_delivery_age_p50_ns", driver.deliveryAgePercentile(.50)).put("front_delivery_age_p95_ns", driver.deliveryAgePercentile(.95))
                .put("front_delivery_age_p99_ns", driver.deliveryAgeP99())
                .put("front_acquisition_age_p50_ns", driver.acquisitionAgePercentile(.50)).put("front_acquisition_age_p95_ns", driver.acquisitionAgePercentile(.95))
                .put("front_acquisition_age_p99_ns", driver.acquisitionAgeP99())
                .put("front_capture_result_age_p99_ns", driver.captureResultAgePercentile(.99))
                .put("front_capture_result_timestamp_errors", driver.captureResultTimestampErrors)
                .put("front_capture_samples_ns", array(driver.captureSamples()));
            if (driver.frames < durationMs * 15 / 1000 || driver.timestampErrors != 0 || driver.packingErrors != 0)
                out.put("transport_status", "failed");
        }
        if (!previewSamples.isEmpty()) {
            CameraPreviewSample sample = previewSamples.get(0);
            out.put("rear_preview_frames", sample.frames).put("preview_sample_timestamp_ns", sample.sampleTimestampNs)
                .put("preview_sample_texture_transform", array(sample.sampleTextureTransform))
                .put("thumbnail_sample_timestamp_ns", thumbnailSampleTimestamp[0])
                .put("preview_timestamp_errors", sample.timestampErrors).put("preview_coalesced_callbacks", sample.coalescedCallbacks)
                .put("preview_frames", sample.frames + previewFrames[0]);
            out.put("preview_readback_frames", sample.readback.frames).put("preview_readback_p99_ns", sample.readback.percentile(true, .99))
                .put("preview_readback_p50_ns", sample.readback.percentile(true, .50)).put("preview_readback_p95_ns", sample.readback.percentile(true, .95))
                .put("preview_gles_version", sample.glesVersion).put("preview_render_p99_ns", sample.renderWork.percentile(true, .99))
                .put("preview_render_p50_ns", sample.renderWork.percentile(true, .50)).put("preview_render_p95_ns", sample.renderWork.percentile(true, .95))
                .put("preview_submit_p99_ns", sample.submitWork.percentile(true, .99))
                .put("preview_submit_p50_ns", sample.submitWork.percentile(true, .50)).put("preview_submit_p95_ns", sample.submitWork.percentile(true, .95))
                .put("preview_cpu_conversion_p99_ns", sample.conversionWork.percentile(true, .99))
                .put("preview_cpu_conversion_p50_ns", sample.conversionWork.percentile(true, .50)).put("preview_cpu_conversion_p95_ns", sample.conversionWork.percentile(true, .95))
                .put("preview_hash_p99_ns", sample.hashWork.percentile(true, .99))
                .put("preview_hash_p50_ns", sample.hashWork.percentile(true, .50)).put("preview_hash_p95_ns", sample.hashWork.percentile(true, .95))
                .put("preview_sink_delivery_age_p99_ns", sample.sinkDelivery.deliveryAgeP99())
                .put("preview_sink_late_deliveries", sample.sinkDelivery.lateDeliveries)
                .put("preview_publication_p99_ns", sample.publicationWork.percentile(true, .99))
                .put("preview_publication_p50_ns", sample.publicationWork.percentile(true, .50)).put("preview_publication_p95_ns", sample.publicationWork.percentile(true, .95))
                .put("preview_dropped_readbacks", sample.droppedReadbacks)
                .put("preview_superseded_publications", sample.supersededPublications)
                .put("preview_duplicate_texture_callbacks", sample.duplicateTextureCallbacks)
                .put("preview_delivery_age_p50_ns", sample.readback.deliveryAgePercentile(.50)).put("preview_delivery_age_p95_ns", sample.readback.deliveryAgePercentile(.95))
                .put("preview_delivery_age_p99_ns", sample.readback.deliveryAgeP99()).put("preview_late_deliveries", sample.readback.lateDeliveries)
                .put("preview_acquisition_age_p50_ns", sample.acquisition.acquisitionAgePercentile(.50)).put("preview_acquisition_age_p95_ns", sample.acquisition.acquisitionAgePercentile(.95))
                .put("preview_acquisition_age_p99_ns", sample.acquisition.acquisitionAgeP99())
                .put("preview_future_deliveries", sample.readback.futureDeliveries);
            out.put("clock_sample_columns", "sensor_raw,elapsed_before,monotonic,elapsed_after")
                .put("preview_clock_samples_ns", array(sample.readback.clockSamples()))
                .put("thumbnail_clock_samples_ns", array(rear.clockSamples()));
            if (publish) out.put("rear_candidate_skew_p99_ns", pairingSkew(sample.readback.timestamps(), rear.timestamps()))
                .put("rear_candidate_overlap_ns", Math.max(0, Math.min(sample.readback.lastTimestamp, rear.lastTimestamp) -
                    Math.max(sample.readback.firstTimestamp, rear.firstTimestamp)))
                .put("preview_changed_frames", sample.readback.changedFrames).put("thumbnail_changed_frames", rear.changedFrames);
            if (sample.error != null) out.put("preview_error", sample.error);
            if (sample.frames < durationMs * 15 / 1000 || sample.timestampErrors != 0 || sample.sampleTimestampNs == 0 || sample.error != null)
                out.put("transport_status", "failed");
        }
        if (error != null) out.put("error", error);
        return out;
    }

    JSONObject attempt(String[] ids, String logical, int width, int height, long durationMs) throws Exception {
        return attempt(ids, logical, width, height, durationMs, "record", false, false);
    }

    JSONObject attempt(String[] ids, String logical, int width, int height, long durationMs,
                       String profile, boolean gpu, boolean evidence) throws Exception {
        boolean highSpeed = "high-speed-120".equals(profile);
        long manualExposureNs = "manual-2ms-low-latency".equals(profile) ? 2_000_000L :
            ("manual-8ms".equals(profile) || "manual-8ms-low-latency".equals(profile) ? 8_333_333L : 0L);
        boolean lowLatency = Arrays.asList("low-latency", "low-latency-preview", "low-latency-record", "manual-8ms-low-latency", "manual-2ms-low-latency").contains(profile);
        long streamUseCase = "low-latency-preview".equals(profile) ? CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES_PREVIEW :
            "low-latency-record".equals(profile) ? CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES_VIDEO_RECORD :
            CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES_DEFAULT;
        if (!"record".equals(profile) && !"preview".equals(profile) && !highSpeed && manualExposureNs == 0 && !lowLatency)
            throw new IllegalArgumentException("Unknown public request profile");
        int template = "preview".equals(profile) || lowLatency ? CameraDevice.TEMPLATE_PREVIEW : CameraDevice.TEMPLATE_RECORD;
        JSONObject out = new JSONObject().put("camera_ids", new JSONArray(ids)).put("width", width).put("height", height)
            .put("request_template", profile).put("transport", gpu ? "private_gpu" : "direct_yuv")
            .put("visionipc_geometry_qualified", false).put("rear_identity_qualified", false)
            .put("logical_camera", logical == null ? JSONObject.NULL : logical).put("requested_duration_ms", durationMs)
            .put("startup_order", "open_all_then_configure").put("high_speed_fps", highSpeed ? 120 : 0);
        JSONArray opened = new JSONArray();
        out.put("opened_camera_ids", opened);
        List<CameraDevice> devices = new ArrayList<>();
        List<CameraCaptureSession> sessions = new ArrayList<>();
        List<Stream> streams = new ArrayList<>();
        List<ImageReader> previews = new ArrayList<>();
        List<CountDownLatch> closures = new ArrayList<>();
        AtomicReference<String> failure = new AtomicReference<>();
        long started = android.os.SystemClock.elapsedRealtime();
        VisionPublisher publisher = null;
        java.nio.channels.FileChannel owner = null;
        java.nio.channels.FileLock ownership = null;
        String blocker = publicRouteBlocker(ids, logical, width, height, gpu);
        if (blocker != null) return out.put("status", "blocked").put("error", blocker);
        if (lowLatency) {
            if (logical != null) return out.put("status", "blocked").put("error", "Low-latency profiles require public independent sessions");
            for (String id : ids) {
                long[] advertised = manager.getCameraCharacteristics(id).get(CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES);
                if (streamUseCase != CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES_DEFAULT &&
                    (advertised == null || Arrays.stream(advertised).noneMatch(value -> value == streamUseCase)))
                    return out.put("status", "blocked").put("error", "Stream use case is not advertised: " + id + ":" + streamUseCase);
            }
        }
        if (manualExposureNs > 0) {
            if (logical != null) return out.put("status", "blocked").put("error", "Manual exposure requires public independent camera sessions");
            JSONObject requested = new JSONObject();
            for (String id : ids) {
                CameraCharacteristics c = manager.getCameraCharacteristics(id);
                int[] capabilities = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
                Range<Long> exposure = c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE);
                Range<Integer> sensitivity = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE);
                Long maximumDuration = c.get(CameraCharacteristics.SENSOR_INFO_MAX_FRAME_DURATION);
                StreamConfigurationMap map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                Size size = new Size(width, height);
                long minimumDuration = gpu ? map.getOutputMinFrameDuration(SurfaceTexture.class, size) :
                    map.getOutputMinFrameDuration(ImageFormat.YUV_420_888, size);
                if (!Objects.equals(c.get(CameraCharacteristics.LENS_FACING), CameraCharacteristics.LENS_FACING_BACK)
                    || capabilities == null || Arrays.stream(capabilities).noneMatch(v -> v == CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR)
                    || exposure == null || !exposure.contains(manualExposureNs) || sensitivity == null
                    || maximumDuration == null || maximumDuration < 33_333_333L || minimumDuration > 33_333_333L)
                    return out.put("status", "blocked").put("error", "Manual rear exposure at 30 FPS is not advertised: " + id + ":" + manualExposureNs);
                requested.put(id, new JSONObject().put("exposure_ns", manualExposureNs).put("frame_duration_ns", 33_333_333L)
                    .put("sensitivity_iso", sensitivity.clamp(100)).put("auto_exposure", false));
            }
            out.put("requested_sensor_settings", requested).put("exposure_diagnostic_only", true);
        }
        if (highSpeed) {
            if (!gpu || logical != null || ids.length != 1 || evidence)
                return out.put("status", "blocked").put("error", "High-speed diagnostics require one public GPU surface");
            StreamConfigurationMap map = manager.getCameraCharacteristics(ids[0]).get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Size size = new Size(width, height);
            if (map == null || !Arrays.asList(map.getHighSpeedVideoSizes()).contains(size) ||
                !Arrays.asList(map.getHighSpeedVideoFpsRangesFor(size)).contains(new Range<>(120, 120)))
                return out.put("status", "blocked").put("error", "Fixed 120 FPS is not advertised for this camera and size");
        }
        try {
            checkBenchThermalStatus();
            owner = java.nio.channels.FileChannel.open(new java.io.File(context.getFilesDir(), "camera-owner.lock").toPath(),
                java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.WRITE);
            ownership = owner.tryLock();
            if (ownership == null) throw new IllegalStateException("Another camera diagnostic is still closing; retry shortly");
            CameraCharacteristics first = manager.getCameraCharacteristics(ids[0]);
            int rotation = Math.floorMod(first.get(CameraCharacteristics.SENSOR_ORIENTATION) +
                (Objects.equals(first.get(CameraCharacteristics.LENS_FACING), CameraCharacteristics.LENS_FACING_FRONT) ? displayDegrees : -displayDegrees), 360);
            // Keep the advertised high-speed capture size while using the existing
            // model-size GPU renderer. Preserve aspect ratio and record both sizes.
            Size renderedSize = highSpeed && width > 960 && width * 9 == height * 16 ? new Size(960, 540) : new Size(width, height);
            out.put("rendered_width", renderedSize.getWidth()).put("rendered_height", renderedSize.getHeight());
            publisher = new VisionPublisher(ipcDirectory, ids, rotation % 180 == 0 ? renderedSize.getWidth() : renderedSize.getHeight(),
                rotation % 180 == 0 ? renderedSize.getHeight() : renderedSize.getWidth());
            for (String id : ids) streams.add(new Stream(id, width, height, publisher, gpu, evidence, renderedSize));
            if (logical != null) {
                Set<String> physicalIds = manager.getCameraCharacteristics(logical).getPhysicalCameraIds();
                List<Stream> physical = new ArrayList<>(), independent = new ArrayList<>();
                for (Stream stream : streams) {
                    if (physicalIds.contains(stream.label)) physical.add(stream);
                    else independent.add(stream);
                }
                if (physical.size() < 2) throw new IllegalArgumentException("Logical camera does not expose two requested physical sensors");
                // Mark the auxiliary logical output as a GPU viewfinder. PRIVATE
                // with usage 0 is rejected by Samsung's multi-camera pipeline.
                ImageReader preview = ImageReader.newInstance(width, height, ImageFormat.PRIVATE, 3,
                    android.hardware.HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE);
                previews.add(preview);
                preview.setOnImageAvailableListener(source -> {
                    try (Image image = source.acquireLatestImage()) { }
                }, handler);
                out.put("auxiliary_logical_preview", true);
                if (!independent.isEmpty()) {
                    CameraCaptureSession.StateCallback callback = new CameraCaptureSession.StateCallback() {
                        public void onConfigured(CameraCaptureSession session) {}
                        public void onConfigureFailed(CameraCaptureSession session) {}
                    };
                    Map<String, SessionConfiguration> config = new LinkedHashMap<>();
                    config.put(logical, new SessionConfiguration(SessionConfiguration.SESSION_REGULAR,
                        outputs(physical, true, preview), Runnable::run, callback));
                    for (Stream stream : independent)
                        config.put(stream.label, new SessionConfiguration(SessionConfiguration.SESSION_REGULAR,
                            outputs(List.of(stream), false, null), Runnable::run, callback));
                    if (!manager.isConcurrentSessionConfigurationSupported(config))
                        throw new UnsupportedOperationException("public concurrent output configuration was rejected");
                }
                CameraDevice device = open(logical, failure, closures); devices.add(device); opened.put(logical);
                for (Stream stream : independent) {
                    devices.add(open(stream.label, failure, closures)); opened.put(stream.label);
                }
                sessions.add(session(device, physical, true, preview, failure, template, false, 0L, false, 0));
                for (int i = 0; i < independent.size(); i++)
                    sessions.add(session(devices.get(i + 1), Collections.singletonList(independent.get(i)), false, null, failure, template, false, 0L, false, 0));
            } else {
                for (Stream stream : streams) {
                    devices.add(open(stream.label, failure, closures)); opened.put(stream.label);
                }
                // Opening after configuring another camera can reserve resources needed for concurrency.
                for (int i = 0; i < streams.size(); i++)
                    sessions.add(session(devices.get(i), Collections.singletonList(streams.get(i)), false, null, failure, template, highSpeed, manualExposureNs, lowLatency, streamUseCase));
            }
            long captureStarted = android.os.SystemClock.elapsedRealtime();
            long end = captureStarted + durationMs;
            while (android.os.SystemClock.elapsedRealtime() < end && failure.get() == null) {
                Thread.sleep(Math.max(1, Math.min(1000, end - android.os.SystemClock.elapsedRealtime())));
                checkBenchThermalStatus();
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
        } catch (UnsupportedOperationException e) { out.put("support_blocker", e.getMessage()); }
        catch (Exception e) { failure.set(e.getClass().getSimpleName() + ": " + e.getMessage()); }
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
                try {
                    for (Stream stream : streams) stream.close();
                    for (ImageReader preview : previews) { preview.setOnImageAvailableListener(null, null); preview.close(); }
                }
                finally { drained.countDown(); }
            });
            boolean callbacksDrained = drained.await(3, TimeUnit.SECONDS);
            boolean camerasClosed = true;
            for (CountDownLatch closed : closures) camerasClosed &= closed.await(3, TimeUnit.SECONDS);
            out.put("image_callbacks_drained", callbacksDrained).put("camera_close_callbacks_complete", camerasClosed)
                .put("stabilization", "disabled when advertised");
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
                && stream.metadataFailures == 0 && stream.lostBuffers == 0 && stream.physicalMetadataFrames >= 10
                && Objects.equals(stream.metadata.opt("timestamp_source"), CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME)
                && stream.stats.futureDeliveries == 0 && stream.stats.lateDeliveries == 0
                && (stream.gpuPreview == null || stream.gpuPreview.error == null && stream.gpuPreview.timestampErrors == 0
                    && stream.gpuPreview.sinkDelivery.futureDeliveries == 0 && stream.gpuPreview.sinkDelivery.lateDeliveries == 0);
        }
        JSONArray pairs = new JSONArray();
        for (int i = 0; i < streams.size(); i++) for (int j = i + 1; j < streams.size(); j++) {
            Stream left = streams.get(i), right = streams.get(j);
            long overlap = Math.min(left.stats.lastTimestamp, right.stats.lastTimestamp) -
                Math.max(left.stats.firstTimestamp, right.stats.firstTimestamp);
            boolean differentContent = left.stats.changedFrames > 0 && right.stats.changedFrames > 0
                && left.stats.firstHash != right.stats.firstHash;
            boolean distinct = !left.label.equals(right.label) && left.hasIdentityEvidence() && right.hasIdentityEvidence();
            pairs.put(new JSONObject().put("camera_ids", new JSONArray(new String[]{left.label, right.label}))
                .put("overlap_ns", Math.max(0, overlap)).put("distinct_sensor_evidence", distinct)
                .put("different_image_content", differentContent)
                .put("nearest_skew_p99_ns", pairingSkew(left.stats.timestamps(), right.stats.timestamps())));
            healthy &= overlap > 1_000_000_000L && distinct && differentContent;
        }
        out.put("sensor_pairs", pairs);
        if (streams.size() >= 2) {
            FrameStats a = streams.get(0).stats, b = streams.get(1).stats;
            long overlap = Math.min(a.lastTimestamp, b.lastTimestamp) - Math.max(a.firstTimestamp, b.firstTimestamp);
            out.put("overlap_ns", Math.max(0, overlap)).put("nearest_skew_p99_ns", pairingSkew(a.timestamps(), b.timestamps()));
            out.put("skew_sampling", "last 4096 timestamps per stream, shared overlap only");
            out.put("distinct_sensor_evidence", pairs.getJSONObject(0).getBoolean("distinct_sensor_evidence"));
        }
        boolean withinBudget = healthy;
        for (Stream stream : streams) {
            FrameStats delivered = stream.gpuPreview == null ? stream.stats : stream.gpuPreview.sinkDelivery;
            withinBudget &= delivered.maxDeliveryAge > 0 && delivered.deliveryOver50ms == 0 && delivered.futureDeliveries == 0
                && stream.rejectedTimestamps == 0 && stream.stats.fps() >= 20
                && stream.publishedFrames >= durationMs * 20 / 1000;
        }
        out.put("camera_only_50ms_status", withinBudget && failure.get() == null && !out.has("support_blocker") ? "passed" : "failed");
        out.put("streams", results).put("elapsed_ms", android.os.SystemClock.elapsedRealtime() - started)
            .put("completed_requested_duration", failure.get() == null)
            .put("status", failure.get() == null && healthy ? "passed" : "failed");
        if (failure.get() != null) out.put("error", failure.get());
        else if (out.has("support_blocker")) out.put("status", "blocked").put("error", out.getString("support_blocker"));
        else if (!healthy) out.put("error", "frame loss/metadata failure or insufficient independent, overlapping, monotonic frames");
        return out;
    }
    private String publicRouteBlocker(String[] ids, String logical, int width, int height, boolean gpu) throws CameraAccessException {
        // On the stock Fold 3, closing concurrent wide/front PRIVATE sessions
        // has crashed CamX in PipelineDestroyCb. Leave this route closed.
        if (gpu && logical == null && "SM-F926B".equals(Build.MODEL)
            && Arrays.asList(ids).containsAll(List.of("0", "1")))
            return "Fold 3 wide/front PRIVATE sessions excluded after camera provider teardown crash";
        Set<String> publicIds = Set.of(manager.getCameraIdList());
        Set<String> physical = Set.of();
        if (logical != null) {
            if (!publicIds.contains(logical)) return "logical ID is outside the public camera list";
            physical = manager.getCameraCharacteristics(logical).getPhysicalCameraIds();
            Set<String> devices = new HashSet<>();
            devices.add(logical);
            int matched = 0;
            for (String id : ids) {
                if (physical.contains(id)) matched++;
                else devices.add(id);
            }
            if (matched < 2) return "public logical ID does not expose two requested physical sensors";
            if (devices.size() > 1 && manager.getConcurrentCameraIds().stream().noneMatch(set -> set.containsAll(devices)))
                return "logical rear and front combination is not advertised";
        } else if (ids.length == 3 && manager.getConcurrentCameraIds().stream().noneMatch(set -> set.containsAll(Arrays.asList(ids)))) {
            return "three independent devices are not advertised for concurrent capture";
        }
        for (String id : ids) {
            if (!publicIds.contains(id) && !physical.contains(id)) return "sensor is outside the public camera configuration";
            StreamConfigurationMap map = manager.getCameraCharacteristics(id).get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Size[] sizes = map == null ? null : gpu ? map.getOutputSizes(SurfaceTexture.class) : map.getOutputSizes(ImageFormat.YUV_420_888);
            if (sizes == null || !Arrays.asList(sizes).contains(new Size(width, height)))
                return "requested output size is not advertised for sensor " + id;
        }
        return null;
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
