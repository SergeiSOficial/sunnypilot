package ai.sunnypilot.fold3;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.hardware.camera2.params.OutputConfiguration;
import android.os.Build;
import android.os.SystemClock;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class CameraPreviewSampleTest {
    @Test public void manualRearExposureRejectsFrontWithoutOpeningCamera() throws Exception {
        android.content.Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        android.hardware.camera2.CameraManager manager = context.getSystemService(android.hardware.camera2.CameraManager.class);
        for (String id : manager.getCameraIdList()) {
            android.hardware.camera2.CameraCharacteristics info = manager.getCameraCharacteristics(id);
            if (!Integer.valueOf(android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT).equals(
                    info.get(android.hardware.camera2.CameraCharacteristics.LENS_FACING))) continue;
            android.util.Size size = info.get(android.hardware.camera2.CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                .getOutputSizes(android.graphics.ImageFormat.YUV_420_888)[0];
            try (CameraProbe probe = new CameraProbe(context, message -> {})) {
                org.json.JSONObject result = probe.attempt(new String[]{id}, null, size.getWidth(), size.getHeight(),
                    1000, "manual-8ms", false, false);
                assertEquals("blocked", result.getString("status"));
                assertEquals(0, result.getJSONArray("opened_camera_ids").length());
                assertTrue(result.toString(), result.getString("error").toLowerCase(java.util.Locale.ROOT).contains("manual rear exposure"));
            }
            return;
        }
        org.junit.Assume.assumeTrue("No front camera advertised", false);
    }

    @Test public void cameraOutputUsesExposureStartWithoutPreviewSmoothing() throws Exception {
        try (CameraPreviewSample preview = new CameraPreviewSample(null, null)) {
            OutputConfiguration output = CameraProbe.sensorOutput(new OutputConfiguration(preview.surface()));
            if (Build.VERSION.SDK_INT >= 33)
                assertEquals(OutputConfiguration.TIMESTAMP_BASE_SENSOR, output.getTimestampBase());
            if (Build.VERSION.SDK_INT >= 34) assertFalse(output.isReadoutTimestampEnabled());
        }
    }

    @Test public void opaquePreviewReadbackPreservesPixelsAndOrientation() throws Exception {
        File file = new File(InstrumentationRegistry.getInstrumentation().getTargetContext().getCacheDir(), "preview-test.png");
        AtomicLong delivered = new AtomicLong();
        try (CameraPreviewSample preview = new CameraPreviewSample(file, (nv12, frame, timestamp) -> {
            // Same orientation as the PNG; BT.601 limited-range red and blue.
            assertEquals(82, nv12.get(100 * 960 + 100) & 255);
            assertEquals(41, nv12.get(400 * 960 + 800) & 255);
            assertEquals(90, nv12.get(960 * 540 + 50 * 960 + 100) & 255);
            assertEquals(240, nv12.get(960 * 540 + 50 * 960 + 101) & 255);
            assertTrue(timestamp > 0); delivered.set(frame);
        }, 1280, 720)) {
            Paint red = new Paint(); red.setColor(Color.RED);
            long deadline = SystemClock.elapsedRealtime() + 8000;
            while (preview.sampleTimestampNs == 0 && preview.error == null && SystemClock.elapsedRealtime() < deadline) {
                Canvas canvas = preview.surface().lockCanvas(null);
                assertEquals(1280, canvas.getWidth()); assertEquals(720, canvas.getHeight());
                canvas.drawColor(Color.BLUE); canvas.drawRect(0, 0, canvas.getWidth(), canvas.getHeight() / 2, red);
                preview.surface().unlockCanvasAndPost(canvas);
                SystemClock.sleep(35);
            }
            while (delivered.get() < 30 && preview.error == null && SystemClock.elapsedRealtime() < deadline) SystemClock.sleep(5);
            assertNull(preview.error); assertTrue(preview.frames >= 30); assertTrue(preview.sampleTimestampNs > 0);
            assertTrue(delivered.get() >= 30); assertTrue(preview.readback.percentile(true, .99) > 0);
            Bitmap image = BitmapFactory.decodeFile(file.getAbsolutePath()); assertNotNull(image);
            try {
                assertEquals(960, image.getWidth()); assertEquals(540, image.getHeight());
                assertEquals(Color.RED, image.getPixel(100, 100)); assertEquals(Color.BLUE, image.getPixel(800, 400));
            } finally { image.recycle(); }
        } finally { file.delete(); }
    }

    @Test public void slowPublisherOwnsItsBufferAndDropsQueuedFrames() throws Exception {
        AtomicLong delivered = new AtomicLong();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        CameraPreviewSample preview = new CameraPreviewSample(null, (nv12, frame, timestamp) -> {
            try {
                int y = nv12.get(100 * 960 + 100) & 255;
                SystemClock.sleep(120);
                assertEquals("Staging buffer changed while publication was in flight", y, nv12.get(100 * 960 + 100) & 255);
                delivered.incrementAndGet();
            } catch (Throwable error) { failure.set(error); }
        });
        try {
            // Functional backpressure check; emulator rendering is not a camera FPS benchmark.
            long deadline = SystemClock.elapsedRealtime() + 8000;
            int frame = 0;
            while ((preview.frames <= 30 || delivered.get() <= 3 || preview.supersededPublications == 0)
                    && SystemClock.elapsedRealtime() < deadline && preview.error == null) {
                Canvas canvas = preview.surface().lockCanvas(null);
                canvas.drawColor(frame++ % 2 == 0 ? Color.RED : Color.BLUE);
                preview.surface().unlockCanvasAndPost(canvas);
                SystemClock.sleep(20);
            }
            assertNull(preview.error);
            assertNull(failure.get());
            assertTrue("Acquisition stalled behind sink", preview.frames > 30);
            assertTrue(delivered.get() > 3);
            assertTrue("Pending publications were not bounded", preview.supersededPublications > 0);
        } finally { preview.close(); }
        long stopped = delivered.get();
        SystemClock.sleep(150);
        assertEquals("Sink ran after close", stopped, delivered.get());
        assertNull(failure.get());
    }

    @Test public void rgbaConversionRejectsTruncatedOddAndAliasedBuffers() {
        ByteBuffer valid = ByteBuffer.allocateDirect(16);
        for (int mode = 0; mode < 5; mode++) {
            try {
                NativeBridge.rgbaToNv12(mode == 0 ? null : mode == 1 ? ByteBuffer.allocateDirect(15) : valid,
                    mode == 2 ? 1 : 2, 2, mode == 3 ? ByteBuffer.allocateDirect(5) : mode == 4 ? valid : ByteBuffer.allocateDirect(6));
                fail("Invalid conversion accepted: " + mode);
            } catch (IllegalArgumentException expected) { }
        }
    }
}
