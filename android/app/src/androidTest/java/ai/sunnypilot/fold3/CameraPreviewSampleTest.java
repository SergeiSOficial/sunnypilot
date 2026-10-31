package ai.sunnypilot.fold3;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.SystemClock;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class CameraPreviewSampleTest {
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
        })) {
            Paint red = new Paint(); red.setColor(Color.RED);
            long deadline = SystemClock.elapsedRealtime() + 8000;
            while (preview.sampleTimestampNs == 0 && preview.error == null && SystemClock.elapsedRealtime() < deadline) {
                Canvas canvas = preview.surface().lockCanvas(null);
                canvas.drawColor(Color.BLUE); canvas.drawRect(0, 0, canvas.getWidth(), canvas.getHeight() / 2, red);
                preview.surface().unlockCanvasAndPost(canvas);
                SystemClock.sleep(35);
            }
            assertNull(preview.error); assertTrue(preview.frames >= 30); assertTrue(preview.sampleTimestampNs > 0);
            assertTrue(delivered.get() >= 30); assertTrue(preview.readback.percentile(true, .99) > 0);
            Bitmap image = BitmapFactory.decodeFile(file.getAbsolutePath()); assertNotNull(image);
            try {
                assertEquals(960, image.getWidth()); assertEquals(540, image.getHeight());
                assertEquals(Color.RED, image.getPixel(100, 100)); assertEquals(Color.BLUE, image.getPixel(800, 400));
            } finally { image.recycle(); }
        } finally { file.delete(); }
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
