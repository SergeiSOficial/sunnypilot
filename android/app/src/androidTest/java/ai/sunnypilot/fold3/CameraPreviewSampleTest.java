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
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class CameraPreviewSampleTest {
    @Test public void opaquePreviewReadbackPreservesPixelsAndOrientation() throws Exception {
        File file = new File(InstrumentationRegistry.getInstrumentation().getTargetContext().getCacheDir(), "preview-test.png");
        try (CameraPreviewSample preview = new CameraPreviewSample(file)) {
            Paint red = new Paint(); red.setColor(Color.RED);
            long deadline = SystemClock.elapsedRealtime() + 8000;
            while (preview.sampleTimestampNs == 0 && preview.error == null && SystemClock.elapsedRealtime() < deadline) {
                Canvas canvas = preview.surface().lockCanvas(null);
                canvas.drawColor(Color.BLUE); canvas.drawRect(0, 0, canvas.getWidth(), canvas.getHeight() / 2, red);
                preview.surface().unlockCanvasAndPost(canvas);
                SystemClock.sleep(35);
            }
            assertNull(preview.error); assertTrue(preview.frames >= 30); assertTrue(preview.sampleTimestampNs > 0);
            Bitmap image = BitmapFactory.decodeFile(file.getAbsolutePath()); assertNotNull(image);
            try {
                assertEquals(960, image.getWidth()); assertEquals(540, image.getHeight());
                assertEquals(Color.RED, image.getPixel(100, 100)); assertEquals(Color.BLUE, image.getPixel(800, 400));
            } finally { image.recycle(); }
        } finally { file.delete(); }
    }
}
