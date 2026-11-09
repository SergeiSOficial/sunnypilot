package ai.sunnypilot.fold3;

import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.GLES30;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import java.nio.ByteBuffer;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class GpuNv12Test {
    @Test public void gpuMatchesCpuForEveryByteAndBoundsReadbacks() throws Exception {
        EGLDisplay display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        assertTrue(EGL14.eglInitialize(display, null, 0, null, 0));
        EGLConfig[] configs = new EGLConfig[1]; int[] count = new int[1];
        assertTrue(EGL14.eglChooseConfig(display, new int[]{EGL14.EGL_RENDERABLE_TYPE, 0x40,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT, EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8, EGL14.EGL_ALPHA_SIZE, 8, EGL14.EGL_NONE},
            0, configs, 0, 1, count, 0));
        assertEquals(1, count[0]);
        EGLContext context = EGL14.eglCreateContext(display, configs[0], EGL14.EGL_NO_CONTEXT,
            new int[]{EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE}, 0);
        EGLSurface surface = EGL14.eglCreatePbufferSurface(display, configs[0],
            new int[]{EGL14.EGL_WIDTH, 32, EGL14.EGL_HEIGHT, 16, EGL14.EGL_NONE}, 0);
        assertTrue(EGL14.eglMakeCurrent(display, surface, surface, context));
        try (GpuNv12 gpu = new GpuNv12(32, 16)) {
            for (int seed : new int[]{42, 7, 123}) {
                byte[] pixels = new byte[32 * 16 * 4]; new Random(seed).nextBytes(pixels);
                ByteBuffer rgba = ByteBuffer.allocateDirect(pixels.length); rgba.put(pixels).rewind();
                ByteBuffer expected = ByteBuffer.allocateDirect(gpu.bytes);
                long hash = NativeBridge.rgbaToNv12(rgba, 32, 16, expected);
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, gpu.textures[0]);
                GLES30.glTexSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0, 32, 16, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, rgba);
                gpu.submit(1, seed, 1); gpu.submit(2, seed, 1); gpu.submit(3, seed, 1);
                assertEquals(1, gpu.dropped); assertTrue(gpu.pending());
                // Blocking is confined to this equivalence test, never the camera callback.
                GLES30.glFinish(); int[] delivered = {0};
                gpu.drain((actual, frame, timestamp, started) -> {
                    assertEquals(++delivered[0], frame); assertEquals(seed, timestamp);
                    for (int i = 0; i < gpu.bytes; i++) assertEquals("byte " + i + " seed " + seed, expected.get(i), actual.get(i));
                    assertEquals(hash, NativeBridge.nv12Hash(actual, 32, 16));
                });
                assertEquals(2, delivered[0]); assertFalse(gpu.pending()); gpu.dropped = 0;
            }
        } finally {
            EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
            EGL14.eglDestroySurface(display, surface); EGL14.eglDestroyContext(display, context);
            EGL14.eglTerminate(display); EGL14.eglReleaseThread();
        }
    }
}
