package ai.sunnypilot.fold3;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.Surface;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** Drain an opaque preview, optionally publishing reusable NV12 buffers on its GL thread. */
final class CameraPreviewSample implements AutoCloseable {
    private static final int WIDTH = 960, HEIGHT = 540;
    interface FrameSink { void receive(ByteBuffer nv12, long frame, long timestamp) throws Exception; }
    private final HandlerThread thread = new HandlerThread("camera-preview-sample");
    private final Handler handler;
    private final File sample;
    private final FrameSink sink;
    final FrameStats readback = new FrameStats();
    private final float[] transform = new float[16];
    private final FloatBuffer vertices = ByteBuffer.allocateDirect(16 * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
    private final ByteBuffer pixels = ByteBuffer.allocateDirect(WIDTH * HEIGHT * 4);
    private final ByteBuffer nv12 = ByteBuffer.allocateDirect(WIDTH * HEIGHT * 3 / 2);
    private EGLDisplay display = EGL14.EGL_NO_DISPLAY;
    private EGLContext context = EGL14.EGL_NO_CONTEXT;
    private EGLSurface eglSurface = EGL14.EGL_NO_SURFACE;
    private SurfaceTexture texture;
    private Surface surface;
    private int textureName, program;
    volatile long frames, sampleTimestampNs, timestampErrors, coalescedCallbacks, lastArrivalNs;
    volatile float[] sampleTextureTransform;
    volatile String error;
    private long lastTimestamp;

    CameraPreviewSample(File sample) throws Exception {
        this(sample, null);
    }

    CameraPreviewSample(File sample, FrameSink sink) throws Exception {
        this.sample = sample; this.sink = sink;
        vertices.put(new float[]{-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1}).rewind();
        thread.start(); handler = new Handler(thread.getLooper());
        CountDownLatch ready = new CountDownLatch(1);
        handler.post(() -> {
            try { initialize(); }
            catch (Exception e) { error = e.toString(); }
            finally { ready.countDown(); }
        });
        if (!ready.await(10, TimeUnit.SECONDS) || error != null) {
            close(); throw new IllegalStateException("Preview initialization failed: " + error);
        }
    }

    Surface surface() { return surface; }

    private void initialize() {
        display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (!EGL14.eglInitialize(display, null, 0, null, 0)) throw new IllegalStateException("eglInitialize");
        EGLConfig[] configs = new EGLConfig[1]; int[] count = new int[1];
        int[] attributes = {EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT, EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8, EGL14.EGL_ALPHA_SIZE, 8, EGL14.EGL_NONE};
        if (!EGL14.eglChooseConfig(display, attributes, 0, configs, 0, 1, count, 0) || count[0] == 0)
            throw new IllegalStateException("No preview EGL config");
        context = EGL14.eglCreateContext(display, configs[0], EGL14.EGL_NO_CONTEXT,
            new int[]{EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE}, 0);
        eglSurface = EGL14.eglCreatePbufferSurface(display, configs[0],
            new int[]{EGL14.EGL_WIDTH, WIDTH, EGL14.EGL_HEIGHT, HEIGHT, EGL14.EGL_NONE}, 0);
        if (!EGL14.eglMakeCurrent(display, eglSurface, eglSurface, context)) throw new IllegalStateException("eglMakeCurrent");
        int vertex = shader(GLES20.GL_VERTEX_SHADER,
            "attribute vec2 p; attribute vec2 uv; uniform mat4 transform; varying vec2 v; " +
            "void main(){gl_Position=vec4(p,0.,1.);v=(transform*vec4(uv,0.,1.)).xy;}");
        int fragment = shader(GLES20.GL_FRAGMENT_SHADER,
            "#extension GL_OES_EGL_image_external : require\nprecision mediump float;" +
            "uniform samplerExternalOES camera; varying vec2 v; void main(){gl_FragColor=texture2D(camera,v);}");
        program = GLES20.glCreateProgram(); GLES20.glAttachShader(program, vertex); GLES20.glAttachShader(program, fragment);
        GLES20.glLinkProgram(program); GLES20.glDeleteShader(vertex); GLES20.glDeleteShader(fragment);
        int[] linked = new int[1]; GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) throw new IllegalStateException(GLES20.glGetProgramInfoLog(program));
        int[] textures = new int[1]; GLES20.glGenTextures(1, textures, 0); textureName = textures[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureName);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        texture = new SurfaceTexture(textureName); texture.setDefaultBufferSize(1920, 1080);
        surface = new Surface(texture);
        texture.setOnFrameAvailableListener(ignored -> {
            if (error != null) return;
            try {
                texture.updateTexImage();
                long timestamp = texture.getTimestamp();
                if (timestamp == lastTimestamp) { coalescedCallbacks++; return; }
                if (timestamp < lastTimestamp) timestampErrors++;
                lastTimestamp = timestamp; frames++;
                lastArrivalNs = android.os.SystemClock.elapsedRealtimeNanos();
                if (sink != null || frames == 30 && sample != null) {
                    long started = System.nanoTime();
                    render();
                    if (sink != null) {
                        long hash = NativeBridge.rgbaToNv12(pixels, WIDTH, HEIGHT, nv12);
                        readback.add(timestamp, hash, System.nanoTime() - started);
                        readback.deliveryAge(android.os.SystemClock.elapsedRealtimeNanos() - timestamp);
                        sink.receive(nv12, frames, timestamp);
                    }
                    if (frames == 30 && sample != null) save();
                }
                if (frames == 30) { sampleTimestampNs = timestamp; sampleTextureTransform = transform.clone(); }
            } catch (Exception e) { error = e.toString(); }
        }, handler);
    }

    private static int shader(int type, String source) {
        int shader = GLES20.glCreateShader(type); GLES20.glShaderSource(shader, source); GLES20.glCompileShader(shader);
        int[] compiled = new int[1]; GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            String log = GLES20.glGetShaderInfoLog(shader); GLES20.glDeleteShader(shader);
            throw new IllegalStateException(log);
        }
        return shader;
    }

    private void render() {
        texture.getTransformMatrix(transform);
        vertices.position(0);
        GLES20.glViewport(0, 0, WIDTH, HEIGHT); GLES20.glUseProgram(program);
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(program, "transform"), 1, false, transform, 0);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "camera"), 0);
        int position = GLES20.glGetAttribLocation(program, "p"), uv = GLES20.glGetAttribLocation(program, "uv");
        GLES20.glEnableVertexAttribArray(position); GLES20.glEnableVertexAttribArray(uv);
        GLES20.glVertexAttribPointer(position, 2, GLES20.GL_FLOAT, false, 16, vertices);
        vertices.position(2); GLES20.glVertexAttribPointer(uv, 2, GLES20.GL_FLOAT, false, 16, vertices);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        pixels.clear();
        GLES20.glReadPixels(0, 0, WIDTH, HEIGHT, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixels);
        int result = GLES20.glGetError(); if (result != GLES20.GL_NO_ERROR) throw new IllegalStateException("Preview GL error: " + result);
    }

    private void save() throws Exception {
        pixels.rewind();
        Bitmap raw = Bitmap.createBitmap(WIDTH, HEIGHT, Bitmap.Config.ARGB_8888); raw.copyPixelsFromBuffer(pixels);
        Matrix flip = new Matrix(); flip.setScale(1, -1);
        Bitmap image = Bitmap.createBitmap(raw, 0, 0, WIDTH, HEIGHT, flip, false);
        try (FileOutputStream file = new FileOutputStream(sample)) {
            if (!image.compress(Bitmap.CompressFormat.PNG, 100, file)) throw new IllegalStateException("PNG encoding failed");
        } finally { image.recycle(); raw.recycle(); }
    }

    @Override public void close() throws Exception {
        CountDownLatch closed = new CountDownLatch(1);
        handler.post(() -> {
            try {
                if (texture != null) texture.setOnFrameAvailableListener(null);
                if (surface != null) surface.release();
                if (texture != null) texture.release();
                if (program != 0) GLES20.glDeleteProgram(program);
                if (textureName != 0) GLES20.glDeleteTextures(1, new int[]{textureName}, 0);
                if (display != EGL14.EGL_NO_DISPLAY) {
                    EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
                    if (eglSurface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(display, eglSurface);
                    if (context != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(display, context);
                    EGL14.eglTerminate(display); EGL14.eglReleaseThread();
                }
            } finally { closed.countDown(); }
        });
        boolean drained = closed.await(10, TimeUnit.SECONDS); thread.quitSafely(); thread.join(2000);
        if (!drained || thread.isAlive()) throw new IllegalStateException("Preview GL cleanup timed out");
    }
}
