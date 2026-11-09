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
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/** Latest-only GL acquisition and publication with explicitly owned NV12 buffers. */
final class CameraPreviewSample implements AutoCloseable {
    private final int width, height;
    interface FrameSink { void receive(ByteBuffer nv12, long frame, long timestamp) throws Exception; }
    private final HandlerThread thread = new HandlerThread("camera-preview-sample");
    private final Handler handler;
    private final HandlerThread callbacks = new HandlerThread("camera-preview-events");
    private final HandlerThread publications = new HandlerThread("camera-preview-publish");
    private final Handler callbackHandler, publicationHandler;
    private final AtomicBoolean frameScheduled = new AtomicBoolean(), publicationScheduled = new AtomicBoolean();
    private final AtomicReference<Delivery> pending = new AtomicReference<>();
    private final ArrayBlockingQueue<Delivery> available = new ArrayBlockingQueue<>(3);
    private final class Delivery {
        final ByteBuffer data = ByteBuffer.allocateDirect(width * height * 3 / 2);
        long frame, timestamp, started;
    }
    private final File sample;
    private final int inputWidth, inputHeight;
    private final FrameSink sink;
    final FrameStats readback = new FrameStats();
    final FrameStats acquisition = new FrameStats(), sinkDelivery = new FrameStats();
    final FrameStats renderWork = new FrameStats(), submitWork = new FrameStats();
    final FrameStats conversionWork = new FrameStats(), hashWork = new FrameStats(), publicationWork = new FrameStats();
    private final float[] transform = new float[16];
    private final FloatBuffer vertices = ByteBuffer.allocateDirect(16 * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
    private final ByteBuffer pixels, nv12;
    private EGLDisplay display = EGL14.EGL_NO_DISPLAY;
    private EGLContext context = EGL14.EGL_NO_CONTEXT;
    private EGLSurface eglSurface = EGL14.EGL_NO_SURFACE;
    private SurfaceTexture texture;
    private Surface surface;
    private int textureName, program, transformLocation, positionLocation, uvLocation;
    private GpuNv12 gpu;
    private boolean draining;
    private volatile boolean closing;
    volatile int glesVersion;
    volatile long frames, sampleTimestampNs, timestampErrors, coalescedCallbacks, lastArrivalNs;
    volatile long droppedReadbacks;
    volatile long supersededPublications, duplicateTextureCallbacks;
    volatile float[] sampleTextureTransform;
    volatile String error;
    private long lastTimestamp;

    CameraPreviewSample(File sample) throws Exception {
        this(sample, null);
    }

    CameraPreviewSample(File sample, FrameSink sink) throws Exception {
        this(sample, sink, 1920, 1080);
    }

    CameraPreviewSample(File sample, FrameSink sink, int inputWidth, int inputHeight) throws Exception {
        this(sample, sink, inputWidth, inputHeight, 960, 540);
    }

    CameraPreviewSample(File sample, FrameSink sink, int inputWidth, int inputHeight, int width, int height) throws Exception {
        this.sample = sample; this.sink = sink; this.inputWidth = inputWidth; this.inputHeight = inputHeight;
        this.width = width; this.height = height;
        pixels = ByteBuffer.allocateDirect(width * height * 4);
        nv12 = ByteBuffer.allocateDirect(width * height * 3 / 2);
        vertices.put(new float[]{-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1}).rewind();
        thread.start(); handler = new Handler(thread.getLooper());
        callbacks.start(); callbackHandler = new Handler(callbacks.getLooper());
        publications.start(); publicationHandler = new Handler(publications.getLooper());
        for (int i = 0; i < 3; i++) available.add(new Delivery());
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
    boolean gpuConversionActive() { return glesVersion == 3 && sink != null; }

    private void initialize() {
        display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (!EGL14.eglInitialize(display, null, 0, null, 0)) throw new IllegalStateException("eglInitialize");
        EGLConfig[] configs = new EGLConfig[1]; int[] count = new int[1];
        for (int version : new int[]{3, 2}) {
            int[] attributes = {EGL14.EGL_RENDERABLE_TYPE, version == 3 ? 0x40 : EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT, EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8, EGL14.EGL_ALPHA_SIZE, 8, EGL14.EGL_NONE};
            if (!EGL14.eglChooseConfig(display, attributes, 0, configs, 0, 1, count, 0) || count[0] == 0) continue;
            context = EGL14.eglCreateContext(display, configs[0], EGL14.EGL_NO_CONTEXT,
                new int[]{EGL14.EGL_CONTEXT_CLIENT_VERSION, version, EGL14.EGL_NONE}, 0);
            if (context != EGL14.EGL_NO_CONTEXT) { glesVersion = version; break; }
            EGL14.eglGetError();
        }
        if (context == EGL14.EGL_NO_CONTEXT) throw new IllegalStateException("No preview EGL context");
        eglSurface = EGL14.eglCreatePbufferSurface(display, configs[0],
            new int[]{EGL14.EGL_WIDTH, width, EGL14.EGL_HEIGHT, height, EGL14.EGL_NONE}, 0);
        if (!EGL14.eglMakeCurrent(display, eglSurface, eglSurface, context)) throw new IllegalStateException("eglMakeCurrent");
        if (glesVersion == 3 && sink != null) gpu = new GpuNv12(width, height);
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
        GLES20.glUseProgram(program);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "camera"), 0);
        transformLocation = GLES20.glGetUniformLocation(program, "transform");
        positionLocation = GLES20.glGetAttribLocation(program, "p");
        uvLocation = GLES20.glGetAttribLocation(program, "uv");
        int[] textures = new int[1]; GLES20.glGenTextures(1, textures, 0); textureName = textures[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureName);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        texture = new SurfaceTexture(textureName); texture.setDefaultBufferSize(inputWidth, inputHeight);
        surface = new Surface(texture);
        texture.setOnFrameAvailableListener(ignored -> {
            if (closing || error != null) return;
            if (frameScheduled.getAndSet(true)) { coalescedCallbacks++; return; }
            handler.post(() -> {
                frameScheduled.set(false);
                acquire();
            });
        }, callbackHandler);
    }

    private void acquire() {
            if (error != null || closing) return;
            try {
                texture.updateTexImage();
                long timestamp = texture.getTimestamp();
                if (timestamp == lastTimestamp) { duplicateTextureCallbacks++; return; }
                if (timestamp < lastTimestamp) timestampErrors++;
                lastTimestamp = timestamp; frames++;
                lastArrivalNs = android.os.SystemClock.elapsedRealtimeNanos();
                acquisition.acquired(timestamp, lastArrivalNs);
                if (sink != null || frames == 30 && sample != null) {
                    long started = System.nanoTime();
                    boolean sampleFrame = frames == 30 && sample != null;
                    render(gpu == null || sampleFrame);
                    renderWork.add(timestamp, 0, System.nanoTime() - started);
                    if (sink != null) {
                        if (gpu == null) {
                            long converting = System.nanoTime();
                            NativeBridge.rgbaToNv12(pixels, width, height, nv12);
                            conversionWork.add(timestamp, 0, System.nanoTime() - converting);
                            deliver(nv12, frames, timestamp, started);
                        } else {
                            long packing = System.nanoTime();
                            gpu.submit(frames, timestamp, started);
                            droppedReadbacks = gpu.dropped;
                            submitWork.add(timestamp, 0, System.nanoTime() - packing);
                            if (!draining) { draining = true; handler.post(this::drainGpu); }
                        }
                    }
                    if (sampleFrame) save();
                }
                if (frames == 30) { sampleTimestampNs = timestamp; sampleTextureTransform = transform.clone(); }
            } catch (Exception e) { error = e.toString(); }
    }

    private void deliver(ByteBuffer data, long frame, long timestamp, long started) {
        if (closing) return;
        Delivery delivery = available.poll();
        if (delivery == null) { supersededPublications++; return; }
        // GPU/PBO and CPU staging buffers are reused as soon as this callback returns.
        delivery.data.clear(); data.position(0); data.limit(width * height * 3 / 2);
        delivery.data.put(data).flip();
        delivery.frame = frame; delivery.timestamp = timestamp; delivery.started = started;
        Delivery old = pending.getAndSet(delivery);
        if (old != null) { available.add(old); supersededPublications++; }
        if (!publicationScheduled.getAndSet(true)) publicationHandler.post(this::publishLatest);
    }

    private void publishLatest() {
        Delivery delivery = pending.getAndSet(null);
        try {
            if (delivery != null && !closing) publish(delivery.data, delivery.frame, delivery.timestamp, delivery.started);
        } catch (Exception e) { error = e.toString(); }
        finally {
            if (delivery != null) available.add(delivery);
            publicationScheduled.set(false);
            if (pending.get() != null && !publicationScheduled.getAndSet(true)) publicationHandler.post(this::publishLatest);
        }
    }

    private void publish(ByteBuffer data, long frame, long timestamp, long started) throws Exception {
        long hashing = System.nanoTime();
        long hash = NativeBridge.nv12Hash(data, width, height);
        hashWork.add(timestamp, 0, System.nanoTime() - hashing);
        long monotonicTimestamp = CameraProbe.captureTimestamp(readback, timestamp);
        readback.add(timestamp, hash, System.nanoTime() - started);
        readback.deliveryAge(android.os.SystemClock.elapsedRealtimeNanos() - timestamp);
        long publishing = System.nanoTime();
        sink.receive(data, frame, monotonicTimestamp);
        publicationWork.add(timestamp, 0, System.nanoTime() - publishing);
        sinkDelivery.deliveryAge(android.os.SystemClock.elapsedRealtimeNanos() - timestamp);
    }

    private void drainGpu() {
        if (closing || error != null) { draining = false; return; }
        try {
            gpu.drain(this::deliver);
            if (gpu.pending()) handler.postDelayed(this::drainGpu, 1);
            else draining = false;
        } catch (Exception e) { error = e.toString(); draining = false; }
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

    private void render(boolean readPixels) {
        if (gpu != null) gpu.bindSource();
        texture.getTransformMatrix(transform);
        vertices.position(0);
        GLES20.glViewport(0, 0, width, height); GLES20.glUseProgram(program);
        GLES20.glUniformMatrix4fv(transformLocation, 1, false, transform, 0);
        GLES20.glEnableVertexAttribArray(positionLocation); GLES20.glEnableVertexAttribArray(uvLocation);
        GLES20.glVertexAttribPointer(positionLocation, 2, GLES20.GL_FLOAT, false, 16, vertices);
        vertices.position(2); GLES20.glVertexAttribPointer(uvLocation, 2, GLES20.GL_FLOAT, false, 16, vertices);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        if (readPixels) {
            pixels.clear();
            GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixels);
        }
        int result = GLES20.glGetError(); if (result != GLES20.GL_NO_ERROR) throw new IllegalStateException("Preview GL error: " + result);
    }

    private void save() throws Exception {
        pixels.rewind();
        Bitmap raw = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888); raw.copyPixelsFromBuffer(pixels);
        Matrix flip = new Matrix(); flip.setScale(1, -1);
        Bitmap image = Bitmap.createBitmap(raw, 0, 0, width, height, flip, false);
        try (FileOutputStream file = new FileOutputStream(sample)) {
            if (!image.compress(Bitmap.CompressFormat.PNG, 100, file)) throw new IllegalStateException("PNG encoding failed");
        } finally { image.recycle(); raw.recycle(); }
    }

    @Override public void close() throws Exception {
        closing = true;
        CountDownLatch closed = new CountDownLatch(1);
        handler.post(() -> {
            try {
                closing = true;
                if (texture != null) texture.setOnFrameAvailableListener(null);
                if (surface != null) surface.release();
                if (texture != null) texture.release();
                if (gpu != null) gpu.close();
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
        callbacks.quitSafely(); callbacks.join(2000);
        publications.quitSafely(); publications.join(5000);
        Delivery old = pending.getAndSet(null);
        if (old != null) available.add(old);
        if (!drained || thread.isAlive() || callbacks.isAlive() || publications.isAlive())
            throw new IllegalStateException("Preview thread cleanup timed out");
    }
}
