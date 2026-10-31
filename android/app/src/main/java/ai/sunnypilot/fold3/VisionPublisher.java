package ai.sunnypilot.fold3;

import java.nio.ByteBuffer;

/** Serialize publisher lifetime against ImageReader callbacks. All frames are uncalibrated. */
final class VisionPublisher implements AutoCloseable {
    private long handle;
    static int stream(String id) {
        // Pinned Cereal VisionStreamType values, checked by the native constructor.
        return id.equals("0") ? 0 : id.equals("2") ? 2 : 1;
    }
    VisionPublisher(String directory, String[] ids, int width, int height) {
        int[] streams = new int[ids.length];
        for (int i = 0; i < ids.length; i++) streams[i] = stream(ids[i]);
        handle = NativeBridge.createCameraPublisher(directory, streams, width, height);
    }
    synchronized void send(String id, ByteBuffer buffer, int width, int height, long frame, long timestamp) {
        if (handle != 0) NativeBridge.sendCameraFrame(handle, stream(id), buffer, width, height, frame, timestamp);
    }
    public synchronized void close() {
        if (handle != 0) { NativeBridge.closeCameraPublisher(handle); handle = 0; }
    }
}
