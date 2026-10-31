package ai.sunnypilot.fold3;

import java.nio.ByteBuffer;

public final class NativeBridge {
    static { System.loadLibrary("fold3"); }
    private NativeBridge() { }
    public static native String openClProbe();
    // Only vendor IN requests are compiled into this diagnostic path.
    public static native String pandaUsbProbe(int authorizedDescriptor);
    public static native String visionIpcRoundTrip(String privateDirectory);
    public static native long createCameraPublisher(String privateDirectory, int[] streams, int width, int height);
    public static native void sendCameraFrame(long publisher, int stream, ByteBuffer nv12, int width, int height,
                                              long frame, long timestamp);
    public static native void closeCameraPublisher(long publisher);
    public static native long packNv12(ByteBuffer y, int yRow, int yPixel,
            ByteBuffer u, int uRow, int uPixel, ByteBuffer v, int vRow, int vPixel,
            int width, int height, ByteBuffer output);
}
