package ai.sunnypilot.fold3;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.nio.ByteBuffer;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class NativeBridgeTest {
    @Test public void javaPublisherReachesPythonVisionIpcWithoutClaimingCalibration() throws Exception {
        android.content.Context app = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().getTargetContext();
        PythonRuntime.start(app);
        String directory = new java.io.File(app.getFilesDir(), "ipc").getAbsolutePath();
        long handle = NativeBridge.createCameraPublisher(directory, new int[]{0}, 4, 2);
        java.util.concurrent.CountDownLatch ready = new java.util.concurrent.CountDownLatch(1);
        Thread producer = new Thread(() -> {
            try { if (!ready.await(15, java.util.concurrent.TimeUnit.SECONDS)) return; }
            catch (InterruptedException e) { Thread.currentThread().interrupt(); return; }
            ByteBuffer pixels = ByteBuffer.allocateDirect(12);
            for (int frame = 1; frame <= 100; frame++) {
                pixels.clear();
                for (int i = 0; i < 12; i++) pixels.put((byte)(frame % 128));
                pixels.rewind();
                NativeBridge.sendCameraFrame(handle, 0, pixels, 4, 2, frame, 1000000 + frame);
                android.os.SystemClock.sleep(20);
            }
        });
        try {
            producer.start();
            int frames = com.chaquo.python.Python.getInstance().getModule("fold3_runtime")
                .callAttr("receive_camera_test", app.getFilesDir().getAbsolutePath(), ready).toInt();
            assertEquals(10, frames);
        } finally { producer.join(); NativeBridge.closeCameraPublisher(handle); }
    }
    @Test public void cameraPublisherRejectsWrongDimensionsAndUnallocatedStreams() {
        android.content.Context app = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().getTargetContext();
        String directory = new java.io.File(app.getFilesDir(), "ipc").getAbsolutePath();
        long handle = NativeBridge.createCameraPublisher(directory, new int[]{0}, 4, 2);
        try {
            for (int[] shape : new int[][]{{0, 8, 2}, {1, 4, 2}}) {
                try {
                    NativeBridge.sendCameraFrame(handle, shape[0], ByteBuffer.allocateDirect(24), shape[1], shape[2], 1, 1000001);
                    fail("Invalid camera stream accepted");
                } catch (IllegalArgumentException expected) { }
            }
        } finally { NativeBridge.closeCameraPublisher(handle); }
    }
    @Test public void pinnedVisionIpcPreservesFramesAndMetadata() throws Exception {
        android.content.Context app = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().getTargetContext();
        String path = app.getDir("ipc", 0).getAbsolutePath();
        android.system.Os.chmod(path, 0700);
        org.json.JSONObject report = new org.json.JSONObject(NativeBridge.visionIpcRoundTrip(path));
        assertEquals("passed", report.getString("status"));
        assertEquals(100, report.getInt("frames"));
        assertFalse(report.getBoolean("vehicle_tx_enabled"));
    }
    private static ByteBuffer direct(byte... data) {
        ByteBuffer b = ByteBuffer.allocateDirect(data.length); b.put(data); b.rewind(); return b;
    }
    @Test public void paddedLumaAndSlicedInterleavedChromaBecomeNv12() {
        ByteBuffer y = direct((byte)1,(byte)2,(byte)3,(byte)4,(byte)99,(byte)99,
                              (byte)5,(byte)6,(byte)7,(byte)8);
        ByteBuffer u = direct((byte)88,(byte)9,(byte)99,(byte)10,(byte)88);
        u.position(1); u.limit(4);
        ByteBuffer v = direct((byte)11,(byte)99,(byte)12);
        ByteBuffer output = ByteBuffer.allocateDirect(12);
        NativeBridge.packNv12(y, 6, 1, u.slice(), 4, 2, v, 4, 2, 4, 2, output);
        byte[] actual = new byte[12]; output.get(actual);
        assertArrayEquals(new byte[]{1,2,3,4,5,6,7,8,9,11,10,12}, actual);
    }
    @Test public void invalidSizesAreRejected() {
        ByteBuffer small = ByteBuffer.allocateDirect(1);
        try {
            NativeBridge.packNv12(small, 4, 1, small, 2, 1, small, 2, 1, 4, 2, small);
            fail("undersized planes accepted");
        } catch (IllegalArgumentException expected) { }
        ByteBuffer plane = ByteBuffer.allocateDirect(16);
        try {
            NativeBridge.packNv12(plane, 1, 1, plane, 2, 1, plane, 2, 1, 4, 2, plane);
            fail("overlapping row stride accepted");
        } catch (IllegalArgumentException expected) { }
        try {
            NativeBridge.packNv12(small, 1, 1, small, 1, 1, small, 1, 1, 1, 1, null);
            fail("odd dimensions accepted");
        } catch (IllegalArgumentException expected) { }
    }
    @Test public void physicalTransmissionCannotBeEnabled() { assertFalse(BuildConfig.VEHICLE_TX_ENABLED); }
    @Test public void invalidUsbDescriptorCannotOpenAPanda() throws Exception {
        org.json.JSONObject report = new org.json.JSONObject(NativeBridge.pandaUsbProbe(-1));
        assertEquals("failed", report.getString("status"));
        assertFalse(report.getBoolean("authorized_fd_wrapped"));
        assertFalse(report.getBoolean("vehicle_tx_enabled"));
    }
}
