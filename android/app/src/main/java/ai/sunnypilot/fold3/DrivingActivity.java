package ai.sunnypilot.fold3;

import android.app.NativeActivity;
import android.content.Intent;
import android.os.Bundle;
import com.chaquo.python.Python;
import java.io.File;
import java.nio.file.Files;

/** Surface ownership is independent of every driving service process. */
public final class DrivingActivity extends NativeActivity {
    private boolean visionStarted;
    private native void prepareNative();
    @Override public void onCreate(Bundle saved) {
        AndroidHardware.initialize(this);
        PythonRuntime.start(this);
        System.loadLibrary("raylib");
        prepareNative();
        super.onCreate(saved);
        getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getOnBackInvokedDispatcher().registerOnBackInvokedCallback(android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT, this::finish);
        getWindow().getDecorView().addOnLayoutChangeListener((view, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom) -> {
            AndroidHardware.surfaceWidth = right - left;
            AndroidHardware.surfaceHeight = bottom - top;
            startVisionWhenReady(right - left, bottom - top);
        });
        getWindow().getDecorView().post(() -> startVisionWhenReady(
            getWindow().getDecorView().getWidth(), getWindow().getDecorView().getHeight()));
    }
    private void startVisionWhenReady(int width, int height) {
        // A service can start before sensorLandscape has rotated the display.
        // Wait for the actual landscape surface before Camera2 snapshots its
        // orientation, and only start once across later surface recreation.
        if (!visionStarted && !isFinishing() && width > height && getIntent().getBooleanExtra("start_vision", false)) {
            visionStarted = true;
            startForegroundService(new Intent(this, VisionService.class)
                .putExtra("duration_seconds", getIntent().getIntExtra("vision_seconds", 1800)));
        }
    }
    public void runUi() {
        try { Python.getInstance().getModule("fold3_ui").callAttr("main", getFilesDir().getAbsolutePath(),
            getIntent().getBooleanExtra("simulation", false), getIntent().getBooleanExtra("preview", false)); }
        catch (Exception e) {
            try { Files.write(new File(getFilesDir(), "ui-error.txt").toPath(), e.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8)); }
            catch (Exception ignored) { }
        }
    }
    @Override public void onDestroy() {
        super.onDestroy();
        // Raylib and Python retain process-global GPU objects. A completed UI
        // Activity gets a fresh renderer next time; driving services survive.
        android.os.Process.killProcess(android.os.Process.myPid());
    }
}
