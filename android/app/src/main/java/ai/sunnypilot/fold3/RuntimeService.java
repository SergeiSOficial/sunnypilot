package ai.sunnypilot.fold3;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import java.io.File;
import java.nio.file.Files;

/** Isolated Python diagnostics; missing native modules never start a driving daemon. */
public final class RuntimeService extends Service {
    @Override public void onCreate() {
        super.onCreate();
        new Thread(() -> {
            try {
                Files.write(new File(getFilesDir(), "hardware.json").toPath(), HardwareProbe.measure(this).toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
                if (!Python.isStarted()) Python.start(new AndroidPlatform(this));
                String report = Python.getInstance().getModule("fold3_runtime").callAttr("probe", getFilesDir().getAbsolutePath()).toString();
                Files.write(new File(getFilesDir(), "runtime.json").toPath(), report.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            } catch (Exception e) {
                try { Files.write(new File(getFilesDir(), "runtime-error.txt").toPath(), e.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8)); }
                catch (Exception ignored) { }
            } finally { stopSelf(); }
        }, "python-runtime-probe").start();
    }
    @Override public int onStartCommand(Intent intent, int flags, int startId) { return START_NOT_STICKY; }
    @Override public IBinder onBind(Intent intent) { return null; }
}
