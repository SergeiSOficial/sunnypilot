package ai.sunnypilot.fold3;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import com.chaquo.python.Python;
import java.io.File;
import java.nio.file.Files;
import java.nio.charset.StandardCharsets;

/** An isolated benchmark process. It has no Panda or engagement path. */
public final class ModelService extends Service {
    private final java.util.concurrent.atomic.AtomicBoolean running = new java.util.concurrent.atomic.AtomicBoolean();
    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (!running.compareAndSet(false, true)) return START_NOT_STICKY;
        String profile = intent == null ? "baseline" : intent.getStringExtra("model_profile");
        final String selected = profile == null ? "baseline" : profile;
        new Thread(() -> {
            try {
                PythonRuntime.start(this);
                Python.getInstance().getModule("fold3_models").callAttr("benchmark", getFilesDir().getAbsolutePath(), selected);
            } catch (Exception e) {
                try {
                    Files.write(new File(getFilesDir(), "model-progress.txt").toPath(), e.toString().getBytes(StandardCharsets.UTF_8));
                    String failure = new org.json.JSONObject().put("status", "blocked").put("error", e.toString())
                        .put("completed_utc", java.time.Instant.now().toString()).put("vehicle_tx_enabled", false).toString(2);
                    Files.write(new File(getFilesDir(), "model-report.json").toPath(), failure.getBytes(StandardCharsets.UTF_8));
                } catch (Exception writeError) { android.util.Log.e("Fold3Models", "Cannot save model failure", writeError); }
            } finally { running.set(false); stopSelf(); }
        }, "model-validation").start();
        return START_NOT_STICKY;
    }
    @Override public IBinder onBind(Intent intent) { return null; }
}
