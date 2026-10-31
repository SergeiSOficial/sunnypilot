package ai.sunnypilot.fold3;

import android.app.*;
import android.content.Intent;
import android.os.*;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

public final class ProbeService extends Service {
    private final AtomicBoolean running = new AtomicBoolean();
    private JSONObject report;
    private final JSONArray attempts = new JSONArray();
    private volatile boolean stopping;
    @Override public IBinder onBind(Intent intent) { return null; }
    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        NotificationManager notifications = getSystemService(NotificationManager.class);
        notifications.createNotificationChannel(new NotificationChannel("validation", "Port validation", NotificationManager.IMPORTANCE_LOW));
        startForeground(1, new Notification.Builder(this, "validation").setContentTitle("sunnypilot camera validation")
            .setSmallIcon(android.R.drawable.ic_menu_camera).setContentText("Vehicle transmission disabled").build());
        if (running.compareAndSet(false, true)) {
            boolean soak = intent != null && intent.getBooleanExtra("soak", false);
            boolean matrix = intent == null || !intent.getBooleanExtra("soak_only", false);
            Intent options = intent == null ? new Intent() : new Intent(intent);
            new Thread(() -> run(soak, matrix, options), "camera-probe").start();
        }
        return START_NOT_STICKY;
    }
    private void progress(String message) {
        if (stopping) throw new IllegalStateException("validation cancelled");
        try {
            PowerManager power = getSystemService(PowerManager.class);
            String text = message + "\nThermal status: " + power.getCurrentThermalStatus()
                + "\nNative heap: " + Debug.getNativeHeapAllocatedSize() / 1048576 + " MiB"
                + "\nCompleted attempts: " + attempts.length() + "\nVehicle transmission: disabled";
            Files.write(new File(getFilesDir(), "progress.txt").toPath(), text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        } catch (java.io.IOException e) { android.util.Log.e("Fold3Probe", "Cannot save progress", e); }
    }
    private void save() throws Exception {
        Files.write(new File(getFilesDir(), "camera-report.json").toPath(), report.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }
    private void run(boolean soak, boolean matrix, Intent options) {
        try (CameraProbe probe = new CameraProbe(this, this::progress)) {
            report = new JSONObject().put("device", Build.MODEL).put("sdk", Build.VERSION.SDK_INT).put("pid", android.os.Process.myPid())
                .put("fingerprint", Build.FINGERPRINT).put("started_utc", java.time.Instant.now().toString())
                .put("package", getPackageName()).put("logical_access", probe.logicalAccess())
                .put("vehicle_tx_enabled", false).put("driving_runtime_ready", false)
                .put("opencl", new JSONObject(NativeBridge.openClProbe())).put("capabilities", probe.capabilities())
                .put("attempts", attempts);
            save();
            if (BuildConfig.DEBUG && options.getBooleanExtra("samsung_multiview", false)) {
                int seconds = Math.max(3, Math.min(1800, options.getIntExtra("duration_seconds", 10)));
                attempts.put(probe.samsungMultiview(options.getBooleanExtra("front_camera", true), seconds * 1000L));
                report.put("completed_utc", java.time.Instant.now().toString()); save();
                return;
            }
            if (BuildConfig.DEBUG && options.hasExtra("camera_ids")) {
                String[] ids = options.getStringExtra("camera_ids").split(",");
                int seconds = Math.max(3, Math.min(1800, options.getIntExtra("duration_seconds", 10)));
                attempts.put(probe.attempt(ids, options.getStringExtra("logical_camera"),
                    options.getIntExtra("width", 1280), options.getIntExtra("height", 720), seconds * 1000L));
                report.put("completed_utc", java.time.Instant.now().toString()); save();
                return;
            }
            for (String[] ids : matrix ? probe.combinations() : Collections.<String[]>emptyList()) {
                if (stopping) break;
                for (int[] size : new int[][]{{1280, 720}, {640, 480}}) {
                    progress("Opening " + Arrays.toString(ids));
                    attempts.put(probe.attempt(ids, null, size[0], size[1], 4000)); save();
                    Thread.sleep(500);
                }
            }
            for (String logical : matrix ? probe.logicalCandidates() : Collections.<String>emptySet()) {
                for (int[] size : new int[][]{{1280, 720}, {640, 480}}) {
                    progress("Trying logical rear camera " + logical + " with physical IDs 0 + 2");
                    attempts.put(probe.attempt(new String[]{"0", "2"}, logical, size[0], size[1], 4000)); save();
                }
            }
            if (soak && !stopping) {
                progress("Starting 30-minute road + driver camera smoke test; full driving workload is not available");
                JSONObject result = probe.attempt(new String[]{"0", "1"}, null, 1280, 720, 30 * 60 * 1000);
                report.put("camera_soak", result).put("combined_driving_workload", "blocked: native driving runtime and model parity incomplete");
            }
            report.put("completed_utc", java.time.Instant.now().toString()); save();
            progress("Camera tests complete. Read camera-report.json and runtime.json for actual results.");
        } catch (Throwable e) {
            try {
                if (report != null) { report.put("fatal_error", e.toString()); save(); }
                Files.write(new File(getFilesDir(), "progress.txt").toPath(), ("Validation stopped: " + e).getBytes(java.nio.charset.StandardCharsets.UTF_8));
            } catch (Exception ignored) { }
            android.util.Log.e("Fold3Probe", "Validation failed", e);
        } finally { running.set(false); stopForeground(STOP_FOREGROUND_REMOVE); stopSelf(); }
    }
    @Override public void onDestroy() { stopping = true; super.onDestroy(); }
}
