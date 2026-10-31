package ai.sunnypilot.fold3;

import android.app.*;
import android.content.Intent;
import android.os.*;
import org.json.*;
import java.nio.file.Files;

/** Live road/cabin diagnostic streams. Every VisionIPC buffer remains invalid. */
public final class VisionService extends Service {
    private volatile boolean stopping;
    private Thread worker;
    @Override public IBinder onBind(Intent intent) { return null; }
    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (worker != null) return START_NOT_STICKY;
        NotificationManager notifications = getSystemService(NotificationManager.class);
        notifications.createNotificationChannel(new NotificationChannel("vision", "Live camera pipeline", NotificationManager.IMPORTANCE_LOW));
        startForeground(5, new Notification.Builder(this, "vision").setContentTitle("Road and driver camera diagnostics")
            .setSmallIcon(android.R.drawable.ic_menu_camera).setContentText("Uncalibrated; vehicle transmission disabled").build());
        int duration = intent == null ? 1800 : intent.getIntExtra("duration_seconds", 1800);
        final int seconds = Math.max(5, Math.min(1800, duration));
        final boolean samsung = BuildConfig.DEBUG && getPackageName().equals("com.samsung.sunnypilot.fold3") &&
            intent != null && intent.getBooleanExtra("samsung_multiview", false);
        worker = new Thread(() -> run(seconds, samsung), "live-cameras");
        worker.start();
        return START_NOT_STICKY;
    }
    private void run(int seconds, boolean samsung) {
        JSONObject report = new JSONObject();
        JSONArray sessions = new JSONArray();
        try {
            report.put("vehicle_tx_enabled", false).put("valid", false).put("sessions", sessions)
                .put("requested_seconds", seconds).put("recovery_limit", 3).put("started_utc", java.time.Instant.now().toString())
                .put("dual_rear", samsung ? "bench candidate feeds; physical identities and calibration unverified" : "not requested")
                .put("backend", samsung ? "samsung_multiview" : "camera2");
            Files.deleteIfExists(new java.io.File(getFilesDir(), "vision-sources.json").toPath());
            report.put("status", "running");
            save(report);
            long end = SystemClock.elapsedRealtime() + seconds * 1000L;
            for (int retry = 0; !stopping && retry <= 3 && SystemClock.elapsedRealtime() < end; retry++) {
                if (retry > 0) Thread.sleep(250L << (2 * (retry - 1)));
                try (CameraProbe camera = new CameraProbe(this, value -> {
                    if (stopping) throw new IllegalStateException("Live camera session stopped");
                })) {
                    long remaining = Math.max(1, end - SystemClock.elapsedRealtime());
                    JSONObject session = samsung ? camera.samsungMultiview(true, remaining, true) :
                        camera.attempt(new String[]{"0", "1"}, null, 1280, 720, remaining);
                    sessions.put(session);
                    report.put("recoveries", retry).put("thermal_status", getSystemService(PowerManager.class).getCurrentThermalStatus());
                    save(report);
                    if (session.optString(samsung ? "transport_status" : "status").equals("passed") ||
                        session.optString("error").contains("critical_thermal_status:")) break;
                }
            }
            boolean passed = !stopping && sessions.length() == 1 &&
                sessions.getJSONObject(0).optString(samsung ? "transport_status" : "status").equals("passed");
            report.put("transport_status", passed ? "passed" : "failed")
                .put("status", stopping ? "stopped" : passed ? (samsung ? "blocked" : "passed") : "failed")
                .put("completed_utc", java.time.Instant.now().toString());
            save(report);
        } catch (Exception e) {
            try { report.put("status", "failed").put("error", e.toString()); save(report); }
            catch (Exception ignored) { android.util.Log.e("Fold3Vision", "Cannot save report", ignored); }
        } finally { stopSelf(); }
    }
    private void save(JSONObject report) throws Exception {
        java.nio.file.Path destination = new java.io.File(getFilesDir(), "vision-report.json").toPath();
        java.nio.file.Path temporary = destination.resolveSibling("vision-report.tmp");
        Files.write(temporary, report.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
        Files.move(temporary, destination, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }
    @Override public void onDestroy() { stopping = true; super.onDestroy(); }
}
