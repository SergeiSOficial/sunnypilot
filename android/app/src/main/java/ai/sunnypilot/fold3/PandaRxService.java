package ai.sunnypilot.fold3;

import android.app.*;
import android.content.Intent;
import android.hardware.usb.*;
import android.os.IBinder;
import com.chaquo.python.Python;
import java.nio.file.Files;
import org.json.JSONObject;

/** User-started receive-only session, isolated from simulator and control workers. */
public final class PandaRxService extends Service {
    private volatile boolean stopping;
    private Thread worker;
    @Override public IBinder onBind(Intent intent) { return null; }
    public boolean isStopping() { return stopping; }
    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (worker != null) return START_NOT_STICKY;
        UsbDevice device = intent == null ? null : intent.getParcelableExtra("device", UsbDevice.class);
        UsbManager manager = getSystemService(UsbManager.class);
        if (device == null || !UsbProbe.recognized(device) || !manager.hasPermission(device)) {
            failure("Panda absent or USB permission unavailable"); stopSelf(); return START_NOT_STICKY;
        }
        NotificationManager notifications = getSystemService(NotificationManager.class);
        notifications.createNotificationChannel(new NotificationChannel("panda-rx", "Panda receive-only", NotificationManager.IMPORTANCE_LOW));
        startForeground(4, new Notification.Builder(this, "panda-rx").setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setContentTitle("Panda receive-only diagnostic").setContentText("Vehicle transmission disabled").build());
        worker = new Thread(() -> {
            UsbDeviceConnection connection = null;
            try {
                connection = manager.openDevice(device);
                if (connection == null) throw new IllegalStateException("Panda disconnected or permission lost");
                PythonRuntime.start(this);
                Python.getInstance().getModule("fold3_panda").callAttr("receive", getFilesDir().getAbsolutePath(), this, connection.getFileDescriptor());
            } catch (Exception e) { failure(e.toString()); }
            finally { if (connection != null) connection.close(); stopSelf(); }
        }, "panda-receive-only");
        worker.start();
        return START_NOT_STICKY;
    }
    private void failure(String error) {
        try { Files.write(new java.io.File(getFilesDir(), "panda-rx.json").toPath(),
            new JSONObject().put("status", "blocked").put("error", error).put("vehicle_tx_enabled", false)
                .put("updated_elapsed_ms", android.os.SystemClock.elapsedRealtime()).toString()
                .getBytes(java.nio.charset.StandardCharsets.UTF_8)); }
        catch (Exception e) { android.util.Log.e("Fold3Panda", "Cannot save receive report", e); }
    }
    @Override public void onDestroy() { stopping = true; super.onDestroy(); }
}
