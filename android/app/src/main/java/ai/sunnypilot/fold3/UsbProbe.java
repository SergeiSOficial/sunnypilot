package ai.sunnypilot.fold3;

import android.app.PendingIntent;
import android.content.*;
import android.hardware.usb.*;
import org.json.JSONObject;
import java.nio.file.Files;

/** Explicit user-started descriptor handoff. Native code exposes read-only identity requests. */
final class UsbProbe implements AutoCloseable {
    private static final String PERMISSION = "ai.sunnypilot.fold3.USB_PERMISSION";
    private final Context context;
    private final UsbManager manager;
    private boolean receiveOnly;
    private int pendingDevice = -1;
    private final BroadcastReceiver receiver = new BroadcastReceiver() {
        @Override public void onReceive(Context c, Intent intent) {
            UsbDevice device = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE, UsbDevice.class);
            if (!PERMISSION.equals(intent.getAction()) || device == null || device.getDeviceId() != pendingDevice) return;
            pendingDevice = -1;
            if (PERMISSION.equals(intent.getAction()) && device != null && recognized(device) && manager.hasPermission(device))
                authorized(device);
            else save("{\"status\":\"blocked\",\"error\":\"USB permission denied or device unavailable\",\"vehicle_tx_enabled\":false}");
        }
    };
    UsbProbe(Context context) {
        this.context = context;
        manager = context.getSystemService(UsbManager.class);
        context.registerReceiver(receiver, new IntentFilter(PERMISSION), Context.RECEIVER_NOT_EXPORTED);
    }
    static boolean recognized(UsbDevice device) {
        return (device.getVendorId() == 0xbbaa || device.getVendorId() == 0x3801) && device.getProductId() == 0xddcc;
    }
    void start() {
        start(false);
    }
    void start(boolean receiveOnly) {
        if (pendingDevice != -1) return;
        this.receiveOnly = receiveOnly;
        for (UsbDevice device : manager.getDeviceList().values()) if (recognized(device)) {
            if (manager.hasPermission(device)) authorized(device);
            else {
                pendingDevice = device.getDeviceId();
                manager.requestPermission(device, PendingIntent.getBroadcast(context, 0,
                    new Intent(PERMISSION).setPackage(context.getPackageName()), PendingIntent.FLAG_MUTABLE | PendingIntent.FLAG_UPDATE_CURRENT));
            }
            return;
        }
        save("{\"status\":\"blocked\",\"error\":\"No application-visible Panda USB device\",\"vehicle_tx_enabled\":false}");
    }
    private void authorized(UsbDevice device) {
        if (receiveOnly) context.startForegroundService(new Intent(context, PandaRxService.class).putExtra("device", device));
        else inspect(device);
    }
    private void inspect(UsbDevice device) {
        new Thread(() -> {
            UsbDeviceConnection connection = null;
            try {
                connection = manager.openDevice(device);
                if (connection == null) throw new IllegalStateException("USB permission lost or device disconnected");
                save(NativeBridge.pandaUsbProbe(connection.getFileDescriptor()));
            } catch (Exception e) {
                try { save(new JSONObject().put("status", "failed").put("error", e.toString()).put("vehicle_tx_enabled", false).toString()); }
                catch (org.json.JSONException invalid) { android.util.Log.e("Fold3Usb", "Cannot serialize failure", invalid); }
            } finally { if (connection != null) connection.close(); }
        }, "panda-usb-probe").start();
    }
    private void save(String value) {
        try {
            JSONObject report = new JSONObject(value).put("updated_elapsed_ms", android.os.SystemClock.elapsedRealtime());
            Files.write(new java.io.File(context.getFilesDir(), receiveOnly ? "panda-rx.json" : "usb-report.json").toPath(),
                report.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
        } catch (java.io.IOException | org.json.JSONException e) { android.util.Log.e("Fold3Usb", "Cannot save report", e); }
    }
    public void close() { context.unregisterReceiver(receiver); }
}
