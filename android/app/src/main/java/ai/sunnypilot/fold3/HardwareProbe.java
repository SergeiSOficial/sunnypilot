package ai.sunnypilot.fold3;

import android.content.*;
import android.hardware.*;
import android.hardware.usb.*;
import android.location.LocationManager;
import android.media.AudioManager;
import android.os.*;
import org.json.JSONArray;
import org.json.JSONObject;
import java.util.*;

/** Android API capability evidence, separate from upstream hardware messages. */
final class HardwareProbe {
    static JSONObject measure(Context context) throws Exception {
        SensorManager manager = context.getSystemService(SensorManager.class);
        JSONArray sensors = new JSONArray();
        for (Sensor sensor : manager.getSensorList(Sensor.TYPE_ALL)) {
            sensors.put(new JSONObject().put("name", sensor.getName()).put("vendor", sensor.getVendor())
                .put("type", sensor.getType()).put("min_delay_us", sensor.getMinDelay()).put("resolution", sensor.getResolution()));
        }
        HandlerThread thread = new HandlerThread("imu-probe"); thread.start();
        Map<Integer, long[]> counts = new HashMap<>();
        SensorEventListener listener = new SensorEventListener() {
            public void onAccuracyChanged(Sensor sensor, int accuracy) { }
            public void onSensorChanged(SensorEvent event) {
                synchronized (counts) {
                    long[] sample = counts.computeIfAbsent(event.sensor.getType(), k -> new long[4]);
                    if (sample[0] == 0) sample[1] = event.timestamp;
                    else if (event.timestamp <= sample[2]) sample[3]++;
                    sample[0]++; sample[2] = event.timestamp;
                }
            }
        };
        try {
            for (int type : new int[]{Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE}) {
                Sensor sensor = manager.getDefaultSensor(type);
                if (sensor != null) manager.registerListener(listener, sensor, 5000, new Handler(thread.getLooper()));
            }
            Thread.sleep(5000);
        } finally { manager.unregisterListener(listener); thread.quitSafely(); thread.join(2000); }
        JSONArray imu = new JSONArray();
        synchronized (counts) {
            for (Map.Entry<Integer, long[]> entry : counts.entrySet()) {
                long[] s = entry.getValue();
                imu.put(new JSONObject().put("sensor_type", entry.getKey()).put("events", s[0]).put("timestamp_errors", s[3])
                    .put("hz", s[2] > s[1] ? (s[0] - 1) * 1e9 / (s[2] - s[1]) : 0));
            }
        }
        JSONArray usb = new JSONArray();
        UsbManager usbManager = context.getSystemService(UsbManager.class);
        for (UsbDevice device : usbManager.getDeviceList().values()) {
            usb.put(new JSONObject().put("vendor_id", device.getVendorId()).put("product_id", device.getProductId())
                .put("permission", usbManager.hasPermission(device)));
        }
        Intent battery = context.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        AudioManager audio = context.getSystemService(AudioManager.class);
        LocationManager location = context.getSystemService(LocationManager.class);
        return new JSONObject().put("pid", android.os.Process.myPid()).put("sensors", sensors).put("imu_observations", imu)
            .put("thermal_status", context.getSystemService(PowerManager.class).getCurrentThermalStatus())
            .put("battery_level", battery == null ? JSONObject.NULL : battery.getIntExtra(BatteryManager.EXTRA_LEVEL, -1))
            .put("battery_temperature_deci_c", battery == null ? JSONObject.NULL : battery.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1))
            .put("battery_current_ua", context.getSystemService(BatteryManager.class).getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW))
            .put("storage_free_bytes", context.getFilesDir().getUsableSpace()).put("usb_devices", usb)
            .put("panda_transport", "blocked: native USB transport / firmware verification not integrated")
            .put("gnss_provider", location.getAllProviders().contains(LocationManager.GPS_PROVIDER))
            .put("gnss_fix", "blocked: location permission and integration not implemented")
            .put("audio_sample_rate", audio.getProperty(AudioManager.PROPERTY_OUTPUT_SAMPLE_RATE))
            .put("audio_alert_validation", "blocked: upstream alerts not integrated");
    }
}
