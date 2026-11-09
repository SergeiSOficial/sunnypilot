package ai.sunnypilot.fold3;

import android.Manifest;
import android.content.*;
import android.content.pm.PackageManager;
import android.hardware.*;
import android.location.*;
import android.media.*;
import android.os.*;
import java.util.ArrayDeque;

/** Public Android APIs only. Queues are bounded; timestamps retain nanoseconds. */
public final class AndroidHardware implements SensorEventListener {
    private static Context context;
    static volatile int surfaceWidth, surfaceHeight;
    public static int[] surfaceSize() { return new int[]{surfaceWidth, surfaceHeight}; }
    private final SensorManager sensors;
    private final HandlerThread thread = new HandlerThread("android-imu");
    private final ArrayDeque<double[]> queue = new ArrayDeque<>();
    private long dropped;
    public static void initialize(Context ctx) { context = ctx.getApplicationContext(); }
    public static double powerWatts() {
        BatteryManager battery = context.getSystemService(BatteryManager.class);
        Intent state = context.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        if (state == null) return Double.NaN;
        int currentMicroamps = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW);
        int voltageMillivolts = state.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0);
        if (currentMicroamps == Integer.MIN_VALUE || voltageMillivolts <= 0) return Double.NaN;
        return -(double) currentMicroamps * voltageMillivolts / 1e9;
    }
    public static int thermalStatus() { return context.getSystemService(PowerManager.class).getCurrentThermalStatus(); }
    public static String applicationApk() { return context.getApplicationInfo().sourceDir; }
    public static String nativeLibraries() { return context.getApplicationInfo().nativeLibraryDir; }
    public static void openNetworkSettings() {
        context.startActivity(new Intent(android.provider.Settings.Panel.ACTION_INTERNET_CONNECTIVITY)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
    }
    public static void openApplicationSettings() {
        context.startActivity(new Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
            android.net.Uri.parse("package:" + context.getPackageName())).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
    }
    public AndroidHardware() {
        sensors = context.getSystemService(SensorManager.class);
        thread.start();
        for (int type : new int[]{Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE}) {
            Sensor sensor = sensors.getDefaultSensor(type);
            if (sensor != null) sensors.registerListener(this, sensor, 5000, new Handler(thread.getLooper()));
        }
    }
    public void onAccuracyChanged(Sensor sensor, int accuracy) { }
    public void onSensorChanged(SensorEvent event) {
        synchronized (queue) {
            if (queue.size() == 256) { queue.removeFirst(); dropped++; }
            queue.addLast(new double[]{event.sensor.getType(), event.timestamp, event.values[0], event.values[1], event.values[2], event.accuracy});
        }
    }
    public double[][] drain() {
        synchronized (queue) { double[][] out = queue.toArray(new double[0][]); queue.clear(); return out; }
    }
    public long droppedSamples() { synchronized (queue) { return dropped; } }
    public void close() { sensors.unregisterListener(this); thread.quitSafely(); }
    public static Location latestLocation() {
        if (context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) return null;
        LocationManager manager = context.getSystemService(LocationManager.class);
        return manager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
    }
    public static void playPcm(short[] samples, int sampleRate) {
        AudioFormat format = new AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(sampleRate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build();
        AudioTrack track = new AudioTrack.Builder().setAudioAttributes(new AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION).setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION).build())
            .setAudioFormat(format).setTransferMode(AudioTrack.MODE_STATIC).setBufferSizeInBytes(samples.length * 2).build();
        track.write(samples, 0, samples.length);
        track.play();
        new Handler(Looper.getMainLooper()).postDelayed(track::release, 100L + 1000L * samples.length / sampleRate);
    }
}
