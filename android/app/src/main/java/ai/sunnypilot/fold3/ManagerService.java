package ai.sunnypilot.fold3;

import android.app.*;
import android.content.*;
import android.os.*;
import org.json.JSONObject;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.*;

/** Binder death + bounded backoff supervision. Missing inputs never mean ready. */
public final class ManagerService extends Service {
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final List<Child> children = new ArrayList<>();
    private final ControlsMetrics controlsMetrics = new ControlsMetrics();
    private boolean simulation, fullChain, stopping;
    private final class Child implements ServiceConnection {
        final Class<?> type;
        IBinder remote;
        int restarts;
        boolean bound;
        JSONObject state = new JSONObject();
        Child(Class<?> type) { this.type = type; }
        void bind() { if (!stopping) bound = bindService(new Intent(ManagerService.this, type), this, BIND_AUTO_CREATE); }
        void unbind() { if (bound) { unbindService(this); bound = false; } }
        public void onServiceConnected(ComponentName name, IBinder service) {
            remote = service;
            Parcel request = Parcel.obtain(), reply = Parcel.obtain();
            try { request.writeInt(simulation ? 1 : 0); request.writeInt(fullChain ? 1 : 0);
                remote.transact(WorkerService.START, request, reply, 0); reply.readException(); }
            catch (Exception e) { disconnected(); }
            finally { request.recycle(); reply.recycle(); }
        }
        private void disconnected() {
            if (remote == null && !bound) return;
            remote = null;
            unbind();
            try { state = new JSONObject().put("phase", "process_died").put("restarts", restarts); }
            catch (Exception e) { throw new IllegalStateException(e); }
            if (!stopping && restarts < 3) { long delay = 250L << (2 * restarts++); handler.postDelayed(this::bind, delay); }
        }
        public void onServiceDisconnected(ComponentName name) { disconnected(); }
        public void onBindingDied(ComponentName name) { disconnected(); }
        public void onNullBinding(ComponentName name) { disconnected(); }
        void poll() {
            if (remote == null) return;
            Parcel request = Parcel.obtain(), reply = Parcel.obtain();
            try {
                remote.transact(WorkerService.STATE, request, reply, 0); reply.readException(); state = new JSONObject(reply.readString());
                state.put("restarts", restarts);
                if (type == WorkerService.Controls.class) {
                    try { controlsMetrics.collect(getFilesDir()); }
                    catch (Exception e) { state.put("phase", "failed").put("error", "Controls metrics: " + e); }
                }
                if (state.optString("phase").equals("running") &&
                    SystemClock.elapsedRealtime() - state.getLong("last_progress_ms") > 2000) {
                    state.put("watchdog", "stale progress; process terminated");
                    remote.transact(WorkerService.TERMINATE, request, null, IBinder.FLAG_ONEWAY);
                }
                if ((state.optString("phase").equals("failed") || state.optString("phase").equals("exited")) && restarts < 3) {
                    remote.transact(WorkerService.TERMINATE, request, null, IBinder.FLAG_ONEWAY);
                }
            } catch (Exception e) { disconnected(); }
            finally { request.recycle(); reply.recycle(); }
        }
        void terminate() {
            IBinder target = remote; remote = null;
            unbind();
            if (target != null) {
                Parcel request = Parcel.obtain();
                try { target.transact(WorkerService.TERMINATE, request, null, IBinder.FLAG_ONEWAY); } catch (Exception ignored) { }
                finally { request.recycle(); }
            }
        }
    }
    private final Runnable monitor = new Runnable() {
        public void run() {
            try {
                JSONObject report = new JSONObject().put("pid", android.os.Process.myPid()).put("simulation", simulation).put("full_chain", fullChain)
                    .put("vehicle_tx_enabled", false).put("engagement_allowed", false)
                    .put("engagement_blocker", "camera calibration, model parity/performance and physical Panda qualification incomplete");
                JSONObject processes = new JSONObject();
                for (Child child : children) { child.poll(); processes.put(child.type.getSimpleName(), child.state); }
                report.put("processes", processes).put("updated_elapsed_ms", SystemClock.elapsedRealtime());
                File tmp = new File(getFilesDir(), "manager.json.tmp"), output = new File(getFilesDir(), "manager.json");
                Files.write(tmp.toPath(), report.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
                Files.move(tmp.toPath(), output.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (Exception e) { android.util.Log.e("sunnypilot-manager", "Supervisor report failed", e); }
            if (!stopping) handler.postDelayed(this, 500);
        }
    };
    @Override public void onCreate() {
        super.onCreate();
        getSystemService(NotificationManager.class).createNotificationChannel(new NotificationChannel("runtime", "Android driving runtime", NotificationManager.IMPORTANCE_LOW));
        startForeground(2, new Notification.Builder(this, "runtime").setContentTitle("sunnypilot runtime · transmission disabled")
            .setSmallIcon(android.R.drawable.ic_dialog_info).build());
    }
    @Override public int onStartCommand(Intent intent, int flags, int id) {
        if (!children.isEmpty()) return START_NOT_STICKY;
        simulation = BuildConfig.DEBUG && intent != null && intent.getBooleanExtra("simulation", false);
        fullChain = simulation && intent.getBooleanExtra("full_chain", false);
        File batches = new File(getFilesDir(), "controlsd-batches");
        if (batches.exists()) try (var paths = Files.walk(batches.toPath())) {
            for (var path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
        } catch (java.io.IOException e) { throw new IllegalStateException("Cannot clear controls timing batches", e); }
        // The /dev/shm replacement is volatile across manager runs. Synthetic
        // CarParams must also be generated afresh from this test's passive CAN.
        for (String name : fullChain ? List.of("sim-params", "sim-memory-params") :
                List.of(simulation ? "sim-memory-params" : "memory-params")) {
            File synthetic = new File(getFilesDir(), name);
            if (synthetic.exists()) try (var paths = Files.walk(synthetic.toPath())) {
                for (var path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
            } catch (java.io.IOException e) { throw new IllegalStateException("Cannot clear synthetic parameters", e); }
        }
        for (Class<?> type : List.of(WorkerService.Controls.class, WorkerService.Monitoring.class)) {
            Child child = new Child(type); children.add(child); child.bind();
        }
        if (!simulation) for (Class<?> type : List.of(WorkerService.Sensors.class, WorkerService.RoadModel.class, WorkerService.DriverModel.class)) {
            Child child = new Child(type); children.add(child); child.bind();
        }
        if (!simulation || fullChain) for (Class<?> type : List.of(WorkerService.Vehicle.class, WorkerService.Localization.class, WorkerService.Calibration.class,
            WorkerService.Parameters.class, WorkerService.Torque.class, WorkerService.Delay.class,
            WorkerService.Radar.class, WorkerService.Planning.class, WorkerService.Selfdrive.class)) {
            Child child = new Child(type); children.add(child); child.bind();
        }
        if (simulation) { Child child = new Child(WorkerService.Simulator.class); children.add(child); child.bind(); }
        handler.post(monitor);
        return START_NOT_STICKY;
    }
    @Override public void onDestroy() { stopping = true; handler.removeCallbacksAndMessages(null); for (Child c : children) c.terminate(); super.onDestroy(); }
    @Override public IBinder onBind(Intent intent) { return null; }
}
