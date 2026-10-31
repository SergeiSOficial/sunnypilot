package ai.sunnypilot.fold3;

import android.app.Service;
import android.content.Intent;
import android.os.*;
import com.chaquo.python.Python;
import org.json.JSONObject;

/** One interpreter/daemon per manifest process; never forks or executes APK data. */
public abstract class WorkerService extends Service {
    static final int START = 1, STATE = 2, TERMINATE = 3;
    private volatile String phase = "created", error = "";
    private volatile long lastProgress, ticks;
    private Thread worker;
    protected abstract String daemon();
    public void progress(String next) { phase = next; lastProgress = SystemClock.elapsedRealtime(); ticks++; }
    private final Binder binder = new Binder() {
        @Override protected boolean onTransact(int code, Parcel data, Parcel reply, int flags) {
            if (code == START) {
                boolean simulation = data.readInt() != 0;
                synchronized (WorkerService.this) {
                    if (worker == null) {
                        progress("starting");
                        worker = new Thread(() -> {
                            try {
                                AndroidHardware.initialize(WorkerService.this);
                                PythonRuntime.start(WorkerService.this);
                                Python.getInstance().getModule("fold3_daemons").callAttr("run", daemon(),
                                    getFilesDir().getAbsolutePath(), WorkerService.this, simulation && BuildConfig.DEBUG);
                                phase = "exited";
                            } catch (Throwable e) { error = e.toString(); phase = "failed"; }
                        }, daemon());
                        worker.start();
                    }
                }
                reply.writeNoException(); return true;
            }
            if (code == STATE) {
                try {
                    reply.writeNoException();
                    reply.writeString(new JSONObject().put("daemon", daemon()).put("pid", android.os.Process.myPid())
                        .put("phase", phase).put("error", error).put("last_progress_ms", lastProgress).put("ticks", ticks).toString());
                    return true;
                } catch (Exception e) { throw new IllegalStateException(e); }
            }
            if (code == TERMINATE) { android.os.Process.killProcess(android.os.Process.myPid()); return true; }
            return false;
        }
    };
    @Override public IBinder onBind(Intent intent) { return binder; }
    public static final class Controls extends WorkerService { protected String daemon() { return "controlsd"; } }
    public static final class Monitoring extends WorkerService { protected String daemon() { return "dmonitoringd"; } }
    public static final class Sensors extends WorkerService { protected String daemon() { return "sensord"; } }
    public static final class Simulator extends WorkerService { protected String daemon() { return "sim_bridge"; } }
}
