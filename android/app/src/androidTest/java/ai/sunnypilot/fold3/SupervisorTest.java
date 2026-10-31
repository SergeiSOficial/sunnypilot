package ai.sunnypilot.fold3;

import android.content.Context;
import android.content.Intent;
import android.os.SystemClock;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.File;
import java.nio.file.Files;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class SupervisorTest {
    private JSONObject waitForControls(Context app, int previous) throws Exception {
        File path = new File(app.getFilesDir(), "manager.json");
        long deadline = SystemClock.elapsedRealtime() + 45000;
        while (SystemClock.elapsedRealtime() < deadline) {
            if (path.isFile()) {
                JSONObject manager = new JSONObject(new String(Files.readAllBytes(path.toPath()), java.nio.charset.StandardCharsets.UTF_8));
                JSONObject controls = manager.getJSONObject("processes").optJSONObject("Controls");
                if (controls != null && controls.optInt("pid") != 0 && controls.optInt("pid") != previous
                    && controls.optString("phase").equals("waiting_for_car_params")) return manager;
            }
            SystemClock.sleep(200);
        }
        throw new AssertionError("Controls did not reach a supervised input wait");
    }
    @Test public void controlProcessRestartsAndActivityRecreationPreservesManager() throws Exception {
        Context app = InstrumentationRegistry.getInstrumentation().getTargetContext();
        app.stopService(new Intent(app, ManagerService.class));
        SystemClock.sleep(1000);
        new File(app.getFilesDir(), "manager.json").delete();
        try (ActivityScenario<MainActivity> activity = ActivityScenario.launch(MainActivity.class)) {
            activity.onActivity(a -> a.startForegroundService(new Intent(a, ManagerService.class)));
            JSONObject first = waitForControls(app, -1);
            assertFalse(first.getBoolean("vehicle_tx_enabled"));
            assertFalse(first.getBoolean("engagement_allowed"));
            int managerPid = first.getInt("pid");
            int controlsPid = first.getJSONObject("processes").getJSONObject("Controls").getInt("pid");
            activity.recreate();
            android.os.Process.killProcess(controlsPid);
            JSONObject restarted = waitForControls(app, controlsPid);
            assertEquals(managerPid, restarted.getInt("pid"));
            assertEquals(1, restarted.getJSONObject("processes").getJSONObject("Controls").getInt("restarts"));
        } finally { app.stopService(new Intent(app, ManagerService.class)); }
    }
}
