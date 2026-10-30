package ai.sunnypilot.fold3;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class RuntimeTest {
    @Test public void androidCerealAndMsgqPreservePayloadAndValidity() throws Exception {
        android.content.Context app = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().getTargetContext();
        if (!Python.isStarted()) Python.start(new AndroidPlatform(app));
        JSONObject report = new JSONObject(Python.getInstance().getModule("fold3_runtime")
            .callAttr("probe", app.getFilesDir().getAbsolutePath()).toString());
        assertEquals("passed", report.getJSONObject("modules").getJSONObject("capnp").getString("status"));
        assertEquals("passed", report.getJSONObject("modules").getJSONObject("openpilot.cereal").getString("status"));
        assertEquals("passed", report.getJSONObject("cereal_msgq_round_trip").getString("status"));
        assertFalse(report.getJSONObject("cereal_msgq_round_trip").getBoolean("valid"));
        assertFalse(report.getBoolean("driving_runtime_ready"));
        assertFalse(report.getBoolean("vehicle_tx_enabled"));
    }
}
