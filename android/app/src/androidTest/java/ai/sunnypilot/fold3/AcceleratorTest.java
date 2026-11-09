package ai.sunnypilot.fold3;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.chaquo.python.Python;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class AcceleratorTest {
    @Test public void diagnosticEnginesCannotSelectCpuOrIncompleteQualification() throws Exception {
        android.content.Context app = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().getTargetContext();
        PythonRuntime.start(app);
        JSONObject report = new JSONObject(Python.getInstance().getModule("fold3_checks").callAttr("accelerator_contract").toString());
        assertEquals("passed", report.getString("status"));
        assertTrue(report.getBoolean("cpu_selection_rejected"));
        assertTrue(report.getBoolean("incomplete_qualification_rejected"));
        assertFalse(report.getBoolean("vehicle_tx_enabled"));
    }
}
