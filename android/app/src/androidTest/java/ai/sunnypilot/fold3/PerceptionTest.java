package ai.sunnypilot.fold3;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.chaquo.python.Python;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.nio.ByteBuffer;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class PerceptionTest {
    @Test public void modelFixtureOutputsUseUpstreamParsersAndStayInvalid() throws Exception {
        android.content.Context app = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().getTargetContext();
        PythonRuntime.start(app);
        JSONObject report = new JSONObject(Python.getInstance().getModule("fold3_checks").callAttr("output_messages").toString());
        assertEquals("passed", report.getString("status"));
        assertEquals(3, report.getJSONArray("invalid_messages").length());
        assertFalse(report.getBoolean("vehicle_tx_enabled"));
    }
    @Test public void originalOnnxRuntimeAcceptsChangingInputsAndRejectsWrongLayouts() throws Exception {
        android.content.Context app = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().getTargetContext();
        PythonRuntime.start(app);
        JSONObject report = new JSONObject(Python.getInstance().getModule("fold3_checks").callAttr("ort").toString());
        assertEquals("passed", report.getString("status"));
        assertFalse(report.getBoolean("vehicle_tx_enabled"));
    }
    @Test public void pinnedPandaDecoderRejectsUnsafeStatesWithoutHardware() throws Exception {
        android.content.Context app = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().getTargetContext();
        PythonRuntime.start(app);
        JSONObject report = new JSONObject(Python.getInstance().getModule("fold3_checks").callAttr("panda_decoder").toString());
        assertEquals(6, report.getInt("unsafe_states_rejected"));
        assertFalse(report.getBoolean("physical_hardware_verified"));
        try { NativeBridge.openPandaReceiver(-1); fail("Invalid USB descriptor accepted"); }
        catch (java.io.IOException expected) { }
        try { NativeBridge.readPandaControl(0, 0xdc, 0, 1); fail("Safety mode USB write request exposed"); }
        catch (java.io.IOException expected) { }
    }
    @Test public void rotationsPreserveLumaAndChromaPairs() {
        byte[] original = {1,2,3,4,5,6,7,8,9,10,11,12};
        byte[][] expected = {original, {5,1,6,2,7,3,8,4,9,10,11,12}, {8,7,6,5,4,3,2,1,11,12,9,10}, {4,8,3,7,2,6,1,5,11,12,9,10}};
        for (int i = 0; i < 4; i++) {
            ByteBuffer input = ByteBuffer.allocateDirect(12), output = ByteBuffer.allocateDirect(12);
            input.put(original).rewind();
            NativeBridge.rotateNv12(input, 4, 2, i * 90, output);
            byte[] actual = new byte[12]; output.get(actual);
            assertArrayEquals("rotation " + i * 90, expected[i], actual);
        }
        try { NativeBridge.rotateNv12(ByteBuffer.allocateDirect(12), 4, 2, 45, ByteBuffer.allocateDirect(12)); fail("Invalid angle accepted"); }
        catch (IllegalArgumentException expectedFailure) { }
    }
}
