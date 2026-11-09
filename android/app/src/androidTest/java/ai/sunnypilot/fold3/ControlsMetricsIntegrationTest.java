package ai.sunnypilot.fold3;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class ControlsMetricsIntegrationTest {
    private File directory() {
        File result = new File(InstrumentationRegistry.getInstrumentation().getTargetContext().getCacheDir(),
            "controls-metrics-" + System.nanoTime());
        assertTrue(result.mkdirs());
        assertTrue(new File(result, "controlsd-batches").mkdir());
        return result;
    }
    private void clean(File path) {
        File[] children = path.listFiles();
        if (children != null) for (File child : children) clean(child);
        assertTrue(path.delete());
    }
    private void batch(File directory, int sequence, long samples, Object value) throws Exception {
        JSONObject data = new JSONObject().put("pid", 321).put("work_samples", samples).put("phase", "running")
            .put("vehicle_tx_enabled", false).put("engagement_allowed", false);
        JSONObject batch = new JSONObject().put("version", 2).put("sequence", sequence).put("data", data)
            .put("columns", new JSONArray().put("work_ns").put("complete_cycle_ms").put("cycle_interval_ms"))
            .put("rows", new JSONArray().put(new JSONArray().put(value).put(12.).put(10.)));
        Files.write(new File(directory, "controlsd-batches/321-" + sequence + ".json").toPath(),
            batch.toString().getBytes(StandardCharsets.UTF_8));
    }
    @Test public void consumerRetainsRawIntervalsAndComputesStatisticsOutsideControls() throws Exception {
        File dir = directory();
        try {
            batch(dir, 1, 1, 2_000_000);
            new ControlsMetrics().collect(dir);
            JSONObject report = new JSONObject(new String(Files.readAllBytes(new File(dir, "controlsd.json").toPath()), StandardCharsets.UTF_8));
            assertEquals(2., report.getDouble("p99_work_ms"), 0.);
            assertEquals(12., report.getDouble("p99_complete_cycle_ms"), 0.);
            assertEquals(1, report.getInt("metrics_observed_samples"));
            assertEquals("manager", report.getString("metrics_process"));
            assertTrue(report.getBoolean("metrics_coverage_complete"));
            assertTrue(new File(dir, "controlsd-timing-321.jsonl").length() > 0);
            assertEquals(0, new File(dir, "controlsd-batches").list().length);
        } finally { clean(dir); }
    }
    @Test public void missingBatchesFailRatherThanLosingDeadlineMisses() throws Exception {
        File dir = directory();
        try {
            batch(dir, 2, 1, 2_000_000);
            try { new ControlsMetrics().collect(dir); fail("Accepted missing timing batch"); }
            catch (java.io.IOException expected) { assertTrue(expected.getMessage().contains("Missing")); }
        } finally { clean(dir); }
    }
    @Test public void nonfiniteSamplesCannotQualifyTiming() throws Exception {
        File dir = directory();
        try {
            batch(dir, 1, 1, "NaN");
            try { new ControlsMetrics().collect(dir); fail("Accepted nonfinite timing"); }
            catch (IllegalArgumentException expected) { assertTrue(expected.getMessage().contains("Non-finite")); }
        } finally { clean(dir); }
    }
    @Test public void binaryHandoffPreservesEveryDoubleAndCounter() throws Exception {
        File dir = directory();
        try {
            batch(dir, 1, 1, 2_000_000);
            File file = new File(dir, "controlsd-batches/321-1.json");
            JSONObject batch = new JSONObject(new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8));
            batch.remove("rows");
            byte[] raw = java.nio.ByteBuffer.allocate(24).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                .putDouble(2_000_000.).putDouble(12.).putDouble(10.).array();
            batch.put("version", 3).put("row_count", 1).put("rows_base64", java.util.Base64.getEncoder().encodeToString(raw));
            Files.write(file.toPath(), batch.toString().getBytes(StandardCharsets.UTF_8));
            new ControlsMetrics().collect(dir);
            JSONObject report = new JSONObject(new String(Files.readAllBytes(new File(dir, "controlsd.json").toPath()), StandardCharsets.UTF_8));
            assertEquals(2., report.getDouble("p99_work_ms"), 0.);
            assertEquals(12., report.getDouble("p99_complete_cycle_ms"), 0.);
            assertEquals(1, report.getInt("metrics_observed_samples"));
        } finally { clean(dir); }
    }
    @Test public void truncatedBinaryHandoffCannotQualifyTiming() throws Exception {
        File dir = directory();
        try {
            batch(dir, 1, 1, 2_000_000);
            File file = new File(dir, "controlsd-batches/321-1.json");
            JSONObject batch = new JSONObject(new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8));
            batch.remove("rows");
            batch.put("version", 3).put("row_count", 1).put("rows_base64", "AAAA");
            Files.write(file.toPath(), batch.toString().getBytes(StandardCharsets.UTF_8));
            try { new ControlsMetrics().collect(dir); fail("Accepted truncated buffer"); }
            catch (java.io.IOException expected) { assertTrue(expected.getMessage().contains("buffer")); }
        } finally { clean(dir); }
    }
}
