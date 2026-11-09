package ai.sunnypilot.fold3;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONObject;

/** Consumes bounded timing batches in :manager, outside controlsd's Python GIL. */
final class ControlsMetrics {
    static final class Window {
        private final double[] values;
        private int count, next;
        Window(int capacity) { values = new double[capacity]; }
        void add(double value) {
            if (!Double.isFinite(value)) throw new IllegalArgumentException("Non-finite controls timing");
            values[next] = value;
            next = (next + 1) % values.length;
            count = Math.min(count + 1, values.length);
        }
        double p99() {
            if (count == 0) return 0.;
            double[] ordered = Arrays.copyOf(values, count);
            Arrays.sort(ordered);
            return ordered[(int)Math.ceil(count * .99) - 1];
        }
        int size() { return count; }
    }
    private final Map<String, Window> windows = new LinkedHashMap<>();
    private long pid = -1, sequence, observedRows;
    private String[] columns;

    void collect(File directory) throws Exception {
        File batchDirectory = new File(directory, "controlsd-batches");
        File[] batches = batchDirectory.listFiles((dir, name) -> name.endsWith(".json"));
        if (batches == null || batches.length == 0) return;
        if (batches.length > 8) throw new IOException("Controls timing consumer overflow");
        Arrays.sort(batches);
        JSONObject latest = null;
        for (File batchFile : batches) {
            JSONObject batch = new JSONObject(new String(Files.readAllBytes(batchFile.toPath()), StandardCharsets.UTF_8));
            JSONObject data = batch.getJSONObject("data");
            int version = batch.getInt("version");
            if ((version != 2 && version != 3) || data.getBoolean("vehicle_tx_enabled") || data.getBoolean("engagement_allowed"))
                throw new IOException("Invalid controls timing batch");
            if (pid != data.getLong("pid")) {
                pid = data.getLong("pid"); sequence = observedRows = 0; windows.clear(); columns = null;
            }
            if (batch.getLong("sequence") != sequence + 1) throw new IOException("Missing controls timing batch");
            JSONArray names = batch.getJSONArray("columns");
            JSONArray rows = version == 2 ? batch.getJSONArray("rows") : null;
            int rowCount = version == 2 ? rows.length() : batch.getInt("row_count");
            if (rowCount < 0 || rowCount > 1024 || names.length() > 64) throw new IOException("Oversized controls timing batch");
            java.nio.ByteBuffer binary = null;
            if (version == 3) {
                byte[] bytes = java.util.Base64.getDecoder().decode(batch.getString("rows_base64"));
                if (bytes.length != rowCount * names.length() * 8) throw new IOException("Invalid controls timing buffer");
                binary = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            }
            if (names.length() != 0) {
                String[] incoming = new String[names.length()];
                for (int i = 0; i < incoming.length; i++) incoming[i] = names.getString(i);
                if (columns == null) {
                    columns = incoming;
                    for (String name : columns) {
                        if (windows.put(name, new Window(10000)) != null) throw new IOException("Duplicate timing column");
                    }
                } else if (!Arrays.equals(columns, incoming)) throw new IOException("Controls timing columns changed");
            }
            for (int i = 0; i < rowCount; i++) {
                JSONArray row = rows == null ? null : rows.getJSONArray(i);
                if (columns == null || (row != null && row.length() != columns.length)) throw new IOException("Invalid controls timing row");
                for (int j = 0; j < columns.length; j++) windows.get(columns[j]).add(binary == null ? row.getDouble(j) : binary.getDouble());
                observedRows++;
            }
            if (observedRows != data.getLong("work_samples")) throw new IOException("Controls timing sample count mismatch");
            // Preserve the raw intervals and counters without enlarging the
            // frequently polled summary or writing from the controls process.
            Files.write(new File(directory, "controlsd-timing-" + pid + ".jsonl").toPath(),
                (batch.toString() + "\n").getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            sequence++;
            latest = data;
            Files.delete(batchFile.toPath());
        }
        if (latest == null) return;
        JSONObject stages = new JSONObject();
        for (var entry : windows.entrySet()) {
            String name = entry.getKey();
            if (name.equals("work_ns")) latest.put("p99_work_ms", entry.getValue().p99() / 1e6);
            else if (name.equals("complete_cycle_ms")) latest.put("p99_complete_cycle_ms", entry.getValue().p99());
            else stages.put(name, entry.getValue().p99());
        }
        latest.put("p99_stages_ms", stages).put("metrics_observed_samples", observedRows)
            .put("metrics_sequence", sequence).put("metrics_coverage_complete", true)
            .put("metrics_process", "manager").put("metrics_pid", android.os.Process.myPid());
        File output = new File(directory, "controlsd.json"), temporary = new File(directory, "controlsd-metrics.tmp");
        byte[] encoded = latest.toString(2).getBytes(StandardCharsets.UTF_8);
        Files.write(temporary.toPath(), encoded);
        Files.move(temporary.toPath(), output.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        if (latest.optString("phase").equals("failed"))
            Files.write(new File(directory, "controlsd-last-failure.json").toPath(), encoded);
    }
}
