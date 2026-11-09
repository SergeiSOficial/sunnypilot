package ai.sunnypilot.fold3;

import java.util.Arrays;

/** Bounded samples: the soak test must not retain every camera frame. */
public final class FrameStats {
    static final String CAPTURE_SAMPLE_COLUMNS = "sensor_raw,frame_number,exposure,duration,elapsed_before,monotonic,elapsed_after,"
        + "exposure_start_callback_elapsed,readout_start_raw,readout_start_callback_elapsed,rolling_shutter_skew,pipeline_depth";
    private final long[] intervals = new long[4096];
    private final long[] packTimes = new long[4096];
    private final long[] timestamps = new long[4096];
    private final long[] deliveryAges = new long[4096];
    private final long[] acquisitionAges = new long[4096];
    private int acquisitionCount;
    private final long[][] captureSamples = new long[64][];
    // Frame number, exposure callback timestamp/arrival, readout timestamp/arrival.
    private final long[][] captureEvents = new long[captureSamples.length][5];
    private long captureCount;
    private final long[] captureResultAges = new long[4096];
    private int captureResultAgeCount;
    public long captureResultTimestampErrors;
    private int ageCount;
    private final long[][] clockSamples = new long[64][];
    private long clockCount;
    public long lateDeliveries, futureDeliveries, deliveryOver50ms, maxDeliveryAge;
    private int intervalCount;
    private int packCount;
    public long frames;
    public long firstTimestamp;
    public long lastTimestamp;
    public long timestampErrors;
    public long firstHash;
    public long changedFrames;
    public long packingErrors;
    /** REALTIME camera timestamps use elapsed realtime, which includes suspend. Never replace a bad sensor timestamp with arrival time. */
    public synchronized long sensorToMonotonic(long raw, long elapsedBefore, long monotonic, long elapsedAfter) {
        if (elapsedAfter < elapsedBefore) throw new IllegalArgumentException("Clock sample order");
        clockSamples[(int)(clockCount++ % clockSamples.length)] = new long[]{raw, elapsedBefore, monotonic, elapsedAfter};
        return raw - (elapsedBefore + (elapsedAfter - elapsedBefore) / 2 - monotonic);
    }
    public synchronized long[][] clockSamples() {
        return Arrays.copyOf(clockSamples, (int)Math.min(clockCount, clockSamples.length));
    }
    public synchronized void acquired(long sensorTimestamp, long elapsedRealtime) {
        acquisitionAges[acquisitionCount++ % acquisitionAges.length] = elapsedRealtime - sensorTimestamp;
    }
    public synchronized long acquisitionAgeP99() {
        return acquisitionAgePercentile(.99);
    }
    public synchronized long acquisitionAgePercentile(double quantile) {
        return boundedPercentile(acquisitionAges, acquisitionCount, quantile);
    }
    synchronized void captureEvent(long frame, long sensor, long elapsedRealtime, boolean readout) {
        long[] event = captureEvents[(int)(frame % captureEvents.length)];
        if (event[0] != frame) {
            Arrays.fill(event, 0);
            event[0] = frame;
        }
        int index = readout ? 3 : 1;
        event[index] = sensor;
        event[index + 1] = elapsedRealtime;
    }
    public synchronized void capture(long sensor, long frame, long exposure, long duration,
                                     long elapsedBefore, long monotonic, long elapsedAfter,
                                     long rollingShutterSkew, int pipelineDepth) {
        long[] event = captureEvents[(int)(frame % captureEvents.length)];
        long exposureArrival = event[0] == frame && sensor > 0 && event[1] == sensor ? event[2] : -1;
        long readout = exposureArrival > 0 && event[3] >= sensor ? event[3] : -1;
        captureSamples[(int)(captureCount++ % captureSamples.length)] =
            new long[]{sensor, frame, exposure, duration, elapsedBefore, monotonic, elapsedAfter,
                exposureArrival, readout, readout > 0 ? event[4] : -1, rollingShutterSkew, pipelineDepth};
        if (sensor <= 0 || elapsedAfter < elapsedBefore || sensor > elapsedAfter) captureResultTimestampErrors++;
        else captureResultAges[captureResultAgeCount++ % captureResultAges.length] = elapsedBefore - sensor;
    }
    public synchronized long captureResultAgePercentile(double quantile) {
        return boundedPercentile(captureResultAges, captureResultAgeCount, quantile);
    }
    public synchronized long[][] captureSamples() {
        return Arrays.copyOf(captureSamples, (int)Math.min(captureCount, captureSamples.length));
    }
    public synchronized void add(long timestamp, long hash, long packingNs) {
        if (frames == 0) { firstTimestamp = timestamp; firstHash = hash; }
        else if (timestamp <= lastTimestamp) timestampErrors++;
        else intervals[intervalCount++ % intervals.length] = timestamp - lastTimestamp;
        if (hash != firstHash) changedFrames++;
        packTimes[packCount++ % packTimes.length] = packingNs;
        timestamps[(int)(frames % timestamps.length)] = timestamp;
        lastTimestamp = timestamp;
        frames++;
    }
    public synchronized double fps() { return frames < 2 ? 0 : (frames - 1) * 1e9 / (lastTimestamp - firstTimestamp); }
    public synchronized void deliveryAge(long nanoseconds) {
        deliveryAges[ageCount++ % deliveryAges.length] = nanoseconds;
        maxDeliveryAge = Math.max(maxDeliveryAge, nanoseconds);
        if (nanoseconds > 50_000_000L) deliveryOver50ms++;
        if (nanoseconds > 100_000_000L) lateDeliveries++;
        if (nanoseconds < 0) futureDeliveries++;
    }
    public synchronized long deliveryAgeP99() {
        return deliveryAgePercentile(.99);
    }
    public synchronized long deliveryAgePercentile(double quantile) {
        return boundedPercentile(deliveryAges, ageCount, quantile);
    }
    public synchronized long percentile(boolean packing, double quantile) {
        long[] source = packing ? packTimes : intervals;
        return boundedPercentile(source, packing ? packCount : intervalCount, quantile);
    }
    private static long boundedPercentile(long[] source, int count, double quantile) {
        int n = Math.min(count, source.length);
        if (n == 0) return 0;
        long[] copy = Arrays.copyOf(source, n);
        Arrays.sort(copy);
        return copy[(int)Math.ceil((n - 1) * quantile)];
    }
    public synchronized long[] timestamps() {
        long[] copy = Arrays.copyOf(timestamps, (int)Math.min(frames, timestamps.length));
        Arrays.sort(copy);
        return copy;
    }
    public synchronized int matchingCaptureTimestamps() {
        long[] images = timestamps();
        int matches = 0;
        for (long[] capture : captureSamples())
            if (capture[0] > 0 && Arrays.binarySearch(images, capture[0]) >= 0) matches++;
        return matches;
    }
}
