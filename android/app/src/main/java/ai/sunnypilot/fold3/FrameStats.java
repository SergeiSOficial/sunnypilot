package ai.sunnypilot.fold3;

import java.util.Arrays;

/** Bounded samples: the soak test must not retain every camera frame. */
public final class FrameStats {
    private final long[] intervals = new long[4096];
    private final long[] packTimes = new long[4096];
    private final long[] timestamps = new long[4096];
    private final long[] deliveryAges = new long[4096];
    private int ageCount;
    public long lateDeliveries, futureDeliveries;
    private int intervalCount;
    private int packCount;
    public long frames;
    public long firstTimestamp;
    public long lastTimestamp;
    public long timestampErrors;
    public long firstHash;
    public long changedFrames;
    public long packingErrors;
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
        if (nanoseconds > 100_000_000L) lateDeliveries++;
        if (nanoseconds < 0) futureDeliveries++;
    }
    public synchronized long deliveryAgeP99() {
        int count = Math.min(ageCount, deliveryAges.length);
        if (count == 0) return 0;
        long[] copy = Arrays.copyOf(deliveryAges, count);
        Arrays.sort(copy);
        return copy[(int)Math.ceil((count - 1) * .99)];
    }
    public synchronized long percentile(boolean packing, double quantile) {
        long[] source = packing ? packTimes : intervals;
        int n = Math.min(packing ? packCount : intervalCount, source.length);
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
}
