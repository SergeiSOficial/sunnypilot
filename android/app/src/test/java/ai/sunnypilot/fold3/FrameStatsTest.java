package ai.sunnypilot.fold3;

import org.junit.Test;
import static org.junit.Assert.*;

public class FrameStatsTest {
    @Test public void monotonicTimestampsAndBoundedSamples() {
        FrameStats stats = new FrameStats();
        for (int i = 0; i < 10000; i++) stats.add(1_000_000_000L + i * 33_333_333L, i, 1000 + i);
        assertEquals(10000, stats.frames);
        assertEquals(30.0, stats.fps(), .001);
        assertEquals(4096, stats.timestamps().length);
        assertEquals(33_333_333L, stats.percentile(false, .99));
        assertEquals(0, stats.timestampErrors);
        stats.add(stats.lastTimestamp, 0, 0);
        assertEquals(1, stats.timestampErrors);
    }
    @Test public void nearestTimestampPairing() {
        assertEquals(2, CameraProbe.pairingSkew(new long[]{100, 200, 300}, new long[]{102, 202, 302}));
        assertEquals(-1, CameraProbe.pairingSkew(new long[]{100}, new long[]{}));
        assertEquals(2, CameraProbe.pairingSkew(new long[]{1, 2, 100, 200, 300}, new long[]{102, 202, 302}));
        assertEquals(-1, CameraProbe.pairingSkew(new long[]{1, 2}, new long[]{102, 202, 302}));
    }
}
