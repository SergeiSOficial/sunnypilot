package ai.sunnypilot.fold3;

import org.junit.Test;
import static org.junit.Assert.*;

public class FrameStatsTest {
    @Test public void agePercentilesIncludeFutureAndLateSamples() {
        FrameStats stats = new FrameStats();
        assertEquals(0, stats.deliveryAgePercentile(.50));
        for (int i = -1; i < 100; i++) {
            stats.acquired(100, 100 + i);
            stats.deliveryAge(i);
        }
        assertEquals(49, stats.acquisitionAgePercentile(.50));
        assertEquals(94, stats.deliveryAgePercentile(.95));
        assertEquals(98, stats.deliveryAgeP99());
        assertEquals(1, stats.futureDeliveries);
    }
    @Test public void acquisitionAndCaptureSamplesRetainRawClocksWithBoundedStorage() {
        FrameStats stats = new FrameStats();
        stats.acquired(120, 100);
        assertEquals(-20, stats.acquisitionAgeP99());
        for (int i = 0; i < 5000; i++) {
            stats.acquired(i, i + 30);
            stats.capture(i, i + 1, 10, 33, i + 30, i + 20, i + 31, -1, -1);
        }
        assertEquals(30, stats.acquisitionAgeP99());
        assertEquals(64, stats.captureSamples().length);
        assertArrayEquals(new long[]{4999, 5000, 10, 33, 5029, 5019, 5030, -1, -1, -1, -1, -1},
            stats.captureSamples()[4999 % 64]);
    }
    @Test public void readoutIsJoinedOnlyToTheMatchingExposureAndFrame() {
        FrameStats stats = new FrameStats();
        stats.captureEvent(1, 100, 103, false);
        stats.captureEvent(1, 110, 113, true);
        stats.capture(100, 1, 10, 33, 135, 130, 136, 8, 3);
        assertArrayEquals(new long[]{100, 1, 10, 33, 135, 130, 136, 103, 110, 113, 8, 3}, stats.captureSamples()[0]);

        stats.capture(101, 1, 10, 33, 135, 130, 136, 8, 3);
        stats.capture(200, 65, 10, 33, 235, 230, 236, 8, 3);
        for (int index : new int[]{1, 2})
            assertArrayEquals(new long[]{-1, -1, -1}, java.util.Arrays.copyOfRange(stats.captureSamples()[index], 7, 10));
    }
    @Test public void sensorClockConversionPreservesAgeIncludingFutureFaults() {
        FrameStats stats = new FrameStats();
        assertEquals(89, stats.sensorToMonotonic(1090, 1100, 100, 1102));
        assertEquals(200, stats.sensorToMonotonic(1200, 1100, 100, 1100));
        for (int i = 0; i < 100; i++) stats.sensorToMonotonic(i, 1100, 100, 1100);
        assertEquals(64, stats.clockSamples().length);
        assertArrayEquals(new long[]{99, 1100, 100, 1100}, stats.clockSamples()[101 % 64]);
    }
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
    @Test public void sensorDeliveryBudgetCountsEveryLateSample() {
        FrameStats stats = new FrameStats();
        stats.deliveryAge(50_000_000L);
        stats.deliveryAge(50_000_001L);
        stats.deliveryAge(101_000_000L);
        stats.deliveryAge(-1);
        assertEquals(2, stats.deliveryOver50ms);
        assertEquals(1, stats.lateDeliveries);
        assertEquals(1, stats.futureDeliveries);
        assertEquals(101_000_000L, stats.maxDeliveryAge);
    }
}
