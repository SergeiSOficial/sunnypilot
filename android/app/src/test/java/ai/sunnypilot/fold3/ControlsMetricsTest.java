package ai.sunnypilot.fold3;

import org.junit.Test;
import static org.junit.Assert.*;

public class ControlsMetricsTest {
    @Test public void boundedWindowUsesTheExistingNearestRankPercentile() {
        ControlsMetrics.Window window = new ControlsMetrics.Window(100);
        assertEquals(0., window.p99(), 0.);
        for (int i = 0; i < 200; i++) window.add(i);
        assertEquals(100, window.size());
        assertEquals(198., window.p99(), 0.);
    }
    @Test(expected = IllegalArgumentException.class) public void numericalFaultsCannotDisappearFromStatistics() {
        new ControlsMetrics.Window(10).add(Double.NaN);
    }
}
