package ai.sunnypilot.fold3;

import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class LifecycleTest {
    @Test public void activitySurvivesRecreation() {
        try (ActivityScenario<MainActivity> activity = ActivityScenario.launch(MainActivity.class)) {
            for (int i = 0; i < 3; i++) activity.recreate();
            activity.onActivity(a -> assertFalse(a.isFinishing()));
        }
    }
}
