package ai.sunnypilot.fold3;

import android.content.Context;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import java.io.File;
import java.io.RandomAccessFile;
import java.nio.channels.FileLock;

/** Chaquopy extracts shared APK assets; serialize startup across service processes. */
final class PythonRuntime {
    private PythonRuntime() { }
    static synchronized void start(Context context) {
        if (Python.isStarted()) return;
        try (RandomAccessFile file = new RandomAccessFile(new File(context.getFilesDir(), "python-start.lock"), "rw");
             FileLock lock = file.getChannel().lock()) {
            if (!lock.isValid()) throw new IllegalStateException("Python startup lock unavailable");
            Python.start(new AndroidPlatform(context.getApplicationContext()));
        } catch (java.io.IOException e) { throw new IllegalStateException("Cannot initialize Android Python runtime", e); }
    }
}
