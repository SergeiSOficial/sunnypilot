package ai.sunnypilot.fold3;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/** Debug-only clip with monotonic capture timestamps and packed NV12 frames. */
final class CameraEvidence implements AutoCloseable {
    private final int frameBytes;
    private static final long PERIOD_NS = 250_000_000L;
    private static final int MAX_FRAMES = 241;
    private final FileChannel file;
    private final ByteBuffer header = ByteBuffer.allocate(Long.BYTES);
    private long lastBucket = -1;
    private int frames;

    CameraEvidence(File directory, String stream) throws IOException {
        this(directory, "multiview-" + stream, 960, 540);
    }

    CameraEvidence(File directory, String name, int width, int height) throws IOException {
        frameBytes = width * height * 3 / 2;
        file = new FileOutputStream(new File(directory, name + ".clip")).getChannel();
    }

    void write(ByteBuffer data, long timestamp) throws IOException {
        long bucket = timestamp / PERIOD_NS;
        if (timestamp <= 0 || bucket <= lastBucket || frames >= MAX_FRAMES) return;
        header.clear();
        header.putLong(timestamp).flip();
        ByteBuffer frame = data.duplicate();
        frame.position(0).limit(frameBytes);
        while (header.hasRemaining()) file.write(header);
        while (frame.hasRemaining()) file.write(frame);
        lastBucket = bucket;
        frames++;
    }

    int frames() { return frames; }

    @Override public void close() throws IOException { file.close(); }
}
