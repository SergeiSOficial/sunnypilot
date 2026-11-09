package ai.sunnypilot.fold3;

import android.opengl.GLES30;
import java.nio.ByteBuffer;

/** GLES3 BT.601 packing with two bounded, nonblocking readbacks. GL-thread only. */
final class GpuNv12 implements AutoCloseable {
    interface Sink { void receive(ByteBuffer data, long frame, long timestamp, long started) throws Exception; }
    final int width, height, bytes;
    final int[] textures = new int[2], framebuffers = new int[2], buffers = new int[2];
    private final long[] fences = new long[2], frames = new long[2], stamps = new long[2], starts = new long[2];
    private final int program;
    long dropped;

    GpuNv12(int width, int height) {
        if (width <= 0 || height <= 0 || width % 4 != 0 || height % 2 != 0) throw new IllegalArgumentException("NV12 GPU dimensions");
        this.width = width; this.height = height; bytes = width * height * 3 / 2;
        GLES30.glGenTextures(2, textures, 0); GLES30.glGenFramebuffers(2, framebuffers, 0);
        for (int i = 0; i < 2; i++) {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textures[i]);
            GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, GLES30.GL_RGBA8, i == 0 ? width : width / 4, i == 0 ? height : height * 3 / 2);
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST);
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST);
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, framebuffers[i]);
            GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, textures[i], 0);
            if (GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER) != GLES30.GL_FRAMEBUFFER_COMPLETE)
                throw new IllegalStateException("GPU NV12 framebuffer incomplete");
        }
        int vertex = shader(GLES30.GL_VERTEX_SHADER, "#version 300 es\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0.,1.);}");
        int fragment = shader(GLES30.GL_FRAGMENT_SHADER, "#version 300 es\nprecision highp float;precision highp int;"
            + "uniform sampler2D source;uniform int h;out vec4 color;"
            + "int dot3(ivec3 a,ivec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}"
            + "ivec3 rgb(int x,int y){return ivec3(round(texelFetch(source,ivec2(x,h-1-y),0).rgb*255.));}"
            + "int value(int x,int row){if(row<h){ivec3 c=rgb(x,row);return ((dot3(c,ivec3(66,129,25))+128)>>8)+16;}"
            + "int y=(row-h)*2;int sx=x&~1;ivec3 c=(rgb(sx,y)+rgb(sx+1,y)+rgb(sx,y+1)+rgb(sx+1,y+1)+2)/4;"
            + "ivec3 k=(x&1)==0?ivec3(-38,-74,112):ivec3(112,-94,-18);return clamp(((dot3(c,k)+128)>>8)+128,0,255);}"
            + "void main(){ivec2 q=ivec2(gl_FragCoord.xy);int x=q.x*4;color=vec4(value(x,q.y),value(x+1,q.y),value(x+2,q.y),value(x+3,q.y))/255.;}");
        program = GLES30.glCreateProgram(); GLES30.glAttachShader(program, vertex); GLES30.glAttachShader(program, fragment);
        GLES30.glLinkProgram(program); GLES30.glDeleteShader(vertex); GLES30.glDeleteShader(fragment);
        int[] linked = new int[1]; GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) throw new IllegalStateException(GLES30.glGetProgramInfoLog(program));
        GLES30.glUseProgram(program);
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "source"), 0);
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "h"), height);
        GLES30.glGenBuffers(2, buffers, 0);
        for (int buffer : buffers) {
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, buffer);
            GLES30.glBufferData(GLES30.GL_PIXEL_PACK_BUFFER, bytes, null, GLES30.GL_STREAM_READ);
        }
        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0); GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        check();
    }

    private static int shader(int type, String source) {
        int shader = GLES30.glCreateShader(type); GLES30.glShaderSource(shader, source); GLES30.glCompileShader(shader);
        int[] compiled = new int[1]; GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) throw new IllegalStateException(GLES30.glGetShaderInfoLog(shader));
        return shader;
    }

    void bindSource() { GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, framebuffers[0]); }
    boolean pending() { return fences[0] != 0 || fences[1] != 0; }

    void submit(long frame, long timestamp, long started) {
        int slot = fences[0] == 0 ? 0 : fences[1] == 0 ? 1 : -1;
        if (slot < 0) { dropped++; return; }
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, framebuffers[1]);
        GLES30.glViewport(0, 0, width / 4, height * 3 / 2); GLES30.glUseProgram(program);
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0); GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textures[0]);
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3);
        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, buffers[slot]);
        GLES30.glReadPixels(0, 0, width / 4, height * 3 / 2, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, 0);
        fences[slot] = GLES30.glFenceSync(GLES30.GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        if (fences[slot] == 0) throw new IllegalStateException("GPU NV12 fence allocation failed");
        frames[slot] = frame; stamps[slot] = timestamp; starts[slot] = started;
        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0); GLES30.glFlush(); check();
    }

    void drain(Sink sink) throws Exception {
        // Submission order matters even when the second PBO completes first.
        for (int attempt = 0; attempt < 2; attempt++) {
            int slot = fences[0] == 0 ? 1 : fences[1] == 0 || frames[0] < frames[1] ? 0 : 1;
            if (fences[slot] == 0) break;
            int state = GLES30.glClientWaitSync(fences[slot], 0, 0);
            if (state == GLES30.GL_WAIT_FAILED) throw new IllegalStateException("GPU NV12 fence failed");
            if (state == GLES30.GL_TIMEOUT_EXPIRED) break;
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, buffers[slot]);
            ByteBuffer data = (ByteBuffer) GLES30.glMapBufferRange(GLES30.GL_PIXEL_PACK_BUFFER, 0, bytes, GLES30.GL_MAP_READ_BIT);
            if (data == null) throw new IllegalStateException("GPU NV12 readback mapping failed");
            try { sink.receive(data, frames[slot], stamps[slot], starts[slot]); }
            finally {
                boolean intact = GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER);
                GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0);
                GLES30.glDeleteSync(fences[slot]); fences[slot] = 0;
                if (!intact) throw new IllegalStateException("GPU NV12 readback corrupted");
            }
        }
        check();
    }

    static void check() {
        int error = GLES30.glGetError();
        if (error != GLES30.GL_NO_ERROR) throw new IllegalStateException("GPU NV12 GL error " + error);
    }

    @Override public void close() {
        for (long fence : fences) if (fence != 0) GLES30.glDeleteSync(fence);
        GLES30.glDeleteBuffers(2, buffers, 0); GLES30.glDeleteFramebuffers(2, framebuffers, 0);
        GLES30.glDeleteTextures(2, textures, 0); GLES30.glDeleteProgram(program);
    }
}
