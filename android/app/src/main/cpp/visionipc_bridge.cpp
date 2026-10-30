#include <jni.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include "msgq/visionipc/visionipc_client.h"
#include "msgq/visionipc/visionipc_server.h"
#include "openpilot/cereal/visionstream.h"

extern "C" JNIEXPORT jstring JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_visionIpcRoundTrip(JNIEnv *env, jclass, jstring directory) {
  const char *path = env->GetStringUTFChars(directory, nullptr);
  setenv("MSGQ_ANDROID_PATH", path, 1);
  env->ReleaseStringUTFChars(directory, path);
  try {
    const std::string name = "android_test_" + std::to_string(getpid());
    VisionIpcServer server(name);
    server.create_buffers(VISION_STREAM_NARROW_ROAD, 4, 4, 2);
    server.start_listener();
    VisionIpcClient client(name, VISION_STREAM_NARROW_ROAD, false);
    bool connected = false;
    for (int retry = 0; retry < 100 && !connected; ++retry) {
      connected = client.connect(false);
      if (!connected) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!connected) throw std::runtime_error("VisionIPC client connection timed out");
    for (uint32_t frame = 1; frame <= 100; ++frame) {
      VisionBuf *output = server.get_buffer(VISION_STREAM_NARROW_ROAD);
      std::memset(output->addr, static_cast<int>(frame), output->len);
      output->set_frame_id(frame);
      VisionIpcBufExtra sent{frame, frame * 10000000ULL, frame * 10000000ULL + 1000, false};
      server.send(output, &sent);
      VisionIpcBufExtra received{};
      VisionBuf *input = client.recv(&received, 1000);
      if (!input || input->len != 12 || input->stride != 4 || input->uv_offset != 8 ||
          received.frame_id != frame || received.timestamp_sof != sent.timestamp_sof ||
          received.timestamp_eof != sent.timestamp_eof || received.valid ||
          input->get_frame_id() != frame || std::memcmp(input->addr, output->addr, 12)) {
        throw std::runtime_error("VisionIPC payload, layout or metadata mismatch");
      }
    }
    return env->NewStringUTF("{\"status\":\"passed\",\"frames\":100,\"shared_mapping\":\"memfd/SCM_RIGHTS\",\"vehicle_tx_enabled\":false}");
  } catch (const std::exception &error) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
    return nullptr;
  }
}
