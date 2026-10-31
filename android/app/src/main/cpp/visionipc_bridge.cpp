#include <jni.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <filesystem>
#include <memory>
#include <set>
#include <sys/stat.h>

#include "msgq/visionipc/visionipc_client.h"
#include "msgq/visionipc/visionipc_server.h"
#include "openpilot/cereal/visionstream.h"

struct CameraPublisher {
  VisionIpcServer server{"camerad"};
  std::set<int> streams;
  int width, height;
  CameraPublisher(int w, int h) : width(w), height(h) {}
};

extern "C" JNIEXPORT jlong JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_createCameraPublisher(JNIEnv *env, jclass, jstring directory, jintArray streams,
                                                          jint width, jint height) {
  const char *raw = env->GetStringUTFChars(directory, nullptr);
  const std::string path(raw);
  env->ReleaseStringUTFChars(directory, raw);
  try {
    if (width <= 0 || height <= 0 || width > 4096 || height > 4096 || ((width | height) & 1) != 0)
      throw std::runtime_error("invalid camera dimensions");
    std::filesystem::create_directories(path);
    chmod(path.c_str(), 0700);
    setenv("MSGQ_ANDROID_PATH", path.c_str(), 1);
    auto publisher = std::make_unique<CameraPublisher>(width, height);
    const jsize count = env->GetArrayLength(streams);
    if (count < 1 || count > 3) throw std::runtime_error("invalid stream count");
    jint ids[3];
    env->GetIntArrayRegion(streams, 0, count, ids);
    for (int i = 0; i < count; ++i) {
      if (ids[i] != VISION_STREAM_NARROW_ROAD && ids[i] != VISION_STREAM_WIDE_ROAD && ids[i] != VISION_STREAM_CABIN)
        throw std::runtime_error("unsupported camera stream");
      if (!publisher->streams.insert(ids[i]).second) throw std::runtime_error("duplicate camera stream");
      publisher->server.create_buffers(static_cast<VisionStreamType>(ids[i]), 4, width, height);
    }
    publisher->server.start_listener();
    return reinterpret_cast<jlong>(publisher.release());
  } catch (const std::exception &error) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
    return 0;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_sendCameraFrame(JNIEnv *env, jclass, jlong handle, jint stream, jobject packed,
                                                    jint width, jint height, jlong frame, jlong timestamp) {
  auto *publisher = reinterpret_cast<CameraPublisher *>(handle);
  auto *source = static_cast<const unsigned char *>(env->GetDirectBufferAddress(packed));
  if (!publisher || !source || width != publisher->width || height != publisher->height ||
      !publisher->streams.count(stream) || frame < 0 || timestamp <= 0 ||
      env->GetDirectBufferCapacity(packed) < width * height * 3LL / 2) {
    env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "invalid VisionIPC frame");
    return;
  }
  VisionBuf *buffer = publisher->server.get_buffer(static_cast<VisionStreamType>(stream));
  // VisionIPC owns a bounded ring. The Camera2 Image can be released immediately
  // after this copy; no Java or ImageReader buffer is retained by the consumer.
  for (int y = 0; y < height; ++y) std::memcpy(static_cast<char *>(buffer->addr) + y * buffer->stride, source + y * width, width);
  for (int y = 0; y < height / 2; ++y)
    std::memcpy(static_cast<char *>(buffer->addr) + buffer->uv_offset + y * buffer->stride, source + width * height + y * width, width);
  buffer->set_frame_id(frame);
  // Raw Android frames cannot establish camera/mount calibration.
  VisionIpcBufExtra extra{static_cast<uint32_t>(frame), static_cast<uint64_t>(timestamp), static_cast<uint64_t>(timestamp), false};
  publisher->server.send(buffer, &extra);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_closeCameraPublisher(JNIEnv *, jclass, jlong handle) {
  delete reinterpret_cast<CameraPublisher *>(handle);
}

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
