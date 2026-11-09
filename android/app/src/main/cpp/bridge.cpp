#include <jni.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

extern "C" JNIEXPORT jlong JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_nv12Hash(JNIEnv *env, jclass, jobject data, jint w, jint h) {
  auto *source = data ? static_cast<const uint8_t *>(env->GetDirectBufferAddress(data)) : nullptr;
  if (!source || w <= 0 || h <= 0 || w > 4096 || h > 4096 || ((w | h) & 1) ||
      env->GetDirectBufferCapacity(data) < static_cast<int64_t>(w) * h * 3 / 2) {
    env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Invalid NV12 buffer");
    return 0;
  }
  uint64_t hash = 1469598103934665603ULL;
  for (int64_t i = 0; i < static_cast<int64_t>(w) * h; i += 64) hash = (hash ^ source[i]) * 1099511628211ULL;
  return static_cast<jlong>(hash);
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_rgbaToNv12(JNIEnv *env, jclass, jobject rgba, jint w, jint h, jobject output) {
  auto *src = rgba ? static_cast<uint8_t *>(env->GetDirectBufferAddress(rgba)) : nullptr;
  auto *dst = output ? static_cast<uint8_t *>(env->GetDirectBufferAddress(output)) : nullptr;
  if (!src || !dst || src == dst || w <= 0 || h <= 0 || w > 4096 || h > 4096 || ((w | h) & 1) ||
      env->GetDirectBufferCapacity(rgba) < static_cast<int64_t>(w) * h * 4 ||
      env->GetDirectBufferCapacity(output) < static_cast<int64_t>(w) * h * 3 / 2) {
    env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Invalid RGBA/NV12 dimensions or buffer");
    return 0;
  }
  for (int y = 0; y < h; y += 2) for (int x = 0; x < w; x += 2) {
    int r = 0, g = 0, b = 0;
    for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
      const uint8_t *p = src + ((h - 1 - y - dy) * w + x + dx) * 4;
      dst[(y + dy) * w + x + dx] = ((66 * p[0] + 129 * p[1] + 25 * p[2] + 128) >> 8) + 16;
      r += p[0]; g += p[1]; b += p[2];
    }
    r = (r + 2) / 4; g = (g + 2) / 4; b = (b + 2) / 4;
    size_t uv = w * h + (y / 2) * w + x;
    dst[uv] = std::clamp(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128, 0, 255);
    dst[uv + 1] = std::clamp(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128, 0, 255);
  }
  uint64_t hash = 1469598103934665603ULL;
  for (int64_t i = 0; i < static_cast<int64_t>(w) * h; i += 64) hash = (hash ^ dst[i]) * 1099511628211ULL;
  return static_cast<jlong>(hash);
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_packNv12(JNIEnv *env, jclass, jobject y, jint yr, jint yp,
    jobject u, jint ur, jint up, jobject v, jint vr, jint vp, jint w, jint h, jobject output) {
  auto valid = [&](jobject b, int row, int pixel, int width, int height) {
    return b && row > 0 && pixel > 0 && static_cast<int64_t>(width - 1) * pixel + 1 <= row && env->GetDirectBufferAddress(b) &&
      env->GetDirectBufferCapacity(b) >= static_cast<int64_t>(height - 1) * row + static_cast<int64_t>(width - 1) * pixel + 1;
  };
  if (w <= 0 || h <= 0 || w % 2 || h % 2 || !valid(y, yr, yp, w, h) ||
      !valid(u, ur, up, w / 2, h / 2) || !valid(v, vr, vp, w / 2, h / 2) ||
      !output || !env->GetDirectBufferAddress(output) || env->GetDirectBufferCapacity(output) < static_cast<int64_t>(w) * h * 3 / 2) {
    env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Invalid YUV plane size, dimensions or stride");
    return 0;
  }
  auto *dst = static_cast<uint8_t *>(env->GetDirectBufferAddress(output));
  auto *ys = static_cast<uint8_t *>(env->GetDirectBufferAddress(y));
  auto *us = static_cast<uint8_t *>(env->GetDirectBufferAddress(u));
  auto *vs = static_cast<uint8_t *>(env->GetDirectBufferAddress(v));
  const size_t width = w, height = h;
  for (size_t r = 0; r < height; r++) {
    if (yp == 1) std::memcpy(dst + r * width, ys + r * yr, width);
    else for (size_t c = 0; c < width; c++) dst[r * width + c] = ys[r * yr + c * yp];
  }
  for (size_t r = 0; r < height / 2; r++) for (size_t c = 0; c < width / 2; c++) {
    dst[width * height + r * width + 2 * c] = us[r * ur + c * up];
    dst[width * height + r * width + 2 * c + 1] = vs[r * vr + c * vp];
  }
  uint64_t hash = 1469598103934665603ULL;
  for (int64_t i = 0; i < static_cast<int64_t>(w) * h; i += 64) hash = (hash ^ dst[i]) * 1099511628211ULL;
  return static_cast<jlong>(hash);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_rotateNv12(JNIEnv *env, jclass, jobject input, jint w, jint h, jint rotation, jobject output) {
  auto *src = static_cast<uint8_t *>(env->GetDirectBufferAddress(input));
  auto *dst = static_cast<uint8_t *>(env->GetDirectBufferAddress(output));
  int64_t size = static_cast<int64_t>(w) * h * 3 / 2;
  if (!src || !dst || src == dst || w <= 0 || h <= 0 || w > 4096 || h > 4096 || (w | h) % 2 ||
      (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270) ||
      env->GetDirectBufferCapacity(input) < size || env->GetDirectBufferCapacity(output) < size) {
    env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Invalid NV12 rotation/buffer"); return;
  }
  auto rotate = [&](int width, int height, int channels, const uint8_t *source, uint8_t *target) {
    int target_width = rotation % 180 ? height : width;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
      int tx = rotation == 90 ? height - 1 - y : rotation == 180 ? width - 1 - x : rotation == 270 ? y : x;
      int ty = rotation == 90 ? x : rotation == 180 ? height - 1 - y : rotation == 270 ? width - 1 - x : y;
      for (int c = 0; c < channels; ++c) target[(ty * target_width + tx) * channels + c] = source[(y * width + x) * channels + c];
    }
  };
  rotate(w, h, 1, src, dst);
  rotate(w / 2, h / 2, 2, src + w * h, dst + w * h);
}

// Load the public vendor library through Android's linker namespace.
extern "C" JNIEXPORT jstring JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_openClProbe(JNIEnv *env, jclass) {
  void *lib = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
  if (!lib) return env->NewStringUTF("{\"status\":\"blocked\",\"error\":\"libOpenCL.so inaccessible to app\"}");
  using GetPlatforms = int (*)(unsigned, void **, unsigned *);
  using GetDevices = int (*)(void *, uint64_t, unsigned, void **, unsigned *);
  using GetInfo = int (*)(void *, unsigned, size_t, void *, size_t *);
  auto platforms = reinterpret_cast<GetPlatforms>(dlsym(lib, "clGetPlatformIDs"));
  auto devices = reinterpret_cast<GetDevices>(dlsym(lib, "clGetDeviceIDs"));
  auto info = reinterpret_cast<GetInfo>(dlsym(lib, "clGetDeviceInfo"));
  void *platform = nullptr, *device = nullptr;
  int err = platforms && devices && info ? platforms(1, &platform, nullptr) : -1;
  if (!err) err = devices(platform, 4 /* CL_DEVICE_TYPE_GPU */, 1, &device, nullptr);
  char name[256] = {};
  if (!err) err = info(device, 0x102B /* CL_DEVICE_NAME */, sizeof(name), name, nullptr);
  uint64_t half_config = 0, single_config = 0;
  int half_error = err ? err : info(device, 0x1033 /* CL_DEVICE_HALF_FP_CONFIG */, sizeof(half_config), &half_config, nullptr);
  int single_error = err ? err : info(device, 0x101B /* CL_DEVICE_SINGLE_FP_CONFIG */, sizeof(single_config), &single_config, nullptr);
  std::string safeName;
  for (unsigned char c : std::string(name)) if (c >= 32 && c != '"' && c != '\\') safeName += static_cast<char>(c);
  std::ostringstream result;
  result << "{\"status\":\"" << (err ? "blocked" : "passed") << "\",\"device\":\"" << safeName << "\",\"error_code\":" << err
         << ",\"half_fp_config\":" << half_config << ",\"half_query_error\":" << half_error
         << ",\"single_fp_config\":" << single_config << ",\"single_query_error\":" << single_error << "}";
  dlclose(lib);
  return env->NewStringUTF(result.str().c_str());
}
