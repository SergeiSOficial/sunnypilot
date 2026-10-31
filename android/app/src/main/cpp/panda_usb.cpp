#include <jni.h>
#include <libusb.h>
#include <unistd.h>
#include <array>
#include <cstdint>
#include <sstream>
#include <mutex>
#include <memory>
#include <vector>

namespace {
struct Receiver {
  libusb_context *context = nullptr;
  libusb_device_handle *handle = nullptr;
  int fd = -1;
  bool claimed = false;
  ~Receiver() {
    if (claimed) libusb_release_interface(handle, 0);
    if (handle) libusb_close(handle);
    if (context) libusb_exit(context);
    if (fd >= 0) close(fd);
  }
};
std::mutex rx_lock;
std::unique_ptr<Receiver> receiver;
jlong generation = 0;
void fail(JNIEnv *env, const char *reason) {
  env->ThrowNew(env->FindClass("java/io/IOException"), reason);
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_openPandaReceiver(JNIEnv *env, jclass, jint descriptor) {
  std::lock_guard<std::mutex> guard(rx_lock);
  if (receiver) { fail(env, "Panda receiver already open"); return 0; }
  auto next = std::make_unique<Receiver>();
  next->fd = descriptor < 0 ? -1 : dup(descriptor);
  libusb_init_option option{};
  option.option = LIBUSB_OPTION_NO_DEVICE_DISCOVERY;
  int error = next->fd < 0 ? LIBUSB_ERROR_INVALID_PARAM : libusb_init_context(&next->context, &option, 1);
  if (!error) error = libusb_wrap_sys_device(next->context, static_cast<intptr_t>(next->fd), &next->handle);
  libusb_device_descriptor device{};
  if (!error) error = libusb_get_device_descriptor(libusb_get_device(next->handle), &device);
  if (!error && ((device.idVendor != 0xbbaa && device.idVendor != 0x3801) || device.idProduct != 0xddcc)) error = LIBUSB_ERROR_NO_DEVICE;
  if (!error) error = libusb_claim_interface(next->handle, 0);
  if (error) { fail(env, libusb_error_name(error)); return 0; }
  next->claimed = true;
  receiver = std::move(next);
  return ++generation;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_readPandaControl(JNIEnv *env, jclass, jlong token, jint request, jint value, jint length) {
  std::lock_guard<std::mutex> guard(rx_lock);
  // This whitelist is deliberately IN-only. No generic USB transfer API exists.
  bool allowed = request == 0xc1 || request == 0xdd || request == 0xd6 || request == 0xd3 ||
                 request == 0xd4 || request == 0xd2 || request == 0xc2;
  if (!receiver || token != generation || !allowed || length < 1 || length > 128 ||
      (request == 0xc2 ? value < 0 || value > 2 : value != 0)) {
    fail(env, "Invalid read-only Panda request or closed receiver"); return nullptr;
  }
  std::array<unsigned char, 128> bytes{};
  int size = libusb_control_transfer(receiver->handle, LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
                                     request, value, 0, bytes.data(), length, 200);
  if (size < 0) { fail(env, libusb_error_name(size)); return nullptr; }
  jbyteArray result = env->NewByteArray(size);
  if (result) env->SetByteArrayRegion(result, 0, size, reinterpret_cast<jbyte *>(bytes.data()));
  return result;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_readPandaCan(JNIEnv *env, jclass, jlong token) {
  std::lock_guard<std::mutex> guard(rx_lock);
  if (!receiver || token != generation) { fail(env, "Panda receiver closed"); return nullptr; }
  std::array<unsigned char, 16384> bytes{};
  int size = 0;
  int error = libusb_bulk_transfer(receiver->handle, 0x81, bytes.data(), bytes.size(), &size, 100);
  // A timed-out transfer can still contain a complete or partial packet.
  if (error && error != LIBUSB_ERROR_TIMEOUT) { fail(env, libusb_error_name(error)); return nullptr; }
  jbyteArray result = env->NewByteArray(size);
  if (result) env->SetByteArrayRegion(result, 0, size, reinterpret_cast<jbyte *>(bytes.data()));
  return result;
}

extern "C" JNIEXPORT void JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_closePandaReceiver(JNIEnv *, jclass, jlong token) {
  std::lock_guard<std::mutex> guard(rx_lock);
  if (token == generation) receiver.reset();
}

// The pinned Panda protocol uses these vendor IN requests (panda/python/__init__.py).
// No vendor OUT, bulk OUT, safety-mode change, heartbeat, reset or flashing API is exposed.
extern "C" JNIEXPORT jstring JNICALL
Java_ai_sunnypilot_fold3_NativeBridge_pandaUsbProbe(JNIEnv *env, jclass, jint descriptor) {
  libusb_context *context = nullptr;
  libusb_device_handle *handle = nullptr;
  int owned_fd = descriptor < 0 ? -1 : dup(descriptor);
  int error = owned_fd < 0 ? LIBUSB_ERROR_INVALID_PARAM : 0;
  libusb_init_option option{};
  option.option = LIBUSB_OPTION_NO_DEVICE_DISCOVERY;
  if (!error) error = libusb_init_context(&context, &option, 1);
  if (!error) error = libusb_wrap_sys_device(context, static_cast<intptr_t>(owned_fd), &handle);
  libusb_device_descriptor device{};
  if (!error) error = libusb_get_device_descriptor(libusb_get_device(handle), &device);
  if (!error && ((device.idVendor != 0xbbaa && device.idVendor != 0x3801) || device.idProduct != 0xddcc))
    error = LIBUSB_ERROR_NO_DEVICE;
  std::array<unsigned char, 1> type{};
  std::array<unsigned char, 8> versions{};
  std::array<unsigned char, 64> version{};
  auto read = [&](uint8_t request, auto &bytes) {
    return libusb_control_transfer(handle, LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE,
                                   request, 0, 0, bytes.data(), bytes.size(), 1000);
  };
  int type_size = -1, versions_size = -1, version_size = -1;
  if (!error) {
    type_size = read(0xc1, type);
    versions_size = read(0xdd, versions);
    version_size = read(0xd6, version);
    if (type_size != 1 || versions_size != 8 || version_size <= 0) error = LIBUSB_ERROR_IO;
  }
  std::ostringstream json;
  json << "{\"status\":\"" << (error ? "failed" : "passed") << "\",\"libusb_error\":" << error
       << ",\"authorized_fd_wrapped\":" << (handle ? "true" : "false")
       << ",\"hardware_type\":" << static_cast<unsigned>(type[0]) << ",\"firmware_version\":\"";
  for (int i = 0; i < version_size; i++) if (version[i] >= 32 && version[i] < 127 && version[i] != '"' && version[i] != '\\')
    json << version[i];
  json << "\",\"packet_versions_hex\":\"";
  const char *hex = "0123456789abcdef";
  if (versions_size == 8) for (auto b : versions) json << hex[b >> 4] << hex[b & 15];
  json << "\",\"vehicle_tx_enabled\":false,\"firmware_compatibility\":\"blocked: signature qualification not implemented\","
          "\"can_receive\":\"blocked: silent mode and firmware must be verified first\"}";
  if (handle) libusb_close(handle);
  if (context) libusb_exit(context);
  if (owned_fd >= 0) close(owned_fd);
  return env->NewStringUTF(json.str().c_str());
}
