#include <jni.h>
#include <libusb.h>
#include <unistd.h>
#include <array>
#include <cstdint>
#include <sstream>

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
