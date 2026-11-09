// Standalone diagnostic only. It cannot start controls or access vehicle hardware.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using Handle = void *;
using Size = size_t;
using Flags = uint64_t;
static void check(int error, const char *where) {
  if (error) { std::fprintf(stderr, "%s: %d\n", where, error); std::exit(1); }
}
static std::vector<float> read_floats(const std::string &path, Size count) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in || in.tellg() != static_cast<std::streamoff>(count * sizeof(float))) {
    std::fprintf(stderr, "invalid fixture: %s\n", path.c_str()); std::exit(1);
  }
  in.seekg(0);
  std::vector<float> values(count);
  if (!in.read(reinterpret_cast<char *>(values.data()), count * sizeof(float))) std::exit(1);
  return values;
}
int main(int argc, char **argv) {
  bool parallel = false, save_output = false;
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--parallel") == 0 && !parallel) parallel = true;
    else if (std::strcmp(argv[i], "--save-output") == 0 && !save_output) save_output = true;
    else { std::fprintf(stderr, "unknown or repeated option: %s\n", argv[i]); return 1; }
  }
  if (argc < 3) { std::fprintf(stderr, "usage: probe fixture-directory kernel.cl [--parallel] [--save-output]\n"); return 1; }
  const std::string path = argv[1];
  auto input = read_floats(path + "/input.f32", 1024);
  auto weights = read_floats(path + "/weights.f32", 2048 * 1024);
  auto bias = read_floats(path + "/bias.f32", 2048);
  auto expected = read_floats(path + "/expected.f32", 2048);
  std::ifstream kernel_file(argv[2]);
  if (!kernel_file) return 1;
  std::string source((std::istreambuf_iterator<char>(kernel_file)), {});
  void *library = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
  if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
#define API(name, result, ...) auto name = reinterpret_cast<result (*)(__VA_ARGS__)>(dlsym(library, #name)); if (!name) { std::fprintf(stderr, "missing %s\n", #name); return 1; }
  API(clGetPlatformIDs, int, unsigned, Handle *, unsigned *)
  API(clGetDeviceIDs, int, Handle, Flags, unsigned, Handle *, unsigned *)
  API(clCreateContext, Handle, const intptr_t *, unsigned, const Handle *, void *, void *, int *)
  API(clCreateCommandQueue, Handle, Handle, Handle, Flags, int *)
  API(clCreateBuffer, Handle, Handle, Flags, Size, void *, int *)
  API(clCreateProgramWithSource, Handle, Handle, unsigned, const char **, const Size *, int *)
  API(clBuildProgram, int, Handle, unsigned, const Handle *, const char *, void *, void *)
  API(clGetProgramBuildInfo, int, Handle, Handle, unsigned, Size, void *, Size *)
  API(clCreateKernel, Handle, Handle, const char *, int *)
  API(clSetKernelArg, int, Handle, unsigned, Size, const void *)
  API(clEnqueueNDRangeKernel, int, Handle, Handle, unsigned, const Size *, const Size *, const Size *, unsigned, const Handle *, Handle *)
  API(clEnqueueReadBuffer, int, Handle, Handle, unsigned, Size, Size, void *, unsigned, const Handle *, Handle *)
  API(clGetEventProfilingInfo, int, Handle, unsigned, Size, void *, Size *)
  API(clReleaseEvent, int, Handle)
  API(clReleaseKernel, int, Handle)
  API(clReleaseProgram, int, Handle)
  API(clReleaseMemObject, int, Handle)
  API(clReleaseCommandQueue, int, Handle)
  API(clReleaseContext, int, Handle)
#undef API
  Handle platform = nullptr, device = nullptr;
  check(clGetPlatformIDs(1, &platform, nullptr), "platform");
  check(clGetDeviceIDs(platform, 4, 1, &device, nullptr), "GPU device");
  int error = 0;
  Handle context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error); check(error, "context");
  Handle queue = clCreateCommandQueue(context, device, 2, &error); check(error, "profiling queue");
  std::vector<float> actual(2048);
  std::vector<float> *vectors[] = {&input, &weights, &bias, &actual};
  Handle buffers[4] = {};
  for (int i = 0; i < 4; ++i) {
    buffers[i] = clCreateBuffer(context, i == 3 ? 1 : 4 | 32, vectors[i]->size() * sizeof(float),
                               i == 3 ? nullptr : vectors[i]->data(), &error);
    check(error, "buffer");
  }
  const char *text = source.c_str();
  Handle program = clCreateProgramWithSource(context, 1, &text, nullptr, &error); check(error, "program");
  error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  if (error) {
    char log[16384] = {};
    clGetProgramBuildInfo(program, device, 0x1183, sizeof(log), log, nullptr);
    std::fprintf(stderr, "%s\n", log); check(error, "build");
  }
  Handle kernel = clCreateKernel(program, "road_dense", &error); check(error, "kernel");
  for (unsigned i = 0; i < 4; ++i) check(clSetKernelArg(kernel, i, sizeof(Handle), &buffers[i]), "argument");
  const Size global = parallel ? 2048 * 8 : 2048, local = 32;
  Handle event = nullptr;
  check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, &event), "enqueue");
  check(clEnqueueReadBuffer(queue, buffers[3], 1, 0, actual.size() * sizeof(float), actual.data(), 0, nullptr, nullptr), "readback");
  uint64_t start = 0, end = 0;
  check(clGetEventProfilingInfo(event, 0x1282, sizeof(start), &start, nullptr), "event start");
  check(clGetEventProfilingInfo(event, 0x1283, sizeof(end), &end, nullptr), "event end");
  if (end < start) { std::fprintf(stderr, "invalid event timestamps\n"); return 1; }
  Size mismatch = 0, nonfinite = 0, exact_mismatch = 0;
  double max_abs = 0;
  for (Size i = 0; i < actual.size(); ++i) {
    const double delta = std::abs(double(actual[i]) - expected[i]);
    nonfinite += !std::isfinite(actual[i]);
    mismatch += !std::isfinite(actual[i]) || delta > 0.0001 + 0.001 * std::abs(expected[i]);
    exact_mismatch += actual[i] != expected[i];
    if (std::isfinite(actual[i])) max_abs = std::max(max_abs, delta);
  }
  if (save_output) {
    std::ofstream out(path + "/actual.f32", std::ios::binary | std::ios::trunc);
    if (!out.write(reinterpret_cast<const char *>(actual.data()), actual.size() * sizeof(float))) {
      std::fprintf(stderr, "cannot save device output\n"); return 1;
    }
  }
  std::printf("{\"diagnostic_only\":true,\"vehicle_tx_enabled\":false,\"gpu_only\":true,\"count\":%zu,\"mismatches\":%zu,\"exact_mismatches\":%zu,\"nonfinite\":%zu,\"max_abs_error\":%.12g,\"one_kernel_event_ms\":%.9f}\n", actual.size(), mismatch, exact_mismatch, nonfinite, max_abs, (end - start) / 1e6);
  clReleaseEvent(event);
  clReleaseKernel(kernel); clReleaseProgram(program);
  for (Handle buffer : buffers) clReleaseMemObject(buffer);
  clReleaseCommandQueue(queue); clReleaseContext(context); dlclose(library);
  return mismatch || nonfinite ? 1 : 0;
}
