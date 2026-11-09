// Standalone bench diagnostic. It cannot start the app or access a Panda.
// Compare ordinary bounded FP32/half-weight dot products; this is not model
// qualification. Build with the Android NDK and run through experiment.py.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

using Handle = void *;
using Flags = uint64_t;
using Size = size_t;

static void require(int error, const char *operation) {
  if (error) { std::fprintf(stderr, "%s: OpenCL error %d\n", operation, error); std::exit(1); }
}

int main() {
  void *library = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
  if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
#define API(name, result, ...) \
  auto name = reinterpret_cast<result (*)(__VA_ARGS__)>(dlsym(library, #name)); \
  if (!name) { std::fprintf(stderr, "Missing %s\n", #name); return 1; }
  API(clGetPlatformIDs, int, unsigned, Handle *, unsigned *)
  API(clGetDeviceIDs, int, Handle, Flags, unsigned, Handle *, unsigned *)
  API(clGetDeviceInfo, int, Handle, unsigned, Size, void *, Size *)
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
  API(clWaitForEvents, int, unsigned, const Handle *)
  API(clGetEventProfilingInfo, int, Handle, unsigned, Size, void *, Size *)
  API(clReleaseEvent, int, Handle)
  API(clReleaseKernel, int, Handle)
  API(clReleaseProgram, int, Handle)
  API(clReleaseMemObject, int, Handle)
  API(clReleaseCommandQueue, int, Handle)
  API(clReleaseContext, int, Handle)
#undef API
  Handle platform = nullptr, device = nullptr;
  require(clGetPlatformIDs(1, &platform, nullptr), "platform");
  require(clGetDeviceIDs(platform, 4, 1, &device, nullptr), "GPU device");
  Flags fp = 0;
  require(clGetDeviceInfo(device, 0x101B, sizeof(fp), &fp, nullptr), "FP32 capabilities");
  int error = 0;
  Handle context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
  require(error, "context");
  Handle queue = clCreateCommandQueue(context, device, 2, &error);
  require(error, "profiling queue");
  constexpr Size count = 65536, terms = 128;
  std::vector<float> inputs(count + terms), weights(terms), expected(count), actual(count);
  uint32_t random = 42;
  auto sample = [&]() { random = random * 1664525U + 1013904223U; return (int32_t(random) / 2147483648.0f) * .25f; };
  for (float &value : inputs) value = sample();
  for (float &value : weights) value = static_cast<float>(static_cast<_Float16>(sample()));
  for (Size i = 0; i < count; ++i) {
    float value = .125f;
    for (Size k = 0; k < terms; ++k) value = std::fma(inputs[i + k], weights[k], value);
    expected[i] = value;
  }
  Handle buffers[] = {
    clCreateBuffer(context, 4 | 32, inputs.size() * sizeof(float), inputs.data(), &error), nullptr, nullptr};
  require(error, "input buffer");
  buffers[1] = clCreateBuffer(context, 4 | 32, weights.size() * sizeof(float), weights.data(), &error);
  require(error, "weight buffer");
  buffers[2] = clCreateBuffer(context, 1, count * sizeof(float), nullptr, &error);
  require(error, "output buffer");
  std::printf("{\"diagnostic_only\":true,\"vehicle_tx_enabled\":false,\"single_fp_config\":%llu,\"outputs\":%zu,\"terms\":%zu,\"variants\":[",
              static_cast<unsigned long long>(fp), count, terms);
  const char *names[] = {"fma", "mad", "separate"};
  const char *expressions[] = {"fma(a[i+k],b[k],v)", "mad(a[i+k],b[k],v)", "(a[i+k]*b[k])+v"};
  for (unsigned mode = 0; mode < 3; ++mode) {
    std::string source = std::string("#pragma OPENCL FP_CONTRACT OFF\n__kernel void dot(__global const float*a,__global const float*b,__global float*c){int i=get_global_id(0);float v=.125f;for(int k=0;k<128;k++)v=") + expressions[mode] + ";c[i]=v;}";
    const char *text = source.c_str();
    Handle program = clCreateProgramWithSource(context, 1, &text, nullptr, &error);
    require(error, "program");
    error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (error) {
      char log[16384] = {};
      clGetProgramBuildInfo(program, device, 0x1183, sizeof(log), log, nullptr);
      std::fprintf(stderr, "%s\n", log);
      require(error, "build");
    }
    Handle kernel = clCreateKernel(program, "dot", &error);
    require(error, "kernel");
    for (unsigned i = 0; i < 3; ++i) require(clSetKernelArg(kernel, i, sizeof(Handle), &buffers[i]), "argument");
    double elapsed = 0;
    for (unsigned repeat = 0; repeat < 4; ++repeat) {
      Handle event = nullptr;
      const Size local = 32;
      require(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &count, &local, 0, nullptr, &event), "enqueue");
      require(clWaitForEvents(1, &event), "wait");
      uint64_t start = 0, end = 0;
      require(clGetEventProfilingInfo(event, 0x1282, sizeof(start), &start, nullptr), "start");
      require(clGetEventProfilingInfo(event, 0x1283, sizeof(end), &end, nullptr), "end");
      if (repeat) elapsed += (end - start) / 1e6;
      clReleaseEvent(event);
    }
    require(clEnqueueReadBuffer(queue, buffers[2], 1, 0, count * sizeof(float), actual.data(), 0, nullptr, nullptr), "readback");
    Size different = 0, outside = 0, nonfinite = 0;
    double maximum = 0;
    for (Size i = 0; i < count; ++i) {
      different += std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0;
      nonfinite += !std::isfinite(actual[i]);
      double delta = std::abs(double(actual[i]) - expected[i]);
      maximum = std::max(maximum, delta);
      outside += !std::isfinite(actual[i]) || delta > 1e-4 + 1e-3 * std::abs(expected[i]);
    }
    std::printf("%s{\"name\":\"%s\",\"event_mean_ms\":%.9f,\"different_elements\":%zu,\"outside_original_tolerance\":%zu,\"nonfinite\":%zu,\"max_abs_error\":%.12g}",
                mode ? "," : "", names[mode], elapsed / 3, different, outside, nonfinite, maximum);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
  }
  std::printf("]}\n");
  for (Handle buffer : buffers) clReleaseMemObject(buffer);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  dlclose(library);
}
