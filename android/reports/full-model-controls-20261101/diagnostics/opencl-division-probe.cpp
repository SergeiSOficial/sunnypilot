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
#include <fstream>
#include <iterator>

using Handle = void *;
using Flags = uint64_t;
using Size = size_t;

static void require(int error, const char *operation) {
  if (error) { std::fprintf(stderr, "%s: OpenCL error %d\n", operation, error); std::exit(1); }
}

int main(int argc, char **argv) {
  if (argc != 2) { std::fprintf(stderr,"Expected OpenCL source file\n"); return 1; }
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

  constexpr Size groups=256, modes=3, stages=8, count=groups*4;
  std::vector<float> left(count), right(count), expected(count), actual(groups*stages*4);
  for(Size i=0;i<count;++i) {
    left[i]=float(i%256)-127.5f; right[i]=i<512?63.75f:255.0f;
    expected[i]=float(double(left[i])/double(right[i]));
  }
  Handle buffers[]={clCreateBuffer(context,4|32,count*sizeof(float),left.data(),&error),nullptr,nullptr};
  require(error,"left");
  buffers[1]=clCreateBuffer(context,4|32,count*sizeof(float),right.data(),&error); require(error,"right");
  buffers[2]=clCreateBuffer(context,1,actual.size()*sizeof(float),nullptr,&error); require(error,"output");
  std::ifstream file(argv[1]);
  std::string source((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
  if (source.empty()) { std::fprintf(stderr,"Missing source\n"); return 1; }
  std::printf("{\"diagnostic_only\":true,\"vehicle_tx_enabled\":false,\"single_fp_config\":%llu,\"outputs\":%zu,\"variants\":[",(unsigned long long)fp,count);
  const char* options[]={"", "-cl-mad-enable -w"};
  for(int variant=0; variant<2; ++variant) {
    const char* text=source.c_str();
    Handle program=clCreateProgramWithSource(context,1,&text,nullptr,&error); require(error,"program");
    error=clBuildProgram(program,1,&device,options[variant],nullptr,nullptr);
    if(error) { Size n=0; clGetProgramBuildInfo(program,device,0x1183,0,nullptr,&n); std::vector<char> log(n+1); clGetProgramBuildInfo(program,device,0x1183,n,log.data(),nullptr); std::fprintf(stderr,"%s\n",log.data()); require(error,"build"); }
    Handle kernel=clCreateKernel(program,"bench_divide",&error); require(error,"kernel");
    for(int i=0;i<3;++i) require(clSetKernelArg(kernel,i,sizeof(Handle),&buffers[i]),"argument");
    Size global=groups; require(clEnqueueNDRangeKernel(queue,kernel,1,nullptr,&global,nullptr,0,nullptr,nullptr),"enqueue");
    require(clEnqueueReadBuffer(queue,buffers[2],1,0,actual.size()*sizeof(float),actual.data(),0,nullptr,nullptr),"read");
    if(variant) std::printf(",");
    std::printf("{\"options\":\"%s\",\"checks\":[",options[variant]);
    for(Size mode=0;mode<modes;++mode) {
      Size unequal=0; double max_error=0;
      for(Size i=0;i<count;++i) { float value=actual[(i/4*stages+mode)*4+i%4]; if(value!=expected[i]) ++unequal; max_error=std::max(max_error,double(std::fabs(value-expected[i]))); }
      if(mode)std::printf(",");
      std::printf("{\"mode\":%zu,\"unequal\":%zu,\"max_abs_error\":%.12g}",mode,unequal,max_error);
    }
    std::printf("],\"examples\":[");
    for(Size i=0;i<4;++i) {
      if(i) std::printf(",");
      std::printf("{\"left\":%.10g,\"right\":%.10g,\"expected\":%.10g,\"stages\":[",left[i],right[i],expected[i]);
      for(Size stage=0;stage<stages;++stage) { if(stage)std::printf(",");std::printf("%.12g",actual[(i/4*stages+stage)*4+i%4]); }
      std::printf("]}");
    }
    std::printf("]}"); clReleaseKernel(kernel);clReleaseProgram(program);
  }
  std::printf("]}\n");
  for(auto buffer:buffers)clReleaseMemObject(buffer);
  clReleaseCommandQueue(queue);clReleaseContext(context);dlclose(library);return 0;
}
