// Diagnostic only: compare pinned Conv, vectorized loads and static geometry.
// GPU events are read after batch submission; no per-kernel waits or CPU fallback.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cl_int code) { if (code != CL_SUCCESS) throw std::runtime_error("OpenCL error " + std::to_string(code)); }
using Clock = std::chrono::steady_clock;

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("Pass pinned fold3_host_conv_buf.cl");
    void *library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) library = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(dlerror());
#define API(name) auto p_##name = reinterpret_cast<decltype(&name)>(dlsym(library, #name)); if (!p_##name) throw std::runtime_error(#name " unavailable")
    API(clGetPlatformIDs); API(clGetDeviceIDs); API(clGetDeviceInfo); API(clCreateContext);
    API(clCreateCommandQueue); API(clCreateBuffer); API(clCreateProgramWithSource); API(clBuildProgram);
    API(clGetProgramBuildInfo); API(clCreateKernel); API(clSetKernelArg); API(clEnqueueNDRangeKernel);
    API(clFinish); API(clGetEventProfilingInfo); API(clEnqueueReadBuffer); API(clReleaseEvent);
    API(clReleaseMemObject); API(clReleaseKernel); API(clReleaseProgram); API(clReleaseCommandQueue); API(clReleaseContext);
#undef API
    cl_uint count = 0; check(p_clGetPlatformIDs(0, nullptr, &count));
    std::vector<cl_platform_id> platforms(count); check(p_clGetPlatformIDs(count, platforms.data(), nullptr));
    cl_device_id device = nullptr;
    for (auto platform : platforms) if (p_clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) == CL_SUCCESS) break;
    if (!device) throw std::runtime_error("No exposed GPU; CPU fallback prohibited");
    char name[256] = {}, version[256] = {};
    check(p_clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr));
    check(p_clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(version), version, nullptr));
    cl_int error = 0;
    auto context = p_clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error); check(error);
    auto queue = p_clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error); check(error);
    std::ifstream input(argv[1]); if (!input) throw std::runtime_error("Kernel source missing");
    std::string source((std::istreambuf_iterator<char>(input)), {}), modified = source;
    source.resize(source.find("// Original FP16 NoTrans Gemm"));
    modified = source;
    size_t position = 0;
    while ((position = modified.find("volatile float4 sum", position)) != std::string::npos) modified.erase(position, 9);
    const auto loop_start = modified.find("            for (int ci = 0; ci < inputChannels; ++ci) {");
    const auto loop_end = modified.find("\n#endif", loop_start);
    if (loop_start == std::string::npos || loop_end == std::string::npos) throw std::runtime_error("Pinned loop changed");
    modified.replace(loop_start, loop_end-loop_start, R"CL(            int ci = 0;
            for (; ci + 3 < inputChannels; ci += 4) {
                float4 value = valid ? vload4(0, input + (((ci / 4 * batch + n) * inHW.x + iy) * inHW.y + ix) * 4) : (float4)0.0f;
                int wi = (((cb * kernelHW.x + ky) * kernelHW.y + kx) * inputChannels + ci);
                sum = fma((float4)value.x, weights[wi], sum);
                sum = fma((float4)value.y, weights[wi + 1], sum);
                sum = fma((float4)value.z, weights[wi + 2], sum);
                sum = fma((float4)value.w, weights[wi + 3], sum);
            }
            for (; ci < inputChannels; ++ci) {
                float value = valid ? input[(((ci / 4 * batch + n) * inHW.x + iy) * inHW.y + ix) * 4 + ci % 4] : 0.0f;
                float4 weight = weights[(((cb * kernelHW.x + ky) * kernelHW.y + kx) * inputChannels + ci)];
                sum = fma((float4)value, weight, sum);
            })CL");
    std::vector<double> build_ms;
    std::cout << "{\"diagnostic_only\":true,\"synthetic_operands\":true,\"bias_order\":\"last\","
                 "\"cpu_fallback\":false,\"vehicle_tx_enabled\":false,\"device\":\"" << name
              << "\",\"driver\":\"" << version << "\",\"local_work_size\":[32,1],\"variant_names\":[\"pinned\",\"vector4\",\"vector4_static\"],\"cases\":[";
    bool first = true, all_exact = true;
    // Actual graph geometries: first road/driver Conv and late road pointwise.
    const int shapes[3][6] = {{24,64,128,256,3,2}, {16,32,240,360,3,2}, {1536,512,4,8,1,1}};
    for (int shape = 0; shape < 3; ++shape) for (unsigned seed : {42u, 7u, 123u}) {
      const auto &s = shapes[shape]; int ci=s[0], co=s[1], h=s[2], w=s[3], kh=s[4], stride=s[5];
      int oh=(h+2*(kh/2)-kh)/stride+1, ow=(w+2*(kh/2)-kh)/stride+1;
      unsigned random = seed;
      auto fill = [&](std::vector<float> &values) { for (auto &x : values) { random = random*1664525u+1013904223u; x=(static_cast<int>(random>>16)-32768)/1048576.0f; } };
      std::vector<float> data(ci*h*w), weights(co*ci*kh*kh), bias(co), outputs[3];
      fill(data); fill(weights); fill(bias);
      cl_mem buffers[4]; std::vector<float> *values[3] = {&data, &weights, &bias};
      for (int i=0;i<3;++i) { buffers[i]=p_clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, values[i]->size()*4, values[i]->data(), &error); check(error); }
      size_t bytes=static_cast<size_t>(co)*oh*ow*4;
      buffers[3]=p_clCreateBuffer(context,CL_MEM_WRITE_ONLY,bytes,nullptr,&error); check(error);
      std::vector<double> times[3]; double wall_ms[3];
      cl_program programs[3]; cl_kernel kernels[3];
      for (int variant=0;variant<3;++variant) {
        auto build_start=Clock::now();
        std::string code = variant ? modified : source;
        if (variant==2) {
          const auto body=code.find("    int cb =");
          auto pair=[](int a,int b){return "((int2)("+std::to_string(a)+","+std::to_string(b)+"))";};
          std::string constants="\n#define batch 1\n#define inputChannels "+std::to_string(ci)+"\n#define outputChannels "+std::to_string(co)+
            "\n#define inHW "+pair(h,w)+"\n#define outHW "+pair(oh,ow)+"\n#define kernelHW "+pair(kh,kh)+
            "\n#define stride "+pair(stride,stride)+"\n#define dilation ((int2)(1,1))\n#define padding "+pair(kh/2,kh/2)+"\n";
          code.insert(body,constants);
        }
        const char *code_pointer=code.c_str();
        programs[variant]=p_clCreateProgramWithSource(context,1,&code_pointer,nullptr,&error);check(error);
        error=p_clBuildProgram(programs[variant],1,&device,"-cl-mad-enable -w",nullptr,nullptr);
        if(error!=CL_SUCCESS){char log[16384]={};p_clGetProgramBuildInfo(programs[variant],device,CL_PROGRAM_BUILD_LOG,sizeof(log),log,nullptr);throw std::runtime_error(log);}
        kernels[variant]=p_clCreateKernel(programs[variant],"fold3_host_conv",&error);check(error);
        build_ms.push_back(std::chrono::duration<double,std::milli>(Clock::now()-build_start).count());
        auto kernel=kernels[variant]; int argument=0, batch=1;
        for (auto buffer : buffers) check(p_clSetKernelArg(kernel,argument++,sizeof(buffer),&buffer));
        for (auto v : {batch,ci,co}) check(p_clSetKernelArg(kernel,argument++,sizeof(v),&v));
        cl_int2 pairs[6] = {{{h,w}},{{oh,ow}},{{kh,kh}},{{stride,stride}},{{1,1}},{{kh/2,kh/2}}};
        for (auto pair : pairs) check(p_clSetKernelArg(kernel,argument++,sizeof(pair),&pair));
        size_t global[2]={static_cast<size_t>(co/4)*ow,static_cast<size_t>(oh)}, local[2]={32,1};
        check(p_clEnqueueNDRangeKernel(queue,kernel,2,nullptr,global,local,0,nullptr,nullptr)); check(p_clFinish(queue));
        auto started=Clock::now(); cl_event events[5];
        for (auto &event : events) check(p_clEnqueueNDRangeKernel(queue,kernel,2,nullptr,global,local,0,nullptr,&event));
        check(p_clFinish(queue)); wall_ms[variant]=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
        for (auto event : events) {
          cl_ulong start=0,end=0; check(p_clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start),&start,nullptr));
          check(p_clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(end),&end,nullptr));
          if (end<start) throw std::runtime_error("GPU clock order invalid");
          times[variant].push_back((end-start)/1e6); check(p_clReleaseEvent(event));
        }
        outputs[variant].resize(bytes/4); check(p_clEnqueueReadBuffer(queue,buffers[3],CL_TRUE,0,bytes,outputs[variant].data(),0,nullptr,nullptr));
      }
      size_t mismatches=0,nonfinite=0; float max_error=0;
      for (size_t i=0;i<bytes/4;++i) {
        for (int v=1;v<3;++v) {
          mismatches += std::memcmp(&outputs[0][i],&outputs[v][i],4)!=0;
          nonfinite += !std::isfinite(outputs[0][i]) || !std::isfinite(outputs[v][i]);
          max_error=std::max(max_error,std::abs(outputs[0][i]-outputs[v][i]));
        }
      }
      all_exact &= mismatches==0 && nonfinite==0;
      if (!first) std::cout << ','; first=false;
      std::cout << "{\"shape\":" << shape << ",\"seed\":" << seed << ",\"elements\":" << bytes/4
                << ",\"bit_mismatches\":" << mismatches << ",\"nonfinite\":" << nonfinite << ",\"max_abs_error\":" << max_error << ",\"variants\":[";
      for (int v=0;v<3;++v) { if (v) std::cout << ','; std::cout << "{\"batch_wall_ms\":" << wall_ms[v] << ",\"gpu_ms\":[";
        for(size_t i=0;i<times[v].size();++i) {if(i) std::cout << ',';std::cout << times[v][i];} std::cout << "]}"; }
      std::cout << "]}";
      for(int v=0;v<3;++v) {check(p_clReleaseKernel(kernels[v]));check(p_clReleaseProgram(programs[v]));}
      for(auto buffer:buffers) check(p_clReleaseMemObject(buffer));
    }
    std::cout << "],\"build_ms\":["; for(size_t i=0;i<build_ms.size();++i){if(i)std::cout<<',';std::cout<<build_ms[i];}
    std::cout << "],\"all_exact\":" << (all_exact ? "true" : "false") << "}\n";
    check(p_clReleaseCommandQueue(queue));check(p_clReleaseContext(context));dlclose(library);
    return all_exact ? 0 : 1;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 2; }
}
