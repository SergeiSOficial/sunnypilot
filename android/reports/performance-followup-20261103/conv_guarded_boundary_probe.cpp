// Diagnostic only: guarded exact-half products versus pinned sequential FMA.
// No model or APK is changed. Non-half operands must retain strict FMA.
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
    cl_device_fp_config fp=0;cl_uint units=0;
    check(p_clGetDeviceInfo(device,CL_DEVICE_SINGLE_FP_CONFIG,sizeof(fp),&fp,nullptr));
    check(p_clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(units),&units,nullptr));
    cl_int error = 0;
    auto context = p_clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error); check(error);
    auto queue = p_clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error); check(error);
    std::ifstream input(argv[1]); if (!input) throw std::runtime_error("Kernel source missing");
    std::string source((std::istreambuf_iterator<char>(input)), {});
    source.resize(source.find("// Original FP16 NoTrans Gemm"));
    std::vector<double> build_ms;
    std::cout << "{\"diagnostic_only\":true,\"synthetic_operands\":true,\"rtol\":0.001,\"atol\":0.0001,"
                 "\"cpu_fallback\":false,\"vehicle_tx_enabled\":false,\"device\":\"" << name
              << "\",\"driver\":\"" << version << "\",\"single_fp_config\":" << fp << ",\"native_fma_flag\":" << ((fp & CL_FP_FMA) ? "true" : "false") << ",\"compute_units\":" << units << ",\"local_work_size\":[32,1],\"variant_names\":[\"pinned_fma\",\"guarded_half_product\"],\"cases\":[";
    bool first = true, all_exact = true, all_tolerance_passed = true;
    // Actual graph geometries: first road/driver Conv and late road pointwise.
    const int shapes[6][6] = {{24,64,128,256,3,2}, {16,32,240,360,3,2}, {1536,512,4,8,1,1}, {3,5,9,11,3,2}, {7,9,13,15,3,1}, {1,1,8,7,1,1}};
    for (int shape = 0; shape < 6; ++shape) for (unsigned seed : {42u, 7u, 123u}) for (int bias_first : {0, 1}) for (bool half_operands : {false, true}) {
      const auto &s = shapes[shape]; int ci=s[0], co=s[1], h=s[2], w=s[3], kh=s[4], stride=s[5];
      int dilation=shape==4 ? 2 : 1, pad=kh/2, batch=shape>=3 ? 2 : 1;
      int oh=(h+2*pad-dilation*(kh-1)-1)/stride+1, ow=(w+2*pad-dilation*(kh-1)-1)/stride+1;
      unsigned random = seed;
      auto fill = [&](std::vector<float> &values) { for (auto &x : values) { random = random*1664525u+1013904223u; x=(static_cast<int>(random>>16)-32768)/32768.0f; } };
      int padded_ci=(ci+3)/4*4, padded_co=(co+3)/4*4;
      std::vector<float> data(padded_ci*batch*h*w), weights(padded_co*ci*kh*kh), bias(padded_co), outputs[2];
      fill(data); fill(weights); fill(bias);
      if (half_operands) {
        for (auto *values : {&data, &weights}) for (auto &value : *values) {
          if (seed == 123) {
            random = random*1664525u+1013904223u;
            unsigned short bits = random >> 16;
            if ((bits & 0x7c00) == 0x7c00) bits &= 0x8000;
            _Float16 half;
            std::memcpy(&half, &bits, sizeof(half));
            value = static_cast<float>(half);
          } else value = static_cast<float>(static_cast<_Float16>(value));
        }
      }
      cl_mem buffers[4]; std::vector<float> *values[3] = {&data, &weights, &bias};
      for (int i=0;i<3;++i) { buffers[i]=p_clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, values[i]->size()*4, values[i]->data(), &error); check(error); }
      size_t bytes=static_cast<size_t>(padded_co)*batch*oh*ow*4;
      buffers[3]=p_clCreateBuffer(context,CL_MEM_WRITE_ONLY,bytes,nullptr,&error); check(error);
      std::vector<double> times[2]; double wall_ms[2];
      cl_program programs[2]; cl_kernel kernels[2];
      for (int variant=0;variant<2;++variant) {
        auto build_start=Clock::now();
        std::string code = source;
        std::string options = "-cl-mad-enable -w -DFOLD3_VECTOR_FMA_CONV";
        if (variant) {
          const std::string needle = "fma(";
          for (size_t pos = 0; (pos = code.find(needle, pos)) != std::string::npos; pos += 12)
            code.replace(pos, needle.size(), "guarded_fma(");
          code = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF
inline float4 guarded_fma(float4 a, float4 b, float4 c) {
    // Finite binary16 products are exactly representable in binary32.
    // Preserve a rounded FP32 accumulator update; never approximate an FMA
    // whose product needs rounding. The runtime guards require both operands.
    int4 exact = isfinite(a) & isfinite(b)
        & (a == convert_float4(convert_half4_rte(a)))
        & (b == convert_float4(convert_half4_rte(b)));
    if (all(exact)) {
        volatile float4 product = a * b;
        return product + c;
    }
    return fma(a, b, c);
}
)CLC" + code;
        }
        if (bias_first) options += " -DFOLD3_BIAS_FIRST";
        if (seed==7) options += " -DFOLD3_RELU";
        const char *code_pointer=code.c_str();
        programs[variant]=p_clCreateProgramWithSource(context,1,&code_pointer,nullptr,&error);check(error);
        error=p_clBuildProgram(programs[variant],1,&device,options.c_str(),nullptr,nullptr);
        if(error!=CL_SUCCESS){char log[16384]={};p_clGetProgramBuildInfo(programs[variant],device,CL_PROGRAM_BUILD_LOG,sizeof(log),log,nullptr);throw std::runtime_error(log);}
        kernels[variant]=p_clCreateKernel(programs[variant],"fold3_host_conv",&error);check(error);
        build_ms.push_back(std::chrono::duration<double,std::milli>(Clock::now()-build_start).count());
        auto kernel=kernels[variant]; int argument=0;
        for (auto buffer : buffers) check(p_clSetKernelArg(kernel,argument++,sizeof(buffer),&buffer));
        for (auto v : {batch,ci,co}) check(p_clSetKernelArg(kernel,argument++,sizeof(v),&v));
        cl_int2 pairs[6] = {{{h,w}},{{oh,ow}},{{kh,kh}},{{stride,stride}},{{dilation,dilation}},{{pad,pad}}};
        for (auto pair : pairs) check(p_clSetKernelArg(kernel,argument++,sizeof(pair),&pair));
        size_t global[2]={static_cast<size_t>((padded_co/4*ow+31)/32*32),static_cast<size_t>(batch*oh)}, local[2]={32,1};
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
      size_t mismatches=0,nonfinite=0,tolerance_failures=0; float max_error=0;
      for (size_t i=0;i<bytes/4;++i) {
        for (int v=1;v<2;++v) {
          tolerance_failures += std::abs(outputs[0][i]-outputs[v][i]) > 1e-4f+1e-3f*std::abs(outputs[0][i]);
          mismatches += std::memcmp(&outputs[0][i],&outputs[v][i],4)!=0;
          nonfinite += !std::isfinite(outputs[0][i]) || !std::isfinite(outputs[v][i]);
          max_error=std::max(max_error,std::abs(outputs[0][i]-outputs[v][i]));
        }
      }
      all_exact &= mismatches==0 && nonfinite==0;
      all_tolerance_passed &= tolerance_failures==0 && nonfinite==0;
      if (!first) std::cout << ','; first=false;
      std::cout << "{\"shape\":" << shape << ",\"seed\":" << seed << ",\"bias_first\":" << bias_first << ",\"half_operands\":" << (half_operands ? "true" : "false") << ",\"batch\":" << batch << ",\"dilation\":" << dilation << ",\"relu\":" << (seed==7 ? "true" : "false") << ",\"elements\":" << bytes/4
                << ",\"bit_mismatches\":" << mismatches << ",\"tolerance_failures\":" << tolerance_failures << ",\"nonfinite\":" << nonfinite << ",\"max_abs_error\":" << max_error << ",\"variants\":[";
      for (int v=0;v<2;++v) { if (v) std::cout << ','; std::cout << "{\"batch_wall_ms\":" << wall_ms[v] << ",\"gpu_ms\":[";
        for(size_t i=0;i<times[v].size();++i) {if(i) std::cout << ',';std::cout << times[v][i];} std::cout << "]}"; }
      std::cout << "]}";
      for(int v=0;v<2;++v) {check(p_clReleaseKernel(kernels[v]));check(p_clReleaseProgram(programs[v]));}
      for(auto buffer:buffers) check(p_clReleaseMemObject(buffer));
    }
    std::cout << "],\"build_ms\":["; for(size_t i=0;i<build_ms.size();++i){if(i)std::cout<<',';std::cout<<build_ms[i];}
    std::cout << "],\"all_exact\":" << (all_exact ? "true" : "false") << ",\"all_tolerance_passed\":" << (all_tolerance_passed ? "true" : "false") << "}\n";
    check(p_clReleaseCommandQueue(queue));check(p_clReleaseContext(context));dlclose(library);
    return all_tolerance_passed ? 0 : 1;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 2; }
}
