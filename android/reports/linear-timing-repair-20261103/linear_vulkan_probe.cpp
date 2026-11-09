// Diagnostic only: no cameras, CAN, application runtime or CPU fallback.
#include <net.h>
#include <gpu.h>
#include <layer.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc, char** argv) {
  try {
    if(argc != 2 && argc != 3) throw std::runtime_error("Expected diagnostic directory");
    const std::string path=argv[1];
    const bool cpu_rejection = argc == 3 && std::string(argv[2]) == "cpu-rejection";
    if(!cpu_rejection && ncnn::get_gpu_count() < 1) throw std::runtime_error("No Vulkan device; CPU fallback rejected");
    ncnn::Net net;
    net.opt.use_vulkan_compute=!cpu_rejection;
    net.opt.use_fp16_storage=false;
    net.opt.use_fp16_packed=false;
    net.opt.use_fp16_arithmetic=false;
    net.opt.use_bf16_storage=false;
    net.opt.use_int8_inference=false;
    net.opt.use_packing_layout=true; // Exercise dispatch from an automatically packed input.
    net.opt.num_threads=1;
    if(!cpu_rejection) net.set_vulkan_device(0);
    if(net.load_param((path+"/model.param").c_str()) || net.load_model((path+"/model.bin").c_str()))
      throw std::runtime_error("Model/pipeline load failed");
    if(!net.opt.use_vulkan_compute) throw std::runtime_error("Vulkan disabled");
    for(const auto* layer: net.layers())
      if(layer->type != "Input" && (!layer->support_vulkan || layer->featmask))
        throw std::runtime_error("CPU fallback rejected: "+layer->name);
    std::vector<float> data(1280);
    std::ifstream input(path+"/input.bin",std::ios::binary);
    input.read(reinterpret_cast<char*>(data.data()),data.size()*sizeof(float));
    if(!input || input.peek()!=EOF) throw std::runtime_error("Input contract mismatch");
    ncnn::Mat tensor(1280,size_t(4),1,1);
    std::memcpy(tensor.data,data.data(),data.size()*sizeof(float));
    auto extractor=net.create_extractor();
    if(extractor.input("in",tensor)) throw std::runtime_error("Input rejected");
    ncnn::Mat output;
    if(extractor.extract("out",output)) throw std::runtime_error("Vulkan execution failed");
    if(output.empty() || output.total()!=512 || output.elembits()!=32 || output.elempack!=1)
      throw std::runtime_error("Output contract mismatch");
    std::ofstream out(path+"/output.bin",std::ios::binary);
    out.write(static_cast<const char*>(output.data),512*sizeof(float));
    if(!out) throw std::runtime_error("Output write failed");
    std::cout<<"VULKAN_LINEAR_OUTPUT 512\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<"\n";
    return 1;
  }
}
