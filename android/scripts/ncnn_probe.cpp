// Shell-only model correctness probe; no Android services or CAN access.
#include <net.h>
#include <gpu.h>
#include <layer.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void save(const ncnn::Mat &value, const std::string &path) {
  if (value.empty() || value.elembits() != 32 || value.elempack != 1)
    throw std::runtime_error("Non-FP32 or packed diagnostic output");
  std::ofstream out(path, std::ios::binary);
  const size_t plane = size_t(value.w) * value.h * value.d;
  for (int batch = 0; batch < value.n; ++batch)
    for (int channel = 0; channel < value.c; ++channel)
      out.write(reinterpret_cast<const char *>(static_cast<const float *>(value.data) +
                batch * value.nstep + channel * value.cstep), plane * sizeof(float));
  if (!out) throw std::runtime_error("Output write failed");
}

int main(int argc, char **argv) {
  try {
    std::cout << "FOLD3_REFERENCE_FLOAT_GEMM " << ncnn::fold3_reference_float_gemm_version() << '\n';
    std::cout << "FOLD3_REFERENCE_LAYERNORM " << ncnn::fold3_reference_layernorm_version() << '\n';
    if (argc != 6) throw std::runtime_error("Usage: f3ncnnprobe directory trace-count trace-skip trace-stride all|weighted");
    const std::string trace_mode = argv[5];
    if (trace_mode != "all" && trace_mode != "weighted") throw std::runtime_error("Invalid trace mode");
    const std::string directory = argv[1];
    const int count = std::stoi(argv[2]);
    const int skip = std::stoi(argv[3]);
    const int stride = std::stoi(argv[4]);
    if (stride < 1 || stride > 64) throw std::runtime_error("Invalid trace stride");
    if (skip < 0 || skip > 4096) throw std::runtime_error("Invalid trace skip");
    if (count < 0 || count > 64) throw std::runtime_error("Invalid trace count");
    if (ncnn::get_gpu_count() < 1) throw std::runtime_error("No Vulkan GPU; fallback rejected");
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = false;
    net.opt.use_int8_inference = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 2;
    net.set_vulkan_device(0);
    if (net.load_param((directory + "/model.param").c_str()) || net.load_model((directory + "/model.bin").c_str()))
      throw std::runtime_error("Model load failed");
    if (!net.opt.use_vulkan_compute) throw std::runtime_error("Vulkan disabled");
    for (const auto *layer : net.layers()) {
      if (layer->type != "Input" && (!layer->support_vulkan || layer->featmask))
        throw std::runtime_error("CPU operation rejected: " + layer->name + ":" + layer->type);
    }
    auto extractor = net.create_extractor();
    // Tracing retains intermediates; a trace-free invocation remains separate.
    if (count) extractor.set_light_mode(false);
    std::ifstream shapes(directory + "/shapes.txt");
    for (const char *name : net.input_names()) {
      std::string expected;
      int rank, n, w, h = 1, d = 1, c = 1;
      if (!(shapes >> expected >> rank >> n >> w) || expected != name || rank < 1 || rank > 4)
        throw std::runtime_error("Input layout contract mismatch");
      if (rank >= 2) shapes >> h;
      if (rank == 4) shapes >> d;
      if (rank >= 3) shapes >> c;
      if (!shapes || n < 1 || w < 1 || h < 1 || d < 1 || c < 1)
        throw std::runtime_error("Invalid input dimensions");
      ncnn::Mat tensor;
      if (rank == 1) tensor.create(w, size_t(4), 1, n, nullptr);
      else if (rank == 2) tensor.create(w, h, 4u, 1, n);
      else if (rank == 3) tensor.create(w, h, c, 4u, 1, n);
      else tensor.create(w, h, d, c, 4u, 1, n);
      if (tensor.empty()) throw std::runtime_error("Input allocation failed");
      const size_t plane = size_t(w) * h * d;
      std::ifstream in(directory + "/" + name + ".bin", std::ios::binary);
      for (int batch = 0; batch < n; ++batch)
        for (int channel = 0; channel < c; ++channel)
          in.read(reinterpret_cast<char *>(static_cast<float *>(tensor.data) + batch * tensor.nstep + channel * tensor.cstep),
                  plane * sizeof(float));
      if (!in || in.peek() != EOF) throw std::runtime_error("Input payload contract mismatch");
      if (extractor.input(name, tensor)) throw std::runtime_error("Input rejected");
    }
    int output_count;
    if (!(shapes >> output_count) || output_count < 1 || output_count > 8)
      throw std::runtime_error("Output contract mismatch");
    const auto start = std::chrono::steady_clock::now();
    std::vector<ncnn::Mat> outputs(output_count);
    for (int i = 0; i < output_count; ++i)
      if (extractor.extract(("out" + std::to_string(i)).c_str(), outputs[i]))
        throw std::runtime_error("Vulkan execution failed");
    const auto end = std::chrono::steady_clock::now();
    for (int i = 0; i < output_count; ++i)
      save(outputs[i], directory + "/out" + std::to_string(i) + ".bin");
    std::cout << "FOLD3_OUTPUT " << output_count << " "
              << std::chrono::duration<double, std::milli>(end - start).count() << "\n";
    int captured = 0;
    for (int index = skip; index < int(net.blobs().size()) && captured < count; index += stride) {
      const auto &blob = net.blobs()[index];
      const auto *layer = net.layers()[blob.producer];
      if (trace_mode == "weighted" && layer->type != "Convolution" &&
          layer->type != "ConvolutionDepthWise" && layer->type != "InnerProduct") continue;
      ++captured;
      ncnn::Mat tensor;
      if (extractor.extract(index, tensor)) throw std::runtime_error("Trace extraction failed");
      save(tensor, directory + "/trace-" + std::to_string(index) + ".bin");
      std::cout << "FOLD3_TENSOR " << index << " " << blob.name << " " << layer->name << " " << layer->type
                << " " << tensor.w << " " << tensor.h << " " << tensor.d << " " << tensor.c << " " << tensor.n << "\n";
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
