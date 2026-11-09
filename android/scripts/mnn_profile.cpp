// Shell-only diagnostic. No camera, CAN, control or vehicle interfaces.
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "core/Backend.hpp"
#include <cstring>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  try {
    std::printf("FOLD3_PROFILE_PID %d\n", getpid());
    std::fflush(stdout);
    std::unique_ptr<MNN::Interpreter> model(MNN::Interpreter::createFromFile(argv[1]));
    if (!model) throw std::runtime_error("model load failed");
    model->setSessionMode(MNN::Interpreter::Session_Debug);
    MNN::BackendConfig precision;
    precision.precision = MNN::BackendConfig::Precision_High;
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_OPENCL;
    config.backendConfig = &precision;
    config.numThread = MNN_GPU_TUNING_HEAVY | MNN_GPU_MEMORY_BUFFER;
    auto *session = model->createSession(config);
    if (!session) throw std::runtime_error("session creation failed");
    int backends[2] = {-1, -1};
    if (!model->getSessionInfo(session, MNN::Interpreter::BACKENDS, backends) || backends[0] != MNN_FORWARD_OPENCL)
      throw std::runtime_error("OpenCL unavailable; fallback rejected");
    for (const auto &[name, tensor] : model->getSessionInputAll(session)) {
      MNN::Tensor host(tensor, MNN::Tensor::CAFFE);
      std::memset(host.host<char>(), 0, host.size());
      if (!tensor->copyFromHostTensor(&host)) throw std::runtime_error("input upload failed");
    }
    bool fallback = false;
    // Audit once without exposing intermediate values. Event profiling drains
    // after submission, unlike tensor tracing which serializes each readback.
    auto code = model->runSessionWithCallBackInfo(session,
      [](const std::vector<MNN::Tensor *> &, const MNN::OperatorInfo *) { return true; },
      [&](const std::vector<MNN::Tensor *> &tensors, const MNN::OperatorInfo *op) {
        for (auto *tensor : tensors) {
          const auto *backend = model->getBackend(session, tensor);
          if (!backend || backend->type() != MNN_FORWARD_OPENCL) {
            std::fprintf(stderr, "CPU fallback rejected: %s\n", op->name().c_str());
            fallback = true;
          }
        }
        std::printf("FOLD3_ASSIGN %s:%s:%s\n", op->name().c_str(), op->type().c_str(), fallback ? "CPU" : "OpenCL");
        return !fallback;
      }, false);
    if (fallback || code != MNN::NO_ERROR) throw std::runtime_error("assignment audit failed");
    for (int run = 0; run < 3; ++run) {
      std::printf("FOLD3_PROFILE_BEGIN %d\n", run);
      if (model->runSession(session) != MNN::NO_ERROR) throw std::runtime_error("execution failed");
      // Complete execution and materialize all outputs, with no intermediate
      // taps. Zero inputs are diagnostic and cannot establish numerical parity.
      for (const auto &[name, tensor] : model->getSessionOutputAll(session)) {
        MNN::Tensor host(tensor, MNN::Tensor::CAFFE);
        if (!tensor->copyToHostTensor(&host)) throw std::runtime_error("output readback failed");
      }
      std::printf("FOLD3_PROFILE_END %d\n", run);
      std::fflush(stdout);
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  return 0;
}
