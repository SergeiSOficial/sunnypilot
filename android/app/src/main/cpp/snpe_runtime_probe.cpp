// Availability is diagnostic evidence only, never model/provider qualification.
#include <cstdio>
#include <dlfcn.h>
#include "DlSystem/DlEnums.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    std::fprintf(stderr, "SNPE loader: %s\n", dlerror());
    return 3;
  }
  using Available = int (*)(Snpe_Runtime_t);
  auto available = reinterpret_cast<Available>(dlsym(library, "Snpe_Util_IsRuntimeAvailable"));
  if (!available) { std::fprintf(stderr, "Missing SNPE availability API\n"); return 4; }
  // Do not request CPU or DSP: this experiment is GPU-only and never executes a model.
  const bool gpu = available(SNPE_RUNTIME_GPU_FLOAT16) != 0;
  std::printf("{\"runtime\":\"GPU_FLOAT16\",\"available\":%s,"
              "\"model_executed\":false,\"provider_assignment_verified\":false,"
              "\"qualified\":false,\"vehicle_tx_enabled\":false}\n", gpu ? "true" : "false");
  dlclose(library);
  return gpu ? 0 : 5;
}
