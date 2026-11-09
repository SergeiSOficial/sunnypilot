// Diagnostic only: refuses partial GPU delegation before executing a model.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert_opaque_options.h"

#define CHECK(call) do { auto status = (call); if (status != kLiteRtStatusOk) { \
  fprintf(stderr, "%s status=%d\n", #call, int(status)); return 2; } } while (0)
struct Resources {
  LiteRtEnvironment env = nullptr; LiteRtModel model = nullptr;
  LiteRtOptions options = nullptr; LiteRtCompiledModel compiled = nullptr;
  LiteRtTensorBuffer input = nullptr, output = nullptr;
  ~Resources() {
    if (output) LiteRtDestroyTensorBuffer(output);
    if (input) LiteRtDestroyTensorBuffer(input);
    if (compiled) LiteRtDestroyCompiledModel(compiled);
    if (options) LiteRtDestroyOptions(options);
    if (model) LiteRtDestroyModel(model);
    if (env) LiteRtDestroyEnvironment(env);
  }
};
int main(int argc, char** argv) {
  const bool compile_only = argc == 3 && std::strcmp(argv[2], "--compile-only") == 0;
  if (argc != 4 && !compile_only) return 64;
  Resources r;
  CHECK(LiteRtCreateEnvironment(0, nullptr, &r.env));
  CHECK(LiteRtCreateModelFromFile(r.env, argv[1], &r.model));
  CHECK(LiteRtCreateOptions(&r.options));
  CHECK(LiteRtSetOptionsHardwareAccelerators(r.options, kLiteRtHwAcceleratorGpu));
  const char* precision = std::getenv("FOLD3_LITERT_PRECISION");
  if (precision && *precision) {
    // LiteRT 2.3.0 serializes GPU opaque options as TOML; keep this
    // diagnostic pinned to that version and reject unrecognized modes.
    const char* option = nullptr;
    if (std::strcmp(precision, "fp32") == 0) option = "precision = 2\n";
    else if (std::strcmp(precision, "fp16-fp32-accum") == 0) option = "precision = 3\n";
    else { fprintf(stderr, "Unsupported GPU precision\n"); return 64; }
    char* payload = strdup(option);
    if (!payload) return 2;
    LiteRtOpaqueOptions opaque = nullptr;
    auto status = LiteRtCreateOpaqueOptions("gpu_options", payload, std::free, &opaque);
    if (status != kLiteRtStatusOk) { std::free(payload); return 2; }
    status = LiteRtAddOpaqueOptions(r.options, opaque);
    if (status != kLiteRtStatusOk) { LiteRtDestroyOpaqueOptions(opaque); return 2; }
  }
  CHECK(LiteRtCreateCompiledModel(r.env, r.model, r.options, &r.compiled));
  bool fully = false;
  CHECK(LiteRtCompiledModelIsNonCpuFullyAccelerated(r.compiled, &fully));
  printf("{\"gpu_only\":%s,\"precision\":\"%s\",\"qualified\":false,\"vehicle_tx_enabled\":false}\n", fully ? "true" : "false", precision && *precision ? precision : "default");
  if (!fully) { fprintf(stderr, "Rejected CPU partition\n"); return 3; }
  if (compile_only) return 0;
  LiteRtSignature signature; CHECK(LiteRtGetModelSignature(r.model, 0, &signature));
  LiteRtTensor in_tensor, out_tensor;
  CHECK(LiteRtGetSignatureInputTensorByIndex(signature, 0, &in_tensor));
  CHECK(LiteRtGetSignatureOutputTensorByIndex(signature, 0, &out_tensor));
  LiteRtRankedTensorType in_type, out_type;
  CHECK(LiteRtGetRankedTensorType(in_tensor, &in_type));
  CHECK(LiteRtGetRankedTensorType(out_tensor, &out_type));
  LiteRtTensorBufferRequirements in_req, out_req;
  CHECK(LiteRtGetCompiledModelInputBufferRequirements(r.compiled, 0, 0, &in_req));
  CHECK(LiteRtGetCompiledModelOutputBufferRequirements(r.compiled, 0, 0, &out_req));
  CHECK(LiteRtCreateManagedTensorBufferFromRequirements(r.env, &in_type, in_req, &r.input));
  CHECK(LiteRtCreateManagedTensorBufferFromRequirements(r.env, &out_type, out_req, &r.output));
  std::ifstream file(argv[2], std::ios::binary);
  if (!file) return 4;
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
  size_t input_size; CHECK(LiteRtGetTensorBufferSize(r.input, &input_size));
  size_t input_count = 1;
  for (int i = 0; i < in_type.layout.rank; ++i) input_count *= in_type.layout.dimensions[i];
  if (in_type.element_type != kLiteRtElementTypeFloat16 && in_type.element_type != kLiteRtElementTypeFloat32) return 5;
  size_t expected_bytes = input_count * (in_type.element_type == kLiteRtElementTypeFloat16 ? 2 : 4);
  if (bytes.size() != expected_bytes || bytes.size() > input_size) return 5;
  void* data; CHECK(LiteRtLockTensorBuffer(r.input, &data, kLiteRtTensorBufferLockModeWrite));
  __builtin_memcpy(data, bytes.data(), bytes.size()); CHECK(LiteRtUnlockTensorBuffer(r.input));
  CHECK(LiteRtRunCompiledModel(r.compiled, 0, 1, &r.input, 1, &r.output));
  CHECK(LiteRtLockTensorBuffer(r.output, &data, kLiteRtTensorBufferLockModeRead));
  // These probes preserve shape and dtype for input/output except Conv's shape.
  size_t count = 1;
  for (int i = 0; i < out_type.layout.rank; ++i) count *= out_type.layout.dimensions[i];
  size_t width = out_type.element_type == kLiteRtElementTypeFloat16 ? 2 : 4;
  std::ofstream output(argv[3], std::ios::binary); output.write(static_cast<char*>(data), count * width);
  bool written = bool(output); CHECK(LiteRtUnlockTensorBuffer(r.output));
  return written ? 0 : 6;
}
