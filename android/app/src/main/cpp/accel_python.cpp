#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "core/Backend.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include <net.h>
#include <gpu.h>
#include <layer.h>
#include <command.h>
#ifdef FOLD3_NCNN_EXPLICIT_CASTS
#include <benchmark.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
#ifdef FOLD3_NCNN_EXPLICIT_CASTS
constexpr bool ncnn_explicit_casts = true;
#else
constexpr bool ncnn_explicit_casts = false;
#endif
int reference_layernorm_version() {
#ifdef FOLD3_NCNN_EXPLICIT_CASTS
  return ncnn::fold3_reference_layernorm_version();
#else
  return 0;
#endif
}
int reference_float_gemm_version() {
#ifdef FOLD3_NCNN_EXPLICIT_CASTS
  return ncnn::fold3_reference_float_gemm_version();
#else
  return 0;
#endif
}
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
struct ReleasedGIL { PyThreadState *state = PyEval_SaveThread(); ~ReleasedGIL() { PyEval_RestoreThread(state); } };
struct Buffer {
  Py_buffer value{};
  ~Buffer() { if (value.obj) PyBuffer_Release(&value); }
};
size_t logical_elements(const ncnn::Mat &tensor) {
  return static_cast<size_t>(tensor.w) * tensor.h * tensor.d * tensor.c * tensor.n;
}
void copy_to_ncnn(const float *dense, ncnn::Mat &tensor) {
  const size_t plane = static_cast<size_t>(tensor.w) * tensor.h * tensor.d;
  for (int batch = 0; batch < tensor.n; ++batch)
    for (int channel = 0; channel < tensor.c; ++channel)
      std::memcpy(static_cast<float *>(tensor.data) + batch * tensor.nstep + channel * tensor.cstep,
                  dense + (static_cast<size_t>(batch) * tensor.c + channel) * plane, plane * sizeof(float));
}
void copy_from_ncnn(const ncnn::Mat &tensor, float *dense) {
  const size_t plane = static_cast<size_t>(tensor.w) * tensor.h * tensor.d;
  for (int batch = 0; batch < tensor.n; ++batch)
    for (int channel = 0; channel < tensor.c; ++channel)
      std::memcpy(dense + (static_cast<size_t>(batch) * tensor.c + channel) * plane,
                  static_cast<const float *>(tensor.data) + batch * tensor.nstep + channel * tensor.cstep, plane * sizeof(float));
}
PyObject *check_tensor_copy(PyObject *, PyObject *) {
  // Unaligned vectors, channel padding and 4 KiB batch strides must round-trip
  // without treating padding as values or mixing independent camera batches.
  for (const auto &shape : std::vector<std::vector<int>>{{19, 1, 1, 1, 1}, {553, 1, 1, 1, 2}, {9, 7, 2, 3, 2}}) {
    ncnn::Mat tensor(shape[0], shape[1], shape[2], shape[3], size_t(4), 1, shape[4]);
    std::vector<float> input(logical_elements(tensor)), output(input.size());
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i) - 256.5f;
    copy_to_ncnn(input.data(), tensor);
    copy_from_ncnn(tensor, output.data());
    if (input != output) { PyErr_SetString(PyExc_AssertionError, "ncnn padded tensor copy failed"); return nullptr; }
  }
  Py_RETURN_TRUE;
}
PyObject *check_profile_clock(PyObject *, PyObject *) {
#ifdef FOLD3_NCNN_EXPLICIT_CASTS
  auto &profile = ncnn::fold3_vulkan_profile();
  const auto previous = profile;
  profile = {};
  profile.enabled = true;
  profile.timestamp_bits = 8;
  ncnn::Layer layer;
  layer.name = "wrap-test"; layer.type = "diagnostic";
  ncnn::fold3_record_vulkan_profile(&layer, 250, 6, 1000.f);
  bool passed = profile.valid && profile.records.size() == 1 && profile.records[0].milliseconds == .012;
  profile.timestamp_bits = 0;
  ncnn::fold3_record_vulkan_profile(&layer, 1, 2, 1000.f);
  passed = passed && !profile.valid;
  profile = previous;
  if (!passed) { PyErr_SetString(PyExc_AssertionError, "Vulkan timestamp wrap validation failed"); return nullptr; }
  Py_RETURN_TRUE;
#else
  PyErr_SetString(PyExc_RuntimeError, "Patched Vulkan profiling required"); return nullptr;
#endif
}
struct Session {
  std::string engine;
  std::string memory_mode;
  struct Trace {
    std::string op, type, tensor, backend, layout, dtype;
    std::vector<int> shape;
    std::vector<char> data;
  };
  std::vector<Trace> traces;
  std::vector<std::string> tensor_names;
  int diagnostic_dispatch_limit = 0;
  int trace_limit = 0;
  int trace_skip = 0, trace_seen = 0;
  size_t trace_bytes = 0;
  std::unique_ptr<MNN::Interpreter> mnn;
  MNN::Session *session = nullptr;
  std::map<std::string, std::unique_ptr<MNN::Tensor>> input_hosts, output_hosts;
  ncnn::Net net;
  std::map<std::string, ncnn::Mat> ncnn_inputs;
  ncnn::VkAllocator *blob_allocator = nullptr, *staging_allocator = nullptr;
  bool profiling = false, profile_valid = false;
  uint32_t timestamp_bits = 0;
  double preparation_ms = 0, record_ms = 0, submit_wait_ms = 0, output_copy_ms = 0;
#ifdef FOLD3_NCNN_EXPLICIT_CASTS
  std::vector<ncnn::Fold3VulkanTiming> gpu_timings;
  uint32_t internal_submissions = 0, dispatch_limit = 0, rough_score = 0;
  uint64_t submitted_groups = 0;
  double internal_submit_wait_ms = 0.;
#endif
  std::vector<std::string> assignments;
  bool audited = false;
  double upload_ms = 0, compute_ms = 0, readback_ms = 0;
  explicit Session(const std::string &path, const std::string &backend, const std::string &memory, int limit, int skip, bool profile, int dispatch)
      : engine(backend), memory_mode(memory), trace_limit(limit), trace_skip(skip), profiling(profile) {
    diagnostic_dispatch_limit = dispatch;
    const auto basename = path.substr(path.find_last_of("/\\") + 1);
    if (dispatch && (engine != "ncnn-vulkan" || basename != "driving_supercombo.ncnn.param" ||
        (dispatch != 131072 && dispatch != 262144 && dispatch != 524288)))
      throw std::invalid_argument("Dispatch override requires an original road Vulkan graph and supported limit");
    if (profiling && (engine != "ncnn-vulkan" || limit)) throw std::invalid_argument("GPU profiling requires untapped Vulkan execution");
#ifndef FOLD3_NCNN_EXPLICIT_CASTS
    if (profiling || dispatch) throw std::runtime_error("Patched Vulkan diagnostics unavailable");
#endif
    if (engine == "mnn-opencl") {
      mnn.reset(MNN::Interpreter::createFromFile(path.c_str()));
      if (!mnn) throw std::runtime_error("MNN model load failed");
      mnn->setSessionMode(MNN::Interpreter::Session_Debug);
      MNN::BackendConfig precision;
      precision.precision = MNN::BackendConfig::Precision_High;
      MNN::ScheduleConfig config;
      config.type = MNN_FORWARD_OPENCL;
      config.backendConfig = &precision;
      if (memory != "image" && memory != "buffer") throw std::invalid_argument("Unknown MNN memory mode");
      config.numThread = MNN_GPU_TUNING_HEAVY | (memory == "buffer" ? MNN_GPU_MEMORY_BUFFER : MNN_GPU_MEMORY_IMAGE);
      if (trace_limit) {
        std::ifstream source(path, std::ios::binary);
        std::vector<char> bytes{std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>()};
        flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size());
        if (!MNN::VerifyNetBuffer(verifier)) throw std::runtime_error("MNN trace model verification failed");
        const auto *net_info = MNN::GetNet(bytes.data());
        if (net_info->tensorName()) for (auto name : *net_info->tensorName()) tensor_names.push_back(name->str());
      }
      session = mnn->createSession(config);
      if (!session) throw std::runtime_error("MNN OpenCL session initialization failed");
      int resize_status = -1;
      if (!mnn->getSessionInfo(session, MNN::Interpreter::RESIZE_STATUS, &resize_status) || resize_status != 0)
        throw std::runtime_error("MNN OpenCL session is not ready; resize status " + std::to_string(resize_status));
      int backends[2] = {-1, -1};
      if (!mnn->getSessionInfo(session, MNN::Interpreter::BACKENDS, backends) || backends[0] != MNN_FORWARD_OPENCL)
        throw std::runtime_error("MNN OpenCL unavailable; CPU fallback rejected");
      for (const auto &[name, tensor] : mnn->getSessionInputAll(session))
        input_hosts[name] = std::make_unique<MNN::Tensor>(tensor, MNN::Tensor::CAFFE);
      for (const auto &[name, tensor] : mnn->getSessionOutputAll(session))
        output_hosts[name] = std::make_unique<MNN::Tensor>(tensor, MNN::Tensor::CAFFE);
    } else if (engine == "ncnn-vulkan") {
      if (ncnn::get_gpu_count() <= 0) throw std::runtime_error("No Vulkan compute device; CPU fallback rejected");
      net.opt.use_vulkan_compute = true;
      net.opt.use_fp16_packed = false;
      net.opt.use_fp16_storage = false;
      net.opt.use_fp16_arithmetic = false;
      net.opt.use_bf16_storage = false;
      net.opt.use_int8_inference = false;
      net.opt.use_packing_layout = false;
      net.opt.num_threads = 2;
#ifdef FOLD3_NCNN_EXPLICIT_CASTS
      net.opt.fold3_diagnostic_dispatch_limit = dispatch;
#endif
      if (dispatch && (ncnn::get_gpu_info(0).vendor_id() != 0x5143 ||
                       std::string(ncnn::get_gpu_info(0).device_name()) != "Adreno (TM) 660"))
        throw std::runtime_error("Dispatch experiment requires the measured Fold 3 GPU");
      net.set_vulkan_device(0);
      if (net.load_param(path.c_str()) != 0) throw std::runtime_error("ncnn graph load failed");
      const std::string weights = path.substr(0, path.rfind(".param")) + ".bin";
      if (net.load_model(weights.c_str()) != 0) throw std::runtime_error("ncnn weight load failed");
      if (!net.opt.use_vulkan_compute) throw std::runtime_error("ncnn disabled Vulkan; CPU fallback rejected");
      // Inspect actual assignments after parameter, weight and pipeline loading.
      // Input is a source with no computation.
      std::string unsupported;
      for (const auto *layer : net.layers()) {
        assignments.push_back(layer->name + ":" + layer->type + (layer->support_vulkan ? ":Vulkan" : ":CPU"));
        if (layer->type != "Input" && (!layer->support_vulkan || layer->featmask))
          unsupported += layer->name + ":" + layer->type + ";";
      }
      if (!unsupported.empty()) throw std::runtime_error("ncnn CPU operations rejected: " + unsupported);
      // Retain session-owned allocators across extractions. All queued work is
      // fenced before returning; no input can be overwritten while in use.
      if (profiling) {
        std::unique_ptr<void, int(*)(void*)> loader(dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL), dlclose);
        if (!loader) throw std::runtime_error("Vulkan loader unavailable");
        auto queue_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
          dlsym(loader.get(), "vkGetPhysicalDeviceQueueFamilyProperties"));
        if (!queue_properties) throw std::runtime_error("Vulkan queue capability query unavailable");
        uint32_t count = 0;
        queue_properties(net.vulkan_device()->info.physical_device(), &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        queue_properties(net.vulkan_device()->info.physical_device(), &count, queues.data());
        timestamp_bits = queues.at(net.vulkan_device()->info.compute_queue_family_index()).timestampValidBits;
        if (!timestamp_bits || timestamp_bits > 64) throw std::runtime_error("Vulkan timestamp queries unavailable");
      }
      blob_allocator = net.vulkan_device()->acquire_blob_allocator();
      staging_allocator = net.vulkan_device()->acquire_staging_allocator();
      net.opt.blob_vkallocator = blob_allocator;
      net.opt.workspace_vkallocator = blob_allocator;
      net.opt.staging_vkallocator = staging_allocator;
      audited = true;
    } else throw std::invalid_argument("Unsupported accelerator");
  }
  ~Session() {
    net.clear();
    if (blob_allocator) net.vulkan_device()->reclaim_blob_allocator(blob_allocator);
    if (staging_allocator) net.vulkan_device()->reclaim_staging_allocator(staging_allocator);
  }
};
Session *get(PyObject *capsule) { return static_cast<Session *>(PyCapsule_GetPointer(capsule, "fold3.accel")); }
void destroy(PyObject *capsule) {
  auto *session = get(capsule);
  // Provider teardown can wait for GPU work. Do not block Chaquopy finalizers
  // (or another Python thread) behind the interpreter lock while it does so.
  ReleasedGIL release;
  delete session;
}
PyObject *create(PyObject *, PyObject *args) {
  const char *path, *engine;
  const char *memory = "image";
  int trace_limit = 0;
  int trace_skip = 0;
  int profiling = 0, dispatch = 0;
  if (!PyArg_ParseTuple(args, "ss|siipi", &path, &engine, &memory, &trace_limit, &trace_skip, &profiling, &dispatch)) return nullptr;
  if (trace_limit < 0 || trace_limit > 2048 || trace_skip < 0 || trace_skip > 4096) { PyErr_SetString(PyExc_ValueError, "Invalid diagnostic trace window"); return nullptr; }
  try {
    std::unique_ptr<Session> session;
    { ReleasedGIL release; session = std::make_unique<Session>(path, engine, memory, trace_limit, trace_skip, profiling, dispatch); }
    return PyCapsule_New(session.release(), "fold3.accel", destroy);
  } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return nullptr; }
}
const char *dtype(const MNN::Tensor *tensor) {
  const auto type = tensor->getType();
  if (type.code == halide_type_float && type.bits == 32) return "float32";
  if (type.code == halide_type_float && type.bits == 16) return "float16";
  if (type.code == halide_type_uint && type.bits == 8) return "uint8";
  if (type.code == halide_type_int && type.bits == 32) return "int32";
  throw std::runtime_error("Unsupported MNN boundary type");
}
PyObject *tensor_info(const MNN::Tensor *tensor) {
  const auto shape = tensor->shape();
  PyObject *dims = PyTuple_New(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) PyTuple_SET_ITEM(dims, i, PyLong_FromLong(shape[i]));
  return Py_BuildValue("{s:N,s:s}", "shape", dims, "dtype", dtype(tensor));
}
PyObject *describe(PyObject *, PyObject *capsule) {
  auto *s = get(capsule);
  if (!s) return nullptr;
  try {
    PyObject *inputs = PyDict_New(), *outputs = PyDict_New(), *assignments = PyList_New(0);
    for (const auto &item : s->input_hosts) {
      PyObject *info = tensor_info(item.second.get()); PyDict_SetItemString(inputs, item.first.c_str(), info); Py_DECREF(info);
    }
    for (const auto &item : s->output_hosts) {
      PyObject *info = tensor_info(item.second.get()); PyDict_SetItemString(outputs, item.first.c_str(), info); Py_DECREF(info);
    }
    for (const auto &item : s->assignments) { PyObject *value = PyUnicode_FromString(item.c_str()); PyList_Append(assignments, value); Py_DECREF(value); }
    return Py_BuildValue("{s:s,s:s,s:i,s:i,s:N,s:N,s:N,s:O,s:O,s:O,s:O,s:i,s:i,s:d,s:d,s:d,s:O,s:i}", "backend", s->engine.c_str(),
                         "memory_mode", s->memory_mode.c_str(), "trace_limit", s->trace_limit,
                         "trace_skip", s->trace_skip,
                         "inputs", inputs, "outputs", outputs, "assignments", assignments,
                         "no_cpu_fallback_verified", s->audited ? Py_True : Py_False,
                         "explicit_half_rounding", ncnn_explicit_casts ? Py_True : Py_False,
                         "ordered_half_gemv", ncnn_explicit_casts ? Py_True : Py_False,
                         "ordered_half_dot2", ncnn_explicit_casts ? Py_True : Py_False,
                         "reference_layernorm", reference_layernorm_version(),
                         "reference_float_gemm", reference_float_gemm_version(),
                         "upload_ms", s->upload_ms, "compute_ms", s->compute_ms, "readback_ms", s->readback_ms,
                         "profiling_enabled", s->profiling ? Py_True : Py_False,
                         "diagnostic_dispatch_limit", s->diagnostic_dispatch_limit);
  } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return nullptr; }
}
PyObject *run(PyObject *, PyObject *args) {
  PyObject *capsule, *inputs;
  if (!PyArg_ParseTuple(args, "OO!", &capsule, &PyDict_Type, &inputs)) return nullptr;
  auto *s = get(capsule);
  if (!s) return nullptr;
  try {
    PyObject *result = PyDict_New();
    try {
      if (s->engine == "mnn-opencl") {
        if (PyDict_Size(inputs) != static_cast<Py_ssize_t>(s->input_hosts.size())) throw std::invalid_argument("MNN input set mismatch");
        auto started = Clock::now();
        for (auto &[name, host] : s->input_hosts) {
          PyObject *value = PyDict_GetItemString(inputs, name.c_str());
          Buffer buffer;
          if (!value || PyObject_GetBuffer(value, &buffer.value, PyBUF_FORMAT | PyBUF_STRIDES) != 0) throw std::invalid_argument("MNN input buffer missing");
          const auto &b = buffer.value;
          const auto shape = host->shape();
          bool matches = b.ndim == static_cast<int>(shape.size()) && b.len == host->size() && PyBuffer_IsContiguous(&b, 'C');
          const std::string expected = dtype(host.get());
          const char *format = expected == "float32" ? "f" : expected == "float16" ? "e" : expected == "uint8" ? "B" : "i";
          matches = matches && std::strcmp(format, b.format) == 0;
          for (int i = 0; matches && i < b.ndim; ++i) matches = b.shape[i] == shape[i];
          if (!matches) throw std::invalid_argument("MNN input shape/type mismatch: " + name);
          std::memcpy(host->host<void>(), b.buf, b.len);
          ReleasedGIL release;
          if (!s->mnn->getSessionInput(s->session, name.c_str())->copyFromHostTensor(host.get())) throw std::runtime_error("MNN input upload failed");
        }
        s->upload_ms = ms(started);
        started = Clock::now();
        MNN::ErrorCode code;
        std::string fallback, trace_error;
        {
          ReleasedGIL release;
          if (!s->audited) {
            s->assignments.clear();
            code = s->mnn->runSessionWithCallBackInfo(s->session,
              [](const std::vector<MNN::Tensor *> &, const MNN::OperatorInfo *) { return true; },
              [&](const std::vector<MNN::Tensor *> &tensors, const MNN::OperatorInfo *op) {
                bool gpu = true;
                for (const auto *tensor : tensors) {
                  const auto *backend = s->mnn->getBackend(s->session, tensor);
                  gpu = gpu && backend && backend->type() == MNN_FORWARD_OPENCL;
                }
                s->assignments.push_back(op->name() + ":" + op->type() + (gpu ? ":OpenCL" : ":CPU"));
                if (s->trace_limit && s->traces.size() < static_cast<size_t>(s->trace_limit)) {
                  try {
                  for (const auto *tensor : tensors) {
                    if (s->trace_seen++ < s->trace_skip) continue;
                    if (s->traces.size() >= static_cast<size_t>(s->trace_limit)) break;
                    const int index = MNN::TensorUtils::getDescribe(tensor)->index;
                    MNN::Tensor host(tensor, MNN::Tensor::CAFFE);
                    if (host.size() < 0 || s->trace_bytes + host.size() > 128u * 1024u * 1024u)
                      throw std::runtime_error("MNN diagnostic trace exceeds owned buffer limit");
                    if (!tensor->copyToHostTensor(&host)) throw std::runtime_error("MNN trace readback failed");
                    const auto type = host.getType();
                    std::string host_dtype = type.code == halide_type_float ? "float" : type.code == halide_type_uint ? "uint" : "int";
                    host_dtype += std::to_string(type.bits);
                    Session::Trace record{op->name(), op->type(), index >= 0 && index < static_cast<int>(s->tensor_names.size()) ? s->tensor_names[index] : "",
                                 gpu ? "OpenCL" : "CPU", "NCHW", host_dtype, host.shape(), {}};
                    record.data.assign(host.host<char>(), host.host<char>() + host.size());
                    s->trace_bytes += record.data.size();
                    s->traces.push_back(std::move(record));
                  }
                  } catch (const std::exception &error) {
                    // Returning false lets MNN leave its execute scope normally.
                    // Throwing through its callback skips backend cleanup.
                    trace_error = error.what();
                    return false;
                  }
                }
                if (!gpu) fallback += op->name() + ":" + op->type() + ";";
                return gpu;
              }, true);
          } else code = s->mnn->runSession(s->session);
          if (!trace_error.empty()) throw std::runtime_error(trace_error);
          if (!fallback.empty()) throw std::runtime_error("MNN CPU operation rejected: " + fallback);
          if (code != MNN::NO_ERROR) throw std::runtime_error("MNN execution error " + std::to_string(code));
        }
        s->compute_ms = ms(started);
        s->audited = true;
        started = Clock::now();
        for (const auto &[name, host] : s->output_hosts) {
          {
            ReleasedGIL release;
            if (!s->mnn->getSessionOutput(s->session, name.c_str())->copyToHostTensor(host.get())) throw std::runtime_error("MNN output readback failed");
          }
          const auto shape = host->shape();
          PyObject *dims = PyTuple_New(shape.size());
          for (size_t i = 0; i < shape.size(); ++i) PyTuple_SET_ITEM(dims, i, PyLong_FromLong(shape[i]));
          PyObject *entry = Py_BuildValue("(NsN)", dims, dtype(host.get()), PyBytes_FromStringAndSize(host->host<char>(), host->size()));
          PyDict_SetItemString(result, name.c_str(), entry); Py_DECREF(entry);
        }
        s->readback_ms = ms(started);
      } else {
#ifdef FOLD3_NCNN_EXPLICIT_CASTS
        struct ProfileScope {
          Session *session;
          explicit ProfileScope(Session *s) : session(s) {
            auto &p = ncnn::fold3_vulkan_profile();
            p.internal_submissions = 0; p.submitted_groups = 0; p.internal_submit_wait_ms = 0.;
            p.dispatch_limit = 0; p.rough_score = 0;
            p.records.clear(); p.valid = true; p.timestamp_bits = s->timestamp_bits; p.enabled = s->profiling;
          }
          ~ProfileScope() {
            auto &p = ncnn::fold3_vulkan_profile();
            session->profile_valid = p.enabled && p.valid && !p.records.empty();
            session->internal_submissions = p.internal_submissions; session->submitted_groups = p.submitted_groups;
            session->internal_submit_wait_ms = p.internal_submit_wait_ms; session->dispatch_limit = p.dispatch_limit;
            session->rough_score = p.rough_score;
            session->gpu_timings.swap(p.records); p.enabled = false;
          }
        } profile_scope(s);
#endif
        auto extractor = s->net.create_extractor();
        auto started = Clock::now();
        PyObject *key, *value; Py_ssize_t position = 0;
        while (PyDict_Next(inputs, &position, &key, &value)) {
          Buffer buffer;
          if (!PyTuple_Check(value) || PyTuple_Size(value) != 2 || !PyBool_Check(PyTuple_GetItem(value, 1)))
            throw std::invalid_argument("ncnn input requires explicit converter batch metadata");
          const bool batched = PyTuple_GetItem(value, 1) == Py_True;
          if (PyObject_GetBuffer(PyTuple_GetItem(value, 0), &buffer.value, PyBUF_FORMAT | PyBUF_STRIDES) != 0) throw std::invalid_argument("ncnn input missing");
          const auto &b = buffer.value;
          if (std::strcmp(b.format, "f") || !PyBuffer_IsContiguous(&b, 'C') || b.ndim < 1 || b.ndim > 5)
            throw std::invalid_argument("ncnn adapter requires contiguous FP32 rank 1..5");
          ncnn::Mat &tensor = s->ncnn_inputs[PyUnicode_AsUTF8(key)];
          const int rank = b.ndim - (batched ? 1 : 0);
          if (rank < 1 || rank > 4) throw std::invalid_argument("ncnn unrepresentable input rank/batch");
          const auto *shape = b.shape + (batched ? 1 : 0);
          const int n = batched ? b.shape[0] : 1;
          if (rank == 1) tensor.create(static_cast<int>(shape[0]), size_t(4), 1, n, nullptr);
          else if (rank == 2) tensor.create(shape[1], shape[0], 4u, 1, n);
          else if (rank == 3) tensor.create(shape[2], shape[1], shape[0], 4u, 1, n);
          else tensor.create(shape[3], shape[2], shape[1], shape[0], 4u, 1, n);
          if (tensor.empty() || logical_elements(tensor) * sizeof(float) != static_cast<size_t>(b.len)) throw std::runtime_error("ncnn input allocation/packing mismatch");
          copy_to_ncnn(static_cast<const float *>(b.buf), tensor);
          if (extractor.input(PyUnicode_AsUTF8(key), tensor) != 0) throw std::runtime_error("ncnn input name rejected");
        }
        s->preparation_ms = s->upload_ms = ms(started);
        started = Clock::now();
        std::map<std::string, ncnn::VkMat> gpu_outputs;
        std::map<std::string, ncnn::Mat> cpu_outputs;
        {
          ReleasedGIL release;
          ncnn::VkCompute cmd(s->net.vulkan_device());
#ifdef FOLD3_NCNN_EXPLICIT_CASTS
          if (s->profiling && cmd.create_query_pool(s->net.layers().size() * 2) != 0)
            throw std::runtime_error("Vulkan profile query allocation failed");
#endif
          const auto& graph_inputs = s->net.input_names();
          for (const auto *name : s->net.output_names()) {
            if (std::any_of(graph_inputs.begin(), graph_inputs.end(),
                            [name](const char *input) { return std::strcmp(name, input) == 0; })) continue;
            auto &output = gpu_outputs[name];
            const int code = extractor.extract(name, output, cmd);
            if (code || output.empty())
              throw std::runtime_error("ncnn Vulkan output extraction failed: " + std::string(name) + " code=" + std::to_string(code));
          }
          for (auto &[name, gpu] : gpu_outputs) cmd.record_download(gpu, cpu_outputs[name], s->net.opt);
          s->record_ms = ms(started);
          const auto wait_started = Clock::now();
          if (cmd.submit_and_wait() != 0) throw std::runtime_error("ncnn Vulkan submission failed");
          s->submit_wait_ms = ms(wait_started);
#ifdef FOLD3_NCNN_EXPLICIT_CASTS
          if (s->profiling) {
            std::vector<uint64_t> queries(s->net.layers().size() * 2);
            if (cmd.get_query_pool_results(0, queries.size(), queries) != 0) ncnn::fold3_vulkan_profile().valid = false;
            for (size_t i = 0; i < s->net.layers().size(); ++i)
              ncnn::fold3_record_vulkan_profile(s->net.layers()[i], queries[2*i], queries[2*i+1], s->net.vulkan_device()->info.timestamp_period());
          }
#endif
        }
        s->compute_ms = ms(started); // Legacy field includes GPU transfers and waits.
        started = Clock::now();
        for (const auto &[name, output] : cpu_outputs) {
          // Internal Vulkan tensors may be packed. record_download performs
          // the same unpacking/cast as Extractor's verified Mat overload.
          if (output.empty() || output.elembits() != 32 || output.elempack != 1)
            throw std::runtime_error("ncnn output download layout failed: " + name);
          PyObject *bytes = PyBytes_FromStringAndSize(nullptr, logical_elements(output) * sizeof(float));
          if (!bytes) throw std::runtime_error("ncnn output allocation failed");
          copy_from_ncnn(output, reinterpret_cast<float *>(PyBytes_AS_STRING(bytes)));
          PyObject *entry = Py_BuildValue("(O s N)", Py_None, "float32", bytes);
          PyDict_SetItemString(result, name.c_str(), entry); Py_DECREF(entry);
        }
        s->output_copy_ms = s->readback_ms = ms(started);
      }
      return result;
    } catch (...) { Py_DECREF(result); throw; }
  } catch (const std::exception &e) { PyErr_Clear(); PyErr_SetString(PyExc_RuntimeError, e.what()); return nullptr; }
}
PyObject *profile(PyObject *, PyObject *capsule) {
  auto *s = get(capsule);
  if (!s) return nullptr;
  PyObject *layers = PyList_New(0);
#ifdef FOLD3_NCNN_EXPLICIT_CASTS
  for (const auto &item : s->gpu_timings) {
    PyObject *entry = Py_BuildValue("{s:s,s:s,s:K,s:K,s:d}", "node", item.name.c_str(), "operator", item.type.c_str(),
      "start_ticks", static_cast<unsigned long long>(item.start), "end_ticks", static_cast<unsigned long long>(item.end),
      "gpu_ms", item.milliseconds);
    PyList_Append(layers, entry); Py_DECREF(entry);
  }
#endif
  PyObject *result = Py_BuildValue("{s:i,s:O,s:O,s:I,s:d,s:d,s:d,s:d,s:N}", "version", 1,
    "enabled", s->profiling ? Py_True : Py_False, "valid", s->profile_valid ? Py_True : Py_False,
    "timestamp_valid_bits", s->timestamp_bits, "host_input_ms", s->preparation_ms, "command_record_ms", s->record_ms,
    "submit_wait_ms", s->submit_wait_ms, "host_output_ms", s->output_copy_ms, "gpu_layers", layers);
#ifdef FOLD3_NCNN_EXPLICIT_CASTS
  PyObject *submissions = Py_BuildValue("{s:i,s:I,s:I,s:I,s:K,s:d}", "version", 1,
    "internal_count", s->internal_submissions, "dispatch_limit", s->dispatch_limit, "rough_score", s->rough_score,
    "submitted_groups", static_cast<unsigned long long>(s->submitted_groups), "wait_ms", s->internal_submit_wait_ms);
  PyDict_SetItemString(result, "internal_submissions", submissions); Py_DECREF(submissions);
#endif
  return result;
}
PyObject *trace(PyObject *, PyObject *capsule) {
  auto *s = get(capsule);
  if (!s) return nullptr;
  PyObject *result = PyList_New(0);
  for (const auto &item : s->traces) {
    PyObject *dims = PyTuple_New(item.shape.size());
    for (size_t i = 0; i < item.shape.size(); ++i) PyTuple_SET_ITEM(dims, i, PyLong_FromLong(item.shape[i]));
    PyObject *entry = Py_BuildValue("{s:s,s:s,s:s,s:s,s:s,s:s,s:N,s:N}",
      "operator", item.op.c_str(), "operator_type", item.type.c_str(), "tensor", item.tensor.c_str(),
      "backend", item.backend.c_str(), "layout", item.layout.c_str(), "dtype", item.dtype.c_str(),
      "shape", dims, "data", PyBytes_FromStringAndSize(item.data.data(), item.data.size()));
    PyList_Append(result, entry); Py_DECREF(entry);
  }
  return result;
}
PyMethodDef methods[] = {{"create", create, METH_VARARGS, nullptr}, {"describe", describe, METH_O, nullptr},
                        {"trace", trace, METH_O, nullptr}, {"profile", profile, METH_O, nullptr},
                        {"check_tensor_copy", check_tensor_copy, METH_NOARGS, nullptr},
                        {"check_profile_clock", check_profile_clock, METH_NOARGS, nullptr},
                        {"run", run, METH_VARARGS, nullptr}, {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "fold3_accel_native", nullptr, -1, methods};
}
PyMODINIT_FUNC PyInit_fold3_accel_native() { return PyModule_Create(&module); }
