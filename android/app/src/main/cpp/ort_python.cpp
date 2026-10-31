#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <onnxruntime_cxx_api.h>
#include <nnapi_provider_factory.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "fold3");
struct Session {
  Ort::Session model{nullptr};
  std::vector<std::string> inputs, outputs;
  explicit Session(const char *path, const char *backend, int threads, bool spinning, bool strict) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(threads).SetInterOpNumThreads(1);
    options.AddConfigEntry("session.intra_op.allow_spinning", spinning ? "1" : "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", spinning ? "1" : "0");
    if (strict) options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL).SetDeterministicCompute(true);
    if (std::strcmp(backend, "nnapi") == 0)
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(options, NNAPI_FLAG_CPU_DISABLED));
    else if (std::strcmp(backend, "cpu") != 0) throw std::invalid_argument("unsupported ORT backend");
    model = Ort::Session(environment, path, options);
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < model.GetInputCount(); ++i) inputs.emplace_back(model.GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < model.GetOutputCount(); ++i) outputs.emplace_back(model.GetOutputNameAllocated(i, allocator).get());
  }
};
constexpr const char *capsule_name = "fold3.ort.session";
void destroy(PyObject *capsule) { delete static_cast<Session *>(PyCapsule_GetPointer(capsule, capsule_name)); }
struct Buffer {
  Py_buffer value{};
  ~Buffer() { if (value.obj) PyBuffer_Release(&value); }
};
struct ReleasedGIL {
  PyThreadState *state = PyEval_SaveThread();
  ~ReleasedGIL() { PyEval_RestoreThread(state); }
};
const char *format(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "f";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "e";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "B";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "i";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "q";
    default: throw std::invalid_argument("unsupported model tensor dtype");
  }
}
const char *numpy_type(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    default: throw std::invalid_argument("unsupported output dtype");
  }
}
PyObject *create(PyObject *, PyObject *args) {
  const char *path, *backend;
  int threads, spinning = 0, strict = 0;
  if (!PyArg_ParseTuple(args, "ssi|pp", &path, &backend, &threads, &spinning, &strict)) return nullptr;
  if (threads < 1 || threads > 4) { PyErr_SetString(PyExc_ValueError, "ORT thread count must be 1..4"); return nullptr; }
  try {
    auto session = std::make_unique<Session>(path, backend, threads, spinning, strict);
    PyObject *result = PyCapsule_New(session.get(), capsule_name, destroy);
    if (result) session.release();
    return result;
  } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return nullptr; }
}
PyObject *run(PyObject *, PyObject *args) {
  PyObject *capsule, *values;
  if (!PyArg_ParseTuple(args, "OO!", &capsule, &PyDict_Type, &values)) return nullptr;
  auto *session = static_cast<Session *>(PyCapsule_GetPointer(capsule, capsule_name));
  if (!session) return nullptr;
  try {
    if (PyDict_Size(values) != static_cast<Py_ssize_t>(session->inputs.size())) throw std::invalid_argument("model input set differs");
    std::vector<Buffer> buffers(session->inputs.size());
    std::vector<Ort::Value> tensors;
    std::vector<const char *> input_names, output_names;
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (size_t i = 0; i < session->inputs.size(); ++i) {
      const auto &name = session->inputs[i];
      PyObject *value = PyDict_GetItemString(values, name.c_str());
      if (!value) throw std::invalid_argument("missing model input: " + name);
      if (PyObject_GetBuffer(value, &buffers[i].value, PyBUF_FORMAT | PyBUF_STRIDES) != 0) return nullptr;
      auto &buffer = buffers[i].value;
      auto input_type = session->model.GetInputTypeInfo(i);
      auto info = input_type.GetTensorTypeAndShapeInfo();
      auto shape = info.GetShape();
      auto type = info.GetElementType();
      bool shape_matches = buffer.ndim == static_cast<int>(shape.size());
      for (size_t d = 0; shape_matches && d < shape.size(); ++d) shape_matches = shape[d] == buffer.shape[d];
      // Native NumPy int64 uses 'l' on ARM64 and 'q' on other Python builds.
      bool dtype_matches = std::strcmp(buffer.format, format(type)) == 0 ||
        (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && std::strcmp(buffer.format, "l") == 0 && buffer.itemsize == 8);
      if (!shape_matches || !dtype_matches || !PyBuffer_IsContiguous(&buffer, 'C') ||
          buffer.len != static_cast<Py_ssize_t>(info.GetElementCount() * buffer.itemsize))
        throw std::invalid_argument("model input shape, dtype or stride mismatch: " + name);
      tensors.emplace_back(Ort::Value::CreateTensor(memory, buffer.buf, buffer.len, shape.data(), shape.size(), type));
      input_names.push_back(name.c_str());
    }
    for (const auto &name : session->outputs) output_names.push_back(name.c_str());
    std::vector<Ort::Value> outputs;
    {
      ReleasedGIL release;
      outputs = session->model.Run(Ort::RunOptions{nullptr}, input_names.data(), tensors.data(), tensors.size(), output_names.data(), output_names.size());
    }
    PyObject *result = PyDict_New();
    if (!result) return nullptr;
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto info = outputs[i].GetTensorTypeAndShapeInfo();
      auto dims = info.GetShape();
      PyObject *shape = PyTuple_New(dims.size());
      if (!shape) { Py_DECREF(result); return nullptr; }
      for (size_t d = 0; d < dims.size(); ++d) PyTuple_SET_ITEM(shape, d, PyLong_FromLongLong(dims[d]));
      PyObject *data = PyBytes_FromStringAndSize(static_cast<const char *>(outputs[i].GetTensorRawData()), info.GetElementCount() *
        (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ? 1 : info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? 2 :
         info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8 : 4));
      if (!data) { Py_DECREF(shape); Py_DECREF(result); return nullptr; }
      PyObject *entry = Py_BuildValue("(NsN)", shape, numpy_type(info.GetElementType()), data);
      if (!entry || PyDict_SetItemString(result, session->outputs[i].c_str(), entry) != 0) {
        Py_XDECREF(entry); Py_DECREF(result); return nullptr;
      }
      Py_DECREF(entry);
    }
    return result;
  } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return nullptr; }
}
PyObject *gather_rows_u8(PyObject *, PyObject *args) {
  PyObject *source, *rows, *columns, *destination;
  if (!PyArg_ParseTuple(args, "OOOO", &source, &rows, &columns, &destination)) return nullptr;
  Buffer input, row_offsets, column_offsets, output;
  if (PyObject_GetBuffer(source, &input.value, PyBUF_FORMAT | PyBUF_STRIDES) != 0 ||
      PyObject_GetBuffer(rows, &row_offsets.value, PyBUF_FORMAT | PyBUF_STRIDES) != 0 ||
      PyObject_GetBuffer(columns, &column_offsets.value, PyBUF_FORMAT | PyBUF_STRIDES) != 0 ||
      PyObject_GetBuffer(destination, &output.value, PyBUF_FORMAT | PyBUF_STRIDES | PyBUF_WRITABLE) != 0) return nullptr;
  auto &s = input.value, &r = row_offsets.value, &c = column_offsets.value, &d = output.value;
  auto index_buffer = [](const Py_buffer &b) {
    return b.ndim == 1 && b.itemsize == sizeof(Py_ssize_t) &&
      (!std::strcmp(b.format, "l") || !std::strcmp(b.format, "q") || !std::strcmp(b.format, "n")) && PyBuffer_IsContiguous(&b, 'C');
  };
  if (s.itemsize != 1 || std::strcmp(s.format, "B") || d.itemsize != 1 || std::strcmp(d.format, "B") ||
      !PyBuffer_IsContiguous(&s, 'C') || !PyBuffer_IsContiguous(&d, 'C') || !index_buffer(r) || !index_buffer(c) ||
      c.len == 0 || r.len == 0 || d.len % (c.len / c.itemsize) || d.len / (c.len / c.itemsize) != r.len / r.itemsize) {
    PyErr_SetString(PyExc_ValueError, "gather requires contiguous uint8 source/output and matching nonempty intp rows/columns");
    return nullptr;
  }
  try {
    ReleasedGIL release;
    auto index = [](const Py_buffer &b, Py_ssize_t n) {
      Py_ssize_t value;
      std::memcpy(&value, static_cast<const char *>(b.buf) + n * sizeof(value), sizeof(value));
      return value;
    };
    const Py_ssize_t width = c.len / c.itemsize, height = r.len / r.itemsize;
    Py_ssize_t max_column = 0;
    for (Py_ssize_t x = 0; x < width; ++x) {
      const auto column = index(c, x);
      if (column < 0 || column >= s.len) throw std::invalid_argument("column outside source");
      max_column = std::max(max_column, column);
    }
    for (Py_ssize_t y = 0; y < height; ++y) {
      const auto row = index(r, y);
      if (row < 0 || row >= s.len - max_column) throw std::invalid_argument("row outside source");
    }
    for (Py_ssize_t y = 0; y < height; ++y) {
      const auto row = index(r, y);
      auto *dest = static_cast<uint8_t *>(d.buf) + y * width;
      if (y && row == index(r, y - 1)) std::memcpy(dest, dest - width, width);
      else for (Py_ssize_t x = 0; x < width; ++x) dest[x] = static_cast<const uint8_t *>(s.buf)[row + index(c, x)];
    }
  } catch (const std::exception &e) { PyErr_SetString(PyExc_ValueError, e.what()); return nullptr; }
  Py_RETURN_NONE;
}
PyMethodDef methods[] = {{"create", create, METH_VARARGS, nullptr}, {"run", run, METH_VARARGS, nullptr},
                        {"gather_rows_u8", gather_rows_u8, METH_VARARGS, nullptr}, {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "fold3_ort_native", nullptr, -1, methods, nullptr, nullptr, nullptr, nullptr};
}
PyMODINIT_FUNC PyInit_fold3_ort_native() { return PyModule_Create(&module); }
