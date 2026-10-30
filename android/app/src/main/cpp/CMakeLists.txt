cmake_minimum_required(VERSION 3.22)
project(fold3 LANGUAGES C CXX)
set(LIBUSB_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../build/native/libusb")
if(NOT EXISTS "${LIBUSB_ROOT}/android/config.h")
  message(FATAL_ERROR "Run python3 android/scripts/prepare_native.py before Gradle")
endif()
# Same sources and Android config as upstream android/jni/libusb.mk.
add_library(usb1.0 SHARED
  ${LIBUSB_ROOT}/libusb/core.c ${LIBUSB_ROOT}/libusb/descriptor.c
  ${LIBUSB_ROOT}/libusb/hotplug.c ${LIBUSB_ROOT}/libusb/io.c
  ${LIBUSB_ROOT}/libusb/sync.c ${LIBUSB_ROOT}/libusb/strerror.c
  ${LIBUSB_ROOT}/libusb/os/linux_usbfs.c ${LIBUSB_ROOT}/libusb/os/linux_netlink.c
  ${LIBUSB_ROOT}/libusb/os/events_posix.c ${LIBUSB_ROOT}/libusb/os/threads_posix.c)
target_include_directories(usb1.0 PRIVATE ${LIBUSB_ROOT}/android ${LIBUSB_ROOT}/libusb ${LIBUSB_ROOT}/libusb/os)
target_compile_options(usb1.0 PRIVATE -fvisibility=hidden)
target_link_libraries(usb1.0 PRIVATE log)
set(MSGQ_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../build/native/msgq")
if(NOT EXISTS "${MSGQ_ROOT}/msgq/android_ipc.h")
  message(FATAL_ERROR "Run python3 android/scripts/prepare_native.py before Gradle")
endif()
add_library(msgq_android STATIC
  ${MSGQ_ROOT}/msgq/ipc.cc ${MSGQ_ROOT}/msgq/event.cc
  ${MSGQ_ROOT}/msgq/impl_msgq.cc ${MSGQ_ROOT}/msgq/impl_fake.cc ${MSGQ_ROOT}/msgq/msgq.cc
  ${MSGQ_ROOT}/msgq/visionipc/visionipc.cc ${MSGQ_ROOT}/msgq/visionipc/visionipc_server.cc
  ${MSGQ_ROOT}/msgq/visionipc/visionipc_client.cc ${MSGQ_ROOT}/msgq/visionipc/visionbuf.cc)
target_include_directories(msgq_android PUBLIC ${MSGQ_ROOT})
target_compile_options(msgq_android PRIVATE -UNDEBUG -Wno-error)
set(PYTHON_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../build/native/python")
foreach(extension IN ITEMS ipc_pyx visionipc_pyx)
  if(extension STREQUAL "visionipc_pyx")
    set(package "msgq/visionipc")
  else()
    set(package "msgq")
  endif()
  add_library(${extension} MODULE ${MSGQ_ROOT}/${package}/${extension}.cpp)
  target_include_directories(${extension} PRIVATE ${PYTHON_ROOT}/include/python3.12)
  target_compile_options(${extension} PRIVATE -Wno-error -UNDEBUG)
  target_link_libraries(${extension} PRIVATE msgq_android ${PYTHON_ROOT}/jniLibs/arm64-v8a/libpython3.12.so)
  set_target_properties(${extension} PROPERTIES PREFIX "" SUFFIX ".cpython-312.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../../../build/native-python/${package}")
endforeach()
set(CAPNP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../build/native/capnproto-c++-1.0.1")
set(PYCAPNP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../build/native/pycapnp-2.1.0")
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(WITH_OPENSSL OFF CACHE STRING "" FORCE)
set(WITH_ZLIB OFF CACHE STRING "" FORCE)
set(WITH_FIBERS OFF CACHE STRING "" FORCE)
add_subdirectory(${CAPNP_ROOT} capnproto EXCLUDE_FROM_ALL)
foreach(library IN ITEMS kj kj-async kj-http capnp capnp-rpc capnp-json capnpc)
  if(TARGET ${library})
    target_compile_options(${library} PRIVATE -Wno-error)
  endif()
endforeach()
add_library(capnp_python MODULE ${PYCAPNP_ROOT}/capnp/lib/capnp.cpp ${PYCAPNP_ROOT}/capnp/helpers/capabilityHelper.cpp)
target_include_directories(capnp_python PRIVATE ${PYTHON_ROOT}/include/python3.12 ${PYCAPNP_ROOT} ${CAPNP_ROOT}/src)
target_compile_options(capnp_python PRIVATE -Wno-error)
target_link_libraries(capnp_python PRIVATE capnpc capnp-rpc kj-async ${PYTHON_ROOT}/jniLibs/arm64-v8a/libpython3.12.so)
set_target_properties(capnp_python PROPERTIES PREFIX "" OUTPUT_NAME "capnp" SUFFIX ".cpython-312.so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../../../build/native-python/capnp/lib")
add_library(fold3 SHARED bridge.cpp panda_usb.cpp visionipc_bridge.cpp)
target_include_directories(fold3 PRIVATE ${LIBUSB_ROOT}/libusb ../../../../../)
target_link_libraries(fold3 PRIVATE dl usb1.0 msgq_android)
