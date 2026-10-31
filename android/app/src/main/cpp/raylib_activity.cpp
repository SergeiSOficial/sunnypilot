#include <jni.h>
#include <android/log.h>
#include <mutex>

static JavaVM *vm = nullptr;
static jobject activity = nullptr;
static std::mutex lock;

extern "C" JNIEXPORT void JNICALL
Java_ai_sunnypilot_fold3_DrivingActivity_prepareNative(JNIEnv *env, jobject self) {
  std::lock_guard<std::mutex> guard(lock);
  env->GetJavaVM(&vm);
  if (activity != nullptr) env->DeleteGlobalRef(activity);
  activity = env->NewGlobalRef(self);
}

// Raylib's Android backend invokes this on its native app thread. All OpenGL
// calls, including the existing Python UI, must stay on this thread.
extern "C" int main(int, char **) {
  JNIEnv *env = nullptr;
  if (vm == nullptr || vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return 1;
  jobject current;
  {
    std::lock_guard<std::mutex> guard(lock);
    current = env->NewLocalRef(activity);
  }
  if (current == nullptr) { vm->DetachCurrentThread(); return 1; }
  jclass cls = env->GetObjectClass(current);
  jmethodID run = env->GetMethodID(cls, "runUi", "()V");
  env->CallVoidMethod(current, run);
  if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); }
  env->DeleteLocalRef(cls);
  {
    std::lock_guard<std::mutex> guard(lock);
    if (env->IsSameObject(activity, current)) { env->DeleteGlobalRef(activity); activity = nullptr; }
  }
  env->DeleteLocalRef(current);
  vm->DetachCurrentThread();
  return 0;
}
