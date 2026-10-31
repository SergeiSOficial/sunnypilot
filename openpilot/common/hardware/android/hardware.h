#pragma once
#include "common/hardware/base.h"

// UNKNOWN is deliberate: the Fold is not a calibrated comma camera platform.
class HardwareAndroid : public HardwareNone {
public:
  static std::string get_name() { return "android"; }
  static bool PC() { return true; }  // Unprivileged paths; no AGNOS/system writes.
};
