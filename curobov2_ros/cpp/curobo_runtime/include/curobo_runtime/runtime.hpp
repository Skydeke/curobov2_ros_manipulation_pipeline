// SPDX-License-Identifier: Apache-2.0
// PORT: (design section 4) Runtime singleton, defined in libcurobo_runtime.so
// only. Components and libcurobo_core.so link to it dynamically; there is no
// static linking and no header-only singleton. The singleton test dlopens two
// libraries and asserts instance() is one address and the interpreter was
// initialised once.
#ifndef CUROBO_RUNTIME__RUNTIME_HPP_
#define CUROBO_RUNTIME__RUNTIME_HPP_

#include <memory>
#include <string>

#include "curobo_runtime/bridge.hpp"
#include "curobo_runtime/gpu_lanes.hpp"

namespace curobo_runtime
{

class Runtime
{
public:
  // Defined in libcurobo_runtime.so only.
  static Runtime & instance();

  // The process's single bridge. Created on first start().
  IBridge & bridge();
  const IBridge & bridge() const;

  // Creates the one interpreter (PyBridge) and calls bridge->init(cfg).
  // Idempotent: subsequent calls are no-ops.
  void start(const std::string & cfg = "");
  void stop();
  bool started() const {return started_;}

  // The bridge handle created by start() (curobo_bridge.init).
  RobotHandle handle() const {return handle_;}

  // PORT: node.gpu.exclusive() / gpu_lock - the process-global CaptureBarrier.
  // Normal (shared) GPU jobs take shared(); a CUDA-graph capture, solver
  // rebuild or world push takes exclusive() (drains shared holders, blocks new
  // ones). This is what keeps a capture from being invalidated by a concurrent
  // camera/ESDF/segmentation kernel.
  CaptureBarrier & barrier() {return barrier_;}

  // How many times the interpreter init ran (1 for the expected singleton).
  int interpreter_init_count() const;

  // Test seam: install a bridge before start(). Must not be called after start().
  void set_bridge_for_test(std::unique_ptr<IBridge> bridge);

  Runtime(const Runtime &) = delete;
  Runtime & operator=(const Runtime &) = delete;

private:
  Runtime();
  ~Runtime();

  std::unique_ptr<IBridge> bridge_;
  CaptureBarrier barrier_;
  RobotHandle handle_ = -1;
  bool started_ = false;
};

}  // namespace curobo_runtime

#endif  // CUROBO_RUNTIME__RUNTIME_HPP_
