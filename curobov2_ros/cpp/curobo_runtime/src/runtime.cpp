// SPDX-License-Identifier: Apache-2.0
# include "curobo_runtime/runtime.hpp"

#include <mutex>
#include <stdexcept>

#include "curobo_runtime/py_bridge.hpp"

namespace curobo_runtime
{

// PORT: (design section 4 singleton rule) one definition, in this library only.
Runtime & Runtime::instance()
{
  static Runtime instance;
  return instance;
}

Runtime::Runtime() = default;

Runtime::~Runtime()
{
  try {
    stop();
  } catch (...) {
  }
}

IBridge & Runtime::bridge()
{
  if (!bridge_) {
    throw std::runtime_error("Runtime::bridge() called before set_bridge_for_test()/start()");
  }
  return *bridge_;
}

const IBridge & Runtime::bridge() const
{
  if (!bridge_) {
    throw std::runtime_error("Runtime::bridge() called before set_bridge_for_test()/start()");
  }
  return *bridge_;
}

void Runtime::start(const std::string & cfg)
{
  if (started_) {
    return;
  }
  if (!bridge_) {
    bridge_ = std::make_unique<PyBridge>();
  }
  handle_ = bridge_->init(cfg);
  started_ = true;
}

void Runtime::stop()
{
  if (!started_) {
    return;
  }
  bridge_->shutdown(handle_);
  handle_ = -1;
  started_ = false;
}

int Runtime::interpreter_init_count() const
{
  if (!bridge_) {
    return 0;
  }
  return bridge_->interpreter_init_count();
}

void Runtime::set_bridge_for_test(std::unique_ptr<IBridge> bridge)
{
  if (started_) {
    throw std::runtime_error("set_bridge_for_test() after start()");
  }
  bridge_ = std::move(bridge);
}

}  // namespace curobo_runtime
