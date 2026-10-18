// SPDX-License-Identifier: Apache-2.0
// Probe library for the Runtime singleton test (design section 4). Built twice
// (runtime_probe_a / runtime_probe_b) so the test can dlopen two independent
// shared objects and check they resolve the SAME Runtime::instance() and see a
// single interpreter.
#include <Python.h>

#include "curobo_runtime/runtime.hpp"

extern "C" void * curobo_probe_runtime_instance()
{
  return static_cast<void *>(&curobo_runtime::Runtime::instance());
}

extern "C" int curobo_probe_py_initialized()
{
  return Py_IsInitialized();
}

extern "C" int curobo_probe_init_count()
{
  return curobo_runtime::Runtime::instance().interpreter_init_count();
}
