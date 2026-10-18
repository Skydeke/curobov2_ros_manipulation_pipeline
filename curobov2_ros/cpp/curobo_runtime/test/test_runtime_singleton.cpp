// SPDX-License-Identifier: Apache-2.0
// M1 acceptance: Runtime singleton across two dlopen'd libraries, and one
// interpreter (design sections 4, 10 M1).
#include <gtest/gtest.h>
#include <dlfcn.h>

#include "curobo_runtime/runtime.hpp"

using curobo_runtime::Runtime;

namespace
{
using instance_fn = void * (*) ();
using int_fn = int (*) ();
}  // namespace

TEST(Runtime, SingletonAcrossTwoDlopenLibraries)
{
  void * handle_a = dlopen(PROBE_A_PATH, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(handle_a, nullptr) << dlerror();
  void * handle_b = dlopen(PROBE_B_PATH, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(handle_b, nullptr) << dlerror();

  auto instance_a = reinterpret_cast<instance_fn>(
    dlsym(handle_a, "curobo_probe_runtime_instance"));
  auto instance_b = reinterpret_cast<instance_fn>(
    dlsym(handle_b, "curobo_probe_runtime_instance"));
  ASSERT_NE(instance_a, nullptr);
  ASSERT_NE(instance_b, nullptr);

  // Same address in both, and equal to the in-process reference.
  EXPECT_EQ(instance_a(), instance_b());
  EXPECT_EQ(instance_a(), static_cast<void *>(&Runtime::instance()));

  // One interpreter, created once.
  Runtime::instance().start();
  EXPECT_TRUE(Runtime::instance().started());

  auto py_a = reinterpret_cast<int_fn>(dlsym(handle_a, "curobo_probe_py_initialized"));
  auto py_b = reinterpret_cast<int_fn>(dlsym(handle_b, "curobo_probe_py_initialized"));
  auto init_count_a = reinterpret_cast<int_fn>(dlsym(handle_a, "curobo_probe_init_count"));
  ASSERT_NE(py_a, nullptr);
  ASSERT_NE(py_b, nullptr);
  ASSERT_NE(init_count_a, nullptr);

  EXPECT_EQ(py_a(), 1);
  EXPECT_EQ(py_b(), 1);
  EXPECT_EQ(init_count_a(), 1);

  Runtime::instance().stop();
}
