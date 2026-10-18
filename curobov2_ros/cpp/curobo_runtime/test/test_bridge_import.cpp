// SPDX-License-Identifier: Apache-2.0
// M1 acceptance: numpy and torch import inside the embedded interpreter
// (design sections 5, 10 M1), and the interpreter is created once.
#include <gtest/gtest.h>

#include "curobo_runtime/py_bridge.hpp"

using curobo_runtime::PyBridge;
using curobo_runtime::RobotHandle;

TEST(PyBridge, NumpyAndTorchImportInsideInterpreter)
{
  PyBridge bridge;
  EXPECT_FALSE(bridge.interpreter_running());

  RobotHandle handle = bridge.init("");
  EXPECT_EQ(handle, 1);
  EXPECT_TRUE(bridge.initialized());
  EXPECT_TRUE(bridge.interpreter_running());

  EXPECT_TRUE(bridge.module_available("numpy"));
  EXPECT_TRUE(bridge.module_available("torch"));
  EXPECT_TRUE(bridge.module_available("curobo_bridge"));

  // Re-init is idempotent: the interpreter is created once.
  bridge.init("");
  EXPECT_EQ(bridge.interpreter_init_count(), 1);

  bridge.shutdown(handle);
  EXPECT_FALSE(bridge.initialized());
}

TEST(PyBridge, BindLaneIsRecorded)
{
  PyBridge bridge;
  bridge.init("");
  bridge.bind_lane("plan");
  bridge.bind_lane("kin");
  auto lanes = bridge.bound_lanes();
  ASSERT_EQ(lanes.size(), 2u);
  EXPECT_EQ(lanes[0], "plan");
  EXPECT_EQ(lanes[1], "kin");
  bridge.shutdown(1);
}
