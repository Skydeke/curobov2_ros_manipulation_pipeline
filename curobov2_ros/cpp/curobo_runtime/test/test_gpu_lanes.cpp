// SPDX-License-Identifier: Apache-2.0
// C++ port of curobov2_ros/test/test_gpu_lanes.py (design section 14.2).
// Same test names (TEST(GpuLanes, FooBar) carries the Python name in a comment),
// same fixtures, same inputs, same expected values. No assertion weakened.
//
// Python fixture: GpuScheduler() + shutdown(). In C++ each test builds a local
// GpuScheduler; its destructor shuts the lanes down.
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "curobo_runtime/gpu_lanes.hpp"

using namespace curobo_runtime;
using namespace std::chrono_literals;

namespace
{

// threading.Event
class Event
{
public:
  void set()
  {
    std::lock_guard<std::mutex> lk(m_);
    flag_ = true;
    cv_.notify_all();
  }
  bool wait(double seconds)
  {
    std::unique_lock<std::mutex> lk(m_);
    return cv_.wait_for(lk, std::chrono::duration<double>(seconds), [&] {return flag_;});
  }

private:
  std::mutex m_;
  std::condition_variable cv_;
  bool flag_ = false;
};

GpuLaneOptions no_stream()
{
  GpuLaneOptions o;
  o.use_cuda_stream = false;
  return o;
}

SubmitOptions opts(int prio)
{
  SubmitOptions o;
  o.prio = prio;
  return o;
}

}  // namespace

// PORT: test_gpu_lanes.py:test_result_and_exception_roundtrip
TEST(GpuLanes, ResultAndExceptionRoundtrip)
{
  GpuScheduler sch;
  auto & ln = sch.lane("a", no_stream());
  EXPECT_EQ(ln.call([](int x) {return x + 1;}, 41), 42);
  EXPECT_THROW(
    ln.call([]() -> int {throw std::invalid_argument("boom");}),
    std::invalid_argument);
  EXPECT_FALSE(ln.poisoned());  // ordinary errors do not poison
}

// PORT: test_gpu_lanes.py:test_single_thread_ownership
TEST(GpuLanes, SingleThreadOwnership)
{
  GpuScheduler sch;
  auto & ln = sch.lane("a", no_stream());
  std::set<std::string> names;
  for (int i = 0; i < 20; ++i) {
    names.insert(ln.call([]() {return current_thread_name();}));
  }
  EXPECT_EQ(names.size(), 1u);
  EXPECT_EQ(*names.begin(), "gpu-lane-a");
  EXPECT_THROW(ln.assert_owner("solver"), NotLaneOwner);
  ln.call([&ln]() {ln.assert_owner("solver");});  // inside lane: ok
}

// PORT: test_gpu_lanes.py:test_submit_from_inside_lane_runs_inline_no_deadlock
TEST(GpuLanes, SubmitFromInsideLaneRunsInlineNoDeadlock)
{
  GpuScheduler sch;
  auto & ln = sch.lane("a", no_stream());
  EXPECT_EQ(ln.call_for(SubmitOptions{}, 2.0, [&ln]() {return ln.call([]() {return 7;});}), 7);
}

// PORT: test_gpu_lanes.py:test_priority_order
TEST(GpuLanes, PriorityOrder)
{
  GpuScheduler sch;
  auto & ln = sch.lane("a", no_stream());
  Event gate;
  std::vector<std::string> order;
  ln.submit([&gate]() {gate.wait(2);});          // occupy the lane
  std::this_thread::sleep_for(50ms);
  auto f1 = ln.submit(opts(PRIO_BULK), [&order]() {order.push_back("bulk");});
  auto f2 = ln.submit(opts(PRIO_PLAN), [&order]() {order.push_back("plan");});
  auto f3 = ln.submit(opts(PRIO_REACTIVE), [&order]() {order.push_back("react");});
  gate.set();
  f1.get();
  f2.get();
  f3.get();
  EXPECT_EQ(order, (std::vector<std::string>{"react", "plan", "bulk"}));
}

// PORT: test_gpu_lanes.py:test_fifo_within_priority
TEST(GpuLanes, FifoWithinPriority)
{
  GpuScheduler sch;
  auto & ln = sch.lane("a", no_stream());
  Event gate;
  std::vector<int> out;
  ln.submit([&gate]() {gate.wait(2);});
  std::this_thread::sleep_for(50ms);
  std::vector<std::future<void>> fs;
  for (int i = 0; i < 10; ++i) {
    fs.push_back(ln.submit([&out, i]() {out.push_back(i);}));
  }
  gate.set();
  for (auto & f : fs) {f.get();}
  std::vector<int> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_EQ(out, expected);
}

// PORT: test_gpu_lanes.py:test_cancel_before_start
TEST(GpuLanes, CancelBeforeStart)
{
  GpuScheduler sch;
  auto & ln = sch.lane("a", no_stream());
  Event gate;
  auto ev = std::make_shared<CancelToken>();
  ln.submit([&gate]() {gate.wait(2);});
  std::this_thread::sleep_for(50ms);
  SubmitOptions o;
  o.cancel = ev;
  auto f = ln.submit(o, []() {return 1;});
  ev->set();
  gate.set();
  EXPECT_THROW(f.get(), JobCancelled);
}

// PORT: test_gpu_lanes.py:test_poison_fails_fast_and_callback
TEST(GpuLanes, PoisonFailsFastAndCallback)
{
  std::vector<std::string> seen;
  GpuScheduler s(
    nullptr, [&seen](GpuLane * l, std::exception_ptr) {seen.push_back(l->name());});
  auto & ln = s.lane("p", no_stream());
  EXPECT_THROW(
    ln.call([]() -> int {
      throw std::runtime_error("CUDA error: an illegal memory access was encountered");
    }),
    std::runtime_error);
  EXPECT_TRUE(ln.poisoned());
  ASSERT_EQ(seen.size(), 1u);
  EXPECT_EQ(seen[0], "p");
  EXPECT_TRUE(s.any_poisoned());
  EXPECT_THROW(ln.call([]() {return 1;}), LanePoisoned);
  s.shutdown();
}

// PORT: test_gpu_lanes.py:test_exclusive_quiesces_other_lanes
TEST(GpuLanes, ExclusiveQuiescesOtherLanes)
{
  GpuScheduler sch;
  auto & a = sch.lane("a", no_stream());
  auto & b = sch.lane("b", no_stream());
  Event running_b;
  Event stop_b;
  std::vector<std::string> violations;
  auto state = std::make_shared<std::atomic<bool>>(false);

  auto b_job = [&]() {
      if (state->load()) {
        violations.push_back("b ran during exclusive");
      }
      running_b.set();
      std::this_thread::sleep_for(20ms);
    };

  auto a_capture = [&]() {
      auto g = sch.exclusive();
      state->store(true);
      std::this_thread::sleep_for(200ms);   // "capture" window
      state->store(false);
    };

  std::thread hammer([&]() {
      while (!stop_b.wait(0)) {
        b.call(b_job);
      }
    });
  running_b.wait(2);
  a.call_for(SubmitOptions{}, 5.0, a_capture);
  stop_b.set();
  hammer.join();
  EXPECT_EQ(violations, (std::vector<std::string>{}));
}

// PORT: test_gpu_lanes.py:test_two_lanes_both_want_exclusive_no_deadlock
TEST(GpuLanes, TwoLanesBothWantExclusiveNoDeadlock)
{
  GpuScheduler sch;
  auto & a = sch.lane("a", no_stream());
  auto & b = sch.lane("b", no_stream());
  std::mutex m;
  std::vector<std::string> done;

  auto cap = [&](const std::string & tag) {
      auto g = sch.exclusive();
      std::this_thread::sleep_for(50ms);
      std::lock_guard<std::mutex> lk(m);
      done.push_back(tag);
    };

  auto fa = a.submit(cap, std::string("a"));
  auto fb = b.submit(cap, std::string("b"));
  fa.get();
  fb.get();
  std::sort(done.begin(), done.end());
  EXPECT_EQ(done, (std::vector<std::string>{"a", "b"}));
}

// PORT: test_gpu_lanes.py:test_lanes_actually_overlap
TEST(GpuLanes, LanesActuallyOverlap)
{
  GpuScheduler sch;
  auto & a = sch.lane("a", no_stream());
  auto & b = sch.lane("b", no_stream());
  auto t0 = std::chrono::steady_clock::now();
  auto fa = a.submit([]() {std::this_thread::sleep_for(300ms);});
  auto fb = b.submit([]() {std::this_thread::sleep_for(300ms);});
  fa.get();
  fb.get();
  double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  EXPECT_LT(elapsed, 0.5);  // parallel, not 0.6
}

// PORT: test_gpu_lanes.py:test_writer_not_starved
TEST(GpuLanes, WriterNotStarved)
{
  GpuScheduler sch;
  auto & a = sch.lane("a", no_stream());
  auto & b = sch.lane("b", no_stream());
  Event stop;

  std::thread hammer([&]() {
      while (!stop.wait(0)) {
        b.call([]() {std::this_thread::sleep_for(5ms);});
      }
    });
  auto t0 = std::chrono::steady_clock::now();
  a.call_for(SubmitOptions{}, 2.0, [&]() {
      auto g = sch.exclusive();
    });
  double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  EXPECT_LT(elapsed, 1.0);
  stop.set();
  hammer.join();
}

// PORT: test_gpu_lanes.py:test_submit_if_idle_drops_when_busy
TEST(GpuLanes, SubmitIfIdleDropsWhenBusy)
{
  GpuScheduler sch;
  auto & ln = sch.lane("v", no_stream());
  Event gate;
  ln.submit([&gate]() {gate.wait(2);});
  std::this_thread::sleep_for(50ms);
  auto r = ln.submit_if_idle([]() {return 1;});
  EXPECT_FALSE(r.has_value());
  gate.set();
  std::this_thread::sleep_for(50ms);
  auto r2 = ln.submit_if_idle([]() {return 5;});
  ASSERT_TRUE(r2.has_value());
  EXPECT_EQ(r2->get(), 5);
}

// PORT: test_gpu_lanes.py:test_shutdown_fails_pending
TEST(GpuLanes, ShutdownFailsPending)
{
  auto s = std::make_unique<GpuScheduler>();
  auto & ln = s->lane("x", no_stream());
  Event gate;
  ln.submit([&gate]() {gate.wait(1);});
  std::this_thread::sleep_for(50ms);
  auto f = ln.submit([]() {return 1;});
  gate.set();
  s->shutdown();
  EXPECT_EQ(future_ready(f), FutureStatus::kReady);
}

// PORT: test_gpu_lanes.py:test_try_exclusive_fails_while_reader_and_state_is_restored
TEST(GpuLanes, TryExclusiveFailsWhileReaderAndStateIsRestored)
{
  CaptureBarrier b;
  Event ready;
  Event release;

  std::thread reader([&]() {
      auto g = b.shared();
      ready.set();
      release.wait(2);
    });

  ready.wait(2);
  EXPECT_FALSE(b.acquire_exclusive(false));
  // a thread that holds shared and fails a try-upgrade must keep its shared hold
  {
    auto g = b.shared();
    EXPECT_FALSE(b.acquire_exclusive(false));
    EXPECT_TRUE(b.holds_shared());
  }
  release.set();
  reader.join();
  EXPECT_TRUE(b.acquire_exclusive(false));
  b.release_exclusive();
}

// PORT: test_gpu_lanes.py:test_upgrade_restores_shared_hold
TEST(GpuLanes, UpgradeRestoresSharedHold)
{
  CaptureBarrier b;
  {
    auto s = b.shared();
    {
      auto e = b.exclusive();
      EXPECT_TRUE(b.holds_exclusive());
    }
    EXPECT_TRUE(b.holds_shared());
    EXPECT_FALSE(b.holds_exclusive());
  }
  EXPECT_FALSE(b.holds_shared());
  EXPECT_TRUE(b.acquire_exclusive(false));
  b.release_exclusive();
}

// PORT: test_gpu_lanes.py:test_legacy_lock_semantics
TEST(GpuLanes, LegacyLockSemantics)
{
  CaptureBarrier b;
  LegacyGpuLock lk(b);
  bool ok = true;
  bool ok2 = false;
  {
    auto g = lk.lock();
    {
      auto g2 = lk.lock();  // re-entrant like RLock
    }
    std::thread t([&]() {ok = lk.acquire(false);});
    t.join();
    EXPECT_FALSE(ok);  // other thread cannot take it
  }
  std::thread t2([&]() {ok2 = lk.acquire(false);});
  t2.join();
  EXPECT_TRUE(ok2);
}

// PORT: test_gpu_lanes.py:test_waiting_on_lane_while_holding_legacy_lock_raises_instead_of_deadlocking
TEST(GpuLanes, WaitingOnLaneWhileHoldingLegacyLockRaises)
{
  GpuScheduler sch;
  auto & ln = sch.lane("a", no_stream());
  auto lk = sch.legacy_lock();
  auto g = lk.lock();
  EXPECT_THROW(ln.submit([]() {return 1;}), DeadlockRisk);
}

// PORT: test_gpu_lanes.py:test_legacy_lock_blocks_lane_jobs_until_released
TEST(GpuLanes, LegacyLockBlocksLaneJobsUntilReleased)
{
  GpuScheduler sch;
  auto & ln = sch.lane("a", no_stream());
  auto lk = sch.legacy_lock();
  std::vector<int> out;
  std::future<void> fut;
  lk.acquire();
  // submit from ANOTHER thread (the holder itself would trip the deadlock guard)
  std::thread t([&]() {fut = ln.submit([&out]() {out.push_back(1);});});
  t.join();
  std::this_thread::sleep_for(150ms);
  EXPECT_EQ(out, (std::vector<int>{}));  // lane job waits for the legacy holder
  lk.release();
  fut.get();
  EXPECT_EQ(out, (std::vector<int>{1}));
}

// PORT: test_gpu_lanes.py:test_sync_hook_runs_once_per_outermost_exclusive_enter_and_exit
TEST(GpuLanes, SyncHookRunsOncePerOutermostExclusive)
{
  CaptureBarrier b;
  int calls = 0;
  b.sync_hook = [&calls]() {calls += 1;};
  {
    auto e = b.exclusive();
    {
      auto e2 = b.exclusive();  // nested: no extra calls
    }
    EXPECT_EQ(calls, 1);
  }
  EXPECT_EQ(calls, 2);
  {
    auto s = b.shared();        // shared never syncs
  }
  EXPECT_EQ(calls, 2);
}

// PORT: test_gpu_lanes.py:test_sync_hook_runs_on_upgrade_from_shared
TEST(GpuLanes, SyncHookRunsOnUpgradeFromShared)
{
  CaptureBarrier b;
  int calls = 0;
  b.sync_hook = [&calls]() {calls += 1;};
  {
    auto s = b.shared();
    {
      auto e = b.exclusive();
    }
  }
  EXPECT_EQ(calls, 2);
}

// PORT: test_gpu_lanes.py:test_scheduler_installs_a_sync_hook_that_is_safe_without_cuda
TEST(GpuLanes, SchedulerInstallsSafeSyncHook)
{
  GpuScheduler sch;
  EXPECT_TRUE(static_cast<bool>(sch.barrier.sync_hook));
  {
    auto e = sch.exclusive();  // must not raise on a CPU-only box
  }
}
