// SPDX-License-Identifier: Apache-2.0
// PORT: curobov2_ros/core/gpu_lanes.py
//
// C++ port of the GPU lane scheduler. Semantics, priorities, poison marker
// list, barrier fairness and error messages are kept identical to the Python
// source (design section 14.4). Names follow the Python identifiers.
//
// One lane is one thread and the only thread allowed to touch its group of
// solvers/graphs/device tensors. Callers submit a callable and get a
// std::future back; results are plain CPU data.
#ifndef CUROBO_RUNTIME__GPU_LANES_HPP_
#define CUROBO_RUNTIME__GPU_LANES_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "curobo_runtime/bridge.hpp"

namespace curobo_runtime
{

// Priorities (lower = more urgent). PORT: gpu_lanes.py:PRIO_*
inline constexpr int PRIO_REACTIVE = 0;
inline constexpr int PRIO_PLAN = 10;
inline constexpr int PRIO_KIN = 20;
inline constexpr int PRIO_PERCEPTION = 30;
inline constexpr int PRIO_BULK = 40;
// cuTAMP chunk priority (design section 8: just below PLAN).
inline constexpr int PRIO_TAMP = 11;

// PORT: gpu_lanes.py:_CUDA_FATAL_MARKERS
extern const std::vector<std::string> kCudaFatalMarkers;

// PORT: gpu_lanes.py:GpuLane._is_fatal's marker scan, exposed as a message
// predicate so call sites that CATCH a CUDA error (and turn it into a service
// result string instead of letting it reach GpuLane::run_job) can still detect a
// fatal fault and stop serving a corrupt context.
bool is_cuda_fatal_message(const std::string & message);

// PORT: gpu_lanes.py:LanePoisoned / LaneStopped / JobCancelled / NotLaneOwner /
// DeadlockRisk
class LanePoisoned : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};
class LaneStopped : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};
class JobCancelled : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};
class NotLaneOwner : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};
class DeadlockRisk : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

// Status of a shutdown-safe future, mirroring concurrent.futures.
enum class FutureStatus { kReady, kTimeout };
template <class R>
FutureStatus future_ready(const std::future<R> & f)
{
  return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready ?
         FutureStatus::kReady : FutureStatus::kTimeout;
}

// Cancellation token replacing threading.Event (PORT: cancel= kwarg).
class CancelToken
{
public:
  void set() { flag_.store(true); }
  bool is_set() const { return flag_.load(); }

private:
  std::atomic<bool> flag_{false};
};

// --------------------------------------------------------------------------
// Capture barrier (readers = normal jobs, writer = capture / reset / rebuild)
// --------------------------------------------------------------------------
// PORT: gpu_lanes.py:CaptureBarrier
class CaptureBarrier
{
public:
  CaptureBarrier() = default;
  CaptureBarrier(const CaptureBarrier &) = delete;
  CaptureBarrier & operator=(const CaptureBarrier &) = delete;

  // Called right after the outermost exclusive acquire and right before the
  // outermost exclusive release. The scheduler points it at a device-wide
  // CUDA synchronize (through the bridge).
  std::function<void()> sync_hook;

  // -- shared
  void acquire_shared();
  void release_shared();
  // -- exclusive
  bool acquire_exclusive(bool blocking = true);
  void release_exclusive();

  bool holds_exclusive() const;
  bool holds_shared() const;

  // RAII guards replacing the Python context managers.
  class SharedGuard
  {
public:
    explicit SharedGuard(CaptureBarrier & b) : b_(&b) { b_->acquire_shared(); }
    ~SharedGuard() { b_->release_shared(); }
    SharedGuard(const SharedGuard &) = delete;
    SharedGuard & operator=(const SharedGuard &) = delete;
    SharedGuard(SharedGuard && o) noexcept : b_(o.b_) { o.b_ = nullptr; }
    SharedGuard & operator=(SharedGuard && o) noexcept
    {
      b_ = o.b_;
      o.b_ = nullptr;
      return *this;
    }

private:
    CaptureBarrier * b_;
  };
  class ExclusiveGuard
  {
public:
    explicit ExclusiveGuard(CaptureBarrier & b) : b_(&b) { b_->acquire_exclusive(true); }
    ~ExclusiveGuard() { if (b_) {b_->release_exclusive();} }
    ExclusiveGuard(const ExclusiveGuard &) = delete;
    ExclusiveGuard & operator=(const ExclusiveGuard &) = delete;
    ExclusiveGuard(ExclusiveGuard && o) noexcept : b_(o.b_) { o.b_ = nullptr; }
    ExclusiveGuard & operator=(ExclusiveGuard && o) noexcept
    {
      b_ = o.b_;
      o.b_ = nullptr;
      return *this;
    }

private:
    CaptureBarrier * b_;
  };

  SharedGuard shared() {return SharedGuard(*this);}
  ExclusiveGuard exclusive() {return ExclusiveGuard(*this);}

private:
  struct ThreadState
  {
    int shared_depth = 0;
    int excl_depth = 0;
    int saved_shared = 0;
  };
  static ThreadState & tls(const CaptureBarrier * key);

  void run_sync_hook();

  std::mutex mutex_;
  std::condition_variable cond_;
  int readers_ = 0;
  std::optional<std::thread::id> writer_;
  int writers_waiting_ = 0;
};

// PORT: gpu_lanes.py:LegacyGpuLock
class LegacyGpuLock
{
public:
  explicit LegacyGpuLock(CaptureBarrier & barrier) : barrier_(&barrier) {}
  bool acquire(bool blocking = true) {return barrier_->acquire_exclusive(blocking);}
  void release() {barrier_->release_exclusive();}
  // Guard form of `with lk:`.
  class Guard
  {
public:
    explicit Guard(LegacyGpuLock & lk) : lk_(&lk) { lk_->acquire(true); }
    ~Guard() { if (lk_) {lk_->release();} }
    Guard(const Guard &) = delete;
    Guard & operator=(const Guard &) = delete;

private:
    LegacyGpuLock * lk_;
  };
  Guard lock() {return Guard(*this);}

private:
  CaptureBarrier * barrier_;
};

// --------------------------------------------------------------------------
// Lane
// --------------------------------------------------------------------------
class GpuLane;

// Thread-name helper (PORT: threading.current_thread().name); lanes are named
// "gpu-lane-<name>" and the tests assert on it.
std::string current_thread_name();
void set_current_thread_name(const std::string & name);

struct GpuLaneOptions
{
  bool use_cuda_stream = true;
  std::string device;                                             // empty = "cuda"
  std::function<void(class GpuLane *, std::exception_ptr)> on_poisoned;
};

// Extra submit options (Python keyword arguments).
struct SubmitOptions
{
  int prio = PRIO_BULK;
  std::shared_ptr<CancelToken> cancel;
  std::string name;
};

namespace detail
{
struct Job
{
  int prio = PRIO_BULK;
  std::uint64_t seq = 0;
  std::string name;
  std::function<void()> invoke;                          // runs fn; may throw
  std::function<void()> commit;                          // sets the promise value
  std::function<void(std::exception_ptr)> fail;          // sets the promise exception
  std::shared_ptr<CancelToken> cancel;
  std::chrono::steady_clock::time_point t_submit;
};

struct JobGreater
{
  bool operator()(const Job & a, const Job & b) const
  {
    return std::tie(a.prio, a.seq) > std::tie(b.prio, b.seq);
  }
};
}  // namespace detail

// PORT: gpu_lanes.py:GpuLane
class GpuLane
{
public:
  GpuLane(
    std::string name, CaptureBarrier & barrier, IBridge * bridge = nullptr,
    GpuLaneOptions options = {});
  ~GpuLane();
  GpuLane(const GpuLane &) = delete;
  GpuLane & operator=(const GpuLane &) = delete;

  const std::string & name() const {return name_;}
  bool poisoned() const {return poisoned_.load();}
  bool owns_current_thread() const {return std::this_thread::get_id() == thread_id_;}
  // PORT: gpu_lanes.py:GpuLane.assert_owner
  void assert_owner(const std::string & what = "");
  bool idle();
  bool stopped() const {return stop_.load();}
  IBridge * bridge() const {return bridge_;}

  // PORT: gpu_lanes.py:GpuLane.submit
  template <class F, class... Args>
  auto submit(F && f, Args &&... args)
  -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
  {
    return submit_opts(SubmitOptions{}, std::forward<F>(f), std::forward<Args>(args)...);
  }

  template <class F, class... Args>
  auto submit(const SubmitOptions & opts, F && f, Args &&... args)
  -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
  {
    return submit_opts(opts, std::forward<F>(f), std::forward<Args>(args)...);
  }

  // PORT: gpu_lanes.py:GpuLane.call
  template <class F, class... Args>
  auto call(F && f, Args &&... args)
  -> std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>
  {
    auto fut = submit(std::forward<F>(f), std::forward<Args>(args)...);
    if constexpr (std::is_void_v<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>) {
      fut.get();
    } else {
      return fut.get();
    }
  }

  template <class F, class... Args>
  auto call(const SubmitOptions & opts, F && f, Args &&... args)
  -> std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>
  {
    auto fut = submit(opts, std::forward<F>(f), std::forward<Args>(args)...);
    if constexpr (std::is_void_v<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>) {
      fut.get();
    } else {
      return fut.get();
    }
  }

  // call() with a bound, so a deadlock surfaces as a failure instead of a hang
  // (Python: call(..., timeout=...)).
  template <class F, class... Args>
  auto call_for(const SubmitOptions & opts, double timeout_s, F && f, Args &&... args)
  -> std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>
  {
    using R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
    auto fut = submit(opts, std::forward<F>(f), std::forward<Args>(args)...);
    if (fut.wait_for(std::chrono::duration<double>(timeout_s)) != std::future_status::ready) {
      throw std::runtime_error("call timed out");
    }
    if constexpr (std::is_void_v<R>) {
      fut.get();
    } else {
      return fut.get();
    }
  }

  // PORT: gpu_lanes.py:GpuLane.submit_if_idle
  template <class F, class... Args>
  auto submit_if_idle(F && f, Args &&... args)
  -> std::optional<std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>>
  {
    if (!idle() || poisoned() || stopped()) {
      return std::nullopt;
    }
    SubmitOptions opts;
    opts.prio = PRIO_BULK;
    return std::make_optional(submit(opts, std::forward<F>(f), std::forward<Args>(args)...));
  }

  template <class F, class... Args>
  auto submit_if_idle(const SubmitOptions & opts, F && f, Args &&... args)
  -> std::optional<std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>>
  {
    if (!idle() || poisoned() || stopped()) {
      return std::nullopt;
    }
    SubmitOptions o = opts;
    return std::make_optional(submit(o, std::forward<F>(f), std::forward<Args>(args)...));
  }

  // PORT: gpu_lanes.py:GpuLane.shutdown
  void shutdown(double timeout_s = 5.0);

  struct Stats
  {
    std::uint64_t jobs = 0;
    std::uint64_t failed = 0;
    double max_wait_s = 0.0;
    double max_run_s = 0.0;
  };
  Stats stats();

  // The stored poison exception message (empty when not poisoned).
  std::string poison_message() const;

private:
  template <class F, class... Args>
  auto submit_opts(const SubmitOptions & opts, F && f, Args &&... args)
  -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
  {
    using R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
    auto promise = std::make_shared<std::promise<R>>();
    std::future<R> fut = promise->get_future();

    auto fail = [promise](std::exception_ptr e) {
        try {
          promise->set_exception(e);
        } catch (...) {
        }
      };

    if (stopped()) {
      fail(std::make_exception_ptr(LaneStopped(name_)));
      return fut;
    }
    if (poisoned()) {
      fail(std::make_exception_ptr(LanePoisoned(poison_message())));
      return fut;
    }

    detail::Job job;
    job.prio = opts.prio;
    job.name = opts.name.empty() ? std::string("job") : opts.name;
    job.cancel = opts.cancel;
    job.fail = fail;
    job.t_submit = std::chrono::steady_clock::now();

    auto args_tuple = std::make_tuple(std::forward<Args>(args)...);
    if constexpr (std::is_void_v<R>) {
      auto holder = std::make_shared<char>(0);
      (void)holder;
      job.invoke = [f = std::forward<F>(f), args_tuple]() mutable {
          std::apply([&](auto &&... a) { std::invoke(f, a...); }, args_tuple);
        };
      job.commit = [promise]() {promise->set_value();};
    } else {
      auto result = std::make_shared<std::optional<R>>();
      job.invoke =
        [f = std::forward<F>(f), args_tuple, result]() mutable {
          *result = std::apply([&](auto &&... a) { return std::invoke(f, a...); }, args_tuple);
        };
      job.commit = [promise, result]() {promise->set_value(std::move(**result));};
    }

    if (owns_current_thread()) {
      // Called from inside the lane itself: run inline (no deadlock).
      execute_inline(job);
      return fut;
    }
    if (barrier_->holds_exclusive()) {
      throw DeadlockRisk(
        "thread holds the exclusive GPU barrier (legacy gpu_lock?) and tried to "
        "wait on lane '" + name_ + "'. Release it first or move this code into the lane.");
    }
    job.seq = next_seq();
    enqueue(std::move(job));
    return fut;
  }

  void enqueue(detail::Job job);
  void run();
  void run_job(detail::Job & job);
  void execute_inline(detail::Job & job);
  static bool is_fatal(const std::exception_ptr & e);
  static std::uint64_t next_seq();

  std::string name_;
  CaptureBarrier * barrier_;
  IBridge * bridge_;
  GpuLaneOptions options_;
  std::thread thread_;
  std::thread::id thread_id_;

  std::mutex q_mutex_;
  std::condition_variable q_cond_;
  std::priority_queue<detail::Job, std::vector<detail::Job>, detail::JobGreater> queue_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> busy_{false};
  // Set by run() on the stop sentinel; lets shutdown() bound its join
  // (std::thread has no timed join - PORT: gpu_lanes.py join(timeout)).
  bool worker_done_ = false;
  std::condition_variable done_cv_;

  mutable std::mutex poison_mutex_;
  std::atomic<bool> poisoned_{false};
  std::string poison_message_;

  std::mutex stats_mutex_;
  Stats stats_;

  std::promise<void> started_;
};

// PORT: gpu_lanes.py:GpuScheduler
class GpuScheduler
{
public:
  GpuScheduler();
  explicit GpuScheduler(IBridge * bridge);
  GpuScheduler(IBridge * bridge, std::function<void(GpuLane *, std::exception_ptr)> on_poisoned);
  GpuScheduler(const GpuScheduler &) = delete;
  GpuScheduler & operator=(const GpuScheduler &) = delete;
  ~GpuScheduler() {shutdown();}

  CaptureBarrier barrier;

  // PORT: gpu_lanes.py:GpuScheduler.lane
  GpuLane & lane(const std::string & name);
  GpuLane & lane(const std::string & name, GpuLaneOptions options);

  // PORT: gpu_lanes.py:GpuScheduler.exclusive
  CaptureBarrier::ExclusiveGuard exclusive() {return barrier.exclusive();}

  // PORT: gpu_lanes.py:GpuScheduler.legacy_lock
  LegacyGpuLock legacy_lock() {return LegacyGpuLock(barrier);}

  // PORT: gpu_lanes.py:GpuScheduler.any_poisoned
  bool any_poisoned() const;

  // PORT: gpu_lanes.py:GpuScheduler.stats
  std::unordered_map<std::string, GpuLane::Stats> stats();

  // PORT: gpu_lanes.py:GpuScheduler.shutdown
  void shutdown();

private:
  void install_sync_hook();

  IBridge * bridge_ = nullptr;
  std::function<void(GpuLane *, std::exception_ptr)> on_poisoned_;
  mutable std::mutex lanes_mutex_;
  std::vector<std::unique_ptr<GpuLane>> lanes_;
  std::unordered_map<std::string, GpuLane *> lane_by_name_;
};

}  // namespace curobo_runtime

#endif  // CUROBO_RUNTIME__GPU_LANES_HPP_
