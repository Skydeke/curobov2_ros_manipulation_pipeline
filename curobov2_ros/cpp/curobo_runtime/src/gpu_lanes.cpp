// SPDX-License-Identifier: Apache-2.0
// 1:1 C++ port of curobov2_ros/core/gpu_lanes.py.
// PORT markers name the Python symbol each function copies.
#include "curobo_runtime/gpu_lanes.hpp"

#include <pthread.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <stdexcept>

namespace curobo_runtime
{

// PORT: gpu_lanes.py:_CUDA_FATAL_MARKERS
const std::vector<std::string> kCudaFatalMarkers = {
  "illegal memory access",
  "cudaerrorstreamcapture",
  "cuda error",
  "cuda_error",
  "device-side assert",
  "cugraphlaunch",
};

// PORT: threading.current_thread().name / the lane thread's name
std::string current_thread_name()
{
  char buf[64] = {0};
  if (pthread_getname_np(pthread_self(), buf, sizeof(buf)) != 0) {
    return {};
  }
  return std::string(buf);
}

void set_current_thread_name(const std::string & name)
{
  // Linux caps thread names at 16 bytes including the NUL.
  std::string truncated = name.substr(0, 15);
  pthread_setname_np(pthread_self(), truncated.c_str());
}

// --------------------------------------------------------------------------
// CaptureBarrier
// --------------------------------------------------------------------------

// Per-(thread, barrier) depths, mirroring Python's threading.local() instance
// attributes.
// PORT: gpu_lanes.py:CaptureBarrier._tl
CaptureBarrier::ThreadState & CaptureBarrier::tls(const CaptureBarrier * key)
{
  static thread_local std::unordered_map<const CaptureBarrier *, ThreadState> map;
  return map[key];
}

void CaptureBarrier::run_sync_hook()
{
  if (sync_hook) {
    sync_hook();
  }
}

// PORT: gpu_lanes.py:CaptureBarrier.acquire_shared
void CaptureBarrier::acquire_shared()
{
  if (tls(this).shared_depth > 0 || tls(this).excl_depth > 0) {  // nested
    tls(this).shared_depth += 1;
    return;
  }
  {
    std::unique_lock<std::mutex> lk(mutex_);
    cond_.wait(lk, [&] {return !writer_.has_value() && writers_waiting_ == 0;});
    readers_ += 1;
  }
  tls(this).shared_depth = 1;
}

// PORT: gpu_lanes.py:CaptureBarrier.release_shared
void CaptureBarrier::release_shared()
{
  int d = tls(this).shared_depth;
  if (d <= 0) {
    throw std::runtime_error("release_shared without acquire_shared");
  }
  tls(this).shared_depth = d - 1;
  if (d == 1 && tls(this).excl_depth == 0) {
    std::lock_guard<std::mutex> lk(mutex_);
    readers_ -= 1;
    cond_.notify_all();
  }
}

// PORT: gpu_lanes.py:CaptureBarrier.acquire_exclusive
bool CaptureBarrier::acquire_exclusive(bool blocking)
{
  if (tls(this).excl_depth > 0) {  // re-entrant
    tls(this).excl_depth += 1;
    return true;
  }
  const std::thread::id me = std::this_thread::get_id();
  const bool held_shared = tls(this).shared_depth > 0;
  {
    std::unique_lock<std::mutex> lk(mutex_);
    if (held_shared) {
      readers_ -= 1;  // drop our own read hold while we wait
    }
    if (!blocking && (writer_.has_value() || readers_ > 0)) {
      if (held_shared) {
        readers_ += 1;
      }
      return false;
    }
    writers_waiting_ += 1;
    try {
      cond_.wait(lk, [&] {return !writer_.has_value() && readers_ == 0;});
    } catch (...) {
      // PORT: gpu_lanes.py:acquire_exclusive's `finally: self._writers_waiting
      // -= 1`. std::condition_variable::wait can throw (std::system_error); a
      // leaked counter makes acquire_shared's predicate (`writers_waiting_ ==
      // 0`) permanently false, deadlocking every later shared acquisition.
      writers_waiting_ -= 1;
      if (held_shared) {
        readers_ += 1;  // re-assert the read hold we dropped while waiting
      }
      throw;
    }
    writers_waiting_ -= 1;
    writer_ = me;
  }
  tls(this).saved_shared = held_shared ? tls(this).shared_depth : 0;
  tls(this).shared_depth = 0;
  tls(this).excl_depth = 1;
  run_sync_hook();
  return true;
}

// PORT: gpu_lanes.py:CaptureBarrier.release_exclusive
void CaptureBarrier::release_exclusive()
{
  int d = tls(this).excl_depth;
  if (d <= 0) {
    throw std::runtime_error("release_exclusive without acquire_exclusive");
  }
  if (d == 1) {
    run_sync_hook();
  }
  tls(this).excl_depth = d - 1;
  if (d > 1) {
    return;
  }
  const int saved = tls(this).saved_shared;
  tls(this).saved_shared = 0;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    writer_.reset();
    if (saved > 0) {
      readers_ += 1;  // restore the read hold we dropped
    }
    cond_.notify_all();
  }
  tls(this).shared_depth = saved;
}

bool CaptureBarrier::holds_exclusive() const {return tls(this).excl_depth > 0;}
bool CaptureBarrier::holds_shared() const {return tls(this).shared_depth > 0;}

// --------------------------------------------------------------------------
// GpuLane
// --------------------------------------------------------------------------

namespace
{
std::string describe_exception(std::exception_ptr e)
{
  if (!e) {
    return {};
  }
  try {
    std::rethrow_exception(e);
  } catch (const std::exception & ex) {
    return ex.what();
  } catch (...) {
    return "unknown exception";
  }
}
}  // namespace

// PORT: gpu_lanes.py:GpuLane.__init__
GpuLane::GpuLane(
  std::string name, CaptureBarrier & barrier, IBridge * bridge, GpuLaneOptions options)
: name_(std::move(name)), barrier_(&barrier), bridge_(bridge), options_(std::move(options))
{
  auto started_fut = started_.get_future();
  thread_ = std::thread(&GpuLane::run, this);
  thread_id_ = thread_.get_id();
  started_fut.wait_for(std::chrono::seconds(10));
}

GpuLane::~GpuLane()
{
  if (thread_.joinable()) {
    shutdown();
  }
}

// PORT: gpu_lanes.py:GpuLane.assert_owner
void GpuLane::assert_owner(const std::string & what)
{
  if (!owns_current_thread()) {
    throw NotLaneOwner(
      (what.empty() ? std::string("GPU state") : what) + " owned by lane '" + name_ +
      "' was touched from thread '" + current_thread_name() + "'. Submit a job instead.");
  }
}

// PORT: gpu_lanes.py:GpuLane.idle
bool GpuLane::idle()
{
  if (busy_.load()) {
    return false;
  }
  std::lock_guard<std::mutex> lk(q_mutex_);
  return queue_.empty();
}

std::string GpuLane::poison_message() const
{
  std::lock_guard<std::mutex> lk(poison_mutex_);
  return poison_message_;
}

GpuLane::Stats GpuLane::stats()
{
  std::lock_guard<std::mutex> lk(stats_mutex_);
  return stats_;
}

std::uint64_t GpuLane::next_seq()
{
  static std::atomic<std::uint64_t> seq{0};
  return seq.fetch_add(1);
}

void GpuLane::enqueue(detail::Job job)
{
  {
    std::lock_guard<std::mutex> lk(q_mutex_);
    queue_.push(std::move(job));
  }
  q_cond_.notify_one();
}

// PORT: gpu_lanes.py:GpuLane.shutdown
void GpuLane::shutdown(double timeout_s)
{
  const bool was_stopped = stop_.exchange(true);
  if (!was_stopped) {
    detail::Job sentinel;
    sentinel.prio = 1000000000;
    sentinel.name = "stop";
    sentinel.invoke = []() {};
    sentinel.commit = []() {};
    sentinel.fail = [](std::exception_ptr) {};
    enqueue(std::move(sentinel));
  }
  if (thread_.joinable() && std::this_thread::get_id() != thread_id_) {
    // PORT: gpu_lanes.py's `self._thread.join(timeout)` - a lane wedged inside a
    // job (e.g. blocked on the capture barrier) must never hang teardown.
    // std::thread cannot express a timed join, so wait on the worker's
    // completion flag, then join (immediate once it has fired).
    bool finished = true;
    if (timeout_s > 0.0) {
      std::unique_lock<std::mutex> lk(q_mutex_);
      finished = done_cv_.wait_for(
        lk, std::chrono::duration<double>(timeout_s), [&] {return worker_done_;});
    }
    if (!finished) {
      std::fprintf(
        stderr,
        "lane '%s': shutdown timed out after %.1fs; leaving the worker thread "
        "running (it was blocked inside a job)\n", name_.c_str(), timeout_s);
      thread_.detach();
      return;
    }
    thread_.join();
  }
  // fail anything left
  while (true) {
    detail::Job job;
    {
      std::unique_lock<std::mutex> lk(q_mutex_);
      if (queue_.empty()) {
        break;
      }
      job = queue_.top();
      queue_.pop();
    }
    job.fail(std::make_exception_ptr(LaneStopped(name_)));
  }
}

// PORT: gpu_lanes.py:_is_fatal (marker scan over a lower-cased message)
bool is_cuda_fatal_message(const std::string & message)
{
  std::string msg = message;
  std::transform(
    msg.begin(), msg.end(), msg.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  for (const auto & marker : kCudaFatalMarkers) {
    if (msg.find(marker) != std::string::npos) {
      return true;
    }
  }
  return false;
}

// PORT: gpu_lanes.py:GpuLane._is_fatal
bool GpuLane::is_fatal(const std::exception_ptr & e)
{
  if (!e) {
    return false;
  }
  try {
    std::rethrow_exception(e);
  } catch (const std::runtime_error & re) {
    return is_cuda_fatal_message(re.what());
  } catch (...) {
    return false;
  }
}

// PORT: gpu_lanes.py:GpuLane._execute_inline
void GpuLane::execute_inline(detail::Job & job)
{
  try {
    job.invoke();
    job.commit();
  } catch (...) {
    job.fail(std::current_exception());
  }
}

// PORT: gpu_lanes.py:GpuLane._run
void GpuLane::run()
{
  set_current_thread_name("gpu-lane-" + name_);
  if (options_.use_cuda_stream && bridge_ != nullptr) {
    try {
      bridge_->bind_lane(name_);
    } catch (...) {
      // PORT: "lane %s: no CUDA stream (%s)" -> continue without a stream.
    }
  }
  started_.set_value();
  while (true) {
    detail::Job job;
    {
      std::unique_lock<std::mutex> lk(q_mutex_);
      q_cond_.wait(lk, [&] {return !queue_.empty();});
      job = queue_.top();
      queue_.pop();
    }
    if (stop_.load() && job.prio >= 1000000000) {
      {
        std::lock_guard<std::mutex> lk(q_mutex_);
        worker_done_ = true;
      }
      done_cv_.notify_all();
      return;
    }
    if (stop_.load()) {
      job.fail(std::make_exception_ptr(LaneStopped(name_)));
      continue;
    }
    run_job(job);
  }
}

// PORT: gpu_lanes.py:GpuLane._run_job
void GpuLane::run_job(detail::Job & job)
{
  if (poisoned()) {
    job.fail(std::make_exception_ptr(LanePoisoned(poison_message())));
    return;
  }
  if (job.cancel && job.cancel->is_set()) {
    job.fail(std::make_exception_ptr(JobCancelled(job.name)));
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lk(stats_mutex_);
    const double waited =
      std::chrono::duration<double>(t0 - job.t_submit).count();
    stats_.max_wait_s = std::max(stats_.max_wait_s, waited);
  }
  busy_.store(true);
  // PORT: gpu_lanes.py:GpuLane._run_job wraps the body in `try/except` and does
  // `finally: self._busy = False; stats[...] = ...`. Keeping that tail OUTSIDE
  // the failure handler matters: if the handler itself throws (poison message
  // formatting, a throwing on_poisoned), busy_ would otherwise stay true and
  // `submit_if_idle` would drop every later job for this lane - permanently
  // starving perception/viz.
  try {
    try {
      {
        auto shared = barrier_->shared();
        job.invoke();
        if (options_.use_cuda_stream && bridge_ != nullptr) {
          // Finish this lane's GPU work BEFORE leaving the shared side.
          bridge_->sync_lane();
        }
      }
      job.commit();
    } catch (...) {
      const std::exception_ptr e = std::current_exception();
      {
        std::lock_guard<std::mutex> lk(stats_mutex_);
        stats_.failed += 1;
      }
      if (is_fatal(e)) {
        {
          std::lock_guard<std::mutex> lk(poison_mutex_);
          poisoned_.store(true);
          poison_message_ = "lane '" + name_ + "': " + describe_exception(e);
        }
        if (options_.on_poisoned) {
          try {
            options_.on_poisoned(this, e);
          } catch (...) {
          }
        }
      }
      job.fail(e);
    }
  } catch (...) {
    // The failure path itself threw. Report and fall through to the `finally`
    // below, exactly as the Python's unconditional tail would.
    std::fprintf(stderr, "lane '%s': failure handling threw\n", name_.c_str());
  }
  busy_.store(false);
  {
    std::lock_guard<std::mutex> lk(stats_mutex_);
    stats_.jobs += 1;
    const double ran = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stats_.max_run_s = std::max(stats_.max_run_s, ran);
  }
}

// --------------------------------------------------------------------------
// GpuScheduler
// --------------------------------------------------------------------------

// PORT: gpu_lanes.py:_cuda_device_sync (through the bridge; no-op on a CPU box)
void GpuScheduler::install_sync_hook()
{
  if (bridge_ != nullptr) {
    barrier.sync_hook = [this]() {bridge_->device_sync();};
  } else {
    barrier.sync_hook = []() {};
  }
}

GpuScheduler::GpuScheduler()
{
  install_sync_hook();
}

GpuScheduler::GpuScheduler(IBridge * bridge)
: bridge_(bridge)
{
  install_sync_hook();
}

GpuScheduler::GpuScheduler(
  IBridge * bridge, std::function<void(GpuLane *, std::exception_ptr)> on_poisoned)
: bridge_(bridge), on_poisoned_(std::move(on_poisoned))
{
  install_sync_hook();
}

// PORT: gpu_lanes.py:GpuScheduler.lane
GpuLane & GpuScheduler::lane(const std::string & name)
{
  return lane(name, GpuLaneOptions{});
}

GpuLane & GpuScheduler::lane(const std::string & name, GpuLaneOptions options)
{
  std::lock_guard<std::mutex> lk(lanes_mutex_);
  auto it = lane_by_name_.find(name);
  if (it != lane_by_name_.end()) {
    return *it->second;
  }
  if (!options.on_poisoned && on_poisoned_) {
    options.on_poisoned = on_poisoned_;
  }
  auto lane_ptr = std::make_unique<GpuLane>(name, barrier, bridge_, std::move(options));
  GpuLane * raw = lane_ptr.get();
  lanes_.push_back(std::move(lane_ptr));
  lane_by_name_[name] = raw;
  return *raw;
}

// PORT: gpu_lanes.py:GpuScheduler.any_poisoned
bool GpuScheduler::any_poisoned() const
{
  std::lock_guard<std::mutex> lk(lanes_mutex_);
  for (const auto & lane : lanes_) {
    if (lane->poisoned()) {
      return true;
    }
  }
  return false;
}

// PORT: gpu_lanes.py:GpuScheduler.stats
std::unordered_map<std::string, GpuLane::Stats> GpuScheduler::stats()
{
  std::lock_guard<std::mutex> lk(lanes_mutex_);
  std::unordered_map<std::string, GpuLane::Stats> out;
  for (const auto & lane : lanes_) {
    out[lane->name()] = lane->stats();
  }
  return out;
}

// PORT: gpu_lanes.py:GpuScheduler.shutdown
void GpuScheduler::shutdown()
{
  std::vector<GpuLane *> lanes;
  {
    std::lock_guard<std::mutex> lk(lanes_mutex_);
    for (const auto & lane : lanes_) {
      lanes.push_back(lane.get());
    }
  }
  for (GpuLane * lane : lanes) {
    lane->shutdown();
  }
}

}  // namespace curobo_runtime
