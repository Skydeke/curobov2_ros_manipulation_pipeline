#!/usr/bin/env python3
"""Hammer the unified_planner node with concurrent plan / IK / FK calls.

UNTESTED against a live node when written (no ROS/GPU in the authoring
sandbox). Check field names with `ros2 interface show curobov2_ros_interfaces/srv/...`
if anything raises AttributeError.

Usage (node already running, joint states publishing):
  python3 concurrency_stress.py --seconds 60 --plan-threads 2 --ik-threads 2 --fk-threads 2
Exit code 0 = node survived and error rate below --max-error-rate; 1 otherwise.
Prints per-kind count / ok / p50 / p95 / max latency (ms).
"""
import argparse, random, statistics, sys, threading, time
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from curobov2_ros_interfaces.srv import FkBatch, IkBatch, TrajectoryGeneration
from curobov2_ros_interfaces.msg import Goalset, TrajectoryGoal

NODE = "unified_planner"


class Stress(Node):
    def __init__(self, a):
        super().__init__("concurrency_stress")
        self.a = a
        cg = ReentrantCallbackGroup()
        self.js = None
        self.create_subscription(JointState, a.joint_states_topic, self._js, 10, callback_group=cg)
        self.fk = self.create_client(FkBatch, f"{NODE}/fk_batch", callback_group=cg)
        self.ik = self.create_client(IkBatch, f"{NODE}/ik_batch", callback_group=cg)
        self.tg = self.create_client(TrajectoryGeneration, f"{NODE}/generate_trajectory", callback_group=cg)
        self.lock = threading.Lock()
        self.stats = {"plan": [], "ik": [], "fk": []}   # (ok, ms)
        self.base_poses = None
        self.stop = threading.Event()
        self.dead = False

    def _js(self, m):
        self.js = m

    def _call(self, cli, req, timeout):
        if not cli.service_is_ready():
            self.dead = True
            return None
        fut = cli.call_async(req)
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > timeout:
                return None
            time.sleep(0.002)
        return fut.result()

    def _rec(self, kind, ok, t0):
        with self.lock:
            self.stats[kind].append((ok, (time.monotonic() - t0) * 1e3))

    def _jittered_js(self, amp):
        m = JointState()
        m.name = list(self.js.name)
        m.position = [p + random.uniform(-amp, amp) for p in self.js.position]
        return m

    def setup(self):
        while self.js is None:
            time.sleep(0.1)
        r = FkBatch.Request(); r.joint_states = [self.js]
        for _ in range(100):
            if self.fk.wait_for_service(timeout_sec=0.5):
                break
        resp = self._call(self.fk, r, 30)
        if resp is None or not resp.success:
            sys.exit("FK warmup failed; is the node up and fk warmed?")
        self.base_poses = list(resp.poses)

    # ---- workers
    def w_fk(self):
        while not self.stop.is_set():
            n = random.choice([1, 8, 32])
            r = FkBatch.Request(); r.joint_states = [self._jittered_js(0.05) for _ in range(n)]
            t0 = time.monotonic(); resp = self._call(self.fk, r, 20)
            self._rec("fk", bool(resp and resp.success), t0)

    def w_ik(self):
        while not self.stop.is_set():
            n = random.choice([1, 4, 16, 32])   # batch-size changes force IK re-init: intentional
            r = IkBatch.Request(); r.poses = [self.base_poses[0] for _ in range(n)]
            for p in r.poses:
                p.position.x += random.uniform(-0.03, 0.03)
            t0 = time.monotonic(); resp = self._call(self.ik, r, 30)
            self._rec("ik", bool(resp and resp.success), t0)

    def w_plan(self):
        names = [n for n in self.js.name][: self.a.arm_dof]
        while not self.stop.is_set():
            start = JointState(); start.name = list(self.js.name); start.position = list(self.js.position)
            g = Goalset()
            tgt = [p + random.uniform(-0.4, 0.4) for p in self.js.position[: self.a.arm_dof]]
            g.target_joint_positions = tgt; g.target_joint_names = names
            tg = TrajectoryGoal(); tg.start_pose = start; tg.goalsets = [g]
            r = TrajectoryGeneration.Request(); r.request = tg
            t0 = time.monotonic(); resp = self._call(self.tg, r, 60)
            # "ok" = the server answered; planning failure on a random goal is legitimate.
            self._rec("plan", resp is not None, t0)

    def run(self):
        ws = ([self.w_plan] * self.a.plan_threads + [self.w_ik] * self.a.ik_threads
              + [self.w_fk] * self.a.fk_threads)
        ts = [threading.Thread(target=w, daemon=True) for w in ws]
        [t.start() for t in ts]
        t_end = time.monotonic() + self.a.seconds
        while time.monotonic() < t_end and not self.dead:
            time.sleep(0.5)
        self.stop.set(); [t.join(5) for t in ts]


def pct(v, q):
    return statistics.quantiles(v, n=100)[q - 1] if len(v) >= 2 else (v[0] if v else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--plan-threads", type=int, default=2)
    ap.add_argument("--ik-threads", type=int, default=2)
    ap.add_argument("--fk-threads", type=int, default=2)
    ap.add_argument("--arm-dof", type=int, default=7)
    ap.add_argument("--joint-states-topic", default="/joint_states")
    ap.add_argument("--max-error-rate", type=float, default=0.02)
    a = ap.parse_args()
    rclpy.init()
    n = Stress(a)
    ex = MultiThreadedExecutor(num_threads=16); ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    n.setup(); n.run()
    bad = n.dead
    print(f"node_alive={not n.dead}")
    for k, v in n.stats.items():
        if not v:
            continue
        lat = [ms for _, ms in v]; ok = sum(1 for o, _ in v if o)
        err = 1 - ok / len(v); bad |= err > a.max_error_rate
        print(f"{k:5s} n={len(v):5d} ok={ok:5d} err={err:5.1%} p50={pct(lat,50):8.1f} "
              f"p95={pct(lat,95):8.1f} max={max(lat):8.1f} ms  thr={len(v)/a.seconds:6.1f}/s")
    rclpy.shutdown()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
