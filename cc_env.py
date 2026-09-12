"""ns3-gym bridge + 6-feature layer + stability-aware reward.

Drives the rl-cc-train fork (examples/rl-cc-train), NOT the frozen pilot
config at examples/rl-cc. One ns-3 process per episode, spawned directly
from the compiled binary; ns3gym attaches to the already-listening port.
"""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ns3gym import ns3env
from paths import NS3, SIM_DIR, assert_fork, find_binary

assert_fork()
BINARY = str(find_binary())
_ENV = {**os.environ,
        "LD_LIBRARY_PATH": f"{NS3/'build'/'lib'}:{os.environ.get('LD_LIBRARY_PATH','')}"}

# ---- raw obs schema (order matters; must match tcp-rl-env.cc) --------------
SEG_SIZE, INTERVAL, RTT_MEAN, GRAD, MIN_RTT, BYTES_SENT, \
    BYTES_ACKED, PKTS_SENT, PKTS_ACKED, LOSS, CWND, TPUT = range(12)

FEATURES = ("latency_gradient", "latency_ratio", "sending_ratio",
            "cwnd_norm", "loss_rate", "tput_norm")

# ---- topology constants (Sec 3.3.2) ---------------------------------------
MTU_BYTES       = 1500    # -> MSS 1448 B
ACCESS_DELAY_MS = 0.01    # negligible: path RTT is set by the bottleneck leg
ACCESS_BW_MBPS  = 20.0    # must exceed max bottleneck bw (6 Mbps)
MAX_SEGS        = 1500    # MUST equal the C++ cwnd cap (--cwndCapSegs)
EP_STEPS        = 400     # timesteps per episode (Sec 3.3.5)


@dataclass
class Link:
    bw_mbps: float
    rtt_ms: float          # target end-to-end path RTT
    queue_pkts: int
    loss_rate: float

    @property
    def bottleneck_delay_ms(self) -> float:
        """One-way bottleneck delay giving the target path RTT.

        RTT = 4*access_delay + 2*bottleneck_delay (two access legs each way).
        """
        return max((self.rtt_ms - 4.0 * ACCESS_DELAY_MS) / 2.0, 0.01)


@dataclass
class RewardCfg:
    alpha: float = 1.0      # throughput weight
    beta: float = 1.0       # latency weight
    zeta: float = 10.0      # loss weight
    delta: float = 0.0      # variance weight  <- swept
    n_var: int = 10         # variance window  <- swept


# ---- domain randomization (Table 3.3.2 training ranges) --------------------
def sample_link(rng) -> Link:
    return Link(bw_mbps=float(rng.uniform(1.2, 6.0)),
                rtt_ms=float(rng.uniform(50.0, 500.0)),
                queue_pkts=int(rng.integers(2, 2982)),
                loss_rate=float(rng.uniform(0.0, 0.05)))


# Nominal hold-values: midpoints of bw / rtt / loss; queue at ~1 BDP
# (3.6 Mbps * 275 ms / 1448 B ~= 86 packets) so the six non-queue scenarios
# are not confounded by persistent buffer saturation.
NOMINAL = Link(bw_mbps=3.6, rtt_ms=275.0, queue_pkts=86, loss_rate=0.025)


def eval_scenarios() -> dict[str, Link]:
    """S = 8: low/high of each of the four link parameters, others nominal."""
    s = {}
    for name, field, val in [
        ("bw_low",   "bw_mbps",    1.2),  ("bw_high",   "bw_mbps",    6.0),
        ("lat_low",  "rtt_ms",    50.0),  ("lat_high",  "rtt_ms",   500.0),
        ("q_low",    "queue_pkts",  10),  ("q_high",    "queue_pkts", 1000),
        ("loss_low", "loss_rate",  0.0),  ("loss_high", "loss_rate", 0.05),
    ]:
        d = asdict(NOMINAL)
        d[field] = val
        s[name] = Link(**d)
    return s


class TcpCcEnv(gym.Env):
    """One ns-3 process per episode; link params resampled on every reset."""

    metadata = {"render_modes": []}

    def __init__(self, reward: RewardCfg, seed: int = 0, port: int = 5555,
                 fixed_link: Link | None = None, debug: bool = False):
        super().__init__()
        self.rw, self.fixed_link, self.debug = reward, fixed_link, debug
        self._port, self._seed = port, seed
        self._sim = None
        self._proc = None
        self._ep = -1
        self.link: Link | None = None

        self.observation_space = spaces.Box(
            low=np.array([-9.0, 1.0, 0.0, 0.0, 0.0, 0.0], np.float32),
            high=np.array([9.0, 10.0, 10.0, 1.0, 1.0, 1.0], np.float32),
            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

    # -- process lifecycle --------------------------------------------------
    def _close_sim(self):
        if self._sim is not None:
            with contextlib.suppress(Exception):
                self._sim.close()
            self._sim = None
        if self._proc is not None:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                self._proc.wait(timeout=5)
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            self._proc = None

    def _launch(self, link: Link, ep_run: int):
        duration = max(60.0, EP_STEPS * (link.rtt_ms / 1000.0) * 1.5 + 30.0)
        # seed folded into --run: sim.cc binds --simSeed and --run to the same
        # variable, so a distinct --run is the only way to give each (seed,
        # episode) an independent ns-3 RNG stream.
        argv = [BINARY,
                f"--openGymPort={self._port}",
                f"--run={self._seed * 100000 + ep_run}",
                "--transport_prot=TcpRlTimeBased",
                f"--bottleneck_bandwidth={link.bw_mbps:.4f}Mbps",
                f"--bottleneck_delay={link.bottleneck_delay_ms:.4f}ms",
                f"--access_bandwidth={ACCESS_BW_MBPS:.1f}Mbps",
                f"--access_delay={ACCESS_DELAY_MS}ms",
                f"--queueSize={link.queue_pkts}",
                f"--error_p={link.loss_rate:.5f}",
                f"--mtu={MTU_BYTES}",
                f"--duration={duration:.1f}",
                "--data=0",                            # unlimited transfer
                f"--cwndCapSegs={MAX_SEGS}"]

        self._proc = subprocess.Popen(
            argv, cwd=str(SIM_DIR), env=_ENV, start_new_session=True,
            stdout=None if self.debug else subprocess.DEVNULL,
            stderr=None if self.debug else subprocess.DEVNULL)

        # ns-3's ZMQ socket doesn't answer a bare TCP connect, so poll by
        # attempting the actual Ns3Env handshake. Check process liveness each
        # try so a crashed ns-3 errors fast instead of blocking forever.
        deadline = time.time() + 30.0
        last = None
        while True:
            if self._proc.poll() is not None:
                self._close_sim()
                raise RuntimeError(
                    f"ns-3 exited early (rc={self._proc.returncode}); "
                    f"rerun with debug=True to see its output")
            try:
                self._sim = ns3env.Ns3Env(port=self._port, stepTime=0.0,
                                          startSim=False, simSeed=self._seed,
                                          simArgs={}, debug=self.debug)
                return
            except Exception as e:
                last = e
                if time.time() > deadline:
                    self._close_sim()
                    raise RuntimeError(
                        f"could not attach on port {self._port}: {last}")
                time.sleep(0.2)

    # -- feature layer (12 raw -> 6 features, Sec 3.2.1) --------------------
    @staticmethod
    def _signed_log(x: float) -> float:
        """obs[3] spans roughly -5000..+100 ms/s; compress before the MLP."""
        return float(np.sign(x) * np.log1p(abs(x)))

    def _features(self, raw: np.ndarray) -> np.ndarray:
        seg = max(raw[SEG_SIZE], 1.0)
        min_rtt = max(raw[MIN_RTT], 1e-6)
        rtt = max(raw[RTT_MEAN], min_rtt)
        sent = max(raw[PKTS_SENT], 0.0)
        acked = max(raw[PKTS_ACKED], 0.0)

        # dead frame: watchdog fired with nothing in flight. Don't let it
        # move the running max used for throughput normalization.
        self._dead = (raw[BYTES_SENT] == 0.0 and acked == 0.0) or raw[RTT_MEAN] <= 0.0
        if not self._dead:
            self._tput_max = max(self._tput_max, float(raw[TPUT]))

        f = np.array([
            self._signed_log(raw[GRAD]),                          # ms/s
            np.clip(rtt / min_rtt, 1.0, 10.0),
            np.clip(sent / max(acked, 1.0), 0.0, 10.0),
            np.clip(raw[CWND] / (MAX_SEGS * seg), 0.0, 1.0),
            np.clip(max(raw[LOSS], 0.0) / max(sent, 1.0), 0.0, 1.0),
            np.clip(raw[TPUT] / max(self._tput_max, 1e-9), 0.0, 1.0),
        ], dtype=np.float32)
        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    # -- reward: R = aT - bD - zL - dV (Sec 3.2.2) --------------------------
    def _reward(self, raw: np.ndarray, feat: np.ndarray):
        min_rtt = max(raw[MIN_RTT], 1e-6)
        rtt = max(raw[RTT_MEAN], min_rtt)

        T = float(np.clip(raw[TPUT] / self.link.bw_mbps, 0.0, 1.5))  # utilization
        if raw[RTT_MEAN] <= 0.0:            # dead frame: no delay signal
            D = 0.0
        else:
            D = float(np.clip(1.0 - min_rtt / rtt, 0.0, 1.0))
        L = float(feat[4])

        self._hist.append(T)
        V = float(np.var(self._hist, ddof=1)) if len(self._hist) >= 2 else 0.0

        r = (self.rw.alpha * T - self.rw.beta * D
             - self.rw.zeta * L - self.rw.delta * V)
        return r, {"T": T, "D": D, "L": L, "V": V}

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._close_sim()
        self.link = self.fixed_link or sample_link(self.np_random)
        self._ep += 1

        self._t = 0
        self._tput_max = 1e-9
        self._dead = False
        self._hist = deque(maxlen=self.rw.n_var)

        self._launch(self.link, self._ep)
        out = self._sim.reset()
        raw = np.asarray(out[0] if isinstance(out, tuple) else out, np.float64)
        self._last = self._features(raw)
        return self._last, {"link": asdict(self.link)}

    def step(self, action):
        a = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
        out = self._sim.step(np.array([a], dtype=np.float32))
        if len(out) == 5:
            obs, _, term, trunc, _ = out
            done = bool(term or trunc)
        else:
            obs, _, done, _ = out

        self._t += 1
        if obs is None:                       # ns-3 ended before EP_STEPS
            return self._last, 0.0, True, False, {}

        raw = np.asarray(obs, np.float64)
        feat = self._features(raw)
        r, parts = self._reward(raw, feat)
        self._last = feat

        info = {"action": a, "reward": r, "watchdog": self._dead,
                "feat": feat.tolist(),          # raw 6 features, pre-normalization
                # full 12-field raw telemetry (Table 3.3.4 schema)
                "seg_size": float(raw[SEG_SIZE]),
                "interval_s": float(raw[INTERVAL]),
                "rtt_ms": float(raw[RTT_MEAN]),
                "grad_ms_s": float(raw[GRAD]),
                "min_rtt_ms": float(raw[MIN_RTT]),
                "bytes_sent": float(raw[BYTES_SENT]),
                "bytes_acked": float(raw[BYTES_ACKED]),
                "pkts_sent": float(raw[PKTS_SENT]),
                "pkts_acked": float(raw[PKTS_ACKED]),
                "loss_ct": float(raw[LOSS]),
                "cwnd_b": float(raw[CWND]),
                "tput_mbps": float(raw[TPUT]),
                        **parts}
        return feat, r, bool(done), self._t >= EP_STEPS, info

    def close(self):
        self._close_sim()

    def __del__(self):
        with contextlib.suppress(Exception):
            self._close_sim()
