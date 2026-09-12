"""CUBIC / BBR baselines through the same fork topology (Sec 3.4.2).

No gym agent: ns-3 runs the baseline transport and logs per-interval telemetry
via the sim.cc hook. Metrics use the same time-weighted, capacity-clipped
formula as the RL variants (evaluate.episode_metrics), so proposed / vanilla /
cubic / bbr are directly comparable.
"""
import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from cc_env import (
    ACCESS_BW_MBPS,
    ACCESS_DELAY_MS,
    EP_STEPS,
    MTU_BYTES,
    Link,
    eval_scenarios,
)
from paths import DATA_EVAL, NS3, SIM_DIR, TABLES, find_binary

BINARY = str(find_binary())
_ENV = {**os.environ,
        "LD_LIBRARY_PATH": f"{NS3/'build'/'lib'}:{os.environ.get('LD_LIBRARY_PATH','')}"}
TRANSPORT = {"cubic": "TcpCubic", "bbr": "TcpBbr"}


def run_one(variant: str, link: Link, seed: int, episode: int, out_csv: Path):
    duration = max(60.0, EP_STEPS * (link.rtt_ms / 1000.0) * 1.5 + 30.0)
    argv = [BINARY,
            f"--run={seed * 100000 + episode}",
            f"--transport_prot={TRANSPORT[variant]}",
            f"--bottleneck_bandwidth={link.bw_mbps:.4f}Mbps",
            f"--bottleneck_delay={link.bottleneck_delay_ms:.4f}ms",
            f"--access_bandwidth={ACCESS_BW_MBPS:.1f}Mbps",
            f"--access_delay={ACCESS_DELAY_MS}ms",
            f"--queueSize={link.queue_pkts}",
            f"--error_p={link.loss_rate:.5f}",
            f"--mtu={MTU_BYTES}",
            f"--duration={duration:.1f}",
            "--data=0",
            f"--telemetryOut={out_csv}"]
    subprocess.run(argv, cwd=str(SIM_DIR), env=_ENV, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def metrics_from_csv(path: Path, cap_mbps: float | None = None) -> dict:
    """Time-weighted metrics with capacity clipping, matching
    evaluate.episode_metrics so RL and baseline variants are comparable.

    Throughput samples above cap_mbps are ack-batching artifacts (a batch of
    acks closing a short interval); clipping to the link rate keeps the variance
    metric on genuine instability rather than finite-interval division noise.
    """
    df = pd.read_csv(path)
    df = df[df["rtt_ms"] > 0].reset_index(drop=True)
    if len(df) < 2:
        return dict(mean_throughput_mbps=0.0, throughput_variance_mbps2=0.0,
                    jitter_ms=0.0, packet_loss_pct=0.0)
    w = df["interval_s"].to_numpy()
    tput = df["tput_mbps"].to_numpy()
    if cap_mbps is not None:
        tput = np.clip(tput, 0.0, cap_mbps)
    rtt = df["rtt_ms"].to_numpy()
    mean_tput = float(np.average(tput, weights=w))
    var_tput = float(np.average((tput - mean_tput) ** 2, weights=w))
    sent = float(df["pkts_sent"].sum())
    lost = float(df["loss_ct"].sum())
    return {
        "mean_throughput_mbps": mean_tput,
        "throughput_variance_mbps2": var_tput,
        "jitter_ms": float(np.mean(np.abs(np.diff(rtt)))),
        "packet_loss_pct": float(100.0 * lost / max(sent, 1.0)),
    }


def evaluate_baseline(variant: str, seed: int, n_episodes: int) -> pd.DataFrame:
    recs = []
    for name, link in eval_scenarios().items():
        for ep in range(n_episodes):
            tmp = DATA_EVAL / f"{variant}_seed{seed}_{name}_ep{ep}_raw.csv"
            run_one(variant, link, seed, ep, tmp)
            m = metrics_from_csv(tmp, cap_mbps=link.bw_mbps)
            recs.append({"variant": variant, "seed": seed,
                         "scenario": name, "episode": ep, **m})
    return pd.DataFrame(recs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["cubic", "bbr"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-episodes", type=int, default=10)
    a = p.parse_args()

    df = evaluate_baseline(a.variant, a.seed, a.n_episodes)
    out = TABLES / f"metrics_{a.variant}_seed{a.seed}.csv"
    df.to_csv(out, index=False)
    print(df.groupby("scenario")[
        ["mean_throughput_mbps", "throughput_variance_mbps2",
         "jitter_ms", "packet_loss_pct"]].mean().round(3).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
