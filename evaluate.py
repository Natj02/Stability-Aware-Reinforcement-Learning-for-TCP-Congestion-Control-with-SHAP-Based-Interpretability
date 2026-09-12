"""Deterministic evaluation over the 8 held-out scenarios (Sec 3.4).

Produces, per (variant, seed, scenario):
  - a per-interval telemetry CSV (12 raw fields + 6 features + reward parts),
    the Sec 3.3.4 schema and the source for SHAP (Sec 3.4.5)
  - one row of episode-level metrics (Sec 3.4.3)
Baselines (CUBIC, BBR) run through baselines.py; metrics are computed with the
same time-weighted, capacity-clipped formula here so all four variants are
directly comparable.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from cc_env import EP_STEPS, FEATURES, Link, RewardCfg, TcpCcEnv, eval_scenarios
from paths import DATA_EVAL, MODELS, TABLES
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

RAW_FIELDS = ["seg_size", "interval_s", "rtt_ms", "grad_ms_s", "min_rtt_ms",
              "bytes_sent", "bytes_acked", "pkts_sent", "pkts_acked",
              "loss_ct", "cwnd_b", "tput_mbps"]
FEAT_FIELDS = [f"f_{n}" for n in FEATURES]
PART_FIELDS = ["reward", "T", "D", "L", "V", "action", "watchdog"]


def episode_metrics(rows: list[dict], cap_mbps: float | None = None) -> dict:
    """Four Sec 3.4.3 metrics, time-weighted over per-interval telemetry.

    Throughput samples exceeding the bottleneck capacity (cap_mbps) are
    ack-batching artifacts: when a batch of acknowledgments closes a very short
    monitoring interval, bytes/interval overshoots the physically achievable
    rate. These are clipped to the link capacity before aggregation so the
    variance metric reflects genuine throughput instability rather than
    finite-interval division noise. Dead (watchdog) frames are excluded.

    Averaging matches baselines.metrics_from_csv so RL and baseline variants
    are directly comparable (Sec 3.4.2).
    """
    live = [r for r in rows if not r.get("watchdog", False)]
    if len(live) < 2:
        return dict(mean_throughput_mbps=0.0, throughput_variance_mbps2=0.0,
                    jitter_ms=0.0, packet_loss_pct=0.0)
    w = np.array([r.get("interval_s", 0.0) for r in live], float)
    tput = np.array([r["tput_mbps"] for r in live], float)
    if cap_mbps is not None:
        tput = np.clip(tput, 0.0, cap_mbps)
    rtt = np.array([r["rtt_ms"] for r in live], float)
    sent = float(np.sum([r["pkts_sent"] for r in live]))
    lost = float(np.sum([r["loss_ct"] for r in live]))
    if w.sum() <= 0:                       # interval_s missing/zero: fall back
        w = np.ones_like(w)
    mean_tput = float(np.average(tput, weights=w))
    var_tput = float(np.average((tput - mean_tput) ** 2, weights=w))
    return {
        "mean_throughput_mbps": mean_tput,
        "throughput_variance_mbps2": var_tput,
        "jitter_ms": float(np.mean(np.abs(np.diff(rtt)))) if len(rtt) > 1 else 0.0,
        "packet_loss_pct": float(100.0 * lost / max(sent, 1.0)),
    }


def run_episode(model, venv, telem_writer, meta) -> dict:
    """One deterministic episode; stream telemetry, return metric row."""
    obs = venv.reset()
    rows, done, t = [], False, 0
    while not done and t < EP_STEPS:
        if model is not None:
            act, _ = model.predict(obs, deterministic=True)
        else:
            act = np.zeros((1, 1), np.float32)     # baseline: agent inert
        obs, _, dones, infos = venv.step(act)
        done = bool(dones[0])
        t += 1
        info = infos[0]
        if "reward" not in info:
            continue
        rows.append(info)
        if telem_writer:
            telem_writer.writerow(
                [meta["variant"], meta["seed"], meta["scenario"],
                 meta["episode"], t]
                + [info.get(k, "") for k in RAW_FIELDS]
                + list(info.get("feat", [""] * 6))
                + [info.get("reward", ""), info.get("T", ""), info.get("D", ""),
                   info.get("L", ""), info.get("V", ""), info.get("action", ""),
                   int(info.get("watchdog", False))])
    return episode_metrics(rows, cap_mbps=meta.get("cap_mbps"))


def make_venv(link: Link, rw: RewardCfg, seed: int, port: int,
              vecnorm_path: Path | None):
    venv = DummyVecEnv([lambda: TcpCcEnv(reward=rw, seed=seed, port=port,
                                         fixed_link=link)])
    if vecnorm_path and vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
    return venv


def evaluate(variant: str, model_dir: Path | None, seed: int,
             rw: RewardCfg, n_episodes: int, port: int) -> pd.DataFrame:
    model = None
    vecnorm = None
    if model_dir is not None:
        model = PPO.load(str(model_dir / "final"), device="cpu")
        vecnorm = model_dir / "vecnorm.pkl"

    telem_path = DATA_EVAL / f"{variant}_seed{seed}_telemetry.csv"
    with open(telem_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "seed", "scenario", "episode", "step"]
                   + RAW_FIELDS + FEAT_FIELDS + PART_FIELDS)
        recs = []
        for name, link in eval_scenarios().items():
            venv = make_venv(link, rw, seed, port, vecnorm)
            for ep in range(n_episodes):
                meta = dict(variant=variant, seed=seed, scenario=name,
                            episode=ep, cap_mbps=link.bw_mbps)
                m = run_episode(model, venv, w, meta)
                recs.append({"variant": variant, "seed": seed,
                             "scenario": name, "episode": ep, **m})
            venv.close()
    return pd.DataFrame(recs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   help="proposed | vanilla | cubic | bbr")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--n-var", type=int, default=10)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--port", type=int, default=5600)
    p.add_argument("--model-dir", default=None,
                   help="models/<tag>; omit for cubic/bbr baselines")
    a = p.parse_args()

    rw = RewardCfg(delta=a.delta, n_var=a.n_var)
    model_dir = Path(a.model_dir) if a.model_dir else None
    df = evaluate(a.variant, model_dir, a.seed, rw, a.n_episodes, a.port)

    out = TABLES / f"metrics_{a.variant}_seed{a.seed}.csv"
    df.to_csv(out, index=False)
    print(df.groupby("scenario")[
        ["mean_throughput_mbps", "throughput_variance_mbps2",
         "jitter_ms", "packet_loss_pct"]].mean().round(3).to_string())
    print(f"\nmetrics -> {out}")
    print(f"telemetry -> {DATA_EVAL / f'{a.variant}_seed{a.seed}_telemetry.csv'}")


if __name__ == "__main__":
    main()
