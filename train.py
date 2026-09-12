"""PPO training, one seed per invocation. δ and N are pure Python config."""
import argparse
import csv
import json
from pathlib import Path

from cc_env import RewardCfg, TcpCcEnv
from paths import DATA_TRAIN, MODELS, MONITOR, TB
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

BASE_PORT = 5555

# raw telemetry pulled from info (subset of the 12 ns-3 fields most useful offline)
RAW_COLS = ["tput_mbps", "rtt_ms", "cwnd_b", "loss_ct", "pkts_sent"]
FEAT_COLS = ["f_lat_grad", "f_lat_ratio", "f_send_ratio",
             "f_cwnd_norm", "f_loss_rate", "f_tput_norm"]
REWARD_COLS = ["reward", "T", "D", "L", "V", "action", "watchdog"]


class TelemetryCSV(BaseCallback):
    """Append per-step raw telemetry + 6 features + reward decomposition to CSV.

    Off unless --log-telemetry. At full scale this file is very large; intended
    for short diagnostic runs, not the 10-seed production run.
    """
    def __init__(self, path, verbose=0):
        super().__init__(verbose)
        self.path = str(path)
        self._fh = None
        self._w = None

    def _on_training_start(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(["step", "env"] + RAW_COLS + FEAT_COLS + REWARD_COLS)

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals["infos"]):
            if "reward" not in info:            # skip reset/empty frames
                continue
            raw = [info.get(c, "") for c in RAW_COLS]
            feat = info.get("feat", [""] * 6)
            rew = [info.get("reward", ""), info.get("T", ""), info.get("D", ""),
                   info.get("L", ""), info.get("V", ""), info.get("action", ""),
                   int(info.get("watchdog", False))]
            self._w.writerow([self.num_timesteps, i] + raw + list(feat) + rew)
        return True

    def _on_training_end(self):
        if self._fh:
            self._fh.close()


def make_env(rank, seed, rw, log_dir):
    def _init():
        # distinct port per (seed, rank) so parallel workers never collide
        env = TcpCcEnv(reward=rw, seed=seed * 1000 + rank,
                       port=BASE_PORT + seed * 100 + rank)
        return Monitor(env, str(Path(log_dir) / f"monitor_{rank}"))
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--delta", type=float, required=True)
    p.add_argument("--n-var", type=int, default=10)
    p.add_argument("--timesteps", type=int, default=4_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--out", default=None)   # defaults to MODELS/<tag>
    p.add_argument("--log-telemetry", action="store_true",
                   help="write per-step telemetry CSV (large; off by default)")
    a = p.parse_args()

    tag = f"delta{a.delta:g}_N{a.n_var}_seed{a.seed}"
    out = Path(a.out) if a.out else (MODELS / tag)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = MONITOR / tag
    log_dir.mkdir(parents=True, exist_ok=True)

    rw = RewardCfg(delta=a.delta, n_var=a.n_var)
    venv = SubprocVecEnv([make_env(i, a.seed, rw, log_dir) for i in range(a.n_envs)])
    # normalize OBSERVATIONS only. Normalizing reward would rescale delta and
    # break the sensitivity analysis, which compares reward magnitudes across delta.
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", venv,
        n_steps=1024 // a.n_envs,   # 1024 transitions per policy update (Table 3.2.3)
        batch_size=256,             # minibatch within the update
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=a.seed, verbose=1, device="cpu",   # 64x64 MLP: CPU faster than GPU
        tensorboard_log=str(TB / tag),
    )

    callbacks = [CheckpointCallback(
        save_freq=max(200_000 // a.n_envs, 1),
        save_path=str(out / "ckpt"), name_prefix="ppo")]
    if a.log_telemetry:
        callbacks.append(TelemetryCSV(DATA_TRAIN / f"{tag}_telemetry.csv"))

    model.learn(total_timesteps=a.timesteps, callback=callbacks)

    model.save(out / "final")
    venv.save(str(out / "vecnorm.pkl"))
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    venv.close()
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
