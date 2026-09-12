import glob
import numpy as np
import pandas as pd
from paths import TABLES

for variant in ["proposed", "vanilla", "cubic", "bbr"]:
    fs = glob.glob(str(TABLES / f"metrics_{variant}_seed*.csv"))
    if not fs:
        print(f"{variant:10s} NO FILES FOUND at {TABLES}")
        continue
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["cv_ep"] = np.sqrt(df["throughput_variance_mbps2"]) / df["mean_throughput_mbps"]
    print(f"{variant:10s} aggregate CV = {df['cv_ep'].mean():.3f}")
