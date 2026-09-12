"""Single source of truth for the training/validation/testing phase."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("THESIS_ROOT", Path.home() / "thesis-cc"))
NS3 = ROOT / "ns-allinone-3.40" / "ns-3.40"
EXP = ROOT / "experiments"

SIM_DIR = NS3 / "contrib" / "opengym" / "examples" / "rl-cc-train"
PROGRAM = SIM_DIR.name

DATA_TRAIN = EXP / "data" / "train"
DATA_EVAL  = EXP / "data" / "eval"
DATA_SWEEP = EXP / "data" / "sweep"
MODELS     = EXP / "models"
TABLES     = EXP / "results" / "tables"
SHAP_OUT   = EXP / "results" / "shap"
STATS      = EXP / "results" / "stats"
MONITOR    = EXP / "logs" / "monitor"
TB         = EXP / "logs" / "tb"
CONFIGS    = EXP / "configs"

PILOT_DATA = ROOT / "data" / "pilot"   # read-only reference

for _d in (DATA_TRAIN, DATA_EVAL, DATA_SWEEP, MODELS,
           TABLES, SHAP_OUT, STATS, MONITOR, TB, CONFIGS):
    _d.mkdir(parents=True, exist_ok=True)


def assert_fork():
    if not SIM_DIR.exists():
        raise RuntimeError(f"{SIM_DIR} missing — run setup_experiments.sh")
    if (SIM_DIR / "FROZEN.md").exists():
        raise RuntimeError("SIM_DIR points at the frozen pilot config")


def find_binary() -> Path:
    """Locate the built rl-cc-train executable."""
    hits = sorted((NS3 / "build").rglob(f"*{PROGRAM}*"))
    hits = [h for h in hits if h.is_file() and os.access(h, os.X_OK)]
    if not hits:
        raise RuntimeError(f"no built binary for {PROGRAM}; run ./ns3 build {PROGRAM}")
    return hits[0]
