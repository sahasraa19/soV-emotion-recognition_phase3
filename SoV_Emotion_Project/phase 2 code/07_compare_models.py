import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
"""Step 7 (CPU) -- compare models (e.g. Qwen2-VL vs LLaVA-1.5) and the paper's claim.
Usage: python 07_compare_models.py --models qwen,llava
Output: analysis_folders/analysis_phase2_compare/accuracy_by_model.csv + sov_minus_plain.csv"""
import argparse
import pandas as pd
from src import config, plots

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="qwen,llava")
a = ap.parse_args()
by = {}
for m in a.models.split(","):
    f = ROOT / "analysis_folders" / f"analysis_phase2_{m}" / "metrics.csv"
    if f.exists(): by[m] = pd.read_csv(f)
    else: print("missing", f)
vs = [v for v in config.ALL_VARIANTS if all(v in M.variant.values for M in by.values())]
out = ROOT / "analysis_folders/analysis_phase2_compare"; out.mkdir(parents=True, exist_ok=True)
t = plots.cross_model_table(by, vs); t.to_csv(out / "accuracy_by_model.csv", index=False); print(t.to_string(index=False))
d = {m: (M[(M.variant == "V5_sov_full") & (M.difficulty == "All")].accuracy.iloc[0] - M[(M.variant == "V0_plain") & (M.difficulty == "All")].accuracy.iloc[0]) * 100 for m, M in by.items()
     if {"V0_plain", "V5_sov_full"} <= set(M.variant)}
pd.Series(d, name="SoV minus plain (accuracy, points)").to_csv(out / "sov_minus_plain.csv"); print(d)
