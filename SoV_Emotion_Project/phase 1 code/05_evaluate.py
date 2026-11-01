import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
"""Phase 1, step 5 (CPU) -- metrics + figures for your own photos. Usage: python 05_evaluate.py --model qwen
(The saved Qwen2-VL-7B result for the original 17 photos is in results/phase1_own_photos_qwen2vl7b.)"""
import argparse
import matplotlib; matplotlib.use("Agg")
from src import config, evaluation, plots, utils
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen"); a = ap.parse_args()
d = config.results_dir(a.model)
raw = utils.load_raw_outputs(d / "raw_outputs.jsonl")
V = [v for v in config.ALL_VARIANTS if any(k[1] == v for k in raw)]
res = evaluation.evaluate(utils.load_dataset(), raw, V)
res["M"].to_csv(d / "metrics.csv", index=False)
(d / "results_summary.md").write_text(evaluation.summary_markdown(res, V, d.name))
plots.accuracy_by_variant(res["M"], V, res["maj"], res["maj_class"], fig_dir=d / "figures")
print((d / "results_summary.md").read_text())
