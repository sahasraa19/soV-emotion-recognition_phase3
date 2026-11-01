import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
"""Step 6 (CPU) -- metrics, bootstrap CIs vs the plain image, per-class recall, figures.
Usage: python 06_evaluate.py --model qwen
Output: results/phase2_fer_wider_<model>/*.csv + results_summary.md ; analysis_folders/analysis_phase2_<model>/ (figures, tables)"""
import argparse
import matplotlib; matplotlib.use("Agg")
from src import config, evaluation, plots, utils

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mock")
ap.add_argument("--dataset", default=str(ROOT / "data/phase2/dataset_eval_only.json"))
a = ap.parse_args()
res_dir = ROOT / "results" / f"phase2_fer_wider_{a.model}"
ana = ROOT / "analysis_folders" / f"analysis_phase2_{a.model}"; (ana / "figures").mkdir(parents=True, exist_ok=True)
raw = utils.load_raw_outputs(res_dir / "raw_outputs.jsonl")
variants = [v for v in config.ALL_VARIANTS if any(k[1] == v for k in raw)]
res = evaluation.evaluate(utils.load_json(a.dataset), raw, variants)
M = res["M"]
for d in (res_dir, ana):
    M.to_csv(d / "metrics.csv", index=False)
    res["per_class"].to_csv(d / "per_class_recall.csv")
    if len(res["ci"]): res["ci"].to_csv(d / "bootstrap_vs_baseline.csv", index=False)
    res["faces"].to_csv(d / "per_face_predictions.csv", index=False)
(res_dir / "results_summary.md").write_text(evaluation.summary_markdown(res, variants, res_dir.name))
(ana / "summary.md").write_text(evaluation.summary_markdown(res, variants, res_dir.name))
fd = ana / "figures"
plots.accuracy_by_variant(M, variants, res["maj"], res["maj_class"], fig_dir=fd)
plots.per_class(res["per_class"], variants, fig_dir=fd)
if "V5_sov_full" in variants and "V0_plain" in variants:
    plots.confusion(res["faces"], fig_dir=fd)
print((ana / "summary.md").read_text())
