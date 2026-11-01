import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
"""Step 5 (GPU for qwen/llava; mock needs none) -- ask the model about every (image, variant).
Safe to stop and re-run: finished pairs are skipped.
Colab:  !pip install -q -U transformers accelerate bitsandbytes qwen-vl-utils
        !python "phase 2 code/05_run_model.py" --model qwen --variants all
Variants: core = V0 plain, V3 boxes, V4 boxes+ids, V5 full SoV, V6 ids only, V7 landmarks only
          all  = core + V1, V2 (baselines) + V8, V9 (small-dot ablation)"""
import argparse
from src import config, models, utils

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mock", choices=sorted(config.MODEL_NAMES))
ap.add_argument("--variants", default="all", help="core | all | comma list")
ap.add_argument("--dataset", default=str(ROOT / "data/phase2/dataset.json"))
ap.add_argument("--limit_images", type=int, default=None)
a = ap.parse_args()
V = {"core": ["V0_plain", "V3_boxes", "V4_boxes_ids", "V5_sov_full", "V6_ids_only", "V7_landmarks_only"],
     "all": config.ALL_VARIANTS}.get(a.variants) or a.variants.split(",")
out = ROOT / "results" / f"phase2_fer_wider_{a.model}"; out.mkdir(parents=True, exist_ok=True)
models.run_experiment(utils.load_json(a.dataset), a.model, variants=V, raw_path=out / "raw_outputs.jsonl", limit_images=a.limit_images)
