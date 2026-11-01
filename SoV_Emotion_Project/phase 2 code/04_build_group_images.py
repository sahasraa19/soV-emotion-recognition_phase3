import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
"""Step 4 (CPU) -- build the Phase-2 evaluation set: multi-face images made from FER-2013 faces.
Each image holds 1-8 faces placed on a canvas (WIDER photo, blurred, as background when available).
Boxes, landmarks and emotion labels are KNOWN, so no detector or manual labelling is needed.
Usage: python 04_build_group_images.py --n_images 60 --min_faces 2 --max_faces 8
Output: data/phase2/images/*.jpg, data/phase2/dataset.json, data/phase2/dataset_eval_only.json, dataset_stats.md"""
import argparse, random
import pandas as pd
from src import composites, config, labeling, utils

ap = argparse.ArgumentParser()
ap.add_argument("--fer_sample", default=str(ROOT / "data/phase2/fer_sample.csv"))
ap.add_argument("--wider", default=str(ROOT / "data/phase2/wider_selected.json"))
ap.add_argument("--n_images", type=int, default=60)
ap.add_argument("--min_faces", type=int, default=2)
ap.add_argument("--max_faces", type=int, default=8)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

rng = random.Random(a.seed)
fer = pd.read_csv(a.fer_sample)
wider = utils.load_json(a.wider) if (ROOT / "data/phase2/wider_selected.json").exists() else []
img_dir = ROOT / "data/phase2/images"; img_dir.mkdir(parents=True, exist_ok=True)
pool = list(fer.index); rng.shuffle(pool)
data, i = [], 0
while len(data) < a.n_images:
    k = rng.randint(a.min_faces, a.max_faces)
    if len(pool) < k:                       # refill: each face is used once per pass
        pool = list(fer.index); rng.shuffle(pool)
    rows, pool = pool[:k], pool[k:]
    bg = rng.choice(wider)["path"] if wider else None
    data.append(composites.make_group_image(rows, fer, img_dir / f"group{len(data)+1:03d}.jpg", rng, bg))
utils.save_json(data, ROOT / "data/phase2/dataset.json")
utils.save_json(labeling.eval_only_view(data), ROOT / "data/phase2/dataset_eval_only.json")
st = labeling.dataset_stats(data)
txt = "# Phase-2 dataset\n\nImages: %d, faces: %d\n\n%s\n" % (len(data), sum(r["n_faces"] for r in data), "\n\n".join(s.to_string() for s in (st if isinstance(st, tuple) else (st,))))
(ROOT / "data/phase2/dataset_stats.md").write_text(txt); print(txt)
