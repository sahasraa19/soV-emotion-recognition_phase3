import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
"""Step 1 (CPU) -- sample a class-balanced set of FER-2013 faces.
Usage: python 01_prepare_fer.py --fer data/fer2013 --split test --per_class 60
Input : fer2013.csv  OR  Kaggle folders  data/fer2013/{train,test}/<emotion>/*.png
Output: data/phase2/fer_sample.csv  (emotion, pixels, kind)  + fer_sample_stats.txt
FER-2013 caveat: labels are noisy (humans agree on only ~65%); we use them as the reference label and say so in the report."""
import argparse
import cv2
from src import datasets, config

ap = argparse.ArgumentParser()
ap.add_argument("--fer", default=str(ROOT / "data/fer2013"))
ap.add_argument("--split", default="test", choices=["train", "test", "all"])
ap.add_argument("--per_class", type=int, default=60)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

df = datasets.load_fer(a.fer, a.split)
s = datasets.sample_balanced(df, a.per_class, a.seed)
s["pixels"] = [" ".join(map(str, datasets.fer_image(r).flatten())) for _, r in s.iterrows()]
s = s.assign(src=s.pixels, kind="pixels")[["emotion", "src", "kind"]]
out = ROOT / "data/phase2"; out.mkdir(parents=True, exist_ok=True)
s.to_csv(out / "fer_sample.csv", index=False)
txt = "FER-2013 sample (split=%s, per_class=%d, seed=%d)\n%s\n" % (a.split, a.per_class, a.seed, s.emotion.value_counts().reindex(config.EMOTIONS).to_string())
(out / "fer_sample_stats.txt").write_text(txt); print(txt)
