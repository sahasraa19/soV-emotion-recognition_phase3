import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
"""Step 2 (CPU) -- select WIDER FACE photos that contain 2-10 CLEAR faces (used as realistic backgrounds
and for the face-detector check in step 3).
Usage: python 02_prepare_wider.py --split_txt data/wider_face/wider_face_split/wider_face_val_bbx_gt.txt \
                                  --images data/wider_face/WIDER_val/images --n 100
Output: data/phase2/wider_selected.json  [{image, path, boxes}]"""
import argparse, random
from src import datasets, utils

ap = argparse.ArgumentParser()
ap.add_argument("--split_txt", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--min_face", type=int, default=32)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

sel = datasets.parse_wider_split(a.split_txt, a.images, a.min_face)
random.Random(a.seed).shuffle(sel)
sel = sel[:a.n]
out = ROOT / "data/phase2"; out.mkdir(parents=True, exist_ok=True)
utils.save_json(sel, out / "wider_selected.json")
print(f"selected {len(sel)} WIDER photos ->", out / "wider_selected.json")
