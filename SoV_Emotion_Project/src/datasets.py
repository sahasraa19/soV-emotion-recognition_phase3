"""Dataset loaders for FER-2013 (emotion-labelled 48x48 faces) and WIDER FACE (many faces + boxes)."""
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config

FER_ID2EMO = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
# 5-point template for a tight, roughly aligned FER crop (x, y as fractions of the face box)
LM_TEMPLATE = {"left_eye": (0.30, 0.38), "right_eye": (0.70, 0.38), "nose": (0.50, 0.60),
               "mouth_left": (0.33, 0.78), "mouth_right": (0.67, 0.78)}


def load_fer(path, split="test"):
    """Return a DataFrame [emotion, pixels(np.uint8 48x48 or path), split] from either layout:
    (a) fer2013.csv  (emotion, pixels, Usage)   (b) Kaggle folders  <path>/{train,test}/<emotion>/*.jpg|png"""
    path = Path(path)
    csv = path if path.suffix == ".csv" else next(iter(path.glob("*.csv")), None)
    if csv is not None and csv.exists():
        df = pd.read_csv(csv)
        if "Usage" in df:
            keep = {"train": ["Training"], "test": ["PublicTest", "PrivateTest"], "all": df.Usage.unique()}[split]
            df = df[df.Usage.isin(keep)]
        df = df.reset_index(drop=True)
        return pd.DataFrame({"emotion": df.emotion.map(FER_ID2EMO), "src": df.pixels, "kind": "pixels"})
    rows = []
    for sp in (["train", "test"] if split == "all" else [split]):
        for d in sorted((path / sp).glob("*")):
            if d.is_dir() and d.name.lower() in FER_ID2EMO.values():
                for p in sorted(d.glob("*")):
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        rows.append({"emotion": d.name.lower(), "src": str(p), "kind": "file"})
    if not rows:
        raise FileNotFoundError(f"No FER-2013 data found under {path} (expected fer2013.csv or train/test/<emotion>/ folders)")
    return pd.DataFrame(rows)


def fer_image(row):
    """48x48 uint8 grayscale array of one FER row."""
    if row["kind"] == "pixels":
        return np.array(row["src"].split(), dtype=np.uint8).reshape(48, 48)
    g = cv2.imread(row["src"], cv2.IMREAD_GRAYSCALE)
    return cv2.resize(g, (48, 48))


def sample_balanced(df, per_class, seed=42):
    """per_class faces for each of the 7 emotions (fewer if the class is smaller, e.g. disgust)."""
    rng = random.Random(seed)
    out = []
    for emo in config.EMOTIONS:
        idx = list(df.index[df.emotion == emo])
        rng.shuffle(idx)
        out += idx[:per_class]
    return df.loc[out].sample(frac=1, random_state=seed).reset_index(drop=True)


def parse_wider_split(txt_path, images_dir, min_face=32, min_n=2, max_n=10):
    """Parse wider_face_{train,val}_bbx_gt.txt. Keeps clear faces only (blur=0, invalid=0, occlusion<=1, size>=min_face).
    Returns [{'image','path','boxes':[[x,y,w,h],...]}] for images with min_n..max_n kept faces AND no other (unusable) faces."""
    lines = Path(txt_path).read_text().splitlines()
    i, out = 0, []
    while i < len(lines):
        name = lines[i].strip()
        n = int(lines[i + 1]); i += 2
        rows = [lines[i + k].split() for k in range(max(n, 1))]
        i += max(n, 1)
        good, bad = [], 0
        for r in rows:
            if n == 0:
                break
            x, y, w, h, blur, expr, illum, invalid, occ, pose = [int(t) for t in r[:10]]
            if blur == 0 and invalid == 0 and occ <= 1 and min(w, h) >= min_face:
                good.append([x, y, w, h])
            else:
                bad += 1
        if min_n <= len(good) <= max_n and bad == 0:
            p = next(iter(Path(images_dir).rglob(Path(name).name)), None)
            if p is not None:
                out.append({"image": Path(name).name, "path": str(p), "boxes": good})
    return out
