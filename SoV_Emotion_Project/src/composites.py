"""Build multi-face test images with KNOWN boxes and labels by placing FER-2013 faces on a canvas
(optionally over a blurred WIDER photo). No manual labelling and no face detector needed."""
import random

import cv2
import numpy as np

from . import config
from .datasets import LM_TEMPLATE, fer_image


def _background(W, H, bg_path, rng):
    if bg_path:
        bg = cv2.imread(str(bg_path))
        if bg is not None:
            s = max(W / bg.shape[1], H / bg.shape[0])
            bg = cv2.resize(bg, None, fx=s, fy=s)
            y0, x0 = rng.randint(0, bg.shape[0] - H), rng.randint(0, bg.shape[1] - W)
            bg = cv2.GaussianBlur(bg[y0:y0 + H, x0:x0 + W], (0, 0), 9)
            return (bg * 0.55 + 90 * 0.45).astype(np.uint8)
    g = rng.randint(110, 170)
    return np.full((H, W, 3), g, np.uint8)


def make_group_image(rows, fer_df, out_path, rng, bg_path=None, face_px=(150, 210)):
    """rows = indices into fer_df. Returns a dataset record (image path, boxes, 5 landmarks, labels, ids left->right)."""
    n = len(rows)
    per_row = n if n <= 4 else (n + 1) // 2
    nrows = 1 if n <= 4 else 2
    W, H = 1280, 400 if nrows == 1 else 760
    canvas = _background(W, H, bg_path, rng)
    placed = []
    for k, ri in enumerate(rows):
        r, c = divmod(k, per_row)
        in_row = per_row if r == 0 else n - per_row
        s = rng.randint(*face_px)
        slot = W / (in_row + 0.0)
        cx = slot * (c + 0.5) + (45 if r == 1 else 0) * (1 if in_row < per_row else 0) + rng.randint(-15, 15)
        cy = (H / 2 if nrows == 1 else H * (0.27 + 0.46 * r)) + rng.randint(-15, 15)
        x = int(np.clip(cx - s / 2, 5, W - s - 5)); y = int(np.clip(cy - s / 2, 5, H - s - 5))
        face = cv2.resize(fer_image(fer_df.loc[ri]), (s, s), interpolation=cv2.INTER_CUBIC)
        canvas[y:y + s, x:x + s] = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        placed.append({"box": [x, y, s, s], "emotion": fer_df.loc[ri, "emotion"], "fer_row": int(ri),
                       "landmarks": {k2: [x + fx * s, y + fy * s] for k2, (fx, fy) in LM_TEMPLATE.items()}})
    placed.sort(key=lambda f: (f["box"][0], f["box"][1]))        # ids go left -> right
    for i, f in enumerate(placed, 1):
        f["id"] = i
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"image": out_path.name, "path": str(out_path), "width": W, "height": H, "n_faces": n,
            "difficulty": config.difficulty(n), "faces": placed}
