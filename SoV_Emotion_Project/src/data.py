"""Stage 01: copy and resize the photos."""
import re

import cv2

from . import config


def preprocess_images(src_dir, out_dir=config.IMG_DIR, max_side=config.MAX_SIDE):
    """Copy every photo under src_dir into out_dir as JPEG with the longest side <= max_side. Returns names."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for p in sorted(Path(src_dir).rglob("*")):
        if p.suffix.lower() not in config.EXTS or "__MACOSX" in p.parts or p.name.startswith("."):
            continue
        img = cv2.imread(str(p))
        if img is None:
            print("cannot read, skipped:", p.name)
            continue
        H, W = img.shape[:2]
        s = min(1.0, max_side / max(H, W))
        if s < 1.0:
            img = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        base = re.sub(r"[^A-Za-z0-9._-]", "_", p.stem)
        name, k = base + ".jpg", 1
        while (out_dir / name).exists():
            name = f"{base}_{k}.jpg"
            k += 1
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        names.append(name)
    return names
