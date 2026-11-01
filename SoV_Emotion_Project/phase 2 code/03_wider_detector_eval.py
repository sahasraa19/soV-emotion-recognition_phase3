import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
"""Step 3 (CPU/GPU, needs `retina-face`) -- how good is the face detector that SoV depends on?
Runs our detector (src.detection: RetinaFace, YuNet fallback, min face 32 px, overlap removal) on the selected WIDER photos
and compares with the WIDER ground-truth boxes (IoU >= 0.5): recall, precision, face-count accuracy.
Output: results/phase2_detector_wider/detector_eval.csv + summary.md"""
import argparse
import cv2
import pandas as pd
from src import detection, utils

def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    i = iw * ih
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)

ap = argparse.ArgumentParser()
ap.add_argument("--selected", default=str(ROOT / "data/phase2/wider_selected.json"))
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
sel = utils.load_json(a.selected)
sel = sel[:a.limit] if a.limit else sel
rows = []
for r in utils.progress(sel, desc="detector vs WIDER"):
    rec = detection.process_image(r["path"], label=False)
    pred = rec["faces"] if rec else []
    gt, used, tp = r["boxes"], set(), 0
    for f in pred:
        best = max(range(len(gt)), key=lambda j: iou(f["box"], gt[j]) if j not in used else -1)
        if best not in used and iou(f["box"], gt[best]) >= 0.5:
            used.add(best); tp += 1
    rows.append({"image": r["image"], "gt": len(gt), "pred": len(pred), "tp": tp})
df = pd.DataFrame(rows)
out = ROOT / "results/phase2_detector_wider"; out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "detector_eval.csv", index=False)
rec_, pre = df.tp.sum() / df.gt.sum(), df.tp.sum() / max(1, df.pred.sum())
cnt = (df.gt == df.pred).mean()
txt = f"# Detector vs WIDER FACE ({len(df)} photos)\n- recall {rec_:.1%}\n- precision {pre:.1%}\n- exact face-count accuracy {cnt:.1%}\n"
(out / "summary.md").write_text(txt); print(txt)
