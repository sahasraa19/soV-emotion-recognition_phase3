import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
"""Phase 1, step 2 (CPU/GPU) -- detect faces (RetinaFace, YuNet fallback), drop faces < 32 px, remove overlaps
(paper Algorithm 1, threshold 0.5), number faces left->right, first-guess emotion with DeepFace.
Output: data/faces_auto.json"""
from src import config, detection, utils
records, skipped = detection.detect_all()
utils.save_json(records, config.FACES_JSON)
print(len(records), "images with faces; skipped:", skipped)
