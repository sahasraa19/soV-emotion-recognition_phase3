import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
"""Phase 1, step 1 (CPU) -- resize your own photos (data/raw) to max side 1280 px JPEG -> data/images_proc."""
from src import data
names = data.preprocess_images(ROOT / "data/raw")
print(len(names), "images prepared")
