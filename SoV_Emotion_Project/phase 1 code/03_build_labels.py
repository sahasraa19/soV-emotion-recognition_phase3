import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
"""Phase 1, step 3 (CPU) -- apply manual label corrections (data/labels_corrections.json) -> data/dataset.json."""
from src import config, labeling, utils
rec = utils.load_json(config.FACES_JSON)
corr = utils.load_corrections()
data, changed, total = labeling.build_dataset(rec, corr)
utils.save_json(data, config.DATASET_JSON)
utils.save_json(labeling.eval_only_view(data), config.DATASET_EVAL_JSON)
print(f"{len(data)} images, {total} faces, {changed} labels corrected")
