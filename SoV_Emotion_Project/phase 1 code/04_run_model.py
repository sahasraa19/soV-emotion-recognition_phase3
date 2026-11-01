import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
"""Phase 1, step 4 (GPU) -- run a VLM on your own photos with all 10 visual-prompt variants.
Usage: python 04_run_model.py --model qwen|llava|gemini|mock"""
import argparse
from src import config, models, utils
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="mock", choices=sorted(config.MODEL_NAMES))
a = ap.parse_args()
models.run_experiment(utils.load_dataset(), a.model, variants=config.ALL_VARIANTS)
