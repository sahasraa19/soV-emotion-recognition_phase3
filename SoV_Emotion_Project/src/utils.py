"""Small I/O helpers shared by the notebooks."""
import json
from pathlib import Path

from . import config


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))


def load_json(path):
    return json.loads(Path(path).read_text())


def load_corrections(path=config.CORRECTIONS_JSON):
    """Corrections file format: {"image10.jpg": {"1": "angry"}, ...}; also {"drop_images": [...]}.
    Returns (corrections with int face ids, list of dropped images)."""
    p = Path(path)
    if not p.exists():
        return {}, []
    raw = load_json(p)
    drop = raw.pop("drop_images", [])
    return {img: {int(k): v for k, v in fixes.items()} for img, fixes in raw.items()}, drop


def load_dataset(prefer_full=True):
    """Final dataset: full version (with boxes) if present, else the labels-only version."""
    if prefer_full and config.DATASET_JSON.exists():
        return load_json(config.DATASET_JSON)
    if config.DATASET_EVAL_JSON.exists():
        return load_json(config.DATASET_EVAL_JSON)
    raise FileNotFoundError("No dataset found. Run notebooks 01-03 first (or provide data/dataset_eval_only.json).")


def load_raw_outputs(raw_path):
    """Latest answer per (image, variant) from a raw_outputs.jsonl file."""
    last = {}
    for line in Path(raw_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[(r["image"], r["variant"])] = r
    return last


def progress(iterable, desc=""):
    """tqdm progress bar if installed (it is in Colab), otherwise the plain iterable."""
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc)
    except ImportError:
        return iterable
