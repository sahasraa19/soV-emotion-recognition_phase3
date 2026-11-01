"""Shared constants, paths and tiny helpers used by every stage."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                      # original photos go here
IMG_DIR = DATA_DIR / "images_proc"              # resized photos (stage 01)
FACES_JSON = DATA_DIR / "faces_auto.json"       # detections + DeepFace guesses (stage 02)
CORRECTIONS_JSON = DATA_DIR / "labels_corrections.json"   # manual label fixes (stage 03)
DATASET_JSON = DATA_DIR / "dataset.json"        # final dataset with boxes + landmarks (stage 03)
DATASET_EVAL_JSON = DATA_DIR / "dataset_eval_only.json"   # labels only (no boxes): lets stages 06-07 run without photos
RESULTS_DIR = ROOT / "results"

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]

# The 8 variants of the main study (reproduces the saved results) ...
VARIANTS = ["V0_plain", "V1_redcircle", "V2_blur_outside", "V3_boxes", "V4_boxes_ids",
            "V5_sov_full", "V6_ids_only", "V7_landmarks_only"]
# ... and the 2 extra variants of the Phase 3 landmark ablation (same cues, much smaller dots).
EXT_VARIANTS = ["V8_sov_small_dots", "V9_landmarks_small_dots"]
ALL_VARIANTS = VARIANTS + EXT_VARIANTS

VARIANT_TITLES = {
    "V0_plain": "V0 Plain (baseline)",
    "V1_redcircle": "V1 RedCircle-style (circle + blur)",
    "V2_blur_outside": "V2 ReCLIP-style (box + blur)",
    "V3_boxes": "V3 Boxes only",
    "V4_boxes_ids": "V4 Boxes + numbers",
    "V5_sov_full": "V5 Full SoV (boxes+numbers+landmarks)",
    "V6_ids_only": "V6 Numbers only",
    "V7_landmarks_only": "V7 Landmarks only",
    "V8_sov_small_dots": "V8 Full SoV, small dots",
    "V9_landmarks_small_dots": "V9 Landmarks only, small dots",
}
# variants whose answers are matched to ground truth by the number tag drawn on the image
ID_VARIANTS = {"V4_boxes_ids", "V5_sov_full", "V6_ids_only", "V8_sov_small_dots"}

MIN_FACE = 32            # ignore detected faces smaller than this many pixels
OVERLAP_THRESHOLD = 0.5  # paper Algorithm 1
MAX_SIDE = 1280
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

MODEL_NAMES = {
    "qwen": "Qwen/Qwen2-VL-7B-Instruct",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "gemini": "gemini-2.5-flash",
    "mock": "mock",
}

SYNONYMS = {"anger": "angry", "mad": "angry", "annoyed": "angry", "disgusted": "disgust", "fearful": "fear",
            "afraid": "fear", "scared": "fear", "anxious": "fear", "happiness": "happy", "joy": "happy",
            "joyful": "happy", "smiling": "happy", "sadness": "sad", "unhappy": "sad", "surprised": "surprise",
            "shocked": "surprise", "calm": "neutral"}


def difficulty(n_faces):
    """Paper Appendix A.1 splits by number of faces. We use Easy 1-3, Medium 4-7, Hard 8+."""
    return "Easy" if n_faces <= 3 else ("Medium" if n_faces <= 7 else "Hard")


def canon_emotion(x):
    """Map free text to one of the 7 labels, or 'invalid'."""
    if not isinstance(x, str):
        return "invalid"
    x = x.strip().lower()
    x = SYNONYMS.get(x, x)
    return x if x in EMOTIONS else "invalid"


def run_name(provider):
    """Folder name for a provider's results, e.g. 'Qwen_Qwen2-VL-7B-Instruct'."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", MODEL_NAMES[provider])


def results_dir(provider, make=True):
    d = RESULTS_DIR / run_name(provider)
    if make:
        (d / "figures").mkdir(parents=True, exist_ok=True)
    return d
