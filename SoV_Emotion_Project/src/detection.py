"""Stage 02: face detection, overlap removal (paper Algorithm 1) and DeepFace first-guess labels."""
import urllib.request
from pathlib import Path

import cv2

from . import config, utils

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_YUNET = {"net": None}
DETECTOR = {"name": None}   # 'retinaface' or 'yunet' once the first image has been processed


def _detect_retina(img, thr):
    from retinaface import RetinaFace
    out = RetinaFace.detect_faces(img, threshold=thr)
    faces = []
    if not isinstance(out, dict):
        return faces
    for v in out.values():
        x1, y1, x2, y2 = [int(round(t)) for t in v["facial_area"]]
        lm = {k: [float(p[0]), float(p[1])] for k, p in v.get("landmarks", {}).items()}
        faces.append({"box": [x1, y1, x2 - x1, y2 - y1], "score": float(v.get("score", 1.0)), "landmarks": lm})
    return faces


def _detect_yunet(img, thr, model_dir=config.DATA_DIR):
    if _YUNET["net"] is None:
        mp = Path(model_dir) / "yunet.onnx"
        if not mp.exists():
            urllib.request.urlretrieve(YUNET_URL, str(mp))
        _YUNET["net"] = cv2.FaceDetectorYN.create(str(mp), "", (320, 320), thr, 0.3, 5000)
    H, W = img.shape[:2]
    _YUNET["net"].setInputSize((W, H))
    _, res = _YUNET["net"].detect(img)
    faces = []
    if res is not None:
        names = ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]
        for r in res:
            x, y, w, h = [int(round(float(t))) for t in r[:4]]
            lm = {n: [float(r[4 + 2 * i]), float(r[5 + 2 * i])] for i, n in enumerate(names)}
            faces.append({"box": [x, y, w, h], "score": float(r[14]), "landmarks": lm})
    return faces


def detect_faces(img, thr=0.9):
    """Detect faces with RetinaFace (paper), falling back to OpenCV YuNet if RetinaFace cannot run."""
    if DETECTOR["name"] in (None, "retinaface"):
        try:
            faces = _detect_retina(img, thr)
            DETECTOR["name"] = "retinaface"
        except Exception as e:
            print(f"RetinaFace failed ({type(e).__name__}: {str(e)[:150]}) -> using OpenCV YuNet instead (say so in the report)")
            DETECTOR["name"] = "yunet"
            faces = _detect_yunet(img, thr)
    else:
        faces = _detect_yunet(img, thr)
    H, W = img.shape[:2]
    out = []
    for f in faces:
        x, y, w, h = f["box"]
        x2, y2 = min(W, x + w), min(H, y + h)
        x, y = max(0, x), max(0, y)
        if x2 - x > 0 and y2 - y > 0:
            f["box"] = [x, y, x2 - x, y2 - y]
            out.append(f)
    return out


def remove_overlaps(faces, eps=config.OVERLAP_THRESHOLD):
    """Paper Algorithm 1: go from the largest to the smallest face and drop a face if
    intersection / min(area) with an already kept face exceeds eps."""
    kept = []
    for f in sorted(faces, key=lambda f: f["box"][2] * f["box"][3], reverse=True):
        x, y, w, h = f["box"]
        drop = False
        for g in kept:
            gx, gy, gw, gh = g["box"]
            iw = max(0, min(x + w, gx + gw) - max(x, gx))
            ih = max(0, min(y + h, gy + gh) - max(y, gy))
            if iw * ih / max(1, min(w * h, gw * gh)) > eps:
                drop = True
                break
        if not drop:
            kept.append(f)
    return kept


_DF_WARNED = {"x": False}


def guess_emotion(img, box, margin=0.2):
    """DeepFace first-guess label for one face (paper: DeepFace pre-annotation, then human revision)."""
    try:
        from deepface import DeepFace
        H, W = img.shape[:2]
        x, y, w, h = box
        mx, my = int(w * margin), int(h * margin)
        crop = img[max(0, y - my): min(H, y + h + my), max(0, x - mx): min(W, x + w + mx)]
        res = DeepFace.analyze(crop, actions=["emotion"], detector_backend="skip", enforce_detection=False, silent=True)
        if isinstance(res, list):
            res = res[0]
        lab = config.canon_emotion(res["dominant_emotion"])
        return lab if lab != "invalid" else "neutral"
    except Exception as e:
        if not _DF_WARNED["x"]:
            print(f"DeepFace failed ({type(e).__name__}: {str(e)[:150]}) -> every auto label becomes 'neutral'; label by hand.")
            _DF_WARNED["x"] = True
        return "neutral"


def process_image(path, min_face=config.MIN_FACE, overlap=config.OVERLAP_THRESHOLD, label=True):
    """Detect, filter, de-overlap and number the faces (left to right) of one image. Returns a record or None."""
    img = cv2.imread(str(path))
    H, W = img.shape[:2]
    faces = [f for f in detect_faces(img) if min(f["box"][2], f["box"][3]) >= min_face]
    faces = remove_overlaps(faces, overlap)
    faces.sort(key=lambda f: f["box"][0])
    if not faces:
        return None
    for i, f in enumerate(faces, 1):
        f["id"] = i
        f["auto_emotion"] = guess_emotion(img, f["box"]) if label else "neutral"
    return {"image": Path(path).name, "path": str(path), "width": W, "height": H, "n_faces": len(faces), "faces": faces}


def detect_all(img_dir=config.IMG_DIR, **kw):
    """Run process_image over every JPEG in img_dir. Returns (records, names of images without faces)."""
    records, skipped = [], []
    for p in utils.progress(sorted(Path(img_dir).glob("*.jpg")), desc="detecting + labelling"):
        rec = process_image(p, **kw)
        if rec is None:
            skipped.append(p.name)
        else:
            records.append(rec)
    return records, skipped
