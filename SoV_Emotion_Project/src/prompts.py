"""Stage 04: draw the visual prompts, build the text question, parse the model's answer."""
import json
import re

import cv2
import numpy as np

from . import config

PALETTE = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 200, 255), (200, 0, 200),
           (255, 160, 0), (0, 255, 255), (128, 0, 255), (0, 128, 255), (255, 0, 128)]

SMALL_DOT_VARIANTS = {"V8_sov_small_dots", "V9_landmarks_small_dots"}


def _landmark_radius(w, h, small):
    """Dot radius. Main study: ~1/16 of the face. Phase 3 'small dots': ~1/45, so eyes and mouth stay visible."""
    return max(2, int(round(min(w, h) / 45))) if small else max(3, int(round(min(w, h) / 16)))


def render(img, faces, variant):
    """Return a copy of img with the visual prompt for `variant` drawn on it."""
    out = img.copy()
    u = max(img.shape[:2]) / 800.0
    th = max(2, int(round(3 * u)))
    if variant == "V0_plain":
        return out
    if variant in ("V1_redcircle", "V2_blur_outside"):
        k = int(max(img.shape[:2]) / 25) | 1
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        mask = np.zeros(img.shape[:2], np.uint8)
        for f in faces:
            x, y, w, h = f["box"]
            cv2.ellipse(mask, (int(x + w / 2), int(y + h / 2)), (int(w * 0.7), int(h * 0.8)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 3 * u)[..., None] / 255.0
        out = (img * mask + blurred * (1 - mask)).astype(np.uint8)
        for f in faces:
            x, y, w, h = f["box"]
            if variant == "V1_redcircle":
                cv2.ellipse(out, (int(x + w / 2), int(y + h / 2)), (int(w * 0.65), int(h * 0.72)), 0, 0, 360, (0, 0, 255), th, cv2.LINE_AA)
            else:
                cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 255), th)
        return out
    no_box = ("V6_ids_only", "V7_landmarks_only", "V9_landmarks_small_dots")
    with_id = ("V4_boxes_ids", "V5_sov_full", "V6_ids_only", "V8_sov_small_dots")
    with_lm = ("V5_sov_full", "V7_landmarks_only", "V8_sov_small_dots", "V9_landmarks_small_dots")
    for f in faces:
        col = PALETTE[(f["id"] - 1) % len(PALETTE)]
        x, y, w, h = [int(round(v)) for v in f["box"]]
        if variant not in no_box:
            cv2.rectangle(out, (x, y), (x + w, y + h), col, th)
        if variant in with_id:
            txt = str(f["id"])
            fs = max(0.7, 0.9 * u * min(1.4, max(0.6, h / (120 * u))))
            tt = max(2, int(round(2 * u)))
            (tw, tth), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, tt)
            pad = max(3, int(4 * u))
            by1 = y
            by0 = by1 - tth - 2 * pad
            if by0 < 0:
                by0, by1 = y, y + tth + 2 * pad
            cv2.rectangle(out, (x, by0), (x + tw + 2 * pad, by1), col, -1)
            lum = 0.114 * col[0] + 0.587 * col[1] + 0.299 * col[2]
            cv2.putText(out, txt, (x + pad, by1 - pad - base // 2 + 1), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 0, 0) if lum > 150 else (255, 255, 255), tt, cv2.LINE_AA)
        if variant in with_lm:
            r = _landmark_radius(w, h, variant in SMALL_DOT_VARIANTS)
            for px, py in f.get("landmarks", {}).values():
                cv2.circle(out, (int(round(px)), int(round(py))), r, col, -1, cv2.LINE_AA)
                cv2.circle(out, (int(round(px)), int(round(py))), r, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_record(rec, variant):
    return render(cv2.imread(rec["path"]), rec["faces"], variant)


def encode_jpeg(img, quality=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------- text prompt (the standardized protocol)
# One fixed question; only the sentence describing the drawing changes between variants.
COMMON = ("You are an expert in facial expression analysis.\n"
          "Task: (1) count the human faces that are clearly visible in the image and (2) classify the facial "
          "emotion of every one of those faces.\n"
          "Allowed emotion labels (exactly one per face, lowercase): " + ", ".join(config.EMOTIONS) + ".\n")
HOW = {
    "V0_plain": "List the faces from left to right (by horizontal position in the image) and number them 1, 2, 3, ...",
    "V1_redcircle": "Every face is circled in red and the rest of the image is blurred. List the circled faces from left to right and number them 1, 2, 3, ...",
    "V2_blur_outside": "Every face is marked with a yellow box and the rest of the image is blurred. List the boxed faces from left to right and number them 1, 2, 3, ...",
    "V3_boxes": "Every face is marked with a coloured bounding box. List the boxed faces from left to right and number them 1, 2, 3, ...",
    "V4_boxes_ids": "Every face is marked with a coloured bounding box and a number tag. Use the number tag of each box as the face id.",
    "V5_sov_full": "Every face is marked with a coloured bounding box, a number tag and five small dots (eyes, nose, mouth corners). Use the number tag of each box as the face id.",
    "V6_ids_only": "Every face has a number tag on it. Use the number tag of each face as the face id.",
    "V7_landmarks_only": "Every face is marked with five small coloured dots (eyes, nose, mouth corners). List the marked faces from left to right and number them 1, 2, 3, ...",
    "V8_sov_small_dots": "Every face is marked with a coloured bounding box, a number tag and five tiny dots (eyes, nose, mouth corners). Use the number tag of each box as the face id.",
    "V9_landmarks_small_dots": "Every face is marked with five tiny coloured dots (eyes, nose, mouth corners). List the marked faces from left to right and number them 1, 2, 3, ...",
}
FORMAT = ('Answer ONLY with JSON in exactly this format, nothing else:\n'
          '{"num_faces": <integer>, "faces": [{"id": <integer>, "emotion": "<label>"}, ...]}')


def build_prompt(variant):
    return COMMON + HOW[variant] + "\n" + FORMAT


def parse_answer(text):
    """-> {'num_faces': int, 'faces': [(id, emotion), ...]} or None if unparsable."""
    if not isinstance(text, str):
        return None
    t = re.sub(r"```(?:json)?", "", text).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None
    raw = obj.get("faces")
    if not isinstance(raw, list):
        return None
    faces = []
    for k, f in enumerate(raw, 1):
        if not isinstance(f, dict):
            continue
        try:
            fid = int(f.get("id", k))
        except (TypeError, ValueError):
            fid = k
        faces.append((fid, config.canon_emotion(f.get("emotion"))))
    try:
        n = int(obj.get("num_faces", len(faces)))
    except (TypeError, ValueError):
        n = len(faces)
    return {"num_faces": n, "faces": faces}
