"""Stage 03: apply the manual label review and build the final dataset."""
import pandas as pd

from . import config


def build_dataset(records, corrections, drop=()):
    """records: output of stage 02. corrections: {image: {face_id: label}}. Returns (dataset, n_changed, n_faces)."""
    data, changed, total = [], 0, 0
    allowed = set(config.EMOTIONS) | {"ignore"}
    drop = set(drop)
    for rec in records:
        if rec["image"] in drop:
            continue
        fix = corrections.get(rec["image"], {})
        bad = {k: v for k, v in fix.items() if str(v).lower() not in allowed}
        if bad:
            raise ValueError(f"Unknown label(s) for {rec['image']}: {bad}. Allowed: {sorted(allowed)}")
        faces = []
        for f in sorted(rec["faces"], key=lambda f: f["box"][0]):
            lab = str(fix.get(f["id"], f["auto_emotion"])).lower()
            if lab == "ignore":
                continue
            faces.append({"orig_id": f["id"], "box": f["box"], "landmarks": f.get("landmarks", {}),
                          "emotion": lab, "auto_emotion": f["auto_emotion"]})
        if not faces:
            continue
        for i, f in enumerate(faces, 1):
            f["id"] = i
            total += 1
            changed += int(f["emotion"] != f["auto_emotion"])
        data.append({"image": rec["image"], "path": rec["path"], "width": rec["width"], "height": rec["height"],
                     "n_faces": len(faces), "difficulty": config.difficulty(len(faces)), "faces": faces})
    return data, changed, total


def dataset_stats(data):
    """Images/faces per difficulty and emotion distribution (DataFrames)."""
    rows = [{"image": r["image"], "difficulty": r["difficulty"], "emotion": f["emotion"]} for r in data for f in r["faces"]]
    df = pd.DataFrame(rows)
    per_diff = (df.groupby("difficulty").agg(images=("image", "nunique"), faces=("emotion", "size"))
                .reindex(config.DIFFICULTIES).fillna(0).astype(int))
    emo = (df.groupby(["difficulty", "emotion"]).size().unstack(fill_value=0)
           .reindex(index=config.DIFFICULTIES, columns=config.EMOTIONS, fill_value=0))
    emo.loc["All"] = emo.sum()
    return per_diff, emo


def eval_only_view(data):
    """Labels-only copy of the dataset (no boxes/paths) so evaluation can run without the photos."""
    return [{"image": r["image"], "n_faces": r["n_faces"], "difficulty": r["difficulty"],
             "faces": [{"id": f["id"], "emotion": f["emotion"]} for f in r["faces"]]} for r in data]
