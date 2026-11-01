"""Stage 06: match model answers to ground truth and compute all metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from . import config

NONE = "none"


def match(rec, variant, parsed):
    """Pair every ground-truth face with the model's answer for it. Returns ([(truth, pred)], n_faces_predicted).
    Numbered variants: answer with id k <-> ground-truth face k. Others: k-th answer <-> k-th face from the left.
    A face with no answer, an invalid label or an unparsable answer counts as wrong ('none')."""
    gt = rec["faces"]
    if not parsed:
        return [(f["emotion"], NONE) for f in gt], -1
    preds = parsed["faces"]
    if variant in config.ID_VARIANTS:
        by_id = {}
        for fid, emo in preds:
            by_id.setdefault(int(fid), emo)
        pairs = [(f["emotion"], by_id.get(f["id"], NONE)) for f in gt]
    else:
        pairs = [(f["emotion"], preds[i][1] if i < len(preds) else NONE) for i, f in enumerate(gt)]
    return [(g, NONE if p == "invalid" else p) for g, p in pairs], parsed["num_faces"]


def build_tables(data, raw, variants):
    rf, ri = [], []
    for rec in data:
        for v in variants:
            r = raw.get((rec["image"], v))
            parsed = r["parsed"] if r and r["status"] == "ok" else None
            pairs, n_pred = match(rec, v, parsed)
            for fid, (g, p) in enumerate(pairs, 1):
                rf.append({"image": rec["image"], "variant": v, "difficulty": rec["difficulty"], "face_id": fid,
                           "truth": g, "pred": p, "correct": int(g == p)})
            ri.append({"image": rec["image"], "variant": v, "difficulty": rec["difficulty"], "n_gt": rec["n_faces"],
                       "n_pred": n_pred, "count_ok": int(n_pred == rec["n_faces"]),
                       "n_correct": sum(g == p for g, p in pairs)})
    return pd.DataFrame(rf), pd.DataFrame(ri)


def compute_metrics(faces, imgs, variants):
    out = []
    for v in variants:
        for d in config.DIFFICULTIES + ["All"]:
            f = faces[(faces.variant == v) & ((faces.difficulty == d) | (d == "All"))]
            i = imgs[(imgs.variant == v) & ((imgs.difficulty == d) | (d == "All"))]
            if f.empty:
                continue
            labels = [e for e in config.EMOTIONS if (f.truth == e).any()]
            out.append({"variant": v, "difficulty": d, "images": len(i), "faces": len(f), "accuracy": f.correct.mean(),
                        "recall": recall_score(f.truth, f.pred, labels=labels, average="macro", zero_division=0),
                        "precision": precision_score(f.truth, f.pred, labels=labels, average="macro", zero_division=0),
                        "f1": f1_score(f.truth, f.pred, labels=labels, average="macro", zero_division=0),
                        "face_count": i.count_ok.mean()})
    return pd.DataFrame(out)


def bootstrap_vs_baseline(imgs, variants, base="V0_plain", B=2000, seed=0):
    """Accuracy difference vs the baseline with a 95% bootstrap interval (resampling images)."""
    rng = np.random.default_rng(seed)
    piv = imgs.pivot(index="image", columns="variant", values="n_correct")
    n_gt = imgs[imgs.variant == base].set_index("image")["n_gt"].reindex(piv.index).values
    out = []
    for v in variants:
        if v == base:
            continue
        cv_, cb = piv[v].values, piv[base].values
        diffs = []
        for _ in range(B):
            idx = rng.integers(0, len(n_gt), len(n_gt))
            diffs.append(cv_[idx].sum() / n_gt[idx].sum() - cb[idx].sum() / n_gt[idx].sum())
        out.append({"variant": v, "acc_diff": cv_.sum() / n_gt.sum() - cb.sum() / n_gt.sum(),
                    "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5))})
    return pd.DataFrame(out)


def wide(m, col, variants):
    """One metric as a variants x difficulty table."""
    t = m.pivot(index="variant", columns="difficulty", values=col).reindex(variants)
    t = t[[c for c in config.DIFFICULTIES + ["All"] if c in t.columns]]
    t.insert(0, "Variant", [config.VARIANT_TITLES[v] for v in t.index])
    return t.reset_index(drop=True)


def per_class_recall(faces, variants):
    return faces.groupby(["variant", "truth"]).correct.mean().unstack().reindex(variants)


def majority_baseline(faces, base="V0_plain"):
    truth = faces[faces.variant == base].truth
    cls = truth.value_counts().idxmax()
    return cls, float((truth == cls).mean())


def collapsed_variants(faces, variants, threshold=0.8):
    """Variants where one answer makes up more than `threshold` of all faces (accuracy is then not meaningful)."""
    out, rows = [], []
    for v in variants:
        top = faces[faces.variant == v].pred.value_counts(normalize=True)
        rows.append((v, top.index[0], float(top.iloc[0])))
        if top.iloc[0] > threshold:
            out.append(v)
    return out, rows


def evaluate(data, raw, variants):
    """Everything stage 06 needs, in one dict. Only images answered for every variant are evaluated."""
    eval_data = [r for r in data if all((r["image"], v) in raw for v in variants)]
    if not eval_data:
        raise RuntimeError("No image has all variants answered yet - run the inference notebook first.")
    faces, imgs = build_tables(eval_data, raw, variants)
    M = compute_metrics(faces, imgs, variants)
    ci = bootstrap_vs_baseline(imgs, variants) if "V0_plain" in variants else pd.DataFrame()
    maj_cls, maj = majority_baseline(faces) if "V0_plain" in variants else (None, None)
    collapsed, top_answers = collapsed_variants(faces, variants)
    status = pd.Series({k[1] + "|" + k[0]: v["status"] for k, v in raw.items()
                        if k[0] in {r["image"] for r in eval_data} and k[1] in variants}).value_counts()
    return {"eval_data": eval_data, "faces": faces, "imgs": imgs, "M": M, "ci": ci,
            "per_class": per_class_recall(faces, variants), "maj_class": maj_cls, "maj": maj,
            "collapsed": collapsed, "top_answers": top_answers, "status": status}


def summary_markdown(res, variants, run_label, extra_lines=()):
    """Plain-text summary with only numbers (the interpretation is written by a human)."""
    M, ci, faces = res["M"], res["ci"], res["faces"]
    allm = M[M.difficulty == "All"].set_index("variant")
    lines = [f"# Results summary ({run_label})", "",
             f"Images evaluated: {len(res['eval_data'])}; faces: {int((faces.variant == variants[0]).sum())}.", "",
             "## Overall"]
    for v in variants:
        lines.append(f"- {config.VARIANT_TITLES[v]}: accuracy {allm.loc[v, 'accuracy']:.1%}, macro recall {allm.loc[v, 'recall']:.1%}, "
                     f"F1 {allm.loc[v, 'f1']:.1%}, face-count accuracy {allm.loc[v, 'face_count']:.1%}")
    if len(ci):
        lines += ["", "## Difference vs plain image (accuracy, percentage points, 95% bootstrap CI)"]
        for _, r in ci.iterrows():
            verdict = "interval excludes 0" if (r.ci_low > 0 or r.ci_high < 0) else "interval includes 0 (not clearly different)"
            lines.append(f"- {config.VARIANT_TITLES[r.variant]}: {r.acc_diff * 100:+.1f} (CI {r.ci_low * 100:+.1f} to {r.ci_high * 100:+.1f}) - {verdict}")
    best = allm.accuracy.idxmax()
    lines += ["", f"Highest overall accuracy: {config.VARIANT_TITLES[best]} ({allm.loc[best, 'accuracy']:.1%})."]
    if res["maj"] is not None:
        lines.append(f"- Majority-class baseline (always '{res['maj_class']}'): accuracy {res['maj']:.1%}.")
    if res["collapsed"]:
        lines.append("- WARNING: these variants gave one answer for >80% of faces, so their accuracy is not meaningful: "
                     + ", ".join(config.VARIANT_TITLES[v] for v in res["collapsed"]))
    lines += list(extra_lines)
    return "\n".join(lines)
