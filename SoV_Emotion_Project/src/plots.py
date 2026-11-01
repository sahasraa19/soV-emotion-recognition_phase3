"""Stage 07: figures, failure cases and the comparison with the paper."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, prompts
from .evaluation import NONE

# Zhang et al. 2024 (GPT-4V, own 119-image dataset), Tables 2-3: accuracy % Easy / Medium / Hard / Total, Recall@1 total
PAPER = {
    "V0_plain":        (48.85, 47.95, 32.45, 44.44, 22.11),
    "V1_redcircle":    (51.72, 48.53, 45.61, 49.01, 23.89),
    "V2_blur_outside": (54.02, 46.19, 42.10, 48.14, 22.98),
    "V3_boxes":        (47.12, 45.61, 39.47, 44.66, 23.52),
    "V4_boxes_ids":    (58.04, 51.46, 42.10, 51.63, 28.24),
    "V5_sov_full":     (60.91, 53.21, 50.00, 55.33, 28.69),
}


def _save(fig_dir, name, show=True):
    plt.tight_layout()
    if fig_dir is not None:
        plt.savefig(fig_dir / name, dpi=130)
    if show:
        plt.show()


def accuracy_by_variant(M, variants, maj, maj_class, fig_dir=None, title="Per-face emotion accuracy by visual prompt"):
    groups = config.DIFFICULTIES + ["All"]
    fig, ax = plt.subplots(figsize=(max(14, 1.4 * len(variants)), 4.8))
    w = 0.2
    for k, d in enumerate(groups):
        vals = [M[(M.variant == v) & (M.difficulty == d)].accuracy.mean() for v in variants]
        ax.bar(np.arange(len(variants)) + (k - 1.5) * w, vals, w, label=d)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.split("_")[0] for v in variants])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.axhline(maj, color="gray", ls="--", lw=1, label=f"always '{maj_class}'")
    ax.legend(title="Difficulty / baseline")
    ax.set_title(title)
    _save(fig_dir, "accuracy_by_variant.png")


def ablation(M, fig_dir=None, chain=("V0_plain", "V3_boxes", "V4_boxes_ids", "V5_sov_full"), labels=("Plain", "Box", "Box+Number", "SoV")):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for col, a, t in [("accuracy", ax[0], "Accuracy"), ("face_count", ax[1], "Face-count accuracy")]:
        for d in config.DIFFICULTIES + ["All"]:
            a.plot(list(labels), [M[(M.variant == v) & (M.difficulty == d)][col].mean() for v in chain], marker="o", label=d)
        a.set_title("Ablation - " + t)
        a.set_ylim(0, 1)
        a.legend()
    _save(fig_dir, "ablation.png")


def confusion(faces, pair=("V0_plain", "V5_sov_full"), fig_dir=None, name="confusion_plain_vs_sov.png"):
    labs = config.EMOTIONS + [NONE]
    fig, axes = plt.subplots(1, len(pair), figsize=(7 * len(pair), 6))
    for ax_, v in zip(np.atleast_1d(axes), pair):
        f = faces[faces.variant == v]
        cm = np.zeros((len(labs), len(labs)), int)
        for g, p in zip(f.truth, f.pred):
            cm[labs.index(g), labs.index(p if p in labs else NONE)] += 1
        ax_.imshow(cm, cmap="Blues")
        ax_.set_xticks(range(len(labs)))
        ax_.set_yticks(range(len(labs)))
        ax_.set_xticklabels(labs, rotation=45, ha="right")
        ax_.set_yticklabels(labs)
        ax_.set_xlabel("Predicted")
        ax_.set_ylabel("Ground truth")
        ax_.set_title("Confusion matrix - " + config.VARIANT_TITLES[v])
        for i in range(len(labs)):
            for j in range(len(labs)):
                if cm[i, j]:
                    ax_.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    _save(fig_dir, name)


def per_class(per_class_df, variants, fig_dir=None):
    per_class_df[[c for c in config.EMOTIONS if c in per_class_df.columns]].T.plot(kind="bar", figsize=(11, 4.5))
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.title("Per-emotion recall by variant (emotions with few faces are very noisy)")
    plt.legend([v.split("_")[0] for v in variants], ncol=3)
    _save(fig_dir, "per_class_recall.png")


def cue_isolation(M, maj, maj_class, fig_dir=None,
                  cue=("V0_plain", "V3_boxes", "V6_ids_only", "V7_landmarks_only", "V4_boxes_ids", "V5_sov_full"),
                  names=("Plain", "Box", "Number", "Landmarks", "Box+Number", "Box+Number+Landmarks"), name="cue_isolation.png"):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    for a, col, t in [(ax[0], "accuracy", "Accuracy"), (ax[1], "f1", "Macro F1")]:
        a.bar(list(names), [M[(M.variant == v) & (M.difficulty == "All")][col].iloc[0] for v in cue])
        a.set_title("Single cues vs combinations - " + t)
        a.tick_params(axis="x", rotation=30)
        a.set_ylim(0, 1)
        if col == "accuracy":
            a.axhline(maj, color="gray", ls="--", lw=1, label=f"always '{maj_class}'")
            a.legend()
    _save(fig_dir, name)


def failure_cases(eval_data, faces, imgs, out_dir, variant_a="V0_plain", variant_b="V5_sov_full", k=3, show=True):
    """Side-by-side pictures of the images where `variant_b` gained / lost the most vs `variant_a` (needs the photos)."""
    by_img = {r["image"]: r for r in eval_data}
    piv = imgs.pivot(index="image", columns="variant", values="n_correct")
    gain = (piv[variant_b] - piv[variant_a]).sort_values()
    picks = [(f"{variant_b.split('_')[0]} better", i) for i in gain[::-1].index[:k] if gain[i] > 0] + \
            [(f"{variant_a.split('_')[0]} better", i) for i in gain.index[:k] if gain[i] < 0]
    out_dir.mkdir(parents=True, exist_ok=True)
    notes = []
    for kind, img_name in picks:
        rec = by_img[img_name]
        a, b = prompts.render_record(rec, variant_a), prompts.render_record(rec, variant_b)
        h = min(a.shape[0], b.shape[0], 600)
        a, b = [cv2.resize(x, (round(x.shape[1] * h / x.shape[0]), h)) for x in (a, b)]
        both = np.hstack([a, b])
        cv2.imwrite(str(out_dir / f"{kind.replace(' ', '_')}_{img_name}"), both)
        if show:
            plt.figure(figsize=(14, 5))
            plt.imshow(cv2.cvtColor(both, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            ta, tb = faces[(faces.image == img_name) & (faces.variant == variant_a)], faces[(faces.image == img_name) & (faces.variant == variant_b)]
            plt.title(f"{kind}: {img_name} | {variant_a.split('_')[0]} {ta.correct.sum()}/{len(ta)} | {variant_b.split('_')[0]} {tb.correct.sum()}/{len(tb)}")
            plt.show()
        ta = faces[(faces.image == img_name) & (faces.variant == variant_a)]
        tb = faces[(faces.image == img_name) & (faces.variant == variant_b)]
        notes.append(f"{kind} | {img_name} | truth={list(ta.truth)} | {variant_a}={list(ta.pred)} | {variant_b}={list(tb.pred)}")
    (out_dir / "notes.txt").write_text("\n".join(notes))
    return notes


def paper_comparison(M, variants):
    """Paper's accuracy next to ours (the pattern is comparable, the absolute numbers are not)."""
    rows = []
    for v in variants:
        mine = {d: M[(M.variant == v) & (M.difficulty == d)] for d in config.DIFFICULTIES + ["All"]}
        g = lambda d, c: (mine[d][c].iloc[0] * 100) if len(mine[d]) else float("nan")
        p = PAPER.get(v, (float("nan"),) * 5)
        rows.append({"Variant": config.VARIANT_TITLES[v],
                     "Easy (paper)": p[0], "Easy (ours)": g("Easy", "accuracy"),
                     "Medium (paper)": p[1], "Medium (ours)": g("Medium", "accuracy"),
                     "Hard (paper)": p[2], "Hard (ours)": g("Hard", "accuracy"),
                     "Total acc (paper)": p[3], "Total acc (ours)": g("All", "accuracy"),
                     "Recall (paper R@1)": p[4], "Recall (ours, macro)": g("All", "recall")})
    return pd.DataFrame(rows)


def cross_model_table(metrics_by_model, variants, col="accuracy"):
    """metrics_by_model: {'Qwen2-VL-7B': M_df, 'LLaVA-1.5-7B': M_df}. One row per variant, one column per model (% overall)."""
    rows = []
    for v in variants:
        row = {"Variant": config.VARIANT_TITLES[v]}
        for name, M in metrics_by_model.items():
            sub = M[(M.variant == v) & (M.difficulty == "All")]
            row[name] = sub[col].iloc[0] * 100 if len(sub) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
