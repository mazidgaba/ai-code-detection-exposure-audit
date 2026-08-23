"""Figures for the GPU run: the uncontaminated strong-model result.

Branch A was trained only on our training split, so the five shift conditions
are genuinely held out for it. That makes it the control the published
detectors cannot be: DroidDetect saw every generator family, language and
source we withhold, so evaluating it on these slices measures nothing about
shift.

Everything here is read from the recovered Kaggle reports.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aicd import config as C

COL1, COL2 = 3.5, 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.2,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

INK, ACC, WARN, BAD, MUT = "#1a1a1a", "#0F6E6B", "#8E6A12", "#A6402B", "#65458C"
ORDER = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
         "s4_unseen_domain", "s5_compound"]
LABEL = {"s1_in_distribution": "S1\nin-dist.", "s2_unseen_generator": "S2\nunseen gen.",
         "s3_unseen_language": "S3\nunseen lang.", "s4_unseen_domain": "S4\nunseen src.",
         "s5_compound": "S5\ncompound"}
CLASSES = ["human", "machine", "hybrid", "advers."]


def kag(name):
    p = C.ROOT / "eval" / "reports" / "kaggle" / name
    return json.loads(io.open(p, encoding="utf-8").read())


def outdir() -> Path:
    d = C.ROOT / "docs" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ Fig 10
def fig_collapse(out: Path) -> None:
    """The headline: a strong model losing two thirds of its accuracy."""
    a = kag("branch_a_base.json")["slices"]
    b = kag("branch_b_xgb.json")["slices"]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4))
    x = np.arange(len(ORDER))

    ax = axes[0]
    fa = [a[s]["macro_f1"] for s in ORDER]
    fb = [b[s]["macro_f1"] for s in ORDER]
    ax.plot(x, fa, "o-", color=ACC, ms=3.6, label="Branch A (ModernBERT)")
    ax.plot(x, fb, "s--", color=WARN, ms=3.2, label="Branch B (TF-IDF + XGBoost)")
    ax.axhline(0.25, color="#999", ls=":", lw=0.7)
    ax.text(-0.45, 0.27, "chance", fontsize=5.9, color="#777", ha="left")
    for i, v in enumerate(fa):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=6.0, color=ACC)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[s] for s in ORDER], fontsize=6.2)
    ax.set_ylabel("four-class macro-F1"); ax.set_ylim(0, 1.02)
    ax.set_title("(a) accuracy under held-out shift", fontsize=7.6)
    ax.legend(frameon=False, fontsize=6.0, loc="lower left")

    ax = axes[1]
    fp = [a[s]["human_fpr"] for s in ORDER]
    auc = [a[s]["binary_auc"] for s in ORDER]
    ax.bar(x, fp, 0.55, color=BAD, ec=INK, lw=0.4, label="human false-positive rate", zorder=3)
    ax.plot(x, auc, "o--", color=MUT, ms=3.2, lw=1.0, label="binary AUC", zorder=4)
    for i, v in enumerate(fp):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=6.0, color=BAD)
    ax.axhline(0.01, color=INK, ls="--", lw=0.8, zorder=5)
    # Sit the label in the gap between the S1 and S2 bars, and low enough that
    # it reads against the dashed line rather than joining the row of value
    # labels above it. Against the right edge it overprinted the S5 bar.
    ax.text(0.5, 0.020, "1% target", fontsize=5.9, color=INK,
            ha="center", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[s] for s in ORDER], fontsize=6.2)
    ax.set_ylabel("rate")
    # Headroom above the AUC curve, which tops out at 1.0, so the legend has
    # somewhere to sit that is not on top of the line it describes.
    ax.set_ylim(0, 1.30)
    ax.set_title("(b) Branch A: false accusations of humans", fontsize=7.6)
    ax.legend(frameon=False, fontsize=6.0, loc="upper left", borderaxespad=0.3)

    for a_ in axes:
        a_.grid(axis="y", ls=":", alpha=0.45, zorder=0)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig10_collapse.pdf"); fig.savefig(out / "fig10_collapse.png")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 11
def fig_confusion_shift(out: Path) -> None:
    """Where the accuracy goes: human code relabelled as disguised machine code."""
    a = kag("branch_a_base.json")["slices"]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.55))
    for ax, s, t in zip(axes, ["s1_in_distribution", "s5_compound"],
                        ["(a) S1 in-distribution", "(b) S5 compound shift"]):
        cm = np.array(a[s]["confusion"], dtype=float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="OrRd", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.4,
                        color="white" if cmn[i, j] > 0.55 else INK,
                        fontweight="bold" if (i == 0 and cmn[i, j] > 0.5) else "normal")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CLASSES, fontsize=6.2, rotation=30, ha="right")
        ax.set_yticklabels(CLASSES, fontsize=6.2)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(t, fontsize=7.6)
    fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02).ax.tick_params(labelsize=6)
    fig.savefig(out / "fig11_confusion_shift.pdf")
    fig.savefig(out / "fig11_confusion_shift.png")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 12
def fig_per_class(out: Path) -> None:
    """Every class collapses; hybrid survives only because errors pile into it."""
    a = kag("branch_a_base.json")["slices"]
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    x = np.arange(len(ORDER)); w = 0.2
    cols = {"human": BAD, "machine": ACC, "hybrid": WARN, "adversarial": MUT}
    for k, (cls, c) in enumerate(cols.items()):
        v = [a[s]["per_class_f1"][cls] for s in ORDER]
        ax.bar(x + (k - 1.5) * w, v, w, color=c, ec=INK, lw=0.3,
               label=cls, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[s] for s in ORDER], fontsize=6.2)
    ax.set_ylabel("per-class F1"); ax.set_ylim(0, 1.05)
    # Above the axes, not inside them: the S1 human bar reaches 0.98 and an
    # inset legend lands on top of it.
    ax.legend(frameon=False, fontsize=6.0, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0,
              columnspacing=1.0, handlelength=1.2)
    ax.grid(axis="y", ls=":", alpha=0.45, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(out / "fig12_per_class.pdf"); fig.savefig(out / "fig12_per_class.png")
    plt.close(fig)


def main() -> None:
    out = outdir()
    made = []
    for name, fn in [("fig10_collapse", fig_collapse),
                     ("fig11_confusion_shift", fig_confusion_shift),
                     ("fig12_per_class", fig_per_class)]:
        try:
            fn(out)
            made.append(name)
            print(f"  [ok]   {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n-> {out}")

    # The manuscript reads paper/figures, not docs/figures. Without this the
    # two silently diverge and the paper keeps compiling the previous figure.
    paper = C.ROOT.parent / "paper" / "figures"
    if paper.is_dir():
        import shutil
        n = 0
        for name in made:
            for ext in (".pdf", ".png"):
                src = out / (name + ext)
                if src.exists():
                    shutil.copy(src, paper / (name + ext))
                    n += 1
        print(f"-> {paper}  ({n} files mirrored)")


if __name__ == "__main__":
    main()
