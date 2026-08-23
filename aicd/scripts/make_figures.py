"""Publication figures, generated from the measured artifacts only.

Nothing here is hand-drawn or illustrative: every value is read back from the
JSON/NPY files written by the pipeline, so a figure cannot silently drift from
the result it depicts.

IEEE geometry: single-column 3.5in, double-column 7.16in, 8pt type.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from aicd import config as C

COL1, COL2 = 3.5, 7.16
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.2,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

INK, ACC, WARN, BAD, MUT = "#1a1a1a", "#0F6E6B", "#8E6A12", "#A6402B", "#65458C"
SLICE_LABEL = {
    "s1_in_distribution": "S1\nin-dist.",
    "s2_unseen_generator": "S2\nunseen gen.",
    "s3_unseen_language": "S3\nunseen lang.",
    "s4_unseen_domain": "S4\nunseen dom.",
    "s5_compound": "S5\ncompound",
}
ORDER = list(SLICE_LABEL)


def outdir(cfg) -> Path:
    d = C.ROOT / "docs" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(cfg, name):
    p = C.reports(cfg) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --------------------------------------------------------------- Fig 1
def fig_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL2, 2.5))
    ax.set_xlim(-0.2, 16.6); ax.set_ylim(-1.15, 6.1); ax.axis("off")

    def box(x, y, w, h, title, sub, color, fs=7.4, trained=True):
        # Components carrying measured results are drawn solid on white;
        # components that are implemented but not trained at scale are drawn
        # dashed on grey, so the figure cannot be read as claiming more
        # evidence than the paper reports.
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08", lw=0.9, ec=color,
            fc="white" if trained else "#EDEDED",
            ls="-" if trained else (0, (3, 2)), zorder=2))
        ax.text(x + w / 2, y + h - 0.42, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=color, zorder=3)
        for i, line in enumerate(sub):
            ax.text(x + w / 2, y + h - 0.86 - i * 0.35, line, ha="center",
                    va="center", fontsize=6.3, color=INK, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.8, color="#666", zorder=1))

    box(0.1, 2.0, 2.3, 2.0, "Source file", ["code +", "language"], INK)
    box(3.1, 4.05, 4.0, 1.75, "Branch A", ["ModernBERT-base, 4-class",
                                           "CE + 0.1 x batch-hard triplet"], ACC,
        trained=False)
    box(3.1, 2.05, 4.0, 1.75, "Branch B", ["char n-gram TF-IDF + 93",
                                           "stylometric / AST feats -> XGBoost"], WARN)
    box(3.1, 0.05, 4.0, 1.75, "Branch C", ["Fast-DetectGPT curvature",
                                           "Qwen2.5-Coder, zero-shot"], MUT,
        trained=False)
    box(7.8, 2.05, 2.6, 1.75, "Stacker", ["multinomial LR", "fit on val only"], INK,
        trained=False)
    box(10.9, 2.05, 2.4, 1.75, "Calibration", ["isotonic,", "per language"], INK)
    box(13.7, 2.05, 2.2, 1.75, "Policy", ["thresholds +", "abstain band"], BAD)

    # Arrowheads stop just short of each target box, so the glyph never lands on
    # a border and get read as part of it.
    arrow(2.4, 3.0, 3.02, 4.9); arrow(2.4, 3.0, 3.02, 2.9); arrow(2.4, 3.0, 3.02, 0.9)
    arrow(7.15, 4.9, 7.72, 3.2); arrow(7.15, 2.9, 7.72, 2.9); arrow(7.15, 0.9, 7.72, 2.6)
    arrow(10.45, 2.9, 10.82, 2.9); arrow(13.35, 2.9, 13.62, 2.9)

    ax.text(9.1, 1.72, "4+4+1 features", ha="center", va="top", fontsize=6.1,
            color="#666", style="italic")

    # Legend for the evidence convention.
    ax.add_patch(FancyBboxPatch((0.15, -0.95), 0.42, 0.30,
                                boxstyle="round,pad=0.04", lw=0.8, ec=INK,
                                fc="white", zorder=2))
    ax.text(0.72, -0.80, "trained and evaluated at scale (this paper)",
            va="center", fontsize=5.9, color=INK)
    ax.add_patch(FancyBboxPatch((8.1, -0.95), 0.42, 0.30,
                                boxstyle="round,pad=0.04", lw=0.8, ec=INK,
                                fc="#EDEDED", ls=(0, (3, 2)), zorder=2))
    ax.text(8.67, -0.80, "implemented and execution-verified, not trained at scale",
            va="center", fontsize=5.9, color=INK)
    # Three short lines keep the outcome list inside the Policy box's own width
    # instead of spilling across its border.
    ax.text(14.8, 1.80, "human / machine /\nhybrid / adversarial /\nabstain",
            ha="center", va="top", linespacing=1.45, fontsize=5.7,
            color="#666", style="italic")
    fig.savefig(out / "fig1_architecture.pdf")
    fig.savefig(out / "fig1_architecture.png")
    plt.close(fig)


# --------------------------------------------------------------- Fig 2
def fig_degradation(cfg, out: Path) -> None:
    res = load(cfg, "branch_b_xgb.json")
    if not res:
        return
    f1 = [res["slices"][s]["macro_f1"] for s in ORDER]
    auc = [res["slices"][s].get("binary_auc", np.nan) for s in ORDER]

    fig, ax = plt.subplots(figsize=(COL1, 2.15))
    x = np.arange(len(ORDER))
    ax.bar(x, f1, 0.56, color=ACC, ec=INK, lw=0.4, label="4-class macro-F1", zorder=3)
    ax.plot(x, auc, "o--", color=BAD, ms=3.2, lw=1.0,
            label="binary AUC (human vs rest)", zorder=4)
    ax.axhline(0.25, color="#999", ls=":", lw=0.7, zorder=2)
    ax.text(4.42, 0.275, "chance", fontsize=5.9, color="#777", ha="right")

    for i, v in enumerate(f1):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=6.0)
    ax.set_xticks(x); ax.set_xticklabels([SLICE_LABEL[s] for s in ORDER], fontsize=6.2)
    ax.set_ylabel("score"); ax.set_ylim(0, 1.10)
    ax.grid(axis="y", ls=":", alpha=0.45, zorder=0)
    ax.legend(frameon=False, loc="lower left", fontsize=6.1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(out / "fig2_degradation.pdf"); fig.savefig(out / "fig2_degradation.png")
    plt.close(fig)


# --------------------------------------------------------------- Fig 3
def fig_calibration(cfg, out: Path) -> None:
    cal = load(cfg, "calibration_b.json")
    if not cal:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.25))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", lw=0.7, color="#999", label="perfect")
    for s, c in zip(["s1_in_distribution", "s2_unseen_generator", "s4_unseen_domain"],
                    [ACC, WARN, BAD]):
        r = cal["slices"].get(s, {}).get("reliability")
        if r and r["pred"]:
            ax.plot(r["pred"], r["obs"], "o-", ms=2.6, lw=1.0, color=c,
                    label=SLICE_LABEL[s].replace("\n", " "))
    ax.set_xlabel("predicted P(machine)"); ax.set_ylabel("observed frequency")
    ax.set_title("(a) reliability after isotonic", fontsize=7.6)
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")
    ax.grid(ls=":", alpha=0.45); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1]
    x = np.arange(len(ORDER)); w = 0.38
    raw = [cal["slices"].get(s, {}).get("ece_raw", np.nan) for s in ORDER]
    calv = [cal["slices"].get(s, {}).get("ece_calibrated", np.nan) for s in ORDER]
    ax.bar(x - w / 2, raw, w, label="raw", color="#c9d3d1", ec=INK, lw=0.4, zorder=3)
    ax.bar(x + w / 2, calv, w, label="isotonic", color=ACC, ec=INK, lw=0.4, zorder=3)
    ax.axhline(0.05, color=BAD, ls="--", lw=0.8, zorder=4)
    # Anchor the gate label over the short S1/S2 bars on the left. Right-aligning
    # it put the text on top of the tall S4 bar.
    ax.text(-0.42, 0.058, "promotion gate 0.05", fontsize=5.9, color=BAD,
            ha="left", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([SLICE_LABEL[s] for s in ORDER], fontsize=6.2)
    ax.set_ylabel("expected calibration error")
    ax.set_title("(b) ECE by slice", fontsize=7.6)
    ax.legend(frameon=False, fontsize=6.2); ax.grid(axis="y", ls=":", alpha=0.45, zorder=0)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.savefig(out / "fig3_calibration.pdf"); fig.savefig(out / "fig3_calibration.png")
    plt.close(fig)


# --------------------------------------------------------------- Fig 4
def fig_threshold_transfer(cfg, out: Path) -> None:
    pol = load(cfg, "policy_b.json")
    if not pol:
        return
    fig, ax = plt.subplots(figsize=(COL1, 2.15))
    x = np.arange(len(ORDER)); w = 0.27
    fpr = [pol["slices"].get(s, {}).get("human_fpr_on_called") or 0 for s in ORDER]
    ab = [pol["slices"].get(s, {}).get("abstain_rate") or 0 for s in ORDER]
    rec = [pol["slices"].get(s, {}).get("machine_recall_on_called") or 0 for s in ORDER]

    ax.bar(x - w, fpr, w, label="human FPR", color=BAD, ec=INK, lw=0.4, zorder=3)
    ax.bar(x, rec, w, label="machine recall", color=ACC, ec=INK, lw=0.4, zorder=3)
    ax.bar(x + w, ab, w, label="abstain rate", color="#c9d3d1", ec=INK, lw=0.4, zorder=3)
    ax.axhline(0.01, color=INK, ls="--", lw=0.8, zorder=4)
    ax.text(-0.45, 0.035, "1% target", fontsize=5.9, color=INK)
    ax.annotate(f"{fpr[3]:.3f}", xy=(3 - w, fpr[3]), xytext=(3 - w, fpr[3] + 0.09),
                ha="center", fontsize=6.4, color=BAD, fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=BAD))
    ax.set_xticks(x); ax.set_xticklabels([SLICE_LABEL[s] for s in ORDER], fontsize=6.2)
    ax.set_ylabel("rate"); ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=6.1, ncol=1, loc="upper left")
    ax.grid(axis="y", ls=":", alpha=0.45, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(out / "fig4_threshold_transfer.pdf")
    fig.savefig(out / "fig4_threshold_transfer.png")
    plt.close(fig)


# --------------------------------------------------------------- Fig 5
def fig_shap(cfg, out: Path) -> None:
    p = C.reports(cfg) / "shap_top20.csv"
    if not p.exists():
        return
    d = pd.read_csv(p).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    cols = [ACC if not f.startswith("ast_") else MUT for f in d["feature"]]
    ax.barh(range(len(d)), d["mean_abs_shap"], color=cols, ec=INK, lw=0.4, zorder=3)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f.replace("_", "\\_") if False else f for f in d["feature"]],
                       fontsize=6.2, family="monospace")
    ax.set_xlabel("mean |SHAP|")
    ax.grid(axis="x", ls=":", alpha=0.45, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=ACC, ec=INK, lw=0.4),
          plt.Rectangle((0, 0), 1, 1, fc=MUT, ec=INK, lw=0.4)]
    ax.legend(hs, ["stylometric", "AST"], frameon=False, fontsize=6.2, loc="lower right")
    fig.savefig(out / "fig5_shap.pdf"); fig.savefig(out / "fig5_shap.png")
    plt.close(fig)


# --------------------------------------------------------------- Fig 6
def fig_formatter(cfg, out: Path) -> None:
    p = C.reports(cfg) / "format_evasion.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(COL1, 2.15))
    x = np.arange(len(d)); w = 0.38
    ax.bar(x - w / 2, d["f1_raw"], w, label="raw input", color=ACC, ec=INK, lw=0.4, zorder=3)
    ax.bar(x + w / 2, d["f1_formatted"], w, label="normalized input",
           color="#c9d3d1", ec=INK, lw=0.4, zorder=3)
    for i, (a, b) in enumerate(zip(d["f1_raw"], d["f1_formatted"])):
        if b < a:
            ax.annotate("", xy=(i + w / 2, b), xytext=(i - w / 2, a),
                        arrowprops=dict(arrowstyle="->", lw=0.7, color=BAD))
            ax.text(i, max(a, b) + 0.035, f"{b - a:+.2f}", ha="center",
                    fontsize=6.0, color=BAD, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SLICE_LABEL.get(s, s) for s in d["slice"]], fontsize=6.2)
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, fontsize=6.2)
    ax.grid(axis="y", ls=":", alpha=0.45, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(out / "fig6_formatter.pdf"); fig.savefig(out / "fig6_formatter.png")
    plt.close(fig)


# --------------------------------------------------------------- Fig 7
def fig_confusion(cfg, out: Path) -> None:
    res = load(cfg, "branch_b_xgb.json")
    if not res:
        return
    names = ["human", "machine", "hybrid", "advers."]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for ax, s, t in zip(axes, ["s1_in_distribution", "s4_unseen_domain"],
                        ["(a) S1 in-distribution", "(b) S4 unseen domain"]):
        cm = np.array(res["slices"][s]["confusion"], dtype=float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="BuGn", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.2, color="white" if cmn[i, j] > 0.55 else INK)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(names, fontsize=6.2, rotation=30, ha="right")
        ax.set_yticklabels(names, fontsize=6.2)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(t, fontsize=7.6)
    fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02).ax.tick_params(labelsize=6)
    fig.savefig(out / "fig7_confusion.pdf"); fig.savefig(out / "fig7_confusion.png")
    plt.close(fig)


# --------------------------------------------------------------- Fig 8
def fig_experiments(cfg, out: Path) -> None:
    e = load(cfg, "experiments.json")
    if not e:
        return
    panels = [k for k in ("e1", "e2", "e3", "e4") if k in e]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(COL2, 2.3))
    if len(panels) == 1:
        axes = [axes]

    for ax, key in zip(axes, panels):
        if key == "e1":
            v = [e["e1"]["problem_wise"]["macro_f1"], e["e1"]["random_row"]["macro_f1"]]
            ax.bar(["problem-wise", "random row"], v, 0.5,
                   color=[ACC, BAD], ec=INK, lw=0.4, zorder=3)
            for i, y in enumerate(v):
                ax.text(i, y + 0.012, f"{y:.3f}", ha="center", fontsize=6.4)
            ax.set_ylabel("macro-F1")
            ax.set_title(f"(a) split leakage\n$\\Delta$ = {e['e1']['inflation']:+.3f}",
                         fontsize=7.4)
            ax.set_ylim(0, max(v) * 1.25)
        elif key == "e2":
            v = [e["e2"]["binary"]["hybrid_recall"],
                 e["e2"]["ternary"]["hybrid_recall"],
                 e["e2"]["four_class"]["hybrid_recall"]]
            ax.bar(["binary", "ternary", "4-class"], v, 0.5,
                   color=[BAD, WARN, ACC], ec=INK, lw=0.4, zorder=3)
            for i, y in enumerate(v):
                ax.text(i, y + 0.015, f"{y:.3f}", ha="center", fontsize=6.4)
            ax.set_ylabel("recall on hybrid code")
            ax.set_title("(b) label granularity", fontsize=7.4)
            ax.set_ylim(0, 1.12)
        elif key == "e4":
            names = ["tfidf_only", "stylometry_only", "stylometry_ast", "all"]
            # Short enough to sit horizontally; rotated labels collided here.
            short = ["TF-IDF", "styl.", "sty+AST", "all"]
            s1 = [e["e4"][n]["s1_in_distribution"] for n in names]
            s5 = [e["e4"][n].get("s5_compound", np.nan) for n in names]
            xx = np.arange(len(names)); w2 = 0.38
            ax.bar(xx - w2 / 2, s1, w2, label="S1", color=ACC, ec=INK, lw=0.4, zorder=3)
            ax.bar(xx + w2 / 2, s5, w2, label="S5", color=BAD, ec=INK, lw=0.4, zorder=3)
            ax.set_xticks(xx); ax.set_xticklabels(short, fontsize=5.5)
            ax.set_ylabel("macro-F1")
            ax.set_title("(d) representation", fontsize=7.4)
            ax.legend(frameon=False, fontsize=5.8, ncol=2)
            ax.set_ylim(0, max(s1) * 1.32)
        else:
            gp, ga = e["e3"]["gap_plain"], e["e3"]["gap_augmented"]
            ax.bar(["plain", "augmented"], [gp, ga], 0.5,
                   color=[BAD, ACC], ec=INK, lw=0.4, zorder=3)
            for i, y in enumerate([gp, ga]):
                ax.text(i, y + 0.006, f"{y:+.3f}", ha="center", fontsize=6.4)
            ax.set_ylabel("macro-F1 lost to formatting")
            ax.set_title("(c) evasion defence", fontsize=7.4)
        ax.grid(axis="y", ls=":", alpha=0.45, zorder=0)
        ax.tick_params(labelsize=6.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig8_experiments.pdf"); fig.savefig(out / "fig8_experiments.png")
    plt.close(fig)


# --------------------------------------------------------------- Fig 9
def fig_risk_coverage(cfg, out: Path) -> None:
    """Risk-coverage curves, the standard view for a selective classifier.

    A single operating point says nothing about the accuracy/coverage
    trade-off. These curves show what abstention can and cannot buy on each
    slice, and mark where the paper's fitted policy actually sits.
    """
    import pandas as _pd
    from aicd.eval.selective import risk_coverage

    art = C.ROOT / cfg.data.cache_dir
    sel = load(cfg, "selective.json")
    if not sel:
        return
    df = _pd.read_parquet(art / "splits.parquet")
    apath = C.ROOT / cfg.project.artifacts_dir

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5))
    cols = {"s1_in_distribution": ACC, "s2_unseen_generator": WARN,
            "s3_unseen_language": MUT, "s4_unseen_domain": BAD,
            "s5_compound": "#444"}

    for s, c in cols.items():
        f = apath / f"proba_b_{s}.npy"
        if not f.exists():
            continue
        proba = np.load(f)
        y = df.loc[df["slice"] == s, "label"].to_numpy()
        if len(y) != len(proba):
            continue
        pred, conf = proba.argmax(1), proba.max(1)
        lab = SLICE_LABEL[s].replace("\n", " ")

        cov, risk = risk_coverage(y, conf, pred != y)
        axes[0].plot(cov, risk, lw=1.1, color=c, label=lab)

        cov_h, risk_h = risk_coverage(y, conf, (y == 0) & (pred != 0))
        axes[1].plot(cov_h, risk_h, lw=1.1, color=c, label=lab)

        pol = sel["slices"].get(s, {}).get("policies", {}).get("fixed")
        if pol and not np.isnan(pol.get("human_fpr", np.nan)):
            axes[1].plot([pol["coverage"]], [pol["human_fpr"]], "o", ms=4.2,
                         mfc="white", mec=c, mew=1.2, zorder=5)

    axes[0].set_xlabel("coverage"); axes[0].set_ylabel("selective risk (all errors)")
    axes[0].set_title("(a) risk-coverage", fontsize=7.6)
    axes[0].legend(frameon=False, fontsize=5.8, loc="upper left")

    axes[1].set_xlabel("coverage"); axes[1].set_ylabel("false accusations of humans")
    axes[1].set_title("(b) human false-positive rate vs coverage", fontsize=7.6)
    axes[1].axhline(0.01, color=INK, ls="--", lw=0.7)
    axes[1].text(0.985, 0.022, "1% target", fontsize=5.8, color=INK, ha="right")

    for a in axes:
        a.grid(ls=":", alpha=0.45); a.set_xlim(0, 1)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig9_risk_coverage.pdf")
    fig.savefig(out / "fig9_risk_coverage.png")
    plt.close(fig)


def main() -> None:
    cfg = C.load("cpu.yaml")
    out = outdir(cfg)
    made = []
    for name, fn in [
        ("fig1_architecture", lambda: fig_architecture(out)),
        ("fig2_degradation", lambda: fig_degradation(cfg, out)),
        ("fig3_calibration", lambda: fig_calibration(cfg, out)),
        ("fig4_threshold_transfer", lambda: fig_threshold_transfer(cfg, out)),
        ("fig5_shap", lambda: fig_shap(cfg, out)),
        ("fig6_formatter", lambda: fig_formatter(cfg, out)),
        ("fig7_confusion", lambda: fig_confusion(cfg, out)),
        ("fig8_experiments", lambda: fig_experiments(cfg, out)),
        ("fig9_risk_coverage", lambda: fig_risk_coverage(cfg, out)),
    ]:
        try:
            fn()
            ok = (out / f"{name}.pdf").exists()
            made.append((name, ok))
            print(f"  {'[ok]  ' if ok else '[skip]'} {name}")
        except Exception as ex:
            print(f"  [FAIL] {name}: {type(ex).__name__}: {ex}")
    print(f"\n{sum(1 for _, o in made if o)}/{len(made)} figures -> {out}")


if __name__ == "__main__":
    main()
