"""E2: is the collapse a representational failure, or a correctable prior shift?

The paper's deployment argument rests on the model being confident and wrong,
and on the collapse under compound shift being a real loss of discriminative
power. Its own S5 confusion matrix argues against the second half of that: the
mass does not scatter, it merges, with human folding into adversarial and
machine into hybrid while binary AUC stays at 0.76. That is the signature of a
shifted decision boundary rather than a destroyed representation, and a shifted
boundary is often fixable without retraining.

If it is fixable, the paper's headline is wrong and the honest result is a
better one: the shift penalty decomposes into a representational part and a
correctable part, and reporting only aggregate macro-F1 attributes all of it to
the former. If it is not fixable, the thesis survives having been attacked
properly, which is worth more than never having asked.

Four measurements, in the order they answer the question:

  E2a  How badly calibrated is the model, per condition and per class?
       ECE and Brier, plus reliability bins.

  E2b  Can the target label distribution be recovered without target labels?
       BBSE (Lipton et al. 2018) and EM/SLD (Saerens et al. 2002), both scored
       against the true target priors, which we have and they do not.

  E2c  How much of the loss does a correction actually recover?
       Temperature scaling and prior adjustment fitted on k labelled target
       examples, k in {10, 50, 100, 500}, evaluated on the rows NOT used for
       fitting, repeated over many draws.

  E2d  The oracle arm: correction using the TRUE target priors.
       This is the ceiling for anything prior correction can achieve. If the
       oracle does not recover S5, no amount of recalibration will, and the
       failure is representational. That single number decides the branch.

Everything here is post-hoc arithmetic on probability arrays that already
exist, so it needs no GPU and no retraining.

    python -m aicd.eval.shift_diagnosis --model a
    python -m aicd.eval.shift_diagnosis --model matched_on_original
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

LABEL_NAMES = ["human", "machine", "hybrid", "adversarial"]
HUMAN = 0
CONDITIONS = ["s1_in_distribution", "s2_unseen_generator", "s3_unseen_language",
              "s4_unseen_domain", "s5_compound"]

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "aicd" / "artifacts" / "kaggle" / "labels.parquet"
SOURCES = {
    # D1, the 196,854-row model the paper's control experiments use.
    "a": ROOT / "aicd" / "artifacts" / "kaggle" / "proba_a_{slice}.npy",
    # The 394,624-row matched-scale model, scored on the SAME evaluation rows
    # so the two are directly comparable. The other matched arrays are scored
    # on that build's own larger eval set and do not align with these labels.
    "matched_on_original":
        ROOT / "kaggle_runs" / "results" / "matched-scale" / "arrays"
        / "proba_a_matched_on_original_{slice}.npy",
}


# --------------------------------------------------------------------------
# probabilities and logits
# --------------------------------------------------------------------------

def as_logits(p: np.ndarray) -> np.ndarray:
    """Recover logits from probabilities, exactly.

    Temperature scaling is defined on logits and we only stored probabilities.
    That is not a loss: if p = softmax(z) then log p = z - logsumexp(z), and
    softmax is invariant to a constant shift, so softmax(log(p) / T) is the
    same as softmax(z / T). The clip only guards log(0).
    """
    return np.log(np.clip(p, 1e-12, 1.0))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def priors(y: np.ndarray, k: int = 4) -> np.ndarray:
    c = np.bincount(y, minlength=k).astype(float)
    return c / max(c.sum(), 1.0)


# --------------------------------------------------------------------------
# E2a: calibration
# --------------------------------------------------------------------------

def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Confidence-based ECE over the four-way decision the model actually makes."""
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, n = 0.0, len(y)
    for i in range(bins):
        hi = edges[i + 1]
        m = (conf >= edges[i]) & (conf < hi if i < bins - 1 else conf <= 1.0)
        if not m.any():
            continue
        total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def classwise_ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """ECE averaged over the four one-vs-rest probabilities.

    Confidence-based ECE only looks at the top class, so a model can score well
    on it while the probabilities it assigns to the other three are nonsense.
    """
    out = []
    for k in range(p.shape[1]):
        yk = (y == k).astype(float)
        pk = p[:, k]
        edges = np.linspace(0.0, 1.0, bins + 1)
        tot = 0.0
        for i in range(bins):
            hi = edges[i + 1]
            m = (pk >= edges[i]) & (pk < hi if i < bins - 1 else pk <= 1.0)
            if not m.any():
                continue
            tot += m.sum() / len(y) * abs(yk[m].mean() - pk[m].mean())
        out.append(tot)
    return float(np.mean(out))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot truth."""
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return float(((p - oh) ** 2).sum(axis=1).mean())


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 15) -> list[dict]:
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        hi = edges[i + 1]
        m = (conf >= edges[i]) & (conf < hi if i < bins - 1 else conf <= 1.0)
        if m.sum() < 5:
            continue
        out.append({"bin_lo": float(edges[i]), "bin_hi": float(hi),
                    "confidence": float(conf[m].mean()),
                    "accuracy": float(correct[m].mean()), "n": int(m.sum())})
    return out


# --------------------------------------------------------------------------
# E2b: estimating the target label distribution without target labels
# --------------------------------------------------------------------------

def bbse_priors(y_src: np.ndarray, p_src: np.ndarray, p_tgt: np.ndarray,
                k: int = 4) -> np.ndarray:
    """Black-box shift estimation (Lipton, Wang and Smola, 2018).

    Builds the source confusion matrix C[i, j] = P(predict i, true j), reads
    the predicted-label distribution q on the target, and solves C w = q for
    the prior ratios w. Least squares rather than an inverse, because C is
    close to singular whenever two classes are confusable, which is precisely
    the case here.
    """
    yhat_src = p_src.argmax(axis=1)
    C = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            C[i, j] = np.mean((yhat_src == i) & (y_src == j))
    q = np.bincount(p_tgt.argmax(axis=1), minlength=k).astype(float) / len(p_tgt)
    w, *_ = np.linalg.lstsq(C, q, rcond=None)
    est = np.clip(w, 0.0, None) * priors(y_src, k)
    s = est.sum()
    return est / s if s > 0 else priors(y_src, k)


def em_priors(p_src_prior: np.ndarray, p_tgt: np.ndarray,
              iters: int = 500, tol: float = 1e-9) -> np.ndarray:
    """EM prior estimation (Saerens, Latinne and Decaestecker, 2002).

    Iterates between reweighting the target probabilities by the current prior
    estimate and re-reading the priors off the reweighted probabilities. Uses
    no labels at all, which is what makes it deployable.
    """
    pi = p_src_prior.copy()
    for _ in range(iters):
        r = p_tgt * (pi / np.clip(p_src_prior, 1e-12, None))
        r /= np.clip(r.sum(axis=1, keepdims=True), 1e-12, None)
        new = r.mean(axis=0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return pi


# --------------------------------------------------------------------------
# corrections
# --------------------------------------------------------------------------

def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    """Temperature minimising negative log-likelihood."""
    z = as_logits(p)

    def nll(log_t: float) -> float:
        q = softmax(z / np.exp(log_t))
        return float(-np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None)).mean())

    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(r.x))


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return softmax(as_logits(p) / t)


def apply_prior_shift(p: np.ndarray, pi_src: np.ndarray,
                      pi_tgt: np.ndarray) -> np.ndarray:
    """Reweight by the ratio of target to source priors, then renormalise."""
    r = p * (pi_tgt / np.clip(pi_src, 1e-12, None))
    return r / np.clip(r.sum(axis=1, keepdims=True), 1e-12, None)


# --------------------------------------------------------------------------
# the numbers the paper quotes
# --------------------------------------------------------------------------

def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1_score(y, p.argmax(axis=1), average="macro", zero_division=0))


def human_fpr_argmax(y: np.ndarray, p: np.ndarray) -> float:
    """Share of human-authored rows the model does not call human."""
    m = y == HUMAN
    return float((p[m].argmax(axis=1) != HUMAN).mean()) if m.any() else float("nan")


def human_threshold(y_val: np.ndarray, p_val: np.ndarray,
                    rate: float = 0.01) -> float:
    """The operating point the paper fits: P(human) below which we accuse.

    Chosen on validation so that exactly `rate` of genuinely human rows fall
    below it. The paper's headline is what this rate becomes off-distribution.
    """
    ph = p_val[y_val == HUMAN][:, HUMAN]
    return float(np.quantile(ph, rate)) if len(ph) else 0.0


def human_fpr_at(y: np.ndarray, p: np.ndarray, tau: float) -> float:
    m = y == HUMAN
    return float((p[m][:, HUMAN] < tau).mean()) if m.any() else float("nan")


# --------------------------------------------------------------------------
# E2c / E2d: how much can be recovered
# --------------------------------------------------------------------------

def kshot_sweep(y_t: np.ndarray, p_t: np.ndarray, pi_src: np.ndarray,
                tau: float, ks: list[int], repeats: int,
                rng: np.random.Generator) -> dict:
    """Fit corrections on k labelled target rows, score on the rest.

    Held-out scoring matters. Fitting and evaluating on the same k rows would
    report a recovery that no deployment could reproduce.
    """
    out = {}
    for k in ks:
        if k >= len(y_t):
            continue
        names = ["none", "temperature", "prior", "both"]
        arms = {n: [] for n in names}
        fprs = {n: [] for n in names}
        fprs_refit = {n: [] for n in names}
        for _ in range(repeats):
            idx = rng.choice(len(y_t), size=k, replace=False)
            hold = np.ones(len(y_t), dtype=bool)
            hold[idx] = False
            y_fit, p_fit = y_t[idx], p_t[idx]
            y_ev, p_ev = y_t[hold], p_t[hold]

            t = fit_temperature(y_fit, p_fit)
            pi_hat = priors(y_fit)

            tp = apply_temperature(p_ev, t)
            variants = {
                # "none" is the control arm: spend the k labels only on moving
                # the operating point, and leave the probabilities alone. Any
                # arm that does not beat this has not earned its complexity.
                "none": p_ev,
                "temperature": tp,
                "prior": apply_prior_shift(p_ev, pi_src, pi_hat),
                "both": apply_prior_shift(tp, pi_src, pi_hat),
            }
            # The same k labels also let us re-fit the threshold, which is what
            # a deployment holding target labels would obviously do. Without
            # this the FPR column is an artefact: temperature rescales every
            # probability against a threshold frozen on validation, so the rate
            # collapses to zero without a single decision having improved.
            fit_variants = {
                "none": p_fit,
                "temperature": apply_temperature(p_fit, t),
                "prior": apply_prior_shift(p_fit, pi_src, pi_hat),
                "both": apply_prior_shift(apply_temperature(p_fit, t),
                                          pi_src, pi_hat),
            }
            for name, pv in variants.items():
                arms[name].append(macro_f1(y_ev, pv))
                fprs[name].append(human_fpr_at(y_ev, pv, tau))
                tau_k = human_threshold(y_fit, fit_variants[name], 0.01)
                fprs_refit[name].append(human_fpr_at(y_ev, pv, tau_k))

        out[str(k)] = {
            name: {"macro_f1_mean": float(np.mean(v)),
                   "macro_f1_sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                   "human_fpr_val_threshold": float(np.nanmean(fprs[name])),
                   "human_fpr_refit_threshold": float(np.nanmean(fprs_refit[name]))}
            for name, v in arms.items()
        }
    return out


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="a", choices=sorted(SOURCES))
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--repeats", type=int, default=20,
                    help="draws per k; the spread over draws is reported")
    ap.add_argument("--ks", default="10,50,100,500")
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)
    tmpl = str(SOURCES[args.model])

    lab = pd.read_parquet(LABELS)
    def load(sl: str):
        y = lab.loc[lab["slice"] == sl, "label"].to_numpy()
        p = np.load(tmpl.format(slice=sl))
        if len(y) != len(p):
            raise SystemExit(
                f"{sl}: {len(y)} labels against {len(p)} rows of probabilities. "
                "These arrays were scored on a different corpus build.")
        return y, p

    y_val, p_val = load("val")
    pi_src = priors(y_val)
    tau = human_threshold(y_val, p_val, 0.01)

    print(f"model            : {args.model}")
    print("source priors    : "
          + "  ".join(f"{n}={v:.4f}" for n, v in zip(LABEL_NAMES, pi_src)))
    print(f"1% threshold     : P(human) < {tau:.6f}")
    print(f"realised on val  : {human_fpr_at(y_val, p_val, 0.01 * 0 + tau):.4f}\n")

    report = {"model": args.model, "seed": args.seed, "repeats": args.repeats,
              "ks": ks, "threshold_human": tau,
              "source_priors": pi_src.tolist(), "conditions": {}}

    hdr = (f"{'condition':22s} {'macroF1':>8s} {'ECE':>7s} {'cwECE':>7s} "
           f"{'Brier':>7s} {'FPR@1%':>8s} {'oracle':>8s} {'k=500':>8s}")
    print(hdr)
    print("-" * len(hdr))

    for cond in CONDITIONS:
        y_t, p_t = load(cond)
        pi_true = priors(y_t)

        base_f1 = macro_f1(y_t, p_t)
        row = {
            "n": int(len(y_t)),
            "support": np.bincount(y_t, minlength=4).tolist(),
            "true_priors": pi_true.tolist(),
            # E2a
            "ece": ece(y_t, p_t, args.bins),
            "classwise_ece": classwise_ece(y_t, p_t, args.bins),
            "brier": brier(y_t, p_t),
            "reliability": reliability(y_t, p_t, args.bins),
            "macro_f1": base_f1,
            "human_fpr_argmax": human_fpr_argmax(y_t, p_t),
            "human_fpr_at_threshold": human_fpr_at(y_t, p_t, tau),
        }

        # E2b: can the target priors be recovered blind?
        pi_bbse = bbse_priors(y_val, p_val, p_t)
        pi_em = em_priors(pi_src, p_t)
        row["priors_estimated"] = {
            "bbse": pi_bbse.tolist(), "em": pi_em.tolist(),
            "bbse_l1_error": float(np.abs(pi_bbse - pi_true).sum()),
            "em_l1_error": float(np.abs(pi_em - pi_true).sum()),
        }

        # E2d: the ceiling. True priors, no labels spent.
        p_oracle = apply_prior_shift(p_t, pi_src, pi_true)
        row["oracle_prior_correction"] = {
            "macro_f1": macro_f1(y_t, p_oracle),
            "human_fpr_at_threshold": human_fpr_at(y_t, p_oracle, tau),
            "recovered": macro_f1(y_t, p_oracle) - base_f1,
        }

        # A harder ceiling, and the one the conclusion actually rests on.
        # Prior reweighting multiplies the probabilities by a factor of order
        # one, so against a model this overconfident it almost never changes
        # which class wins, and "true priors recover nothing" could be an
        # artefact of overconfidence rather than a statement about the
        # representation. Softening first removes that objection: sweep the
        # temperature, apply the true priors at each, and keep the best result
        # any such correction could possibly reach.
        grid = np.exp(np.linspace(np.log(0.2), np.log(50.0), 60))
        best, best_t = -1.0, 1.0
        for t in grid:
            f = macro_f1(y_t, apply_prior_shift(
                apply_temperature(p_t, float(t)), pi_src, pi_true))
            if f > best:
                best, best_t = f, float(t)
        row["oracle_temperature_and_prior"] = {
            "macro_f1": float(best), "temperature": best_t,
            "recovered": float(best) - base_f1,
        }
        # And the blind versions, which is what a deployment could actually do.
        for name, pi_hat in (("bbse", pi_bbse), ("em", pi_em)):
            pc = apply_prior_shift(p_t, pi_src, pi_hat)
            row[f"{name}_correction"] = {
                "macro_f1": macro_f1(y_t, pc),
                "human_fpr_at_threshold": human_fpr_at(y_t, pc, tau),
                "recovered": macro_f1(y_t, pc) - base_f1,
            }

        # E2c
        row["kshot"] = kshot_sweep(y_t, p_t, pi_src, tau, ks, args.repeats, rng)

        report["conditions"][cond] = row

        best_k = row["kshot"].get(str(ks[-1]), {}).get("both", {})
        print(f"{cond:22s} {base_f1:8.4f} {row['ece']:7.4f} "
              f"{row['classwise_ece']:7.4f} {row['brier']:7.4f} "
              f"{row['human_fpr_at_threshold']:8.4f} "
              f"{row['oracle_prior_correction']['macro_f1']:8.4f} "
              f"{best_k.get('macro_f1_mean', float('nan')):8.4f}")

    dest = Path(args.out) if args.out else (
        ROOT / "aicd" / "eval" / "reports" / f"shift_diagnosis_{args.model}.json")
    os.makedirs(dest.parent, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest}")

    # The decision the whole experiment exists to settle.
    s5 = report["conditions"]["s5_compound"]
    orc = s5["oracle_prior_correction"]["macro_f1"]
    print("\n" + "=" * 72)
    print("DECISION GATE (Part G)")
    print("=" * 72)
    print(f"  S5 macro-F1 as measured          : {s5['macro_f1']:.4f}")
    otp = s5["oracle_temperature_and_prior"]
    print(f"  S5 with TRUE target priors       : {orc:.4f}   "
          f"(ceiling for prior correction)")
    print(f"  S5 with best temperature + true  : {otp['macro_f1']:.4f}   "
          f"(T={otp['temperature']:.2f}, ceiling for ANY such correction)")
    for k in ks:
        b = s5["kshot"].get(str(k), {}).get("both")
        if b:
            print(f"  S5 with k={k:<4d} labelled target   : "
                  f"{b['macro_f1_mean']:.4f} +/- {b['macro_f1_sd']:.4f}")
    ceiling = max(orc, otp["macro_f1"])
    verdict = ("BRANCH 2: recalibration recovers much of the loss. The headline "
               "must change." if ceiling > 0.55 else
               "BRANCH 1: the loss is not a correctable prior shift. The thesis "
               "holds and strengthens.")
    print(f"\n  {verdict}")


if __name__ == "__main__":
    main()
