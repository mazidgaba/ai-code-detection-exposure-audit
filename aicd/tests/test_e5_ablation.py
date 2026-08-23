"""E5's ablation, and the comparison that would quietly invalidate it.

Each arm's notebook printed its drop against a single number labelled
"reference": 0.6599. That figure is real, but it is the largest of the six seed
drops in the reference configuration. Comparing four arms against the maximum
of a noisy baseline makes all four look like they shrink the collapse, and
three of them then appear to "improve" on a difference that is entirely seed
noise. The consolidation module exists to replace that comparison, so the first
test here is that it has not drifted back to it.

The second is the finding itself, stated as a property rather than a number:
the spread across four deliberate design changes must remain no larger than the
spread across six random seeds. If a future arm broke that, E5 would no longer
support the sentence the paper draws from it, and this test should fail rather
than let the sentence stand.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from aicd.eval import e5_ablation as E

REPORT = pathlib.Path(E.REPORTS) / "e5_ablation.json"


@pytest.fixture(scope="module")
def rep():
    if not REPORT.exists():
        pytest.skip("run `python -m aicd.eval.e5_ablation` first")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_reference_is_the_seed_spread_not_its_maximum(rep):
    """The yardstick must be the range, and 0.6599 must be inside it, not it."""
    ref = rep["reference"]
    assert ref["drop_max"] == pytest.approx(0.6599, abs=5e-4)
    assert ref["drop_min"] < ref["drop_max"], "a spread, not a point"
    # The maximum is what the notebooks used. It must not be what this reports
    # as the reference level.
    assert ref["drop_mean"] < ref["drop_max"]
    assert ref["ci95"][0] < ref["drop_mean"] < ref["ci95"][1]


def test_no_arm_removes_the_collapse(rep):
    """Every arm must still collapse. If one did not, the paper's claim would
    be about our architecture and the contribution would evaporate."""
    assert rep["any_arm_removes_collapse"] is False
    for tag, a in rep["arms"].items():
        assert a["drop"] > 0.55, f"{tag} drop {a['drop']:.4f} is not a collapse"


def test_design_choices_move_it_less_than_seeds_do(rep):
    """The sentence the paper draws from E5, as an assertion."""
    assert rep["arm_spread"] <= rep["seed_spread"], (
        f"design choices now spread {rep['arm_spread']:.4f}, seeds "
        f"{rep['seed_spread']:.4f}; E5 no longer says what the paper says")


def test_exposure_dominates_architecture(rep):
    """The twin control must move the drop by far more than any design choice.

    This is the comparison the paper actually rests on: if changing the encoder
    mattered as much as changing what the training set contains, the thesis
    would be about model capacity rather than about withholding.
    """
    twin = rep.get("twin_control")
    if twin is None:
        pytest.skip("twin control reports not present")
    assert twin["move"] > 5 * rep["arm_spread"]


def test_log_recovery_cross_checks_against_the_printed_table(tmp_path):
    """Arms recovered from kernel logs take the drop from the summary line, and
    must reject a log whose summary disagrees with its own table."""
    good = (
        "slice                        n  macroF1     wF1     acc     AUC  humFPR\n"
        "s1_in_distribution      31,041   0.9000  0.9326  0.9323  0.9984  0.0201\n"
        "s2_unseen_generator     41,058   0.8678  0.9048  0.9039  0.9988  0.0216\n"
        "s3_unseen_language      56,626   0.5810  0.6783  0.6477  0.9215  0.3138\n"
        "s4_unseen_domain        56,728   0.4092  0.4560  0.4341  0.8534  0.6825\n"
        "s5_compound              4,168   0.3000  0.3130  0.3093  0.8037  0.7717\n"
        "S1 -> S5: 0.9000 -> 0.3000  (drop 0.6000)\n"
    )
    p = tmp_path / "ok.log"
    p.write_text(good, encoding="utf-8")
    by, drop = E.from_log(p)
    assert drop == pytest.approx(0.6000)
    assert by["s5_compound"] == pytest.approx(0.3000)

    bad = tmp_path / "bad.log"
    bad.write_text(good.replace("(drop 0.6000)", "(drop 0.2000)"), encoding="utf-8")
    with pytest.raises(SystemExit):
        E.from_log(bad)
