"""Numerical correctness tests for the Fast-DetectGPT curvature statistic.

Why this file exists. The paper originally justified Branch C by comparing our
measured AUC of 0.683 against numbers reported for Fast-DetectGPT in the Droid
suite. Review established that those published figures are weighted F1, not
AUC, so the comparison was between incompatible metrics and was withdrawn.

That left the implementation with no evidence of correctness. These tests
supply it directly, by checking the estimator against values computed in closed
form from Eq. (2) rather than against a published score for a different metric:

    d(x) = sum_t [ log p(x_t | x_<t) - mu_t ]  /  sqrt( sum_t var_t )

with mu_t = E_{v~p}[log p(v)] and var_t = Var_{v~p}[log p(v)].

Run:  python -m pytest aicd/tests/test_fastdetect.py -q
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from aicd.models.fastdetect import curvature


class FakeEncoding(dict):
    """Stands in for transformers' BatchEncoding, which supports .to(device)."""

    def to(self, device):
        return FakeEncoding({k: v.to(device) for k, v in self.items()})


class FakeTok:
    """Tokeniser stub returning a fixed id sequence."""

    def __init__(self, ids):
        self.ids = ids

    def __call__(self, code, truncation=True, max_length=512, return_tensors="pt"):
        return FakeEncoding({"input_ids": torch.tensor([self.ids])})


class FakeOut:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Language-model stub emitting fixed logits, independent of input."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, **kw):
        return FakeOut(self._logits)


def reference_d(logits, target) -> float:
    """Eq. (2) evaluated directly, in plain Python, as the oracle."""
    num, var_sum = 0.0, 0.0
    for t in range(logits.shape[0]):
        row = logits[t].tolist()
        m = max(row)
        z = sum(math.exp(v - m) for v in row)
        logp = [(v - m) - math.log(z) for v in row]
        p = [math.exp(lp) for lp in logp]

        mu = sum(pi * lpi for pi, lpi in zip(p, logp))
        var = sum(pi * lpi * lpi for pi, lpi in zip(p, logp)) - mu * mu

        num += logp[target[t]] - mu
        var_sum += var
    return num / math.sqrt(max(var_sum, 1e-9))


def run(logits, ids):
    """Drive curvature() with stubs. ids has one more element than logits rows."""
    tok, model = FakeTok(ids), FakeModel(logits.unsqueeze(0))
    return curvature("x", tok, model, torch.device("cpu"))


def test_matches_closed_form():
    """The estimator must equal Eq. (2) computed independently."""
    torch.manual_seed(0)
    n_pos, vocab = 11, 7
    logits = torch.randn(n_pos + 1, vocab)
    ids = torch.randint(0, vocab, (n_pos + 1,)).tolist()

    got = run(logits, ids)
    want = reference_d(logits[:-1], ids[1:])
    assert got == pytest.approx(want, rel=1e-4, abs=1e-4), (
        f"curvature() {got} disagrees with the closed form {want}"
    )


def test_uniform_distribution_gives_zero():
    """Under a uniform predictive distribution every token is the mean.

    log p is identical for all v, so observed == mu at every position and the
    numerator vanishes. The variance also vanishes, and the clamp keeps this
    finite rather than dividing by zero.
    """
    n_pos, vocab = 9, 5
    logits = torch.zeros(n_pos + 1, vocab)
    ids = [i % vocab for i in range(n_pos + 1)]
    assert abs(run(logits, ids)) < 1e-3


def test_sign_separates_likely_from_unlikely_tokens():
    """The statistic's sign is the whole detection signal.

    A sequence made of the argmax token at every step sits above the
    conditional mean and must score positive; a sequence of the least likely
    token must score negative. Machine text behaves like the former.
    """
    torch.manual_seed(1)
    n_pos, vocab = 13, 9
    logits = torch.randn(n_pos + 1, vocab) * 2.0

    top = logits[:-1].argmax(dim=-1).tolist()
    bot = logits[:-1].argmin(dim=-1).tolist()

    d_top = run(logits, [0] + top)
    d_bot = run(logits, [0] + bot)

    assert d_top > 0 > d_bot, f"expected {d_top} > 0 > {d_bot}"
    assert d_top > d_bot


def test_short_input_returns_zero():
    """Sequences below the minimum length are declined, not extrapolated."""
    logits = torch.randn(4, 6)
    assert run(logits, [1, 2, 3, 4]) == 0.0


def test_invariant_to_logit_shift():
    """Adding a constant to a row leaves the softmax, and so d(x), unchanged."""
    torch.manual_seed(2)
    n_pos, vocab = 10, 6
    logits = torch.randn(n_pos + 1, vocab)
    ids = torch.randint(0, vocab, (n_pos + 1,)).tolist()

    a = run(logits, ids)
    b = run(logits + 4.2, ids)
    assert a == pytest.approx(b, rel=1e-4, abs=1e-4)


# ---------------------------------------------------------------------------
# Batching. Added when the scorer was moved off one-sample-at-a-time so it
# could run on a GPU inside a session limit.
#
# Two failures would be invisible in the output and fatal to the result. If the
# padding mask is wrong, pad positions contribute to the numerator and the
# variance, so a sequence's score depends on which other sequences happened to
# share its batch. If the chunked vocabulary reduction is wrong, every score is
# wrong by an amount that varies with chunk size. Both are asserted against the
# single-sample path, which the tests above tie to the closed form.
# ---------------------------------------------------------------------------

from aicd.models.fastdetect import _row_stats, curvature_batch  # noqa: E402


class ListTok:
    """Tokeniser stub: "3,1,4" -> [3, 1, 4]. Supports padding, like the real one."""

    pad_token = "<pad>"
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, code, truncation=True, max_length=512, padding=False,
                 return_tensors="pt"):
        seqs = [code] if isinstance(code, str) else list(code)
        ids = [[int(x) for x in s.split(",")][:max_length] for s in seqs]
        width = max(len(i) for i in ids) if padding else len(ids[0])
        out, mask = [], []
        for i in ids:
            pad = width - len(i)
            out.append(i + [0] * pad)
            mask.append([1] * len(i) + [0] * pad)
        return FakeEncoding({"input_ids": torch.tensor(out),
                             "attention_mask": torch.tensor(mask)})


class SeqModel:
    """Logits determined by the input ids alone.

    This is what makes the comparison meaningful: a sequence produces the same
    logits whether it is scored alone or sitting inside a padded batch, so any
    difference in the resulting score is the batching code's doing.
    """

    def __init__(self, vocab: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.table = torch.randn(vocab, vocab, generator=g) * 1.5

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        return FakeOut(self.table[input_ids])


VOCAB = 32
SEQS = ["3,1,4,1,5,9,2,6,5,3,5",                      # 11
        "2,7,1,8,2,8,1,8,2,8,4,5,9,0,4,5",            # 16
        "1,1,2,3,5,8,13,21,3,4",                      # 10
        "9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7,8,9,0"]    # 20


def test_batched_equals_single_across_ragged_lengths():
    """The padding mask, asserted rather than assumed.

    The four sequences differ in length, so in one batch three of them are
    padded. Each must score exactly what it scores alone.
    """
    tok, model, dev = ListTok(), SeqModel(VOCAB), torch.device("cpu")
    one = [curvature(s, tok, model, dev) for s in SEQS]
    many = curvature_batch(SEQS, tok, model, dev)
    for s, a, b in zip(SEQS, one, many):
        assert a == pytest.approx(b, rel=1e-4, abs=1e-4), (
            f"{s[:12]}... scored {a} alone and {b} in a batch")


def test_score_does_not_depend_on_batch_neighbours():
    """The same sequence in three different batches must give one answer."""
    tok, model, dev = ListTok(), SeqModel(VOCAB), torch.device("cpu")
    target = SEQS[2]
    alone = curvature_batch([target], tok, model, dev)[0]
    with_longest = curvature_batch([target, SEQS[3]], tok, model, dev)[0]
    with_all = curvature_batch([target] + SEQS, tok, model, dev)[0]
    assert alone == pytest.approx(with_longest, rel=1e-4, abs=1e-4)
    assert alone == pytest.approx(with_all, rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("chunk", [1, 3, 8, 1024])
def test_chunked_reduction_is_invariant_to_chunk_size(chunk):
    """Chunking exists for memory. It must not touch the arithmetic."""
    tok, model, dev = ListTok(), SeqModel(VOCAB), torch.device("cpu")
    ref = curvature(SEQS[1], tok, model, dev, chunk=256)
    got = curvature(SEQS[1], tok, model, dev, chunk=chunk)
    assert ref == pytest.approx(got, rel=1e-5, abs=1e-5)


def test_row_stats_matches_a_direct_computation():
    """_row_stats against the same quantities computed densely in one shot."""
    torch.manual_seed(3)
    logits = torch.randn(37, 19) * 2.0
    targets = torch.randint(0, 19, (37,))

    obs, mu, var = _row_stats(logits, targets, chunk=5)

    lp = torch.log_softmax(logits.float(), dim=-1)
    p = lp.exp()
    want_mu = (p * lp).sum(-1)
    want_var = (p * lp.square()).sum(-1) - want_mu.square()
    want_obs = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(obs, want_obs, atol=1e-5)
    assert torch.allclose(mu, want_mu, atol=1e-5)
    assert torch.allclose(var, want_var, atol=1e-5)


def test_short_sequences_are_declined_inside_a_batch():
    """The minimum-length rule must survive batching.

    A three-token file batched with long ones would otherwise be scored on two
    positions and reported as a confident prediction.
    """
    tok, model, dev = ListTok(), SeqModel(VOCAB), torch.device("cpu")
    got = curvature_batch(["1,2,3", SEQS[0]], tok, model, dev)
    assert got[0] == 0.0
    assert got[1] != 0.0
