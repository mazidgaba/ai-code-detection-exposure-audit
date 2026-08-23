"""The gradient-checkpointing path, tested without downloading a model.

ModernBERT-large exhausted a 16 GB T4 with 57 MiB to spare, and gradient
checkpointing is the fix that leaves the experiment unchanged: identical
gradients and updates, memory traded for time. That makes it worth exactly one
thing going wrong to be caught here rather than an hour into a GPU session.

Two failures matter. The flag silently doing nothing would send the run back
into the same out-of-memory crash, and an encoder that cannot checkpoint would
do the same. Both are asserted below against a stub encoder, so these run
offline and in milliseconds.
"""
from __future__ import annotations

import sys
import types

import pytest

import aicd.models.modernbert_triplet as M


class _Cfg(dict):
    """Attribute access over a dict, matching how configs are read.

    Raising AttributeError rather than KeyError for a missing key is the whole
    point of this stub. `build()` reads the flag with a getattr default, and
    getattr only falls back on AttributeError. A stub that raised KeyError
    would fail a test the real config passes, which is exactly what the first
    version of this file did.
    """

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return _Cfg(v) if isinstance(v, dict) else v


def _cfg(**over):
    base = {"base_model": "stub/encoder", "projection_dim": 8, "num_classes": 4,
            "triplet_weight": 0.1}
    base.update(over)
    return _Cfg({"modernbert": base})


class _Encoder:
    """Stands in for a HuggingFace encoder, recording whether it was asked to
    checkpoint and with which arguments."""

    def __init__(self, supports=True):
        self.config = types.SimpleNamespace(hidden_size=16)
        self.enabled = None
        if not supports:
            del self.__class__.gradient_checkpointing_enable

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.enabled = gradient_checkpointing_kwargs or {}

    # TLModel only needs the encoder to be a module-like object here; build()
    # does not run a forward pass.
    def parameters(self):
        return iter(())


@pytest.fixture
def patched(monkeypatch):
    """Intercept the transformers import inside build()."""
    made = {}

    class _Auto:
        @staticmethod
        def from_pretrained(name):
            enc = _Encoder(supports=made.get("supports", True))
            made["enc"] = enc
            return enc

    class _Tok:
        @staticmethod
        def from_pretrained(name):
            return object()

    mod = types.ModuleType("transformers")
    mod.AutoModel = _Auto
    mod.AutoTokenizer = _Tok
    monkeypatch.setitem(sys.modules, "transformers", mod)
    monkeypatch.setattr(M, "TLModel", lambda enc, hid, **kw: ("model", enc))
    return made


def test_flag_off_leaves_the_encoder_alone(patched):
    M.build(_cfg())
    assert patched["enc"].enabled is None


def test_flag_on_actually_enables_it(patched):
    """The failure this guards against is the flag being accepted and ignored,
    which would send the run straight back into the same OOM."""
    M.build(_cfg(gradient_checkpointing=True))
    assert patched["enc"].enabled is not None


def test_it_asks_for_non_reentrant_checkpointing(patched):
    """Reentrant checkpointing is the older implementation and interacts badly
    with AMP and with parameters that do not require grad."""
    M.build(_cfg(gradient_checkpointing=True))
    assert patched["enc"].enabled.get("use_reentrant") is False


def test_an_encoder_that_cannot_checkpoint_fails_loudly(patched, monkeypatch):
    """Better to stop here than to discover it when the card fills up."""
    class _NoSupport:
        def __init__(self):
            self.config = types.SimpleNamespace(hidden_size=16)

    mod = sys.modules["transformers"]
    monkeypatch.setattr(mod.AutoModel, "from_pretrained",
                        staticmethod(lambda name: _NoSupport()))
    with pytest.raises(SystemExit):
        M.build(_cfg(gradient_checkpointing=True))
