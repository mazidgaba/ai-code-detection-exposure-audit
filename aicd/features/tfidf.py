"""Character n-gram TF-IDF.

Deliberately character-level, not token-level: it is language-agnostic and,
per AICD Bench, it is the representation that holds up best under distribution
shift -- SVM/LR over TF-IDF beat all six neural encoders on their hardest task.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build(cfg) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(cfg.features.tfidf_ngram_min, cfg.features.tfidf_ngram_max),
        max_features=cfg.features.tfidf_max_features,
        sublinear_tf=cfg.features.tfidf_sublinear,
        min_df=3,
        lowercase=False,
    )
