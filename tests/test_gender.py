"""
Tests for the optional embedding name-gender prior (gender.py).

Skipped entirely when the `coref` extra (sentence-transformers) is not
installed — the prior is optional and the resolver falls back to its
lexicon-free uniqueness behavior without it.
"""

import pytest

pytest.importorskip("sentence_transformers")

from triplet_extract.gender import NameGenderPrior  # noqa: E402


@pytest.fixture(scope="module")
def prior():
    return NameGenderPrior()


def test_confident_female_names(prior):
    for name in ["Sarah", "Maria", "Emily"]:
        assert prior.gender(name) == "FEMALE", name


def test_confident_male_names(prior):
    for name in ["Tom", "Obama"]:
        assert prior.gender(name) == "MALE", name


def test_unisex_names_abstain(prior):
    # Genuinely ambiguous given names fall inside the abstain band
    assert prior.gender("Alex") is None


def test_cache_returns_stable_result(prior):
    first = prior.gender("Sarah")
    assert prior.gender("Sarah") == first
    assert "sarah" in prior._cache


def test_empty_name_is_unknown(prior):
    assert prior.gender("") is None
    assert prior.gender("   ") is None
