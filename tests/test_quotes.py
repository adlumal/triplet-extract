"""
Tests for direct-quote speech normalization (quotes.py).

Quoted direct speech must not leak its content as an author-level
assertion, and must not produce cross-boundary garbage where the speaker
attaches as an object. After normalization, quoted speech behaves like
its unquoted reported form: content stays in the object of the speech
triple, and transitive inner clauses are attributed to the speaker.
"""

import spacy

from triplet_extract import extract
from triplet_extract.quotes import strip_speech_quotes


def rendered(triplets):
    return {(t.subject, t.relation, t.object) for t in triplets}


def test_trailing_attribution_no_escape():
    """'"The food was amazing," said Tom.' must not yield an author-direct
    (food, was, amazing) nor garbage (food, was, Tom)."""
    triplets = extract('"The food was amazing," said Tom.')
    r = rendered(triplets)
    assert ("Tom", "said", "The food was amazing") in r
    for s, rel, o in r:
        assert not (s in ("food", "The food") and rel == "was" and o == "amazing"), r
        assert o != "Tom", f"cross-boundary garbage: {(s, rel, o)}"


def test_leading_attribution_no_escape():
    """The leading form leaks the same way without the quote-setoff comma
    being dropped; check it is also clean."""
    triplets = extract('Tom said, "The food was amazing."')
    r = rendered(triplets)
    assert ("Tom", "said", "The food was amazing") in r
    for s, rel, o in r:
        assert not (s in ("food", "The food") and rel == "was" and o == "amazing"), r
        assert "said" not in s, f"token-join corruption: {(s, rel, o)}"


def test_quoted_transitive_clause_attributed_to_speaker():
    """A transitive inner clause inside a quote is attributed to the
    speaker, exactly like unquoted reported speech."""
    for text in ['"Sarah ate the cake," said Tom.', 'Tom said, "Sarah ate the cake."']:
        triplets = extract(text)
        attributed = [
            t for t in triplets if (t.subject, t.relation, t.object) == ("Sarah", "ate", "the cake")
        ]
        assert attributed, f"{text}: {rendered(triplets)}"
        assert attributed[0].asserter_chain == ["Tom"]


def test_scare_quotes_are_left_untouched():
    """Quotation marks that are not a speech frame (no ccomp-governing
    verb) are not stripped — abstain rather than mangle."""
    doc = spacy.load("en_core_web_sm")('The so-called "experts" were wrong.')
    assert strip_speech_quotes(doc) is None


def test_quoted_noun_phrase_not_a_speech_frame():
    doc = spacy.load("en_core_web_sm")('He called it a "masterpiece".')
    assert strip_speech_quotes(doc) is None


def test_no_quotes_is_noop():
    doc = spacy.load("en_core_web_sm")("Tom said the food was cold.")
    assert strip_speech_quotes(doc) is None


def test_unquoted_extraction_unchanged_by_quote_module():
    """The presence of the normalization step must not alter extraction of
    ordinary sentences."""
    triplets = extract("Cats love milk.")
    assert rendered(triplets) == {("Cats", "love", "milk")}
