"""
Tests for gated pronominal coreference resolution (resolve_coref).

Port of dcoref's pronoun sieve (Lee et al., 2011) with a uniqueness gate:
substitute a pronoun only when exactly one agreeing PERSON antecedent
exists within the sentence window; otherwise abstain. Opt-in because it
changes rendered triplet strings (default OFF).
"""

import pytest

from triplet_extract import OpenIEExtractor


@pytest.fixture(scope="module")
def coref_extractor():
    return OpenIEExtractor(resolve_coref=True)


@pytest.fixture(scope="module")
def plain_extractor():
    return OpenIEExtractor()


def subjects(triplets):
    return {t.subject for t in triplets}


def rendered(triplets):
    return {(t.subject, t.relation, t.object) for t in triplets}


def test_unique_antecedent_resolves(coref_extractor, plain_extractor):
    """One PERSON in scope: she -> Sarah."""
    text = "My friend Sarah loved them. She is probably lying."

    on = coref_extractor.extract_triplet_objects(text)
    assert ("Sarah", "is", "lying") in rendered(on)
    assert "She" not in subjects(on)

    off = plain_extractor.extract_triplet_objects(text)
    assert ("She", "is", "lying") in rendered(off)


def test_unique_antecedent_resolves_inside_reported_clause(coref_extractor):
    """Resolution applies before extraction, so embedded clauses see it
    too — and attribution metadata composes with it."""
    text = "Obama is the president. Everyone says he has a nice smile."

    on = coref_extractor.extract_triplet_objects(text)
    embedded = [
        t for t in on if (t.subject, t.relation, t.object) == ("Obama", "has", "a nice smile")
    ]
    assert embedded, rendered(on)
    # The resolved clause is still reported speech: asserted by "Everyone"
    assert embedded[0].asserter_chain == ["Everyone"]


def test_two_same_gender_persons_abstain(coref_extractor):
    """Two candidates the gender prior cannot tell apart (both female) are
    ambiguous, so the pronoun is left untouched."""
    text = "Sarah met Mary at the cafe. She ordered coffee."

    on = coref_extractor.extract_triplet_objects(text)
    assert ("She", "ordered", "coffee") in rendered(on)
    assert ("Sarah", "ordered", "coffee") not in rendered(on)
    assert ("Mary", "ordered", "coffee") not in rendered(on)


def test_mixed_gender_pair_disambiguated_by_prior(coref_extractor):
    """When the prior is active, a mixed-gender pair is NOT ambiguous: the
    pronoun's gender selects the one agreeing candidate. Without the prior
    this abstains (both candidates eligible)."""
    text = "Tom met Sarah at the cafe. She ordered coffee."
    on = coref_extractor.extract_triplet_objects(text)
    if coref_extractor._gender_prior is not None:
        assert ("Sarah", "ordered", "coffee") in rendered(on)
        assert ("Tom", "ordered", "coffee") not in rendered(on)
    else:
        assert ("She", "ordered", "coffee") in rendered(on)


def test_winograd_abstains(coref_extractor, plain_extractor):
    """World-knowledge pronouns ("it": no grammatical gender in scope)
    are never substituted."""
    text = "The trophy does not fit in the suitcase because it is too big."

    on = rendered(coref_extractor.extract_triplet_objects(text))
    off = rendered(plain_extractor.extract_triplet_objects(text))
    assert on == off


def test_default_is_off(plain_extractor):
    """resolve_coref defaults to False and leaves pronouns untouched."""
    assert plain_extractor.resolve_coref is False

    text = "Sarah cooked dinner. She burned the pasta."
    off = plain_extractor.extract_triplet_objects(text)
    assert ("She", "burned", "the pasta") in rendered(off)


def test_opt_in_resolves_simple_case(coref_extractor):
    text = "Sarah cooked dinner. She burned the pasta."
    on = coref_extractor.extract_triplet_objects(text)
    assert ("Sarah", "burned", "the pasta") in rendered(on)
    assert "She" not in subjects(on)


def test_dangling_pronoun_gender_depends_on_prior(coref_extractor):
    """A dangling gendered pronoun (referent absent from the text).

    Behavior is bidirectional on the optional name-gender prior:
    - With the `coref` extra: the prior knows "Sarah" is female, conflicts
      with "He", and vetoes the resolution — "He" stays (the v0.4.0
      precision limit is fixed).
    - Without it: candidate gender is UNKNOWN, so the uniqueness gate
      resolves "He" to the lone PERSON "Sarah" — the documented fallback.
    """
    text = "Sarah arrived early. He was annoyed about the delay."
    on = coref_extractor.extract_triplet_objects(text)
    resolved_to_sarah = any(t.subject == "Sarah" and "annoyed" in t.relation for t in on)
    he_left_alone = any(t.subject == "He" for t in on)

    if coref_extractor._gender_prior is not None:
        assert he_left_alone and not resolved_to_sarah, "gender prior should veto He->Sarah"
    else:
        assert resolved_to_sarah, "lexicon-free fallback should resolve He->Sarah"


def test_plural_anaphora_unique_antecedent(coref_extractor):
    """ "they"/"them" resolve to a unique plural noun-phrase antecedent."""
    text = "I love these headphones. They sound amazing."
    on = coref_extractor.extract_triplet_objects(text)
    assert ("these headphones", "sound", "amazing") in rendered(on)
    assert "They" not in subjects(on)


def test_plural_anaphora_ambiguous_abstains(coref_extractor):
    """Two plural candidates -> abstain (no substitution)."""
    text = "The cats chased the dogs. They ran off."
    on = coref_extractor.extract_triplet_objects(text)
    assert not any(
        t.subject in ("The cats", "the dogs", "cats", "dogs") and "ran" in t.relation for t in on
    )
