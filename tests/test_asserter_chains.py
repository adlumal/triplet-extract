"""
Tests for asserter-chain attribution metadata.

Content embedded under attitude/speech verbs (Stanford's
INDIRECT_SPEECH_LEMMAS, ClauseSplitterSearchProblem.java:100-102) is not
asserted by the document author; the asserter_chain field records who
asserts each triplet, outermost asserter first. None = asserted directly.

Hard constraint: this is metadata only — rendered triplet strings must be
identical with and without the chain computation.
"""

import spacy

from triplet_extract import extract
from triplet_extract.corenlp_patterns import CoreNLPStyleExtractor


def find(triplets, subject, relation, obj):
    """Return the first triplet matching (subject, relation, object)."""
    for t in triplets:
        if t.subject == subject and t.relation == relation and t.object == obj:
            return t
    return None


def test_reported_clause_carries_asserter():
    """The reported clause is asserted by the speaker, not the author."""
    triplets = extract("Tom said Sarah claimed the food was cold.")

    reported = find(triplets, "Sarah", "claimed", "the food was cold")
    assert reported is not None, [t.to_dict() for t in triplets]
    assert reported.asserter_chain == ["Tom"]

    author = find(triplets, "Tom", "said", "Sarah claimed the food was cold")
    assert author is not None
    assert author.asserter_chain is None


def test_nested_reported_speech_chain_is_outermost_first():
    """Two levels of reporting produce a two-element chain."""
    triplets = extract("Tom said Sarah claimed the chef burned the pasta.")

    innermost = find(triplets, "the chef", "burned", "the pasta")
    assert innermost is not None, [t.to_dict() for t in triplets]
    assert innermost.asserter_chain == ["Tom", "Sarah"]

    middle = find(triplets, "Sarah", "claimed", "the chef burned the pasta")
    assert middle is not None
    assert middle.asserter_chain == ["Tom"]

    outer = find(triplets, "Tom", "said", "Sarah claimed the chef burned the pasta")
    assert outer is not None
    assert outer.asserter_chain is None


def test_attitude_verb_carries_asserter():
    """Attitude verbs ("think", "believe") attribute like speech verbs."""
    triplets = extract("The waiter thinks the chef believes the soup is ready.")

    inner = find(triplets, "the chef", "believes", "the soup is ready")
    assert inner is not None, [t.to_dict() for t in triplets]
    assert inner.asserter_chain == ["The waiter"]


def test_conj_shared_subject_resolution():
    """
    Conjoined verbs share the first conjunct's subject: in "Sarah loved
    them and said ...", "said" has no nsubj of its own — the resolver
    must walk the conj chain to find "My friend Sarah".
    """
    nlp = spacy.load("en_core_web_sm")
    segmenter = CoreNLPStyleExtractor(nlp=nlp)
    doc = nlp("My friend Sarah loved them and said they were perfect.")

    said = next(t for t in doc if t.text == "said")
    subject = segmenter._resolve_clause_subject(said)
    assert subject is not None
    assert segmenter._get_subtree_text(subject) == "My friend Sarah"


def test_adversarial_review_praise_is_never_author_direct():
    """
    "My friend Sarah loved them and said they were perfect.": natural
    logic refuses to emit the praise as a free-standing author assertion
    (ccomp content is not entailed). The praise appears only either (a)
    inside the object of an author-level rendering, or (b) in a "said"-
    rendering that carries an asserter chain — never as an author-direct
    (chain None) standalone claim that "they were perfect".
    """
    triplets = extract("My friend Sarah loved them and said they were perfect.")
    assert len(triplets) > 0

    for t in triplets:
        praise_is_the_claim = "perfect" in t.object and "said" not in t.relation
        if praise_is_the_claim and t.asserter_chain is None:
            # Allowed only when the praise sits inside a larger object that
            # is itself reported (e.g. "... said they were perfect"); a bare
            # author-direct praise claim is the failure we guard against.
            assert "said" in t.object, f"author-direct praise leaked: {t.to_dict()}"

    # The praise content is present somewhere
    assert any("perfect" in t.object for t in triplets)


def test_no_reported_speech_control():
    """Plain assertions carry no chain."""
    for sentence in ["Cats love milk.", "A cheetah runs faster than a dog."]:
        triplets = extract(sentence)
        assert len(triplets) > 0
        for t in triplets:
            assert t.asserter_chain is None, t.to_dict()


def test_metadata_does_not_change_rendered_strings():
    """The chain rides alongside: rendered strings are the same set the
    extraction produced before attribution metadata existed."""
    triplets = extract("Tom said Sarah claimed the food was cold.")
    rendered = {(t.subject, t.relation, t.object) for t in triplets}
    assert ("Tom", "said", "Sarah claimed the food was cold") in rendered
    assert ("Sarah", "claimed", "the food was cold") in rendered


def test_to_dict_includes_chain():
    triplets = extract("Tom said Sarah claimed the food was cold.")
    reported = find(triplets, "Sarah", "claimed", "the food was cold")
    assert reported.to_dict()["asserter_chain"] == ["Tom"]

    author = find(triplets, "Tom", "said", "Sarah claimed the food was cold")
    assert author.to_dict()["asserter_chain"] is None


def link_for(triplets, subject, relation, obj):
    t = find(triplets, subject, relation, obj)
    assert t is not None and t.asserter_links, [x.to_dict() for x in triplets]
    return t.asserter_links


def test_non_speech_complement_verb_attributes():
    """A complement-taking verb outside Stanford's 8 speech lemmas still
    attributes its embedded content; speech_act marks it False."""
    triplets = extract("Tom denied Sarah ate the cake.")
    links = link_for(triplets, "Sarah", "ate", "the cake")
    assert len(links) == 1
    assert links[0].asserter == "Tom"
    assert links[0].verb == "deny"
    assert links[0].speech_act is False
    assert links[0].negated is False


def test_evidential_verb_attributes_to_source():
    """Evidential constructions ("the study shows X") attribute the
    embedded finding to the evidence source."""
    triplets = extract("The study shows smoking causes cancer.")
    links = link_for(triplets, "smoking", "causes", "cancer")
    assert links[0].asserter == "The study"
    assert links[0].verb == "show"
    assert links[0].speech_act is False


def test_negated_speech_verb_sets_negated_flag():
    """ "did not say" must be distinguishable from "said" so a consumer
    does not read denial as endorsement."""
    triplets = extract("Tom did not say Sarah ate the cake.")
    links = link_for(triplets, "Sarah", "ate", "the cake")
    assert links[0].asserter == "Tom"
    assert links[0].verb == "say"
    assert links[0].speech_act is True
    assert links[0].negated is True


def test_speech_act_flag_true_for_canonical_verbs():
    triplets = extract("Tom said Sarah claimed the food was cold.")
    links = link_for(triplets, "Sarah", "claimed", "the food was cold")
    assert links[0].verb == "say"
    assert links[0].speech_act is True
    assert links[0].negated is False


def test_links_to_dict_shape():
    triplets = extract("Tom denied Sarah ate the cake.")
    t = find(triplets, "Sarah", "ate", "the cake")
    d = t.to_dict()
    assert d["asserter_links"] == [
        {
            "asserter": "Tom",
            "verb": "deny",
            "construction": "ccomp",
            "speech_act": False,
            "negated": False,
        }
    ]
