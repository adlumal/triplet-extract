"""
Tests for appositive-promotion renderings.

Entailment shortening deletes dependent subtrees, so from "My friend
Sarah said X" it can produce "My friend said X" (dropping the appositive
name) but never bare "Sarah said X" (it cannot delete the head and
promote the appositive) — exactly backwards for identity: the most
identity-bearing rendering was the one that never existed.

A non-restrictive appositive names the same referent, so promotion is an
equivalence — stronger than the forward entailments the shortener emits.
Stanford's own verb patterns substitute an appositive for its head in
OBJECT position (RelationTripleSegmenter.java VERB_PATTERNS
"?>appos {}=appos"; segmentVerb takes object = m.getNode("appos"));
promotion extends that device to subjects. The gate is purely
structural: a PROPN appos/flat child under a nominal subject head — no
semantic gating, no word lists.

Bidirectional: every promotion assertion is paired with a non-firing
control (PROPN-headed subjects, PROPN inside modifiers), so over-firing
fails as loudly as the original gap.
"""

from triplet_extract import extract


def renderings(triplets):
    return [(t.subject, t.relation, t.object) for t in triplets]


def test_appositive_subject_promotes_to_bare_name():
    triplets = extract("My friend Sarah said the food was cold.")
    r = renderings(triplets)

    # The promoted rendering exists alongside the originals
    assert ("Sarah", "said", "the food was cold") in r, r
    assert ("My friend Sarah", "said", "the food was cold") in r, r


def test_compound_appositive_promotes_whole_name():
    triplets = extract("Her colleague Dr. Chen said the results were wrong.")
    r = renderings(triplets)

    assert ("Dr. Chen", "said", "the results were wrong") in r, r
    # Never a fragment of the name
    assert not any(s == "Chen" or s == "Dr." for s, _, _ in r), r


def test_propn_in_modifier_does_not_promote():
    """ "Ohio" does not name the senator — no promotion may fire."""
    triplets = extract("The senator from Ohio said it.")
    r = renderings(triplets)

    assert ("Ohio", "said", "it") not in r, r


def test_propn_headed_subject_does_not_duplicate():
    """A subject that already IS the name has nothing to promote."""
    triplets = extract("Mary Jane Watson said it.")
    r = renderings(triplets)

    assert r.count(("Mary Jane Watson", "said", "it")) == 1, r
    assert len(r) == len(set(r)), r


def test_reversed_apposition_unaffected():
    """ "Sarah, my friend, said it." gets bare Sarah via appositive
    DELETION (the entailer's existing device); promotion must not fire
    on the PROPN head and must not create duplicates."""
    triplets = extract("Sarah, my friend, said it.")
    r = renderings(triplets)

    assert ("Sarah", "said", "it") in r, r
    assert len(r) == len(set(r)), r


def test_promoted_rendering_keeps_asserter_chain():
    """Promotion of embedded content must not detach attribution."""
    triplets = extract("Tom said his colleague Sarah burned the pasta.")

    promoted = next(
        (
            t
            for t in triplets
            if (t.subject, t.relation, t.object) == ("Sarah", "burned", "the pasta")
        ),
        None,
    )
    assert promoted is not None, renderings(triplets)
    assert promoted.asserter_chain == ["Tom"]
    assert promoted.subject_canonical == "Sarah"


def test_promotion_no_negation_interference():
    """Promoted renderings inherit the relation untouched — negation
    survives promotion."""
    triplets = extract("My friend Sarah did not say the food was cold.")
    r = renderings(triplets)

    assert ("Sarah", "did not say", "the food was cold") in r, r
    assert not any(s == "Sarah" and "not" not in rel and "say" in rel for s, rel, _ in r), r
