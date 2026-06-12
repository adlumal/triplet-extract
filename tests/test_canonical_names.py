"""
Tests for canonical-name metadata.

`subject_canonical` is the proper-noun span naming the subject's referent,
derived structurally from the parse: the name must be the subject head
itself or attached to it by appos/flat. A PROPN elsewhere in the span
("the senator from Ohio") modifies the referent rather than naming it and
must NOT become the canonical. Reference for appositives naming the same
referent: Stanford's verb patterns substitute an appositive for its head
in object position (RelationTripleSegmenter.java VERB_PATTERNS
"?>appos {}=appos") and appos insertion is REVERSE_ENTAILMENT
(NaturalLogicRelation.java:209).

With cluster_sources=True, `subject_cluster_canonical` (and
`cluster_canonical` on asserter links) carry the cluster-wide canonical:
the shortest name span across the cluster's mentions — so a bare
"My friend" mention recovers "Sarah" through its cluster.

Hard constraint: metadata only — rendered triplet strings are unaffected.
"""

import spacy

from triplet_extract import OpenIEExtractor, extract
from triplet_extract.clustering import canonical_name, cluster_canonicals, cluster_mentions


def find(triplets, subject):
    return next((t for t in triplets if t.subject == subject), None)


def test_appositive_subject_yields_bare_name():
    triplets = extract("My friend Sarah said the food was cold.")

    appositive = find(triplets, "My friend Sarah")
    assert appositive is not None
    assert appositive.subject_canonical == "Sarah"

    # The headless variant has no name of its own
    bare_head = find(triplets, "My friend")
    assert bare_head is not None
    assert bare_head.subject_canonical is None


def test_compound_name_stays_whole():
    triplets = extract("Mary Jane Watson said it.")
    t = find(triplets, "Mary Jane Watson")
    assert t is not None
    assert t.subject_canonical == "Mary Jane Watson"


def test_compound_appositive_name_stays_whole():
    triplets = extract("My friend Sarah Connor said it.")
    t = find(triplets, "My friend Sarah Connor")
    assert t is not None
    assert t.subject_canonical == "Sarah Connor"


def test_propn_inside_modifier_is_not_a_name():
    """ "Ohio" modifies the senator; it does not name the senator."""
    triplets = extract("The senator from Ohio said it.")
    t = find(triplets, "The senator from Ohio")
    assert t is not None
    assert t.subject_canonical is None


def test_no_propn_subject_is_none():
    triplets = extract("Sarah burned the pasta.")
    assert find(triplets, "Sarah").subject_canonical == "Sarah"
    triplets = extract("The chef burned the pasta.")
    assert find(triplets, "The chef").subject_canonical is None


def test_cluster_canonical_recovers_name_for_headless_mentions():
    ex = OpenIEExtractor(cluster_sources=True)
    triplets = ex.extract_triplet_objects("My friend Sarah said it. Sarah was sure.")

    headless = find(triplets, "My friend")
    assert headless is not None
    assert headless.subject_canonical is None
    assert headless.subject_cluster_canonical == "Sarah"

    bare = find(triplets, "Sarah")
    assert bare is not None
    assert bare.subject_cluster_canonical == "Sarah"


def test_cluster_canonical_is_shortest_name_span():
    nlp = spacy.load("en_core_web_sm")
    clusters = cluster_mentions({"My friend Sarah Connor", "Sarah", "my friend"}, nlp)
    assert len(set(clusters.values())) == 1
    canonicals = cluster_canonicals(clusters, nlp)
    assert list(canonicals.values()) == ["Sarah"]


def test_canonical_name_on_standalone_mention_parse():
    """The helper must detect the head of a standalone mention (ROOT) as
    well as a subtree inside a sentence — spaCy tokens are per-access
    views, so this is an index-comparison regression pin."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("My friend Sarah")
    assert canonical_name([t for t in doc if not t.is_punct]) == "Sarah"
    doc = nlp("the senator from Ohio")
    assert canonical_name([t for t in doc if not t.is_punct]) is None


def test_metadata_only_rendered_strings_unchanged():
    """Canonical computation must never alter rendered strings."""
    triplets = extract("My friend Sarah said the food was cold.")
    rendered = {(t.subject, t.relation, t.object) for t in triplets}
    assert ("My friend Sarah", "said", "the food was cold") in rendered
    # No rendering uses the canonical as a substitute subject string here
    # (promotion renderings are a separate, deliberate feature)
    for t in triplets:
        assert t.to_dict()["subject"] == t.subject
