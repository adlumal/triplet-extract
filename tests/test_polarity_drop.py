"""
Polarity soundness tests: no rendering may invert the polarity of its source.

The bug class: a triple whose relation text drops a parse-level negation
asserts exactly what the source denied — "Tom did not say that S" rendered
as (Tom | did | say). Three independent failures had to stack for that to
escape:

1. The copula/xcomp-family pattern (corenlp_patterns Pattern 2b) built its
   relation from the auxiliary alone, ignoring the verb's `neg` child.
   Stanford's segmenter refuses to emit rather than strip — it eliminates
   extractions whose chunks carry an unhandled "not"
   (RelationTripleSegmenter.java: segmentVerb's "prohibit 'not'" check;
   the segment() javadoc: "the system has not been written to handle
   negation"). This port's relation convention is richer (aux + negation +
   verb), so the pattern now carries the negation instead.
2. The aux filter classified triples by relation_tokens that disagreed
   with the emitted relation text, and compared dominance per-fragment, so
   a fragment whose only triple was the artifact kept it. Filtering now
   runs at sentence scope with a text-level negation-drop guard
   (aux_filter.drops_parse_negation) that is unconditional.
3. The forward entailer lost polarity annotations at every reparse
   (Stanford deletes graph nodes in place, so polarity persists; the
   reparse is this port's device), letting argument deletion proceed
   inside downward-entailing scopes: "Tom did not say that S" does not
   entail "Tom did not say". Reparses now re-annotate polarity, so the
   per-edge truth check (NaturalLogicRelation.forDependencyDeletion
   projected through Polarity) blocks those deletions again.

Tests are bidirectional house style: every absence assertion on negated
input is paired with a presence assertion on the positive control, so an
overblocking "fix" fails just as loudly as the original bug.
"""

import pytest
import spacy

from triplet_extract import extract
from triplet_extract.openie.forward_entailer import ForwardEntailer
from triplet_extract.openie.polarity_annotator import PolarityAnnotator


def find(triplets, subject, relation, obj):
    """Return the first triplet matching (subject, relation, object)."""
    for t in triplets:
        if t.subject == subject and t.relation == relation and t.object == obj:
            return t
    return None


def relations(triplets):
    return [t.relation for t in triplets]


def renderings(triplets):
    return [f"{t.subject} {t.relation} {t.object}" for t in triplets]


# ---------------------------------------------------------------------------
# The probe sentence and the production-observed instance
# ---------------------------------------------------------------------------


def test_probe_sentence_never_renders_affirmed_say():
    """ "Tom did not say that S" must never render a triple affirming "say"."""
    triplets = extract("Tom did not say that Sarah burned the pasta.")

    # The corrupt renderings: bare-aux relation, or "did say" without "not"
    assert find(triplets, "Tom", "did", "say") is None, renderings(triplets)
    for t in triplets:
        if t.subject == "Tom" and "say" in f"{t.relation} {t.object}":
            assert "not" in t.relation, renderings(triplets)

    # Positive direction: the negation-preserving rendering is present
    assert (
        find(triplets, "Tom", "did not say", "that Sarah burned the pasta") is not None
    ), renderings(triplets)


def test_negated_build_never_renders_affirmed_build():
    """ "Moses did not build the ark." must not yield 'Moses did build'."""
    triplets = extract("Moses did not build the ark.")

    assert "did" not in relations(triplets), renderings(triplets)
    for r in renderings(triplets):
        if "build" in r:
            assert "not" in r, renderings(triplets)

    assert find(triplets, "Moses", "did not build", "the ark") is not None, renderings(triplets)


# ---------------------------------------------------------------------------
# Direct-sentence inversion (no entailer involved): Pattern 2b intransitive
# ---------------------------------------------------------------------------


def test_negated_progressive_keeps_negation_in_relation():
    """Pattern 2b fired on "are not grazing" used to emit (Horses | are | grazing)."""
    triplets = extract("Horses are not grazing peacefully.")

    assert find(triplets, "Horses", "are not", "grazing peacefully") is not None, renderings(
        triplets
    )
    assert find(triplets, "Horses", "are", "grazing peacefully") is None, renderings(triplets)


def test_bare_negated_aux_sentence_keeps_negation():
    """Direct input "Tom did not say." has exactly one sound rendering shape."""
    triplets = extract("Tom did not say.")

    assert find(triplets, "Tom", "did", "say") is None, renderings(triplets)
    assert find(triplets, "Tom", "did not", "say") is not None, renderings(triplets)


# ---------------------------------------------------------------------------
# Contraction forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_relation_part",
    [
        ("Tom didn't say that Sarah burned the pasta.", "didn't say"),
        ("It won't rain.", "won't"),
        ("She doesn't know him.", "doesn't know"),
    ],
)
def test_contracted_negation_survives_in_relation(text, expected_relation_part):
    triplets = extract(text)

    assert any(expected_relation_part in r for r in relations(triplets)), renderings(triplets)
    # No rendering may strip the contraction down to the affirmative aux
    for r in relations(triplets):
        assert r not in ("did", "will", "wo", "does", "did say", "does know"), renderings(triplets)


# ---------------------------------------------------------------------------
# Positive controls — pinned decisions
# ---------------------------------------------------------------------------


def test_positive_aux_artifact_is_dominated_away():
    """Pinned: "Tom did say it" keeps the rich rendering; the redundant
    aux-only artifact (Tom | did | say) is dominated at sentence scope and
    does not survive. (It was merely redundant, never corrupting.)"""
    triplets = extract("Tom did say it.")

    assert find(triplets, "Tom", "did say", "it") is not None, renderings(triplets)
    assert find(triplets, "Tom", "did", "say") is None, renderings(triplets)


def test_positive_progressive_aux_rendering_survives():
    """Pinned: an aux-only rendering with no richer same-subject sibling is
    kept — it is the pattern's raison d'être, and two aux-only siblings
    must not dominate each other."""
    triplets = extract("Horses are grazing peacefully.")

    assert find(triplets, "Horses", "are", "grazing peacefully") is not None, renderings(triplets)
    # The entailed shortening is itself aux-only and must coexist
    assert find(triplets, "Horses", "are", "grazing") is not None, renderings(triplets)


@pytest.mark.parametrize(
    "text, subject, relation, obj",
    [
        (
            "Obama was named 2009 Nobel Peace Prize Laureate.",
            "Obama",
            "was named",
            "2009 Nobel Peace Prize Laureate",
        ),
        ("She knows him.", "She", "knows", "him"),
        ("Sarah burned the pasta.", "Sarah", "burned", "the pasta"),
    ],
)
def test_no_negation_control_sweep(text, subject, relation, obj):
    """Sentences without negation keep their renderings unchanged."""
    triplets = extract(text)
    assert find(triplets, subject, relation, obj) is not None, renderings(triplets)


# ---------------------------------------------------------------------------
# Downward-entailing deletion (entailer level, bidirectional)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def entailer_env():
    nlp = spacy.load("en_core_web_sm")
    return nlp, PolarityAnnotator(nlp), ForwardEntailer(nlp=nlp)


def entail_texts(env, sentence):
    nlp, annotator, entailer = env
    doc = annotator.annotate(nlp(sentence))
    return {str(f).rstrip(".").rstrip() for f in entailer.entail(doc, truth_of_premise=True)}


def test_no_argument_deletion_under_negation(entailer_env):
    """ "Tom did not say that S" does not entail "Tom did not say [anything]":
    argument deletion inside a downward-entailing scope must be blocked,
    even after the determiner-removal reparse."""
    fragments = entail_texts(entailer_env, "Tom did not say that Sarah burned the pasta.")

    assert "Tom did not say" not in fragments, fragments
    assert "Tom did not say Sarah burned" not in fragments, fragments
    # Determiner/complementizer removal is polarity-neutral and stays licensed
    assert "Tom did not say Sarah burned pasta" in fragments, fragments


def test_argument_deletion_in_upward_context_still_licensed(entailer_env):
    """Positive direction: "Tom said that S" DOES entail "Tom said" —
    the gate must not overblock upward-monotone deletions."""
    fragments = entail_texts(entailer_env, "Tom said that Sarah burned the pasta.")

    assert "Tom said" in fragments, fragments
    assert "Tom said Sarah burned pasta" in fragments, fragments


# ---------------------------------------------------------------------------
# The aux_filter backstop in isolation
# ---------------------------------------------------------------------------


def test_drops_parse_negation_guard(entailer_env):
    """The filter-level guard catches a polarity-dropping triple regardless
    of which pattern produced it, and passes the sound rendering."""
    from triplet_extract.corenlp_patterns import Triple
    from triplet_extract.openie.aux_filter import drops_parse_negation

    nlp, _, _ = entailer_env
    doc = nlp("Tom did not say.")
    tom, did, not_, say = doc[0], doc[1], doc[2], doc[3]
    assert not_.dep_ == "neg"

    corrupt = Triple(
        subject="Tom",
        relation="did",
        object="say",
        subject_tokens=[tom],
        relation_tokens=[did],
        object_tokens=[say],
        relation_head=did,
    )
    assert drops_parse_negation(corrupt)

    sound = Triple(
        subject="Tom",
        relation="did not",
        object="say",
        subject_tokens=[tom],
        relation_tokens=[did, not_],
        object_tokens=[say],
        relation_head=did,
    )
    assert not drops_parse_negation(sound)
