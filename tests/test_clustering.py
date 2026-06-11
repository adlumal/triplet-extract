"""
Tests for source-identity clustering (clustering.py).

Mentions of the same source ("My friend Sarah", "my friend", "Sarah")
must share a cluster id so downstream consumers do not fragment a single
source's reliability across its surface forms. Metadata only.
"""

import spacy

from triplet_extract import OpenIEExtractor
from triplet_extract.clustering import cluster_mentions


def test_cluster_mentions_groups_same_source():
    nlp = spacy.load("en_core_web_sm")
    clusters = cluster_mentions(["My friend Sarah", "my friend", "Sarah", "Tom", "Bob"], nlp)
    # The three Sarah surface forms share one id
    assert clusters["My friend Sarah"] == clusters["my friend"] == clusters["Sarah"]
    # Distinct people are distinct clusters
    assert clusters["Tom"] != clusters["Sarah"]
    assert clusters["Bob"] != clusters["Sarah"]
    assert clusters["Tom"] != clusters["Bob"]


def test_cluster_does_not_merge_conflicting_names():
    nlp = spacy.load("en_core_web_sm")
    clusters = cluster_mentions(["My friend Sarah", "My friend Mary"], nlp)
    # Same head noun ("friend") but conflicting names -> separate sources
    assert clusters["My friend Sarah"] != clusters["My friend Mary"]


def test_subject_cluster_assigned_when_enabled():
    ex = OpenIEExtractor(cluster_sources=True)
    triplets = ex.extract_triplet_objects(
        "My friend Sarah loved them. My friend recommended them. Bob disagreed strongly."
    )
    by_subject = {t.subject: t.subject_cluster for t in triplets}
    assert by_subject.get("My friend Sarah") is not None
    assert by_subject["My friend Sarah"] == by_subject["My friend"]


def test_asserter_link_cluster_matches_subject_cluster():
    ex = OpenIEExtractor(cluster_sources=True)
    triplets = ex.extract_triplet_objects("Tom said Sarah ate the cake. Tom left.")
    tom_subject = next((t.subject_cluster for t in triplets if t.subject == "Tom"), None)
    link_cluster = None
    for t in triplets:
        for link in t.asserter_links or []:
            if link.asserter == "Tom":
                link_cluster = link.cluster
    assert tom_subject is not None
    assert link_cluster == tom_subject


def test_default_off_leaves_cluster_none():
    ex = OpenIEExtractor()
    triplets = ex.extract_triplet_objects("My friend Sarah loved them.")
    assert all(t.subject_cluster is None for t in triplets)


def test_clustering_does_not_change_strings():
    """Metadata only: the rendered triples are identical with clustering on
    and off."""
    on = OpenIEExtractor(cluster_sources=True).extract_triplet_objects(
        "My friend Sarah loved them. Sarah is honest."
    )
    off = OpenIEExtractor(cluster_sources=False).extract_triplet_objects(
        "My friend Sarah loved them. Sarah is honest."
    )
    assert {(t.subject, t.relation, t.object) for t in on} == {
        (t.subject, t.relation, t.object) for t in off
    }
