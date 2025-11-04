"""Tests for pytriplets."""

from triplet_extract import OpenIEExtractor, Triplet, extract


def test_basic_extraction():
    """Test basic triplet extraction."""
    triplets = extract("Cats love milk.")
    assert len(triplets) > 0
    assert all(isinstance(t, Triplet) for t in triplets)
    # Should extract at least one triplet with "Cats" as subject
    subjects = [t.subject for t in triplets]
    assert any("Cats" in s for s in subjects)


def test_no_malformed_hyphens():
    """Regression test: no orphaned hyphens in triplets."""
    text = "The emergence of antibiotic-resistant strains is concerning."
    triplets = extract(text)

    # Check no orphaned hyphens (like "of-" or "-resistant")
    for t in triplets:
        assert "of-" not in t.subject, f"Malformed subject: {t.subject}"
        assert "of-" not in t.object, f"Malformed object: {t.object}"
        assert " -" not in t.subject, f"Orphaned hyphen in subject: {t.subject}"
        assert " -" not in t.object, f"Orphaned hyphen in object: {t.object}"


def test_batch_processing():
    """Test batch extraction."""
    extractor = OpenIEExtractor()
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    results = extractor.extract_batch(texts, progress=False)

    assert len(results) == 3
    assert all(isinstance(r, list) for r in results)


def test_extractor_options():
    """Test extractor with different options."""
    # With entailment
    extractor1 = OpenIEExtractor(enable_entailment=True)
    triplets1 = extractor1.extract_triplet_objects("Cats love milk.")

    # Without entailment (should have fewer triplets)
    extractor2 = OpenIEExtractor(enable_entailment=False)
    triplets2 = extractor2.extract_triplet_objects("Cats love milk.")

    assert len(triplets1) >= len(triplets2)


def test_triplet_attributes():
    """Test triplet object attributes."""
    triplets = extract("Dogs eat bones.")
    assert len(triplets) > 0

    t = triplets[0]
    assert hasattr(t, "subject")
    assert hasattr(t, "relation")
    assert hasattr(t, "object")
    assert hasattr(t, "confidence")
    assert isinstance(t.subject, str)
    assert isinstance(t.relation, str)
    assert isinstance(t.object, str)
    assert isinstance(t.confidence, float)


def test_empty_text():
    """Test extraction from empty text."""
    triplets = extract("")
    assert isinstance(triplets, list)
    assert len(triplets) == 0


def test_scientific_text():
    """Test extraction from scientific abstract."""
    text = "Marine natural products have been a source of drug discovery."
    triplets = extract(text)

    # Should extract multiple valid triplets
    assert len(triplets) > 0

    # All triplets should have non-empty components
    for t in triplets:
        assert len(t.subject.strip()) > 0
        assert len(t.relation.strip()) > 0
        assert len(t.object.strip()) > 0
