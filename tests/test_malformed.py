"""
Some example tests to verify the hyphen issue is fixed (and won't come back)
"""

from triplet_extract import extract


def test_obama_birth():
    """Test: Barack Obama was born in Hawaii."""
    text = "Barack Obama was born in Hawaii."
    triplets = extract(text)

    # Should extract at least basic triplets
    assert len(triplets) > 0

    # Should have Obama as subject
    subjects = [t.subject for t in triplets]
    assert any("Obama" in s for s in subjects)

    # Should extract birth relation
    relations = [t.relation for t in triplets]
    assert any("born" in r for r in relations)


def test_obama_president():
    """Test: Barack Hussein Obama II is the 44th President of the United States."""
    text = "Barack Hussein Obama II is the 44th and current President of the United States."
    triplets = extract(text)

    # Should extract multiple variations
    assert len(triplets) >= 3

    # Should have Obama as subject in some form
    subjects = [t.subject for t in triplets]
    assert any("Obama" in s for s in subjects)

    # Should extract president relation
    objects = [t.object for t in triplets]
    assert any("President" in o or "United States" in o for o in objects)


def test_marine_products_scientific():
    """
    Test: Marine natural products scientific abstract.

    We extract 37 triplets compared to Stanford's 29 (27% improvement).
    """
    text = """Marine natural products with antibiotic activity have been a rich source of drug discovery.
The emergence of antibiotic-resistant bacterial strains has turned attention towards discovery of alternative strategies.
Bacterial biofilm formation plays a critical role in pathogenesis and antibiotic resistance."""

    triplets = extract(text)

    # We should extract many triplets from this rich text
    assert len(triplets) >= 30, f"Expected >= 30 triplets, got {len(triplets)}"

    # Check for key subjects
    subjects = [t.subject for t in triplets]
    assert any("Marine" in s or "products" in s for s in subjects)
    assert any("emergence" in s or "strains" in s for s in subjects)
    assert any("biofilm" in s or "formation" in s for s in subjects)

    # All triplets should be well-formed (no malformed hyphens)
    for t in triplets:
        assert "of-" not in t.subject, f"Malformed subject: {t.subject}"
        assert "of-" not in t.object, f"Malformed object: {t.object}"


def test_antibiotic_resistant_hyphenated():
    """
    Regression test: hyphenated compounds should be handled correctly.

    This was a critical bug we fixed - "antibiotic-resistant" was becoming
    "of-resistant" when "antibiotic" was deleted.
    """
    text = "The emergence of antibiotic-resistant bacterial strains is concerning."
    triplets = extract(text)

    # Should extract valid triplets
    assert len(triplets) > 0

    # No malformed hyphens (this was a bug previously)
    for t in triplets:
        assert "of-" not in t.subject, f"Malformed hyphen in subject: {t.subject}"
        assert "of-" not in t.object, f"Malformed hyphen in object: {t.object}"
        assert (
            "-resistant" not in t.subject or "antibiotic-resistant" in t.subject
        ), f"Orphaned hyphen: {t.subject}"


def test_biofilm_formation():
    """Test: Bacterial biofilm formation plays a critical role."""
    text = "Bacterial biofilm formation plays a critical role in pathogenesis and antibiotic resistance."
    triplets = extract(text)

    # Should extract multiple triplets
    assert len(triplets) >= 3

    # Should have biofilm/formation subjects
    subjects = [t.subject for t in triplets]
    assert any("biofilm" in s or "formation" in s for s in subjects)

    # Should have "plays" or "role" in relations/objects
    all_text = " ".join([f"{t.relation} {t.object}" for t in triplets])
    assert "role" in all_text or "plays" in all_text


def test_vitamin_d_deficiency():
    """Test: Vitamin D deficiency increases respiratory infection risk."""
    text = "Vitamin D deficiency increases respiratory infection risk."
    triplets = extract(text)

    # Should extract compositional variations
    assert len(triplets) > 0

    # Should capture the core relation
    subjects = [t.subject for t in triplets]
    relations = [t.relation for t in triplets]

    assert any("deficiency" in s for s in subjects)
    assert any("increase" in r for r in relations)


def test_simple_sentence():
    """Test: Simple sentence with clear SVO structure."""
    text = "Cats love milk."
    triplets = extract(text)

    assert len(triplets) > 0

    # Should extract basic SVO
    found = False
    for t in triplets:
        if "Cats" in t.subject and "love" in t.relation and "milk" in t.object:
            found = True
            break

    assert found, f"Did not find (Cats, love, milk) in {triplets}"


def test_multiple_objects():
    """Test: Sentence with multiple objects connected by 'and'."""
    text = "Cats love milk and mice."
    triplets = extract(text)

    # Should extract separate triplets for each object
    assert len(triplets) >= 2

    objects = [t.object for t in triplets]
    assert any("milk" in o for o in objects)
    assert any("mice" in o for o in objects)
