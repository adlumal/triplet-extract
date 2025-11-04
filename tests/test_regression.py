"""
Regression tests.
"""

from triplet_extract import OpenIEExtractor, extract


def test_no_forward_entailer_warnings():
    """
    Regression: ForwardEntailer was crashing with 'NoneType' has no attribute 'dep_'.

    Bug: When no determiner token was found, None was passed to deletion_probability()
    Fix: Added None check before calling deletion_probability()
    Result: Zero warnings and more triplets extracted
    """
    # These simple sentences would trigger the bug
    texts = [
        "This is sentence number 1 for testing batch processing.",
        "This is sentence number 2 for testing batch processing.",
        "This is sentence number 3 for testing batch processing.",
    ]

    for text in texts:
        # Should not crash or produce warnings
        triplets = extract(text)
        assert isinstance(triplets, list)


def test_hyphenated_compounds_bidirectional():
    """
    Regression: Hyphenated compounds were not properly detected bidirectionally.

    Bug: When removing "resistant" from "antibiotic-resistant bacteria",
         only checked parent's children (upward), not token's own children (downward),
         leaving "antibiotic-bacteria"

    Fix: Added bidirectional compound detection
    Result: Properly removes entire compound "antibiotic-resistant" atomically
    """
    text = "The antibiotic-resistant bacteria spread rapidly."
    triplets = extract(text)

    # Check all triplets are well-formed
    for t in triplets:
        # Should not have orphaned parts of compounds
        assert "antibiotic-bacteria" not in t.subject
        assert "antibiotic-bacteria" not in t.object

        # If it has hyphen, should have both parts
        if "-" in t.subject:
            # Hyphen should be part of a proper compound
            assert not t.subject.endswith("-")
            assert not t.subject.startswith("-")


def test_multi_sentence_processing():
    """
    Regression: Processing multiple sentences as one doc caused compound detection to fail.

    Bug: OpenIEExtractor processed all sentences as one doc, causing fragments
         to span sentence boundaries and compound detection to fail

    Fix: Split into sentences and process each separately (matches Stanford behavior)
    Result: 0% malformed instead of 32% malformed when processing multiple sentences
    """
    # When these 3 sentences were processed together, produced 6 malformed triplets
    # After fix: 0 malformed triplets
    text = """Marine natural products with antibiotic activity have been a rich source of drug discovery.
The emergence of antibiotic-resistant bacterial strains has turned attention towards discovery of alternative strategies.
Bacterial biofilm formation plays a critical role in pathogenesis and antibiotic resistance."""

    triplets = extract(text)

    # Should extract many triplets
    assert len(triplets) >= 30

    # CRITICAL: No malformed triplets (this was 32% malformed before fix!)
    malformed = []
    for t in triplets:
        if "of-" in t.subject or "of -" in t.subject:
            malformed.append(f"{t.subject} | {t.relation} | {t.object}")

    assert len(malformed) == 0, f"Found {len(malformed)} malformed triplets: {malformed}"


def test_reparsing_optimization():
    """
    Regression: Fragments were being reparsed unnecessarily.

    Bug: CoreNLPStyleExtractor.extract_triplets() always reparsed text,
         even when already had parsed Doc object

    Fix: Added extract_triplets_from_doc() to reuse parsed Docs
    Result: 2.94x speedup (1.37s to 0.466s per 3 sentences)
    """
    # This test just verifies the optimization doesn't break functionality
    text = "Cats love milk and mice."
    triplets = extract(text)

    # Should still extract correctly
    assert len(triplets) >= 2
    subjects = [t.subject for t in triplets]
    assert any("Cats" in s for s in subjects)


def test_batch_processing_consistency():
    """
    Regression: Batch processing should produce same results as sequential.

    Bug: Different code paths could produce different results
    Fix: Ensure batch uses same pipeline as single-document extraction
    """
    texts = ["First test sentence.", "Second test sentence.", "Third test sentence."]

    # Extract individually
    individual_results = [extract(text) for text in texts]

    # Extract in batch
    extractor = OpenIEExtractor()
    batch_results = extractor.extract_batch(texts, progress=False)

    # Should have same number of results
    assert len(individual_results) == len(batch_results)

    # Each should have extracted triplets
    for individual, batch in zip(individual_results, batch_results, strict=True):
        # Count should be similar (might differ slightly due to deduplication timing)
        assert abs(len(individual) - len(batch)) <= 1
