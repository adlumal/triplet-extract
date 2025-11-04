"""
triplet-extract - Pure Python Triplet Extraction

A pure Python port of Stanford OpenIE for extracting (subject, relation, object)
triples from natural language text.

Based on: "Leveraging Linguistic Structure For Open Domain Information Extraction"
by Gabor Angeli, Melvin Johnson Premkumar, and Christopher D. Manning (ACL 2015)

This implementation is a faithful port of Stanford's Java implementation to Python,
using spaCy for NLP processing instead of Stanford CoreNLP.

Copyright (C) 2025 - Licensed under GPL-3.0
Original work: Stanford OpenIE, Copyright (C) 2015 Stanford University (GPL-3.0)
"""

from .extractor import OpenIEExtractor, Triplet

__version__ = "0.2.0"
__author__ = "Adrian Lucas Malec"
__license__ = "GPL-3.0"

__all__ = ["OpenIEExtractor", "Triplet", "extract"]

# Global singleton extractor for quick extraction (lazy loaded)
_default_extractor = None


def extract(text: str, **kwargs):
    """
    Quick triplet extraction from text.

    Extracts (subject, relation, object) triples using the full OpenIE pipeline.

    For best performance with multiple calls, either use this function without
    custom kwargs (to reuse the singleton extractor) or create your own
    OpenIEExtractor instance and call extract_triplet_objects() on it.

    Args:
        text: Input text to extract triples from
        **kwargs: Optional arguments passed to OpenIEExtractor:
            - enable_clause_split: Enable clause splitting (default: True)
            - enable_entailment: Enable natural logic entailment (default: True)
            - min_confidence: Minimum confidence threshold (default: 0.3)
            Note: Providing custom kwargs creates a new extractor instance

    Returns:
        List of Triplet objects with (subject, relation, object) attributes

    Example:
        >>> from triplet_extract import extract
        >>> triplets = extract("Cats love milk and mice.")
        >>> for t in triplets:
        ...     print(f"{t.subject} | {t.relation} | {t.object}")
        Cats | love | milk
        Cats | love | mice

    Performance tip:
        >>> # For multiple extractions, reuse an extractor instance:
        >>> extractor = OpenIEExtractor()
        >>> for text in texts:
        ...     triplets = extractor.extract_triplet_objects(text)
    """
    global _default_extractor

    # If custom kwargs provided, create new instance
    if kwargs:
        extractor = OpenIEExtractor(**kwargs)
        return extractor.extract_triplet_objects(text)

    # Otherwise use singleton (lazy load on first use)
    if _default_extractor is None:
        _default_extractor = OpenIEExtractor()

    return _default_extractor.extract_triplet_objects(text)
