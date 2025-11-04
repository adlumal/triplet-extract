"""
Auxiliary Edge Filter

Removes redundant auxiliary-only triples that provide no semantic content.

For example, from "25% of people don't know what GraphRAG is":
  - REMOVE: "25% of people -> do -> know" (auxiliary artifact)
  - KEEP: "25% of people -> do know not -> what GraphRAG is" (real semantic content)

This implements a two-layer filter:
1. Mark triples whose relation is purely auxiliary support
2. Remove aux-only triples dominated by richer triples with the same subject

Reference: Based on community best practices for OpenIE post-processing
"""

# Auxiliary verb lemmas (do, be, have, modals)
AUX_LEMMAS = {
    "do",
    "be",
    "have",
    "will",
    "would",
    "can",
    "could",
    "may",
    "might",
    "shall",
    "should",
    "must",
}


def is_aux_only_triple(triple) -> bool:
    """
    Check if a triple's relation is purely auxiliary support.

    An aux-only triple has:
    1. All relation tokens are auxiliaries or negation
    2. Relation head is tagged as AUX (not main VERB)
    3. Object contains at least one content VERB

    Args:
        triple: Triple object with token-level information

    Returns:
        True if triple is aux-only, False otherwise
    """
    # Need token information to check
    if not triple.relation_tokens or not triple.object_tokens or not triple.relation_head:
        return False

    # Get content tokens from relation (skip punctuation/whitespace)
    content_rel = [t for t in triple.relation_tokens if t.is_alpha]
    if not content_rel:
        return False

    # Check 1: All relation tokens must be auxiliaries or negation
    all_aux_or_neg = all((t.lemma_.lower() in AUX_LEMMAS) or (t.dep_ == "neg") for t in content_rel)
    if not all_aux_or_neg:
        return False

    # Check 2: Relation head must be AUX in the parse (not main VERB)
    if triple.relation_head.pos_ != "AUX":
        return False

    # Check 3: Object must contain at least one content VERB
    # (the real predicate that the aux supports)
    has_content_verb = any(t.pos_ == "VERB" and t.dep_ != "aux" for t in triple.object_tokens)
    if not has_content_verb:
        return False

    return True


def suppress_redundant_aux_edges(triples: list, keep_if_alone: bool = True) -> list:
    """
    Remove aux-only triples that are dominated by richer triples.

    An aux-only triple is dominated if another triple exists with:
    - Same subject (normalized)
    - The aux triple's object verb appears in the other triple's relation or object
    - Equal or higher confidence

    Args:
        triples: List of Triple objects
        keep_if_alone: If True, keep aux-only triple if no better alternative exists

    Returns:
        Filtered list of triples
    """
    # First pass: Mark aux-only triples
    marked = []
    for t in triples:
        is_aux = is_aux_only_triple(t)
        marked.append((t, is_aux))

    # Second pass: Check for dominance
    keep = []
    for triple, is_aux in marked:
        if not is_aux:
            # Not an aux-only triple - always keep
            keep.append(triple)
            continue

        # Check if dominated by another triple
        dominated = False
        for other, _other_is_aux in marked:
            if other is triple:
                continue

            # Must have same subject (normalize for comparison)
            if other.subject.lower().strip() != triple.subject.lower().strip():
                continue

            # Get object lemmas from aux triple (the content verb we're looking for)
            aux_obj_lemmas = {t.lemma_.lower() for t in triple.object_tokens if t.is_alpha}

            # Get lemmas from other triple's relation and object
            other_rel_lemmas = (
                {t.lemma_.lower() for t in other.relation_tokens if t.is_alpha}
                if other.relation_tokens
                else set()
            )
            other_obj_lemmas = (
                {t.lemma_.lower() for t in other.object_tokens if t.is_alpha}
                if other.object_tokens
                else set()
            )

            # Check if aux triple's object verb appears in other triple
            if aux_obj_lemmas & (other_rel_lemmas | other_obj_lemmas):
                # Other triple contains the content verb - this aux triple is dominated
                dominated = True
                break

        # Keep if not dominated, or if alone and keep_if_alone is True
        if not dominated:
            keep.append(triple)
        elif keep_if_alone:
            # Check if there's ANY other triple with same subject
            has_alternative = any(
                other.subject.lower().strip() == triple.subject.lower().strip()
                and other is not triple
                for other, _ in marked
            )
            if not has_alternative:
                # No alternative - keep this one
                keep.append(triple)

    return keep


def filter_triples(
    triples: list,
    enable_aux_filter: bool = True,
    keep_aux_if_alone: bool = True,
) -> list:
    """
    Main entry point for triple filtering.

    Args:
        triples: List of Triple objects
        enable_aux_filter: Whether to remove auxiliary-only triples
        keep_aux_if_alone: Keep aux triple if no better alternative

    Returns:
        Filtered list of triples
    """
    if not enable_aux_filter:
        return triples

    return suppress_redundant_aux_edges(triples, keep_if_alone=keep_aux_if_alone)
