"""
Polarity Annotator

Annotates tokens with their polarity based on operators (quantifiers and negation markers)
in scope. This is critical for determining what mutations are valid while maintaining
truth-preserving inferences.

The annotation process:
1. Detect operators (quantifiers/negation) using dependency patterns
2. Compute scope for each operator (subject and object spans)
3. For each token, collect operators in scope and compute polarity

Ported from NaturalLogicAnnotator.java
"""

import logging
import re

from spacy.language import Language
from spacy.tokens import Doc, Token

from .operator import Operator
from .operator_spec import OperatorSpec
from .polarity import Monotonicity, MonotonicityType, Polarity

logger = logging.getLogger(__name__)


# ===================================================================
# Dependency Relation Patterns (Semgrex equivalents for spaCy)
# ===================================================================

# Determiners and modifiers
DET_RELS = {"det", "amod", "advmod", "neg", "nummod", "compound", "case"}

# Subject relations
SUBJ_RELS = {"nsubj", "nsubj:pass", "nsubjpass"}

# Object relations
OBJ_RELS = {"obj", "iobj", "dobj", "xcomp", "advcl"}

# Copula relations
COP_RELS = {"cop", "aux", "auxpass", "aux:pass"}

# Clause/modifier relations
CLAUSE_RELS = {"nmod", "obl", "acl:relcl", "acl"}

# Prepositional/adverbial relations
PREP_RELS = {"nmod", "obl", "advcl", "ccomp", "advmod"}

# Negation lemmas
NEG_LEMMAS = {"no", "not", "never", "neither", "nobody", "n't", "but", "except"}

# Negation marker patterns
NEG_PATTERN = re.compile(r"\b(no|not|never|neither|nobody|n't)\b", re.IGNORECASE)

# Doubt words (trigger downward polarity)
DOUBT_WORDS = {"doubt", "skeptical", "doubtful"}


# ===================================================================
# Helper Functions for Subtree Span Computation
# ===================================================================


def _get_subtree_span(
    token: Token, valid_rels: set[str] | None = None, include_punct: bool = False
) -> tuple[int, int]:
    """
    Get the span (begin, end) of the subtree rooted at token.

    Args:
        token: Root of subtree
        valid_rels: If provided, only traverse these relations. If None, traverse all.
        include_punct: Whether to include punctuation

    Returns:
        Tuple of (begin_index, end_index) where end is exclusive
    """
    min_idx = token.i
    max_idx = token.i

    # BFS to collect all descendants
    queue = [token]
    visited = {token}

    while queue:
        current = queue.pop(0)

        for child in current.children:
            if child in visited:
                continue

            # Check relation validity
            if valid_rels is not None and child.dep_ not in valid_rels:
                continue

            # Skip punctuation unless requested
            if not include_punct and child.dep_ == "punct":
                continue

            visited.add(child)
            queue.append(child)

            # Update span
            min_idx = min(min_idx, child.i)
            max_idx = max(max_idx, child.i)

    return (min_idx, max_idx + 1)


def _get_modifier_subtree_span(token: Token) -> tuple[int, int]:
    """
    Get subtree span but only traverse modifier relations.

    For negations, use minimal traversal. Otherwise, allow aux/nmod/obl.

    Args:
        token: Root token

    Returns:
        Span tuple (begin, end)
    """
    # Check if this has a neg child
    has_neg = any(child.dep_ == "neg" for child in token.children)

    if has_neg:
        # Only traverse nmod/obl for negations
        valid_rels = {"nmod", "obl"}
    else:
        # Allow more relations
        valid_rels = {"aux", "nmod", "obl"}

    return _get_subtree_span(token, valid_rels=valid_rels)


def _get_proper_noun_subtree_span(token: Token) -> tuple[int, int]:
    """
    Get subtree span for proper nouns (only compounds).

    Args:
        token: Root token

    Returns:
        Span tuple (begin, end)
    """
    return _get_subtree_span(token, valid_rels={"compound"})


def _get_jj_component_span(token: Token) -> tuple[int, int]:
    """
    Get subtree span for adjectives (advmod, amod).

    Args:
        token: Root token

    Returns:
        Span tuple (begin, end)
    """
    return _get_subtree_span(token, valid_rels={"advmod", "amod"})


def _include_in_span(span1: tuple[int, int], span2: tuple[int, int]) -> tuple[int, int]:
    """Merge two spans."""
    return (min(span1[0], span2[0]), max(span1[1], span2[1]))


def _exclude_from_span(span: tuple[int, int], to_exclude: tuple[int, int]) -> tuple[int, int]:
    """
    Exclude a span from another span.

    If to_exclude is on the edge, trim the span.
    If to_exclude is in the middle, keep original (can't split).

    Args:
        span: Main span
        to_exclude: Span to exclude

    Returns:
        Modified span
    """
    if to_exclude[1] <= span[0] or to_exclude[0] >= span[1]:
        # No overlap
        return span
    elif to_exclude[0] <= span[0]:
        # Overlap on the left
        return (to_exclude[1], span[1])
    elif to_exclude[1] >= span[1]:
        # Overlap on the right
        return (span[0], to_exclude[0])
    else:
        # to_exclude is in the middle - can't split, return original
        return span


# ===================================================================
# Operator Detection via Dependency Patterns
# ===================================================================


def _find_quantifier_by_lemma(doc: Doc, lemma: str) -> list[Token]:
    """
    Find tokens matching a quantifier lemma.

    Args:
        doc: spaCy Doc
        lemma: Lemma to search for

    Returns:
        List of matching tokens
    """
    matches = []
    for token in doc:
        if token.lemma_.lower() == lemma.lower():
            matches.append(token)
        # Also check text for contractions like "n't"
        elif token.text.lower() == lemma.lower():
            matches.append(token)
    return matches


def _has_neg_child(token: Token) -> bool:
    """Check if token has a neg dependency child."""
    return any(child.dep_ == "neg" for child in token.children)


def _get_child_by_dep(token: Token, dep_pattern: set[str]) -> Token | None:
    """
    Get first child matching dependency pattern.

    Args:
        token: Parent token
        dep_pattern: Set of dependency labels to match

    Returns:
        First matching child, or None
    """
    for child in token.children:
        if child.dep_ in dep_pattern:
            return child
    return None


def _match_subj_verb_obj_pattern(doc: Doc) -> list[tuple[Token, Token, Token, Token]]:
    """
    Match pattern: {verb} >nsubj {subject >>det quantifier} >obj {object}

    Returns:
        List of (pivot, subject, object, quantifier) tuples
    """
    matches = []

    for token in doc:
        # Must be a verb
        if not token.pos_.startswith("V"):
            continue

        # Get subject
        subj = _get_child_by_dep(token, SUBJ_RELS)
        if not subj:
            continue

        # Get object
        obj = _get_child_by_dep(token, OBJ_RELS)
        if not obj:
            continue

        # Subject must have a det/amod quantifier
        for det in subj.children:
            if det.dep_ in DET_RELS:
                # Check if det is a quantifier
                if Operator.from_string(det.text) is not None:
                    matches.append((token, subj, obj, det))

    return matches


def _match_subj_verb_prep_pattern(doc: Doc) -> list[tuple[Token, Token, Token, Token]]:
    """
    Match pattern: {verb} >nsubj {subject >>det quantifier} >prep {object}

    Returns:
        List of (pivot, subject, object, quantifier) tuples
    """
    matches = []

    for token in doc:
        if not token.pos_.startswith("V"):
            continue

        subj = _get_child_by_dep(token, SUBJ_RELS)
        if not subj:
            continue

        prep_obj = _get_child_by_dep(token, PREP_RELS)
        if not prep_obj:
            continue

        for det in subj.children:
            if det.dep_ in DET_RELS:
                if Operator.from_string(det.text) is not None:
                    matches.append((token, subj, prep_obj, det))

    return matches


def _match_copula_pattern(doc: Doc) -> list[tuple[Token, Token, Token, Token]]:
    """
    Match pattern: {object} >nsubj {subject >>det quantifier} >cop {pivot}

    Returns:
        List of (pivot, subject, object, quantifier) tuples
    """
    matches = []

    for token in doc:
        subj = _get_child_by_dep(token, SUBJ_RELS)
        if not subj:
            continue

        cop = _get_child_by_dep(token, COP_RELS)
        if not cop:
            continue

        for det in subj.children:
            if det.dep_ in DET_RELS:
                if Operator.from_string(det.text) is not None:
                    # For copula, object is the predicate
                    matches.append((cop, subj, token, det))

    return matches


def _match_negation_pattern(doc: Doc) -> list[tuple[Token, Token | None, Token, Token]]:
    """
    Match pattern: {verb} >/neg/ {not} >obj {object}

    Returns:
        List of (pivot, None, object, quantifier) tuples
    """
    matches = []

    for token in doc:
        if not _has_neg_child(token):
            continue

        # Get the neg token
        neg_token = _get_child_by_dep(token, {"neg", "advmod"})
        if not neg_token:
            continue

        # Check if neg token is a negation word
        if neg_token.lemma_.lower() not in NEG_LEMMAS and not NEG_PATTERN.search(neg_token.text):
            continue

        # Get object
        obj = _get_child_by_dep(token, OBJ_RELS | PREP_RELS)
        if obj:
            matches.append((token, None, obj, neg_token))
        else:
            # No object - negation of adjective or subject-only
            subj = _get_child_by_dep(token, SUBJ_RELS)
            if subj:
                matches.append((token, subj, token, neg_token))

    return matches


def _match_proper_noun_pattern(doc: Doc) -> list[tuple[Token, Token, Token]]:
    """
    Match pattern: {verb} >nsubj {NNP} >obj {object}

    Proper nouns have implicit universal quantification.

    Returns:
        List of (pivot, subject, object) tuples
    """
    matches = []

    for token in doc:
        if not token.pos_.startswith("V"):
            continue

        subj = _get_child_by_dep(token, SUBJ_RELS)
        if not subj or subj.pos_ != "PROPN":
            continue

        obj = _get_child_by_dep(token, OBJ_RELS | PREP_RELS)
        if obj:
            matches.append((token, subj, obj))

    return matches


# ===================================================================
# Scope Computation
# ===================================================================


def _compute_scope(
    doc: Doc,
    operator: Operator,
    pivot: Token,
    quantifier_span: tuple[int, int],
    subject: Token | None,
    is_proper_noun: bool,
    obj: Token | None,
) -> OperatorSpec:
    """
    Compute scope for an operator.

    This determines the subject and object spans for the operator.

    Args:
        doc: spaCy Doc
        operator: The Operator instance
        pivot: The pivot token (usually the verb)
        quantifier_span: Span of the quantifier itself
        subject: Subject token (or None)
        is_proper_noun: Whether subject is a proper noun
        obj: Object token (or None)

    Returns:
        OperatorSpec with computed scopes
    """
    # Handle different cases based on what was matched
    if subject is None and obj is None:
        # Unary pattern - take pivot subtree
        subj_span = _get_subtree_span(pivot)

        # Exclude quantifier if it's inside
        if quantifier_span[0] >= subj_span[0] and quantifier_span[1] <= subj_span[1]:
            # Quantifier is inside - take part after it
            subj_span = (max(subj_span[0], quantifier_span[1]), subj_span[1])
            if subj_span[1] <= subj_span[0]:
                subj_span = (subj_span[0], subj_span[0] + 1)
        else:
            subj_span = _exclude_from_span(subj_span, quantifier_span)

        obj_span = (subj_span[1], subj_span[1])

    elif subject is None:
        # Only object - make it the subject
        assert obj is not None
        subj_span = _include_in_span(
            _get_subtree_span(obj), _get_subtree_span(pivot, valid_rels={"nmod", "obl"})
        )
        obj_span = (subj_span[1], subj_span[1])

    else:
        # Both subject and object
        # Get subject subtree
        if is_proper_noun:
            subj_subtree = _get_proper_noun_subtree_span(subject)
        else:
            subj_subtree = _get_subtree_span(subject)

        subj_span = _exclude_from_span(subj_subtree, quantifier_span)

        # Get object subtree
        if obj == pivot:
            vanilla_obj_span = _get_jj_component_span(obj)
        else:
            vanilla_obj_span = _get_subtree_span(obj)

        # Include pivot modifiers if obj != pivot
        if obj == pivot:
            obj_span = vanilla_obj_span
        else:
            obj_span = _include_in_span(vanilla_obj_span, _get_modifier_subtree_span(pivot))

        # Exclude subject from object
        obj_span = _exclude_from_span(obj_span, subj_subtree)

    # Adjust subject span if it overlaps with quantifier end
    if subj_span[0] < quantifier_span[1] and subj_span[1] > quantifier_span[1]:
        subj_span = (quantifier_span[1], subj_span[1])

    return OperatorSpec(
        instance=operator,
        quantifier_begin=quantifier_span[0],
        quantifier_end=quantifier_span[1],
        subject_begin=subj_span[0],
        subject_end=subj_span[1],
        object_begin=obj_span[0],
        object_end=obj_span[1],
        sentence_length=len(doc),
    )


def _validate_quantifier(
    doc: Doc, quantifier_token: Token, is_unary: bool = False
) -> tuple[Operator, int, int] | None:
    """
    Validate and identify the quantifier at a token position.

    Handles multi-word quantifiers by looking backwards.

    Args:
        doc: spaCy Doc
        quantifier_token: Token where quantifier was matched
        is_unary: Whether this is a unary quantifier

    Returns:
        Tuple of (Operator, begin_index, end_index) or None
    """
    tokens = list(doc)
    quant_idx = quantifier_token.i

    # Helper to get gloss (normalize numbers)
    def get_gloss(token):
        if token.pos_ == "NUM" or token.tag_ == "CD":
            return "--num--"
        return token.lemma_.lower()

    # Try different spans (look backwards up to 10 tokens)
    # Try with offset for numbers
    offsets_to_check = [2, 1, 0] if quantifier_token.tag_ == "CD" else [0]

    for offset_end in offsets_to_check:
        end = min(len(tokens), quant_idx + offset_end + 1)

        for start in range(max(0, quant_idx - 10), quant_idx + 1):
            # Build gloss
            gloss = " ".join(get_gloss(tokens[i]) for i in range(start, end))

            # Try to match operator
            for operator in Operator.values_by_length_desc():
                if operator.surface_form == gloss:
                    # Check unary constraint
                    if operator.is_unary() or not is_unary:
                        return (operator, start, end)

    return None


# ===================================================================
# Main Annotation Functions
# ===================================================================


def _annotate_operators(doc: Doc) -> dict[int, OperatorSpec]:
    """
    Annotate operators in the document.

    Finds quantifiers and negations using dependency patterns,
    computes their scopes, and stores them.

    Args:
        doc: spaCy Doc

    Returns:
        Dictionary mapping token index to OperatorSpec
    """
    operator_specs: dict[int, OperatorSpec] = {}

    # Pattern 1: Subject-Verb-Object with quantifier
    for pivot, subj, obj, quant in _match_subj_verb_obj_pattern(doc):
        quant_info = _validate_quantifier(doc, quant, is_unary=False)
        if quant_info:
            operator, q_begin, q_end = quant_info
            spec = _compute_scope(doc, operator, pivot, (q_begin, q_end), subj, False, obj)
            _add_or_merge_spec(operator_specs, quant.i, spec)

    # Pattern 2: Subject-Verb-Prep with quantifier
    for pivot, subj, obj, quant in _match_subj_verb_prep_pattern(doc):
        quant_info = _validate_quantifier(doc, quant, is_unary=False)
        if quant_info:
            operator, q_begin, q_end = quant_info
            spec = _compute_scope(doc, operator, pivot, (q_begin, q_end), subj, False, obj)
            _add_or_merge_spec(operator_specs, quant.i, spec)

    # Pattern 3: Copula pattern
    for pivot, subj, obj, quant in _match_copula_pattern(doc):
        quant_info = _validate_quantifier(doc, quant, is_unary=False)
        if quant_info:
            operator, q_begin, q_end = quant_info
            spec = _compute_scope(doc, operator, pivot, (q_begin, q_end), subj, False, obj)
            _add_or_merge_spec(operator_specs, quant.i, spec)

    # Pattern 4: Negation pattern
    for pivot, subj, obj, quant in _match_negation_pattern(doc):
        quant_info = _validate_quantifier(doc, quant, is_unary=subj is None)
        if quant_info:
            operator, q_begin, q_end = quant_info
            spec = _compute_scope(doc, operator, pivot, (q_begin, q_end), subj, False, obj)
            _add_or_merge_spec(operator_specs, quant.i, spec)

    # Pattern 5: Proper noun pattern (implicit quantification)
    for pivot, subj, obj in _match_proper_noun_pattern(doc):
        # Use implicit named entity operator
        operator = Operator.IMPLICIT_NAMED_ENTITY
        spec = _compute_scope(doc, operator, pivot, (subj.i, subj.i), subj, True, obj)
        _add_or_merge_spec(operator_specs, subj.i, spec)

    return operator_specs


def _add_or_merge_spec(specs: dict[int, OperatorSpec], index: int, new_spec: OperatorSpec):
    """
    Add or merge an operator spec.

    If a spec already exists at this index, merge them.

    Args:
        specs: Dictionary of specs
        index: Token index
        new_spec: New OperatorSpec to add
    """
    if index in specs:
        specs[index] = OperatorSpec.merge(specs[index], new_spec)
    else:
        specs[index] = new_spec


def _annotate_unaries(doc: Doc, operator_specs: dict[int, OperatorSpec]):
    """
    Annotate unary quantifiers not caught by main patterns.

    This catches simple cases like "Some cats" where there's no verb.

    Args:
        doc: spaCy Doc
        operator_specs: Existing operator specs (will be modified)
    """
    # Get indices where we already have operators
    existing_indices = set()
    for _idx, spec in operator_specs.items():
        for i in range(spec.quantifier_begin, spec.quantifier_end):
            existing_indices.add(i)

    # Look for simple noun + det patterns
    for token in doc:
        if not token.pos_.startswith("N"):
            continue

        # Check for determiner/quantifier
        for child in token.children:
            if child.dep_ not in DET_RELS:
                continue

            # Skip if already annotated
            if child.i in existing_indices:
                continue

            # Skip uninformative quantifiers
            if child.text.lower() in ("a", "an", "the") or child.pos_ == "NUM":
                continue

            # Try to validate as quantifier
            quant_info = _validate_quantifier(doc, child, is_unary=True)
            if quant_info:
                operator, q_begin, q_end = quant_info
                spec = _compute_scope(doc, operator, token, (q_begin, q_end), None, False, None)
                _add_or_merge_spec(operator_specs, child.i, spec)


def _annotate_polarity(doc: Doc, operator_specs: dict[int, OperatorSpec]) -> Doc:
    """
    Annotate polarity for every token based on operators in scope.

    Args:
        doc: spaCy Doc
        operator_specs: Dictionary of operator specs

    Returns:
        Doc with polarity annotations added
    """
    # Initialize all tokens with default polarity
    for token in doc:
        token._.polarity = Polarity.DEFAULT

    # For each token, find operators in scope and compute polarity
    for i, token in enumerate(doc):
        # Collect operators in scope with their monotonicity
        in_scope: list[tuple[int, Monotonicity, MonotonicityType]] = []

        for spec in operator_specs.values():
            # Check if token is in subject scope
            if spec.subject_begin <= i < spec.subject_end:
                scope_size = spec.subject_end - spec.subject_begin
                in_scope.append((scope_size, spec.instance.subj_mono, spec.instance.subj_type))

            # Check if token is in object scope
            elif spec.object_begin <= i < spec.object_end:
                scope_size = spec.object_end - spec.object_begin
                in_scope.append((scope_size, spec.instance.obj_mono, spec.instance.obj_type))

        # Sort by scope size (largest first = narrowest scope in operator stack)
        in_scope.sort(key=lambda x: x[0], reverse=True)

        # Build polarity from operator stack
        if in_scope:
            operators = [(mono, mono_type) for _, mono, mono_type in in_scope]
            token._.polarity = Polarity(operators=operators)

    return doc


# ===================================================================
# Main PolarityAnnotator Class
# ===================================================================


class PolarityAnnotator:
    """
    Annotates documents with polarity information for natural logic inference.

    This finds operators (quantifiers and negation markers), computes their scopes,
    and annotates each token with its polarity based on operators in scope.

    Usage:
        annotator = PolarityAnnotator(nlp)
        doc = nlp("People don't like spam")
        doc = annotator.annotate(doc)
        # Now each token has token._.polarity attribute
    """

    def __init__(self, nlp: Language):
        """
        Initialize the polarity annotator.

        Args:
            nlp: spaCy Language instance
        """
        self.nlp = nlp

        # Register custom attributes if not already registered
        if not Token.has_extension("polarity"):
            Token.set_extension("polarity", default=Polarity.DEFAULT)
        if not Token.has_extension("operator_spec"):
            Token.set_extension("operator_spec", default=None)

    def annotate(self, doc: Doc) -> Doc:
        """
        Annotate a document with polarity information.

        Args:
            doc: spaCy Doc

        Returns:
            Doc with polarity annotations (modifies in place and returns)
        """
        # Step 1: Detect operators
        operator_specs = _annotate_operators(doc)

        # Step 2: Detect unary quantifiers
        _annotate_unaries(doc, operator_specs)

        # Step 3: Store operator specs on tokens
        for idx, spec in operator_specs.items():
            if 0 <= idx < len(doc):
                doc[idx]._.operator_spec = spec

        # Step 4: Compute and annotate polarity
        doc = _annotate_polarity(doc, operator_specs)

        logger.debug(f"Annotated {len(operator_specs)} operators in sentence: {doc.text[:50]}...")

        return doc
