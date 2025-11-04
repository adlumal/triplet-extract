"""
Operator Scope Specification

Represents the scope of a quantifier/operator in a sentence, including:
- The quantifier span itself (e.g., "no", "all", "most")
- The subject scope (what the quantifier quantifies over)
- The object scope (what is predicated of the subject)

Ported from OperatorSpec.java
"""

from dataclasses import dataclass

from .operator import Operator


@dataclass
class OperatorSpec:
    """
    Specification of an operator's scope in a sentence.

    Attributes:
        instance: The Operator (quantifier/negation) instance
        quantifier_begin: Start index of quantifier span (inclusive, 0-indexed)
        quantifier_end: End index of quantifier span (exclusive, 0-indexed)
        subject_begin: Start index of subject scope
        subject_end: End index of subject scope
        object_begin: Start index of object scope
        object_end: End index of object scope
        sentence_length: Total length of sentence (for bounds checking)
    """

    instance: Operator
    quantifier_begin: int
    quantifier_end: int
    subject_begin: int
    subject_end: int
    object_begin: int
    object_end: int
    sentence_length: int = 0

    def __post_init__(self):
        """Validate and clamp indices to sentence bounds."""
        if self.sentence_length > 0:
            # Clamp all indices to valid bounds
            self.quantifier_begin = max(0, min(self.sentence_length - 1, self.quantifier_begin))
            self.quantifier_end = max(0, min(self.sentence_length, self.quantifier_end))
            self.subject_begin = max(0, min(self.sentence_length - 1, self.subject_begin))
            self.subject_end = max(0, min(self.sentence_length, self.subject_end))

            # Object can be empty (for unary quantifiers)
            if self.object_begin == self.sentence_length:
                self.object_begin = self.sentence_length
            else:
                self.object_begin = max(0, min(self.sentence_length - 1, self.object_begin))
            self.object_end = max(0, min(self.sentence_length, self.object_end))

    @property
    def quantifier_head(self) -> int:
        """Get the head index of the quantifier (last token)."""
        return self.quantifier_end - 1

    def is_explicit(self) -> bool:
        """
        Check if this is an explicit quantifier.

        Returns:
            True if explicit (e.g., "all", "some"), False if implicit (named entities)
        """
        return self.instance != Operator.IMPLICIT_NAMED_ENTITY

    def is_binary(self) -> bool:
        """
        Check if this is a binary quantifier (has object scope).

        Returns:
            True if object scope is non-empty
        """
        return self.object_end > self.object_begin

    def quantifier_length(self) -> int:
        """Get the length of the quantifier span."""
        return self.quantifier_end - self.quantifier_begin

    def contains_index(self, index: int, scope: str = "both") -> bool:
        """
        Check if an index is within this operator's scope.

        Args:
            index: Token index to check
            scope: Which scope to check ("subject", "object", "both")

        Returns:
            True if index is in the specified scope(s)
        """
        in_subject = self.subject_begin <= index < self.subject_end
        in_object = self.object_begin <= index < self.object_end

        if scope == "subject":
            return in_subject
        elif scope == "object":
            return in_object
        elif scope == "both":
            return in_subject or in_object
        else:
            raise ValueError(f"Unknown scope: {scope}")

    @staticmethod
    def merge(spec1: "OperatorSpec", spec2: "OperatorSpec") -> "OperatorSpec":
        """
        Merge two operator specs for the same quantifier.

        This is used when multiple patterns match the same operator.

        Args:
            spec1: First operator spec
            spec2: Second operator spec

        Returns:
            Merged operator spec with combined scopes
        """
        # Validate they're the same operator
        assert spec1.quantifier_begin == spec2.quantifier_begin, "Quantifiers must match"
        assert spec1.quantifier_end == spec2.quantifier_end, "Quantifiers must match"
        assert spec1.instance == spec2.instance, "Operators must match"

        # Merge by taking union of scopes
        return OperatorSpec(
            instance=spec1.instance,
            quantifier_begin=min(spec1.quantifier_begin, spec2.quantifier_begin),
            quantifier_end=min(spec1.quantifier_end, spec2.quantifier_end),
            subject_begin=min(spec1.subject_begin, spec2.subject_begin),
            subject_end=max(spec1.subject_end, spec2.subject_end),
            object_begin=min(spec1.object_begin, spec2.object_begin),
            object_end=max(spec1.object_end, spec2.object_end),
            sentence_length=spec1.sentence_length,
        )

    def __str__(self):
        return (
            f"OperatorSpec({self.instance.name}, "
            f"quant=[{self.quantifier_begin}:{self.quantifier_end}], "
            f"subj=[{self.subject_begin}:{self.subject_end}], "
            f"obj=[{self.object_begin}:{self.object_end}])"
        )

    def __repr__(self):
        return self.__str__()
