"""
Polarity and Natural Logic Relations

Implements the natural logic framework for determining valid mutations
while maintaining truth-preserving inferences.

Ported from:
- NaturalLogicRelation.java
- Monotonicity.java
- MonotonicityType.java
- Polarity.java
"""

from enum import Enum


class NaturalLogicRelation(Enum):
    """
    The seven natural logic relations.

    Set-theoretically, if A and B are two sets (e.g., denotations):
    - Equivalence: A = B
    - Forward entailment: A ⊂ B
    - Reverse entailment: A ⊃ B
    - Negation: A ∩ B = ∅ and A ∪ B = D (universe)
    - Alternation: A ∩ B = ∅
    - Cover: A ∪ B = D
    - Independence: No specific relation

    Ports NaturalLogicRelation.java:28-84
    """

    # (fixed_index, maintains_truth, negates_truth, maintains_falsehood, negates_falsehood)
    EQUIVALENT = (0, True, False, True, False)
    FORWARD_ENTAILMENT = (1, True, False, False, False)
    REVERSE_ENTAILMENT = (2, False, False, True, False)
    NEGATION = (3, False, True, False, True)
    ALTERNATION = (4, False, True, False, False)
    COVER = (5, False, False, False, True)
    INDEPENDENCE = (6, False, False, False, False)

    def __init__(
        self,
        fixed_index: int,
        maintains_truth: bool,
        negates_truth: bool,
        maintains_falsehood: bool,
        negates_falsehood: bool,
    ):
        self.fixed_index = fixed_index
        self.maintains_truth = maintains_truth
        self.negates_truth = negates_truth
        self.maintains_falsehood = maintains_falsehood
        self.negates_falsehood = negates_falsehood

    @staticmethod
    def by_fixed_index(index: int) -> "NaturalLogicRelation":
        """Get relation by fixed index."""
        for rel in NaturalLogicRelation:
            if rel.fixed_index == index:
                return rel
        raise ValueError(f"Unknown index for Natural Logic relation: {index}")

    def apply_to_truth_value(self, truth_value: bool) -> "TruthValue":
        """
        Apply this natural logic relation to a truth value.

        Returns a TruthValue indicating the result:
        - TRUE if the relation preserves/creates truth
        - FALSE if the relation preserves/creates falsehood
        - UNKNOWN if the result is uncertain

        Args:
            truth_value: Input truth value (True or False)

        Returns:
            TruthValue enum
        """
        if truth_value:
            # Input is TRUE
            if self.maintains_truth:
                return TruthValue.TRUE
            elif self.negates_truth:
                return TruthValue.FALSE
            else:
                return TruthValue.UNKNOWN
        else:
            # Input is FALSE
            if self.maintains_falsehood:
                return TruthValue.FALSE
            elif self.negates_falsehood:
                return TruthValue.TRUE
            else:
                return TruthValue.UNKNOWN


class TruthValue(Enum):
    """
    Truth value for natural logic inference.

    Ports the simple boolean + unknown truth value system from Stanford.
    """

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def is_true(self) -> bool:
        """Check if this truth value is definitely true."""
        return self == TruthValue.TRUE

    def is_false(self) -> bool:
        """Check if this truth value is definitely false."""
        return self == TruthValue.FALSE

    def is_unknown(self) -> bool:
        """Check if this truth value is unknown."""
        return self == TruthValue.UNKNOWN


class Monotonicity(Enum):
    """
    Monotonicity values for operators.

    Ports Monotonicity.java
    """

    MONOTONE = "monotone"
    ANTITONE = "antitone"
    NONMONOTONE = "nonmonotone"
    INVALID = "invalid"


class MonotonicityType(Enum):
    """
    Monotonicity type: additive, multiplicative, both, or neither.

    Ports MonotonicityType.java
    """

    NONE = "none"
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    BOTH = "both"


class Polarity:
    """
    Polarity determines what mutations are valid on a lexical item while
    maintaining valid natural logic inference.

    The polarity is represented as a projection function that maps input
    relations to projected relations after passing through operators in scope.

    Ports Polarity.java:18-280
    """

    # Default permissive polarity
    DEFAULT = None  # Will be set after class definition

    def __init__(
        self,
        operators: list[tuple[Monotonicity, MonotonicityType]] = None,
        projection_function: list[int] = None,
    ):
        """
        Initialize polarity from operators or projection function.

        Args:
            operators: List of (Monotonicity, MonotonicityType) operators in narrowing scope order
            projection_function: Direct projection function (7 elements, indices 0-6)
        """
        if projection_function is not None:
            # Initialize from projection function
            if len(projection_function) != 7:
                raise ValueError(f"Invalid projection function: {projection_function}")
            for idx in projection_function:
                if idx < 0 or idx > 6:
                    raise ValueError(f"Invalid projection function: {projection_function}")
            self.projection_function = list(projection_function)

        else:
            # Initialize from operators
            if not operators:
                # Identity projection
                self.projection_function = list(range(7))
            else:
                # Compute projection
                self.projection_function = [0] * 7
                for rel_idx in range(7):
                    relation = NaturalLogicRelation.by_fixed_index(rel_idx)

                    # Project through each operator (in reverse order)
                    for mono, mono_type in reversed(operators):
                        relation = self._project(relation, mono, mono_type)

                    self.projection_function[rel_idx] = relation.fixed_index

    def _project(
        self, input_rel: NaturalLogicRelation, mono: Monotonicity, mono_type: MonotonicityType
    ) -> NaturalLogicRelation:
        """
        Project a relation through an operator.

        Ports Polarity.java:70-175

        Args:
            input_rel: Input natural logic relation
            mono: Monotonicity of operator
            mono_type: Monotonicity type of operator

        Returns:
            Projected relation
        """
        # EQUIVALENT always stays EQUIVALENT
        if input_rel == NaturalLogicRelation.EQUIVALENT:
            return NaturalLogicRelation.EQUIVALENT

        # FORWARD_ENTAILMENT
        if input_rel == NaturalLogicRelation.FORWARD_ENTAILMENT:
            if mono == Monotonicity.MONOTONE:
                return NaturalLogicRelation.FORWARD_ENTAILMENT
            elif mono == Monotonicity.ANTITONE:
                return NaturalLogicRelation.REVERSE_ENTAILMENT
            else:  # NONMONOTONE or INVALID
                return NaturalLogicRelation.INDEPENDENCE

        # REVERSE_ENTAILMENT
        if input_rel == NaturalLogicRelation.REVERSE_ENTAILMENT:
            if mono == Monotonicity.MONOTONE:
                return NaturalLogicRelation.REVERSE_ENTAILMENT
            elif mono == Monotonicity.ANTITONE:
                return NaturalLogicRelation.FORWARD_ENTAILMENT
            else:  # NONMONOTONE or INVALID
                return NaturalLogicRelation.INDEPENDENCE

        # NEGATION
        if input_rel == NaturalLogicRelation.NEGATION:
            if mono_type == MonotonicityType.NONE:
                return NaturalLogicRelation.INDEPENDENCE
            elif mono_type == MonotonicityType.ADDITIVE:
                if mono == Monotonicity.MONOTONE:
                    return NaturalLogicRelation.COVER
                elif mono == Monotonicity.ANTITONE:
                    return NaturalLogicRelation.ALTERNATION
                else:
                    return NaturalLogicRelation.INDEPENDENCE
            elif mono_type == MonotonicityType.MULTIPLICATIVE:
                if mono == Monotonicity.MONOTONE:
                    return NaturalLogicRelation.ALTERNATION
                elif mono == Monotonicity.ANTITONE:
                    return NaturalLogicRelation.COVER
                else:
                    return NaturalLogicRelation.INDEPENDENCE
            elif mono_type == MonotonicityType.BOTH:
                return NaturalLogicRelation.NEGATION

        # ALTERNATION
        if input_rel == NaturalLogicRelation.ALTERNATION:
            if mono == Monotonicity.MONOTONE:
                if mono_type in (MonotonicityType.NONE, MonotonicityType.ADDITIVE):
                    return NaturalLogicRelation.INDEPENDENCE
                else:  # MULTIPLICATIVE or BOTH
                    return NaturalLogicRelation.ALTERNATION
            elif mono == Monotonicity.ANTITONE:
                if mono_type in (MonotonicityType.NONE, MonotonicityType.ADDITIVE):
                    return NaturalLogicRelation.INDEPENDENCE
                else:  # MULTIPLICATIVE or BOTH
                    return NaturalLogicRelation.COVER
            else:  # NONMONOTONE or INVALID
                return NaturalLogicRelation.INDEPENDENCE

        # COVER
        if input_rel == NaturalLogicRelation.COVER:
            if mono == Monotonicity.MONOTONE:
                if mono_type in (MonotonicityType.NONE, MonotonicityType.MULTIPLICATIVE):
                    return NaturalLogicRelation.INDEPENDENCE
                else:  # ADDITIVE or BOTH
                    return NaturalLogicRelation.COVER
            elif mono == Monotonicity.ANTITONE:
                if mono_type in (MonotonicityType.NONE, MonotonicityType.MULTIPLICATIVE):
                    return NaturalLogicRelation.INDEPENDENCE
                else:  # ADDITIVE or BOTH
                    return NaturalLogicRelation.ALTERNATION
            else:  # NONMONOTONE or INVALID
                return NaturalLogicRelation.INDEPENDENCE

        # INDEPENDENCE
        if input_rel == NaturalLogicRelation.INDEPENDENCE:
            return NaturalLogicRelation.INDEPENDENCE

        raise ValueError(f"Projection table incomplete for {mono}:{mono_type} on {input_rel}")

    def project_lexical_relation(
        self, lexical_relation: NaturalLogicRelation
    ) -> NaturalLogicRelation:
        """
        Project a lexical relation through this polarity.

        Args:
            lexical_relation: The lexical relation to project

        Returns:
            The projected relation
        """
        return NaturalLogicRelation.by_fixed_index(
            self.projection_function[lexical_relation.fixed_index]
        )

    def maintains_truth(self, lexical_relation: NaturalLogicRelation) -> bool:
        """Check if applying this relation maintains truth."""
        return self.project_lexical_relation(lexical_relation).maintains_truth

    def negates_truth(self, lexical_relation: NaturalLogicRelation) -> bool:
        """Check if applying this relation negates truth."""
        return self.project_lexical_relation(lexical_relation).negates_truth

    def maintains_falsehood(self, lexical_relation: NaturalLogicRelation) -> bool:
        """Check if applying this relation maintains falsehood."""
        return self.project_lexical_relation(lexical_relation).maintains_falsehood

    def negates_falsehood(self, lexical_relation: NaturalLogicRelation) -> bool:
        """Check if applying this relation negates falsehood."""
        return self.project_lexical_relation(lexical_relation).negates_falsehood

    def is_upwards(self) -> bool:
        """Check if this polarity is upward monotone (ignoring exclusion)."""
        return (
            self.project_lexical_relation(NaturalLogicRelation.FORWARD_ENTAILMENT)
            == NaturalLogicRelation.FORWARD_ENTAILMENT
            and self.project_lexical_relation(NaturalLogicRelation.REVERSE_ENTAILMENT)
            == NaturalLogicRelation.REVERSE_ENTAILMENT
        )

    def is_downwards(self) -> bool:
        """Check if this polarity is downward monotone (ignoring exclusion)."""
        return (
            self.project_lexical_relation(NaturalLogicRelation.FORWARD_ENTAILMENT)
            == NaturalLogicRelation.REVERSE_ENTAILMENT
            and self.project_lexical_relation(NaturalLogicRelation.REVERSE_ENTAILMENT)
            == NaturalLogicRelation.FORWARD_ENTAILMENT
        )

    def __str__(self):
        if self.is_upwards():
            return "up"
        elif self.is_downwards():
            return "down"
        else:
            return "flat"

    def __eq__(self, other):
        if isinstance(other, str):
            other_lower = other.lower()
            if other_lower in ("down", "downward", "downwards", "v"):
                return self.is_downwards()
            elif other_lower in ("up", "upward", "upwards", "^"):
                return self.is_upwards()
            elif other_lower in ("flat", "none", "-"):
                return not self.is_downwards() and not self.is_upwards()
            return False
        if not isinstance(other, Polarity):
            return False
        return self.projection_function == other.projection_function

    def __hash__(self):
        return hash(tuple(self.projection_function))


# Set default polarity
Polarity.DEFAULT = Polarity(operators=[(Monotonicity.MONOTONE, MonotonicityType.BOTH)])
