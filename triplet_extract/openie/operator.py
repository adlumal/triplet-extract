"""
Operators (Quantifiers and Negation Markers)

Defines all operators recognized by the natural logic system, including
universal quantifiers (all, every), existential quantifiers (some, a),
negation (not, no), and proportional quantifiers (most, many).

Each operator has monotonicity properties that determine how it affects
the validity of inferences in its subject and object scopes.

Ported from Operator.java
"""

from enum import Enum
from typing import Optional

from .polarity import Monotonicity, MonotonicityType, NaturalLogicRelation


def _parse_monotonicity(mono_str: str) -> tuple[Monotonicity, MonotonicityType]:
    """
    Parse a monotonicity string into (Monotonicity, MonotonicityType).

    Args:
        mono_str: String like "additive", "anti-additive", "multiplicative", etc.

    Returns:
        Tuple of (Monotonicity, MonotonicityType)
    """
    mapping = {
        "nonmonotone": (Monotonicity.NONMONOTONE, MonotonicityType.NONE),
        "additive": (Monotonicity.MONOTONE, MonotonicityType.ADDITIVE),
        "multiplicative": (Monotonicity.MONOTONE, MonotonicityType.MULTIPLICATIVE),
        "additive-multiplicative": (Monotonicity.MONOTONE, MonotonicityType.BOTH),
        "anti-additive": (Monotonicity.ANTITONE, MonotonicityType.ADDITIVE),
        "anti-multiplicative": (Monotonicity.ANTITONE, MonotonicityType.MULTIPLICATIVE),
        "anti-additive-multiplicative": (Monotonicity.ANTITONE, MonotonicityType.BOTH),
    }
    if mono_str not in mapping:
        raise ValueError(f"Unknown monotonicity: {mono_str}")
    return mapping[mono_str]


class Operator(Enum):
    """
    Exhaustive list of operators (quantifiers and negation markers) known to the system.

    Each operator is defined with:
    - surface_form: The text pattern to match (e.g., "all", "no", "not")
    - delete_relation: Natural logic relation when operator is deleted
    - subj_mono: Subject monotonicity string
    - obj_mono: Object monotonicity string (None for unary operators)

    Ported from Operator.java
    """

    # ===================================================================
    # "All" quantifiers - Universal quantification
    # ===================================================================
    ALL = ("all", NaturalLogicRelation.FORWARD_ENTAILMENT, "anti-additive", "multiplicative")
    EVERY = ("every", NaturalLogicRelation.FORWARD_ENTAILMENT, "anti-additive", "multiplicative")
    ANY = ("any", NaturalLogicRelation.FORWARD_ENTAILMENT, "anti-additive", "multiplicative")
    EACH = ("each", NaturalLogicRelation.FORWARD_ENTAILMENT, "anti-additive", "multiplicative")
    THE_LOT_OF = (
        "the lot of",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "anti-additive",
        "multiplicative",
    )
    ALL_OF = (
        "all of",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "anti-additive",
        "multiplicative",
    )
    EACH_OF = (
        "each of",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "anti-additive",
        "multiplicative",
    )
    FOR_ALL = (
        "for all",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "anti-additive",
        "multiplicative",
    )
    FOR_EVERY = (
        "for every",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "anti-additive",
        "multiplicative",
    )
    FOR_EACH = (
        "for each",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "anti-additive",
        "multiplicative",
    )
    EVERYONE = (
        "everyone",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "anti-additive",
        "multiplicative",
    )
    NUM = ("--num--", NaturalLogicRelation.FORWARD_ENTAILMENT, "anti-additive", "multiplicative")
    NUM_NUM = (
        "--num-- --num--",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "anti-additive",
        "multiplicative",
    )
    NUM_NUM_NUM = (
        "--num-- --num-- --num--",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "anti-additive",
        "multiplicative",
    )
    NUM_NUM_NUM_NUM = (
        "--num-- --num-- --num-- --num--",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "anti-additive",
        "multiplicative",
    )
    FEW = ("few", NaturalLogicRelation.FORWARD_ENTAILMENT, "anti-additive", "multiplicative")
    IMPLICIT_NAMED_ENTITY = (
        "__implicit_named_entity__",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "additive",
        "multiplicative",
    )

    # ===================================================================
    # "No" quantifiers - Negation
    # ===================================================================
    NO = ("no", NaturalLogicRelation.INDEPENDENCE, "anti-additive", "anti-additive")
    NEITHER = ("neither", NaturalLogicRelation.INDEPENDENCE, "anti-additive", "anti-additive")
    NO_ONE = ("no one", NaturalLogicRelation.INDEPENDENCE, "anti-additive", "anti-additive")
    NOBODY = ("nobody", NaturalLogicRelation.INDEPENDENCE, "anti-additive", "anti-additive")
    NOT = ("not", NaturalLogicRelation.INDEPENDENCE, "anti-additive", "anti-additive")
    BUT = ("but", NaturalLogicRelation.INDEPENDENCE, "anti-additive", "anti-additive")
    EXCEPT = ("except", NaturalLogicRelation.INDEPENDENCE, "anti-additive", "anti-additive")
    UNARY_NO = ("no", NaturalLogicRelation.INDEPENDENCE, "anti-additive", None)
    UNARY_NOT = ("not", NaturalLogicRelation.INDEPENDENCE, "anti-additive", None)
    UNARY_NO_ONE = ("no one", NaturalLogicRelation.INDEPENDENCE, "anti-additive", None)
    UNARY_NT = ("n't", NaturalLogicRelation.INDEPENDENCE, "anti-additive", None)
    UNARY_BUT = ("but", NaturalLogicRelation.INDEPENDENCE, "anti-additive", None)
    UNARY_EXCEPT = ("except", NaturalLogicRelation.INDEPENDENCE, "anti-additive", None)

    # General downward polarity trigger (for "doubt", "skeptical", etc.)
    GENERAL_NEG_POLARITY = (
        "neg_polarity_trigger",
        NaturalLogicRelation.INDEPENDENCE,
        "anti-additive",
        None,
    )

    # ===================================================================
    # "Some" quantifiers - Existential quantification
    # ===================================================================
    SOME = ("some", NaturalLogicRelation.FORWARD_ENTAILMENT, "additive", "additive")
    SEVERAL = ("several", NaturalLogicRelation.FORWARD_ENTAILMENT, "additive", "additive")
    EITHER = ("either", NaturalLogicRelation.FORWARD_ENTAILMENT, "additive", "additive")
    A = (
        "a",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "additive-multiplicative",
        "additive-multiplicative",
    )
    THE = (
        "the",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "additive-multiplicative",
        "additive-multiplicative",
    )
    LESS_THAN = (
        "less than --num--",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "additive",
        "additive",
    )
    SOME_OF = ("some of", NaturalLogicRelation.FORWARD_ENTAILMENT, "additive", "additive")
    ONE_OF = ("one of", NaturalLogicRelation.FORWARD_ENTAILMENT, "additive", "additive")
    AT_LEAST = (
        "at least --num--",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "additive",
        "additive",
    )
    A_FEW = ("a few", NaturalLogicRelation.FORWARD_ENTAILMENT, "additive", "additive")
    AT_LEAST_A_FEW = (
        "at least a few",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "additive",
        "additive",
    )
    THERE_BE = ("there be", NaturalLogicRelation.FORWARD_ENTAILMENT, "additive", "additive")
    THERE_BE_A_FEW = (
        "there be a few",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "additive",
        "additive",
    )
    THERE_EXIST = (
        "there exist",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "additive",
        "additive",
    )
    NUM_OF = ("--num-- of", NaturalLogicRelation.FORWARD_ENTAILMENT, "additive", "additive")

    # ===================================================================
    # "Not All" quantifiers - Negative universals
    # ===================================================================
    NOT_ALL = (
        "not all",
        NaturalLogicRelation.INDEPENDENCE,
        "additive",
        "anti-multiplicative",
    )
    NOT_EVERY = (
        "not every",
        NaturalLogicRelation.INDEPENDENCE,
        "additive",
        "anti-multiplicative",
    )

    # ===================================================================
    # "Most" quantifiers - Proportional/vague quantifiers
    # ===================================================================
    MOST = ("most", NaturalLogicRelation.FORWARD_ENTAILMENT, "nonmonotone", "multiplicative")
    MORE = ("more", NaturalLogicRelation.FORWARD_ENTAILMENT, "nonmonotone", "multiplicative")
    MANY = ("many", NaturalLogicRelation.FORWARD_ENTAILMENT, "nonmonotone", "multiplicative")
    ENOUGH = ("enough", NaturalLogicRelation.FORWARD_ENTAILMENT, "nonmonotone", "multiplicative")
    MORE_THAN = (
        "more than --num--",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "nonmonotone",
        "multiplicative",
    )
    LOTS_OF = ("lots of", NaturalLogicRelation.FORWARD_ENTAILMENT, "nonmonotone", "multiplicative")
    PLENTY_OF = (
        "plenty of",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "nonmonotone",
        "multiplicative",
    )
    HEAPS_OF = ("heap of", NaturalLogicRelation.FORWARD_ENTAILMENT, "nonmonotone", "multiplicative")
    A_LOAD_OF = (
        "a load of",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "nonmonotone",
        "multiplicative",
    )
    LOADS_OF = ("load of", NaturalLogicRelation.FORWARD_ENTAILMENT, "nonmonotone", "multiplicative")
    TONS_OF = ("ton of", NaturalLogicRelation.FORWARD_ENTAILMENT, "nonmonotone", "multiplicative")
    BOTH = ("both", NaturalLogicRelation.FORWARD_ENTAILMENT, "nonmonotone", "multiplicative")
    JUST_NUM = (
        "just --num--",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "nonmonotone",
        "multiplicative",
    )
    ONLY_NUM = (
        "only --num--",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "nonmonotone",
        "multiplicative",
    )

    # ===================================================================
    # Special cases
    # ===================================================================
    AT_MOST_NUM = (
        "at most --num--",
        NaturalLogicRelation.FORWARD_ENTAILMENT,
        "anti-additive",
        "anti-additive",
    )

    def __init__(
        self,
        surface_form: str,
        delete_relation: NaturalLogicRelation,
        subj_mono: str,
        obj_mono: str | None = None,
    ):
        """
        Initialize an operator.

        Args:
            surface_form: Text pattern to match (e.g., "all", "no", "not")
            delete_relation: Natural logic relation when deleted
            subj_mono: Subject monotonicity string
            obj_mono: Object monotonicity string (None for unary)
        """
        self.surface_form = surface_form
        self.delete_relation = delete_relation

        # Parse subject monotonicity
        subj_parsed = _parse_monotonicity(subj_mono)
        self.subj_mono = subj_parsed[0]
        self.subj_type = subj_parsed[1]

        # Parse object monotonicity
        if obj_mono is None:
            # Unary operator
            self.obj_mono = Monotonicity.INVALID
            self.obj_type = MonotonicityType.NONE
        else:
            obj_parsed = _parse_monotonicity(obj_mono)
            self.obj_mono = obj_parsed[0]
            self.obj_type = obj_parsed[1]

    def is_unary(self) -> bool:
        """Check if this is a unary operator (no object scope)."""
        return self.obj_mono == Monotonicity.INVALID

    @staticmethod
    def from_string(text: str) -> Optional["Operator"]:
        """
        Find operator matching the given text.

        Args:
            text: Text to match (case-insensitive, numbers normalized to --num--)

        Returns:
            Matching Operator, or None if no match
        """
        # Normalize: lowercase and replace numbers with --num--
        normalized = text.lower().strip()
        # Replace sequences of digits with --num--
        import re

        normalized = re.sub(r"\b\d+\b", "--num--", normalized).strip()
        normalized = re.sub(r"\s+", " ", normalized)  # Collapse whitespace

        # Try to find matching operator
        for operator in Operator:
            if operator.surface_form == normalized:
                return operator

        return None

    @staticmethod
    def values_by_length_desc() -> list["Operator"]:
        """
        Get operators sorted by surface form length (descending).

        This ensures longest-match-first when detecting operators.

        Returns:
            List of operators sorted by length
        """
        operators = list(Operator)
        operators.sort(key=lambda op: len(op.surface_form.split()), reverse=True)
        return operators
