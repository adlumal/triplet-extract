"""
Clause Search State and Actions

Ports the State and Action classes from Stanford's ClauseSplitterSearchProblem.
These define the search space for clause splitting.

Ported from ClauseSplitterSearchProblem.java:100-740
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from spacy.tokens import Doc, Token

# Hard-coded split rules from ClauseSplitterSearchProblem.java:56-95
# Maps dependency relations to preferred action sequences
HARD_SPLITS = {
    "comp": ["simple"],
    "ccomp": ["simple"],
    "xcomp": ["clone_obj", "clone_nsubj", "simple"],
    "vmod": ["clone_nsubj", "simple"],
    "csubj": ["clone_obj", "simple"],
    "advcl": ["clone_nsubj", "simple"],
    "advcl:*": ["clone_nsubj", "simple"],  # Wildcard for any advcl subtype
    "conj:*": ["clone_nsubj", "clone_obj", "simple"],  # Wildcard for any conj subtype
    "acl:relcl": ["simple"],
    "parataxis": ["simple"],
}

# Indirect speech verbs - we don't split ccomp edges from these
# From ClauseSplitterSearchProblem.java:100-102
INDIRECT_SPEECH_LEMMAS = {"report", "say", "told", "claim", "assert", "think", "believe", "suppose"}


@dataclass
class ClauseSearchState:
    """
    A search state for clause splitting.

    Ports ClauseSplitterSearchProblem.State (lines 143-187)

    Attributes:
        edge: The edge we traversed to get here (None for initial state)
        subject_or_null: The subject edge from the parent tree
        distance_from_subj: Distance from the last subject we saw
        object_or_null: The object edge from the parent tree
        tree_operations: List of operations to apply to the tree
        is_done: Whether this state represents a complete clause
    """

    edge: Token | None
    subject_or_null: Token | None
    distance_from_subj: int
    object_or_null: Token | None
    tree_operations: list[Callable[[Doc], None]]
    is_done: bool

    def __init__(
        self,
        edge: Token | None = None,
        subject_or_null: Token | None = None,
        distance_from_subj: int = -9000,
        object_or_null: Token | None = None,
        tree_operations: list[Callable[[Doc], None]] | None = None,
        is_done: bool = True,
    ):
        """
        Initialize a search state.

        Args:
            edge: The edge token we traversed (None for initial state)
            subject_or_null: Subject token from parent tree
            distance_from_subj: How far we are from the last subject
            object_or_null: Object token from parent tree
            tree_operations: List of tree modification operations
            is_done: Whether this is a complete clause split
        """
        self.edge = edge
        self.subject_or_null = subject_or_null
        self.distance_from_subj = distance_from_subj
        self.object_or_null = object_or_null
        self.tree_operations = tree_operations if tree_operations is not None else []
        self.is_done = is_done

    def with_is_done(self, classifier_label: str) -> "ClauseSearchState":
        """
        Update the is_done flag based on classifier output.

        Args:
            classifier_label: One of 'CLAUSE_SPLIT', 'CLAUSE_INTERM', 'NOT_A_CLAUSE'

        Returns:
            Self (for chaining)
        """
        if classifier_label == "CLAUSE_SPLIT":
            self.is_done = True
        elif classifier_label == "CLAUSE_INTERM":
            self.is_done = False
        # NOT_A_CLAUSE doesn't change is_done
        return self

    def __repr__(self):
        edge_str = f"'{self.edge.text}'" if self.edge else "ROOT"
        return f"State(edge={edge_str}, is_done={self.is_done})"


class ClauseAction(ABC):
    """
    Base class for clause splitting actions.

    Ports ClauseSplitterSearchProblem.Action (lines 192-222)
    """

    @abstractmethod
    def signature(self) -> str:
        """Return the action signature/name."""
        pass

    def prerequisites_met(self, doc: Doc, edge: Token) -> bool:
        """
        Check if prerequisites are met for this action.

        Args:
            doc: The original dependency tree
            edge: The edge we're considering splitting

        Returns:
            True if action can be applied
        """
        return True

    @abstractmethod
    def apply_to(
        self,
        doc: Doc,
        source: ClauseSearchState,
        outgoing_edge: Token,
        subject_or_null: Token | None,
        object_or_null: Token | None,
    ) -> ClauseSearchState | None:
        """
        Apply this action to create a new state.

        Args:
            doc: The original dependency tree
            source: The source state
            outgoing_edge: The edge we're splitting off
            subject_or_null: Subject from parent tree
            object_or_null: Object from parent tree

        Returns:
            New state, or None if action cannot be applied
        """
        pass


class SimpleAction(ClauseAction):
    """
    Simple clause split action.

    Ports the "simple" action from lines 577-606.
    This is the most basic clause split - just take the subtree.
    """

    def signature(self) -> str:
        return "simple"

    def prerequisites_met(self, doc: Doc, edge: Token) -> bool:
        """Only split into V, N, J, P, D tags."""
        if not edge.tag_:
            return False
        tag_first_char = edge.tag_[0]
        return tag_first_char in {"V", "N", "J", "P", "D"}

    def apply_to(
        self,
        doc: Doc,
        source: ClauseSearchState,
        outgoing_edge: Token,
        subject_or_null: Token | None,
        object_or_null: Token | None,
    ) -> ClauseSearchState | None:
        """Create new state with simple split."""

        # Create new tree operations list
        new_operations = source.tree_operations.copy()

        # Add the split operation
        def split_operation(tree_doc: Doc):
            # In the full implementation, this would:
            # 1. Keep only the subtree rooted at outgoing_edge
            # 2. Remove aux and mark edges if this is a comp relation
            pass

        new_operations.append(split_operation)

        # Create new state
        return ClauseSearchState(
            edge=outgoing_edge,
            subject_or_null=subject_or_null if subject_or_null else source.subject_or_null,
            distance_from_subj=0 if subject_or_null else (source.distance_from_subj + 1),
            object_or_null=object_or_null if object_or_null else source.object_or_null,
            tree_operations=new_operations,
            is_done=False,
        )


class CloneSubjectAction(ClauseAction):
    """
    Clone subject action.

    Ports the "clone_nsubj" action from lines 656-695.
    Copies the subject from the parent clause into the new clause.
    """

    def signature(self) -> str:
        return "clone_nsubj"

    def prerequisites_met(self, doc: Doc, edge: Token) -> bool:
        """Only split into V or N tags, and only if no subject already exists."""
        if not edge.tag_:
            return False
        tag_first_char = edge.tag_[0]
        if tag_first_char not in {"V", "N"}:
            return False

        # Check that the edge doesn't already have a subject
        for child in edge.children:
            if "subj" in child.dep_.lower():
                return False

        return True

    def apply_to(
        self,
        doc: Doc,
        source: ClauseSearchState,
        outgoing_edge: Token,
        subject_or_null: Token | None,
        object_or_null: Token | None,
    ) -> ClauseSearchState | None:
        """Create new state with cloned subject."""

        # Only apply if we have a subject and it's not the outgoing edge
        if not subject_or_null or outgoing_edge == subject_or_null:
            return None

        # Create new tree operations
        new_operations = source.tree_operations.copy()

        def clone_subj_operation(tree_doc: Doc):
            # In the full implementation, this would:
            # 1. Do a simple split
            # 2. Copy the subject subtree and attach it to outgoing_edge
            # 3. Strip aux and mark edges
            pass

        new_operations.append(clone_subj_operation)

        # Create new state
        return ClauseSearchState(
            edge=outgoing_edge,
            subject_or_null=subject_or_null,
            distance_from_subj=0,
            object_or_null=object_or_null if object_or_null else source.object_or_null,
            tree_operations=new_operations,
            is_done=False,
        )


class CloneObjectAction(ClauseAction):
    """
    Clone object action.

    Ports the "clone_obj" action from lines 698-740.
    Copies the object from the parent clause as the subject of the new clause.
    """

    def signature(self) -> str:
        return "clone_obj"

    def prerequisites_met(self, doc: Doc, edge: Token) -> bool:
        """Only split into V or N tags, and only if no subject already exists."""
        if not edge.tag_:
            return False
        tag_first_char = edge.tag_[0]
        if tag_first_char not in {"V", "N"}:
            return False

        # Check that the edge doesn't already have a subject
        for child in edge.children:
            if "subj" in child.dep_.lower():
                return False

        return True

    def apply_to(
        self,
        doc: Doc,
        source: ClauseSearchState,
        outgoing_edge: Token,
        subject_or_null: Token | None,
        object_or_null: Token | None,
    ) -> ClauseSearchState | None:
        """Create new state with cloned object as subject."""

        # Only apply if we have an object and it's not the outgoing edge
        if not object_or_null or outgoing_edge == object_or_null:
            return None

        # Create new tree operations
        new_operations = source.tree_operations.copy()

        def clone_obj_operation(tree_doc: Doc):
            # In the full implementation, this would:
            # 1. Do a simple split
            # 2. Copy the object subtree and attach it as nsubj to outgoing_edge
            # 3. Strip aux and mark edges
            pass

        new_operations.append(clone_obj_operation)

        # Create new state
        return ClauseSearchState(
            edge=outgoing_edge,
            subject_or_null=subject_or_null if subject_or_null else source.subject_or_null,
            distance_from_subj=0 if subject_or_null else (source.distance_from_subj + 1),
            object_or_null=object_or_null,
            tree_operations=new_operations,
            is_done=False,
        )


def get_action_space() -> list[ClauseAction]:
    """
    Get the default action space for clause splitting.

    Returns:
        List of all available actions
    """
    return [
        SimpleAction(),
        CloneSubjectAction(),
        CloneObjectAction(),
    ]


def get_hard_split_actions(relation: str) -> list[str] | None:
    """
    Get the hard-coded action sequence for a dependency relation.

    Args:
        relation: Dependency relation (e.g., "ccomp", "xcomp", "advcl:tmod")

    Returns:
        List of action signatures in preference order, or None if no hard split
    """
    # Direct match
    if relation in HARD_SPLITS:
        return HARD_SPLITS[relation]

    # Wildcard match (e.g., "advcl:tmod" matches "advcl:*")
    if ":" in relation:
        base_rel = relation.split(":")[0]
        wildcard_key = f"{base_rel}:*"
        if wildcard_key in HARD_SPLITS:
            return HARD_SPLITS[wildcard_key]

    return None


def should_skip_indirect_speech(edge: Token) -> bool:
    """
    Check if this edge should be skipped due to indirect speech.

    From ClauseSplitterSearchProblem.java:857-864:
    We don't split ccomp edges when the governor is an indirect speech verb.

    Args:
        edge: The edge to check

    Returns:
        True if this edge should be skipped
    """
    if edge.dep_ != "ccomp":
        return False

    # Check if governor is indirect speech verb
    governor = edge.head
    governor_lemma = governor.lemma_.lower() if governor.lemma_ else ""
    governor_word = governor.text.lower()

    return governor_lemma in INDIRECT_SPEECH_LEMMAS or governor_word in INDIRECT_SPEECH_LEMMAS
