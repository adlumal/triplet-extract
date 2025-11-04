"""
SentenceFragment - Python port from Stanford CoreNLP

Represents a sentence fragment with its dependency parse tree.
Used by ClauseSplitter and ForwardEntailer.
"""

from spacy.tokens import Doc, Token

from .text_utils import reconstruct_text


class SentenceFragment:
    """
    A representation of a sentence fragment.

    Attributes:
        doc: Spacy Doc containing the tokens
        token_indices: Set of token indices included in this fragment
        assumed_truth: Whether this fragment is assumed to be true
        score: Confidence score for this fragment (default 1.0)
    """

    def __init__(
        self,
        doc: Doc,
        token_indices: set[int] | None = None,
        assumed_truth: bool = True,
        score: float = 1.0,
    ):
        """
        Initialize a sentence fragment.

        Args:
            doc: Spacy Doc object
            token_indices: Set of token indices to include (None = all tokens)
            assumed_truth: Whether fragment is assumed true
            score: Confidence score
        """
        self.doc = doc
        self.token_indices = token_indices if token_indices is not None else set(range(len(doc)))
        self.assumed_truth = assumed_truth
        self.score = score

    @property
    def tokens(self) -> list[Token]:
        """Get the tokens in this fragment, in sentence order."""
        return [self.doc[i] for i in sorted(self.token_indices)]

    def length(self) -> int:
        """The length of this fragment in tokens."""
        return len(self.token_indices)

    def contains_token(self, token: Token) -> bool:
        """Check if a token is in this fragment."""
        return token.i in self.token_indices

    def text(self) -> str:
        """Get the text representation of this fragment."""
        if not self.token_indices:
            return ""

        # Get tokens in order
        tokens = self.tokens
        return reconstruct_text(tokens)

    def remove_token(self, token: Token) -> "SentenceFragment":
        """
        Create a new fragment without the specified token.

        Automatically detects and removes hyphenated compounds atomically
        to avoid orphaned hyphens (e.g., removing "antibiotic" from
        "antibiotic-resistant" will also remove "-" and "resistant").

        Args:
            token: Token to remove

        Returns:
            New SentenceFragment without the token (and its compound parts if any)
        """
        # Get all tokens that are part of the same compound
        compound_tokens = self._get_compound_tokens(token)

        new_indices = self.token_indices.copy()
        for comp_token in compound_tokens:
            new_indices.discard(comp_token.i)
        return SentenceFragment(self.doc, new_indices, self.assumed_truth, self.score)

    def remove_tokens(self, tokens: set[Token]) -> "SentenceFragment":
        """
        Create a new fragment without the specified tokens.

        Automatically detects and removes hyphenated compounds atomically.

        Args:
            tokens: Set of tokens to remove

        Returns:
            New SentenceFragment without the tokens (and their compound parts if any)
        """
        # Collect all tokens including compound parts
        all_tokens_to_remove = set()
        for token in tokens:
            compound_tokens = self._get_compound_tokens(token)
            all_tokens_to_remove.update(compound_tokens)

        new_indices = self.token_indices.copy()
        for token in all_tokens_to_remove:
            new_indices.discard(token.i)
        return SentenceFragment(self.doc, new_indices, self.assumed_truth, self.score)

    def change_score(self, score: float) -> "SentenceFragment":
        """Update the score of this fragment."""
        self.score = score
        return self

    def copy(self) -> "SentenceFragment":
        """Create a copy of this fragment."""
        return SentenceFragment(self.doc, self.token_indices.copy(), self.assumed_truth, self.score)

    def _get_compound_tokens(self, token: Token) -> set[Token]:
        """
        Get all tokens that are part of the same hyphenated compound as the given token.

        For example, in "antibiotic-resistant", if given "antibiotic", returns
        {"antibiotic", "-", "resistant"} so they can be deleted atomically.

        Args:
            token: Token that may be part of a compound

        Returns:
            Set of tokens in the compound (includes the token itself)
        """
        compound_tokens = {token}

        # Check UPWARD: if token is part of a hyphenated compound by looking at siblings
        parent = token.head
        if parent != token:
            siblings = list(parent.children)

            # Check if any sibling is a hyphen (indicates compound)
            has_hyphen = any(sib.text in ("-", "—", "–", "‐") for sib in siblings)

            if has_hyphen:
                # Include all tokens that are part of this compound:
                # - All amod (adjectival modifier) siblings
                # - All hyphen punctuation siblings
                # - The parent if it's also an amod (e.g., resistant in antibiotic-resistant strains)
                for sibling in siblings:
                    if sibling.dep_ == "amod" or sibling.text in ("-", "—", "–", "‐"):
                        compound_tokens.add(sibling)

                # If parent is also a modifier (e.g., "resistant" modifying "strains"),
                # include it in the compound
                if parent.dep_ == "amod":
                    compound_tokens.add(parent)

        # Check DOWNWARD: if token itself is the head of a hyphenated compound
        children = list(token.children)
        has_hyphen_child = any(child.text in ("-", "—", "–", "‐") for child in children)

        if has_hyphen_child:
            # Include all children that are part of the compound
            for child in children:
                if child.dep_ == "amod" or child.text in ("-", "—", "–", "‐"):
                    compound_tokens.add(child)

        return compound_tokens

    def __len__(self) -> int:
        return self.length()

    def __str__(self) -> str:
        return self.text()

    def __repr__(self) -> str:
        return f"SentenceFragment(text='{self.text()}', score={self.score:.2f}, length={self.length()})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, SentenceFragment):
            return False
        return self.token_indices == other.token_indices and self.doc == other.doc

    def __hash__(self) -> int:
        return hash((id(self.doc), frozenset(self.token_indices)))
