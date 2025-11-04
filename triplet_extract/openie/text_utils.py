"""
Text reconstruction utilities for OpenIE extraction.

Provides functions for properly reconstructing text from spaCy tokens,
handling compound words (hyphenated), punctuation, and spacing correctly.
"""

from collections.abc import Iterable

from spacy.tokens import Token


def reconstruct_text(tokens: Iterable[Token]) -> str:
    """
    Reconstruct text from spaCy tokens with proper spacing.

    Handles:
    - Compound words with hyphens (e.g., "antibiotic-resistant")
    - Punctuation (no space before punctuation)
    - Contractions including special modal forms (can't, won't)
    - Em/en dashes and other Unicode hyphens

    Args:
        tokens: Iterable of spaCy Token objects

    Returns:
        Reconstructed text string with proper spacing

    Examples:
        >>> tokens = doc  # ["antibiotic", "-", "resistant", "strains"]
        >>> reconstruct_text(tokens)
        'antibiotic-resistant strains'

        >>> tokens = doc  # ["Hello", ",", "world", "!"]
        >>> reconstruct_text(tokens)
        'Hello, world!'

        >>> tokens = doc  # ["ca", "n't", "go"]
        >>> reconstruct_text(tokens)
        "can't go"
    """
    tokens_list = list(tokens)

    if not tokens_list:
        return ""

    # Punctuation, hyphens, and contractions that should not have space before them
    NO_SPACE_BEFORE = frozenset(
        (
            ".",
            ",",
            "!",
            "?",
            ";",
            ":",
            ")",
            "]",
            "}",
            "-",
            "—",
            "–",
            "‐",
            # English contractions (attach to previous word)
            "n't",
            "'s",
            "'re",
            "'ve",
            "'ll",
            "'d",
            "'m",
        )
    )
    # Hyphens that should not have space after them (for compounds)
    NO_SPACE_AFTER = frozenset(("-", "—", "–", "‐"))

    # Truncated modal forms that need completion before n't
    # spaCy tokenizes "can't" as ["ca", "n't"] and "won't" as ["wo", "n't"]
    TRUNCATED_MODALS = {"ca": "can't", "wo": "won't"}

    text_parts = []
    skip_next = False

    for i, token in enumerate(tokens_list):
        # Skip this token if previous was a truncated modal + n't
        if skip_next:
            skip_next = False
            continue

        token_text = token.text

        # Fix truncated modals before n't
        if token_text in ("ca", "wo") and i + 1 < len(tokens_list):
            if tokens_list[i + 1].text == "n't":
                token_text = TRUNCATED_MODALS[token_text]
                skip_next = True  # Skip the n't token

        text_parts.append(token_text)

        # Add space after token if not last and spacing rules allow
        # When we skip n't, we need to check i+2 for spacing (the token after n't)
        next_index = i + 2 if skip_next else i + 1

        if next_index < len(tokens_list):
            next_token = tokens_list[next_index]

            # Don't add space before punctuation/hyphens/contractions
            if next_token.text in NO_SPACE_BEFORE:
                continue

            # Don't add space after hyphens (for compound words)
            if token.text in NO_SPACE_AFTER:
                continue

            text_parts.append(" ")

    return "".join(text_parts)


def reconstruct_text_from_strings(token_strings: list[str]) -> str:
    """
    Reconstruct text from token strings (without spaCy Token objects).

    Similar to reconstruct_text() but works with plain strings.
    Useful when you only have text and not Token objects.

    Args:
        token_strings: List of token text strings

    Returns:
        Reconstructed text string with proper spacing
    """
    if not token_strings:
        return ""

    NO_SPACE_BEFORE = frozenset(
        (
            ".",
            ",",
            "!",
            "?",
            ";",
            ":",
            ")",
            "]",
            "}",
            "-",
            "—",
            "–",
            "‐",
            # English contractions (attach to previous word)
            "n't",
            "'s",
            "'re",
            "'ve",
            "'ll",
            "'d",
            "'m",
        )
    )
    NO_SPACE_AFTER = frozenset(("-", "—", "–", "‐"))

    # Truncated modal forms that need completion before n't
    TRUNCATED_MODALS = {"ca": "can't", "wo": "won't"}

    text_parts = []
    skip_next = False

    for i, token_str in enumerate(token_strings):
        # Skip this token if previous was a truncated modal + n't
        if skip_next:
            skip_next = False
            continue

        # Fix truncated modals before n't
        if token_str in ("ca", "wo") and i + 1 < len(token_strings):
            if token_strings[i + 1] == "n't":
                token_str = TRUNCATED_MODALS[token_str]
                skip_next = True  # Skip the n't token

        text_parts.append(token_str)

        # Add space after token if not last and spacing rules allow
        # When we skip n't, we need to check i+2 for spacing (the token after n't)
        next_index = i + 2 if skip_next else i + 1

        if next_index < len(token_strings):
            next_str = token_strings[next_index]

            if next_str not in NO_SPACE_BEFORE and token_str not in NO_SPACE_AFTER:
                text_parts.append(" ")

    return "".join(text_parts)
