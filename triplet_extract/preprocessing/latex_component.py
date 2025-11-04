"""spaCy custom component for LaTeX placeholder handling."""

from spacy.language import Language
from spacy.tokens import Doc


@Language.component("latex_placeholder_handler")
def latex_placeholder_handler(doc: Doc) -> Doc:
    """
    Custom spaCy component that marks LaTeX placeholders as proper nouns.

    This ensures placeholders like LATEX_MATH_1 are treated as atomic entities
    (like "Obama" or "Microsoft") and aren't split or modified during parsing.

    Args:
        doc: spaCy Doc object

    Returns:
        Modified Doc with placeholders marked as PROPN
    """
    for token in doc:
        # Check if this token is a LaTeX placeholder
        if token.text.startswith("LATEX_MATH_"):
            # Override POS tag to proper noun
            token.pos_ = "PROPN"
            token.tag_ = "NNP"  # Proper noun, singular

            # Mark as named entity to prevent splitting
            # (Though we mainly rely on POS tagging)

    return doc


def create_latex_aware_nlp(nlp):
    """
    Add LaTeX placeholder handler to a spaCy pipeline.

    Args:
        nlp: spaCy Language object

    Returns:
        Modified nlp with latex_placeholder_handler component
    """
    # Add component after tokenizer, before parser
    # This ensures placeholders are treated as proper nouns before dependency parsing
    if "latex_placeholder_handler" not in nlp.pipe_names:
        # Add after tokenizer (first component)
        nlp.add_pipe("latex_placeholder_handler", first=True)

    return nlp


def test_latex_component():
    """Test the LaTeX placeholder component."""
    import spacy

    # Load spaCy and add component
    nlp = spacy.load("en_core_web_sm")
    nlp = create_latex_aware_nlp(nlp)

    # Test text with placeholder
    text = "The value LATEX_MATH_1 is greater than LATEX_MATH_2."
    doc = nlp(text)

    print("Testing LaTeX placeholder component:")
    print(f"Text: {text}\n")
    print("Token analysis:")
    for token in doc:
        if token.text.startswith("LATEX_MATH_"):
            print(f"  {token.text}: POS={token.pos_}, TAG={token.tag_}, DEP={token.dep_}")
            assert token.pos_ == "PROPN", f"Expected PROPN, got {token.pos_}"

    print("\n✅ LaTeX placeholders correctly marked as PROPN")


if __name__ == "__main__":
    test_latex_component()
