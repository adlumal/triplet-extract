"""
Direct-quote speech normalization.

Quoted direct speech ("'The food was amazing,' said Tom.") is reported
speech just like its unquoted form ("Tom said the food was amazing."),
but the quotation marks break extraction: they are deletable punctuation
tokens, so the forward entailer strips them along with the speech frame
and the quoted clause reparses as a standalone author assertion — leaking
the quoted content as if the author asserted it (a poisoned premise) and
producing cross-boundary garbage where the speaker attaches as an object.

Stanford CoreNLP solves attribution with a dedicated QuoteAnnotator that
pairs each quoted span with a speaker mention. This is the deterministic
structural core of that idea: detect the speech frame structurally (a
verb governing a clausal complement, with quote-mark punctuation hanging
off it — exactly how spaCy parses said-inversion and leading attribution
alike) and remove just those quotation-mark tokens. What remains is the
canonical reported-speech construction the pipeline already handles
soundly: the content stays inside the object of the speech triple, the
ccomp natural-logic guard keeps it from escaping, and transitive inner
clauses are attributed to the speaker via the existing chain machinery.

Conservative by design: only quote marks that belong to a clausal-
complement-governing verb are removed; scare quotes, quoted noun phrases,
and anything that does not match the speech-frame shape are left
untouched (abstain rather than mangle).
"""

from spacy.tokens import Doc

# Straight and curly double quotation marks. Single quotes are excluded
# deliberately — they collide with apostrophes/contractions.
_QUOTE_CHARS = {'"', "“", "”", "«", "»"}


def _is_quote_token(token) -> bool:
    return token.text in _QUOTE_CHARS


def strip_speech_quotes(doc: Doc) -> str | None:
    """
    Remove quotation-mark tokens that bracket direct speech.

    A quote mark is removed only when it is a punctuation dependent of a
    verb that governs a clausal complement (the speech frame). Returns the
    rewritten text if any such quote mark was removed, otherwise None
    (nothing matched — use the original).

    Args:
        doc: Parsed spaCy Doc

    Returns:
        Text with speech quotation marks removed, or None.
    """
    # Verbs that govern a ccomp are the speech-frame heads (said/declared/
    # claimed ...); spaCy hangs the quote-mark punct tokens off them.
    speech_heads = {
        token.i
        for token in doc
        if token.pos_ == "VERB" and any(c.dep_ == "ccomp" for c in token.children)
    }
    if not speech_heads:
        return None

    drop = {
        token.i
        for token in doc
        if _is_quote_token(token) and token.dep_ == "punct" and token.head.i in speech_heads
    }
    if not drop:
        return None

    # Also drop the quote-setoff comma in leading attribution ("Tom said,
    # 'X'" -> "Tom said X"): a comma punctuation dependent immediately
    # following a speech head. Without it the leading clause detaches and
    # the quoted content escapes as a standalone assertion. The trailing
    # comma ("'X,' said Tom") sits BEFORE the verb and is load-bearing for
    # the parse, so it is left in place.
    for token in doc:
        if (
            token.text == ","
            and token.dep_ == "punct"
            and token.head.i in speech_heads
            and token.i == token.head.i + 1
        ):
            drop.add(token.i)

    # Reconstruct, keeping the whitespace a dropped token carried so its
    # neighbours stay separated ("said," -> "said " not "saidThe"), then
    # collapse the resulting runs of spaces.
    parts = []
    for token in doc:
        if token.i in drop:
            parts.append(token.whitespace_)
        else:
            parts.append(token.text + token.whitespace_)
    return " ".join("".join(parts).split())
