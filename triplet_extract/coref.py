"""
Gated pronominal coreference resolution.

Port of the pronoun sieve from Stanford CoreNLP's deterministic
coreference system (dcoref; Lee et al., 2011, "Stanford's Multi-Pass
Sieve Coreference Resolution System"):

- Pronoun sieve: edu.stanford.nlp.dcoref.sievepasses.PronounMatch.java
  (DO_PRONOUN over DeterministicCorefSieve).
- Sentence window: DeterministicCorefSieve.java:150-152 — a third-person
  pronoun's antecedent must lie within 3 sentences.
- Attribute agreement: Rules.entityAttributesAgree (Rules.java:216-291) —
  number/gender/animacy conflict only when BOTH sides carry known,
  incompatible values; UNKNOWN is compatible with anything.
- Mention attributes: Mention.java setGender (line 513), setNumber (563),
  setAnimacy (619). Where the Java consults word lists (male/female
  pronoun dictionaries, Bergsma-Lin gender lists, animate-word lists),
  this port substitutes model-derived features: spaCy morphology
  (Gender/Number/Person/PronType) for pronouns, and named-entity labels
  for candidate animacy (PERSON -> animate, mirroring the NER switch in
  Mention.java:630-633). Named persons therefore usually carry gender
  UNKNOWN here (there is no Bergsma list in this package by design).

Deliberate divergence from the Java: dcoref RANKS agreeing antecedents
by syntactic salience and links to the best candidate. This port instead
substitutes a pronoun ONLY when exactly one agreeing antecedent exists in
the window, and abstains otherwise. Abstention fails in the safe
direction — an unresolved pronoun is honest ignorance for downstream
consumers, while a wrong substitution is a poisoned premise. Pronouns
that need world knowledge (Winograd-style ambiguity, several candidate
persons in scope) are therefore deliberately left alone.

Scope (high-precision subset):
- Third-person singular gendered personal pronouns ("he", "she", "him",
  "her"), excluding possessives — dcoref clusters mentions without
  rewriting text, so text substitution is the new step here, and
  possessives do not substitute cleanly as surface strings. Antecedents
  are PERSON named entities (the animate mention subset).
- Third-person plural personal pronouns ("they", "them"), with plural
  noun-phrase antecedents (the cheap, high-value non-person case). Singular
  "it" is excluded — its resolution is Winograd-class (world knowledge),
  which this resolver abstains on.

Name gender: English proper nouns carry no Gender morphology, so by
default a gendered pronoun whose referent is absent can resolve to the
lone PERSON in scope regardless of that name's apparent gender. The
optional name-gender prior (gender.py, the `coref` extra) closes this:
a confident name/pronoun gender conflict vetoes the candidate. Without
the extra the resolver stays lexicon-free (uniqueness gate only).
"""

from spacy.tokens import Doc

# DeterministicCorefSieve.java:150-152: antecedent within 3 sentences
# for third-person pronouns
PRONOUN_SENTENCE_WINDOW = 3


def _pronoun_kind(token) -> str | None:
    """
    Classify a token as a resolvable pronoun, or None.

    - "person": third-person singular gendered personal pronoun
      ("he"/"she"/"him"/"her"). Antecedents are PERSON named entities —
      animate mentions, mirroring dcoref's pronominal gating (PronounMatch
      DO_PRONOUN; person/gender from Mention.setPerson/setGender).
    - "plural": third-person plural personal pronoun ("they"/"them").
      Antecedents are plural noun phrases. Singular non-person pronouns
      ("it") are deliberately excluded — their resolution is Winograd-class
      (world knowledge), which this resolver abstains on.

    Possessives are excluded (dcoref clusters mentions without rewriting
    text; possessives do not substitute cleanly as surface strings).
    """
    if token.pos_ != "PRON":
        return None
    morph = token.morph
    if morph.get("PronType") != ["Prs"] or morph.get("Person") != ["3"]:
        return None
    if morph.get("Poss") == ["Yes"]:
        return None
    if morph.get("Number") == ["Sing"] and morph.get("Gender") in (["Masc"], ["Fem"]):
        return "person"
    if morph.get("Number") == ["Plur"]:
        return "plural"
    return None


_PRONOUN_GENDER = {"Masc": "MALE", "Fem": "FEMALE"}


def _attributes_agree(pronoun, entity, gender_prior=None) -> bool:
    """
    Port of Rules.entityAttributesAgree (Rules.java:216-291), reduced to
    the singleton-mention case: for each attribute, a conflict exists only
    when both sides carry known, incompatible values; UNKNOWN (absent
    morphology) is compatible with anything.

    Attributes checked, as in the Java: number, gender, animacy.

    gender_prior (optional, see gender.py): supplies a name-gender signal
    that English proper nouns lack morphologically. When it returns a
    confident gender that conflicts with the pronoun, the candidate is
    rejected; an unknown/abstaining prior leaves the candidate eligible.
    """
    root = entity.root

    # Number (Mention.setNumber, Mention.java:563)
    number = root.morph.get("Number")
    if number and number != ["Sing"]:
        return False

    # Gender (Mention.setGender, Mention.java:513) — English proper nouns
    # carry no Gender morphology, so this usually compares against UNKNOWN
    gender = root.morph.get("Gender")
    if gender and gender != pronoun.morph.get("Gender"):
        return False

    # Optional embedding name-gender prior (gender.py): only a confident
    # conflict vetoes; abstention leaves the candidate eligible.
    if gender_prior is not None and not gender:
        pronoun_gender = next(iter(pronoun.morph.get("Gender")), None)
        want = _PRONOUN_GENDER.get(pronoun_gender)
        if want is not None:
            name_gender = gender_prior.gender(entity.text)
            if name_gender is not None and name_gender != want:
                return False

    # Animacy (Mention.setAnimacy, Mention.java:619): gendered personal
    # pronouns are animate; PERSON entities are animate (NER switch,
    # Mention.java:630-633) — always compatible within this port's scope
    return True


def _plural_antecedent_spans(doc):
    """
    Plural noun-phrase antecedent candidates: noun chunks (and plural
    named entities) headed by a plural common/proper noun. Pronoun-headed
    chunks are excluded — "they" does not antecede "they".
    """
    spans = []
    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.pos_ in ("NOUN", "PROPN") and root.morph.get("Number") == ["Plur"]:
            spans.append(chunk)
    for ent in doc.ents:
        if ent.root.morph.get("Number") == ["Plur"] and ent.root.pos_ in ("NOUN", "PROPN"):
            spans.append(ent)
    return spans


def resolve_unique_pronouns(doc: Doc, gender_prior=None) -> str | None:
    """
    Resolve uniquely-determined pronouns in a parsed document.

    For each in-scope pronoun, collect agreeing antecedent candidates from
    the preceding PRONOUN_SENTENCE_WINDOW sentences (and the current
    sentence, before the pronoun) — PERSON entities for singular gendered
    pronouns, plural noun phrases for "they"/"them". Substitute only when
    exactly one distinct candidate (by text) exists; otherwise abstain.

    Args:
        doc: Parsed spaCy Doc (full document, so cross-sentence antecedents
            are visible)
        gender_prior: optional name-gender prior (see gender.py); only
            vetoes confident gender conflicts for singular pronouns

    Returns:
        The rewritten text if at least one pronoun was resolved, otherwise
        None (nothing to change — use the original).
    """
    sentences = list(doc.sents)
    if not sentences:
        return None

    sent_index = {}
    for i, sent in enumerate(sentences):
        for token in sent:
            sent_index[token.i] = i

    person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    plural_spans = None  # built lazily; noun_chunks parse can be non-trivial

    replacements = {}  # token index -> replacement text
    for token in doc:
        kind = _pronoun_kind(token)
        if kind is None:
            continue

        token_sent = sent_index[token.i]
        window_start = sentences[max(0, token_sent - PRONOUN_SENTENCE_WINDOW)].start

        if kind == "person":
            candidates = [
                ent
                for ent in person_entities
                if ent.end <= token.i
                and ent.start >= window_start
                and _attributes_agree(token, ent, gender_prior)
            ]
        else:  # plural
            if plural_spans is None:
                plural_spans = _plural_antecedent_spans(doc)
            candidates = [
                span for span in plural_spans if span.end <= token.i and span.start >= window_start
            ]

        # Uniqueness gate (divergence from dcoref's salience ranking —
        # see module docstring): repeated mentions of the same surface form
        # are a single candidate; two distinct ones mean ambiguity -> abstain
        unique_texts = {span.text for span in candidates}
        if len(unique_texts) != 1:
            continue

        replacements[token.i] = unique_texts.pop()

    if not replacements:
        return None

    parts = []
    for token in doc:
        text = replacements.get(token.i, token.text)
        parts.append(text + token.whitespace_)
    return "".join(parts)
