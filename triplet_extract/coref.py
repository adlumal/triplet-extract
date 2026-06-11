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
- Third-person singular personal pronouns with grammatical gender
  ("he", "she", "him", "her"), excluding possessives — dcoref clusters
  mentions without rewriting text, so text substitution is the new step
  here, and possessives do not substitute cleanly as surface strings.
- Antecedent candidates are PERSON named entities (the animate mention
  subset of dcoref's mention extraction).

Known precision limit: with no name-gender dictionary, candidate gender
is usually UNKNOWN and therefore agreement-compatible with either
pronoun. A gendered pronoun whose true referent is absent from the text
can thus resolve to the single (apparently other-gendered) PERSON in
scope: "Sarah arrived early. He was annoyed." resolves "He" -> Sarah.
The uniqueness gate cannot catch a lone candidate. Documented in the
README; captured as known behavior in tests/test_coref.py.
"""

from spacy.tokens import Doc

# DeterministicCorefSieve.java:150-152: antecedent within 3 sentences
# for third-person pronouns
PRONOUN_SENTENCE_WINDOW = 3


def _is_candidate_pronoun(token) -> bool:
    """
    Third-person singular gendered personal pronoun, not possessive.

    Mirrors dcoref's pronominal mention gating (PronounMatch DO_PRONOUN;
    person/gender from Mention.setPerson/setGender), with spaCy morphology
    replacing the pronoun dictionaries.
    """
    if token.pos_ != "PRON":
        return False
    morph = token.morph
    return (
        morph.get("PronType") == ["Prs"]
        and morph.get("Person") == ["3"]
        and morph.get("Number") == ["Sing"]
        and morph.get("Gender") in (["Masc"], ["Fem"])
        and morph.get("Poss") != ["Yes"]
    )


def _attributes_agree(pronoun, entity) -> bool:
    """
    Port of Rules.entityAttributesAgree (Rules.java:216-291), reduced to
    the singleton-mention case: for each attribute, a conflict exists only
    when both sides carry known, incompatible values; UNKNOWN (absent
    morphology) is compatible with anything.

    Attributes checked, as in the Java: number, gender, animacy.
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

    # Animacy (Mention.setAnimacy, Mention.java:619): gendered personal
    # pronouns are animate; PERSON entities are animate (NER switch,
    # Mention.java:630-633) — always compatible within this port's scope
    return True


def resolve_unique_pronouns(doc: Doc) -> str | None:
    """
    Resolve uniquely-determined pronouns in a parsed document.

    For each in-scope pronoun, collect PERSON-entity antecedents from the
    preceding PRONOUN_SENTENCE_WINDOW sentences (and the current sentence,
    before the pronoun); substitute only when exactly one agreeing
    candidate exists (repeated mentions of the same name count once).
    Otherwise abstain and leave the pronoun untouched.

    Args:
        doc: Parsed spaCy Doc (full document, so cross-sentence
            antecedents are visible)

    Returns:
        The rewritten text if at least one pronoun was resolved,
        otherwise None (nothing to change — use the original).
    """
    sentences = list(doc.sents)
    if not sentences:
        return None

    sent_index = {}
    for i, sent in enumerate(sentences):
        for token in sent:
            sent_index[token.i] = i

    person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]

    replacements = {}  # token index -> replacement text
    for token in doc:
        if not _is_candidate_pronoun(token):
            continue

        token_sent = sent_index[token.i]
        window_start = sentences[max(0, token_sent - PRONOUN_SENTENCE_WINDOW)].start

        candidates = [
            ent
            for ent in person_entities
            if ent.end <= token.i and ent.start >= window_start and _attributes_agree(token, ent)
        ]

        # Uniqueness gate (divergence from dcoref's salience ranking —
        # see module docstring): repeated mentions of one name are a
        # single candidate; two distinct names mean ambiguity -> abstain
        unique_texts = {ent.text for ent in candidates}
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
