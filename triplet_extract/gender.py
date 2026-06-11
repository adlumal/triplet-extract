"""
Optional embedding-based name-gender prior for coreference.

Stanford CoreNLP's dcoref reads name gender from Bergsma-Lin gender
dictionaries. This package has no such lexicon by design, so a gendered
pronoun whose true referent is absent from the text resolves to the lone
PERSON in scope regardless of that name's apparent gender ("Sarah arrived
early. He was annoyed." -> He resolves to Sarah). This module closes that
hole without a lexicon: a small sentence embedding model supplies the
name-gender signal, and only confident conflicts veto a resolution.

The signal is sim(name, "she") - sim(name, "he") under BAAI/bge-small-en
(deliberately the original, not the v1.5 revision). On a name sample it
separates cleanly: Sarah +0.048, Maria +0.051, Emily +0.046 (female);
Tom -0.021, Obama -0.035 (male); Alex -0.005, Jordan -0.016 (genuinely
unisex, near zero). Names inside the abstain band are treated as
genderless so the resolver falls back to its uniqueness gate.

This is an OPTIONAL capability: it requires the `coref` extra
(sentence-transformers). When that is not installed, the resolver keeps
its lexicon-free uniqueness behavior. The prior can only ever ADD
abstentions (veto a gender-conflicting candidate); it never forces a
resolution the uniqueness gate would not already make, so enabling it is
strictly safer.
"""

# Names whose |sim(name,"she") - sim(name,"he")| is at or below this are
# treated as gender-unknown. Calibrated on the sample above: confident
# names clear it comfortably, genuinely unisex names fall inside it. The
# signal is model-derived, not a word list.
GENDER_ABSTAIN_BAND = 0.02

_MODEL_NAME = "BAAI/bge-small-en"


class NameGenderPrior:
    """Lazily-loaded embedding gender prior with a per-name cache."""

    def __init__(self, model_name: str = _MODEL_NAME, abstain_band: float = GENDER_ABSTAIN_BAND):
        self._model_name = model_name
        self._abstain_band = abstain_band
        self._model = None
        self._ref = None  # (she_vec, he_vec)
        self._cache: dict[str, str | None] = {}

    @staticmethod
    def is_available() -> bool:
        import importlib.util

        return importlib.util.find_spec("sentence_transformers") is not None

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name, device="cpu")
        she, he = self._model.encode(["she", "he"], normalize_embeddings=True)
        self._ref = (she, he)

    def gender(self, name: str) -> str | None:
        """
        Return "MALE", "FEMALE", or None (unknown / within the abstain band)
        for a name string. Results are cached per name.
        """
        key = name.strip().lower()
        if not key:
            return None
        if key in self._cache:
            return self._cache[key]

        self._ensure_model()
        vec = self._model.encode([key], normalize_embeddings=True)[0]
        she, he = self._ref
        delta = float(vec @ she) - float(vec @ he)
        if delta > self._abstain_band:
            result = "FEMALE"
        elif delta < -self._abstain_band:
            result = "MALE"
        else:
            result = None
        self._cache[key] = result
        return result
