"""
Natural Logic Weights for ForwardEntailer

Assigns deletion probabilities to dependency edges based on:
- Hard-coded rules for common edge types
- Affinity model lookups for prepositional attachments
- Special handling for privative adjectives

Ported from:
- NaturalLogicWeights.java (266 lines)
- Util.PRIVATIVE_ADJECTIVES from Util.java
"""

from spacy.tokens import Doc, Token

from .affinity_models import AffinityModels

# Privative adjectives - ported from Util.java:553-600
# These adjectives don't preserve entailment when deleted
# E.g., "fake gun" -> "gun" (a fake gun is not a gun)
PRIVATIVE_ADJECTIVES: set[str] = {
    "believed",
    "debatable",
    "disputed",
    "dubious",
    "hypothetical",
    "impossible",
    "improbable",
    "plausible",
    "putative",
    "questionable",
    "so called",
    "supposed",
    "suspicious",
    "theoretical",
    "uncertain",
    "unlikely",
    "would - be",
    "apparent",
    "arguable",
    "assumed",
    "likely",
    "ostensible",
    "possible",
    "potential",
    "predicted",
    "presumed",
    "probable",
    "seeming",
    "anti",
    "fake",
    "fictional",
    "fictitious",
    "imaginary",
    "mythical",
    "phony",
    "false",
    "artificial",
    "erroneous",
    "mistaken",
    "mock",
    "pseudo",
    "simulated",
    "spurious",
    "deputy",
    "faulty",
    "virtual",
    "doubtful",
}


class NaturalLogicWeights:
    """
    Deletion probability weights for the ForwardEntailer.

    Uses a combination of:
    1. Hard-coded rules for common edges
    2. Affinity model probabilities for PP attachments
    3. Privative adjective filtering

    Ports NaturalLogicWeights.java (lines 1-266)
    """

    def __init__(self, affinity_models: AffinityModels | None = None):
        """
        Initialize natural logic weights.

        Args:
            affinity_models: Loaded affinity models (loads default if None)
        """
        self.affinity_models = affinity_models if affinity_models else AffinityModels()

        # Upper probability cap for affinity scores
        # From NaturalLogicWeights.java:43
        self.upper_probability_cap = 1.0

    def deletion_probability(self, edge: Token, doc: Doc) -> float:
        """
        Get the deletion probability for a dependency edge.

        This is the main entry point for deletion weights.
        Ports NaturalLogicWeights.deletionProbability() from lines 47-105

        Args:
            edge: The dependency edge (child token)
            doc: The Spacy Doc containing the sentence

        Returns:
            Deletion probability (0.0 = must keep, 1.0 = should delete)
        """
        edge_type = edge.dep_
        edge_type_short = self._get_short_relation(edge_type)

        # Hard-coded rules (lines 50-96)

        # 1. Prepositional phrases - use affinity model
        if "prep" in edge_type_short or edge_type_short == "nmod":
            return self._pp_deletion_probability(edge, doc)

        # 2. Objects - special handling
        if "obj" in edge_type_short:
            return self._obj_deletion_probability(edge, doc)

        # 3. Subjects - special handling
        if "subj" in edge_type_short:
            return self._subj_deletion_probability(edge, doc)

        # 4. Adjectival modifiers - check for privative adjectives
        if edge_type_short == "amod":
            word = edge.text.lower()
            lemma = edge.lemma_.lower() if edge.lemma_ else word

            if word in PRIVATIVE_ADJECTIVES or lemma in PRIVATIVE_ADJECTIVES:
                return 0.0  # Don't delete privative adjectives
            else:
                return 1.0  # Safe to delete normal adjectives

        # 5. Adverbial clauses - generally deletable
        if edge_type_short == "advcl":
            return 1.0

        # 6. Determiners - generally keep
        if edge_type_short == "det":
            return 0.0

        # 7. Negations - must keep (critical for meaning)
        if edge_type_short == "neg":
            return 0.0

        # 8. Auxiliaries - keep for tense/mood
        if edge_type_short in ("aux", "auxpass"):
            return 0.0

        # 9. Conjunctions - medium priority
        if edge_type_short in ("cc", "conj"):
            return 0.5

        # 10. Relative clauses - deletable
        if edge_type_short == "acl" or edge_type == "acl:relcl":
            return 1.0

        # 11. Appositives - deletable
        if edge_type_short == "appos":
            return 1.0

        # Default: deletable
        return 1.0

    def _pp_deletion_probability(self, edge: Token, doc: Doc) -> float:
        """
        Get deletion probability for prepositional phrase attachments.

        Uses affinity models with fallback strategy:
        1. Try: subj + obj + pp
        2. Try: subj + pp + pp (if multiple PPs)
        3. Try: subj + pp
        4. Try: verb + pp
        5. Default: 0.9 (high probability of deletion)

        Ports NaturalLogicWeights.ppDeletionProbability() from lines 107-153

        Args:
            edge: The prepositional phrase edge
            doc: The Spacy Doc

        Returns:
            Deletion probability
        """
        parent = edge.head

        # Get preposition (the edge word itself, or the case marker)
        prep = self._get_preposition(edge)
        if not prep:
            return 0.9  # Default for PPs

        # Get neighbors (siblings of this edge)
        neighbors = list(parent.children)

        # Find subject and object
        subj = None
        obj = None
        other_pps = []

        for neighbor in neighbors:
            if neighbor == edge:
                continue

            rel = neighbor.dep_
            if "subj" in rel:
                subj = neighbor
            elif "obj" in rel:
                obj = neighbor
            elif "prep" in rel or rel == "nmod":
                other_pps.append(neighbor)

        # Get head verb lemma
        verb = parent.lemma_.lower() if parent.lemma_ else parent.text.lower()

        # Affinity lookup with fallback strategy

        # 1. Try: verb + subj + obj + pp
        if subj and obj:
            subj_word = subj.lemma_.lower() if subj.lemma_ else subj.text.lower()
            obj_word = obj.lemma_.lower() if obj.lemma_ else obj.text.lower()

            prob = self.affinity_models.get_subj_obj_pp_affinity(verb, subj_word, obj_word, prep)
            if prob is not None:
                return self._normalize_affinity_score(prob)

        # 2. Try: verb + subj + pp + pp (if multiple PPs)
        if subj and other_pps:
            subj_word = subj.lemma_.lower() if subj.lemma_ else subj.text.lower()
            other_prep = self._get_preposition(other_pps[0])

            if other_prep:
                prob = self.affinity_models.get_subj_pp_pp_affinity(
                    verb, subj_word, prep, other_prep
                )
                if prob is not None:
                    return self._normalize_affinity_score(prob)

        # 3. Try: verb + subj + pp
        if subj:
            subj_word = subj.lemma_.lower() if subj.lemma_ else subj.text.lower()

            prob = self.affinity_models.get_subj_pp_affinity(verb, subj_word, prep)
            if prob is not None:
                return self._normalize_affinity_score(prob)

        # 4. Try: verb + pp
        prob = self.affinity_models.get_pp_affinity(verb, prep)
        if prob is not None:
            return self._normalize_affinity_score(prob)

        # 5. Default for PPs
        return 0.9

    def _obj_deletion_probability(self, edge: Token, doc: Doc) -> float:
        """
        Get deletion probability for object edges.

        Objects are generally not deletable (would change meaning).
        But some cases allow deletion based on affinity.

        Ports NaturalLogicWeights.objDeletionProbability() from lines 155-180

        Args:
            edge: The object edge
            doc: The Spacy Doc

        Returns:
            Deletion probability
        """
        parent = edge.head

        # Get neighbors
        neighbors = list(parent.children)

        # Find subject
        subj = None
        for neighbor in neighbors:
            if neighbor == edge:
                continue
            if "subj" in neighbor.dep_:
                subj = neighbor
                break

        if not subj:
            return 0.0  # No subject, keep object

        # Get words
        verb = parent.lemma_.lower() if parent.lemma_ else parent.text.lower()

        # Try affinity lookup for verb's object deletion probability
        prob = self.affinity_models.get_obj_affinity(verb)

        if prob is not None:
            return self._normalize_affinity_score(prob)

        # Default: don't delete objects
        return 0.0

    def _subj_deletion_probability(self, edge: Token, doc: Doc) -> float:
        """
        Get deletion probability for subject edges.

        Subjects are generally not deletable.

        Ports NaturalLogicWeights.subjDeletionProbability() from lines 182-200

        Args:
            edge: The subject edge
            doc: The Spacy Doc

        Returns:
            Deletion probability (usually 0.0)
        """
        # Subjects are critical - don't delete
        # Exception: passive subjects in some cases

        if edge.dep_ == "nsubjpass":
            # Passive subject - might be deletable in some contexts
            # But for now, keep it
            return 0.0

        return 0.0

    def _normalize_affinity_score(self, raw_score: float) -> float:
        """
        Convert affinity model score to deletion probability.

        Affinity scores are counts/frequencies. We normalize them:
        - Higher affinity = more likely attachment = LESS likely deletion
        - Formula: 1.0 - min(1.0, rawScore / upperCap)

        Ports NaturalLogicWeights.java:204-210

        Args:
            raw_score: Raw affinity score from model

        Returns:
            Deletion probability (0.0-1.0)
        """
        capped = min(1.0, raw_score / self.upper_probability_cap)
        return 1.0 - capped

    def _get_preposition(self, edge: Token) -> str | None:
        """
        Get the preposition for a prepositional phrase edge.

        In Spacy, the preposition can be:
        1. The edge word itself (if it's ADP)
        2. A 'case' dependent of the edge

        Args:
            edge: The PP edge

        Returns:
            Preposition string (lowercase), or None
        """
        # Check if edge itself is a preposition
        if edge.pos_ == "ADP":
            return edge.text.lower()

        # Look for case marker child
        for child in edge.children:
            if child.dep_ == "case" and child.pos_ == "ADP":
                return child.text.lower()

        # For nmod edges, look at the relation subtype
        # E.g., "nmod:in" -> "in"
        if ":" in edge.dep_:
            parts = edge.dep_.split(":")
            if len(parts) == 2:
                return parts[1].lower()

        return None

    @staticmethod
    def _get_short_relation(relation: str) -> str:
        """
        Get short form of dependency relation.

        E.g., "nsubj:pass" -> "nsubj"

        Args:
            relation: Full dependency relation string

        Returns:
            Short form (before first underscore or colon)
        """
        if "_" in relation:
            return relation.split("_")[0]
        if ":" in relation:
            return relation.split(":")[0]
        return relation
