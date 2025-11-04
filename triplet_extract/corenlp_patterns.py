"""
Direct port of CoreNLP's RelationTripleSegmenter to Python/spaCy.

This module implements the same Semgrex patterns as:
edu.stanford.nlp.naturalli.RelationTripleSegmenter.VERB_PATTERNS

Reference: CoreNLP/src/edu/stanford/nlp/naturalli/RelationTripleSegmenter.java
"""

from dataclasses import dataclass

import spacy
from spacy.tokens import Doc, Token

from .openie.text_utils import reconstruct_text, reconstruct_text_from_strings


@dataclass
class Triple:
    """A relation triple: (subject, relation, object) with token-level information."""

    subject: str
    relation: str
    object: str

    # Token-level information for auxiliary filtering
    subject_tokens: list[Token] = None
    relation_tokens: list[Token] = None
    object_tokens: list[Token] = None
    relation_head: Token | None = None  # Head verb/predicate of relation

    def to_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)

    def to_dict(self) -> dict:
        return {"subject": self.subject, "relation": self.relation, "object": self.object}


class CoreNLPStyleExtractor:
    """
    Direct port of CoreNLP's RelationTripleSegmenter to Python/spaCy.

    Implements the same Semgrex patterns as RelationTripleSegmenter.VERB_PATTERNS:
    - Pattern 1 (Line 52): Basic SVO - "cats have tails"
    - Pattern 2 (Line 46): Copula - "cats are cute"
    - Pattern 3 (Line 48): Xcomp - "fish like to swim"
    - Pattern 4 (Line 43): Prepositional - "cats play with yarn"
    """

    def __init__(self, nlp: spacy.language.Language | None = None):
        """
        Initialize extractor with spaCy model.

        Args:
            nlp: spaCy language model. If None, loads en_core_web_sm.
        """
        if nlp is None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp

    def extract_triplets(self, text: str) -> list[Triple]:
        """
        Extract relation triples from text.

        Args:
            text: Input sentence

        Returns:
            List of Triple objects
        """
        doc = self.nlp(text)
        return self.extract_triplets_from_doc(doc)

    def extract_triplets_from_doc(self, doc) -> list[Triple]:
        """
        Extract relation triples from an already-parsed spaCy Doc.

        This is faster than extract_triplets() when you already have a parsed Doc,
        as it avoids reparsing the text.

        Args:
            doc: Parsed spaCy Doc object

        Returns:
            List of Triple objects
        """
        triples = []

        # Try each pattern in order (more specific patterns first)
        triples.extend(
            self._pattern_object_in_relation(doc)
        )  # Pattern 6: dobj with PP -> merge dobj into relation
        triples.extend(self._pattern_basic_svo(doc))  # Pattern 1: basic SVO
        triples.extend(self._pattern_copula(doc))  # Pattern 2: copula + adjective
        triples.extend(self._pattern_copula_xcomp(doc))  # Pattern 2b: copula + VBG/VBN
        triples.extend(self._pattern_copula_nominal(doc))  # Pattern 2c: copula + noun
        triples.extend(self._pattern_xcomp(doc))  # Pattern 3: xcomp
        triples.extend(self._pattern_ccomp(doc))  # Pattern 5: ccomp
        triples.extend(self._pattern_prepositional(doc))  # Pattern 4: prepositional
        triples.extend(self._pattern_acl(doc))  # Pattern 7: ACL (participial phrases)

        return triples

    def extract_triplets_as_strings(self, text: str) -> list[str]:
        """
        Extract triplets as "subject relation object" strings.

        This matches the format used by the existing triplet extractors.

        Args:
            text: Input sentence

        Returns:
            List of "subject relation object" strings
        """
        triples = self.extract_triplets(text)
        return [f"{t.subject} {t.relation} {t.object}" for t in triples]

    # ==================================================================
    # Pattern 6: Object in Relation
    # ==================================================================
    # CoreNLP Semgrex (Line 55):
    # {$}=verb >/.subj/ {}=subject >dobj {}=dobj >/nmod|obl:.*/=prepEdge {}=object
    #
    # When verb has BOTH dobj AND PP attached to dobj, merge dobj into relation.
    # Matches: "established production plant outside Södertälje"
    # ==================================================================

    def _pattern_object_in_relation(self, doc: Doc) -> list[Triple]:
        """
        Pattern 6: Object in relation.

        When verb has BOTH dobj AND prepositional phrase attached to dobj,
        merge the dobj into the relation (not the object).

        Example: "established production plant outside Södertälje"
        -> (subject, "established production plant outside", "Södertälje")

        CoreNLP Pattern (line 55):
        {$}=verb >/.subj/ {}=subject >dobj {}=dobj >/nmod|obl:.*/=prepEdge {}=object

        spaCy structure:
        - VERB with nsubj and dobj
        - dobj has prep child (PP attached to object, not verb)
        """
        triples = []

        for token in doc:
            if not token.pos_.startswith("VERB"):
                continue

            # Find subject
            subject = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
            if subject is None:
                continue

            # Find direct object
            dobj = next((c for c in token.children if c.dep_ in ("obj", "dobj")), None)
            if dobj is None:
                continue

            # Check if dobj has prepositional attachment
            # (NOT the verb - that's Pattern 4)
            prep_on_dobj = [c for c in dobj.children if c.dep_ == "prep"]

            if not prep_on_dobj:
                continue  # No PP on dobj, skip (Pattern 1 handles this)

            # Extract prepositional phrase from dobj
            for prep in prep_on_dobj:
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj:
                    subj_text = self._get_subtree_text(subject)

                    # Relation = verb + dobj (with compounds) + preposition
                    verb_text = self._build_relation(token)

                    # Include compound modifiers of dobj
                    dobj_parts = []
                    for child in dobj.children:
                        if child.dep_ == "compound":
                            dobj_parts.append(child.text)
                    dobj_parts.append(dobj.text)
                    dobj_text = reconstruct_text_from_strings(dobj_parts)

                    relation_text = f"{verb_text} {dobj_text} {prep.text}"

                    # Object = prepositional object only
                    obj_text = self._get_subtree_text(pobj)

                    # Extract tokens for auxiliary filtering
                    subj_tokens = self._get_subtree_tokens(subject)
                    rel_tokens, rel_head = self._build_relation_tokens(token)
                    obj_tokens = self._get_subtree_tokens(pobj)

                    triple = Triple(
                        subject=subj_text,
                        relation=relation_text,
                        object=obj_text,
                        subject_tokens=subj_tokens,
                        relation_tokens=rel_tokens,
                        object_tokens=obj_tokens,
                        relation_head=rel_head,
                    )
                    triples.append(triple)

        return triples

    # ==================================================================
    # Pattern 1: Basic SVO
    # ==================================================================
    # CoreNLP Semgrex (Line 52):
    # {$}=verb ?>/aux(:pass)?/ {}=be >/.subj(:pass)?/ {}=subject >/[di]?obj|xcomp/ {}=object
    #
    # Matches: "cats have tails"
    # ==================================================================

    def _pattern_basic_svo(self, doc: Doc) -> list[Triple]:
        """
        Pattern 1: Basic Subject-Verb-Object.

        CoreNLP pattern (line 52):
        {$}=verb ?>/aux(:pass)?/ {}=be >/.subj(:pass)?/ {}=subject >/[di]?obj|xcomp/ {}=object

        Example: "cats have tails" -> (cats, have, tails)

        Note: Skip verbs with xcomp/ccomp children (handled by Patterns 3 & 5)
        Note: Skip verbs that ARE xcomp/ccomp children (to avoid duplicates)
        """
        triples = []

        for token in doc:
            # Must be a verb
            if not token.pos_.startswith("VERB"):
                continue

            # Skip if has xcomp or ccomp (Patterns 3 & 5 handle these)
            has_xcomp = any(child.dep_ == "xcomp" for child in token.children)
            has_ccomp = any(child.dep_ == "ccomp" for child in token.children)
            if has_xcomp or has_ccomp:
                continue

            # Skip if IS xcomp or ccomp (to avoid extracting embedded clauses twice)
            if token.dep_ in ("xcomp", "ccomp"):
                continue

            # Find subject (nsubj or nsubjpass)
            subject = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child
                    break

            if subject is None:
                continue

            # Find object (obj, dobj)
            obj = None
            for child in token.children:
                if child.dep_ in ("obj", "dobj"):
                    obj = child
                    break

            if obj is None:
                continue

            # Skip if dobj has prep children (Pattern 6 handles this - object in relation)
            if any(child.dep_ == "prep" for child in obj.children):
                continue

            # Extract spans
            subj_text = self._get_subtree_text(subject)
            obj_text = self._get_subtree_text(obj)

            # Handle negation: check for "neg" dependency
            relation_text = self._build_relation(token)

            # Extract tokens for auxiliary filtering
            subj_tokens = self._get_subtree_tokens(subject)
            rel_tokens, rel_head = self._build_relation_tokens(token)
            obj_tokens = self._get_subtree_tokens(obj)

            triple = Triple(
                subject=subj_text,
                relation=relation_text,
                object=obj_text,
                subject_tokens=subj_tokens,
                relation_tokens=rel_tokens,
                object_tokens=obj_tokens,
                relation_head=rel_head,
            )
            triples.append(triple)

        return triples

    # ==================================================================
    # Pattern 2: Copula
    # ==================================================================
    # CoreNLP Semgrex (Line 46):
    # {$}=object >/.subj(:pass)?/ {}=subject >/cop|aux(:pass)?/ {}=verb
    #
    # Matches: "cats are cute"
    # ==================================================================

    def _pattern_copula(self, doc: Doc) -> list[Triple]:
        """
        Pattern 2: Copula constructions.

        CoreNLP pattern (line 46):
        {$}=object >/.subj(:pass)?/ {}=subject >/cop|aux(:pass)?/ {}=verb

        In spaCy, copula sentences have "is/are" as ROOT with:
        - nsubj child (subject)
        - acomp child (adjectival complement = object)

        Examples:
        - Simple: "cats are cute" -> (cats, are, cute)
        - Coordinated with PP: "Obama is 44th and current president of US" -> (Obama, is 44th and current president of, US)
        """
        triples = []

        for token in doc:
            # Must be auxiliary verb (is/are/was/were) as ROOT
            if token.pos_ != "AUX" or token.dep_ != "ROOT":
                continue

            # Find subject
            subject = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child
                    break

            if subject is None:
                continue

            # Find adjectival complement (the predicate)
            acomp = None
            for child in token.children:
                if child.dep_ == "acomp":
                    acomp = child
                    break

            if acomp is None:
                continue

            # Extract spans
            subj_text = self._get_subtree_text(subject)

            # Check if ROOT has coordinated predicate (e.g., "44th and current president of US")
            # In this case, acomp and conj are siblings, both children of ROOT
            conj_with_prep = None
            for child in token.children:
                if child.dep_ == "conj" and child != acomp:
                    # Check if this conj has a prep child
                    prep = next((c for c in child.children if c.dep_ == "prep"), None)
                    if prep:
                        conj_with_prep = (child, prep)
                        break

            if conj_with_prep:
                # Case: "Obama is 44th and current president of US"
                # Relation: "is" + acomp text + "and" + conj text (with compounds) + prep
                # Object: pobj
                conj, prep = conj_with_prep
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj:
                    # Build relation: copula + acomp + cc + conj (with modifiers) + prep
                    copula_text = self._build_relation(token)

                    # Get acomp text (just the head)
                    acomp_text = acomp.text

                    # Get coordinating conjunction (and/or)
                    cc = next((c.text for c in token.children if c.dep_ == "cc"), "and")

                    # Get conj text with modifiers (amod, compound)
                    conj_parts = []
                    for child in conj.children:
                        if child.dep_ in ("compound", "amod"):
                            conj_parts.append(child.text)
                    conj_parts.append(conj.text)
                    conj_text = reconstruct_text_from_strings(conj_parts)

                    relation_text = f"{copula_text} {acomp_text} {cc} {conj_text} {prep.text}"
                    obj_text = self._get_subtree_text(pobj)

                    # Extract tokens for auxiliary filtering
                    subj_tokens = self._get_subtree_tokens(subject)
                    rel_tokens, rel_head = self._build_relation_tokens(token)
                    obj_tokens = self._get_subtree_tokens(pobj)

                    triple = Triple(
                        subject=subj_text,
                        relation=relation_text,
                        object=obj_text,
                        subject_tokens=subj_tokens,
                        relation_tokens=rel_tokens,
                        object_tokens=obj_tokens,
                        relation_head=rel_head,
                    )
                    triples.append(triple)
            else:
                # Simple case: just extract acomp subtree
                obj_text = self._get_subtree_text(acomp)

                # Build relation from copula verb (handles negation)
                relation_text = self._build_relation(token)

                # Extract tokens for auxiliary filtering
                subj_tokens = self._get_subtree_tokens(subject)
                rel_tokens, rel_head = self._build_relation_tokens(token)
                obj_tokens = self._get_subtree_tokens(acomp)

                triple = Triple(
                    subject=subj_text,
                    relation=relation_text,
                    object=obj_text,
                    subject_tokens=subj_tokens,
                    relation_tokens=rel_tokens,
                    object_tokens=obj_tokens,
                    relation_head=rel_head,
                )
                triples.append(triple)

        return triples

    # ==================================================================
    # Pattern 2b: Copula + Xcomp (VBG/VBN)
    # ==================================================================
    # Handles "are/is + VBG/VBN" where the verb is ROOT with aux child.
    #
    # Matches: "horses are grazing peacefully"
    # ==================================================================

    def _pattern_copula_xcomp(self, doc: Doc) -> list[Triple]:
        """
        Pattern 2b: Copula with verbal complement.

        Handles "are/is + VBG/VBN" constructions where the main verb is ROOT.
        This is different from Pattern 2 where copula is ROOT with acomp.

        Examples:
        - Active: "horses are grazing peacefully" -> (horses, are, grazing peacefully)
        - Passive: "things are arranged neatly" -> (things, are, arranged neatly)
        - Passive with oprd: "Obama was named 2009 Nobel Peace Prize Laureate" -> (Obama, was named, 2009 Nobel Peace Prize Laureate)

        spaCy structure:
        - VERB(ROOT) with aux/auxpass child (is/are)
        - nsubj/nsubjpass child (subject)
        - Optional advmod (adverbs)
        - Optional oprd (object predicate for passive constructions)
        - No regular object (intransitive, unless oprd present)
        """
        triples = []

        for token in doc:
            # Must be VERB as ROOT
            if not token.pos_.startswith("VERB") or token.dep_ != "ROOT":
                continue

            # Must have aux or auxpass child (copula: is/are/was/were)
            aux = next((c for c in token.children if c.dep_ in ("aux", "auxpass")), None)
            if aux is None:
                continue

            # Find subject
            subject = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
            if subject is None:
                continue

            # Check for oprd (object predicate) - used in passive constructions
            oprd = next((c for c in token.children if c.dep_ == "oprd"), None)
            if oprd:
                # Case: "Obama was named 2009 Nobel Peace Prize Laureate"
                # Relation: auxpass + verb = "was named"
                # Object: oprd subtree = "2009 Nobel Peace Prize Laureate"
                subj_text = self._get_subtree_text(subject)
                relation_text = f"{aux.text} {token.text}"
                obj_text = self._get_subtree_text(oprd)

                # Extract tokens for auxiliary filtering
                subj_tokens = self._get_subtree_tokens(subject)
                rel_tokens, rel_head = self._build_relation_tokens(token)
                obj_tokens = self._get_subtree_tokens(oprd)

                triple = Triple(
                    subject=subj_text,
                    relation=relation_text,
                    object=obj_text,
                    subject_tokens=subj_tokens,
                    relation_tokens=rel_tokens,
                    object_tokens=obj_tokens,
                    relation_head=rel_head,
                )
                triples.append(triple)
                continue

            # Must NOT have object, prepositional attachment, or clausal complement (this is intransitive pattern)
            # This prevents matching "was inaugurated as president" which should use prepositional pattern
            # Also prevents matching "don't know what X is" which should use ccomp pattern
            has_obj = any(c.dep_ in ("obj", "dobj") for c in token.children)
            has_prep = any(c.dep_ == "prep" for c in token.children)
            has_ccomp = any(c.dep_ == "ccomp" for c in token.children)
            has_xcomp = any(c.dep_ == "xcomp" for c in token.children)
            if has_obj or has_prep or has_ccomp or has_xcomp:
                continue

            # Extract spans
            subj_text = self._get_subtree_text(subject)

            # Check if any advmod has prep children (e.g., "standing next to horse")
            advmod_with_prep = None
            for child in token.children:
                if child.dep_ == "advmod":
                    prep_on_advmod = next((c for c in child.children if c.dep_ == "prep"), None)
                    if prep_on_advmod:
                        advmod_with_prep = (child, prep_on_advmod)
                        break

            if advmod_with_prep:
                # Case: "be standing next to horse"
                # Relation: aux + verb + advmod + prep = "be standing next to"
                # Object: pobj = "horse"
                advmod, prep = advmod_with_prep
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj:
                    relation_text = f"{aux.text} {token.text} {advmod.text} {prep.text}"
                    obj_text = self._get_subtree_text(pobj)

                    # Extract tokens for auxiliary filtering
                    subj_tokens = self._get_subtree_tokens(subject)
                    rel_tokens, rel_head = self._build_relation_tokens(token)
                    obj_tokens = self._get_subtree_tokens(pobj)

                    triple = Triple(
                        subject=subj_text,
                        relation=relation_text,
                        object=obj_text,
                        subject_tokens=subj_tokens,
                        relation_tokens=rel_tokens,
                        object_tokens=obj_tokens,
                        relation_head=rel_head,
                    )
                    triples.append(triple)
            else:
                # Case: "horses are grazing peacefully" (intransitive)
                # Relation: just auxiliary
                # Object: verb + advmods
                obj_parts = [token.text]
                for child in token.children:
                    if child.dep_ == "advmod":
                        obj_parts.append(child.text)
                obj_text = reconstruct_text_from_strings(obj_parts)

                # Relation is just the auxiliary (is/are)
                relation_text = aux.text

                # Extract tokens for auxiliary filtering
                subj_tokens = self._get_subtree_tokens(subject)
                rel_tokens, rel_head = self._build_relation_tokens(token)
                # Object tokens include the main verb and its advmods
                obj_tokens = [token]
                for child in token.children:
                    if child.dep_ == "advmod":
                        obj_tokens.append(child)

                triple = Triple(
                    subject=subj_text,
                    relation=relation_text,
                    object=obj_text,
                    subject_tokens=subj_tokens,
                    relation_tokens=rel_tokens,
                    object_tokens=obj_tokens,
                    relation_head=rel_head,
                )
                triples.append(triple)

        return triples

    # ==================================================================
    # Pattern 2c: Copula + Nominal Predicate
    # ==================================================================
    # Handles "is/are + NOUN" constructions (vs adjectives in Pattern 2).
    #
    # Matches: "Obama is President", "Obama is president of US"
    # ==================================================================

    def _pattern_copula_nominal(self, doc: Doc) -> list[Triple]:
        """
        Pattern 2c: Copula with nominal predicate.

        Handles "is/are + NOUN" constructions (vs adjectives in Pattern 2).

        Examples:
        - "Obama is President" -> (Obama, is, President)
        - "Obama is president of US" -> (Obama, is president of, US)
        - "he was community organizer in Chicago" -> (he, was, community organizer in Chicago)

        spaCy structure:
        - AUX(ROOT) with attr child (nominal predicate)
        - nsubj/nsubjpass child
        - attr may have prep children (PP attachment)

        Logic for PP attachment:
        - If attr has compound modifier + prep: keep full subtree in object
        - If attr has prep but NO compound: merge attr + prep into relation
        """
        triples = []

        for token in doc:
            # Must be AUX as ROOT
            if token.pos_ != "AUX" or token.dep_ != "ROOT":
                continue

            # Find subject
            subject = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
            if subject is None:
                continue

            # Find attribute (nominal predicate)
            attr = next((c for c in token.children if c.dep_ == "attr"), None)
            if attr is None:
                continue

            subj_text = self._get_subtree_text(subject)

            # Check if attr has prep child
            prep = next((c for c in attr.children if c.dep_ == "prep"), None)

            if prep:
                # Check if attr has compound modifier
                has_compound = any(c.dep_ == "compound" for c in attr.children)

                if has_compound:
                    # Case: "community organizer in Chicago"
                    # Keep full attr subtree in object
                    obj_text = self._get_subtree_text(attr)
                    relation_text = self._build_relation(token)

                    # Extract tokens for auxiliary filtering
                    subj_tokens = self._get_subtree_tokens(subject)
                    rel_tokens, rel_head = self._build_relation_tokens(token)
                    obj_tokens = self._get_subtree_tokens(attr)

                    triple = Triple(
                        subject=subj_text,
                        relation=relation_text,
                        object=obj_text,
                        subject_tokens=subj_tokens,
                        relation_tokens=rel_tokens,
                        object_tokens=obj_tokens,
                        relation_head=rel_head,
                    )
                    triples.append(triple)
                else:
                    # Case: "president of US"
                    # Merge attr + prep into relation, pobj as object
                    pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                    if pobj:
                        copula_text = self._build_relation(token)
                        relation_text = f"{copula_text} {attr.text} {prep.text}"
                        obj_text = self._get_subtree_text(pobj)

                        # Extract tokens for auxiliary filtering
                        subj_tokens = self._get_subtree_tokens(subject)
                        rel_tokens, rel_head = self._build_relation_tokens(token)
                        obj_tokens = self._get_subtree_tokens(pobj)

                        triple = Triple(
                            subject=subj_text,
                            relation=relation_text,
                            object=obj_text,
                            subject_tokens=subj_tokens,
                            relation_tokens=rel_tokens,
                            object_tokens=obj_tokens,
                            relation_head=rel_head,
                        )
                        triples.append(triple)
            else:
                # Case: "Obama is President" (simple, no PP)
                obj_text = self._get_subtree_text(attr)
                relation_text = self._build_relation(token)

                # Extract tokens for auxiliary filtering
                subj_tokens = self._get_subtree_tokens(subject)
                rel_tokens, rel_head = self._build_relation_tokens(token)
                obj_tokens = self._get_subtree_tokens(attr)

                triple = Triple(
                    subject=subj_text,
                    relation=relation_text,
                    object=obj_text,
                    subject_tokens=subj_tokens,
                    relation_tokens=rel_tokens,
                    object_tokens=obj_tokens,
                    relation_head=rel_head,
                )
                triples.append(triple)

        return triples

    # ==================================================================
    # Pattern 3: Xcomp
    # ==================================================================
    # CoreNLP Semgrex (Line 48):
    # {$}=verb >/.subj(:pass)?/ {}=subject >xcomp ( {}=object ?>appos {}=appos )
    #
    # Matches: "fish like to swim"
    # ==================================================================

    def _pattern_xcomp(self, doc: Doc) -> list[Triple]:
        """
        Pattern 3: Xcomp constructions.

        CoreNLP pattern (line 48):
        {$}=verb >/.subj(:pass)?/ {}=subject >xcomp ( {}=object ?>appos {}=appos )

        In spaCy, xcomp sentences have:
        - Main verb (ROOT) with nsubj and xcomp children
        - xcomp is a verb with its own complements

        Examples:
        - "fish like to swim" -> (fish, like, to swim)
        - "I persuaded Fred to leave the room" ->
            (I, persuaded, Fred to leave the room)
            (Fred, leave, the room)
        """
        triples = []

        for token in doc:
            # Must be a verb
            if not token.pos_.startswith("VERB"):
                continue

            # Find subject
            subject = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child
                    break

            if subject is None:
                continue

            # Find xcomp (clausal complement)
            xcomp = None
            for child in token.children:
                if child.dep_ == "xcomp":
                    xcomp = child
                    break

            if xcomp is None:
                continue

            # Extract subject text
            subj_text = self._get_subtree_text(subject)

            # Build object text
            # For active voice with dobj: include both dobj and xcomp ("Fred to leave the room")
            # For passive/other: just xcomp ("to leave the room")
            obj_text = ""
            dobj = None
            for child in token.children:
                if child.dep_ in ("dobj", "obj"):
                    dobj = child
                    break

            if dobj is not None:
                # Active voice: "I persuaded Fred to leave the room"
                # Object = "Fred to leave the room"
                dobj_text = self._get_subtree_text(dobj)
                xcomp_text = self._get_subtree_text(xcomp)
                obj_text = f"{dobj_text} {xcomp_text}"
            else:
                # Passive/other: "I was persuaded to leave the room"
                # Object = "to leave the room"
                obj_text = self._get_subtree_text(xcomp)

            # Build relation
            relation_text = self._build_relation(token)

            # Extract tokens for auxiliary filtering (main triple)
            subj_tokens = self._get_subtree_tokens(subject)
            rel_tokens, rel_head = self._build_relation_tokens(token)
            # Object tokens depend on whether we have dobj
            if dobj is not None:
                obj_tokens = self._get_subtree_tokens(dobj) + self._get_subtree_tokens(xcomp)
            else:
                obj_tokens = self._get_subtree_tokens(xcomp)

            # Create main triple
            triple = Triple(
                subject=subj_text,
                relation=relation_text,
                object=obj_text,
                subject_tokens=subj_tokens,
                relation_tokens=rel_tokens,
                object_tokens=obj_tokens,
                relation_head=rel_head,
            )
            triples.append(triple)

            # Extract embedded clause from xcomp
            # For active voice with dobj: (Fred, leave, the room)
            # For passive voice: (I, leave, the room)
            embedded_subj = None

            # Check for dobj (active voice: "I persuaded Fred to leave")
            dobj = None
            for child in token.children:
                if child.dep_ in ("dobj", "obj"):
                    dobj = child
                    embedded_subj = dobj
                    break

            # If no dobj, check for nsubjpass (passive voice: "I was persuaded to leave")
            if embedded_subj is None:
                for child in token.children:
                    if child.dep_ == "nsubjpass":
                        embedded_subj = child
                        break

            if embedded_subj is not None:
                # Find the object of the xcomp verb
                xcomp_obj = None
                for child in xcomp.children:
                    if child.dep_ in ("dobj", "obj"):
                        xcomp_obj = child
                        break

                if xcomp_obj is not None:
                    embedded_subj_text = self._get_subtree_text(embedded_subj)
                    xcomp_obj_text = self._get_subtree_text(xcomp_obj)

                    # Build relation from xcomp verb
                    xcomp_relation = self._build_relation(xcomp)

                    # Extract tokens for auxiliary filtering (embedded triple)
                    embedded_subj_tokens = self._get_subtree_tokens(embedded_subj)
                    xcomp_rel_tokens, xcomp_rel_head = self._build_relation_tokens(xcomp)
                    xcomp_obj_tokens = self._get_subtree_tokens(xcomp_obj)

                    # Create embedded triple
                    embedded_triple = Triple(
                        subject=embedded_subj_text,
                        relation=xcomp_relation,
                        object=xcomp_obj_text,
                        subject_tokens=embedded_subj_tokens,
                        relation_tokens=xcomp_rel_tokens,
                        object_tokens=xcomp_obj_tokens,
                        relation_head=xcomp_rel_head,
                    )
                    triples.append(embedded_triple)

        return triples

    # ==================================================================
    # Pattern 5: Ccomp (Clausal Complement)
    # ==================================================================
    # CoreNLP handles clausal complements with "that" similarly to xcomp.
    #
    # Matches: "I suggested that he leave the room"
    # ==================================================================

    def _pattern_ccomp(self, doc: Doc) -> list[Triple]:
        """
        Pattern 5: Clausal complement constructions.

        Handles finite clausal complements introduced by "that".
        Similar to xcomp but for finite clauses.

        In spaCy, ccomp sentences have:
        - Main verb (ROOT) with nsubj and ccomp children
        - ccomp is a verb with mark ("that"), nsubj, and complements

        Example: "I suggested that he leave the room" ->
            ("I", "suggested", "that he leave the room")
            ("he", "leave", "the room")
        """
        triples = []

        for token in doc:
            # Must be a verb
            if not token.pos_.startswith("VERB"):
                continue

            # Find subject
            subject = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child
                    break

            if subject is None:
                continue

            # Find ccomp (clausal complement)
            ccomp = None
            for child in token.children:
                if child.dep_ == "ccomp":
                    ccomp = child
                    break

            if ccomp is None:
                continue

            # Extract subject text
            subj_text = self._get_subtree_text(subject)

            # Build object text (entire ccomp clause including "that")
            obj_text = self._get_subtree_text(ccomp)

            # Build relation
            relation_text = self._build_relation(token)

            # Extract tokens for auxiliary filtering (main triple)
            subj_tokens = self._get_subtree_tokens(subject)
            rel_tokens, rel_head = self._build_relation_tokens(token)
            obj_tokens = self._get_subtree_tokens(ccomp)

            # Create main triple
            triple = Triple(
                subject=subj_text,
                relation=relation_text,
                object=obj_text,
                subject_tokens=subj_tokens,
                relation_tokens=rel_tokens,
                object_tokens=obj_tokens,
                relation_head=rel_head,
            )
            triples.append(triple)

            # Extract embedded clause from ccomp
            # Find subject of ccomp verb
            ccomp_subj = None
            for child in ccomp.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    ccomp_subj = child
                    break

            # Find object of ccomp verb
            ccomp_obj = None
            for child in ccomp.children:
                if child.dep_ in ("dobj", "obj"):
                    ccomp_obj = child
                    break

            if ccomp_subj is not None and ccomp_obj is not None:
                embedded_subj_text = self._get_subtree_text(ccomp_subj)
                ccomp_obj_text = self._get_subtree_text(ccomp_obj)

                # Build relation from ccomp verb
                ccomp_relation = self._build_relation(ccomp)

                # Extract tokens for auxiliary filtering (embedded triple)
                ccomp_subj_tokens = self._get_subtree_tokens(ccomp_subj)
                ccomp_rel_tokens, ccomp_rel_head = self._build_relation_tokens(ccomp)
                ccomp_obj_tokens = self._get_subtree_tokens(ccomp_obj)

                # Create embedded triple
                embedded_triple = Triple(
                    subject=embedded_subj_text,
                    relation=ccomp_relation,
                    object=ccomp_obj_text,
                    subject_tokens=ccomp_subj_tokens,
                    relation_tokens=ccomp_rel_tokens,
                    object_tokens=ccomp_obj_tokens,
                    relation_head=ccomp_rel_head,
                )
                triples.append(embedded_triple)

        return triples

    # ==================================================================
    # Pattern 4: Prepositional
    # ==================================================================
    # CoreNLP Semgrex (Line 43):
    # {$}=verb >/.subj/ {}=subject >/(nmod|obl):.*/=prepEdge {}=object
    #
    # Matches: "cats play with yarn"
    # ==================================================================

    def _pattern_prepositional(self, doc: Doc) -> list[Triple]:
        """
        Pattern 4: Prepositional attachment.

        CoreNLP pattern (line 43):
        {$}=verb >/.subj/ {}=subject >/(nmod|obl):.*/=prepEdge {}=object

        In spaCy, prepositional phrases have:
        - Verb with prep child (preposition)
        - Prep has pobj child (prepositional object)

        Example: "cats play with yarn" -> (cats, play with, yarn)
        """
        triples = []

        for token in doc:
            # Must be a verb or auxiliary (to handle copula "is on")
            if not (token.pos_.startswith("VERB") or token.pos_ == "AUX"):
                continue

            # Find subject
            subject = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child
                    break

            if subject is None:
                continue

            # Find prepositional phrases
            for child in token.children:
                if child.dep_ == "prep":
                    prep = child

                    # Find prepositional object
                    pobj = None
                    for prep_child in prep.children:
                        if prep_child.dep_ == "pobj":
                            pobj = prep_child
                            break

                    if pobj is None:
                        continue

                    # Extract spans
                    subj_text = self._get_subtree_text(subject)
                    pobj_text = self._get_subtree_text(pobj)

                    # Build relation: verb + preposition
                    base_relation = self._build_relation(token)
                    relation_text = f"{base_relation} {prep.text}"

                    # Extract tokens for auxiliary filtering
                    subj_tokens = self._get_subtree_tokens(subject)
                    rel_tokens, rel_head = self._build_relation_tokens(token)
                    obj_tokens = self._get_subtree_tokens(pobj)

                    triple = Triple(
                        subject=subj_text,
                        relation=relation_text,
                        object=pobj_text,
                        subject_tokens=subj_tokens,
                        relation_tokens=rel_tokens,
                        object_tokens=obj_tokens,
                        relation_head=rel_head,
                    )
                    triples.append(triple)

        return triples

    # ==================================================================
    # Pattern 7: ACL (Adjectival Clause / Participial Phrase)
    # ==================================================================
    # Handles participial phrases modifying nouns.
    #
    # Matches: "dogs sitting in heaven"
    # ==================================================================

    def _pattern_acl(self, doc: Doc) -> list[Triple]:
        """
        Pattern 7: ACL (Adjectival Clause / Participial Phrase).

        Handles VBG/VBN forms modifying nouns via acl dependency.

        Example: "dogs sitting in heaven" -> (dogs, sitting in, heaven)

        spaCy structure:
        - NOUN(ROOT) with acl child (VERB)
        - acl verb has prep child with pobj
        """
        triples = []

        for token in doc:
            # Must be a NOUN or PROPN as ROOT or main element
            if token.pos_ not in ("NOUN", "PROPN"):
                continue

            # Find acl child (participial phrase)
            acl_verbs = [c for c in token.children if c.dep_ == "acl"]

            for acl in acl_verbs:
                # Check if acl has prepositional phrase
                for prep in acl.children:
                    if prep.dep_ == "prep":
                        pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                        if pobj:
                            # Subject = just the noun (not the modifying acl)
                            subj_text = token.text

                            # Relation = acl verb + preposition
                            relation_text = f"{acl.text} {prep.text}"

                            # Object = prepositional object
                            obj_text = self._get_subtree_text(pobj)

                            # Extract tokens for auxiliary filtering
                            subj_tokens = [token]  # Just the noun head
                            rel_tokens, rel_head = self._build_relation_tokens(acl)
                            obj_tokens = self._get_subtree_tokens(pobj)

                            triple = Triple(
                                subject=subj_text,
                                relation=relation_text,
                                object=obj_text,
                                subject_tokens=subj_tokens,
                                relation_tokens=rel_tokens,
                                object_tokens=obj_tokens,
                                relation_head=rel_head,
                            )
                            triples.append(triple)

        return triples

    # ==================================================================
    # Helper Methods
    # ==================================================================

    def _build_relation_tokens(self, verb: Token) -> tuple[list[Token], Token]:
        """
        Build relation token list from verb token.

        Returns both the tokens and the head token (main verb or auxiliary).

        Args:
            verb: The verb token

        Returns:
            Tuple of (relation_tokens, relation_head)
        """
        # Collect ALL auxiliary tokens (can have multiple: "can't be treated" has both aux and auxpass)
        aux_tokens = [child for child in verb.children if child.dep_ in ("aux", "auxpass")]

        # Check for negation tokens
        neg_tokens = [child for child in verb.children if child.dep_ == "neg"]

        # Collect adverb modifiers
        advmods = [child for child in verb.children if child.dep_ == "advmod"]

        # Build relation tokens
        # Include ALL auxiliaries + main verb + advmods + negation
        tokens = []

        # Add all auxiliaries
        tokens.extend(aux_tokens)

        # Determine relation head (first auxiliary if any, otherwise main verb)
        relation_head = aux_tokens[0] if aux_tokens else verb

        # Add main verb
        tokens.append(verb)

        # Add adverbs (sorted by position to maintain order)
        for advmod in sorted(advmods, key=lambda t: t.i):
            tokens.append(advmod)

        # Add negation tokens
        for neg in neg_tokens:
            tokens.append(neg)

        # Sort tokens by position to maintain proper word order
        tokens.sort(key=lambda t: t.i)

        return tokens, relation_head

    def _build_relation(self, verb: Token) -> str:
        """
        Build relation text from verb token.

        Handles negation properly: "is not" (NOT "not is").
        Handles passive auxiliary: "was persuaded" (NOT just "persuaded").
        Handles perfect/modal auxiliary: "has joined" (NOT just "joined").
        Handles adverbs: "play quietly with" (NOT just "play with").
        Matches CoreNLP's relation construction.

        Args:
            verb: The verb token

        Returns:
            Relation string (e.g., "have", "is not", "was persuaded", "has joined", "play quietly")
        """
        # Get tokens and reconstruct text
        tokens, _ = self._build_relation_tokens(verb)
        return reconstruct_text(tokens)

    def _get_subtree_tokens(self, token: Token) -> list[Token]:
        """
        Get tokens of token's subtree (the token and all its descendants).

        Args:
            token: The root token

        Returns:
            List of tokens in subtree, sorted by position
        """
        subtree_tokens = list(token.subtree)
        subtree_tokens.sort(key=lambda t: t.i)
        return subtree_tokens

    def _get_subtree_text(self, token: Token) -> str:
        """
        Get text of token's subtree (the token and all its descendants).

        This mirrors CoreNLP's span extraction behavior.

        Args:
            token: The root token

        Returns:
            Text of the subtree
        """
        # Get all tokens in subtree
        subtree_tokens = self._get_subtree_tokens(token)

        # Use utility function for proper text reconstruction
        return reconstruct_text(subtree_tokens)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
