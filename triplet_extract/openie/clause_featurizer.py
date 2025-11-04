"""
Clause Featurizer for ClauseSplitter

Ports the DEFAULT_FEATURIZER from Stanford's ClauseSplitterSearchProblem.
Generates features for the LinearClassifier based on dependency parse structure.

Ported from ClauseSplitterSearchProblem.java:933-1037
"""

from collections import Counter

from spacy.tokens import Doc, Token


class ClauseFeaturizer:
    """
    Featurizes clause splitting decisions for the LinearClassifier.

    Generates features based on:
    - Edge types (dependency relations)
    - Parent and child neighbors in parse tree
    - Subject/object presence
    - POS tags
    - Root vs non-root position
    """

    def __init__(self):
        """Initialize the featurizer."""
        pass

    @staticmethod
    def is_simple_split(features: Counter) -> bool:
        """
        Check if features represent a simple split action.

        Args:
            features: Feature counter

        Returns:
            True if any feature starts with "simple&"
        """
        for feature_name in features:
            if feature_name.startswith("simple&"):
                return True
        return False

    def featurize(
        self,
        parent_token: Token | None,
        edge_token: Token | None,
        action_signature: str,
        doc: Doc,
    ) -> Counter:
        """
        Generate features for a clause splitting decision.

        This matches the Java implementation:
        ClauseSplitterSearchProblem.DEFAULT_FEATURIZER.apply()

        Args:
            parent_token: The parent token in the "from" state (can be None for root)
            edge_token: The edge token in the "to" state (the token being considered)
            action_signature: Action type ("simple", "clone_nsubj", etc.)
            doc: The Spacy Doc containing the parse tree

        Returns:
            Counter of feature name -> count (typically 1.0)
        """
        features = Counter()

        # Handle root case
        if edge_token is None:
            return features  # No features for null edge

        # Get edge relation and short form
        edge_rel = edge_token.dep_
        edge_rel_short = self._get_short_relation(edge_rel)

        # 1. Edge taken features
        features[f"{action_signature}&edge:{edge_rel}"] = 1.0
        features[f"{action_signature}&edge_type:{edge_rel_short}"] = 1.0

        # 2. Last edge taken (at_root vs not_root)
        if parent_token is None or parent_token.dep_ == "ROOT":
            # At root
            features[f"{action_signature}&at_root"] = 1.0

            # Get root POS tag
            root_token = self._get_root_token(doc)
            if root_token:
                features[f"{action_signature}&at_root&root_pos:{root_token.tag_}"] = 1.0
        else:
            # Not at root
            features[f"{action_signature}&not_root"] = 1.0

            # Last edge relation
            last_rel_short = self._get_short_relation(parent_token.dep_)
            features[f"{action_signature}&last_edge:{last_rel_short}"] = 1.0

        # 3. Parent neighbor features (siblings of edge_token)
        parent_has_subj = False
        parent_has_obj = False

        # Get parent node (governor of edge_token)
        parent_node = edge_token.head

        # Iterate over children of parent (siblings of edge_token)
        for sibling in parent_node.children:
            if sibling != edge_token:
                sibling_rel = sibling.dep_

                # Check for subject/object
                if "subj" in sibling_rel.lower():
                    parent_has_subj = True
                if "obj" in sibling_rel.lower():
                    parent_has_obj = True

                # Add neighbor features
                features[f"{action_signature}&parent_neighbor:{sibling_rel}"] = 1.0
                features[
                    f"{action_signature}&edge_type:{edge_rel_short}&parent_neighbor:{sibling_rel}"
                ] = 1.0

        # 4. Child neighbor features (children of edge_token)
        child_has_subj = False
        child_has_obj = False
        child_neighbor_count = 0

        for child in edge_token.children:
            child_rel = child.dep_

            # Check for subject/object
            if "subj" in child_rel.lower():
                child_has_subj = True
            if "obj" in child_rel.lower():
                child_has_obj = True

            child_neighbor_count += 1

            # Add neighbor features
            features[f"{action_signature}&child_neighbor:{child_rel}"] = 1.0
            features[
                f"{action_signature}&edge_type:{edge_rel_short}&child_neighbor:{child_rel}"
            ] = 1.0

        # 4.1 Child neighbor count features
        count_str = str(child_neighbor_count) if child_neighbor_count < 3 else ">2"
        features[f"{action_signature}&child_neighbor_count:{count_str}"] = 1.0
        features[
            f"{action_signature}&edge_type:{edge_rel_short}&child_neighbor_count:{count_str}"
        ] = 1.0

        # 5. Subject/Object presence flags
        features[f"{action_signature}&parent_neighbor_subj:{parent_has_subj}"] = 1.0
        features[f"{action_signature}&parent_neighbor_obj:{parent_has_obj}"] = 1.0
        features[f"{action_signature}&child_neighbor_subj:{child_has_subj}"] = 1.0
        features[f"{action_signature}&child_neighbor_obj:{child_has_obj}"] = 1.0

        # 6. POS tag features
        parent_pos = parent_node.tag_
        child_pos = edge_token.tag_
        pos_signature = f"{parent_pos}_{child_pos}"

        features[f"{action_signature}&parent_pos:{parent_pos}"] = 1.0
        features[f"{action_signature}&child_pos:{child_pos}"] = 1.0
        features[f"{action_signature}&pos_signature:{pos_signature}"] = 1.0
        features[f"{action_signature}&edge_type:{edge_rel_short}&pos_signature:{pos_signature}"] = (
            1.0
        )

        return features

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

    @staticmethod
    def _get_root_token(doc: Doc) -> Token | None:
        """
        Get the root token of the dependency parse.

        Args:
            doc: Spacy Doc

        Returns:
            Root token, or None if not found
        """
        for token in doc:
            if token.dep_ == "ROOT":
                return token
        return None
