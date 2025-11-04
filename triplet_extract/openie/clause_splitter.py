"""
Clause Splitter - Stage 1 of Stanford OpenIE

Implements beam search over clause splitting decisions using a pre-trained
linear classifier.

Ported from ClauseSplitterSearchProblem.java:500-926
"""

import heapq
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from spacy.tokens import Doc

from .clause_featurizer import ClauseFeaturizer
from .clause_search_state import (
    ClauseSearchState,
    get_action_space,
    get_hard_split_actions,
    should_skip_indirect_speech,
)
from .linear_classifier import LinearClassifier
from .sentence_fragment import SentenceFragment


@dataclass(order=True)
class PriorityItem:
    """
    Priority queue item for beam search.

    Items are ordered by priority (higher is better).
    Python's heapq is a min-heap, so we negate priorities.
    """

    priority: float
    state: ClauseSearchState = field(compare=False)
    features_so_far: list[Counter] = field(compare=False)


class ClauseSplitter:
    """
    Stage 1 of Stanford OpenIE: Split sentences into entailed clauses.

    Uses beam search with a pre-trained linear classifier to explore
    the space of clause splitting decisions.

    Ports ClauseSplitterSearchProblem.java
    """

    def __init__(
        self,
        classifier: LinearClassifier | None = None,
        featurizer: ClauseFeaturizer | None = None,
        max_ticks: int = 1000,
    ):
        """
        Initialize the clause splitter.

        Args:
            classifier: Pre-trained linear classifier (loads default if None)
            featurizer: Clause featurizer (creates default if None)
            max_ticks: Maximum search iterations
        """
        self.classifier = classifier if classifier else LinearClassifier()
        self.featurizer = featurizer if featurizer else ClauseFeaturizer()
        self.max_ticks = max_ticks
        self.action_space = get_action_space()

    def split(
        self, doc: Doc, threshold_probability: float = 0.5, max_clauses: int = 100
    ) -> list[SentenceFragment]:
        """
        Split a sentence into entailed clauses.

        Args:
            doc: Spacy Doc containing the sentence
            threshold_probability: Minimum probability for a clause (0-1)
            max_clauses: Maximum number of clauses to return

        Returns:
            List of SentenceFragment objects representing clauses
        """
        results = []

        def candidate_callback(
            log_prob: float,
            features: list[Counter],
            fragment_generator: Callable[[], SentenceFragment],
        ) -> bool:
            """
            Callback for search results.

            Args:
                log_prob: Log probability of this fragment
                features: Feature counters along the path
                fragment_generator: Lazy generator for the fragment

            Returns:
                True to continue searching, False to stop
            """
            # Convert log probability to probability
            prob = math.exp(log_prob) if log_prob > -100 else 0.0

            if prob >= threshold_probability:
                fragment = fragment_generator()
                fragment.score = prob
                results.append(fragment)

                # Stop if we have enough clauses
                if len(results) >= max_clauses:
                    return False

            return True

        # Run beam search
        self._search(doc, candidate_callback)

        # Sort by score descending
        results.sort(key=lambda f: f.score, reverse=True)

        return results[:max_clauses]

    def _search(
        self,
        doc: Doc,
        candidate_callback: Callable[[float, list[Counter], Callable[[], SentenceFragment]], bool],
    ):
        """
        Beam search over clause splitting decisions.

        Ports ClauseSplitterSearchProblem.search() from lines 784-926

        Args:
            doc: Spacy Doc containing the sentence
            candidate_callback: Callback for complete clauses
        """
        # Find the root token
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break

        if not root:
            return  # No root found

        # Priority queue: (negated_log_prob, state, features_so_far)
        # Python heapq is a min-heap, so we negate priorities for max-heap behavior
        fringe: list[PriorityItem] = []

        # Avoid duplicate work
        seen_tokens: set[int] = set()

        # Initial state: at root, done
        initial_state = ClauseSearchState(
            edge=None,
            subject_or_null=None,
            distance_from_subj=-9000,
            object_or_null=None,
            tree_operations=[],
            is_done=True,
        )

        heapq.heappush(
            fringe,
            PriorityItem(
                priority=0.0,  # Negated log prob (so 0.0 = log prob -0.0)
                state=initial_state,
                features_so_far=[],
            ),
        )

        ticks = 0

        while fringe:
            if ticks >= self.max_ticks:
                break

            ticks += 1

            # Pop highest priority state
            item = heapq.heappop(fringe)
            log_prob_so_far = -item.priority  # Un-negate to get actual log prob
            last_state = item.state
            features_so_far = item.features_so_far

            # Get the current root token for this state
            root_token = root if last_state.edge is None else last_state.edge

            # If this state is done, yield the fragment
            if last_state.is_done:
                # Create fragment generator
                def make_fragment(captured_score=log_prob_so_far) -> SentenceFragment:
                    # For now, create a simple fragment
                    # In full implementation, this would apply tree_operations
                    return SentenceFragment(doc, score=math.exp(captured_score))

                # Call the callback
                should_continue = candidate_callback(
                    log_prob_so_far, features_so_far, make_fragment
                )
                if not should_continue:
                    break

            # Find subject and object edges from current root
            subj_or_null = None
            obj_or_null = None

            for child in root_token.children:
                rel = child.dep_
                if "subj" in rel:
                    subj_or_null = child
                if "obj" in rel:
                    obj_or_null = child

            # Iterate over outgoing edges
            for outgoing_edge in root_token.children:
                # Skip indirect speech verbs
                if should_skip_indirect_speech(outgoing_edge):
                    continue

                # Get hard-coded split actions if applicable
                forced_actions = get_hard_split_actions(outgoing_edge.dep_)
                done_forced_arc = False

                # Determine action order
                if forced_actions:
                    # Order actions according to hard splits
                    ordered_actions = []
                    for action_sig in forced_actions:
                        for action in self.action_space:
                            if action.signature() == action_sig:
                                ordered_actions.append(action)
                    # Add remaining actions
                    for action in self.action_space:
                        if action not in ordered_actions:
                            ordered_actions.append(action)
                    action_list = ordered_actions
                else:
                    action_list = self.action_space

                # Try each action
                for action in action_list:
                    # Check prerequisites
                    if not action.prerequisites_met(doc, outgoing_edge):
                        continue

                    # If we already did a forced arc, stop
                    if forced_actions and done_forced_arc:
                        break

                    # Apply action to get candidate state
                    candidate = action.apply_to(
                        doc, last_state, outgoing_edge, subj_or_null, obj_or_null
                    )

                    if candidate is None:
                        continue

                    # Featurize the transition
                    features = self.featurizer.featurize(
                        parent_token=last_state.edge,
                        edge_token=outgoing_edge,
                        action_signature=action.signature(),
                        doc=doc,
                    )

                    # Classify the edge
                    if forced_actions and not done_forced_arc:
                        # Hard-coded split: assign probability 1.0
                        log_probability = 0.0
                        best_label = "CLAUSE_SPLIT"
                        done_forced_arc = True
                    else:
                        # Use classifier
                        probs = self.classifier.predict_probabilities(features)

                        # Special case: always yield on nsubj and obj
                        rel = outgoing_edge.dep_
                        if rel == "nsubj" or rel == "obj":
                            if "NOT_A_CLAUSE" in probs:
                                del probs["NOT_A_CLAUSE"]

                        # Get best label and probability
                        if probs:
                            best_label = max(probs, key=probs.get)
                            log_probability = (
                                math.log(probs[best_label]) if probs[best_label] > 0 else -100.0
                            )
                        else:
                            best_label = "NOT_A_CLAUSE"
                            log_probability = float("-inf")

                    # Only add if not NOT_A_CLAUSE
                    if best_label != "NOT_A_CLAUSE":
                        # Update candidate state with isDone flag
                        candidate.with_is_done(best_label)

                        # Create new features list
                        new_features = features_so_far.copy()
                        new_features.append(features)

                        # Avoid duplicates
                        if outgoing_edge.i not in seen_tokens:
                            # Add to fringe
                            new_log_prob = log_prob_so_far + log_probability
                            heapq.heappush(
                                fringe,
                                PriorityItem(
                                    priority=-new_log_prob,  # Negate for max-heap
                                    state=candidate,
                                    features_so_far=new_features,
                                ),
                            )

            # Mark current token as seen
            seen_tokens.add(root_token.i)

    def split_to_strings(self, doc: Doc, **kwargs) -> list[str]:
        """
        Split and return clause texts as strings.

        Args:
            doc: Spacy Doc
            **kwargs: Arguments to pass to split()

        Returns:
            List of clause texts
        """
        fragments = self.split(doc, **kwargs)
        return [str(fragment) for fragment in fragments]
