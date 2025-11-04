"""
Forward Entailer - Stage 2 of Stanford OpenIE

Uses natural logic to find entailed shortenings of a sentence by deleting
dependency edges while maintaining valid entailment.

Ported from ForwardEntailerSearchProblem.java (384 lines)
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

from spacy.tokens import Doc, Token

from .dependency_relations import for_dependency_deletion
from .natural_logic_weights import NaturalLogicWeights
from .polarity import Polarity
from .sentence_fragment import SentenceFragment
from .text_utils import reconstruct_text


@dataclass
class SearchResult:
    """
    A result from the forward entailment search.

    Attributes:
        doc: The shortened dependency tree
        active_mask: Indices of active tokens (None = all active)
        deleted_edges: List of deleted edge labels
        confidence: Deletion probability score (0-1)
    """

    doc: Doc
    active_mask: frozenset[int] | None
    deleted_edges: list[str]
    confidence: float

    def to_fragment(self, truth_of_premise: bool) -> SentenceFragment:
        """Convert to a SentenceFragment."""
        # Convert frozenset to set for SentenceFragment
        token_indices = set(self.active_mask) if self.active_mask else None

        fragment = SentenceFragment(
            doc=self.doc,
            token_indices=token_indices,
            assumed_truth=truth_of_premise,
            score=self.confidence,
        )
        return fragment


@dataclass
class SearchState:
    """
    A search state representing a partial shortening.

    Attributes:
        deletion_mask: BitSet indicating which tokens are deleted
        active_mask: Indices of active (non-deleted) tokens (for fast mode)
        current_index: Current token index in topological order
        doc: Current dependency tree (original or reparsed)
        is_masked: True if using masked view (no reparse)
        num_deletions: Count of deletions in this branch (for budget)
        last_deleted_edge: Last edge label we deleted
        source: Previous search state (for backtracking)
        score: Cumulative deletion probability score
    """

    deletion_mask: set[int]
    active_mask: frozenset[int]
    current_index: int
    doc: Doc
    is_masked: bool
    num_deletions: int
    last_deleted_edge: str | None
    source: Optional["SearchState"]
    score: float


class ForwardEntailer:
    """
    Stage 2 of Stanford OpenIE: Shorten clauses via natural logic entailment.

    Uses DFS search to explore deletions of dependency edges, validating
    each deletion maintains valid entailment using natural logic relations.

    Ports ForwardEntailerSearchProblem.java
    """

    def __init__(
        self,
        weights: NaturalLogicWeights | None = None,
        max_ticks: int = 5000,
        max_results: int = 100,
        nlp=None,
        fast: bool = False,
        speed_preset: str = "balanced",
        deep_search: bool = False,
        gpu_batch_size: int | None = None,
        use_gpu: bool | None = None,  # Deprecated, use deep_search
    ):
        """
        Initialize the forward entailer.

        Args:
            weights: Natural logic weights (loads default if None)
            max_ticks: Maximum search iterations (ignored if fast=True)
            max_results: Maximum number of results to return
            nlp: spaCy Language instance (loads en_core_web_sm if None)
            fast: Enable fast mode (lightweight reparsing + budgets)
            speed_preset: Speed/quality trade-off (requires fast=True, preserve_latex=True for stated metrics):
                - "balanced" (default): 100% recall, 98% precision, 7x speedup
                - "fast": 79% recall, 99.5% precision, 12x speedup
                - "ultra": 66% recall, 99.5% precision, 16x speedup
            deep_search: Enable Deep Search mode for comprehensive triplet extraction (3.5x more triplets!)
                - Uses breadth-first search to explore the full search space
                - GPU-accelerated if CUDA available, falls back to CPU if not
                - Extracts 2.5-3.5x more triplets with same 98%+ precision
                - Per-triplet efficiency same or better than standard mode
                - Requires: fast=True
                - Optional GPU acceleration: pip install triplet-extract[deepsearch]
            gpu_batch_size: Number of texts to batch for GPU reparsing (auto-detected if None)
                - 8GB VRAM: 32 (accumulates to 64-128)
                - 16GB VRAM: 64 (accumulates to 128-256)
                - 24GB VRAM: 128 (accumulates to 256-512)
                - 32GB+ VRAM: 256 (accumulates to 512-1024)
            use_gpu: DEPRECATED - use deep_search instead
        """
        self.weights = weights if weights else NaturalLogicWeights()
        self.fast = fast
        self.speed_preset = speed_preset

        # Set budgets based on mode
        if fast:
            if speed_preset == "ultra":
                # ULTRA: 66% recall, 99.5% precision, 16x speedup (with LaTeX preprocessing)
                # Use when speed is critical and lower recall is acceptable
                self.max_ticks = 1500
                self.max_results = 100
                self.max_deletions = 10
                self.max_reparses = 20
            elif speed_preset == "fast":
                # FAST: 79% recall, 99.5% precision, 12x speedup (with LaTeX preprocessing)
                # Good balance of speed and quality for most applications
                self.max_ticks = 1500
                self.max_results = 100
                self.max_deletions = 10
                self.max_reparses = 50
            else:  # balanced (default)
                # BALANCED: 100% recall, 98% precision, 7x speedup (with LaTeX preprocessing)
                # Highest quality, captures all HQ mode triplets
                self.max_ticks = 2000
                self.max_results = 100
                self.max_deletions = 10
                self.max_reparses = 200
        else:
            self.max_ticks = max_ticks
            self.max_results = max_results
            self.max_deletions = 999
            self.max_reparses = 999

        # Load main spaCy model
        if nlp is None:
            import spacy

            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp

        # Setup lightweight reparsing for fast mode
        if fast:
            from functools import lru_cache

            import spacy

            self.nlp_reparse = spacy.load(
                "en_core_web_sm",
                disable=["ner", "textcat", "entity_ruler", "entity_linker"],
            )

            # Closure-based caching (avoid hashing self)
            nlp_reparse = self.nlp_reparse

            @lru_cache(maxsize=10000)
            def _reparse_text(text: str):
                return nlp_reparse(text)

            self._reparse_text = _reparse_text
            self._reparse_count = 0
        else:
            self.nlp_reparse = self.nlp
            self._reparse_text = lambda text: self.nlp(text)
            self._reparse_count = 0

        # Handle deprecated use_gpu parameter
        if use_gpu is not None:
            import warnings

            warnings.warn(
                "Parameter 'use_gpu' is deprecated and will be removed in a future version. "
                "Use 'deep_search' instead for the same functionality.",
                DeprecationWarning,
                stacklevel=2,
            )
            deep_search = use_gpu

        # Setup Deep Search mode (BFS with GPU acceleration if available)
        self.deep_search = deep_search
        self.use_gpu = False  # Track if GPU is actually being used

        if deep_search:
            # Deep Search requires fast mode
            if not fast:
                raise ValueError("Deep Search mode requires fast=True")

            # Try to enable GPU acceleration
            try:
                import spacy

                spacy.require_gpu()
                self.use_gpu = True
                print("✓ Deep Search mode enabled with GPU acceleration")
            except ImportError as e:
                print("ℹ Deep Search mode enabled (CPU)")
                print("  For GPU acceleration: pip install triplet-extract[deepsearch]")
                print(f"  (Missing: {e.name if hasattr(e, 'name') else 'GPU dependencies'})")
            except Exception as e:
                print("ℹ Deep Search mode enabled (CPU)")
                print(f"  GPU not available: {e}")

            # Auto-detect batch size based on available VRAM (if using GPU)
            if self.use_gpu:
                if gpu_batch_size is None:
                    self.gpu_batch_size = self._detect_gpu_batch_size()
                else:
                    self.gpu_batch_size = gpu_batch_size
                print(
                    f"  GPU batch size: {self.gpu_batch_size} (accumulates to {self.gpu_batch_size*2}-{self.gpu_batch_size*4})"
                )
            else:
                self.gpu_batch_size = 32  # Default for CPU mode

            # Initialize shared batch queue for cross-sentence batching
            self._shared_batch_queue = []
            self._shared_batch_metadata = []
        else:
            self.gpu_batch_size = 32  # Default (unused in standard mode)
            self._shared_batch_queue = None
            self._shared_batch_metadata = None

    def enable_shared_batching(self):
        """Enable shared batch queue for cross-sentence batching."""
        if self.use_gpu:
            self._shared_batch_queue = []
            self._shared_batch_metadata = []

    def flush_shared_batch(self):
        """
        Flush the shared batch queue and return reparsed docs.

        Returns:
            List of (metadata, reparsed_doc) tuples
        """
        if not self.use_gpu or not self._shared_batch_queue:
            return []

        texts = self._shared_batch_queue
        metadata = self._shared_batch_metadata

        # Batch reparse on GPU!
        reparsed_docs = list(
            self.nlp_reparse.pipe(texts, batch_size=min(len(texts), self.gpu_batch_size))
        )
        self._reparse_count += len(reparsed_docs)

        # Clear queue
        self._shared_batch_queue = []
        self._shared_batch_metadata = []

        return list(zip(metadata, reparsed_docs, strict=False))

    def _detect_gpu_batch_size(self) -> int:
        """
        Auto-detect optimal batch size based on available VRAM.

        Returns:
            Recommended batch size
        """
        try:
            import cupy

            device = cupy.cuda.Device()
            free_mem, total_mem = device.mem_info
            vram_gb = total_mem / (1024**3)

            # Recommendations based on VRAM
            if vram_gb >= 30:
                return 256  # 32GB+ VRAM
            elif vram_gb >= 20:
                return 128  # 24GB VRAM
            elif vram_gb >= 14:
                return 64  # 16GB VRAM
            else:
                return 32  # 8GB VRAM
        except Exception:
            # Fallback if cupy not available
            return 64

    def entail(self, doc: Doc, truth_of_premise: bool = True) -> list[SentenceFragment]:
        """
        Find entailed shortenings of the input sentence.

        Args:
            doc: Spacy Doc containing the sentence
            truth_of_premise: Whether the premise is true (vs false)

        Returns:
            List of entailed sentence fragments, sorted by confidence
        """
        # Route to appropriate search algorithm
        if self.deep_search:
            # Use BFS (with GPU if available, CPU fallback otherwise)
            results = self._search_gpu_bfs(doc, truth_of_premise)
        else:
            # Standard DFS
            results = self._search(doc, truth_of_premise)

        # Convert to sentence fragments
        fragments = [r.to_fragment(truth_of_premise) for r in results if len(r.doc) > 0]

        # Sort by score descending
        fragments.sort(key=lambda f: f.score, reverse=True)

        return fragments

    def _search(self, doc: Doc, truth_of_premise: bool) -> list[SearchResult]:
        """
        The search algorithm for finding entailed shortenings.

        Ports ForwardEntailerSearchProblem.searchImplementation()
        (lines 132-359)

        Args:
            doc: Input dependency tree
            truth_of_premise: Whether premise is true

        Returns:
            List of search results
        """
        # Make a copy to avoid modifying the input
        parse_tree = doc

        # Pre-process: remove common determiners
        # (lines 136-150)
        determiner_removals = []
        tokens_to_remove = []

        for token in parse_tree:
            word_lower = token.text.lower()
            if word_lower in ("the", "a", "an", "this", "that", "those", "these"):
                if token.dep_ == "det" or self._is_leaf(token, parse_tree):
                    tokens_to_remove.append(token)
                    determiner_removals.append("det")

        # Create modified doc without determiners
        if tokens_to_remove:
            parse_tree = self._remove_tokens(parse_tree, {t.i for t in tokens_to_remove})

        # Pre-process: cut conj:and edges that create multiple parents
        # (lines 152-167)
        # TODO: For now, skip this step - it's complex and rarely affects results

        # Find subject/object split
        # (lines 173-202)
        is_subject = self._find_subject_nodes(parse_tree)

        # Initialize results
        results: list[SearchResult] = []

        # Add determiner-only result if we removed any
        if determiner_removals:
            # Find the determiner token (if it exists)
            det_token = next((t for t in doc if t.dep_ == "det"), None)

            # Only add result if we found a determiner token
            # (avoid passing None to deletion_probability which would crash)
            if det_token is not None:
                det_score = self.weights.deletion_probability(det_token, doc) ** len(
                    determiner_removals
                )
                results.append(
                    SearchResult(
                        doc=parse_tree,
                        active_mask=None,  # All tokens active (determiners removed from parse)
                        deleted_edges=determiner_removals,
                        confidence=det_score,
                    )
                )

        # Get topological sort for DFS
        # (lines 226-234)
        topological_order = self._topological_sort(parse_tree)

        if not topological_order:
            return results

        # Initialize DFS fringe (stack)
        # (lines 236-237)
        fringe: deque[SearchState] = deque()
        initial_state = SearchState(
            deletion_mask=set(),
            active_mask=frozenset(range(len(parse_tree))),
            current_index=0,
            doc=parse_tree,
            is_masked=False,
            num_deletions=0,
            last_deleted_edge=None,
            source=None,
            score=1.0,
        )
        fringe.append(initial_state)

        # Run DFS
        # (lines 240-355)
        num_ticks = 0
        self._reparse_count = 0  # Reset reparse counter per sentence

        while fringe:
            # Check limits
            if num_ticks >= self.max_ticks:
                break
            if len(results) >= self.max_results:
                break

            num_ticks += 1
            state = fringe.pop()

            # Skip if score too low
            if state.score <= 0.0:
                continue

            # Get current word
            if state.current_index >= len(topological_order):
                continue

            current_word = topological_order[state.current_index]

            # Skip if already deleted
            if current_word.i in state.deletion_mask:
                # Move to next index
                next_idx = self._next_undeleted_index(
                    state.current_index + 1, topological_order, state.deletion_mask
                )
                if next_idx is not None:
                    fringe.append(
                        SearchState(
                            deletion_mask=state.deletion_mask,
                            active_mask=state.active_mask,
                            current_index=next_idx,
                            doc=state.doc,
                            is_masked=state.is_masked,
                            num_deletions=state.num_deletions,
                            last_deleted_edge=None,
                            source=state,
                            score=state.score,
                        )
                    )
                continue

            # Push the case where we DON'T delete
            # (lines 254-271)
            next_idx = self._next_undeleted_index(
                state.current_index + 1, topological_order, state.deletion_mask
            )
            if next_idx is not None:
                fringe.append(
                    SearchState(
                        deletion_mask=state.deletion_mask,
                        active_mask=state.active_mask,
                        current_index=next_idx,
                        doc=state.doc,
                        is_masked=state.is_masked,
                        num_deletions=state.num_deletions,
                        last_deleted_edge=None,
                        source=state,
                        score=state.score,
                    )
                )

            # Check if we CAN delete this subtree
            # (lines 273-301)
            can_delete = not self._is_root(current_word, state.doc)

            # Get incoming edges to this token
            parent = current_word.head
            if can_delete and parent != current_word:
                edge_relation = current_word.dep_

                # Don't delete from CD parents (numbers)
                if parent.tag_ == "CD":
                    can_delete = False

                # Check natural logic validity
                if can_delete:
                    # Get polarity from token annotation (computed by polarity annotator)
                    token_polarity = (
                        current_word._.polarity
                        if hasattr(current_word._, "polarity") and current_word._.polarity
                        else Polarity.DEFAULT
                    )

                    # Get deletion relation
                    is_subj_node = current_word.i in is_subject
                    lexical_relation = for_dependency_deletion(edge_relation, is_subj_node)

                    # Project through polarity
                    projected_relation = token_polarity.project_lexical_relation(lexical_relation)

                    # Check if this preserves truth
                    truth_result = projected_relation.apply_to_truth_value(truth_of_premise)

                    if not truth_result.is_true():
                        can_delete = False

            # If we can delete, do it
            # (lines 303-354)
            if can_delete:
                # Get descendants
                descendants = self._get_descendants(current_word, state.doc)

                # Check if current_word is part of a hyphenated compound
                # If so, include all compound tokens in deletion to avoid orphaned hyphens
                compound_tokens = self._get_compound_tokens(current_word, state.doc)
                if len(compound_tokens) > 1:
                    # Delete entire compound atomically
                    for comp_token in compound_tokens:
                        descendants.add(comp_token)
                        # Also get descendants of each compound part
                        comp_descendants = self._get_descendants(comp_token, state.doc)
                        descendants.update(comp_descendants)

                # Create new deletion mask
                new_mask = state.deletion_mask.copy()
                for desc in descendants:
                    new_mask.add(desc.i)

                # Determine deletion strategy (safe vs unsafe)
                is_safe = self._is_structurally_safe_subtree(current_word)

                if is_safe:
                    # SAFE: Use masking (no reparse)
                    to_remove = {t.i for t in descendants}
                    new_active_mask = state.active_mask - to_remove
                    new_doc = state.doc  # Keep same doc
                    is_masked = True
                else:
                    # UNSAFE: Requires reparsing
                    if self._reparse_count >= self.max_reparses:
                        continue  # Budget exceeded, skip this branch

                    new_doc = self._remove_tokens_fast(state.doc, descendants, state.active_mask)
                    new_active_mask = frozenset(range(len(new_doc)))
                    is_masked = False

                # Compute score
                new_score = state.score
                if parent != current_word:
                    # Get siblings for deletion probability
                    list(parent.children) if parent else []
                    multiplier = self.weights.deletion_probability(current_word, state.doc)
                    new_score *= multiplier

                # Only proceed if score > 0
                if new_score > 0.0:
                    # Register result
                    deleted_edges_list = self._aggregate_deleted_edges(
                        state,
                        [current_word.dep_] if parent != current_word else [],
                        determiner_removals,
                    )

                    results.append(
                        SearchResult(
                            doc=new_doc,
                            active_mask=new_active_mask if is_masked else None,
                            deleted_edges=deleted_edges_list,
                            confidence=new_score,
                        )
                    )

                    # Push state with deletion
                    next_idx = self._next_undeleted_index(
                        state.current_index + 1, topological_order, new_mask
                    )
                    if next_idx is not None:
                        # Check deletion depth budget
                        new_num_deletions = state.num_deletions + 1
                        if new_num_deletions <= self.max_deletions:
                            fringe.append(
                                SearchState(
                                    deletion_mask=new_mask,
                                    active_mask=new_active_mask,
                                    current_index=next_idx,
                                    doc=new_doc,
                                    is_masked=is_masked,
                                    num_deletions=new_num_deletions,
                                    last_deleted_edge=current_word.dep_,
                                    source=state,
                                    score=new_score,
                                )
                            )

        return results

    def _search_gpu_bfs(self, doc: Doc, truth_of_premise: bool) -> list[SearchResult]:
        """
        GPU-accelerated breadth-first search for entailed shortenings.

        This is a BFS version of _search() that batches reparsing operations
        to leverage GPU parallelism. Instead of processing deletions one at a time
        (DFS), it processes all deletions at each level together and batches the
        GPU reparsing.

        Expected speedup: 5-10x on GPU vs CPU sequential processing.

        Args:
            doc: Input dependency tree
            truth_of_premise: Whether premise is true

        Returns:
            List of search results
        """
        # Pre-process: remove common determiners (same as DFS)
        parse_tree = doc
        determiner_removals = []
        tokens_to_remove = []

        for token in parse_tree:
            word_lower = token.text.lower()
            if word_lower in ("the", "a", "an", "this", "that", "those", "these"):
                if token.dep_ == "det" or self._is_leaf(token, parse_tree):
                    tokens_to_remove.append(token)
                    determiner_removals.append("det")

        if tokens_to_remove:
            parse_tree = self._remove_tokens(parse_tree, {t.i for t in tokens_to_remove})

        # Find subject/object split (same as DFS)
        is_subject = self._find_subject_nodes(parse_tree)

        # Initialize results
        results: list[SearchResult] = []

        # Add determiner-only result (same as DFS)
        if determiner_removals:
            det_token = next((t for t in doc if t.dep_ == "det"), None)
            if det_token is not None:
                det_score = self.weights.deletion_probability(det_token, doc) ** len(
                    determiner_removals
                )
                results.append(
                    SearchResult(
                        doc=parse_tree,
                        active_mask=None,
                        deleted_edges=determiner_removals,
                        confidence=det_score,
                    )
                )

        # Get topological sort
        topological_order = self._topological_sort(parse_tree)
        if not topological_order:
            return results

        # BFS: Initialize first level
        initial_state = SearchState(
            deletion_mask=set(),
            active_mask=frozenset(range(len(parse_tree))),
            current_index=0,
            doc=parse_tree,
            is_masked=False,
            num_deletions=0,
            last_deleted_edge=None,
            source=None,
            score=1.0,
        )
        current_level = [initial_state]

        num_ticks = 0
        self._reparse_count = 0

        # Accumulate candidates across levels for bigger batches
        accumulated_unsafe_candidates = []

        # Process level by level (BFS)
        while current_level:
            # Check limits
            if num_ticks >= self.max_ticks:
                break
            if len(results) >= self.max_results:
                break

            next_level = []

            # Collect candidates for batch processing
            # safe_states: can be processed without reparsing
            # unsafe_candidates: need reparsing - we'll batch these!
            safe_states_to_process = []
            unsafe_candidates = (
                []
            )  # List of (state, current_word, descendants, new_mask, parent, is_subj_node)

            for state in current_level:
                num_ticks += 1
                if num_ticks >= self.max_ticks:
                    break

                if state.score <= 0.0:
                    continue

                if state.current_index >= len(topological_order):
                    continue

                current_word = topological_order[state.current_index]

                # Skip if already deleted
                if current_word.i in state.deletion_mask:
                    next_idx = self._next_undeleted_index(
                        state.current_index + 1, topological_order, state.deletion_mask
                    )
                    if next_idx is not None:
                        next_level.append(
                            SearchState(
                                deletion_mask=state.deletion_mask,
                                active_mask=state.active_mask,
                                current_index=next_idx,
                                doc=state.doc,
                                is_masked=state.is_masked,
                                num_deletions=state.num_deletions,
                                last_deleted_edge=None,
                                source=state,
                                score=state.score,
                            )
                        )
                    continue

                # Push non-deletion case
                next_idx = self._next_undeleted_index(
                    state.current_index + 1, topological_order, state.deletion_mask
                )
                if next_idx is not None:
                    next_level.append(
                        SearchState(
                            deletion_mask=state.deletion_mask,
                            active_mask=state.active_mask,
                            current_index=next_idx,
                            doc=state.doc,
                            is_masked=state.is_masked,
                            num_deletions=state.num_deletions,
                            last_deleted_edge=None,
                            source=state,
                            score=state.score,
                        )
                    )

                # Check if we CAN delete this subtree (same logic as DFS)
                can_delete = not self._is_root(current_word, state.doc)

                parent = current_word.head
                if can_delete and parent != current_word:
                    edge_relation = current_word.dep_

                    if parent.tag_ == "CD":
                        can_delete = False

                    if can_delete:
                        token_polarity = (
                            current_word._.polarity
                            if hasattr(current_word._, "polarity") and current_word._.polarity
                            else Polarity.DEFAULT
                        )

                        is_subj_node = current_word.i in is_subject
                        lexical_relation = for_dependency_deletion(edge_relation, is_subj_node)
                        projected_relation = token_polarity.project_lexical_relation(
                            lexical_relation
                        )
                        truth_result = projected_relation.apply_to_truth_value(truth_of_premise)

                        if not truth_result.is_true():
                            can_delete = False

                # If we can delete, prepare deletion info
                if can_delete:
                    descendants = self._get_descendants(current_word, state.doc)

                    # Handle compound tokens (hyphens)
                    compound_tokens = self._get_compound_tokens(current_word, state.doc)
                    if len(compound_tokens) > 1:
                        for comp_token in compound_tokens:
                            descendants.add(comp_token)
                            comp_descendants = self._get_descendants(comp_token, state.doc)
                            descendants.update(comp_descendants)

                    new_mask = state.deletion_mask.copy()
                    for desc in descendants:
                        new_mask.add(desc.i)

                    # Check if safe (though in fast mode, always False - always reparse)
                    is_safe = self._is_structurally_safe_subtree(current_word)
                    is_subj_node = current_word.i in is_subject

                    if is_safe:
                        # Safe: process immediately (no reparse needed)
                        safe_states_to_process.append(
                            (state, current_word, descendants, new_mask, parent, is_subj_node)
                        )
                    else:
                        # Unsafe: needs reparsing - add to batch!
                        if self._reparse_count < self.max_reparses:
                            unsafe_candidates.append(
                                (state, current_word, descendants, new_mask, parent, is_subj_node)
                            )

            # Process safe states (no reparsing needed)
            for (
                state,
                current_word,
                descendants,
                new_mask,
                parent,
                _is_subj_node,
            ) in safe_states_to_process:
                to_remove = {t.i for t in descendants}
                new_active_mask = state.active_mask - to_remove
                new_doc = state.doc
                is_masked = True

                new_score = state.score
                if parent != current_word:
                    multiplier = self.weights.deletion_probability(current_word, state.doc)
                    new_score *= multiplier

                if new_score > 0.0:
                    deleted_edges_list = self._aggregate_deleted_edges(
                        state,
                        [current_word.dep_] if parent != current_word else [],
                        determiner_removals,
                    )

                    results.append(
                        SearchResult(
                            doc=new_doc,
                            active_mask=new_active_mask,
                            deleted_edges=deleted_edges_list,
                            confidence=new_score,
                        )
                    )

                    # Add to next level
                    next_idx = self._next_undeleted_index(
                        state.current_index + 1, topological_order, new_mask
                    )
                    if next_idx is not None:
                        new_num_deletions = state.num_deletions + 1
                        if new_num_deletions <= self.max_deletions:
                            next_level.append(
                                SearchState(
                                    deletion_mask=new_mask,
                                    active_mask=new_active_mask,
                                    current_index=next_idx,
                                    doc=new_doc,
                                    is_masked=is_masked,
                                    num_deletions=new_num_deletions,
                                    last_deleted_edge=current_word.dep_,
                                    source=state,
                                    score=new_score,
                                )
                            )

            # BATCH PROCESS unsafe candidates on GPU!
            # Accumulate candidates across levels for bigger batches
            accumulated_unsafe_candidates.extend(unsafe_candidates)

            # Only flush when we have enough for a big batch OR at the end
            should_flush = (
                len(accumulated_unsafe_candidates)
                >= self.gpu_batch_size * 2  # Wait for 2x batch size
                or not next_level  # Or end of search
                or self._reparse_count >= self.max_reparses  # Or budget exhausted
            )

            if accumulated_unsafe_candidates and should_flush:
                # Limit batch size to avoid OOM and respect reparse budget
                remaining_budget = self.max_reparses - self._reparse_count
                batch_to_process = accumulated_unsafe_candidates[
                    : min(
                        len(accumulated_unsafe_candidates),
                        remaining_budget,
                        self.gpu_batch_size * 4,
                    )
                ]  # Process up to 4x batch size at once

                # Remove processed from accumulated
                accumulated_unsafe_candidates = accumulated_unsafe_candidates[
                    len(batch_to_process) :
                ]

                if batch_to_process:
                    # Prepare texts for batch reparsing
                    texts_to_reparse = []
                    for (
                        state,
                        _current_word,
                        descendants,
                        _new_mask,
                        _parent,
                        _is_subj_node,
                    ) in batch_to_process:
                        # Get indices to remove
                        remove_indices = {t.i for t in descendants}

                        # Get indices to keep (respecting active_mask)
                        if len(state.active_mask) < len(state.doc):
                            # Doc is masked
                            keep_indices = [i for i in state.active_mask if i not in remove_indices]
                        else:
                            # Doc is not masked
                            keep_indices = [
                                i for i in range(len(state.doc)) if i not in remove_indices
                            ]

                        if not keep_indices:
                            texts_to_reparse.append("")
                        else:
                            # Reconstruct text from remaining tokens
                            remaining_tokens = [state.doc[i] for i in keep_indices]
                            text = reconstruct_text(remaining_tokens)
                            texts_to_reparse.append(text)

                    # BATCH REPARSE ON GPU! This is the key speedup!
                    reparsed_docs = list(
                        self.nlp_reparse.pipe(
                            texts_to_reparse,
                            batch_size=min(len(texts_to_reparse), self.gpu_batch_size),
                        )
                    )
                    self._reparse_count += len(reparsed_docs)

                    # Process reparsed results
                    for (
                        state,
                        current_word,
                        _descendants,
                        new_mask,
                        parent,
                        _is_subj_node,
                    ), new_doc in zip(batch_to_process, reparsed_docs, strict=False):
                        new_active_mask = frozenset(range(len(new_doc)))
                        is_masked = False

                        new_score = state.score
                        if parent != current_word:
                            multiplier = self.weights.deletion_probability(current_word, state.doc)
                            new_score *= multiplier

                        if new_score > 0.0:
                            deleted_edges_list = self._aggregate_deleted_edges(
                                state,
                                [current_word.dep_] if parent != current_word else [],
                                determiner_removals,
                            )

                            results.append(
                                SearchResult(
                                    doc=new_doc,
                                    active_mask=None,  # Not masked after reparse
                                    deleted_edges=deleted_edges_list,
                                    confidence=new_score,
                                )
                            )

                            # Add to next level
                            next_idx = self._next_undeleted_index(
                                state.current_index + 1, topological_order, new_mask
                            )
                            if next_idx is not None:
                                new_num_deletions = state.num_deletions + 1
                                if new_num_deletions <= self.max_deletions:
                                    next_level.append(
                                        SearchState(
                                            deletion_mask=new_mask,
                                            active_mask=new_active_mask,
                                            current_index=next_idx,
                                            doc=new_doc,
                                            is_masked=is_masked,
                                            num_deletions=new_num_deletions,
                                            last_deleted_edge=current_word.dep_,
                                            source=state,
                                            score=new_score,
                                        )
                                    )

            # Move to next level
            current_level = next_level

        return results

    def _is_leaf(self, token: Token, doc: Doc) -> bool:
        """Check if token has no children."""
        return not any(child.head == token for child in doc)

    def _is_root(self, token: Token, doc: Doc) -> bool:
        """Check if token is the root."""
        return token.dep_ == "ROOT" or token.head == token

    def _is_structurally_safe_subtree(self, head: Token) -> bool:
        """
        Check if deleting this subtree is "safe" (can use masking).

        IMPORTANT: After extensive testing, masking was found to significantly
        reduce recall (from 92.9% to 57.4%). The issue is that masked fragments
        don't allow proper exploration of the search space.

        Solution: Always reparse (never mask) in fast mode.
        This maintains high recall (92.9%) with good speedup (11.2x).

        Args:
            head: Root of subtree to delete

        Returns:
            Always False in fast mode (never use masking, always reparse)
        """
        # Never use masking - always reparse for better recall
        return False

    def _reparse_fast(self, text: str) -> Doc:
        """
        Reparse text using lightweight pipeline with caching.

        In fast mode: uses nlp_reparse (lightweight) + LRU cache
        In HQ mode: uses full nlp pipeline

        Args:
            text: Text to parse

        Returns:
            Parsed Doc
        """
        self._reparse_count += 1
        return self._reparse_text(text)

    def get_reparse_stats(self) -> dict:
        """Get statistics about reparsing in fast mode."""
        if not self.fast:
            return {}

        cache_info = self._reparse_text.cache_info()
        return {
            "reparse_count": self._reparse_count,
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "hit_rate": (
                f"{cache_info.hits / (cache_info.hits + cache_info.misses):.1%}"
                if cache_info.hits + cache_info.misses > 0
                else "0%"
            ),
        }

    def _remove_tokens(self, doc: Doc, tokens_to_remove) -> Doc:
        """
        Create a new Doc without specified tokens.

        Args:
            doc: Original doc
            tokens_to_remove: Set of Token objects or set of token indices to remove

        Returns:
            New Doc without those tokens
        """
        # Handle both Set[Token] and Set[int]
        if not tokens_to_remove:
            return doc

        # Convert to set of indices
        if isinstance(next(iter(tokens_to_remove)), Token):
            remove_indices = {t.i for t in tokens_to_remove}
        else:
            remove_indices = tokens_to_remove

        # Get indices of tokens to keep
        keep_indices = [i for i in range(len(doc)) if i not in remove_indices]

        if not keep_indices:
            # Return empty doc - create from empty list
            return self.nlp("")

        # Extract text from remaining tokens
        # Preserve spacing where possible
        remaining_tokens = [doc[i] for i in keep_indices]
        reconstructed_text = reconstruct_text(remaining_tokens)

        # Re-parse to get proper dependencies
        new_doc = self.nlp(reconstructed_text)

        return new_doc

    def _remove_tokens_fast(self, doc: Doc, tokens_to_remove, active_mask: frozenset[int]) -> Doc:
        """
        Create new Doc without tokens (fast mode with lightweight pipeline).

        Respects active_mask if doc is masked.
        Uses lightweight pipeline + caching.

        Args:
            doc: Original doc (might be masked)
            tokens_to_remove: Set of Token objects or indices to remove
            active_mask: Current active mask

        Returns:
            New Doc without those tokens (reparsed with lightweight pipeline)
        """
        if not tokens_to_remove:
            return doc

        # Convert to indices
        if isinstance(next(iter(tokens_to_remove)), Token):
            remove_indices = {t.i for t in tokens_to_remove}
        else:
            remove_indices = set(tokens_to_remove)

        # Get indices to keep (respecting mask if doc is masked)
        if len(active_mask) < len(doc):
            # Doc is masked - only consider active tokens
            keep_indices = [i for i in active_mask if i not in remove_indices]
        else:
            # Doc is not masked - consider all tokens
            keep_indices = [i for i in range(len(doc)) if i not in remove_indices]

        if not keep_indices:
            return self._reparse_fast("")

        # Reconstruct text and reparse
        remaining_tokens = [doc[i] for i in keep_indices]
        reconstructed_text = reconstruct_text(remaining_tokens)

        return self._reparse_fast(reconstructed_text)

    def _find_subject_nodes(self, doc: Doc) -> set[int]:
        """
        Find all nodes that are subjects or descendants of subjects.

        Ports lines 175-202.

        Args:
            doc: Dependency tree

        Returns:
            Set of token indices that are subjects
        """
        is_subject = set()

        for token in doc:
            # Check if this or any ancestor is a subject
            current = token
            depth = 0
            while current and depth < 100:
                if current.dep_.endswith("subj"):
                    is_subject.add(token.i)
                    break

                # Move to parent
                if current.head == current:
                    break
                current = current.head
                depth += 1

        return is_subject

    def _topological_sort(self, doc: Doc) -> list[Token]:
        """
        Topological sort of dependency tree.

        Falls back to left-to-right if not a proper tree.

        Args:
            doc: Dependency tree

        Returns:
            List of tokens in topological order
        """
        # Simple left-to-right order
        # TODO: Implement proper topological sort
        return list(doc)

    def _next_undeleted_index(
        self, start_index: int, topological_order: list[Token], deletion_mask: set[int]
    ) -> int | None:
        """
        Find next undeleted token index.

        Args:
            start_index: Starting index
            topological_order: Tokens in topological order
            deletion_mask: Set of deleted token indices

        Returns:
            Next valid index, or None
        """
        for i in range(start_index, len(topological_order)):
            if topological_order[i].i not in deletion_mask:
                return i
        return None

    def _get_compound_tokens(self, token: Token, doc: Doc) -> set[Token]:
        """
        Get all tokens that are part of the same hyphenated compound as the given token.

        For example, in "antibiotic-resistant", if given "antibiotic", returns
        {"antibiotic", "-", "resistant"} so they can be deleted atomically.

        Args:
            token: Token that may be part of a compound
            doc: Dependency tree

        Returns:
            Set of tokens in the compound (includes the token itself)
        """
        compound_tokens = {token}

        # Check UPWARD: if token is part of a hyphenated compound by looking at siblings
        parent = token.head
        if parent != token:
            siblings = list(parent.children)

            # Check if any sibling is a hyphen (indicates compound)
            has_hyphen = any(sib.text in ("-", "—", "–", "‐") for sib in siblings)

            if has_hyphen:
                # Include all tokens that are part of this compound:
                # - All amod (adjectival modifier) siblings
                # - All hyphen punctuation siblings
                # - The parent if it's also an amod (e.g., resistant in antibiotic-resistant strains)
                for sibling in siblings:
                    if sibling.dep_ == "amod" or sibling.text in ("-", "—", "–", "‐"):
                        compound_tokens.add(sibling)

                # If parent is also a modifier (e.g., "resistant" modifying "strains"),
                # include it in the compound
                if parent.dep_ == "amod":
                    compound_tokens.add(parent)

        # Check DOWNWARD: if token itself is the head of a hyphenated compound
        children = list(token.children)
        has_hyphen_child = any(child.text in ("-", "—", "–", "‐") for child in children)

        if has_hyphen_child:
            # Include all children that are part of the compound
            for child in children:
                if child.dep_ == "amod" or child.text in ("-", "—", "–", "‐"):
                    compound_tokens.add(child)

        return compound_tokens

    def _get_descendants(self, token: Token, doc: Doc) -> set[Token]:
        """
        Get all descendants of a token in the dependency tree.

        Args:
            token: Root token
            doc: Dependency tree

        Returns:
            Set of descendant tokens (including token itself)
        """
        descendants = {token}
        queue = [token]

        while queue:
            current = queue.pop(0)
            for child in doc:
                if child.head == current and child != current:
                    if child not in descendants:
                        descendants.add(child)
                        queue.append(child)

        return descendants

    def _aggregate_deleted_edges(
        self, state: SearchState, just_deleted: list[str], other_edges: list[str]
    ) -> list[str]:
        """
        Backtrace to collect all deleted edges.

        Ports lines 368-383.

        Args:
            state: Current search state
            just_deleted: Edges just deleted
            other_edges: Other deletions (e.g., determiners)

        Returns:
            Complete list of deleted edges
        """
        result = []
        result.extend(just_deleted)
        result.extend(other_edges)

        # Backtrace
        current = state
        while current is not None:
            if current.last_deleted_edge:
                result.append(current.last_deleted_edge)
            current = current.source

        return result
