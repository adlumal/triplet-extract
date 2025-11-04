"""GPU-accelerated batch reparsing for forward entailment search.

Provides a drop-in replacement for sequential reparsing that batches
requests to leverage GPU parallelism.
"""

from spacy.tokens import Doc


class GPUBatchReparser:
    """Batch-aware reparser that accumulates requests and processes them on GPU."""

    def __init__(self, nlp, batch_size: int = 100, cache_size: int = 10000):
        """
        Initialize GPU batch reparser.

        Args:
            nlp: spaCy Language model (should have GPU enabled)
            batch_size: Number of texts to accumulate before batching
            cache_size: Size of LRU cache for reparsed docs
        """
        self.nlp = nlp
        self.batch_size = batch_size

        # Pending requests
        self.pending_texts: list[str] = []
        self.pending_indices: list[int] = []
        self.request_counter = 0

        # Cache for parsed docs

        self.cache = {}
        self.cache_size = cache_size

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_count = 0

    def reparse(self, text: str) -> Doc:
        """
        Reparse text, batching with other requests if possible.

        This is a drop-in replacement for nlp(text) that batches requests.
        """
        # Check cache first
        if text in self.cache:
            self.cache_hits += 1
            return self.cache[text]

        self.cache_misses += 1

        # Add to pending batch
        request_id = self.request_counter
        self.request_counter += 1
        self.pending_texts.append(text)
        self.pending_indices.append(request_id)

        # If batch is full, process it
        if len(self.pending_texts) >= self.batch_size:
            return self._flush_batch(request_id)
        else:
            # Process immediately (simpler for now - can optimize later)
            return self._flush_batch(request_id)

    def _flush_batch(self, request_id: int) -> Doc:
        """Process all pending requests as a batch and return the requested doc."""
        if not self.pending_texts:
            raise ValueError("No pending texts to flush")

        # Find position of requested doc in batch
        try:
            position = self.pending_indices.index(request_id)
        except ValueError:
            raise ValueError(f"Request ID {request_id} not found in pending batch") from None

        # Batch process on GPU
        self.batch_count += 1
        docs = list(self.nlp.pipe(self.pending_texts, batch_size=len(self.pending_texts)))

        # Update cache
        for text, doc in zip(self.pending_texts, docs, strict=False):
            if len(self.cache) < self.cache_size:
                self.cache[text] = doc

        # Get requested doc
        requested_doc = docs[position]

        # Clear pending
        self.pending_texts.clear()
        self.pending_indices.clear()

        return requested_doc

    def get_stats(self) -> dict:
        """Get statistics about caching and batching."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
            "batch_count": self.batch_count,
            "pending": len(self.pending_texts),
        }


class AsyncGPUBatchReparser:
    """
    Advanced batch reparser that collects multiple requests before flushing.

    This version actually batches requests instead of flushing immediately.
    Useful for BFS-style search where we can collect all deletions at a level.
    """

    def __init__(self, nlp, batch_size: int = 100):
        """
        Initialize async batch reparser.

        Args:
            nlp: spaCy Language model (should have GPU enabled)
            batch_size: Number of texts to accumulate before auto-flushing
        """
        self.nlp = nlp
        self.batch_size = batch_size

        # Pending requests
        self.pending: list[tuple[str, int]] = []  # (text, request_id)
        self.request_counter = 0
        self.results: dict[int, Doc] = {}  # request_id -> Doc

        # Cache
        self.cache: dict[str, Doc] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def submit(self, text: str) -> int:
        """
        Submit a reparse request and get a request ID.

        The actual parsing may happen later in a batch.

        Returns:
            request_id: Use this to retrieve the result with get()
        """
        # Check cache
        if text in self.cache:
            request_id = self.request_counter
            self.request_counter += 1
            self.results[request_id] = self.cache[text]
            self.cache_hits += 1
            return request_id

        # Add to pending
        request_id = self.request_counter
        self.request_counter += 1
        self.pending.append((text, request_id))
        self.cache_misses += 1

        # Auto-flush if batch is full
        if len(self.pending) >= self.batch_size:
            self.flush()

        return request_id

    def get(self, request_id: int) -> Doc:
        """
        Get the result of a reparse request.

        If not yet processed, flushes the batch first.
        """
        if request_id in self.results:
            return self.results[request_id]

        # Not ready - flush batch
        self.flush()

        if request_id not in self.results:
            raise ValueError(f"Request ID {request_id} not found after flush")

        return self.results[request_id]

    def flush(self):
        """Process all pending requests as a batch."""
        if not self.pending:
            return

        # Extract texts
        texts = [text for text, _ in self.pending]
        [req_id for _, req_id in self.pending]

        # Batch process on GPU
        docs = list(self.nlp.pipe(texts, batch_size=len(texts)))

        # Store results
        for (text, req_id), doc in zip(self.pending, docs, strict=False):
            self.results[req_id] = doc
            self.cache[text] = doc

        # Clear pending
        self.pending.clear()

    def get_stats(self) -> dict:
        """Get statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
            "pending": len(self.pending),
            "completed": len(self.results),
        }
