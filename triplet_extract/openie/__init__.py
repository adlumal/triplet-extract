"""
Stanford OpenIE Python Port

Full 3-stage OpenIE pipeline:
- Stage 1: ClauseSplitter (sentence -> entailed clauses)
- Stage 2: ForwardEntailer (clauses -> shortened via natural logic)
- Stage 3: RelationTripleSegmenter (clauses -> relation triples)
"""

from .affinity_models import AffinityModels

__all__ = ["AffinityModels"]
