# triplet-extract

GPU-accelerated Python implementation of Stanford OpenIE with comprehensive triplet extraction

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Example

```python
from triplet_extract import extract

text = "95.6% of people don't know what GraphRAG is for"
triplets = extract(text)

for t in triplets:
    print(f"({t.subject}, {t.relation}, {t.object})")
```

Output:
```
(95.6% of people, don't know, what GraphRAG is for)
```

**Features:**
- Comprehensive extraction using breadth-first search 
- Natural formatting with proper contraction spacing
- Quantifiers preserved and normalized (percentages, scientific units)
- LaTeX math preserved for scientific literature
- Optional GPU acceleration for batch processing

## About

This is a GPU-accelerated Python port of Stanford OpenIE that extends the original natural-logic pipeline with breadth-first search for comprehensive triplet extraction. The implementation follows the same three-stage pipeline and uses the trained models from the Stanford NLP Group's research.

### Technical Approach

To our knowledge, this is the first open-source system that GPU-accelerates the natural-logic forward-entailment search itself — via batched reparsing over dependency parses — rather than replacing the natural-logic OpenIE pipeline with a neural model trained on its outputs.

Prior neural OpenIE models typically train on triplets produced by classical OpenIE systems, using GPUs for neural inference over those labels. In contrast, this system keeps the original natural-logic semantics and uses the GPU to accelerate the BFS exploration through batch processing, effectively GPU-accelerating the underlying OpenIE algorithm rather than approximating it with a neural model.

This port uses spaCy for dependency parsing instead of Stanford CoreNLP, providing a pure Python alternative that works without Java dependencies. I'm grateful to the Stanford NLP Group for their groundbreaking research and for making their models available.

**Note:** This implementation supports English text only. The trained models and natural logic rules are language-specific.

## Design Philosophy

This implementation prioritizes preserving rich semantic context in extracted triplets. Unlike some ports that simplify subjects and relations, this port retains qualifiers, quantifiers, and contextual information (e.g., "The U.S. president Barack Obama" rather than just "Barack Obama", or "25% of people" rather than just "people"). This makes the output particularly well-suited for knowledge graph construction, GraphRAG applications, and other systems that benefit from semantically rich representations.

## Installation

**Recommended: GPU-accelerated (more comprehensive extraction):**

```bash
pip install triplet-extract[deepsearch]
python -m spacy download en_core_web_sm
```

**Requires:** CUDA-capable GPU, CUDA 12.x, 8GB+ VRAM recommended
**Benefit:** ~1.9x more triplets with GPU-accelerated BFS (vs default Balanced mode)

**Base install (CPU-optimized):**

```bash
pip install triplet-extract
python -m spacy download en_core_web_sm
```

**Works on:** Any machine, serverless, edge devices
**Performance:** Fast CPU-optimized DFS (13.60/s)

**Local development with `uv`:**

```bash
git clone https://github.com/adlumal/triplet-extract.git
cd triplet-extract
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[deepsearch]"
uv pip install -e ".[dev]"
uv run spacy download en_core_web_sm
```

## Usage

### Basic Extraction

```python
from triplet_extract import extract

text = "Cats love milk and mice."
triplets = extract(text)

for t in triplets:
    print(f"({t.subject}, {t.relation}, {t.object})")
```

### Using the Extractor Class

The `OpenIEExtractor` class provides more control over the extraction pipeline:

```python
from triplet_extract import OpenIEExtractor

extractor = OpenIEExtractor(
    enable_clause_split=True,    # Split complex sentences into clauses
    enable_entailment=True,      # Generate entailed shorter forms
    min_confidence=0.5           # Filter low-confidence triplets
)

triplets = extractor.extract_triplet_objects(text)

for t in triplets:
    print(f"Subject: {t.subject}")
    print(f"Relation: {t.relation}")
    print(f"Object: {t.object}")
    print(f"Confidence: {t.confidence}")
    print()
```

### Extraction Modes

The extractor uses Balanced mode by default, which is CPU-optimized for production use:

```python
# Default: Balanced mode (CPU-optimized, 13.60/s)
extractor = OpenIEExtractor()

# Enable Deep Search for comprehensive extraction (GPU recommended)
# Automatically enables fast=True and speed_preset="fast"
extractor = OpenIEExtractor(deep_search=True)

# CPU speed presets (automatically enable fast=True)
extractor = OpenIEExtractor(speed_preset="fast")    # 17.22/s, fewer triplets
extractor = OpenIEExtractor(speed_preset="ultra")   # 28.22/s, minimal triplets
```

**Performance comparison** (100 scientific abstracts):

| Mode | Triplets/Sent | Total (100s) | Throughput | Time (100s) | Coverage† | Precision | Use Case |
|------|---------------|--------------|------------|-------------|-----------|-----------|----------|
| **Deep Search** | **16.34** | **1634** | **8.86/s** | **11.29s** | **100%** | **98%+** | **Comprehensive extraction (GPU-accelerated)** |
| Baseline (DFS) | 7.93 | 793 | 1.96/s | 51.09s | 48.5% | 100% | Reference quality |
| **Balanced (default)** | **8.55** | **855** | **13.60/s** | **7.35s** | **52.3%** | **98.2%** | **Default:** CPU-optimized production |
| Fast | 6.57 | 657 | 17.22/s | 5.81s | 40.2% | 98.4% | High-throughput APIs |
| Ultra | 5.21 | 521 | 28.22/s | 3.54s | 31.9% | 99.5% | Maximum speed |
| Stanford OpenIE | 13.45 | 1345 | 15.42/s | 6.48s | 82.3% | ~95% | Original Java |

**†Coverage:** Percentage of Deep Search triplets found (Deep Search finds the most = 100% baseline)

**Note:** Stanford OpenIE benchmarks executed via [stanford-openie-python](https://github.com/philipperemy/stanford-openie-python) package. Numbers vary slightly between runs.

**Benchmark Hardware:**
- **GPU tests:** NVIDIA RTX 5090 (32GB VRAM), CUDA 12.x
- **CPU tests:** AMD Ryzen 7 9800X3D (8-Core, 16 threads), 48GB RAM
- **Dataset:** 100 scientific abstracts (LaTeX-free)

### Deployment Notes

#### GPU-Accelerated
*Workstations, GPU servers, batch processing pipelines*

BFS mode with CUDA acceleration - **~1.9x more triplets vs default Balanced mode:**

| Mode | Configuration | Hardware | Use Case |
|------|--------------|----------|----------|
| **Deep Search (GPU)** | `deep_search=True` | GPU (CUDA) | Comprehensive extraction, knowledge graphs |
| Deep Search (CPU fallback) | `deep_search=True` | CPU only | Same quality, slower throughput |

*Estimated based on BFS algorithm complexity. Actual performance varies by CPU.

**GPU Requirements:** CUDA 12.x, 8GB+ VRAM recommended

**Example:**
```python
from triplet_extract import OpenIEExtractor

# GPU-accelerated comprehensive extraction
# Auto-detects optimal batch size based on VRAM (adds ~1s initialization)
extractor = OpenIEExtractor(deep_search=True)

# Process multiple texts efficiently with batching
texts = ["Sentence 1", "Sentence 2", "Sentence 3", ...]
results = extractor.extract_batch(texts)  # Returns list of triplet lists

# Disable auto-detection for faster initialization (use fixed batch size)
extractor = OpenIEExtractor(deep_search=True, gpu_batch_size=128)
```

#### CPU-Optimized (Default)
*AWS Lambda, Cloud Run, serverless functions, edge devices*

All DFS modes - optimized for CPU with LRU caching:

| Mode | Configuration | Use Case |
|------|--------------|----------|
| **Balanced (recommended)** | `deep_search=False, speed_preset="balanced"` | Production default |
| Fast | `deep_search=False, speed_preset="fast"` | High-throughput APIs |
| Ultra | `deep_search=False, speed_preset="ultra"` | Maximum speed priority |
| Baseline | `high_quality=True, fast=False, deep_search=False` | Reference/compatibility |

**Example:**
```python
from triplet_extract import OpenIEExtractor

# CPU-optimized for serverless deployment (default)
extractor = OpenIEExtractor()  # Uses balanced preset

# Process multiple texts efficiently
texts = ["Sentence 1", "Sentence 2", "Sentence 3", ...]
results = extractor.extract_batch(texts)  # 3-5x faster than individual calls

# Or adjust speed/quality tradeoff
extractor = OpenIEExtractor(speed_preset="fast")    # Higher throughput
extractor = OpenIEExtractor(speed_preset="ultra")   # Maximum speed
```

### Pipeline Options

The extractor implements three stages:

**Stage 1: Clause Splitting** (`enable_clause_split`)
Breaks complex sentences into simpler clauses using beam search. For example, "Obama, born in Hawaii, is president" becomes ["Obama is president", "Obama born in Hawaii"].

**Stage 2: Forward Entailment** (`enable_entailment`)
Generates shorter entailed forms using natural logic. For example, "Blue cats play" produces ["Blue cats play", "cats play"]. This applies to all fragments, including those from clause splitting.

**Confidence Threshold** (`min_confidence`)
Filters triplets below the specified confidence score (0.0 to 1.0). Higher values give fewer but higher-quality results.

```python
# Fast extraction without variations
extractor = OpenIEExtractor(
    enable_clause_split=False,
    enable_entailment=False
)

# High-precision extraction
extractor = OpenIEExtractor(
    min_confidence=0.7
)
```

### Batch Processing

For processing multiple texts efficiently:

```python
texts = [
    "First sentence to process.",
    "Second sentence to process.",
    "Third sentence to process."
]

# GPU-accelerated if available, CPU fallback otherwise
results = extractor.extract_batch(texts, progress=True)

for text, triplets in zip(texts, results):
    print(f"\n{text}")
    print(f"  {len(triplets)} triplets extracted")
```

The system automatically uses GPU acceleration if `triplet-extract[deepsearch]` is installed and a CUDA GPU is available. Otherwise, it falls back to CPU with identical extraction quality.

### Performance Tips

Reuse extractor instances when processing multiple texts:

```python
# Good: Reuse the same extractor
extractor = OpenIEExtractor(min_confidence=0.5)
for text in texts:
    triplets = extractor.extract_triplet_objects(text)

# Avoid: Creates new extractor (reloads models) each time
for text in texts:
    triplets = extract(text, min_confidence=0.5)
```

Use batch processing for best performance:

```python
results = extractor.extract_batch(texts, batch_size=32)
```

### Verbose Logging

The library is silent by default. Enable logging to see internal operations:

```python
import logging

logging.basicConfig(level=logging.DEBUG)  # Show all details
# or
logging.basicConfig(level=logging.INFO)   # Show major steps

from triplet_extract import extract
triplets = extract("Your text here")
```

## How It Works

The system implements the three-stage pipeline from the Stanford OpenIE paper:

**Stage 1: Clause Splitting**
Uses a pre-trained linear classifier to break complex sentences into independent clauses. The classifier was trained on the LSOIE dataset and considers dependency parse structure to make splitting decisions.

**Stage 2: Forward Entailment**
Applies natural logic deletion rules to generate shorter entailed forms. Uses prepositional phrase attachment affinities to determine which constituents can be safely deleted while preserving truth.

**Stage 3: Pattern Matching**
Extracts (subject, relation, object) triplets from sentence fragments using dependency patterns. Handles various syntactic constructions including copular sentences, prepositional phrases, and clausal complements.

The trained models (clause splitting classifier and PP attachment affinities) are from the original Stanford implementation and are included in this package.

## Implementation Notes

This implementation uses spaCy for dependency parsing instead of Stanford CoreNLP. While the algorithm and models are the same, the parsers may produce different dependency trees for the same sentence. Differences in tokenization, POS tagging, and dependency labels mean that extraction results won't be identical to the original Java implementation.

In practice, core extractions remain highly compatible with Stanford OpenIE, though edge cases may differ, particularly with unusual capitalization or complex grammatical constructions. If you require exact compatibility with Stanford OpenIE output, please use the original Java implementation.

## Limitations

spaCy's statistical POS tagger can commit to a compound-noun reading of an entire clause when every token in sequence admits a noun-compatible analysis. `extract("Dogs chase cats.")` returns no triplets because the parse is `Dogs/ADJ chase/NOUN cats/NOUN` — no verb anywhere, so no extraction pattern has a predicate to anchor on. The trigger is a noun/verb-ambiguous word in verb position with bare nominals on *both* sides; a bare-plural subject alone is not the decisive factor. Anything that breaks the compound reading anywhere in the clause flips the tagger to the clausal parse and extraction succeeds: a determiner on the **object** (`"Dogs chase the cats."`), an adverb beside the verb (`"Dogs often chase cats."`, `"Cheetahs run faster than dogs."`), a pronoun subject (`"They chase cats."`), unambiguous verb morphology (`"Dogs chased cats."`), or an unambiguous verb (`"Birds eat seeds."`). Conversely, a determiner on the subject alone does **not** reliably fix it (`"The dogs chase cats."` fails on `en_core_web_sm`), and neither does 3sg inflection when the object stays bare (`"The dog chases cats."` fails — "chases" re-reads as a plural noun, like "dog races"); `"The dog chases the cat."` works because of the object's determiner. Larger models shrink the failure surface without eliminating it: `en_core_web_md` resolves the determiner-bearing variants but still misparses fully bare `"Dogs chase cats."` and `"Dogs hunt mice."` (`en_core_web_trf` untested). Stanford CoreNLP's tagger resolves these cases correctly. This rarely impacts real-world usage — formal writing scatters determiners, adverbs, and inflection through clauses — but aphorism-style generic SVO ("X chase Y") is the risk zone, and such sentences fail *silently*, yielding zero triplets.

## Citation

If you use this library in research, please cite both this implementation and the original Stanford OpenIE paper:

**This implementation:**
```bibtex
@software{malec2025tripletextract,
  title={triplet-extract: GPU-accelerated Python implementation of Stanford OpenIE},
  author={Malec, Adrian Lucas},
  year={2025},
  url={https://github.com/adlumal/triplet-extract}
}
```

**Original Stanford OpenIE paper:**
```bibtex
@inproceedings{angeli2015openie,
  title={Leveraging Linguistic Structure For Open Domain Information Extraction},
  author={Angeli, Gabor and Johnson Premkumar, Melvin Jose and Manning, Christopher D},
  booktitle={Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics (ACL 2015)},
  year={2015}
}
```

**Reference:** Angeli, Gabor, Melvin Jose Johnson Premkumar, and Christopher D. Manning. "Leveraging Linguistic Structure For Open Domain Information Extraction." *Association for Computational Linguistics (ACL), 2015.* [Paper](http://nlp.stanford.edu/pubs/2015angeli-openie.pdf) | [Stanford OpenIE](https://stanfordnlp.github.io/CoreNLP/openie.html) | [CoreNLP Github](https://github.com/stanfordnlp/CoreNLP)

## Contributing

Bug reports and feature requests are welcome. Please open an issue on GitHub if you encounter problems or have suggestions for improvements.

## License

GPL-3.0-or-later

This is a derivative work of Stanford OpenIE, which is licensed under GPL-3.0. The trained models included in this package are from the original Stanford implementation and remain under their GPL-3.0 license.

See [LICENSE](LICENSE) for details.

## Links

- [Stanford OpenIE](https://stanfordnlp.github.io/CoreNLP/openie.html)
- [Original Paper](http://nlp.stanford.edu/pubs/2015angeli-openie.pdf)
- [spaCy](https://spacy.io/)

## Related packages

- [stanford-openie-python](https://github.com/philipperemy/stanford-openie-python)
