"""
Benchmark GPU-accelerated Deep Search mode.

Reproduces the Deep Search GPU metrics from README.md performance table.

Requirements:
- NVIDIA GPU with CUDA support
- triplet-extract[deepsearch] installed
- pip install triplet-extract[deepsearch]
"""

import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from triplet_extract import OpenIEExtractor

print("=" * 80)
print("GPU MODE BENCHMARK - Reproducing README Metrics")
print("=" * 80)
print()

# Load test dataset
dataset_path = Path(__file__).parent.parent / "data" / "latex_free_sentences.txt"
print(f"Loading dataset from: {dataset_path}")
with open(dataset_path) as f:
    sentences = [line.strip() for line in f if line.strip()][:100]
print(f"✓ Loaded {len(sentences)} sentences")
print()

print("=" * 80)
print("Benchmarking: Deep Search (GPU)")
print("=" * 80)
print()

print("Creating extractor with deep_search=True...")
extractor = OpenIEExtractor(
    deep_search=True,  # Auto-enables fast=True and speed_preset="fast"
)
print()

print("Running benchmark on 100 sentences...")
start = time.time()
results = extractor.extract_batch(sentences, progress=True)
elapsed = time.time() - start

total_triplets = sum(len(r) for r in results)
throughput = len(sentences) / elapsed
avg_per_sent = total_triplets / len(sentences)

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()
print(f"Time:                {elapsed:.2f}s")
print(f"Throughput:          {throughput:.2f} sent/s")
print(f"Total triplets:      {total_triplets}")
print(f"Avg per sentence:    {avg_per_sent:.2f}")
print()

print("=" * 80)
print("GPU MODE BENCHMARK COMPLETE")
print("=" * 80)
print()
print("Compare these results to the README performance table:")
print(f"  Expected: ~8.86 sent/s, ~16.34 triplets/sentence")
print(f"  Actual:   {throughput:.2f} sent/s, {avg_per_sent:.2f} triplets/sentence")
