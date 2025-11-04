"""
Benchmark all CPU-optimized DFS modes.

Reproduces the CPU mode metrics from README.md performance table:
- Baseline (DFS): high_quality=True, fast=False
- Balanced: fast=True, speed_preset="balanced"
- Fast: fast=True, speed_preset="fast"
- Ultra: fast=True, speed_preset="ultra"
"""

import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from triplet_extract import OpenIEExtractor

print("=" * 80)
print("CPU MODES BENCHMARK - Reproducing README Metrics")
print("=" * 80)
print()

# Load test dataset
dataset_path = Path(__file__).parent.parent / "data" / "latex_free_sentences.txt"
print(f"Loading dataset from: {dataset_path}")
with open(dataset_path) as f:
    sentences = [line.strip() for line in f if line.strip()][:100]
print(f"✓ Loaded {len(sentences)} sentences")
print()

modes = [
    {
        "name": "Baseline (DFS)",
        "config": {"high_quality": True, "fast": False, "deep_search": False, "preserve_latex": True},
    },
    {
        "name": "Balanced",
        "config": {"fast": True, "speed_preset": "balanced", "deep_search": False, "preserve_latex": True},
    },
    {
        "name": "Fast",
        "config": {"fast": True, "speed_preset": "fast", "deep_search": False, "preserve_latex": True},
    },
    {
        "name": "Ultra",
        "config": {"fast": True, "speed_preset": "ultra", "deep_search": False, "preserve_latex": True},
    },
]

results = []

for mode in modes:
    print("=" * 80)
    print(f"Benchmarking: {mode['name']}")
    print("=" * 80)
    print()

    extractor = OpenIEExtractor(**mode['config'])

    start = time.time()
    triplet_results = extractor.extract_batch(sentences, progress=True)
    elapsed = time.time() - start

    total_triplets = sum(len(r) for r in triplet_results)
    throughput = len(sentences) / elapsed
    avg_per_sent = total_triplets / len(sentences)

    results.append({
        "name": mode['name'],
        "time": elapsed,
        "throughput": throughput,
        "total_triplets": total_triplets,
        "triplets_per_sent": avg_per_sent,
    })

    print(f"\n✓ {mode['name']} completed")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} sent/s")
    print(f"  Total triplets: {total_triplets}")
    print(f"  Avg per sentence: {avg_per_sent:.2f}")
    print()

# Summary table
print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()
print(f"{'Mode':<20} {'Time':<10} {'Throughput':<15} {'Total':<10} {'Per Sent'}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<20} {r['time']:>8.2f}s {r['throughput']:>12.2f}/s {r['total_triplets']:>9} {r['triplets_per_sent']:>10.2f}")

print()
print("=" * 80)
print("CPU MODES BENCHMARK COMPLETE")
print("=" * 80)
print()
print("Compare these results to the README performance table.")
