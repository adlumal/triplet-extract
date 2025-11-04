#!/usr/bin/env python3
"""
Basic usage examples for triplet-extract.

Demonstrates the main functionality of the triplet-extract package.
"""

from triplet_extract import OpenIEExtractor, extract

print("triplet-extract - Basic Usage Examples")

# Example 1: Quick one-liner
print("\n1. Quick Extraction")
text = "Cats love milk and mice."
triplets = extract(text)

print(f"Input: {text}")
print(f"Output: {len(triplets)} triplets")
for t in triplets:
    print(f"  • {t.subject} | {t.relation} | {t.object}")

# Example 2: Scientific text
print("\n2. Scientific Text")
text = "Marine natural products have been a rich source of drug discovery."
triplets = extract(text)

print(f"Input: {text}")
print(f"Output: {len(triplets)} triplets")
for i, t in enumerate(triplets[:5], 1):  # Show first 5
    print(f"  {i}. {t.subject} -> {t.relation} -> {t.object}")
if len(triplets) > 5:
    print(f"  ... and {len(triplets)-5} more")

# Example 3: Custom extractor with options
print("\n3. Custom Extractor")
extractor = OpenIEExtractor(enable_clause_split=True, enable_entailment=True, min_confidence=0.5)

text = "The emergence of antibiotic-resistant strains has turned attention towards discovery of alternative strategies."
triplets = extractor.extract_triplet_objects(text)

print(f"Input: {text}")
print(f"Output: {len(triplets)} triplets (confidence >= 0.5)")
for t in triplets[:3]:
    print(f"  • {t.subject}")
    print(f"    {t.relation}")
    print(f"    {t.object}")
    print(f"    (confidence: {t.confidence:.2f})")
    print()

# Example 4: Batch processing
print("\n4. Batch Processing")
texts = [
    "Bacteria develop resistance to antibiotics.",
    "Biofilm formation protects bacterial communities.",
    "Novel compounds show antimicrobial activity.",
]

results = extractor.extract_batch(texts, progress=False)

for text, triplets in zip(texts, results, strict=True):
    print(f"\n{text}")
    print(f"  -> {len(triplets)} triplets")
    if triplets:
        t = triplets[0]
        print(f"     Example: ({t.subject}, {t.relation}, {t.object})")

print("\n" + "=" * 60)
print("✓ Examples complete!")
print("=" * 60)
