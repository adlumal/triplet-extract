# Reproducible Benchmarks

This directory contains clean, reproducible scripts to verify the performance metrics claimed in the main README.md.

## Benchmark Hardware (Reference)

Results in the main README were obtained on:
- **GPU tests:** NVIDIA RTX 5090 (32GB VRAM), CUDA 12.x
- **CPU tests:** AMD Ryzen 7 9800X3D (8-Core, 16 threads), 48GB RAM
- **Dataset:** 100 scientific abstracts (LaTeX-free)

Your results may vary based on your hardware.

## Quick Start

**Run all benchmarks:**
```bash
cd benchmark
./run_all_benchmarks.sh
```

**Run only CPU benchmarks:**
```bash
./run_all_benchmarks.sh cpu
```

**Run only GPU benchmark:**
```bash
./run_all_benchmarks.sh gpu
```

## Individual Benchmarks

### CPU Modes Benchmark

Benchmarks all CPU-optimized DFS modes:
- Baseline (DFS)
- Balanced (default)
- Fast
- Ultra

```bash
python scripts/benchmark_cpu_modes.py
```

**Expected results:**
- Baseline: ~1.96 sent/s, ~7.93 triplets/sentence
- Balanced: ~13.60 sent/s, ~8.55 triplets/sentence
- Fast: ~17.22 sent/s, ~6.57 triplets/sentence
- Ultra: ~28.22 sent/s, ~5.21 triplets/sentence

### GPU Mode Benchmark

Benchmarks Deep Search with GPU acceleration.

**Requirements:**
```bash
pip install triplet-extract[deepsearch]
```

**Run:**
```bash
python scripts/benchmark_gpu_mode.py
```

**Expected results:**
- Deep Search (GPU): ~8.86 sent/s, ~16.34 triplets/sentence

## Dataset

The test dataset (`data/latex_free_sentences.txt`) contains 100 scientific abstracts with LaTeX notation removed. This is the same dataset used for the README benchmarks.

**Source:** SciFact-Open corpus ([Wadden et al., 2022](https://arxiv.org/abs/2210.13777))

```bibtex
@misc{wadden2022scifactopenopendomainscientificclaim,
      title={SciFact-Open: Towards open-domain scientific claim verification},
      author={David Wadden and Kyle Lo and Bailey Kuehl and Arman Cohan and Iz Beltagy and Lucy Lu Wang and Hannaneh Hajishirzi},
      year={2022},
      eprint={2210.13777},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2210.13777},
}
```

## Variance

Hardware differences will cause performance variations. Results within ±10% of the published metrics are normal and expected. The key metrics to validate are:

1. **Relative performance:** Deep Search should find ~2.1x more triplets than Balanced mode
2. **Coverage hierarchy:** Deep Search (100%) > Balanced (~52%) > Fast (~40%) > Ultra (~32%)
3. **Throughput hierarchy:** Ultra > Fast > Balanced > Baseline > Deep Search (per-sentence)

## Troubleshooting

**GPU not detected:**
```
ℹ Deep Search mode enabled (CPU)
  For GPU acceleration: pip install triplet-extract[deepsearch]
```
Solution: Install GPU dependencies and verify CUDA is available.

**Import errors:**
```
ModuleNotFoundError: No module named 'triplet_extract'
```
Solution: Install the package from repository root:
```bash
cd ..
pip install -e ".[deepsearch]"
```

## Directory Structure

```
benchmark/
├── README.md                          # This file
├── run_all_benchmarks.sh              # Master script
├── data/
│   └── latex_free_sentences.txt       # Test dataset (100 sentences)
└── scripts/
    ├── benchmark_cpu_modes.py         # CPU modes benchmark
    └── benchmark_gpu_mode.py          # GPU mode benchmark
```

## Development Benchmarks

If you're looking for experimental benchmark scripts and development work, see the `benchmark-dev/` directory (not tracked in git).
