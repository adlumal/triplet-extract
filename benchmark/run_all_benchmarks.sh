#!/bin/bash
#
# Run all benchmarks to reproduce README.md performance metrics
#
# Usage: ./run_all_benchmarks.sh [cpu|gpu|all]
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-all}"

echo "================================================================================"
echo "BENCHMARK SUITE - Reproducing README.md Metrics"
echo "================================================================================"
echo ""
echo "Mode: $MODE"
echo ""

case "$MODE" in
    cpu)
        echo "Running CPU modes benchmark only..."
        python scripts/benchmark_cpu_modes.py
        ;;
    gpu)
        echo "Running GPU mode benchmark only..."
        python scripts/benchmark_gpu_mode.py
        ;;
    all)
        echo "Running all benchmarks..."
        echo ""
        echo "==> CPU Modes"
        python scripts/benchmark_cpu_modes.py
        echo ""
        echo "==> GPU Mode"
        python scripts/benchmark_gpu_mode.py
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        echo "Usage: $0 [cpu|gpu|all]"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "✅ ALL BENCHMARKS COMPLETE"
echo "================================================================================"
echo ""
echo "Results should match the README.md performance table within ±10% variance"
echo "(Hardware differences may cause minor variations)"
