#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_PATH="${1:-/usr/Project/data/dog1_1024_683.jpg}"
OUT_DIR="${2:-./perf_summary_report}"

source "$SCRIPT_DIR/set_ascend_env.sh"

/usr/bin/python3 "$SCRIPT_DIR/export_perf_total_report.py" \
  --image "$IMAGE_PATH" \
  --image-size 224 \
  --warmup 2 \
  --iters 10 \
  --topk 5 \
  --image-sizes 128,224,320 \
  --micro-sizes 4096,16384,65536,262144,1048576,4194304 \
  --micro-warmup 20 \
  --micro-iters 100 \
  --profiler-output ./torch_npu_profiler_output \
  --run-profiler \
  --profiler-warmup 1 \
  --profiler-active 2 \
  --out-dir "$OUT_DIR"

echo ""
echo "Generated report files:"
echo "  $OUT_DIR/performance_total_summary.csv"
echo "  $OUT_DIR/performance_total_summary.html"
