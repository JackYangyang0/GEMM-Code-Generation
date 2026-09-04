#!/usr/bin/env bash
set -euo pipefail

M="${1:-512}"
K="${2:-512}"
N="${3:-512}"
ARCH="${ARCH:-sm_89}"
BUILD_DIR="${BUILD_DIR:-build}"
OUT="${OUT:-sgpo_gemm}"

mkdir -p "$BUILD_DIR"

sources=()
if [[ -f main.cu ]]; then
  sources+=(main.cu)
elif [[ -f main.cpp ]]; then
  sources+=(main.cpp)
else
  echo "missing main.cu or main.cpp" >&2
  exit 1
fi

for src in *.cu; do
  [[ -e "$src" ]] || continue
  [[ "$src" == "main.cu" ]] && continue
  sources+=("$src")
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found in PATH" >&2
  exit 1
fi

REQUESTED_ARCH="$ARCH"
SUPPORTED_ARCHES="$(nvcc --list-gpu-arch 2>/dev/null || true)"
if [[ -n "$SUPPORTED_ARCHES" ]] && ! grep -qw "$ARCH" <<< "$SUPPORTED_ARCHES"; then
  if [[ "$ARCH" == "sm_89" ]] && grep -qw "sm_87" <<< "$SUPPORTED_ARCHES"; then
    ARCH="sm_87"
  else
    ARCH="$(awk -v requested="${ARCH#sm_}" '
      /^sm_[0-9]+$/ {
        value = substr($0, 4) + 0;
        if (requested == "" || value <= requested) {
          best = value;
        }
      }
      END {
        if (best != "") {
          printf("sm_%d", best);
        }
      }
    ' <<< "$SUPPORTED_ARCHES")"
  fi
fi

if [[ -z "$ARCH" ]]; then
  echo "could not resolve a supported CUDA arch from nvcc --list-gpu-arch" >&2
  exit 1
fi

echo "Compiling SGPO CUDA GEMM"
echo "  requested_arch: $REQUESTED_ARCH"
echo "  resolved_arch: $ARCH"
echo "  sources: ${sources[*]}"
echo "  output: $BUILD_DIR/$OUT"

nvcc -O3 -std=c++17 -arch="$ARCH" -x cu "${sources[@]}" -lcublas -o "$BUILD_DIR/$OUT"

echo "Running SGPO CUDA GEMM"
echo "  problem: M=$M K=$K N=$N"
"./$BUILD_DIR/$OUT" "$M" "$K" "$N"
