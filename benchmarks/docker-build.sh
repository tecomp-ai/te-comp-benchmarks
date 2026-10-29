#!/usr/bin/env bash
#
# docker-build.sh — Build the te-comp memory benchmarks Docker image.
#
# Uses the pre-built te-comp binary from tools/. No source code needed.
# Build context is memory_accuracy_benchmarks/ (self-contained).
#
# Usage:
#   ./docker-build.sh                    # default: te-comp-bench:latest
#   ./docker-build.sh --tag my-image     # custom image name
#   ./docker-build.sh --no-cache         # force full rebuild
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"

IMAGE_TAG="te-comp-bench:latest"
BUILD_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --tag)       IMAGE_TAG="$2"; shift 2 ;;
        --tag=*)     IMAGE_TAG="${1#*=}"; shift ;;
        --no-cache)  BUILD_ARGS+=(--no-cache); shift ;;
        --help|-h)
            sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Verify the pre-built binary exists
if [ ! -f "$SCRIPT_DIR/tools/te-comp" ]; then
    echo "ERROR: Pre-built binary not found at $SCRIPT_DIR/tools/te-comp" >&2
    exit 1
fi

echo "========================================================================"
echo "  Building Docker image: $IMAGE_TAG"
echo "========================================================================"
echo "  Dockerfile:    $DOCKERFILE"
echo "  Build context: $SCRIPT_DIR"
echo "========================================================================"
echo

docker build \
    -f "$DOCKERFILE" \
    -t "$IMAGE_TAG" \
    ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} \
    "$SCRIPT_DIR"

echo
echo "========================================================================"
echo "  Image built: $IMAGE_TAG"
echo "========================================================================"
echo
echo "  Run the evaluation:"
echo "    ./docker-run.sh --output ./results"
echo
echo
