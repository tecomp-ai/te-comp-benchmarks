#!/usr/bin/env bash
#
# docker-run.sh — Run the te-comp TinyStories_v2 code generation inside Docker.
#
# Usage:
#   ./docker-run.sh                          # full evaluation
#   ./docker-run.sh --skip-download          # skip model download
#   ./docker-run.sh --shell                  # drop into bash shell
#   ./docker-run.sh --output ./results       # copy results to host
#

set -euo pipefail

IMAGE_TAG="te-comp-tinystories-v2:latest"
CONTAINER_NAME="te-comp-tinystories-v2-$$"
OUTPUT_DIR=""
SHELL_MODE=false
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --image)    IMAGE_TAG="$2"; shift 2 ;;
        --image=*)  IMAGE_TAG="${1#*=}"; shift ;;
        --output)   OUTPUT_DIR="$2"; shift 2 ;;
        --output=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --shell)    SHELL_MODE=true; shift ;;
        --help|-h)
            sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
            exit 0
            ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if ! docker image inspect "$IMAGE_TAG" &>/dev/null; then
    echo "ERROR: Docker image '$IMAGE_TAG' not found." >&2
    echo "       Run ./docker-build.sh first." >&2
    exit 1
fi

if $SHELL_MODE; then
    echo "Dropping into shell in $IMAGE_TAG ..."
    docker run --rm -it --name "$CONTAINER_NAME" "$IMAGE_TAG" /bin/bash
    exit 0
fi

if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Running code generation (results will be copied to $OUTPUT_DIR) ..."
    # Allow non-zero exit (accuracy verification returns 1 for INT8 rounding diffs)
    EVAL_RC=0
    docker run --name "$CONTAINER_NAME" "$IMAGE_TAG" \
        ./generate_engine.sh ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} || EVAL_RC=$?
    docker cp "$CONTAINER_NAME:/bench/output" \
        "$OUTPUT_DIR/output" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" > /dev/null
    echo
else
    docker run --rm -it "$IMAGE_TAG" ./generate_engine.sh ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi
