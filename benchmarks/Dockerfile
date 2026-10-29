# =============================================================================
# Dockerfile — te-comp memory & accuracy benchmarks artifact
#
# Self-contained environment to reproduce Paper Table 1, Table 2,
# and numerical accuracy verification.
# Uses the pre-built te-comp binary shipped in tools/.
#
# Build context = memory_accuracy_benchmarks/ (self-contained).
#
# Usage:
#   ./docker-build.sh            # build image
#   ./docker-run.sh              # run evaluation
# =============================================================================

ARG TARGETPLATFORM=linux/amd64
FROM --platform=$TARGETPLATFORM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- System packages ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        # ARM cross-compiler (cross-compile + arm-none-eabi-size)
        gcc-arm-none-eabi libnewlib-arm-none-eabi \
        # Python runtime
        python3 python3-pip python3-numpy python3-pil \
        # Host GCC + libc headers (for verify_accuracy.py: gcc -shared engine.c)
        gcc libc6-dev \
        # Utilities
        wget curl ca-certificates \
        # The pre-built te-comp binary was built against LLVM with
        # BUILD_SHARED_LIBS=OFF (default), so LLVM/MLIR are linked in.
        # Only standard system libs are needed at runtime.
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# --- Python packages for TFLite/SafeTensors conversion -----------------------
# Ubuntu 22.04 ships pip 22.0 + Python 3.10: no --break-system-packages needed
# (that flag requires pip >= 23.0.1 and is only needed for PEP 668 / Python 3.11+).
RUN pip3 install --no-cache-dir \
        "numpy<2" tflite==2.10.0 safetensors flatbuffers tflite-runtime

# --- Copy self-contained benchmark directory into /bench/ --------------------
# Build context is memory_accuracy_benchmarks/ (set by docker-build.sh)
COPY . /bench/

# Make the binary executable
RUN chmod +x /bench/tools/te-comp

# Verify the binary runs (the pre-built binary is x86-64, requires glibc >= 2.34;
# Ubuntu 22.04 provides glibc 2.35, and --platform=linux/amd64 ensures x86-64).
RUN /bench/tools/te-comp --help > /dev/null

WORKDIR /bench
CMD ["./run_evaluation.sh"]
