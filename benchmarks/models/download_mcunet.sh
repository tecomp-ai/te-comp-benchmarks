#!/usr/bin/env bash
#
# download_mcunet.sh — Download MCUNet TFLite models from MIT HAN Lab.
#
# Downloads all 10 MCUNet benchmark models (int8 quantized TFLite) from:
#   https://hanlab18.mit.edu/projects/tinyml/mcunet/release/
#
# Source: https://github.com/mit-han-lab/mcunet
# Paper:  "MCUNet: Tiny Deep Learning on IoT Devices" (NeurIPS 2020)
#
# Models are the official pre-trained int8 TFLite files from the MCUNet
# model zoo. No retraining or re-quantization is performed.
#
# Usage:
#   ./benchmarks/models/download_mcunet.sh              # from repo root
#   ./benchmarks/models/download_mcunet.sh --verify      # verify checksums only
#   ./benchmarks/models/download_mcunet.sh --force        # re-download even if exists
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
URL_BASE="https://hanlab18.mit.edu/projects/tinyml/mcunet/release/"

# MCUNet model zoo: net_name -> tflite filename
# net_name is the key used in mcunet.model_zoo.NET_INFO
declare -A MODELS=(
    # ImageNet models
    ["mcunet-10fps_imagenet"]="mcunet-in0"
    ["mcunet-5fps_imagenet"]="mcunet-in1"
    ["mcunet-256kb-1mb_imagenet"]="mcunet-in2"
    ["mcunet-320kb-1mb_imagenet"]="mcunet-in3"
    ["mcunet-512kb-2mb_imagenet"]="mcunet-in4"
    ["mbv2-w0.35-r144_imagenet"]="mbv2-w0.35"
    ["proxyless-w0.3-r176_imagenet"]="proxyless-w0.3"
    # VWW models
    ["mcunet-10fps_vww"]="mcunet-vww0"
    ["mcunet-5fps_vww"]="mcunet-vww1"
    ["mcunet-320kb-1mb_vww"]="mcunet-vww2"
)

# Expected SHA256 checksums (verified against official release)
declare -A CHECKSUMS=(
    ["mcunet-10fps_imagenet"]="4791f13ac1b90b0a1a0b96177d905197a1ebd34f7bc960d92c9d3250aa5b6ef8"
    ["mcunet-5fps_imagenet"]="bf14b8de64caba8630086250b4c6ee660c222d8e48ce3e8c791d0a402e15ef73"
    ["mcunet-256kb-1mb_imagenet"]="38c5d0794fa40196cb5c8da4fc64b116a33f448f3db721458916b54b1e4a11fb"
    ["mcunet-320kb-1mb_imagenet"]="780563f544bccba413b1ee4e051159996ac8b5be39eba8bb38761a51b58a07d1"
    ["mcunet-512kb-2mb_imagenet"]="1ad36261efdc5185d5236e7f8306a9481d9044c56cbb72f9230fbc67ba13545e"
    ["mbv2-w0.35-r144_imagenet"]="00c87d2a5f2d96deb7a7e8afe000b786e5198384ab66f21c714dd3ed43d00f8c"
    ["proxyless-w0.3-r176_imagenet"]="8cfa09445a43c8eb51b94685b72a4aa1c0318865af03636f3f81bcf7042dacc7"
    ["mcunet-10fps_vww"]="4676b97e2fe775929b1538446a120173ab0e1a0f94a1d81e7322f09fee9b3d1d"
    ["mcunet-5fps_vww"]="d878916e93f59e63949ce9b8e1a585382d466d3b94d751b9f1628a7753e03c24"
    ["mcunet-320kb-1mb_vww"]="5180674a058b7c7b627a4142b4c6b194382e6df05ad8c8eb40a86aab8c1c7986"
)

# Model descriptions
declare -A DESCRIPTIONS=(
    ["mcunet-10fps_imagenet"]="MCUNet ImageNet, 10fps on STM32F746 (6.4M MACs, 0.75M params)"
    ["mcunet-5fps_imagenet"]="MCUNet ImageNet, 5fps on STM32F746 (12.8M MACs, 0.64M params)"
    ["mcunet-256kb-1mb_imagenet"]="MCUNet ImageNet, 256KB SRAM + 1MB Flash (67.3M MACs, 0.73M params)"
    ["mcunet-320kb-1mb_imagenet"]="MCUNet ImageNet, 320KB SRAM + 1MB Flash (81.8M MACs, 0.74M params)"
    ["mcunet-512kb-2mb_imagenet"]="MCUNet ImageNet, 512KB SRAM + 2MB Flash (125.9M MACs, 1.73M params)"
    ["mbv2-w0.35-r144_imagenet"]="MobileNetV2 w0.35 r144, ImageNet baseline (23.5M MACs, 0.75M params)"
    ["proxyless-w0.3-r176_imagenet"]="ProxylessNAS w0.3 r176, ImageNet baseline (38.3M MACs, 0.75M params)"
    ["mcunet-10fps_vww"]="MCUNet VWW, 10fps on STM32F746 (6.0M MACs, 0.37M params)"
    ["mcunet-5fps_vww"]="MCUNet VWW, 5fps on STM32F746 (11.6M MACs, 0.43M params)"
    ["mcunet-320kb-1mb_vww"]="MCUNet VWW, 320KB SRAM + 1MB Flash (55.8M MACs, 0.64M params)"
)

# --- Parse arguments ---
VERIFY_ONLY=false
FORCE=false
for arg in "$@"; do
    case "$arg" in
        --verify)  VERIFY_ONLY=true ;;
        --force)   FORCE=true ;;
        --help|-h)
            sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
            exit 0
            ;;
    esac
done

# --- Download / verify ---
echo "MCUNet TFLite Model Downloader"
echo "Source: https://github.com/mit-han-lab/mcunet"
echo "URL:    ${URL_BASE}"
echo "Output: ${SCRIPT_DIR}/"
echo ""

TOTAL=0
OK=0
FAIL=0

# Sort by net_id for consistent output
for net_name in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
    net_id="${MODELS[$net_name]}"
    desc="${DESCRIPTIONS[$net_name]}"
    expected_sha="${CHECKSUMS[$net_name]}"
    tflite_file="${SCRIPT_DIR}/${net_name}.tflite"
    url="${URL_BASE}${net_name}.tflite"
    TOTAL=$((TOTAL + 1))

    if $VERIFY_ONLY; then
        if [ -f "$tflite_file" ]; then
            actual_sha=$(sha256sum "$tflite_file" | cut -d' ' -f1)
            if [ "$actual_sha" = "$expected_sha" ]; then
                printf "  [%2d] %-35s  OK (sha256 match)\n" "$TOTAL" "$net_name"
                OK=$((OK + 1))
            else
                printf "  [%2d] %-35s  FAIL (sha256 mismatch!)\n" "$TOTAL" "$net_name"
                echo "        Expected: $expected_sha"
                echo "        Got:      $actual_sha"
                FAIL=$((FAIL + 1))
            fi
        else
            printf "  [%2d] %-35s  MISSING\n" "$TOTAL" "$net_name"
            FAIL=$((FAIL + 1))
        fi
        continue
    fi

    if [ -f "$tflite_file" ] && ! $FORCE; then
        # File exists — verify checksum
        actual_sha=$(sha256sum "$tflite_file" | cut -d' ' -f1)
        if [ "$actual_sha" = "$expected_sha" ]; then
            printf "  [%2d] %-35s  exists (sha256 OK)\n" "$TOTAL" "$net_name"
            OK=$((OK + 1))
            continue
        else
            echo "  [$TOTAL] $net_name  checksum mismatch, re-downloading..."
        fi
    fi

    # Download
    printf "  [%2d] %-35s  downloading... " "$TOTAL" "$net_name"
    if wget -q "$url" -O "$tflite_file" 2>/dev/null || \
       curl -sL "$url" -o "$tflite_file" 2>/dev/null; then
        # Verify
        actual_sha=$(sha256sum "$tflite_file" | cut -d' ' -f1)
        if [ "$actual_sha" = "$expected_sha" ]; then
            size=$(wc -c < "$tflite_file")
            echo "OK ($(( size / 1024 )) KB)"
            OK=$((OK + 1))
        else
            echo "FAIL (sha256 mismatch!)"
            echo "        Expected: $expected_sha"
            echo "        Got:      $actual_sha"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL (download error)"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Result: $OK/$TOTAL OK, $FAIL failed"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "Some models failed! Try re-running with --force to re-download."
    exit 1
fi

echo ""
echo "Name mapping (MCUNet net_id -> tflite filename -> C identifier):"
echo ""
printf "  %-15s  %-35s  %s\n" "net_id" "tflite filename" "C identifier"
printf "  %-15s  %-35s  %s\n" "---------------" "-----------------------------------" "-----------------------------------"
for net_name in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
    net_id="${MODELS[$net_name]}"
    c_name="${net_name//-/_}"
    c_name="${c_name//./_}"
    printf "  %-15s  %-35s  %s\n" "$net_id" "${net_name}.tflite" "$c_name"
done
