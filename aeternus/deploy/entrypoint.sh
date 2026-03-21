#!/bin/bash
# =============================================================================
# AETERNUS RunPod Entrypoint
# =============================================================================
set -e

echo "================================================================"
echo "  AETERNUS — A100 Runtime"
echo "================================================================"

# Check Vulkan
echo "[1/3] Vulkan check..."
if vulkaninfo --summary 2>/dev/null | grep -q "GPU"; then
    vulkaninfo --summary 2>/dev/null | grep -E "GPU|apiVersion|driverVersion" | head -5
    echo "  Vulkan: OK"
else
    echo "  WARNING: vulkaninfo failed — NVIDIA ICD may not be mapped"
    echo "  Trying AETERNUS directly (ash loads libvulkan.so)..."
fi

# Run tests
echo "[2/3] Running tests..."
cd /workspace/Project-Aeternus
cargo test 2>&1 | tail -3

# Smoke test
echo "[3/3] Smoke test..."
./target/release/aeternus mesh --preset demo 2>&1 | tail -10

echo ""
echo "================================================================"
echo "  AETERNUS ready. GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Run: ./target/release/aeternus --help"
echo "================================================================"

# Keep container alive for RunPod interactive use
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "RunPod detected — keeping alive..."
    sleep infinity
else
    exec /bin/bash
fi
