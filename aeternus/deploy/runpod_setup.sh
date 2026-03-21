#!/usr/bin/env bash
# =============================================================================
# AETERNUS RunPod Setup — A100 SXM 80 GB
# =============================================================================
# Usage:
#   git clone <your-repo> && cd aeternus
#   chmod +x deploy/runpod_setup.sh && ./deploy/runpod_setup.sh
# =============================================================================
set -euo pipefail

echo "================================================================"
echo "  AETERNUS — RunPod A100 Bootstrap"
echo "================================================================"

# --- 1. System deps (Vulkan ICD + loader) ---
echo "[1/5] Installing Vulkan loader..."
apt-get update -qq
apt-get install -y -qq libvulkan1 libvulkan-dev vulkan-tools 2>/dev/null || true

# Verify NVIDIA Vulkan ICD exists.
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ] || [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    echo "  NVIDIA Vulkan ICD: found"
else
    echo "  WARNING: NVIDIA Vulkan ICD not found. Creating fallback..."
    mkdir -p /usr/share/vulkan/icd.d
    cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3"
    }
}
EOF
fi

# --- 2. Rust toolchain ---
echo "[2/5] Installing Rust..."
if command -v cargo &>/dev/null; then
    echo "  Rust already installed: $(rustc --version)"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "  Installed: $(rustc --version)"
fi

# --- 3. Build ---
echo "[3/5] Building AETERNUS (release)..."
source "$HOME/.cargo/env" 2>/dev/null || true
cargo build --release 2>&1 | tail -3

# --- 4. Vulkan smoke test ---
echo "[4/5] Vulkan device check..."
if command -v vulkaninfo &>/dev/null; then
    vulkaninfo --summary 2>/dev/null | head -20 || echo "  vulkaninfo unavailable, trying AETERNUS directly..."
fi

# --- 5. AETERNUS smoke test ---
echo "[5/5] AETERNUS smoke tests..."
echo ""
echo "--- Unit Tests ---"
cargo test 2>&1 | tail -5

echo ""
echo "--- Mesh Demo ---"
./target/release/aeternus mesh --preset demo

echo ""
echo "--- Headless ---"
./target/release/aeternus headless --turns 4

echo ""
echo "================================================================"
echo "  AETERNUS is ready on A100!"
echo "  Run: ./target/release/aeternus --help"
echo "================================================================"
