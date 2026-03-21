# =============================================================================
# AETERNUS RunPod Template
# =============================================================================
# Vulkan + Rust on NVIDIA A100. No CUDA — raw Vulkan compute.
#
# Build:
#   docker build -t aeternus-runpod -f deploy/Dockerfile .
#
# Push to Docker Hub (for RunPod custom template):
#   docker tag aeternus-runpod YOUR_DOCKERHUB/aeternus-runpod:latest
#   docker push YOUR_DOCKERHUB/aeternus-runpod:latest
# =============================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV RUST_LOG=info

# --- Vulkan ICD + loader ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    curl \
    git \
    build-essential \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create NVIDIA Vulkan ICD if not present (NVIDIA Container Runtime maps the real one)
RUN mkdir -p /usr/share/vulkan/icd.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}' \
    > /usr/share/vulkan/icd.d/nvidia_icd.json

# --- Rust toolchain ---
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# --- Clone and build AETERNUS ---
WORKDIR /workspace
RUN git clone https://github.com/CLClabs-Sean/Project-Aeternus.git && \
    cd Project-Aeternus && \
    cargo build --release 2>&1 | tail -5

WORKDIR /workspace/Project-Aeternus

# --- Expose for RunPod ---
# RunPod expects the container to stay alive. We use a simple entrypoint
# that runs smoke tests on boot, then drops to shell.
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
