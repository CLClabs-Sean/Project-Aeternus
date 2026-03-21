#version 450
layout(local_size_x = 256) in;

// ============================================================================
// Fused VQ Magnitude + PCG Sign Reconstruction Kernel — AETERNUS Phase 1
//
// Each thread:
//   1. Extracts a 2-bit magnitude index from a packed u32 word
//   2. Looks up the magnitude from a 4-entry VQ codebook
//   3. Reconstructs the sign via PCG hash
//   4. Writes the fused (magnitude × sign) to the output buffer
//
// Packing: 16 weights per u32 (2 bits each)
// Memory: 70B params × 2 bits = 17.5 GB (tiles into 2 GB VRAM)
// ============================================================================

// --- PCG Hash (identical to Phase 0) ---
uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// --- Buffers ---

// Packed magnitude indices: 16 x 2-bit indices per u32
layout(set = 0, binding = 0) readonly buffer PackedWeights {
    uint packed[];
};

// VQ codebook: 4 magnitude levels (2-bit index -> float magnitude)
layout(set = 0, binding = 1) readonly buffer Codebook {
    float magnitudes[4];
};

// Output: reconstructed float weights (magnitude x sign)
layout(set = 0, binding = 2) buffer Output {
    float weights_out[];
};

// Push constants
layout(push_constant) uniform Params {
    uint seed;       // PCG seed for sign reconstruction
    uint count;      // Total weights in this tile
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= count) return;

    // --- BFE: Extract 2-bit magnitude index from packed word ---
    // Each u32 holds 16 weights. Weight `idx` is in word `idx / 16`,
    // at bit offset `(idx % 16) * 2`.
    uint word_idx = idx >> 4u;            // idx / 16
    uint bit_offset = (idx & 15u) << 1u;  // (idx % 16) * 2
    uint mag_index = (packed[word_idx] >> bit_offset) & 3u;

    // --- VQ Codebook Lookup ---
    float magnitude = magnitudes[mag_index];

    // --- PCG Sign Reconstruction (zero storage cost, proven at 1.68x PCIe) ---
    uint hash = pcg_hash(idx ^ seed);
    float sign = (hash % 2u == 0u) ? 1.0 : -1.0;

    // --- Fused Output ---
    weights_out[idx] = magnitude * sign;
}
