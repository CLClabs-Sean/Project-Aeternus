#version 450
layout(local_size_x = 256) in;

// ============================================================================
// SWAR 16-Wide Extraction + VQ + Sign — AETERNUS Phase 2
//
// Each thread processes ONE u32 word = 16 weights.
// SWAR (SIMD-Within-A-Register) extracts all 16 x 2-bit indices,
// performs codebook lookup and sign reconstruction, writes 16 floats.
// Dispatched threads = total_weights / 16.
// ============================================================================

uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Packed magnitude words: 16 x 2-bit indices per u32
layout(set = 0, binding = 0) readonly buffer PackedWeights {
    uint packed[];
};

// VQ codebook (4 magnitudes)
layout(set = 0, binding = 1) readonly buffer Codebook {
    float magnitudes[4];
};

// Output reconstructed weights
layout(set = 0, binding = 2) buffer Output {
    float weights_out[];
};

layout(push_constant) uniform Params {
    uint seed;
    uint word_count;   // total packed words (= total_weights / 16)
};

// Shared memory codebook for L1 cache residency
shared float s_codebook[4];

void main() {
    uint wid = gl_GlobalInvocationID.x;

    // Load codebook into shared memory (once per workgroup)
    if (gl_LocalInvocationID.x < 4u) {
        s_codebook[gl_LocalInvocationID.x] = magnitudes[gl_LocalInvocationID.x];
    }
    barrier();

    if (wid >= word_count) return;

    uint word = packed[wid];
    uint base_idx = wid << 4u;  // wid * 16

    // SWAR: unroll 16 extractions from the single word.
    // Each iteration: shift, mask 2 bits, codebook lookup, sign, write.
    for (uint i = 0u; i < 16u; i++) {
        uint bit_offset = i << 1u;
        uint mag_idx = (word >> bit_offset) & 3u;

        float magnitude = s_codebook[mag_idx];

        uint weight_idx = base_idx + i;
        uint hash = pcg_hash(weight_idx ^ seed);
        float sign = (hash % 2u == 0u) ? 1.0 : -1.0;

        weights_out[weight_idx] = magnitude * sign;
    }
}
