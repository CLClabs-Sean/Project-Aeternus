#version 450

// ============================================================================
// Fused Reconstruct-GEMV Kernel — AETERNUS Phase 2
//
// y[row] = sum_k( reconstruct(W[row,k]) * x[k] )
// Weights NEVER materialize as f32 in VRAM.
// ============================================================================

layout(local_size_x = 256) in;

uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

layout(set = 0, binding = 0) readonly buffer PackedW { uint packed_w[]; };
layout(set = 0, binding = 1) readonly buffer Codebook { float magnitudes[4]; };
layout(set = 0, binding = 2) readonly buffer InputX { float x[]; };
layout(set = 0, binding = 3) buffer OutputY { float y[]; };

layout(push_constant) uniform Params {
    uint seed;
    uint M;
    uint K;
};

shared float s_partial[256];

void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;

    if (row >= M) {
        return;
    }

    uint words_per_row = (K + 15u) / 16u;
    uint row_word_base = row * words_per_row;

    // Each thread accumulates over a strided slice of K
    float acc = 0.0;
    uint k = tid;
    while (k < K) {
        // BFE: extract 2-bit magnitude index
        uint word_idx = row_word_base + (k >> 4u);
        uint bit_offset = (k & 15u) << 1u;
        uint mag_idx = (packed_w[word_idx] >> bit_offset) & 3u;

        // VQ lookup
        float magnitude = magnitudes[mag_idx];

        // PCG sign
        uint weight_global = row * K + k;
        uint hash = pcg_hash(weight_global ^ seed);
        float sign = (hash % 2u == 0u) ? 1.0 : -1.0;

        // Fused MAC
        acc += (magnitude * sign) * x[k];

        k += 256u;
    }

    // Store partial sum
    s_partial[tid] = acc;
    barrier();

    // Tree reduction with fixed constants
    if (tid < 128u) { s_partial[tid] += s_partial[tid + 128u]; }
    barrier();
    if (tid < 64u) { s_partial[tid] += s_partial[tid + 64u]; }
    barrier();
    if (tid < 32u) { s_partial[tid] += s_partial[tid + 32u]; }
    barrier();
    if (tid < 16u) { s_partial[tid] += s_partial[tid + 16u]; }
    barrier();
    if (tid < 8u) { s_partial[tid] += s_partial[tid + 8u]; }
    barrier();
    if (tid < 4u) { s_partial[tid] += s_partial[tid + 4u]; }
    barrier();
    if (tid < 2u) { s_partial[tid] += s_partial[tid + 2u]; }
    barrier();
    if (tid < 1u) { s_partial[tid] += s_partial[tid + 1u]; }
    barrier();

    if (tid == 0u) {
        y[row] = s_partial[0];
    }
}
