#version 450
layout(local_size_x = 256) in;

// ============================================================================
// Element-wise Activation Kernel — AETERNUS Phase 3
//
// Applies activation function in-place to a float buffer.
// Used between GEMV layers in the micro model forward pass.
// ============================================================================

layout(set = 0, binding = 0) buffer Data {
    float values[];
};

layout(push_constant) uniform Params {
    uint count;
    uint mode;   // 0 = ReLU, 1 = SiLU
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= count) return;

    float x = values[idx];

    if (mode == 0u) {
        // ReLU
        values[idx] = max(x, 0.0);
    } else {
        // SiLU: x * sigmoid(x)
        values[idx] = x / (1.0 + exp(-x));
    }
}
