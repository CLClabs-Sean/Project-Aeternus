# AGENTS.md — Project AETERNUS

> Procedural Weight Reconstruction engine. Sub-1-Bit SPIM via Rust/Vulkan.
> Weights never materialize — they are reconstructed on-the-fly in GPU registers.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    AETERNUS Runtime                              │
│                                                                  │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌────────────┐  │
│  │ Seed     │   │ Codebook  │   │ Vulkan   │   │ Lease      │  │
│  │ Engine   │──▶│ (2-bit VQ)│──▶│ GEMV     │──▶│ Pool       │  │
│  │ (PCG)    │   │           │   │ (fused)  │   │ (zero-copy)│  │
│  └──────────┘   └───────────┘   └──────────┘   └────────────┘  │
│                                       │                          │
│                                       ▼                          │
│                          ┌────────────────────┐                  │
│                          │ Multi-Agent Mesh    │                  │
│                          │ Agent A → Proj → B  │                  │
│                          └────────────────────┘                  │
│                                       │                          │
│                                       ▼                          │
│                          ┌────────────────────┐                  │
│                          │ Headless Pipeline   │                  │
│                          │ (task-to-tensor)    │                  │
│                          └────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Design Principles

1. **Weights never exist in decompressed form in VRAM.** 2-bit packed VQ magnitudes + PCG-hashed signs are reconstructed in GPU registers during the GEMV multiply-accumulate. The weight is used and discarded in the same instruction.

2. **All inter-agent communication is tensor-to-tensor.** No strings, no tokenizer, no detokenizer at any point in the execution loop. Agents exchange latent vectors through zero-copy shared leases.

3. **Everything is procedural.** Model weights are deterministic functions of `(row, col, seed)`. Given the same seed, you always get the same weight — no storage needed.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Rust (edition 2021) |
| GPU API | Vulkan 1.2 via `ash` crate |
| Shaders | GLSL → SPIR-V (pre-compiled, embedded via `include_bytes!`) |
| Memory | `gpu-allocator` for Vulkan buffer management |
| Serialization | `bytemuck` for zero-copy Pod casting |
| CLI | `clap` v4 with derive |
| Logging | `log` + `env_logger` |

## File Map

### Shaders (`shaders/`)

| File | Purpose | Workgroup |
|---|---|---|
| `pim_sign_regen.comp/.spv` | Phase 0: sign-only reconstruction via PCG hash | 256 |
| `fused_vq_sign.comp/.spv` | Phase 1: 2-bit VQ magnitude + PCG sign, fused | 256 |
| `swar_extract.comp/.spv` | SWAR bit-parallelism: 16 weights per thread | 256 |
| `fused_gemv.comp/.spv` | Phase 2: fused reconstruct-GEMV (weights in registers only) | 256 |
| `activation.comp/.spv` | Element-wise ReLU / SiLU, in-place | 256 |

All shaders are pre-compiled to SPIR-V and embedded in the binary. No runtime shader compilation.

### Core Modules (`src/`)

| File | Purpose | Key Functions |
|---|---|---|
| `seed_engine.rs` | PCG hash for deterministic sign generation | `pcg_hash()`, `pcg_sign()` |
| `codebook.rs` | 2-bit VQ codebook, pack/unpack 16 weights per u32 | `Codebook::reconstruct()`, `pack_weights()` |
| `lease.rs` | Zero-copy memory pool with refcounted sharing | `LeasePool::acquire/release()`, `Lease::share()` |
| `benchmark.rs` | Phase 0 sign-only benchmark | `run()` |
| `bench_fused.rs` | Phase 1 fused VQ+sign benchmark (supports tile sweep) | `run()`, `sweep()` |
| `bench_gemv.rs` | Phase 2 fused GEMV benchmark with CPU reference | `run()`, `cpu_gemv()` |
| `micro_model.rs` | Multi-layer forward pass (GPU + CPU), preset models | `forward_gpu()`, `forward_cpu()`, `validate()` |
| `mesh.rs` | Multi-agent mesh: Agent→lease→projector→Agent | `run_gpu()`, `run_cpu()`, `demo_mesh()` |
| `prefetch.rs` | Ping-pong double-buffer tile prefetch | `TilePrefetcher`, `bench_prefetch()` |
| `headless.rs` | Task-to-tensor pipeline, LatentAgent trait | `HeadlessPipeline::run()`, `demo_pipeline()` |

### Vulkan Fabric (`src/vulkan_fabric/`)

| File | Purpose |
|---|---|
| `mod.rs` | `VulkanContext` — instance, device, queue, allocator bootstrapping |
| `buffer.rs` | `AllocatedBuffer` — storage buffer creation, staging upload, readback |
| `pipeline.rs` | Phase 0 sign-regen compute pipeline |
| `fused_pipeline.rs` | Phase 1 fused VQ+sign compute pipeline |
| `swar_pipeline.rs` | SWAR extraction compute pipeline |
| `gemv_pipeline.rs` | Phase 2 fused GEMV pipeline (4-binding descriptor set) |
| `activation_pipeline.rs` | Activation pipeline (push constants for mode + count) |

## GEMV Kernel Details

The fused GEMV shader (`fused_gemv.comp`) is the heart of the system:

```
For each output row:
  Each thread in the 256-thread workgroup handles K/256 columns:
    1. Read packed_weights[word_idx] — 2-bit VQ indices, 16 per u32
    2. BFE extract the 2-bit index → codebook lookup → magnitude
    3. PCG hash (row * K + col) XOR seed → sign bit
    4. weight = sign ? -magnitude : magnitude
    5. MAC: acc += weight * x[col]
  Tree reduction in shared memory → output[row]
```

**Push constants:** `(M, K, seed)` — matrix dims and the per-layer weight seed.
**Descriptor bindings:** `(packed_weights, codebook[4], input_x, output_y)`.

## Lease System

The `LeasePool` allocates from a contiguous byte array with 64-byte aligned blocks:

```rust
let mut pool = LeasePool::new(64 * 1024 * 1024);  // 64 MB
let lease = pool.acquire(4096).unwrap();            // 4 KB block
let shared = lease.share();                         // refcount++, same pointer
pool.release(shared);                               // refcount--
pool.release(lease);                                // freed, coalesced
```

Zero-copy sharing: `lease.share()` returns a new `Lease` pointing to the same memory with an incremented refcount. Release only frees when refcount hits zero. Adjacent freed blocks are coalesced.

## Multi-Agent Mesh

Two agents with different latent dimensions communicate via leases:

```
fast-7B (64→128→64) → lease(latent_A)
                          ↓ zero-copy share
                     [Projector: 64→256, ReLU]
                          ↓
                     lease(latent_B) → deep-70B (256→512→128) → Output
```

The projector is itself a `MicroModel` — a 1-layer GEMV reusing the same fused kernel. No new code needed.

## Headless Pipeline

Multi-turn, multi-agent tensor loop:

```
Turn 0: goal + workspace(zeros) → [thinker] → ws' → [refiner] → ws''
Turn 1: goal + ws'' → [thinker] → ws''' → [refiner] → ws''''
...
Turn N: → final output (Vec<f32>)
```

`LatentAgent` trait: `step_cpu(goal: &[f32], workspace: &[f32]) -> Vec<f32>`.
No tokenizer. No detokenizer. Zero strings.

## CLI Reference

```bash
# Phase 0: Sign-only reconstruction
aeternus bench --params 1000000000 --tile-size 4194304

# Phase 1: Fused VQ+sign
aeternus fused --params 1000000000 --tile-size 4194304
aeternus fused --sweep  # find optimal tile size

# Phase 2: Fused GEMV
aeternus gemv --m 4096 --k 4096 --iterations 100

# Phase 3: Micro model validation
aeternus micro --model nano --validate
aeternus micro --model small --bench

# Phase 3: Multi-agent mesh
aeternus mesh --preset demo
aeternus mesh --preset large
aeternus mesh --preset demo --validate

# Phase 3: Async tile prefetch
aeternus prefetch --tiles 50 --tile-elements 65536

# Phase 3: Headless task-to-tensor
aeternus headless --turns 4
```

## Benchmark Baselines (Intel Arc iGPU, 2 GB VRAM)

| Phase | Metric | Result |
|---|---|---|
| 0 | Sign reconstruction throughput | 26.5 GB/s (1.68x PCIe) |
| 1 | Fused VQ+sign throughput | 56.8 GB/s (3.61x PCIe) |
| 2 | Fused GEMV (4096×4096) | 58.51 GFLOP/s, 0.573 ms |
| 3 | Mesh pipeline time (demo) | 309 ms |
| 3 | Headless pipeline (4 turns) | ~ 1 ms |

## Test Suite (29 tests)

```bash
cargo test
# seed_engine: 5 tests (determinism, distribution, sub-1-bit)
# codebook: 6 tests (pack/unpack, BFE, ternary, storage cost)
# lease: 6 tests (lifecycle, sharing, coalescing, alignment, OOM)
# micro_model: 2 tests (CPU forward, GPU/CPU match)
# mesh: 3 tests (CPU output, GPU/CPU cosine sim, lease cleanup)
# prefetch: 2 tests (swap correctness, double-swap identity)
# headless: 3 tests (single turn, multi-turn, zero strings)
# vulkan_fabric: 1 test (device init)
# benchmark: 1 test (smoke)
```

## Known Issues

1. **GPU/CPU mesh divergence**: Chaining 6 GEMV layers with 256-thread workgroup parallel reductions produces different floating-point accumulation orders than serial CPU. Cosine similarity > 0.9 (validated), but element-wise relative error can be ~60%. This is expected for random-weight procedural models.

2. **Flaky `gpu_mesh_matches_cpu` test**: Passes reliably when run alone, occasionally fails when run with the full suite. The cosine similarity is near the 0.9 threshold.

3. **Build on Windows**: Antivirus real-time protection can cause OS error 32 (file lock) during compilation. Fix: disable real-time protection or use `cargo clean && cargo build -j 1`.

## Deployment

### RunPod (A100)

```bash
git clone <repo> && cd aeternus
chmod +x deploy/runpod_setup.sh
./deploy/runpod_setup.sh
```

The setup script installs Vulkan, Rust, builds release, and runs smoke tests.

### Scale-Up for A100

The 80 GB VRAM enables much larger models. At 2 bits/param:
- **10B params** = 2.5 GB packed (fits easily)
- **100B params** = 25 GB packed (40x headroom)
- **320B params** = 80 GB packed (VRAM-filling)

The GEMV kernel workgroup size (256) works on all GPUs. For A100 optimization, increase to 1024 threads and process multiple rows per workgroup.

## Environment Variables

```bash
RUST_LOG=info      # or debug for verbose output
RUST_BACKTRACE=1   # for stack traces on panics
```
