//! # AETERNUS — Procedural Weight Reconstruction Engine
//!
//! Sub-1-bit SPIM via Rust/Vulkan. Reconstructs neural network weights
//! procedurally, fusing reconstruction directly into GEMV operations.

pub mod seed_engine;
pub mod codebook;
pub mod vulkan_fabric;
pub mod benchmark;
pub mod bench_fused;
pub mod bench_gemv;
pub mod lease;
pub mod micro_model;
pub mod mesh;
pub mod prefetch;
pub mod headless;
pub mod ingestor;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "aeternus", about = "Procedural Weight Reconstruction engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Phase 0: sign-only reconstruction benchmark.
    Bench {
        #[arg(long, default_value = "1000000000")]
        params: u64,
        #[arg(long, default_value = "4194304")]
        tile_size: u32,
    },

    /// Phase 1: fused VQ+sign reconstruction benchmark.
    Fused {
        #[arg(long, default_value = "1000000000")]
        params: u64,
        #[arg(long, default_value = "4194304")]
        tile_size: u32,
        #[arg(long)]
        sweep: bool,
    },

    /// Phase 2: fused Reconstruct-GEMV (weights never materialize).
    Gemv {
        #[arg(long, default_value = "4096")]
        m: u32,
        #[arg(long, default_value = "4096")]
        k: u32,
        #[arg(long, default_value = "100")]
        iterations: u32,
    },

    /// Phase 3: micro model end-to-end validation and benchmarking.
    Micro {
        #[arg(long, default_value = "nano")]
        model: String,
        #[arg(long)]
        validate: bool,
        #[arg(long)]
        bench: bool,
        #[arg(long, default_value = "10")]
        iterations: u32,
    },

    /// Phase 3: model-to-model communication via latent leases.
    Mesh {
        #[arg(long, default_value = "demo")]
        preset: String,
        #[arg(long)]
        validate: bool,
    },

    /// Phase 3: async tile prefetch benchmark.
    Prefetch {
        /// Number of tiles to process.
        #[arg(long, default_value = "50")]
        tiles: usize,
        /// Elements per tile (u32).
        #[arg(long, default_value = "65536")]
        tile_elements: usize,
    },

    /// Phase 3: headless task-to-tensor multi-agent pipeline.
    Headless {
        /// Number of turns.
        #[arg(long, default_value = "4")]
        turns: usize,
    },

    /// Phase 5: ingest safetensors weights into AETERNUS format.
    Ingest {
        /// Path to directory containing .safetensors files.
        #[arg(long)]
        weights_path: String,
        /// Model config: llama3-8b, tiny-llama.
        #[arg(long, default_value = "llama3-8b")]
        config: String,
        /// Run benchmark after ingestion.
        #[arg(long)]
        bench: bool,
        /// Number of benchmark iterations.
        #[arg(long, default_value = "3")]
        iterations: u32,
    },
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Bench { params, tile_size } => {
            println!("AETERNUS Phase 0 — Sign-Only Reconstruction Benchmark");
            println!("Parameters: {}  |  Tile size: {}", params, tile_size);
            let config = benchmark::BenchConfig::new(params, tile_size);
            match benchmark::run(&config) {
                Ok(result) => println!("{}", result),
                Err(e) => { eprintln!("Benchmark failed: {}", e); std::process::exit(1); }
            }
        }

        Commands::Fused { params, tile_size, sweep } => {
            println!("AETERNUS Phase 1 — Fused VQ+Sign Reconstruction");
            println!("Parameters: {}  |  Tile size: {}  |  Bits/param: 2.0", params, tile_size);
            let config = bench_fused::FusedBenchConfig::new(params, tile_size);
            if sweep {
                match bench_fused::sweep(&config) {
                    Ok(()) => {},
                    Err(e) => { eprintln!("Sweep failed: {}", e); std::process::exit(1); }
                }
            } else {
                match bench_fused::run(&config) {
                    Ok(result) => println!("{}", result),
                    Err(e) => { eprintln!("Benchmark failed: {}", e); std::process::exit(1); }
                }
            }
        }

        Commands::Gemv { m, k, iterations } => {
            println!("AETERNUS Phase 2 — Fused Reconstruct-GEMV");
            println!("Matrix: {}x{}  |  Weights never materialize", m, k);
            let mut config = bench_gemv::GemvBenchConfig::new(m, k);
            config.iterations = iterations;
            match bench_gemv::run(&config) {
                Ok(result) => println!("{}", result),
                Err(e) => { eprintln!("GEMV failed: {}", e); std::process::exit(1); }
            }
        }

        Commands::Micro { model, validate, bench, iterations } => {
            let m = micro_model::get_model(&model).unwrap_or_else(|| {
                eprintln!("Unknown model '{}'. Available: nano, micro, mini, small, medium, large, xl", model);
                std::process::exit(1);
            });

            println!("AETERNUS Phase 3 — Micro Model: '{}'", m.name);
            println!("Layers: {}  |  Params: {}  |  Packed: {} bytes",
                     m.layers.len(), m.total_params(), m.packed_bytes());

            if validate || (!validate && !bench) {
                match micro_model::validate(&m) {
                    Ok(result) => println!("{}", result),
                    Err(e) => { eprintln!("Validation failed: {}", e); std::process::exit(1); }
                }
            }

            if bench {
                match micro_model::bench(&m, iterations) {
                    Ok(()) => {},
                    Err(e) => { eprintln!("Benchmark failed: {}", e); std::process::exit(1); }
                }
            }
        }

        Commands::Mesh { preset, validate } => {
            let (agent_a, agent_b, projector) = match preset.as_str() {
                "demo" => mesh::demo_mesh(),
                "large" => mesh::large_mesh(),
                _ => {
                    eprintln!("Unknown preset '{}'. Available: demo, large", preset);
                    std::process::exit(1);
                }
            };

            println!("AETERNUS Phase 3 — Semantic Multi-Agent Mesh");
            println!("  {} ({}→{}) → projector ({}→{}) → {} ({}→{})",
                     agent_a.name, agent_a.input_dim(), agent_a.output_dim(),
                     projector.from_dim, projector.to_dim,
                     agent_b.name, agent_b.input_dim(), agent_b.output_dim());

            let input: Vec<f32> = (0..agent_a.input_dim() as usize)
                .map(|i| ((seed_engine::pcg_hash(i as u32) % 2000) as f32 / 2000.0) - 0.5)
                .collect();
            let config = mesh::MeshConfig::default();

            if validate {
                println!("  Running CPU reference...");
                let cpu_result = mesh::run_cpu(&agent_a, &agent_b, &projector, &input, &config);
                println!("  Running GPU mesh...");
                match mesh::run_gpu(&agent_a, &agent_b, &projector, &input, &config) {
                    Ok(gpu_result) => {
                        let max_cpu = cpu_result.output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                        let max_rel = cpu_result.output.iter().zip(gpu_result.output.iter())
                            .map(|(c, g)| if max_cpu > 0.0 { (c - g).abs() / max_cpu } else { 0.0 })
                            .fold(0.0f32, f32::max);
                        println!("{}", gpu_result);
                        println!("  GPU/CPU max relative error: {:.6}", max_rel);
                        if max_rel < 1e-3 {
                            println!("  >> MESH VALIDATION PASSED");
                        } else {
                            println!("  >> MESH VALIDATION FAILED");
                        }
                    }
                    Err(e) => { eprintln!("GPU mesh failed: {}", e); std::process::exit(1); }
                }
            } else {
                match mesh::run_gpu(&agent_a, &agent_b, &projector, &input, &config) {
                    Ok(result) => println!("{}", result),
                    Err(e) => { eprintln!("Mesh failed: {}", e); std::process::exit(1); }
                }
            }
        }

        Commands::Prefetch { tiles, tile_elements } => {
            println!("AETERNUS Phase 3 — Async Tile Prefetch");
            match prefetch::bench_prefetch(tiles, tile_elements) {
                Ok(result) => println!("{}", result),
                Err(e) => { eprintln!("Prefetch bench failed: {}", e); std::process::exit(1); }
            }
        }

        Commands::Headless { turns } => {
            println!("AETERNUS Phase 3 — Headless Task-to-Tensor");
            println!("  Turns: {}  |  Strings produced: ZERO", turns);

            let mut pipeline = headless::demo_pipeline();
            let goal: Vec<f32> = (0..64)
                .map(|i| ((seed_engine::pcg_hash(i as u32) % 2000) as f32 / 2000.0) - 0.5)
                .collect();

            match pipeline.run(&goal, turns) {
                Ok(result) => println!("{}", result),
                Err(e) => { eprintln!("Headless failed: {}", e); std::process::exit(1); }
            }
        }

        Commands::Ingest { weights_path, config, bench, iterations } => {
            println!("AETERNUS Phase 5 — Safetensors Ingestor");
            println!("Weights: {}  |  Config: {}", weights_path, config);

            let llama_config = match config.as_str() {
                "llama3-8b" => ingestor::LlamaConfig::llama3_8b(),
                "tiny-llama" => ingestor::LlamaConfig::tiny_llama(),
                _ => {
                    eprintln!("Unknown config '{}'. Available: llama3-8b, tiny-llama", config);
                    std::process::exit(1);
                }
            };

            let model = match ingestor::ingest_llama(
                std::path::Path::new(&weights_path),
                &llama_config,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Ingestion failed: {}", e);
                    std::process::exit(1);
                }
            };

            println!("Ingested model: '{}'", model.name);
            println!("  Layers: {}  |  Params: {}  |  Packed: {} bytes",
                     model.layers.len(), model.total_params(), model.packed_bytes());
            println!("  Bits/param: {:.2}", (model.packed_bytes() as f64 * 8.0) / model.total_params() as f64);

            if bench {
                match micro_model::bench(&model, iterations) {
                    Ok(()) => {},
                    Err(e) => { eprintln!("Benchmark failed: {}", e); std::process::exit(1); }
                }
            }
        }
    }
}
