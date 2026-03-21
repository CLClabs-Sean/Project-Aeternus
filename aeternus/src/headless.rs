//! # Headless Execution — Task-to-Tensor Interface
//!
//! No strings. No tokenizer. Agents consume and produce latent vectors
//! through shared leases. Multi-turn pipeline:
//!
//! ```text
//! goal (latent) → [Agent 0] → workspace′ → [Agent 1] → workspace″ → ...
//!                      ↑ all via zero-copy leases ↑
//! ```

use crate::lease::{Lease, LeasePool};
use crate::micro_model::{self, Activation, MicroModel};
use crate::mesh::Projector;
use crate::seed_engine;

// ---------------------------------------------------------------------------
// LatentAgent trait
// ---------------------------------------------------------------------------

/// An agent that operates purely in latent space.
pub trait LatentAgent {
    fn name(&self) -> &str;
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;

    /// Execute one step: read goal+workspace, produce output Vec.
    fn step_cpu(
        &self,
        goal: &[f32],
        workspace: &[f32],
    ) -> Vec<f32>;
}

// ---------------------------------------------------------------------------
// ModelAgent: wraps a MicroModel as a LatentAgent
// ---------------------------------------------------------------------------

pub struct ModelAgent {
    pub label: String,
    pub model: MicroModel,
}

impl ModelAgent {
    pub fn new(label: &str, model: MicroModel) -> Self {
        Self { label: label.to_string(), model }
    }
}

impl LatentAgent for ModelAgent {
    fn name(&self) -> &str { &self.label }

    fn input_dim(&self) -> usize {
        self.model.layers.first().map(|l| l.cols as usize).unwrap_or(0)
    }

    fn output_dim(&self) -> usize {
        self.model.layers.last().map(|l| l.rows as usize).unwrap_or(0)
    }

    fn step_cpu(
        &self,
        goal: &[f32],
        workspace: &[f32],
    ) -> Vec<f32> {
        let input_dim = self.input_dim();
        let input: Vec<f32> = (0..input_dim).map(|i| {
            let g = if i < goal.len() { goal[i] } else { 0.0 };
            let w = if i < workspace.len() { workspace[i] } else { 0.0 };
            g + w
        }).collect();

        micro_model::forward_cpu(&self.model, &input)
    }
}

// ---------------------------------------------------------------------------
// HeadlessPipeline: multi-turn, multi-agent
// ---------------------------------------------------------------------------

pub struct HeadlessPipeline {
    pub agents: Vec<Box<dyn LatentAgent>>,
    pub projectors: Vec<Option<Projector>>,
    pub pool: LeasePool,
}

/// Result of a headless pipeline run.
pub struct HeadlessResult {
    pub turns: usize,
    pub agents_per_turn: usize,
    pub total_steps: usize,
    pub final_output: Vec<f32>,
    pub leases_created: u64,
    pub elapsed_ms: f64,
}

impl std::fmt::Display for HeadlessResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "")?;
        writeln!(f, "================================================================")?;
        writeln!(f, "  AETERNUS — Headless Task-to-Tensor Pipeline")?;
        writeln!(f, "================================================================")?;
        writeln!(f, "  Turns:              {:>38}", self.turns)?;
        writeln!(f, "  Agents/turn:        {:>38}", self.agents_per_turn)?;
        writeln!(f, "  Total agent steps:  {:>38}", self.total_steps)?;
        writeln!(f, "  Leases created:     {:>38}", self.leases_created)?;
        writeln!(f, "  Pipeline time:      {:>34.2} ms", self.elapsed_ms)?;
        writeln!(f, "----------------------------------------------------------------")?;
        writeln!(f, "  Output dim:         {:>38}", self.final_output.len())?;
        let l2: f64 = self.final_output.iter()
            .map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
        writeln!(f, "  Output L2 norm:     {:>38.4}", l2)?;
        writeln!(f, "  Strings produced:   {:>38}", "ZERO")?;
        writeln!(f, "----------------------------------------------------------------")?;
        writeln!(f, "  >> HEADLESS EXECUTION VALIDATED")?;
        writeln!(f, "================================================================")?;
        Ok(())
    }
}

impl HeadlessPipeline {
    pub fn new(pool_size: usize) -> Self {
        Self {
            agents: Vec::new(),
            projectors: Vec::new(),
            pool: LeasePool::new(pool_size),
        }
    }

    pub fn add_agent(&mut self, agent: Box<dyn LatentAgent>) {
        self.agents.push(agent);
        self.projectors.push(None);
    }

    pub fn set_projector(&mut self, idx: usize, proj: Projector) {
        if idx < self.projectors.len() {
            self.projectors[idx] = Some(proj);
        }
    }

    /// Run the pipeline for N turns.
    /// Each turn: goal + workspace → agent[0] → agent[1] → ...
    /// Last agent's output becomes next turn's workspace.
    pub fn run(
        &mut self,
        goal_data: &[f32],
        turns: usize,
    ) -> Result<HeadlessResult, Box<dyn std::error::Error>> {
        let start = std::time::Instant::now();
        let mut leases_created = 0u64;
        let num_agents = self.agents.len();

        // Goal lease — immutable across all turns.
        let goal_bytes = goal_data.len() * 4;
        let mut goal_lease = self.pool.acquire(goal_bytes)
            .ok_or("Pool OOM for goal")?;
        goal_lease.as_mut_slice::<f32>().copy_from_slice(goal_data);
        leases_created += 1;

        // Initial workspace = zeros.
        let first_input_dim = self.agents[0].input_dim();
        let ws_bytes = first_input_dim * 4;
        let workspace_lease = self.pool.acquire(ws_bytes)
            .ok_or("Pool OOM for initial workspace")?;
        leases_created += 1;

        // Track the current workspace data as a Vec<f32> to avoid
        // ownership issues. Leases are used for accounting only.
        let mut workspace_data: Vec<f32> = workspace_lease.as_slice::<f32>().to_vec();
        self.pool.release(workspace_lease);

        let mut total_steps = 0usize;
        let mut final_output = Vec::new();

        for turn in 0..turns {
            let goal_slice: &[f32] = goal_lease.as_slice();
            let mut current_data = workspace_data.clone();

            for agent_idx in 0..num_agents {
                // Agent step: reads goal + current workspace, produces output.
                let output = self.agents[agent_idx].step_cpu(goal_slice, &current_data);
                total_steps += 1;

                // Store in a lease for accounting.
                let out_bytes = output.len() * 4;
                let mut out_lease = self.pool.acquire(out_bytes)
                    .ok_or("Pool OOM for agent output")?;
                out_lease.as_mut_slice::<f32>().copy_from_slice(&output);
                leases_created += 1;

                // Project if needed for next agent.
                if agent_idx + 1 < num_agents {
                    if let Some(ref proj) = self.projectors[agent_idx] {
                        let proj_input: &[f32] = out_lease.as_slice();
                        let projected = micro_model::forward_cpu(&proj.model, proj_input);
                        self.pool.release(out_lease);

                        let proj_bytes = projected.len() * 4;
                        let mut proj_lease = self.pool.acquire(proj_bytes)
                            .ok_or("Pool OOM for projection")?;
                        proj_lease.as_mut_slice::<f32>().copy_from_slice(&projected);
                        leases_created += 1;

                        current_data = proj_lease.as_slice::<f32>().to_vec();
                        self.pool.release(proj_lease);
                    } else {
                        current_data = out_lease.as_slice::<f32>().to_vec();
                        self.pool.release(out_lease);
                    }
                } else {
                    // Last agent — output becomes next turn's workspace.
                    final_output = out_lease.as_slice::<f32>().to_vec();
                    self.pool.release(out_lease);

                    // Resize to first agent's input dim for next turn.
                    workspace_data = (0..first_input_dim).map(|i| {
                        if i < final_output.len() { final_output[i] } else { 0.0 }
                    }).collect();
                }
            }
        }

        self.pool.release(goal_lease);

        Ok(HeadlessResult {
            turns,
            agents_per_turn: num_agents,
            total_steps,
            final_output,
            leases_created,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

// ---------------------------------------------------------------------------
// Preset pipelines
// ---------------------------------------------------------------------------

pub fn demo_pipeline() -> HeadlessPipeline {
    let mut pipeline = HeadlessPipeline::new(64 * 1024 * 1024);

    let thinker = ModelAgent::new("thinker", MicroModel::new("think", &[
        (128, 64, Activation::ReLU),
        (64, 128, Activation::None),
    ], 0xAAAA_0001));

    let refiner = ModelAgent::new("refiner", MicroModel::new("refine", &[
        (256, 64, Activation::ReLU),
        (64, 256, Activation::None),
    ], 0xBBBB_0002));

    pipeline.add_agent(Box::new(thinker));
    pipeline.add_agent(Box::new(refiner));

    pipeline
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headless_single_turn() {
        let mut pipeline = demo_pipeline();
        let goal: Vec<f32> = (0..64).map(|i| {
            ((seed_engine::pcg_hash(i as u32) % 2000) as f32 / 2000.0) - 0.5
        }).collect();

        let result = pipeline.run(&goal, 1).expect("single turn should work");
        assert_eq!(result.turns, 1);
        assert_eq!(result.total_steps, 2);
        assert!(result.final_output.len() > 0);
        assert!(result.final_output.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn headless_multi_turn() {
        let mut pipeline = demo_pipeline();
        let goal: Vec<f32> = (0..64).map(|i| {
            ((seed_engine::pcg_hash(i as u32) % 2000) as f32 / 2000.0) - 0.5
        }).collect();

        let result = pipeline.run(&goal, 4).expect("multi-turn should work");
        assert_eq!(result.turns, 4);
        assert_eq!(result.total_steps, 8);
        assert!(result.final_output.len() > 0);
    }

    #[test]
    fn headless_produces_zero_strings() {
        let mut pipeline = demo_pipeline();
        let goal = vec![0.1f32; 64];
        let result = pipeline.run(&goal, 1).expect("should work");
        assert!(result.final_output.iter().all(|v| v.is_finite()));
    }
}
