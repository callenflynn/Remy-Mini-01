//! Training utilities: LR schedule, gradient clipping, AdamW step loop helpers.

pub mod clip;
pub mod export;
pub mod schedule;
pub mod trainer;

pub use clip::{clip_grad_norm, global_grad_norm};
pub use schedule::LrSchedule;
pub use trainer::{StepStats, Trainer};
