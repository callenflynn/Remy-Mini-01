use candle_core::{Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use lm_model::{Model, ModelConfig, masked_cross_entropy};

use crate::clip::clip_grad_norm;
use crate::schedule::LrSchedule;

/// Owns the parameters, optimizer and schedule for one training run (fp32 throughout).
pub struct Trainer {
    pub varmap: VarMap,
    pub model: Model,
    optim: AdamW,
    schedule: LrSchedule,
    max_grad_norm: f64,
    pub step: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StepStats {
    pub loss: f64,
    pub grad_norm: f64,
    pub lr: f64,
}

impl Trainer {
    pub fn new(
        cfg: &ModelConfig,
        device: &Device,
        schedule: LrSchedule,
        weight_decay: f64,
        max_grad_norm: f64,
        seed: u64,
    ) -> Result<Self> {
        // Weight init uses candle's RNG; seed it where candle supports it (CUDA).
        // candle's CPU RNG cannot be seeded, so CPU runs are not bit-reproducible.
        if device.is_cuda() {
            device.set_seed(seed)?;
        }
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
        let model = Model::new(cfg, vb)?;
        let params = ParamsAdamW {
            lr: schedule.lr_at(0),
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay,
        };
        let optim = AdamW::new(varmap.all_vars(), params)?;
        Ok(Self {
            varmap,
            model,
            optim,
            schedule,
            max_grad_norm,
            step: 0,
        })
    }

    /// Runs one optimizer step on a batch. `tokens`/`targets` are [b, t] u32, `mask` [b, t] f32.
    /// Errors if the loss or gradient norm is NaN/Inf; parameters are not updated in that case.
    pub fn train_step(
        &mut self,
        tokens: &Tensor,
        targets: &Tensor,
        mask: &Tensor,
    ) -> Result<StepStats> {
        let lr = self.schedule.lr_at(self.step);
        self.optim.set_learning_rate(lr);
        let logits = self.model.forward(tokens)?;
        let loss = masked_cross_entropy(&logits, targets, mask)?;
        let loss_v = loss.to_scalar::<f32>()? as f64;
        if !loss_v.is_finite() {
            candle_core::bail!("non-finite loss {loss_v} at step {}", self.step);
        }
        let mut grads = loss.backward()?;
        let vars = self.varmap.all_vars();
        let grad_norm = clip_grad_norm(&mut grads, &vars, self.max_grad_norm)?;
        if !grad_norm.is_finite() {
            candle_core::bail!("non-finite gradient norm at step {}", self.step);
        }
        self.optim.step(&grads)?;
        self.step += 1;
        Ok(StepStats {
            loss: loss_v,
            grad_norm,
            lr,
        })
    }

    /// Saves all model parameters to a safetensors file.
    /// Optimizer moments and the step counter are not saved: candle's AdamW keeps them private.
    pub fn save_weights<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        self.varmap.save(path)
    }

    /// Loads parameters from a safetensors file into this trainer's existing variables.
    /// Errors if the file is missing, unreadable, or lacks a variable or has the wrong shape.
    pub fn load_weights<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        self.varmap.load(path)
    }

    /// Mean masked loss over a batch without updating parameters.
    pub fn eval_loss(&self, tokens: &Tensor, targets: &Tensor, mask: &Tensor) -> Result<f64> {
        let logits = self.model.forward(tokens)?;
        Ok(masked_cross_entropy(&logits, targets, mask)?.to_scalar::<f32>()? as f64)
    }
}
