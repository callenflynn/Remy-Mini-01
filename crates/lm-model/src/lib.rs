//! Decoder-only transformer: pre-norm RMSNorm, RoPE, SwiGLU, grouped-query
//! attention, tied input/output embeddings, bias-free linear layers.

pub mod config;
pub mod layers;

use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub use config::ModelConfig;
use layers::{Block, Rope, rms_norm};

pub struct Model {
    cfg: ModelConfig,
    embed: candle_nn::Embedding,
    blocks: Vec<Block>,
    norm: layers::RmsNorm,
    rope: Rope,
}

impl Model {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        // GPT-2's embedding scale. candle's default normal(0, 1) makes the tied output logits
        // enormous at step 0 (std ~ sqrt(d_model)), so the first losses are ~7x `ln(vocab)`
        // instead of ~`ln(vocab)` and the early gradient steps are dominated by rescaling.
        let embed = candle_nn::Embedding::new(
            vb.pp("embed").get_with_hints(
                (cfg.vocab_size, cfg.d_model),
                "weight",
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            )?,
            cfg.d_model,
        );
        let blocks = (0..cfg.n_layers)
            .map(|i| {
                Block::new(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    cfg.ffn_hidden,
                    cfg.rms_eps,
                    vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = rms_norm(cfg.d_model, cfg.rms_eps, vb.pp("norm"))?;
        let rope = Rope::new(
            cfg.head_dim(),
            cfg.context_len,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;
        Ok(Self {
            cfg: cfg.clone(),
            embed,
            blocks,
            norm,
            rope,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// tokens: [batch, seq] u32. Returns logits [batch, seq, vocab].
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let mut h = self.embed.forward(tokens)?;
        for block in &self.blocks {
            h = block.forward(&h, &self.rope)?;
        }
        let h = self.norm.forward(&h)?;
        // Tied output projection: logits = h @ E^T
        let w = self.embed.embeddings().t()?;
        h.broadcast_matmul(&w)
    }
}

/// Mean cross-entropy over positions where `mask` is 1. `targets` is [batch, seq] u32,
/// `mask` is [batch, seq] f32. Computed in f32 for stability.
pub fn masked_cross_entropy(logits: &Tensor, targets: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    let logits = logits.to_dtype(DType::F32)?.reshape((b * t, v))?;
    let logp = candle_nn::ops::log_softmax(&logits, 1)?;
    let targets = targets.reshape((b * t, 1))?;
    let picked = logp.gather(&targets, 1)?.reshape((b * t,))?;
    let mask = mask.reshape((b * t,))?.to_dtype(DType::F32)?;
    let total = (picked * &mask)?.sum_all()?.neg()?;
    let count = mask.sum_all()?.clamp(1.0, f64::MAX)?;
    total.broadcast_div(&count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    fn tiny_cfg() -> ModelConfig {
        ModelConfig {
            vocab_size: 32,
            context_len: 16,
            d_model: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            ffn_hidden: 24,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
        }
    }

    #[test]
    fn causal_mask_blocks_future_positions() -> Result<()> {
        let m = layers::causal_mask(4, DType::F32, &Device::Cpu)?.to_vec2::<f32>()?;
        for i in 0..4 {
            for j in 0..4 {
                if j <= i {
                    assert_eq!(m[i][j], 0.0);
                } else {
                    assert!(m[i][j].is_infinite() && m[i][j] < 0.0);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn logits_do_not_depend_on_future_tokens() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let model = Model::new(&cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
        let a = Tensor::new(&[[1u32, 2, 3, 4, 5]], &dev)?;
        let b = Tensor::new(&[[1u32, 2, 3, 9, 7]], &dev)?;
        let la = model.forward(&a)?.to_vec3::<f32>()?;
        let lb = model.forward(&b)?.to_vec3::<f32>()?;
        for pos in 0..3 {
            for v in 0..cfg.vocab_size {
                assert!(
                    (la[0][pos][v] - lb[0][pos][v]).abs() < 1e-5,
                    "leak at pos {pos}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn loss_ignores_masked_positions() -> Result<()> {
        let dev = Device::Cpu;
        let logits = Tensor::new(&[[[0f32, 0.0], [0.0, 0.0]]], &dev)?;
        let targets = Tensor::new(&[[0u32, 1]], &dev)?;
        let full = Tensor::new(&[[1f32, 1.0]], &dev)?;
        let only_first = Tensor::new(&[[1f32, 0.0]], &dev)?;
        let wrong = Tensor::new(&[[[10f32, 0.0], [0.0, 0.0]]], &dev)?;
        let l1 = masked_cross_entropy(&logits, &targets, &only_first)?.to_scalar::<f32>()?;
        assert!((l1 - 2f32.ln()).abs() < 1e-5);
        let l2 = masked_cross_entropy(&wrong, &targets, &only_first)?.to_scalar::<f32>()?;
        assert!(l2 < 1e-3, "masked-out position leaked into loss: {l2}");
        let l3 = masked_cross_entropy(&logits, &targets, &full)?.to_scalar::<f32>()?;
        assert!((l3 - 2f32.ln()).abs() < 1e-5);
        Ok(())
    }

    /// A fresh model must be as uncertain as a uniform distribution over the vocabulary:
    /// that is what `ln(vocab)` looks like. Initial logits far from zero make the first steps
    /// huge (they were ~68 instead of ~9 for an 8k vocabulary) and destabilise early training.
    #[test]
    fn fresh_model_starts_near_uniform() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = ModelConfig {
            vocab_size: 1024,
            context_len: 16,
            d_model: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            ffn_hidden: 128,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
        };
        let vm = VarMap::new();
        let model = Model::new(&cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
        let ids: Vec<u32> = (0..16).map(|i| (i * 37 % cfg.vocab_size) as u32).collect();
        let tokens = Tensor::new(&ids[..8], &dev)?.unsqueeze(0)?;
        let targets = Tensor::new(&ids[1..9], &dev)?.unsqueeze(0)?;
        let mask = Tensor::ones((1, 8), DType::F32, &dev)?;
        let loss = masked_cross_entropy(&model.forward(&tokens)?, &targets, &mask)?
            .to_scalar::<f32>()? as f64;
        let uniform = (cfg.vocab_size as f64).ln();
        assert!(
            (loss - uniform).abs() < 1.0,
            "initial loss {loss:.2} is far from uniform {uniform:.2}"
        );
        Ok(())
    }

    #[test]
    fn param_count_matches_var_map() -> Result<()> {
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let _model = Model::new(&cfg, VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu))?;
        let actual: usize = vm.all_vars().iter().map(|v| v.elem_count()).sum();
        assert_eq!(actual, cfg.param_count());
        Ok(())
    }
}
