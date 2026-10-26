//! Load a finished run and sample text from it.
//!
//! `Checkpoint` reads the three files a `train` run leaves behind (`model_config.json`,
//! `weights.safetensors`, `tokenizer.json`) and offers `complete`, which samples one
//! continuation. `ModelGenerator` adapts it to `lm_infer::Generator` so the tool-call harness
//! can drive the real model instead of a scripted test double.
//!
//! The model has no KV cache, so every generated token costs one forward pass over the whole
//! context window. That is fine for a 4 M-parameter model on a 256-token window and keeps the
//! training forward path untouched.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use lm_data::rng::SplitMix64;
use lm_infer::Generator;
use lm_model::{Model, ModelConfig};
use lm_tokenizer::{END_OF_TURN, EOS, LmTokenizer};

/// How to sample. `temperature <= 0` means greedy (argmax), which is the boring but
/// reproducible choice for checking that a checkpoint learned anything at all.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_k: usize,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.8,
            top_k: 40,
            seed: 0,
        }
    }
}

pub struct Checkpoint {
    pub model: Model,
    pub tokenizer: LmTokenizer,
    pub config: ModelConfig,
    pub device: Device,
    /// Ids that end a completion: `<|eos|>` and `<|end_of_turn|>`.
    stop: Vec<u32>,
}

impl Checkpoint {
    /// Load a run directory written by `lm-cli train`.
    pub fn load(run_dir: &Path, device: &Device) -> Result<Self> {
        let cfg_path = run_dir.join("model_config.json");
        let weights_path = run_dir.join("weights.safetensors");
        let tok_path = run_dir.join("tokenizer.json");
        for p in [&cfg_path, &weights_path, &tok_path] {
            if !p.exists() {
                bail!(
                    "missing {}: {} is not a finished training run (train writes all three files only at the end)",
                    p.display(),
                    run_dir.display()
                );
            }
        }
        let config: ModelConfig = serde_json::from_str(
            &std::fs::read_to_string(&cfg_path)
                .with_context(|| format!("reading {}", cfg_path.display()))?,
        )
        .with_context(|| format!("parsing {}", cfg_path.display()))?;
        let tokenizer = LmTokenizer::load(&tok_path)?;
        if tokenizer.vocab_size() != config.vocab_size {
            bail!(
                "tokenizer vocab {} does not match model vocab {} in {}",
                tokenizer.vocab_size(),
                config.vocab_size,
                run_dir.display()
            );
        }
        // Read-only mmap: weights are never modified on this path.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&weights_path),
                DType::F32,
                device,
            )?
        };
        let model = Model::new(&config, vb).with_context(|| {
            format!(
                "weights in {} do not match {} (a run that was interrupted, or a different run's files)",
                weights_path.display(),
                cfg_path.display()
            )
        })?;
        let stop = vec![
            tokenizer.special_id(EOS)?,
            tokenizer.special_id(END_OF_TURN)?,
        ];
        Ok(Self {
            model,
            tokenizer,
            config,
            device: device.clone(),
            stop,
        })
    }

    pub fn param_count(&self) -> usize {
        self.config.param_count()
    }

    /// Mean next-token cross-entropy on one batch. Same loss function the trainer uses, so the
    /// number is comparable to a training log line.
    pub fn batch_loss(&self, tokens: &Tensor, targets: &Tensor, mask: &Tensor) -> Result<f64> {
        let logits = self.model.forward(tokens)?;
        Ok(lm_model::masked_cross_entropy(&logits, targets, mask)?.to_scalar::<f32>()? as f64)
    }

    /// Sample a continuation of `prompt`. The prompt is cropped to the newest
    /// `context_len` tokens so long transcripts keep fitting the window.
    pub fn complete(&self, prompt: &str, s: Sampling) -> Result<String> {
        let ids = self.tokenizer.encode(prompt)?;
        if ids.is_empty() {
            bail!("prompt encoded to zero tokens");
        }
        let ctx = self.config.context_len;
        let mut tokens: Vec<u32> = ids;
        let mut generated: Vec<u32> = Vec::new();
        let mut rng = SplitMix64::new(s.seed);
        for _ in 0..s.max_tokens {
            let start = tokens.len().saturating_sub(ctx);
            let window = &tokens[start..];
            let input = Tensor::new(window.to_vec(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let last = logits.narrow(1, window.len() - 1, 1)?.squeeze(1)?;
            let row = last.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let next = sample(&row[0], s.temperature, s.top_k, &mut rng)?;
            tokens.push(next);
            generated.push(next);
            if self.stop.contains(&next) {
                break;
            }
        }
        self.tokenizer.decode(&generated, false)
    }
}

/// Draw one token from `logits`. `temperature <= 0` picks the argmax; `top_k == 0` samples
/// the full distribution. Returned ids index `logits`.
fn sample(logits: &[f32], temperature: f64, top_k: usize, rng: &mut SplitMix64) -> Result<u32> {
    if logits.is_empty() {
        bail!("empty logits");
    }
    // Descending by logit, so `order[0]` is the argmax and a top-k cut is just a truncation.
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if top_k > 0 && top_k < order.len() {
        order.truncate(top_k);
    }
    if temperature <= 0.0 {
        return Ok(order[0] as u32);
    }
    let max = order
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = order
        .iter()
        .map(|&i| ((logits[i] - max) as f64 / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return Ok(order[0] as u32);
    }
    let mut r = (rng.next_u64() as f64 / u64::MAX as f64) * total;
    for (pos, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return Ok(order[pos] as u32);
        }
    }
    Ok(*order.last().unwrap() as u32)
}

/// Wraps a checkpoint in the harness's `Generator` trait, so `lm_infer::run` can query it.
pub struct ModelGenerator<'a> {
    pub checkpoint: &'a Checkpoint,
    pub sampling: Sampling,
}

impl Generator for ModelGenerator<'_> {
    fn next_turn(&mut self, transcript: &str) -> Result<String> {
        self.checkpoint.complete(transcript, self.sampling)
    }
}

/// Convenience for callers that want the same device the trainer picks.
pub fn default_device() -> Device {
    Device::new_cuda(0).unwrap_or(Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn greedy_ignores_seed_and_picks_argmax() {
        let logits = [0.1f32, 2.0, -1.0, 1.9];
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(999);
        for _ in 0..10 {
            assert_eq!(sample(&logits, 0.0, 0, &mut a).unwrap(), 1);
            assert_eq!(sample(&logits, 0.0, 0, &mut b).unwrap(), 1);
        }
    }

    #[test]
    fn top_k_one_is_argmax_even_with_temperature() {
        let logits = [0.5f32, 0.6, 5.0, 0.4];
        let mut r = SplitMix64::new(7);
        for _ in 0..20 {
            assert_eq!(sample(&logits, 1.5, 1, &mut r).unwrap(), 2);
        }
    }

    #[test]
    fn sampling_is_reproducible_for_a_seed() {
        let logits = [1.0f32, 1.0, 1.0, 1.0, 1.0];
        let draw = |seed: u64| {
            let mut r = SplitMix64::new(seed);
            (0..16)
                .map(|_| sample(&logits, 1.0, 0, &mut r).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(3), draw(3));
        // A uniform distribution over 5 tokens must not collapse to one id.
        assert!(draw(3).iter().any(|&x| x != draw(3)[0]));
    }

    #[test]
    fn sampling_only_returns_tokens_inside_top_k() {
        let logits = [10.0f32, 9.0, 0.0, 0.0];
        let mut r = SplitMix64::new(5);
        for _ in 0..50 {
            let t = sample(&logits, 1.0, 2, &mut r).unwrap();
            assert!(t == 0 || t == 1, "sampled {t} outside top-2");
        }
    }

    #[test]
    fn empty_logits_are_an_error() {
        let mut r = SplitMix64::new(0);
        assert!(sample(&[], 1.0, 10, &mut r).is_err());
    }

    /// Round trip: save a tiny model the way `train` does, load it, complete from it.
    #[test]
    fn checkpoint_loads_and_completes() {
        let dir = std::env::temp_dir().join(format!("lm-cli-ckpt-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();

        let corpus = dir.join("corpus.txt");
        std::fs::write(
            &corpus,
            "the quick brown fox jumps over the lazy dog\n".repeat(50),
        )
        .unwrap();
        let tokenizer = LmTokenizer::train(&[corpus.to_string_lossy().into_owned()], 512).unwrap();
        tokenizer.save(&dir.join("tokenizer.json")).unwrap();

        let cfg = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            context_len: 32,
            d_model: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            ffn_hidden: 32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
        };
        std::fs::write(
            dir.join("model_config.json"),
            serde_json::to_string_pretty(&cfg).unwrap(),
        )
        .unwrap();
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let _model = Model::new(&cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev)).unwrap();
        vm.save(dir.join("weights.safetensors")).unwrap();

        let ckpt = Checkpoint::load(&dir, &dev).unwrap();
        assert_eq!(ckpt.param_count(), cfg.param_count());
        let s = Sampling {
            max_tokens: 8,
            temperature: 0.0,
            top_k: 0,
            seed: 0,
        };
        let out = ckpt.complete("the quick", s).unwrap();
        assert!(!out.is_empty(), "empty completion");
        assert_eq!(out, ckpt.complete("the quick", s).unwrap());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A run directory whose tokenizer and config disagree is rejected by name, rather than
    /// failing later with an out-of-range token id.
    #[test]
    fn mismatched_tokenizer_and_config_are_rejected() {
        let dir = std::env::temp_dir().join(format!("lm-cli-mismatch-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("corpus.txt");
        std::fs::write(
            &corpus,
            "the quick brown fox jumps over the lazy dog\n".repeat(50),
        )
        .unwrap();
        let tokenizer = LmTokenizer::train(&[corpus.to_string_lossy().into_owned()], 512).unwrap();
        tokenizer.save(&dir.join("tokenizer.json")).unwrap();
        let cfg = ModelConfig {
            vocab_size: tokenizer.vocab_size() + 7,
            context_len: 32,
            d_model: 16,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 1,
            ffn_hidden: 32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
        };
        std::fs::write(
            dir.join("model_config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("weights.safetensors"), b"placeholder").unwrap();

        let err = match Checkpoint::load(&dir, &Device::Cpu) {
            Ok(_) => panic!("accepted mismatched files"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("does not match model vocab"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn loading_a_directory_without_weights_names_the_missing_file() {
        let dir = std::env::temp_dir().join(format!("lm-cli-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = match Checkpoint::load(&dir, &Device::Cpu) {
            Ok(_) => panic!("loaded a directory with no files"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("weights.safetensors") || err.contains("model_config.json"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
