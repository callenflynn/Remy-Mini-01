use candle_core::{Device, Tensor};
use lm_model::ModelConfig;
use lm_train::{LrSchedule, Trainer};

fn cfg() -> ModelConfig {
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

fn sched() -> LrSchedule {
    LrSchedule {
        max_lr: 1e-2,
        min_lr: 1e-3,
        warmup_steps: 2,
        total_steps: 20,
    }
}

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("lm-ckpt-{}-{name}.safetensors", std::process::id()))
}

#[test]
fn weights_round_trip_reproduces_logits() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let mut a = Trainer::new(&cfg(), &dev, sched(), 0.0, 1.0, 0)?;
    // Train one step so the weights differ from initialisation.
    let toks = Tensor::new(&[[1u32, 2, 3, 4]], &dev)?;
    let tgt = Tensor::new(&[[2u32, 3, 4, 5]], &dev)?;
    let mask = Tensor::ones((1, 4), candle_core::DType::F32, &dev)?;
    a.train_step(&toks, &tgt, &mask)?;

    let path = tmp("roundtrip");
    a.save_weights(&path)?;

    let mut b = Trainer::new(&cfg(), &dev, sched(), 0.0, 1.0, 999)?;
    let before = b.model.forward(&toks)?.to_vec3::<f32>()?;
    b.load_weights(&path)?;
    let la = a.model.forward(&toks)?.to_vec3::<f32>()?;
    let lb = b.model.forward(&toks)?.to_vec3::<f32>()?;
    assert_ne!(before, la, "fresh model should differ before loading");
    assert_eq!(la, lb, "loaded model must reproduce logits exactly");
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn loading_missing_file_is_an_error() -> candle_core::Result<()> {
    let mut t = Trainer::new(&cfg(), &Device::Cpu, sched(), 0.0, 1.0, 0)?;
    assert!(t.load_weights(tmp("does-not-exist")).is_err());
    Ok(())
}

#[test]
fn loading_mismatched_architecture_is_an_error() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let small = Trainer::new(&cfg(), &dev, sched(), 0.0, 1.0, 0)?;
    let path = tmp("mismatch");
    small.save_weights(&path)?;
    let mut bigger_cfg = cfg();
    bigger_cfg.ffn_hidden = 32;
    let mut other = Trainer::new(&bigger_cfg, &dev, sched(), 0.0, 1.0, 0)?;
    assert!(
        other.load_weights(&path).is_err(),
        "shape mismatch must not load silently"
    );
    std::fs::remove_file(&path).ok();
    Ok(())
}
