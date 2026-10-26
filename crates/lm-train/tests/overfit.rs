use candle_core::{DType, Device, Tensor};
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

#[test]
fn one_step_updates_every_parameter_and_overfit_reduces_loss() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let sched = LrSchedule {
        max_lr: 1e-2,
        min_lr: 1e-3,
        warmup_steps: 2,
        total_steps: 200,
    };
    let mut tr = Trainer::new(&cfg(), &dev, sched, 0.0, 1.0, 7)?;
    let toks = Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8]], &dev)?;
    let tgt = Tensor::new(&[[2u32, 3, 4, 5, 6, 7, 8, 9]], &dev)?;
    let mask = Tensor::ones((1, 8), DType::F32, &dev)?;

    // Keep the Var handles so each parameter is compared with itself (VarMap order is unstable).
    let vars = tr.varmap.all_vars();
    let snapshot = |vars: &[candle_core::Var]| -> candle_core::Result<Vec<Vec<f32>>> {
        vars.iter()
            .map(|v| v.flatten_all()?.to_vec1::<f32>())
            .collect()
    };
    let before = snapshot(&vars)?;
    let first = tr.train_step(&toks, &tgt, &mask)?;
    let after = snapshot(&vars)?;
    for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
        assert!(
            b != a,
            "parameter {i} was not updated by the optimizer step"
        );
    }

    let mut last = first.loss;
    for _ in 0..150 {
        last = tr.train_step(&toks, &tgt, &mask)?.loss;
    }
    assert!(last.is_finite());
    assert!(
        last < first.loss * 0.5,
        "loss did not fall: {} -> {last}",
        first.loss
    );
    Ok(())
}
