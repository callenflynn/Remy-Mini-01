use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use lm_model::{Model, ModelConfig, masked_cross_entropy};

/// Guards against ops without a backward pass (e.g. candle_nn's fused rms_norm
/// and softmax_last_dim), which silently leave parameters untrained.
#[test]
fn every_parameter_receives_a_finite_gradient() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let cfg = ModelConfig {
        vocab_size: 32,
        context_len: 16,
        d_model: 16,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        ffn_hidden: 24,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
    };
    let vm = VarMap::new();
    let model = Model::new(&cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
    let toks = Tensor::new(&[[1u32, 2, 3, 4, 5]], &dev)?;
    let tgt = Tensor::new(&[[2u32, 3, 4, 5, 6]], &dev)?;
    let mask = Tensor::new(&[[0f32, 0.0, 1.0, 1.0, 1.0]], &dev)?;
    let loss = masked_cross_entropy(&model.forward(&toks)?, &tgt, &mask)?;
    let grads = loss.backward()?;
    let mut missing = 0;
    for v in vm.all_vars() {
        match grads.get(v.as_tensor()) {
            None => missing += 1,
            Some(g) => {
                let n: f32 = g.sqr()?.sum_all()?.to_scalar()?;
                assert!(n.is_finite());
            }
        }
    }
    assert_eq!(missing, 0, "{missing} parameters received no gradient");
    Ok(())
}

fn grad_test_config() -> ModelConfig {
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

/// Two sequences, one with leading masked positions and one fully active. Every
/// parameter must get a finite, nonzero gradient, and the embedding (tied to the
/// output projection) must be a single variable receiving gradient from both uses.
#[test]
fn multi_sequence_masks_give_nonzero_grads_for_all_params() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let cfg = grad_test_config();
    let vm = VarMap::new();
    let model = Model::new(&cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
    let toks = Tensor::new(&[[1u32, 2, 3, 4, 5], [6, 7, 8, 9, 10]], &dev)?;
    let tgt = Tensor::new(&[[2u32, 3, 4, 5, 6], [7, 8, 9, 10, 11]], &dev)?;
    let mask = Tensor::new(
        &[[0f32, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
        &dev,
    )?;
    let loss = masked_cross_entropy(&model.forward(&toks)?, &tgt, &mask)?;
    let grads = loss.backward()?;
    let vars = vm.all_vars();
    // embed (tied, one variable) + 9 per layer (2 norms, q/k/v/o, gate/up/down) + final norm
    assert_eq!(
        vars.len(),
        2 + 9 * cfg.n_layers,
        "unexpected number of variables"
    );
    for v in &vars {
        let g = grads
            .get(v.as_tensor())
            .expect("parameter received no gradient");
        let n: f32 = g.sqr()?.sum_all()?.to_scalar()?;
        assert!(n.is_finite(), "non-finite gradient");
        assert!(
            n > 0.0,
            "zero gradient for a parameter that should be trained"
        );
    }
    Ok(())
}

/// An all-masked batch has a zero loss and zero gradients. That is mathematically
/// valid, so it must produce finite zeros rather than NaN.
#[test]
fn fully_masked_batch_gives_finite_zero_grads() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let cfg = grad_test_config();
    let vm = VarMap::new();
    let model = Model::new(&cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
    let toks = Tensor::new(&[[1u32, 2, 3, 4]], &dev)?;
    let tgt = Tensor::new(&[[2u32, 3, 4, 5]], &dev)?;
    let mask = Tensor::zeros((1, 4), DType::F32, &dev)?;
    let loss = masked_cross_entropy(&model.forward(&toks)?, &tgt, &mask)?;
    assert_eq!(loss.to_scalar::<f32>()?, 0.0);
    let grads = loss.backward()?;
    for v in vm.all_vars() {
        if let Some(g) = grads.get(v.as_tensor()) {
            let n: f32 = g.sqr()?.sum_all()?.to_scalar()?;
            assert!(n.is_finite() && n == 0.0);
        }
    }
    Ok(())
}
