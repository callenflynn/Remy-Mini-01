//! Export adapter test: writes an HF Llama-layout package from a trained checkpoint and a
//! golden logits file for cross-checking against HuggingFace transformers (see the audit).
use candle_core::{Device, Tensor};
use lm_model::ModelConfig;
use lm_train::export::export_llama;
use lm_train::{LrSchedule, Trainer};

fn cfg(vocab: usize) -> ModelConfig {
    ModelConfig {
        vocab_size: vocab,
        context_len: 64,
        d_model: 32,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        ffn_hidden: 48,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
    }
}

fn sched() -> LrSchedule {
    LrSchedule {
        max_lr: 1e-2,
        min_lr: 1e-3,
        warmup_steps: 2,
        total_steps: 10,
    }
}

#[test]
fn export_produces_llama_layout_and_golden_logits() {
    let dir = std::env::temp_dir().join("lm-hf-export");
    std::fs::create_dir_all(&dir).unwrap();
    let corpus = dir.join("corpus.txt");
    std::fs::write(&corpus, "the glass frog keeps its heart visible the arctic tern migrates far the axolotl regrows limbs and the honeybee dances for food ").unwrap();
    let tok =
        lm_tokenizer::LmTokenizer::train(&[corpus.to_string_lossy().into_owned()], 256).unwrap();
    let tok_file = dir.join("tokenizer.json");
    tok.save(&tok_file).unwrap();

    let dev = Device::Cpu;
    let mcfg = cfg(tok.vocab_size());
    let mut tr = Trainer::new(&mcfg, &dev, sched(), 0.0, 1.0, 7).unwrap();
    let toks = Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8]], &dev).unwrap();
    let tgt = Tensor::new(&[[2u32, 3, 4, 5, 6, 7, 8, 9]], &dev).unwrap();
    let mask = Tensor::ones((1, 8), candle_core::DType::F32, &dev).unwrap();
    for _ in 0..5 {
        tr.train_step(&toks, &tgt, &mask).unwrap();
    }
    let raw = dir.join("raw.safetensors");
    tr.save_weights(&raw).unwrap();
    let out = dir.join("export");
    export_llama(&raw, &out, &mcfg, &tok, &tok_file).unwrap();

    assert!(out.join("model.safetensors").exists());
    assert!(out.join("config.json").exists());
    let logits = tr.model.forward(&toks).unwrap().to_vec3::<f32>().unwrap();
    let golden = serde_json::json!({
        "ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "logits": logits[0],
    });
    std::fs::write(dir.join("golden_logits.json"), golden.to_string()).unwrap();
}
