//! Training-readiness smoke run.
//!
//! Runs the real stack end to end: `lm-data` pipeline output -> tokenizer -> token batches ->
//! `Trainer` (model + masked loss + AdamW + LR schedule + clipping) -> weights checkpoint ->
//! reload. Prints `AUDIT` lines with the numbers the readiness audit records. It asserts only
//! mechanical properties (finite loss, parameters change, loss falls on the probe batch,
//! checkpoint round-trip is exact); it says nothing about model quality.
//!
//! Note: the repository has no data loader; the batch construction below is audit scaffolding
//! inside this test, not product code.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use lm_model::ModelConfig;
use lm_train::{LrSchedule, Trainer};

const SEQ: usize = 64;
const BATCH: usize = 8;
const STEPS: usize = 40;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("lm-data")
        .join("fixtures")
        .join(name)
}

fn out_dir() -> PathBuf {
    let d = std::env::temp_dir().join("lm-training-readiness-audit");
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn token_stream(tok: &lm_tokenizer::LmTokenizer, texts: &[String]) -> Vec<u32> {
    let mut ids = Vec::new();
    for t in texts {
        ids.extend(tok.encode(t).unwrap());
    }
    ids
}

/// Pack a token stream into `n` batches of [BATCH, SEQ] with next-token targets and an
/// all-ones mask (pretraining-style). Positions wrap around the stream, so a short split
/// still produces full rows; the audit reports each split's true token count separately.
/// Repeat a short stream until it is long enough to cut one batch row from.
fn ensure_len(stream: &[u32]) -> Vec<u32> {
    let mut out = stream.to_vec();
    while out.len() <= SEQ + 1 {
        out.extend_from_slice(stream);
        if stream.is_empty() {
            panic!("empty token stream");
        }
    }
    out
}

fn make_batches(dev: &Device, stream: &[u32], n: usize) -> Vec<(Tensor, Tensor, Tensor)> {
    let stream = ensure_len(stream);
    let stream = &stream;
    assert!(stream.len() > SEQ + 1, "stream too short: {}", stream.len());
    let mut out = Vec::new();
    for b in 0..n {
        let mut xs = vec![vec![0u32; SEQ]; BATCH];
        let mut ts = vec![vec![0u32; SEQ]; BATCH];
        for r in 0..BATCH {
            let start = (b * BATCH + r) * SEQ % (stream.len() - SEQ - 1);
            xs[r].copy_from_slice(&stream[start..start + SEQ]);
            ts[r].copy_from_slice(&stream[start + 1..start + SEQ + 1]);
        }
        let mask = Tensor::ones((BATCH, SEQ), DType::F32, dev).unwrap();
        out.push((
            Tensor::new(xs, dev).unwrap(),
            Tensor::new(ts, dev).unwrap(),
            mask,
        ));
    }
    out
}

/// One SFT-style batch: chat-serialized conversations, assistant-only loss mask, padded to SEQ.
fn sft_batch(dev: &Device, tok: &lm_tokenizer::LmTokenizer, n: usize) -> (Tensor, Tensor, Tensor) {
    let mut rng = lm_data::rng::SplitMix64::new(42);
    let pad = tok.special_id("<|pad|>").unwrap();
    let mut xs = vec![vec![pad; SEQ]; n];
    let mut ts = vec![vec![pad; SEQ]; n];
    let mut ms = vec![vec![0f32; SEQ]; n];
    for r in 0..n {
        let ex = lm_data::synthetic::arithmetic(&mut rng, false);
        let s = lm_data::chat::serialize(tok, &ex.messages).unwrap();
        let take = s.ids.len().min(SEQ);
        for i in 0..take.saturating_sub(1) {
            xs[r][i] = s.ids[i];
            ts[r][i] = s.ids[i + 1];
            ms[r][i] = s.loss_mask[i + 1] as f32;
        }
    }
    (
        Tensor::new(xs, dev).unwrap(),
        Tensor::new(ts, dev).unwrap(),
        Tensor::new(ms, dev).unwrap(),
    )
}

fn pick_device() -> Device {
    if let Ok(dev) = Device::new_cuda(0) {
        // Probe: a forward pass must actually execute, not just allocate.
        let cfg = ModelConfig {
            vocab_size: 32,
            context_len: 8,
            d_model: 16,
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 2,
            ffn_hidden: 24,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
        };
        let vm = candle_nn::VarMap::new();
        let model = lm_model::Model::new(
            &cfg,
            candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev),
        )
        .and_then(|m| {
            let t = Tensor::new(&[[1u32, 2, 3, 4]], &dev)?;
            m.forward(&t).map(|_| ())
        });
        match model {
            Ok(()) => {
                eprintln!("AUDIT device=cuda:0");
                return dev;
            }
            Err(e) => eprintln!("AUDIT cuda_probe_failed={e}"),
        }
    } else {
        eprintln!("AUDIT cuda_unavailable");
    }
    eprintln!("AUDIT device=cpu");
    Device::Cpu
}

fn vm_hwm_mb() -> f64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return f64::NAN;
    };
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmHWM:") {
            let kb: f64 = v.trim().trim_end_matches(" kB").parse().unwrap_or(f64::NAN);
            return kb / 1024.0;
        }
    }
    f64::NAN
}

fn cfg_for(vocab: usize) -> ModelConfig {
    ModelConfig {
        vocab_size: vocab,
        context_len: SEQ,
        d_model: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        ffn_hidden: 128,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
    }
}

fn sched() -> LrSchedule {
    LrSchedule {
        max_lr: 1e-3,
        min_lr: 1e-4,
        warmup_steps: 5,
        total_steps: STEPS,
    }
}

#[test]
fn smoke_run_reports_training_readiness_metrics() {
    // ---------- Data: real pipeline ----------
    let inputs = vec![fixture("tiny.txt"), fixture("tiny.jsonl")];
    // Finding: the default 98/1/1 config cannot run on the data that exists in the repo.
    let default_err = lm_data::prepare::prepare(&lm_data::config::Config::default(), &inputs)
        .map(|_| ())
        .unwrap_err();
    eprintln!("AUDIT default_config_prepare_err={default_err}");

    let mut cfg = lm_data::config::Config::load(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("lm-data")
            .join("configs")
            .join("tiny.json"),
    )
    .unwrap();
    cfg.synthetic.arithmetic = 200;
    cfg.synthetic.logic = 100;
    cfg.synthetic.word_problems = 100;
    cfg.synthetic.instructions = 100; // configured but not generated: see AUDIT line below
    cfg.synthetic.coding = 100; // configured but not generated
    let (records, manifest) = lm_data::prepare::prepare(&cfg, &inputs).unwrap();
    eprintln!("AUDIT synthetic_counts={:?}", manifest.synthetic_counts);

    let mut per_split: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    let mut texts_by_split: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for r in &records {
        let e = per_split.entry(r.split.clone()).or_default();
        e.0 += 1;
        e.1 += r.text.len();
        texts_by_split
            .entry(r.split.clone())
            .or_default()
            .push(r.text.clone());
    }
    for (split, (n, chars)) in &per_split {
        eprintln!("AUDIT records split={split} examples={n} raw_chars={chars}");
    }
    for (origin, splits) in &manifest.counts {
        for (split, n) in splits {
            eprintln!("AUDIT records origin={origin} split={split} examples={n}");
        }
    }

    // Exact-text duplicates across splits = leakage risk (hash split is by id, not content).
    let mut seen: HashSet<&str> = HashSet::new();
    let mut cross_split_dupes = 0usize;
    let mut all: Vec<(&str, &str)> = records
        .iter()
        .map(|r| (r.split.as_str(), r.text.as_str()))
        .collect();
    all.sort();
    for (split, text) in &all {
        if !seen.insert(text) && all.iter().any(|(s2, t2)| t2 == text && s2 != split) {
            cross_split_dupes += 1;
        }
    }
    eprintln!("AUDIT cross_split_duplicate_texts={cross_split_dupes}");

    // ---------- Tokenizer: trained on the training split only ----------
    let corpus_path = out_dir().join("tokenizer_corpus.txt");
    let train_texts = texts_by_split.get("train").cloned().unwrap_or_default();
    std::fs::write(&corpus_path, train_texts.join("\n")).unwrap();
    let tok = lm_tokenizer::LmTokenizer::train(&[corpus_path.to_string_lossy().into_owned()], 512)
        .unwrap();
    let tok_path = out_dir().join("tokenizer.json");
    tok.save(&tok_path).unwrap();
    eprintln!(
        "AUDIT tokenizer vocab={} corpus_examples={} saved={}",
        tok.vocab_size(),
        train_texts.len(),
        tok_path.display()
    );
    for (split, texts) in &texts_by_split {
        let n: usize = texts.iter().map(|t| tok.encode(t).unwrap().len()).sum();
        eprintln!(
            "AUDIT tokens split={split} examples={} tokens={n}",
            texts.len()
        );
    }

    // ---------- Model config matches tokenizer and sequence length ----------
    let mcfg = cfg_for(tok.vocab_size());
    mcfg.validate().unwrap();
    eprintln!(
        "AUDIT model params={} vocab={} context={} d_model={} layers={}",
        mcfg.param_count(),
        mcfg.vocab_size,
        mcfg.context_len,
        mcfg.d_model,
        mcfg.n_layers
    );

    let dev = pick_device();

    // ---------- Real batches consumed by the real model ----------
    let train_stream = token_stream(&tok, &train_texts);
    let val_stream = token_stream(
        &tok,
        texts_by_split.get("val").map(Vec::as_slice).unwrap_or(&[]),
    );
    let test_stream = token_stream(
        &tok,
        texts_by_split.get("test").map(Vec::as_slice).unwrap_or(&[]),
    );
    eprintln!(
        "AUDIT streams train_tokens={} val_tokens={} test_tokens={}",
        train_stream.len(),
        val_stream.len(),
        test_stream.len()
    );
    let train_batches = make_batches(&dev, &train_stream, 5);
    let val_batch = make_batches(&dev, &val_stream, 1).remove(0);
    let test_batch = make_batches(&dev, &test_stream, 1).remove(0);
    let (sft_x, sft_t, sft_m) = sft_batch(&dev, &tok, BATCH);

    // ---------- Training ----------
    let mut tr = Trainer::new(&mcfg, &dev, sched(), 0.01, 1.0, 7).unwrap();
    let probe = &train_batches[0];
    let probe_before = tr.eval_loss(&probe.0, &probe.1, &probe.2).unwrap();
    let val_before = tr
        .eval_loss(&val_batch.0, &val_batch.1, &val_batch.2)
        .unwrap();
    let test_before = tr
        .eval_loss(&test_batch.0, &test_batch.1, &test_batch.2)
        .unwrap();
    let sft_before = tr.eval_loss(&sft_x, &sft_t, &sft_m).unwrap();

    let vars = tr.varmap.all_vars();
    let before: Vec<Vec<f32>> = vars
        .iter()
        .map(|v| v.flatten_all().unwrap().to_vec1::<f32>().unwrap())
        .collect();

    let mut grad_min = f64::INFINITY;
    let mut grad_max = 0f64;
    let t0 = Instant::now();
    for i in 0..STEPS {
        let b = &train_batches[i % train_batches.len()];
        let s = tr.train_step(&b.0, &b.1, &b.2).unwrap();
        assert!(
            s.loss.is_finite() && s.grad_norm.is_finite(),
            "non-finite at {i}"
        );
        grad_min = grad_min.min(s.grad_norm);
        grad_max = grad_max.max(s.grad_norm);
    }
    let dt = t0.elapsed().as_secs_f64();
    let tokens = STEPS * BATCH * SEQ;
    let after: Vec<Vec<f32>> = vars
        .iter()
        .map(|v| v.flatten_all().unwrap().to_vec1::<f32>().unwrap())
        .collect();
    let changed = before.iter().zip(&after).filter(|(a, b)| *a != *b).count();

    let probe_after = tr.eval_loss(&probe.0, &probe.1, &probe.2).unwrap();
    let val_after = tr
        .eval_loss(&val_batch.0, &val_batch.1, &val_batch.2)
        .unwrap();
    let test_after = tr
        .eval_loss(&test_batch.0, &test_batch.1, &test_batch.2)
        .unwrap();
    let sft_after = tr.eval_loss(&sft_x, &sft_t, &sft_m).unwrap();

    eprintln!(
        "AUDIT train steps={STEPS} batch={BATCH}x{SEQ} tokens_processed={tokens} wall_s={dt:.3} tokens_per_s={:.0}",
        tokens as f64 / dt
    );
    eprintln!("AUDIT grad_norm min={grad_min:.4} max={grad_max:.4}");
    eprintln!("AUDIT params_changed={changed}/{}", vars.len());
    eprintln!("AUDIT probe_loss before={probe_before:.4} after={probe_after:.4}");
    eprintln!(
        "AUDIT val_loss before={val_before:.4} ppl_before={:.2} after={val_after:.4} ppl_after={:.2}",
        val_before.exp(),
        val_after.exp()
    );
    eprintln!(
        "AUDIT test_loss before={test_before:.4} ppl_before={:.2} after={test_after:.4} ppl_after={:.2}",
        test_before.exp(),
        test_after.exp()
    );
    eprintln!("AUDIT sft_batch_loss before={sft_before:.4} after={sft_after:.4}");
    eprintln!("AUDIT peak_rss_mb={:.1}", vm_hwm_mb());

    assert!(changed > 0, "no parameters changed");
    assert!(
        probe_after < probe_before,
        "probe loss did not fall: {probe_before} -> {probe_after}"
    );

    // ---------- Checkpoint round-trip ----------
    let ckpt = out_dir().join("smoke_weights.safetensors");
    tr.save_weights(&ckpt).unwrap();
    let mut fresh = Trainer::new(&mcfg, &dev, sched(), 0.01, 1.0, 999).unwrap();
    let before_load = fresh.eval_loss(&probe.0, &probe.1, &probe.2).unwrap();
    fresh.load_weights(&ckpt).unwrap();
    let reloaded = fresh.eval_loss(&probe.0, &probe.1, &probe.2).unwrap();
    eprintln!(
        "AUDIT checkpoint path={} bytes={} fresh_loss_before_load={:.4} reloaded_loss={:.4} original_loss={:.4}",
        ckpt.display(),
        std::fs::metadata(&ckpt).unwrap().len(),
        before_load,
        reloaded,
        probe_after
    );
    assert_eq!(
        reloaded, probe_after,
        "reload must reproduce eval loss exactly"
    );

    // ---------- Resume behaviour: weights-only, optimizer state is not preserved ----------
    // Both trainers hold identical weights but different AdamW state (warm vs zeroed moments).
    // The first step reports the same pre-update loss; divergence shows from the second step.
    let _ = tr.train_step(&probe.0, &probe.1, &probe.2).unwrap();
    let _ = fresh.train_step(&probe.0, &probe.1, &probe.2).unwrap();
    let cont_a = tr.train_step(&probe.0, &probe.1, &probe.2).unwrap().loss;
    let cont_b = fresh.train_step(&probe.0, &probe.1, &probe.2).unwrap().loss;
    eprintln!(
        "AUDIT resume_second_step_loss original_with_moments={cont_a:.4} reloaded_fresh_moments={cont_b:.4} exact_resume={}",
        cont_a == cont_b
    );

    // ---------- Reproducibility on CPU ----------
    let cpu = Device::Cpu;
    let mut r1 = Trainer::new(&cfg_for(16), &cpu, sched(), 0.0, 1.0, 7).unwrap();
    let mut r2 = Trainer::new(&cfg_for(16), &cpu, sched(), 0.0, 1.0, 7).unwrap();
    let probe_t = Tensor::new(&[[1u32, 2, 3, 4]], &cpu).unwrap();
    let l1 = r1
        .model
        .forward(&probe_t)
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    let l2 = r2
        .model
        .forward(&probe_t)
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    eprintln!("AUDIT cpu_same_seed_same_init={}", l1 == l2);
}
