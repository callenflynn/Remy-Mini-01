//! `lm-cli`: prepare training data and run training end to end.
//!
//! Subcommands:
//! - `prepare --config <file> --inputs <files...> --out <dir>`: run the `lm-data` pipeline and
//!   write `records.jsonl` + `manifest.json`.
//! - `train --config <file> --inputs <paths...> --out <dir> [--steps N] [--batch N] [--seq N]
//!   [--mode pretrain|sft] [--vocab N] [--lr X] [--d-model N] [--layers N] [--heads N]`:
//!   prepare, train a tokenizer on the train split, batch with `lm-data::loader`, train with
//!   `lm-train::Trainer`, save weights and a metrics summary.
//!
//! `--inputs` takes files or directories; directories are walked for `.txt`, `.md`, `.jsonl`.
//!
//! Deterministic where the underlying pieces are: fixed config seed gives identical data.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use lm_data::config::Config;
use lm_data::loader::{Batch, PretrainData, SftData};
use lm_model::ModelConfig;
use lm_tokenizer::LmTokenizer;
use lm_train::{LrSchedule, Trainer};

struct Args {
    config: PathBuf,
    inputs: Vec<PathBuf>,
    out: PathBuf,
    steps: usize,
    batch: usize,
    seq: usize,
    vocab: usize,
    mode: String,
    lr: f64,
    d_model: usize,
    layers: usize,
    heads: usize,
}

fn parse_args(argv: &[String]) -> Result<Args> {
    let mut config = None;
    let mut inputs = Vec::new();
    let mut out = None;
    let mut args = Args {
        config: PathBuf::new(),
        inputs: Vec::new(),
        out: PathBuf::new(),
        steps: 50,
        batch: 8,
        seq: 64,
        vocab: 512,
        mode: "pretrain".into(),
        lr: 1e-3,
        d_model: 64,
        layers: 2,
        heads: 4,
    };
    let mut i = 0;
    while i < argv.len() {
        let take = |i: &mut usize| -> Result<String> {
            *i += 1;
            argv.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing value for flag"))
        };
        match argv[i].as_str() {
            "--config" => config = Some(PathBuf::from(take(&mut i)?)),
            "--out" => out = Some(PathBuf::from(take(&mut i)?)),
            "--inputs" => {
                i += 1;
                while i < argv.len() && !argv[i].starts_with("--") {
                    inputs.push(PathBuf::from(&argv[i]));
                    i += 1;
                }
                continue;
            }
            "--steps" => args.steps = take(&mut i)?.parse()?,
            "--batch" => args.batch = take(&mut i)?.parse()?,
            "--seq" => args.seq = take(&mut i)?.parse()?,
            "--vocab" => args.vocab = take(&mut i)?.parse()?,
            "--mode" => args.mode = take(&mut i)?,
            "--lr" => args.lr = take(&mut i)?.parse()?,
            "--d-model" => args.d_model = take(&mut i)?.parse()?,
            "--layers" => args.layers = take(&mut i)?.parse()?,
            "--heads" => args.heads = take(&mut i)?.parse()?,
            other => bail!("unknown flag '{other}'"),
        }
        i += 1;
    }
    args.config = config.ok_or_else(|| anyhow::anyhow!("--config is required"))?;
    args.out = out.ok_or_else(|| anyhow::anyhow!("--out is required"))?;
    args.inputs = inputs;
    if args.inputs.is_empty() {
        bail!("--inputs needs at least one file");
    }
    if args.mode != "pretrain" && args.mode != "sft" {
        bail!("--mode must be 'pretrain' or 'sft'");
    }
    Ok(args)
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().cloned().unwrap_or_default();
    match cmd.as_str() {
        "prepare" => run_prepare(parse_args(&argv[1..])?),
        "train" => run_train(parse_args(&argv[1..])?),
        "export" => run_export(&argv[1..]),
        _ => {
            bail!("usage: lm-cli <prepare|train|export> ...")
        }
    }
}

/// `lm-cli export --weights w.safetensors --model-config model_config.json
///   --tokenizer tokenizer.json --out <dir>`: write an HF LlamaForCausalLM directory that
/// Soup (and transformers) can load directly.
fn run_export(argv: &[String]) -> Result<()> {
    let mut flags: std::collections::HashMap<String, String> = Default::default();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i]
            .strip_prefix("--")
            .ok_or_else(|| anyhow::anyhow!("expected --flag, got {}", argv[i]))?;
        let value = argv
            .get(i + 1)
            .ok_or_else(|| anyhow::anyhow!("missing value for --{key}"))?;
        flags.insert(key.to_string(), value.clone());
        i += 2;
    }
    let need = |k: &str| -> Result<String> {
        flags
            .get(k)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("--{k} is required by export"))
    };
    let weights = PathBuf::from(need("weights")?);
    let model_config = PathBuf::from(need("model-config")?);
    let tokenizer = PathBuf::from(need("tokenizer")?);
    let out = PathBuf::from(need("out")?);

    let mcfg: ModelConfig = serde_json::from_str(&std::fs::read_to_string(&model_config)?)?;
    let tok = LmTokenizer::load(&tokenizer)?;
    lm_train::export::export_llama(&weights, &out, &mcfg, &tok, &tokenizer)?;
    println!("exported HF Llama package -> {}", out.display());
    Ok(())
}

fn run_prepare(a: Args) -> Result<()> {
    let cfg = Config::load(&a.config)?;
    let (records, manifest) = lm_data::prepare::prepare(&cfg, &a.inputs)?;
    std::fs::create_dir_all(&a.out)?;
    let rec_path = a.out.join("records.jsonl");
    let mut buf = String::new();
    for r in &records {
        buf.push_str(&serde_json::to_string(r)?);
        buf.push('\n');
    }
    std::fs::write(&rec_path, buf)?;
    std::fs::write(
        a.out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    println!(
        "prepared {} records ({} deduped) -> {}",
        records.len(),
        manifest.dedup_removed,
        rec_path.display()
    );
    Ok(())
}

/// Tokenize one split's texts into a flat stream length report, per split.
fn split_texts(
    records: &[lm_data::prepare::Record],
) -> std::collections::BTreeMap<String, Vec<String>> {
    let mut m: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for r in records {
        m.entry(r.split.clone()).or_default().push(r.text.clone());
    }
    m
}

fn run_train(a: Args) -> Result<()> {
    let cfg = Config::load(&a.config)?;
    let (records, manifest) = lm_data::prepare::prepare(&cfg, &a.inputs)?;
    std::fs::create_dir_all(&a.out)?;
    std::fs::write(
        a.out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    // Tokenizer trained on the training split only: no eval text in the tokenizer corpus.
    let by_split = split_texts(&records);
    let train_texts = by_split.get("train").cloned().unwrap_or_default();
    if train_texts.is_empty() {
        bail!("training split is empty");
    }
    let corpus_path = a.out.join("tokenizer_corpus.txt");
    std::fs::write(&corpus_path, train_texts.join("\n"))?;
    let tok = LmTokenizer::train(&[corpus_path.to_string_lossy().into_owned()], a.vocab)
        .context("training tokenizer on the training split")?;
    let tok_path = a.out.join("tokenizer.json");
    tok.save(&tok_path)?;
    println!(
        "tokenizer vocab={} saved={}",
        tok.vocab_size(),
        tok_path.display()
    );

    // Batches. Pretraining packs the flattened record texts; SFT regenerates the
    // deterministic synthetic conversations and tool-use trajectories from the config seed
    // (record text is a flattened render, so it cannot be un-flattened into messages).
    let train_batches: Vec<Batch> = if a.mode == "pretrain" {
        PretrainData::new(&tok, &train_texts, a.seq)?.batches(a.batch)?
    } else {
        let convos = sft_conversations(&cfg, false)?;
        SftData::new(&tok, &convos, a.seq)?.batches(&tok, a.batch)?
    };
    let val_batches: Vec<Batch> = if a.mode == "pretrain" {
        match by_split.get("val") {
            Some(v) if !v.is_empty() => PretrainData::new(&tok, v, a.seq)
                .ok()
                .map(|d| d.batches(a.batch).unwrap_or_default())
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    } else {
        let convos = sft_conversations(&cfg, true)?;
        SftData::new(&tok, &convos, a.seq)
            .ok()
            .map(|d| d.batches(&tok, a.batch).unwrap_or_default())
            .unwrap_or_default()
    };
    println!(
        "batches train={} val={} mode={}",
        train_batches.len(),
        val_batches.len(),
        a.mode
    );

    // Model sized to the tokenizer and sequence length; width, depth and head count come from
    // the command line so the same data can be smoke-tested and then trained for real.
    let mcfg = ModelConfig {
        vocab_size: tok.vocab_size(),
        context_len: a.seq,
        d_model: a.d_model,
        n_layers: a.layers,
        n_heads: a.heads,
        n_kv_heads: (a.heads / 2).max(1),
        ffn_hidden: a.d_model * 2,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
    };
    mcfg.validate()?;
    // Saved so `lm-cli export` rebuilds the exact architecture the weights belong to.
    std::fs::write(
        a.out.join("model_config.json"),
        serde_json::to_string_pretty(&mcfg)?,
    )?;

    let dev = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let total = a.steps.max(1);
    let schedule = LrSchedule {
        max_lr: a.lr,
        min_lr: a.lr * 0.1,
        warmup_steps: (total / 10).max(1),
        total_steps: total,
    };
    let mut trainer = Trainer::new(&mcfg, &dev, schedule, 0.01, 1.0, cfg.seed)?;
    let tensors: Vec<(Tensor, Tensor, Tensor)> = train_batches
        .iter()
        .map(|b| to_tensors(b, &dev))
        .collect::<Result<_>>()?;
    if tensors.is_empty() {
        bail!("no training batches");
    }

    let mut first = f64::NAN;
    let mut last = f64::NAN;
    for step in 0..total {
        let b = &tensors[step % tensors.len()];
        let s = trainer.train_step(&b.0, &b.1, &b.2)?;
        if step == 0 {
            first = s.loss;
        }
        last = s.loss;
        if step % 10 == 0 || step + 1 == total {
            println!(
                "step {}/{} loss={:.4} grad_norm={:.4} lr={:.2e}",
                step + 1,
                total,
                s.loss,
                s.grad_norm,
                s.lr
            );
        }
    }

    let mut val_loss = f64::NAN;
    if let Some(vb) = val_batches.first() {
        let (x, t, m) = to_tensors(vb, &dev)?;
        val_loss = trainer.eval_loss(&x, &t, &m)?;
    }
    let weights = a.out.join("weights.safetensors");
    trainer.save_weights(&weights)?;
    let metrics = serde_json::json!({
        "mode": a.mode,
        "steps": total,
        "first_loss": first,
        "last_loss": last,
        "val_loss": val_loss,
        "train_batches": train_batches.len(),
        "vocab": tok.vocab_size(),
        "device": format!("{dev:?}"),
        "weights": weights.display().to_string(),
    });
    std::fs::write(
        a.out.join("metrics.json"),
        serde_json::to_string_pretty(&metrics)?,
    )?;
    println!("metrics written; weights={}", weights.display());
    Ok(())
}

/// Deterministic SFT conversations for one split: the configured synthetic examples (train
/// or held-out templates) plus tool-use trajectories, all labelled synthetic in the source
/// generators. No network access happens anywhere in this path.
fn sft_conversations(cfg: &Config, held_out: bool) -> Result<Vec<Vec<lm_data::chat::Message>>> {
    use lm_data::chat::Message;
    let mut out: Vec<Vec<Message>> = Vec::new();
    let examples = if held_out {
        lm_data::prepare::generate_heldout(cfg)
    } else {
        lm_data::prepare::generate_synthetic(cfg)
    };
    for ex in examples {
        out.push(ex.messages);
    }
    let mut rng = lm_data::rng::SplitMix64::new(cfg.seed ^ 0x544f_4f4c);
    for _ in 0..(cfg.synthetic.arithmetic / 2).max(1) {
        out.push(lm_data::tooluse::generate(&mut rng)?.messages);
    }
    Ok(out)
}

fn to_tensors(b: &Batch, dev: &Device) -> Result<(Tensor, Tensor, Tensor)> {
    let x = Tensor::new(b.tokens.clone(), dev)?.to_dtype(DType::U32)?;
    let t = Tensor::new(b.targets.clone(), dev)?.to_dtype(DType::U32)?;
    let m = Tensor::new(b.mask.clone(), dev)?;
    Ok((x, t, m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_rejects_unknown_flags() {
        let argv: Vec<String> = ["--bogus", "1"].iter().map(|s| s.to_string()).collect();
        assert!(parse_args(&argv).is_err());
    }

    #[test]
    fn parse_args_requires_config_and_out() {
        let argv: Vec<String> = ["--inputs", "a.txt"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(parse_args(&argv).is_err());
    }

    #[test]
    fn parse_args_reads_full_flag_set() {
        let argv: Vec<String> = [
            "--config", "c.json", "--inputs", "a.txt", "b.txt", "--out", "o", "--steps", "7",
            "--batch", "3", "--seq", "16", "--vocab", "128", "--mode", "sft", "--lr", "0.002",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let a = parse_args(&argv).unwrap();
        assert_eq!(a.steps, 7);
        assert_eq!(a.batch, 3);
        assert_eq!(a.seq, 16);
        assert_eq!(a.vocab, 128);
        assert_eq!(a.mode, "sft");
        assert_eq!(a.inputs.len(), 2);
    }

    #[test]
    fn to_tensors_shapes_match_batch() {
        let dev = Device::Cpu;
        let b = Batch {
            tokens: vec![vec![1u32, 2, 3], vec![4, 5, 6]],
            targets: vec![vec![2u32, 3, 4], vec![5, 6, 7]],
            mask: vec![vec![1f32, 0.0, 1.0], vec![0.0, 1.0, 1.0]],
        };
        let (x, t, m) = to_tensors(&b, &dev).unwrap();
        assert_eq!(x.dims(), &[2, 3]);
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(m.dims(), &[2, 3]);
    }
}
