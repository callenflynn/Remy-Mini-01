//! `lm-cli`: prepare training data, run training, and sample from the result.
//!
//! Subcommands:
//! - `prepare --config <file> --inputs <files...> --out <dir>`: run the `lm-data` pipeline and
//!   write `records.jsonl` + `manifest.json`.
//! - `train --config <file> --inputs <paths...> --out <dir> [--steps N] [--batch N] [--seq N]
//!   [--mode pretrain|sft] [--vocab N] [--lr X] [--d-model N] [--layers N] [--heads N]`:
//!   prepare, train a tokenizer on the train split, batch with `lm-data::loader`, train with
//!   `lm-train::Trainer`, save weights and a metrics summary.
//! - `export --weights <f> --model-config <f> --tokenizer <f> --out <dir>`: write an HF
//!   `LlamaForCausalLM` directory.
//! - `generate --run <dir> --prompt <text> [--max-tokens N] [--temperature X] [--top-k N]
//!   [--seed N]`: sample a completion from a finished run.
//! - `harness --run <dir> --prompt <text> [--max-steps N] ...`: drive the tool-call harness
//!   with the trained model, using offline stand-ins for the tools.
//! - `clean --inputs <paths...> --out <dir> [--config <file>] [--min-chars N]`: clean the raw
//!   corpora into a loader-ready directory with a report and a provenance manifest.
//!
//! `--inputs` takes files or directories; directories are walked for `.txt`, `.md`, `.jsonl`.
//!
//! Deterministic where the underlying pieces are: fixed config seed gives identical data.

mod cleanup;
mod generate;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use lm_data::config::Config;
use lm_data::loader::{Batch, PretrainData, SftData};
use lm_infer::ToolExecutor;
use lm_infer::protocol::ToolCall;
use lm_model::ModelConfig;
use lm_tokenizer::LmTokenizer;
use lm_train::{LrSchedule, Trainer};

use generate::{Checkpoint, ModelGenerator, Sampling, default_device};

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
        "" | "help" | "-h" | "--help" => {
            print_usage();
            Ok(())
        }
        "prepare" => run_prepare(parse_args(&argv[1..])?),
        "train" => run_train(parse_args(&argv[1..])?),
        "export" => run_export(&argv[1..]),
        "generate" => run_generate(&argv[1..]),
        "harness" => run_harness(&argv[1..]),
        "clean" => cleanup::run(parse_clean_options(&argv[1..])?),
        "tokenize" => run_tokenize(&argv[1..]),
        "eval" => run_eval(&argv[1..]),
        other => bail!("unknown subcommand '{other}'; run `lm-cli --help`"),
    }
}

fn print_usage() {
    print!(
        "lm-cli — prepare data, train a small decoder-only LM, and sample from it

Subcommands:
  prepare  --config <file> --inputs <paths...> --out <dir>
           run the lm-data pipeline, write records.jsonl + manifest.json
  train    --config <file> --inputs <paths...> --out <dir> [--steps N] [--batch N]
           [--seq N] [--mode pretrain|sft] [--vocab N] [--lr X] [--d-model N]
           [--layers N] [--heads N]
           write model_config.json, tokenizer.json, weights.safetensors, metrics.json
  generate --run <dir> --prompt <text> [--max-tokens N] [--temperature X] [--top-k N]
           [--seed N]
           print one sampled completion
  harness  --run <dir> --prompt <text> [--max-steps N] [--max-tokens N] [--temperature X]
           [--top-k N] [--seed N]
           run the tool-call loop with offline tool stubs
  clean    --inputs <paths...> --out <dir> [--config <file>] [--min-chars N]
           [--alpha-ratio PCT] [--keep-urls] [--no-dedup-paragraphs] [--paragraph-min N]
           strip markup and artifacts, dedupe, and write <source>.jsonl + report.json
  tokenize --run <dir> --input <file> [--limit N]
           tokens/char, tokens/word and markup-ish token share of one file
  eval     --run <dir> --config <file> --inputs <paths...> [--batches N]
           held-out loss of a finished run on the config's val split (default 20 batches)
  export   --weights <f> --model-config <f> --tokenizer <f> --out <dir>
           write an HF LlamaForCausalLM directory (llama.cpp / transformers shape)

--temperature 0 is greedy; --top-k 0 samples the full distribution.
--inputs takes files or directories (.txt, .md, .jsonl).
"
    );
}

/// `--flag value` pairs, the shape every non-training subcommand takes.
struct Flags(HashMap<String, String>);

impl Flags {
    fn parse(argv: &[String]) -> Result<Self> {
        let mut map = HashMap::new();
        let mut i = 0;
        while i < argv.len() {
            let key = argv[i]
                .strip_prefix("--")
                .ok_or_else(|| anyhow::anyhow!("expected --flag, got {}", argv[i]))?;
            let value = argv
                .get(i + 1)
                .ok_or_else(|| anyhow::anyhow!("missing value for --{key}"))?;
            map.insert(key.to_string(), value.clone());
            i += 2;
        }
        Ok(Self(map))
    }

    fn need(&self, k: &str) -> Result<&str> {
        self.0
            .get(k)
            .map(String::as_str)
            .ok_or_else(|| anyhow::anyhow!("--{k} is required"))
    }

    fn opt(&self, k: &str) -> Option<&str> {
        self.0.get(k).map(String::as_str)
    }

    fn number<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T> {
        match self.opt(k) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| anyhow::anyhow!("--{k} must be a number, got '{v}'")),
        }
    }

    /// Reject anything not listed, so a typo cannot silently change nothing.
    fn ensure_only(&self, allowed: &[&str]) -> Result<()> {
        for key in self.0.keys() {
            if !allowed.contains(&key.as_str()) {
                bail!("unknown flag '--{key}'");
            }
        }
        Ok(())
    }
}

const SAMPLING_FLAGS: [&str; 4] = ["max-tokens", "temperature", "top-k", "seed"];

fn sampling_from(f: &Flags) -> Result<Sampling> {
    let d = Sampling::default();
    Ok(Sampling {
        max_tokens: f.number("max-tokens", d.max_tokens)?,
        temperature: f.number("temperature", d.temperature)?,
        top_k: f.number("top-k", d.top_k)?,
        seed: f.number("seed", d.seed)?,
    })
}

/// `lm-cli export --weights w.safetensors --model-config model_config.json
///   --tokenizer tokenizer.json --out <dir>`: write an HF LlamaForCausalLM directory that
/// Soup (and transformers) can load directly.
fn run_export(argv: &[String]) -> Result<()> {
    let f = Flags::parse(argv)?;
    f.ensure_only(&["weights", "model-config", "tokenizer", "out"])?;
    let weights = PathBuf::from(f.need("weights")?);
    let model_config = PathBuf::from(f.need("model-config")?);
    let tokenizer = PathBuf::from(f.need("tokenizer")?);
    let out = PathBuf::from(f.need("out")?);

    let mcfg: ModelConfig = serde_json::from_str(&std::fs::read_to_string(&model_config)?)?;
    let tok = LmTokenizer::load(&tokenizer)?;
    lm_train::export::export_llama(&weights, &out, &mcfg, &tok, &tokenizer)?;
    println!("exported HF Llama package -> {}", out.display());
    Ok(())
}

/// Parse `lm-cli clean` flags. Unlike the sampling subcommands this takes a list of inputs and
/// two valueless switches, so it does not go through `Flags`.
fn parse_clean_options(argv: &[String]) -> Result<cleanup::Options> {
    let mut inputs = Vec::new();
    let mut out = None;
    let mut config = None;
    let mut policy = lm_data::clean::Policy::default();
    let mut dedup_paragraphs = true;
    let mut paragraph_dedup_min = 200usize;
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        let take = |i: &mut usize| -> Result<String> {
            *i += 1;
            argv.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))
        };
        match flag {
            "--inputs" => {
                i += 1;
                while i < argv.len() && !argv[i].starts_with("--") {
                    inputs.push(PathBuf::from(&argv[i]));
                    i += 1;
                }
                continue;
            }
            "--out" => out = Some(PathBuf::from(take(&mut i)?)),
            "--config" => config = Some(PathBuf::from(take(&mut i)?)),
            "--min-chars" => policy.min_chars = take(&mut i)?.parse()?,
            "--alpha-ratio" => policy.min_alpha_ratio_pct = take(&mut i)?.parse()?,
            "--paragraph-min" => paragraph_dedup_min = take(&mut i)?.parse()?,
            "--keep-urls" => policy.strip_urls = false,
            "--no-dedup-paragraphs" => dedup_paragraphs = false,
            other => bail!("unknown flag '{other}' for clean"),
        }
        i += 1;
    }
    Ok(cleanup::Options {
        inputs,
        out: out.ok_or_else(|| anyhow::anyhow!("--out is required"))?,
        config,
        policy,
        paragraph_dedup_min,
        dedup_paragraphs,
    })
}

/// `lm-cli tokenize --run <dir> --input <file>`: how well the tokenizer fits a text. A high
/// tokens/word count means words are being shredded, and a high markup-ish share means the
/// vocabulary is spending capacity on web artifacts.
/// Indices spread evenly over `0..len`, so a sample covers the whole range instead of its start.
/// Taking `first()` made validation look 2.4 nats better than training: the first val batch of a
/// concatenated corpus is whatever source was listed first.
fn spread_indices(len: usize, want: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    let n = want.clamp(1, len);
    (0..n).map(|i| i * len / n).collect()
}

/// `lm-cli eval --run <dir> --config <file> --inputs <paths...> [--batches N]`: held-out loss of
/// a finished run. The split is rebuilt from the same config and seed `train` used, so the
/// documents scored here are exactly the ones the model never trained on.
fn run_eval(argv: &[String]) -> Result<()> {
    let mut run: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut want = 20usize;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--run" => {
                i += 1;
                run = Some(PathBuf::from(
                    argv.get(i).context("--run needs a value")?.clone(),
                ));
            }
            "--config" => {
                i += 1;
                config = Some(PathBuf::from(
                    argv.get(i).context("--config needs a value")?.clone(),
                ));
            }
            "--inputs" => {
                i += 1;
                while i < argv.len() && !argv[i].starts_with("--") {
                    inputs.push(PathBuf::from(&argv[i]));
                    i += 1;
                }
                continue;
            }
            "--batches" => {
                i += 1;
                want = argv
                    .get(i)
                    .context("--batches needs a value")?
                    .parse()
                    .context("--batches must be a number")?;
            }
            other => bail!("unknown argument '{other}'"),
        }
        i += 1;
    }
    let run = run.context("--run is required")?;
    let config = config.context("--config is required")?;
    if inputs.is_empty() {
        bail!("--inputs needs at least one path");
    }

    let device = default_device();
    let ckpt = Checkpoint::load(&run, &device)?;
    let cfg = Config::load(&config)?;
    let (records, _) = lm_data::prepare::prepare(&cfg, &inputs)?;
    let val: Vec<String> = records
        .iter()
        .filter(|r| r.split == "val")
        .map(|r| r.text.clone())
        .collect();
    if val.is_empty() {
        bail!("the val split is empty; nothing to evaluate on");
    }
    let batches = PretrainData::new(&ckpt.tokenizer, &val, ckpt.config.context_len)?.batches(1)?;
    let picked = spread_indices(batches.len(), want);
    let mut total = 0.0;
    for &b in &picked {
        let (x, t, m) = to_tensors(&batches[b], &device)?;
        total += ckpt.batch_loss(&x, &t, &m)?;
    }
    let loss = total / picked.len() as f64;
    println!(
        "run={} val_docs={} val_batches={} evaluated={} val_loss={loss:.4} ppl={:.2}",
        run.display(),
        val.len(),
        batches.len(),
        picked.len(),
        loss.exp()
    );
    Ok(())
}

fn run_tokenize(argv: &[String]) -> Result<()> {
    let f = Flags::parse(argv)?;
    f.ensure_only(&["run", "input", "limit"])?;
    let run = PathBuf::from(f.need("run")?);
    let input = PathBuf::from(f.need("input")?);
    let limit = f.number("limit", 200_000usize)?;

    let tok = LmTokenizer::load(&run.join("tokenizer.json"))?;
    // A `.jsonl` corpus is measured through the loader so JSON escaping is not counted as text.
    let source = if input.extension().and_then(|e| e.to_str()) == Some("jsonl") {
        lm_data::corpus::load_file(&input)?
            .iter()
            .map(|d| d.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        std::fs::read_to_string(&input).with_context(|| format!("reading {}", input.display()))?
    };
    let text: String = source.chars().take(limit).collect();
    let ids = tok.encode(&text)?;
    let words = text.split_whitespace().count().max(1);

    let mut counts: HashMap<u32, usize> = HashMap::new();
    let mut markup = 0usize;
    for id in &ids {
        *counts.entry(*id).or_default() += 1;
        let piece = tok.decode(&[*id], false).unwrap_or_default();
        if [
            "<", ">", "&", "=", "quot", "span", "div", "href", "mw-", "\\",
        ]
        .iter()
        .any(|needle| piece.contains(needle))
        {
            markup += 1;
        }
    }
    let mut top: Vec<(u32, usize)> = counts.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let top_str: Vec<String> = top
        .iter()
        .take(8)
        .map(|(id, n)| {
            let piece = tok.decode(&[*id], false).unwrap_or_default();
            format!("{}×{n}", piece.escape_debug())
        })
        .collect();

    println!("{}", input.display());
    println!(
        "  chars={} words={} tokens={} tokens/char={:.3} tokens/word={:.2} markup-ish={:.1}%",
        text.chars().count(),
        words,
        ids.len(),
        ids.len() as f64 / text.chars().count().max(1) as f64,
        ids.len() as f64 / words as f64,
        100.0 * markup as f64 / ids.len().max(1) as f64
    );
    println!("  most frequent: {}", top_str.join(" "));
    Ok(())
}

/// `lm-cli generate --run <dir> --prompt <text> [...]`: one sampled completion.
fn run_generate(argv: &[String]) -> Result<()> {
    let f = Flags::parse(argv)?;
    let mut allowed = vec!["run", "prompt"];
    allowed.extend(SAMPLING_FLAGS);
    f.ensure_only(&allowed)?;
    let run = PathBuf::from(f.need("run")?);
    let prompt = f.need("prompt")?;
    let sampling = sampling_from(&f)?;

    let device = default_device();
    let ckpt = Checkpoint::load(&run, &device)?;
    println!(
        "loaded {} (params={} vocab={} ctx={} device={device:?})",
        run.display(),
        ckpt.param_count(),
        ckpt.config.vocab_size,
        ckpt.config.context_len
    );
    let out = ckpt.complete(prompt, sampling)?;
    println!("--- completion ---");
    print!("{out}");
    if !out.ends_with('\n') {
        println!();
    }
    Ok(())
}

/// Offline stand-ins for the harness tools: no network, no filesystem, no shell. They return
/// fixed text so the model/harness loop can be exercised without side effects.
#[derive(Default)]
struct OfflineTools {
    calls: Vec<String>,
}

impl ToolExecutor for OfflineTools {
    fn execute(&mut self, call: &ToolCall) -> Result<String> {
        self.calls.push(call.name.clone());
        match call.name.as_str() {
            "web_search" => Ok(format!(
                "offline stub: no network. query was {}",
                call.arguments["query"]
            )),
            "fetch_page" => Ok(format!(
                "offline stub: no network. url was {}",
                call.arguments["url"]
            )),
            other => bail!("no offline implementation for '{other}'"),
        }
    }
}

/// `lm-cli harness --run <dir> --prompt <text> [--max-steps N] [...]`: run one conversation
/// through `lm_infer::run` with the trained model generating every turn.
fn run_harness(argv: &[String]) -> Result<()> {
    let f = Flags::parse(argv)?;
    let mut allowed = vec!["run", "prompt", "max-steps"];
    allowed.extend(SAMPLING_FLAGS);
    f.ensure_only(&allowed)?;
    let run = PathBuf::from(f.need("run")?);
    let prompt = f.need("prompt")?;
    let max_steps = f.number("max-steps", 3usize)?;
    let sampling = sampling_from(&f)?;

    let device = default_device();
    let ckpt = Checkpoint::load(&run, &device)?;
    let registry = lm_data::tooluse::tool_registry();
    let mut generator = ModelGenerator {
        checkpoint: &ckpt,
        sampling,
    };
    let mut executor = OfflineTools::default();
    let (transcript, executed) =
        lm_infer::run(&mut generator, &registry, &mut executor, prompt, max_steps)?;
    println!("{transcript}");
    println!("\n[tool calls executed: {executed}, max steps: {max_steps}]");
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
    // The stream is ordered by hash of record id, not by input order. Without this a run shorter
    // than one epoch only ever sees whichever sources were listed first, which is exactly how a
    // 2,000-step run turned into a Wikipedia-markup model while validation (drawn from every
    // source) stayed bad.
    let mut train_records: Vec<&lm_data::prepare::Record> =
        records.iter().filter(|r| r.split == "train").collect();
    let train_ids: Vec<String> = train_records.iter().map(|r| r.id.clone()).collect();
    let order = lm_data::split::stream_order(&train_ids, cfg.seed);
    train_records = order.iter().map(|&i| train_records[i]).collect();
    let train_texts: Vec<String> = train_records.iter().map(|r| r.text.clone()).collect();
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

    // Mean over batches spread across the val split, not one batch from its start.
    let picked = spread_indices(val_batches.len(), 20);
    let mut val_loss = f64::NAN;
    for &b in &picked {
        let (x, t, m) = to_tensors(&val_batches[b], &dev)?;
        val_loss = if val_loss.is_nan() {
            trainer.eval_loss(&x, &t, &m)?
        } else {
            val_loss + trainer.eval_loss(&x, &t, &m)?
        };
    }
    if !picked.is_empty() {
        val_loss /= picked.len() as f64;
    }
    let weights = a.out.join("weights.safetensors");
    trainer.save_weights(&weights)?;
    let metrics = serde_json::json!({
        "mode": a.mode,
        "steps": total,
        "first_loss": first,
        "last_loss": last,
        "val_loss": val_loss,
        "val_batches_total": val_batches.len(),
        "val_batches_evaluated": picked.len(),
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
    fn spread_indices_covers_the_whole_range() {
        assert_eq!(spread_indices(0, 20), Vec::<usize>::new());
        assert_eq!(spread_indices(3, 20), vec![0, 1, 2]);
        let picks = spread_indices(1_000, 20);
        assert_eq!(picks.len(), 20);
        assert_eq!(picks[0], 0);
        assert!(picks[19] > 900, "last pick must reach the end: {picks:?}");
        // Not just the first batches: consecutive picks are spread apart.
        assert!(picks[1] > 40, "{picks:?}");
        assert!(picks.windows(2).all(|w| w[0] < w[1]));
    }

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
