//! End-to-end data preparation: load local corpora, normalize and filter documents, generate
//! synthetic examples, split deterministically, and write a manifest.
//!
//! Synthetic records are tagged `origin: "synthetic"` and listed separately from external
//! sources in the manifest. Declared external sources must pass license validation before any
//! of their files are read.

use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::config::Config;
use crate::corpus::{self, Document};
use crate::rng::SplitMix64;
use crate::split::{self, Bucket, Ratios};
use crate::synthetic::{self, Example, SYNTHETIC_ORIGIN};

/// Normalization rules, applied in this order:
/// 1. Trim surrounding whitespace.
/// 2. Drop documents shorter than `min_chars` or longer than `max_chars` (no truncation).
/// 3. Drop exact duplicate texts (first occurrence kept).
pub fn normalize(
    docs: Vec<Document>,
    min_chars: usize,
    max_chars: usize,
) -> (Vec<Document>, NormalizeStats) {
    let mut stats = NormalizeStats::default();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for mut d in docs {
        stats.input += 1;
        d.text = d.text.trim().to_string();
        let len = d.text.chars().count();
        if len < min_chars {
            stats.too_short += 1;
        } else if len > max_chars {
            stats.too_long += 1;
        } else if !seen.insert(d.text.clone()) {
            stats.duplicate += 1;
        } else {
            stats.kept += 1;
            out.push(d);
        }
    }
    (out, stats)
}

#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct NormalizeStats {
    pub input: usize,
    pub kept: usize,
    pub too_short: usize,
    pub too_long: usize,
    pub duplicate: usize,
}

/// Serialized split record: one per external document or synthetic example.
#[derive(Debug, Clone, Serialize)]
pub struct Record {
    pub id: String,
    pub origin: String,
    pub family: String,
    pub split: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct Manifest {
    pub seed: u64,
    pub split_strategy: String,
    pub ratios: [f64; 3],
    pub normalize: crate::config::NormalizeConfig,
    pub sources: Vec<crate::license::SourceMeta>,
    pub normalize_stats: NormalizeStats,
    /// Exact-duplicate texts removed before splitting (cross-split leakage prevention).
    pub dedup_removed: usize,
    pub counts: BTreeMap<String, BTreeMap<String, usize>>,
    pub synthetic_counts: BTreeMap<String, usize>,
    pub records_file: String,
}

/// Load every path listed in `inputs`. Each input is a file or a directory (walked
/// recursively for `.txt`, `.md` and `.jsonl`).
pub fn load_inputs(inputs: &[PathBuf]) -> Result<Vec<Document>> {
    if inputs.is_empty() {
        bail!("no input files given");
    }
    let mut docs = Vec::new();
    for p in inputs {
        docs.extend(corpus::load_path(p)?);
    }
    Ok(docs)
}

/// Generate the configured number of examples per family. Training templates and held-out
/// templates are both generated; the held-out set is produced by the `held_out` generators.
pub fn generate_synthetic(cfg: &Config) -> Vec<Example> {
    let mut rng = SplitMix64::new(cfg.seed ^ 0x5157_4e54);
    let mut out = Vec::new();
    for _ in 0..cfg.synthetic.arithmetic {
        out.push(synthetic::arithmetic(&mut rng, false));
    }
    for _ in 0..cfg.synthetic.logic {
        out.push(synthetic::logic(&mut rng, false));
    }
    for _ in 0..cfg.synthetic.word_problems {
        out.push(synthetic::word_problem(&mut rng, false));
    }
    for _ in 0..cfg.synthetic.instructions {
        out.push(synthetic::instructions(&mut rng, false));
    }
    for _ in 0..cfg.synthetic.coding {
        out.push(synthetic::coding(&mut rng, false));
    }
    out
}

/// Held-out evaluation examples, generated from the `held_out` templates. They are kept out
/// of the training split by construction.
pub fn generate_heldout(cfg: &Config) -> Vec<Example> {
    let mut rng = SplitMix64::new(cfg.seed ^ 0x4845_4c44);
    let mut out = Vec::new();
    for _ in 0..cfg.synthetic.arithmetic {
        out.push(synthetic::arithmetic(&mut rng, true));
    }
    for _ in 0..cfg.synthetic.logic {
        out.push(synthetic::logic(&mut rng, true));
    }
    for _ in 0..cfg.synthetic.word_problems {
        out.push(synthetic::word_problem(&mut rng, true));
    }
    for _ in 0..cfg.synthetic.instructions {
        out.push(synthetic::instructions(&mut rng, true));
    }
    for _ in 0..cfg.synthetic.coding {
        out.push(synthetic::coding(&mut rng, true));
    }
    out
}

/// Run the whole pipeline. External documents and synthetic training examples are split by
/// hash of their id. Held-out synthetic examples always go to `test`.
///
/// After normalization, records are deduplicated globally by exact text (first occurrence in
/// generation order wins), so identical texts can never straddle splits: the split is by id,
/// and two ids carrying the same text would otherwise be assigned independently.
pub fn prepare(cfg: &Config, inputs: &[PathBuf]) -> Result<(Vec<Record>, Manifest)> {
    cfg.validate()?;
    let docs = load_inputs(inputs)?;
    let (docs, nstats) = normalize(docs, cfg.normalize.min_chars, cfg.normalize.max_chars);
    if docs.is_empty() && synthetic_total(cfg) == 0 {
        bail!(
            "no usable records: all documents were filtered and no synthetic examples are configured"
        );
    }

    let r = ratios(cfg);
    let mut records = Vec::new();

    let keys: Vec<String> = docs.iter().map(|d| d.id.clone()).collect();
    let buckets = if keys.is_empty() {
        Vec::new()
    } else {
        split::split_hash(&keys, cfg.seed, r)?
    };
    for (d, b) in docs.iter().zip(buckets.iter()) {
        records.push(Record {
            id: d.id.clone(),
            origin: "external".into(),
            family: "document".into(),
            split: bucket_name(*b).into(),
            text: d.text.clone(),
        });
    }

    let train_examples = generate_synthetic(cfg);
    let ex_keys: Vec<String> = (0..train_examples.len())
        .map(|i| format!("synthetic#{i}"))
        .collect();
    let tb = if ex_keys.is_empty() {
        Vec::new()
    } else {
        split::split_hash(&ex_keys, cfg.seed, r)?
    };
    for (i, (ex, b)) in train_examples.iter().zip(tb.iter()).enumerate() {
        records.push(synthetic_record(i, ex, bucket_name(*b)));
    }

    for (i, ex) in generate_heldout(cfg).iter().enumerate() {
        records.push(synthetic_record(1_000_000 + i, ex, "test"));
    }

    let dedup_removed = dedupe_by_text(&mut records);

    let manifest = build_manifest(cfg, nstats, dedup_removed, &records);
    Ok((records, manifest))
}

/// Remove records whose exact text appeared earlier (generation order: external documents,
/// then synthetic training examples, then held-out). Returns how many were removed. This is
/// what keeps identical texts out of two different splits.
pub fn dedupe_by_text(records: &mut Vec<Record>) -> usize {
    let mut seen = std::collections::HashSet::new();
    let before = records.len();
    records.retain(|r| seen.insert(r.text.clone()));
    before - records.len()
}

fn synthetic_total(cfg: &Config) -> usize {
    cfg.synthetic.arithmetic
        + cfg.synthetic.logic
        + cfg.synthetic.word_problems
        + cfg.synthetic.instructions
        + cfg.synthetic.coding
}

fn synthetic_record(i: usize, ex: &Example, split: &str) -> Record {
    let text = ex
        .messages
        .iter()
        .map(|m| format!("{:?}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");
    Record {
        id: format!("synthetic#{i}"),
        origin: SYNTHETIC_ORIGIN.into(),
        family: ex.family.into(),
        split: split.into(),
        text,
    }
}

fn build_manifest(
    cfg: &Config,
    nstats: NormalizeStats,
    dedup_removed: usize,
    records: &[Record],
) -> Manifest {
    let mut counts: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut synthetic_counts: BTreeMap<String, usize> = BTreeMap::new();
    for r in records {
        *counts
            .entry(r.origin.clone())
            .or_default()
            .entry(r.split.clone())
            .or_default() += 1;
        if r.origin == SYNTHETIC_ORIGIN {
            *synthetic_counts.entry(r.family.clone()).or_default() += 1;
        }
    }
    Manifest {
        seed: cfg.seed,
        split_strategy: cfg.split.strategy.clone(),
        ratios: [cfg.split.train, cfg.split.val, cfg.split.test],
        normalize: cfg.normalize.clone(),
        sources: cfg.sources.clone(),
        normalize_stats: nstats,
        dedup_removed,
        counts,
        synthetic_counts,
        records_file: "records.jsonl".into(),
    }
}

pub fn ratios(cfg: &Config) -> Ratios {
    Ratios {
        train: cfg.split.train,
        val: cfg.split.val,
        test: cfg.split.test,
    }
}

pub fn bucket_name(b: Bucket) -> &'static str {
    match b {
        Bucket::Train => "train",
        Bucket::Val => "val",
        Bucket::Test => "test",
    }
}
