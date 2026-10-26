//! `lm-cli clean`: turn raw corpora into a cleaned, deduplicated, loader-ready corpus.
//!
//! Writes one `<source>.jsonl` per input path (the loader already understands `.jsonl` with a
//! `text` field), plus `report.json` (what was kept, dropped and why) and `manifest.json`
//! (policy, output files, and the license/origin metadata carried over from `--config`).
//!
//! Nothing under the input paths is modified. The report and manifest contain no timestamps, so
//! running the command twice over the same inputs produces byte-identical output: cleaning is
//! reproducible and idempotent.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use lm_data::clean::{Deduper, Policy, Rejection};
use lm_data::config::Config;
use lm_data::corpus;
use serde_json::json;

pub struct Options {
    pub inputs: Vec<PathBuf>,
    pub out: PathBuf,
    /// Optional pipeline config; its `sources` metadata is copied into the manifest.
    pub config: Option<PathBuf>,
    pub policy: Policy,
    /// Paragraph deduplication is skipped for paragraphs shorter than this many characters.
    pub paragraph_dedup_min: usize,
    pub dedup_paragraphs: bool,
}

#[derive(Default)]
struct SourceReport {
    input_documents: usize,
    kept_documents: usize,
    duplicate_documents: usize,
    empty: usize,
    too_short: usize,
    not_prose: usize,
    input_chars: usize,
    output_chars: usize,
    paragraphs_removed: usize,
    markup_chars_removed: usize,
    entities_decoded: usize,
    urls_removed: usize,
    chrome_lines_dropped: usize,
    repeated_lines_collapsed: usize,
    repeated_words_collapsed: usize,
    char_runs_collapsed: usize,
    control_chars_removed: usize,
    output_file: Option<String>,
}

impl SourceReport {
    fn rejected(&self) -> usize {
        self.empty + self.too_short + self.not_prose
    }

    fn to_json(&self) -> serde_json::Value {
        json!({
            "input_documents": self.input_documents,
            "kept_documents": self.kept_documents,
            "rejected": {
                "total": self.rejected(),
                "empty": self.empty,
                "too_short": self.too_short,
                "not_prose": self.not_prose,
                "duplicate_document": self.duplicate_documents,
            },
            "paragraphs_removed": self.paragraphs_removed,
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "output_chars_pct_of_input": pct(self.output_chars, self.input_chars),
            "est_tokens_chars_div_4": self.output_chars / 4,
            "cleaning": {
                "markup_chars_removed": self.markup_chars_removed,
                "entities_decoded": self.entities_decoded,
                "urls_removed": self.urls_removed,
                "chrome_lines_dropped": self.chrome_lines_dropped,
                "repeated_lines_collapsed": self.repeated_lines_collapsed,
                "repeated_words_collapsed": self.repeated_words_collapsed,
                "char_runs_collapsed": self.char_runs_collapsed,
                "control_chars_removed": self.control_chars_removed,
            },
            "output_file": self.output_file,
        })
    }
}

fn pct(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        return 0.0;
    }
    (1000.0 * part as f64 / whole as f64).round() / 10.0
}

pub fn run(opts: Options) -> Result<()> {
    if opts.inputs.is_empty() {
        bail!("--inputs needs at least one path");
    }
    // Refuse to write into an input directory: a cleaned corpus must never mix with its source.
    let out = normalize(&opts.out);
    for input in &opts.inputs {
        let src = normalize(input);
        if out.starts_with(&src) || src.starts_with(&out) {
            bail!(
                "output {} overlaps input {}; write the cleaned corpus to a separate directory",
                opts.out.display(),
                input.display()
            );
        }
    }
    std::fs::create_dir_all(&opts.out)
        .with_context(|| format!("creating {}", opts.out.display()))?;

    let mut deduper = Deduper::new();
    let mut sources: BTreeMap<String, SourceReport> = BTreeMap::new();
    let mut used_names: BTreeMap<String, usize> = BTreeMap::new();

    for input in &opts.inputs {
        let name = unique_name(source_name(input), &mut used_names);
        let docs = corpus::load_path(input)
            .with_context(|| format!("loading corpus {}", input.display()))?;
        let mut report = SourceReport {
            input_documents: docs.len(),
            ..Default::default()
        };
        // Paragraph dedup counters are cumulative across sources, so this source's contribution
        // is the difference from where it started.
        let paragraphs_before = deduper.paragraphs_removed;
        let mut buf = String::new();
        let mut kept = 0usize;
        for doc in &docs {
            report.input_chars += doc.text.chars().count();
            match lm_data::clean::clean(&doc.text, &opts.policy) {
                Err(Rejection::Empty) => report.empty += 1,
                Err(Rejection::TooShort { .. }) => report.too_short += 1,
                Err(Rejection::NotProse { .. }) => report.not_prose += 1,
                Ok(cleaned) => {
                    if !deduper.keep_document(&cleaned.text) {
                        report.duplicate_documents += 1;
                        continue;
                    }
                    let text = if opts.dedup_paragraphs {
                        deduper.dedupe_paragraphs(&cleaned.text, opts.paragraph_dedup_min)
                    } else {
                        cleaned.text
                    };
                    if text.trim().is_empty() {
                        report.empty += 1;
                        continue;
                    }
                    report.output_chars += text.chars().count();
                    report.markup_chars_removed += cleaned.stats.markup_chars_removed;
                    report.entities_decoded += cleaned.stats.entities_decoded;
                    report.urls_removed += cleaned.stats.urls_removed;
                    report.chrome_lines_dropped += cleaned.stats.chrome_lines_dropped;
                    report.repeated_lines_collapsed += cleaned.stats.repeated_lines_collapsed;
                    report.repeated_words_collapsed += cleaned.stats.repeated_words_collapsed;
                    report.char_runs_collapsed += cleaned.stats.char_runs_collapsed;
                    report.control_chars_removed += cleaned.stats.control_chars_removed;
                    buf.push_str(&serde_json::to_string(&json!({ "text": text }))?);
                    buf.push('\n');
                    kept += 1;
                }
            }
        }
        report.kept_documents = kept;
        if kept > 0 {
            let file = format!("{name}.jsonl");
            std::fs::write(opts.out.join(&file), &buf)
                .with_context(|| format!("writing {file}"))?;
            report.output_file = Some(file);
        } else {
            eprintln!("warning: {name} produced no documents; nothing written");
        }
        report.paragraphs_removed = deduper.paragraphs_removed - paragraphs_before;
        sources.insert(name, report);
    }

    write_sidecars(&opts, &sources)?;
    print_summary(&opts.out, &sources);
    Ok(())
}

fn write_sidecars(opts: &Options, sources: &BTreeMap<String, SourceReport>) -> Result<()> {
    let policy = json!({
        "min_chars": opts.policy.min_chars,
        "strip_urls": opts.policy.strip_urls,
        "collapse_repeats": opts.policy.collapse_repeats,
        "min_alpha_ratio_pct": opts.policy.min_alpha_ratio_pct,
        "paragraph_dedup": opts.dedup_paragraphs,
        "paragraph_dedup_min_chars": opts.paragraph_dedup_min,
    });
    let mut totals = json!({
        "input_documents": 0,
        "kept_documents": 0,
        "rejected": 0,
        "input_chars": 0,
        "output_chars": 0,
        "est_tokens_chars_div_4": 0,
    });
    for report in sources.values() {
        totals["input_documents"] = json!(
            totals["input_documents"].as_u64().unwrap_or(0) as usize + report.input_documents
        );
        totals["kept_documents"] =
            json!(totals["kept_documents"].as_u64().unwrap_or(0) as usize + report.kept_documents);
        totals["rejected"] = json!(
            totals["rejected"].as_u64().unwrap_or(0) as usize
                + report.rejected()
                + report.duplicate_documents
        );
        totals["input_chars"] =
            json!(totals["input_chars"].as_u64().unwrap_or(0) as usize + report.input_chars);
        totals["output_chars"] =
            json!(totals["output_chars"].as_u64().unwrap_or(0) as usize + report.output_chars);
        totals["est_tokens_chars_div_4"] = json!(
            totals["est_tokens_chars_div_4"].as_u64().unwrap_or(0) as usize
                + report.output_chars / 4
        );
    }
    totals["output_chars_pct_of_input"] = json!(pct(
        totals["output_chars"].as_u64().unwrap_or(0) as usize,
        totals["input_chars"].as_u64().unwrap_or(0) as usize
    ));

    let report = json!({
        "policy": policy,
        "sources": sources.iter().map(|(k, v)| (k.clone(), v.to_json())).collect::<BTreeMap<_, _>>(),
        "totals": totals,
    });
    std::fs::write(
        opts.out.join("report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;

    let declared_sources = match &opts.config {
        Some(path) => {
            let cfg = Config::load(path)
                .with_context(|| format!("reading source metadata from {}", path.display()))?;
            serde_json::to_value(&cfg.sources)?
        }
        None => json!([]),
    };
    let manifest = json!({
        "cleaned_by": "lm-cli clean",
        "cleaning_policy": policy,
        "inputs": opts.inputs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "outputs": sources
            .iter()
            .filter_map(|(k, v)| v.output_file.as_ref().map(|f| (k.clone(), f.clone())))
            .collect::<BTreeMap<_, _>>(),
        "declared_sources": declared_sources,
    });
    std::fs::write(
        opts.out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}

fn print_summary(out: &Path, sources: &BTreeMap<String, SourceReport>) {
    println!(
        "{:<20} {:>7} {:>7} {:>14} {:>14} {:>6}  rejected",
        "source", "docs", "kept", "chars in", "chars out", "kept%"
    );
    for (name, r) in sources {
        println!(
            "{:<20} {:>7} {:>7} {:>14} {:>14} {:>5.1}%  short={} notprose={} empty={} dup={}",
            name,
            r.input_documents,
            r.kept_documents,
            r.input_chars,
            r.output_chars,
            pct(r.output_chars, r.input_chars),
            r.too_short,
            r.not_prose,
            r.empty,
            r.duplicate_documents
        );
    }
    println!("\ncleaned corpus -> {}", out.display());
}

fn source_name(path: &Path) -> String {
    let raw = path
        .file_stem()
        .or_else(|| path.file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "source".to_string());
    raw.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn unique_name(base: String, used: &mut BTreeMap<String, usize>) -> String {
    let n = used.entry(base.clone()).or_insert(0);
    *n += 1;
    if *n == 1 { base } else { format!("{base}-{n}") }
}

/// Compare paths lexically after removing `.` and `..` segments; no filesystem access needed.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        use std::path::Component;
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}
