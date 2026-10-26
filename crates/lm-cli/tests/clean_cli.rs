//! End-to-end tests for `lm-cli clean`: the real binary, real files, real JSON out.
//!
//! These cover what unit tests cannot: that the command writes loader-ready JSONL, that its
//! report and manifest describe what happened, that a second run over the same inputs produces
//! byte-identical output, and that it refuses to write into its own input directory.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_lm-cli")
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lm-clean-cli-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&d).ok();
    std::fs::create_dir_all(&d).unwrap();
    d
}

const PROSE: &str = "The arctic tern migrates between hemispheres each year, following the \
summer sun across open water and coastal staging grounds where it feeds on small fish and \
invertebrates. Researchers tracking tagged birds recorded round trips of more than seventy \
thousand kilometres, among the longest migrations known for any animal.";

fn write_inputs(dir: &Path) {
    let html = dir.join("pages");
    std::fs::create_dir_all(&html).unwrap();
    std::fs::write(
        html.join("tern.md"),
        format!(
            "<div class=\"mw\"><p>{PROSE}</p><script>var x = 1;</script>\
             <span data-mw=\"{{\\\"parts\\\":[\\\"cite\\\"]}}\">Jump to content</span>\
             <p>See [the study](https://example.org/x) for figures.</p></div>"
        ),
    )
    .unwrap();

    // Two identical records plus one short fragment: dedup and length filters must apply.
    let jsonl = dir.join("notes.jsonl");
    std::fs::write(
        &jsonl,
        format!(
            "{}\n{}\n{}\n",
            serde_json::json!({ "text": PROSE }),
            serde_json::json!({ "text": PROSE }),
            serde_json::json!({ "text": "too short" })
        ),
    )
    .unwrap();
}

fn run_clean(inputs: &[&Path], out: &Path) -> std::process::Output {
    let mut cmd = Command::new(bin());
    cmd.arg("clean").arg("--inputs");
    for i in inputs {
        cmd.arg(i);
    }
    cmd.arg("--out").arg(out);
    cmd.output().expect("running lm-cli clean")
}

#[test]
fn clean_writes_loader_ready_jsonl_report_and_manifest() {
    let dir = temp_dir("basic");
    write_inputs(&dir);
    let out = dir.join("cleaned");
    let result = run_clean(&[&dir.join("pages"), &dir.join("notes.jsonl")], &out);
    assert!(
        result.status.success(),
        "clean failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    for name in ["pages.jsonl", "notes.jsonl", "report.json", "manifest.json"] {
        assert!(out.join(name).exists(), "{name} missing");
    }

    // The output must load with the same loader the trainer uses.
    let pages = out.join("pages.jsonl");
    let docs = lm_data::corpus::load_file(&pages).unwrap();
    assert_eq!(docs.len(), 1);
    let text = &docs[0].text;
    assert!(text.contains("arctic tern"), "{text}");
    assert!(!text.contains('<') && !text.contains("data-mw"), "{text}");
    assert!(!text.contains("https://"), "{text}");
    assert!(!text.contains("var x = 1"), "{text}");
    assert!(text.contains("See the study for figures."), "{text}");

    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("report.json")).unwrap()).unwrap();
    let notes = &report["sources"]["notes"];
    assert_eq!(notes["input_documents"], 3);
    assert_eq!(
        notes["kept_documents"], 1,
        "duplicate and short record dropped"
    );
    assert_eq!(notes["rejected"]["duplicate_document"], 1);
    assert_eq!(notes["rejected"]["too_short"], 1);
    assert!(
        report["sources"]["pages"]["cleaning"]["markup_chars_removed"]
            .as_u64()
            .unwrap()
            > 0
    );

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["cleaned_by"], "lm-cli clean");
    assert_eq!(manifest["outputs"]["pages"], "pages.jsonl");
    assert_eq!(
        manifest["cleaning_policy"]["min_chars"],
        lm_data::clean::Policy::default().min_chars
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn clean_is_idempotent_and_byte_identical_across_runs() {
    let dir = temp_dir("idempotent");
    write_inputs(&dir);
    let out = dir.join("cleaned");
    let inputs = [dir.join("pages"), dir.join("notes.jsonl")];
    let inputs: Vec<&Path> = inputs.iter().map(PathBuf::as_path).collect();

    assert!(run_clean(&inputs, &out).status.success());
    let first: Vec<Vec<u8>> = ["pages.jsonl", "notes.jsonl", "report.json", "manifest.json"]
        .iter()
        .map(|n| std::fs::read(out.join(n)).unwrap())
        .collect();

    // Same inputs again, same directory: every byte must match. No timestamps, no reordering.
    assert!(run_clean(&inputs, &out).status.success());
    let second: Vec<Vec<u8>> = ["pages.jsonl", "notes.jsonl", "report.json", "manifest.json"]
        .iter()
        .map(|n| std::fs::read(out.join(n)).unwrap())
        .collect();
    assert_eq!(first, second);

    // Cleaning the cleaned output must not keep peeling text away.
    let again = dir.join("cleaned-again");
    assert!(
        run_clean(&[&out.join("pages.jsonl")], &again)
            .status
            .success()
    );
    let once = std::fs::read_to_string(out.join("pages.jsonl")).unwrap();
    let twice = std::fs::read_to_string(again.join("pages.jsonl")).unwrap();
    let once: serde_json::Value = serde_json::from_str(once.trim()).unwrap();
    let twice: serde_json::Value = serde_json::from_str(twice.trim()).unwrap();
    assert_eq!(once["text"], twice["text"]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn clean_refuses_to_write_inside_an_input_directory() {
    let dir = temp_dir("overlap");
    write_inputs(&dir);
    let result = run_clean(&[&dir], &dir.join("cleaned"));
    assert!(!result.status.success());
    let err = String::from_utf8_lossy(&result.stderr);
    assert!(err.contains("overlaps"), "{err}");
    assert!(!dir.join("cleaned").exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn clean_reports_a_bad_flag_instead_of_ignoring_it() {
    let dir = temp_dir("badflag");
    write_inputs(&dir);
    let result = Command::new(bin())
        .args(["clean", "--inputs"])
        .arg(&dir)
        .args(["--out"])
        .arg(dir.join("out"))
        .arg("--min-charrs")
        .arg("10")
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unknown flag"));
    std::fs::remove_dir_all(&dir).ok();
}
