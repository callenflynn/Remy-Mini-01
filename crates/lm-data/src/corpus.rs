//! Corpus loading for local `.txt`, `.md` and `.jsonl` files, either one at a time or by
//! walking a directory.
//!
//! - `.txt` / `.md`: the whole file is one document.
//! - `.jsonl`: one JSON object per line with a string `text` field. Other fields are ignored.
//!   Blank lines are skipped. A malformed line is an error naming the file and line number.

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Extensions the loader understands. Anything else in a directory is ignored; an explicit
/// path with any other extension is an error.
pub const SUPPORTED_EXTENSIONS: [&str; 3] = ["txt", "md", "jsonl"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    /// Stable id: `<path as given>#<index>`, so splits do not depend on file order alone.
    pub id: String,
    pub text: String,
}

#[derive(Deserialize)]
struct JsonlLine {
    text: String,
}

/// Load documents from a path: a file is read directly, a directory is walked recursively.
pub fn load_path(path: &Path) -> Result<Vec<Document>> {
    if path.is_dir() {
        let mut files = Vec::new();
        collect_supported(path, &mut files)?;
        if files.is_empty() {
            bail!(
                "directory {} contains no .txt, .md or .jsonl files",
                path.display()
            );
        }
        // Sorted so a directory input yields the same document order every run.
        files.sort();
        let mut docs = Vec::new();
        for f in &files {
            docs.extend(load_file(f)?);
        }
        return Ok(docs);
    }
    load_file(path)
}

fn collect_supported(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("reading corpus directory {}", dir.display()))?;
    for entry in entries {
        let p = entry?.path();
        if p.is_dir() {
            collect_supported(&p, out)?;
        } else if is_supported(&p) {
            out.push(p);
        }
    }
    Ok(())
}

fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| SUPPORTED_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Load documents from one file, dispatching on its extension.
pub fn load_file(path: &Path) -> Result<Vec<Document>> {
    let stem = path.to_string_lossy().into_owned();
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading corpus file {}", path.display()))?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("txt") | Some("md") => {
            if raw.trim().is_empty() {
                bail!("corpus file {} is empty", path.display());
            }
            Ok(vec![Document {
                id: format!("{stem}#0"),
                text: raw,
            }])
        }
        Some("jsonl") => parse_jsonl(&stem, &raw, path),
        _ => bail!(
            "unsupported corpus format for {}; use .txt, .md or .jsonl",
            path.display()
        ),
    }
}

/// Parse JSONL text. `name` is used to build document ids.
pub fn parse_jsonl(name: &str, raw: &str, path: &Path) -> Result<Vec<Document>> {
    let mut docs = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: JsonlLine = serde_json::from_str(line).map_err(|e| {
            anyhow!(
                "{}: line {} is not valid JSONL with a string \"text\" field: {e}",
                path.display(),
                lineno + 1
            )
        })?;
        docs.push(Document {
            id: format!("{name}#{}", docs.len()),
            text: parsed.text,
        });
    }
    if docs.is_empty() {
        bail!("corpus file {} contains no records", path.display());
    }
    Ok(docs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_skips_blank_lines_and_reads_text() {
        let raw = "{\"text\":\"a\"}\n\n{\"text\":\"b\",\"x\":1}\n";
        let docs = parse_jsonl("f.jsonl", raw, Path::new("f.jsonl")).unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[1].text, "b");
        assert_eq!(docs[1].id, "f.jsonl#1");
    }

    #[test]
    fn malformed_jsonl_names_the_line() {
        let raw = "{\"text\":\"a\"}\n{not json}\n";
        let err = parse_jsonl("f.jsonl", raw, Path::new("f.jsonl"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("line 2"), "{err}");
    }

    #[test]
    fn jsonl_without_text_field_is_rejected() {
        let err = parse_jsonl("f", "{\"body\":\"a\"}\n", Path::new("f.jsonl"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("line 1"), "{err}");
    }

    #[test]
    fn empty_jsonl_is_rejected() {
        assert!(parse_jsonl("f", "\n  \n", Path::new("f.jsonl")).is_err());
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("lm-corpus-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn markdown_file_is_one_document() {
        let d = temp_dir("md");
        let p = d.join("notes.md");
        std::fs::write(&p, "# Title\n\nBody text.").unwrap();
        let docs = load_file(&p).unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].text.contains("Body text."));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn directory_is_walked_sorted_and_filtered() {
        let d = temp_dir("dir");
        std::fs::create_dir_all(d.join("nested")).unwrap();
        std::fs::write(d.join("b.md"), "second").unwrap();
        std::fs::write(d.join("a.txt"), "first").unwrap();
        std::fs::write(d.join("nested/c.jsonl"), "{\"text\":\"third\"}\n").unwrap();
        std::fs::write(d.join("skip.bin"), "ignored").unwrap();
        let docs = load_path(&d).unwrap();
        let texts: Vec<&str> = docs.iter().map(|x| x.text.as_str()).collect();
        assert_eq!(texts, vec!["first", "second", "third"]);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn directory_without_supported_files_is_an_error() {
        let d = temp_dir("empty");
        std::fs::write(d.join("x.json"), "{}").unwrap();
        let err = load_path(&d).unwrap_err().to_string();
        assert!(err.contains("no .txt, .md or .jsonl"), "{err}");
        std::fs::remove_dir_all(&d).ok();
    }
}
