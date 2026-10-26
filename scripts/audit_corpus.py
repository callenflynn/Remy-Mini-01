#!/usr/bin/env python3
"""Audit the pretraining corpus: formats, sizes, duplicates, web-extraction artifacts.

Reads every file under the given directories, streams large files, and writes a JSON report.
Nothing is modified. Run:

    python3 scripts/audit_corpus.py data/speeches data/forums data/wikipedia-core data/misc \
        > reports/corpus-audit.json

The point is to measure, not to guess: which files the loader can even read, how much of each
source is HTML rather than prose, how much is duplicated, and which fragments repeat.
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

SUPPORTED = {".txt", ".md", ".jsonl"}

# A tag or entity is "markup evidence"; counting characters is more honest than counting
# occurrences because one giant <div> wrapper is one tag but thousands of characters.
TAG_RE = re.compile(r"<[a-zA-Z/!][^>]{0,200}>")
ENTITY_RE = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]{1,9});")
URL_RE = re.compile(r"https?://[^\s\"'<>]{4,}")
# MediaWiki/HTML leftovers that should never survive into prose.
SUSPICIOUS = [
    ("href=", re.compile(r"href=")),
    ("mw-attr", re.compile(r"\bmw-[a-zA-Z-]+|data-mw=")),
    ("div/span", re.compile(r"</?(div|span|table|tr|td|li|ul|p|a|h[1-6])\b")),
    ("template-json", re.compile(r"&quot;(target|wt|parts)&quot;")),
    ("edit-section", re.compile(r"\[edit\]|wikipedia|Wik")),
    ("entity-quot", re.compile(r"&quot;")),
    ("long-punct-run", re.compile(r"([^\w\s])\1{7,}")),
    ("repeated-word", re.compile(r"\b(\w+)(\s+\1){4,}\b", re.IGNORECASE)),
    ("cid-artifact", re.compile(r"\(cid:\d+\)|Ã[\x80-\xbf]|\ufffd")),
]


def fenced(text, limit=20_000_000):
    """Yield text in chunks so a 500 MB file never has to be held twice."""
    for i in range(0, len(text), limit):
        yield text[i : i + limit]


def blank():
    return {
        "files": 0,
        "bytes": 0,
        "supported_files": 0,
        "unsupported_files": [],
        "empty_files": [],
        "unreadable_files": [],
        "documents": 0,
        "jsonl_bad_lines": 0,
        "chars": 0,
        "markup_chars": 0,
        "entity_chars": 0,
        "url_chars": 0,
        "null_bytes": 0,
        "replacement_chars": 0,
        "control_chars": 0,
        "max_char_run": 0,
        "lengths": [],
        "top_fragments": Counter(),
        "exact_duplicate_docs": 0,
        "duplicate_bytes": 0,
        "hashes": set(),
    }


def audit_dir(root: Path):
    st = blank()
    ext_counts = Counter()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        st["files"] += 1
        st["bytes"] += path.stat().st_size
        ext = path.suffix.lower()
        ext_counts[ext or "(none)"] += 1
        if ext not in SUPPORTED:
            st["unsupported_files"].append(str(path))
            continue
        st["supported_files"] += 1
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            st["unreadable_files"].append(f"{path}: {e}")
            continue
        if not raw.strip():
            st["empty_files"].append(str(path))
            continue

        if ext == ".jsonl":
            docs = []
            for lineno, line in enumerate(raw.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    st["jsonl_bad_lines"] += 1
                    continue
                if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
                    st["jsonl_bad_lines"] += 1
                    continue
                docs.append(obj["text"])
        else:
            docs = [raw]

        for text in docs:
            st["documents"] += 1
            st["chars"] += len(text)
            st["lengths"].append(len(text))
            digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
            if digest in st["hashes"]:
                st["exact_duplicate_docs"] += 1
                st["duplicate_bytes"] += len(text.encode("utf-8", "replace"))
            else:
                st["hashes"].add(digest)
            for chunk in fenced(text):
                st["markup_chars"] += sum(len(m.group(0)) for m in TAG_RE.finditer(chunk))
                st["entity_chars"] += sum(len(m.group(0)) for m in ENTITY_RE.finditer(chunk))
                st["url_chars"] += sum(len(m.group(0)) for m in URL_RE.finditer(chunk))
                st["null_bytes"] += chunk.count("\x00")
                st["replacement_chars"] += chunk.count("\ufffd")
                st["control_chars"] += sum(
                    1 for c in chunk if ord(c) < 32 and c not in "\n\r\t"
                )
                for name, rx in SUSPICIOUS:
                    st["top_fragments"][name] += len(rx.findall(chunk))
                for m in re.finditer(r"([^\s])\1{7,}", chunk):
                    st["max_char_run"] = max(st["max_char_run"], len(m.group(0)))
    lengths = sorted(st["lengths"])
    n = len(lengths)
    st.pop("hashes")
    out = {k: v for k, v in st.items() if k not in ("lengths", "top_fragments")}
    out["extensions"] = dict(ext_counts)
    out["est_tokens_chars_div_4"] = st["chars"] // 4
    out["markup_pct"] = round(100.0 * st["markup_chars"] / max(1, st["chars"]), 2)
    out["url_pct"] = round(100.0 * st["url_chars"] / max(1, st["chars"]), 2)
    out["entity_pct"] = round(100.0 * st["entity_chars"] / max(1, st["chars"]), 2)
    out["top_fragments"] = dict(st["top_fragments"].most_common(12))
    if n:
        out["doc_len_min"] = lengths[0]
        out["doc_len_p50"] = lengths[n // 2]
        out["doc_len_p90"] = lengths[int(n * 0.9)]
        out["doc_len_max"] = lengths[-1]
        out["docs_under_200_chars"] = sum(1 for x in lengths if x < 200)
    return out


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    report = {}
    for a in argv:
        root = Path(a)
        if not root.is_dir():
            print(f"skip {a}: not a directory", file=sys.stderr)
            continue
        report[root.name] = audit_dir(root)
    totals = Counter()
    for st in report.values():
        for k in ("files", "bytes", "documents", "chars", "markup_chars", "url_chars",
                  "exact_duplicate_docs", "duplicate_bytes", "jsonl_bad_lines",
                  "unsupported_files"):
            v = st[k]
            totals[k] += len(v) if isinstance(v, list) else v
    report["_totals"] = dict(totals)
    report["_totals"]["markup_pct"] = round(
        100.0 * totals["markup_chars"] / max(1, totals["chars"]), 2
    )
    report["_totals"]["est_tokens_chars_div_4"] = totals["chars"] // 4
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
