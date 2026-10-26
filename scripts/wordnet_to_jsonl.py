#!/usr/bin/env python3
"""Flatten Open English WordNet synset JSON into readable JSONL for lm-data.

The 2025 JSON release stores one file per lexicographer class (noun.food.json,
verb.motion.json, ...) plus per-letter entry files. Only the class files carry
definitions; `entries-*.json` is a gloss-free index of sense ids and `frames.json`
is verb-frame machinery, so both are skipped.

Each synset becomes one line: {"id": <synset id>, "text": <dictionary entry>},
where the entry reads like a dictionary line, e.g.

    able, able — adjective. (usually followed by 'to') having the necessary means
    or skill or know-how or authority to do something. Examples: able to swim;
    she was able to program her computer.

Output is written next to the source as `wordnet.jsonl` and is deterministic:
files are visited in sorted order and lines follow file order.

Usage: scripts/wordnet_to_jsonl.py [source-dir]
"""

import json
import sys
from pathlib import Path

POS_NAMES = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective satellite",
    "r": "adverb",
}

SKIP_PREFIXES = ("entries-", "frames")


def synset_files(src: Path) -> list[Path]:
    """Lexicographer-class files only, in sorted order."""
    return sorted(
        p for p in src.glob("*.json") if not p.name.startswith(SKIP_PREFIXES)
    )


def as_text(value) -> str:
    """Examples are plain strings, except ~1k which are {source, text} dicts."""
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return str(value)


def render(synset: dict) -> str:
    """One dictionary line per synset. Empty when the synset carries no gloss."""
    members = synset.get("members") or []
    pos = POS_NAMES.get(synset.get("partOfSpeech", ""), synset.get("partOfSpeech", ""))
    definitions = synset.get("definition") or []
    examples = [as_text(e) for e in (synset.get("example") or [])]

    gloss = ". ".join(as_text(d).rstrip(".") for d in definitions)
    if not gloss:
        return ""
    lead = ", ".join(members) if members else pos
    text = f"{lead} ({pos}): {gloss}."
    if examples:
        text += " Examples: " + "; ".join(e.rstrip(".") for e in examples) + "."
    return text


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/english-wordnet-2025-json")
    if not src.is_dir():
        print(f"not a directory: {src}", file=sys.stderr)
        return 1
    out_path = src / "wordnet.jsonl"

    written = 0
    chars = 0
    files = synset_files(src)
    if not files:
        print(f"no synset .json files in {src}", file=sys.stderr)
        return 1

    with out_path.open("w", encoding="utf-8") as out:
        for path in files:
            with path.open(encoding="utf-8") as f:
                synsets = json.load(f)
            for sid, synset in synsets.items():
                text = render(synset).strip()
                if not text or not synset.get("definition"):
                    continue
                out.write(json.dumps({"id": sid, "text": text}, ensure_ascii=False) + "\n")
                written += 1
                chars += len(text)

    print(f"{out_path}: {written} entries, {chars} chars from {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
