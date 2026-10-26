# lm-data

Data preparation for the model. It runs without any external corpus.

## Inputs

- `.txt` / `.md`: the whole file is one document.
- `.jsonl`: one JSON object per line with a string `text` field. Blank lines are skipped. A malformed line is an error that names the file and line.
- A directory is walked recursively (sorted, so order is deterministic) and every `.txt`, `.md` and `.jsonl` file in it is read; other files are ignored.

An explicit file with any other extension is an error. Document ids are `<path as given>#<index>`.

## Configuration

`configs/tiny.json` is a test fixture config. Its explicit 60/20/20 split is for testing only. Defaults (`Config::default`) are seed 1, split 98/1/1, normalize `min_chars=20`, `max_chars=100000`, and synthetic counts of 200/100/100/100/100. Those counts are for smoke runs only and are not sufficient to train a capable model.

Unknown config fields are rejected. Ratios must sum to 1.0. `split.strategy` must be `hash`; `template` is rejected until template-level splitting is wired into `prepare` (the helper exists in `split.rs` but is not used).

## Sources and licenses

Each declared source needs `name`, `kind` (`fixture` or `external`), `origin`, and a `license` on the allowlist: CC0-1.0, Public-Domain, Unlicense, MIT, ISC, BSD-2-Clause, BSD-3-Clause, Apache-2.0, CC-BY-4.0, CC-BY-SA-4.0, CC-BY-NC-SA-4.0, No-License-Granted. Missing, GPL, and unknown licenses are errors. `No-License-Granted` exists so locally collected text with no license can be declared and attributed rather than included silently; it grants no redistribution rights. `kind: synthetic` is not allowed; generated data is never declared as a source.

To add external training data:
1. Download it yourself. The pipeline never downloads corpora.
2. Convert it to `.txt`, `.md`, or `.jsonl`.
3. Add a source entry with its origin URL and verified license to your config.
4. Pass the file or directory path to `prepare`.

## Normalization

Documents are trimmed, then dropped if shorter than `min_chars` or longer than `max_chars` (no truncation), then exact duplicates are dropped (first kept). Counts are recorded in the manifest.

## Splits

- Each record's id (external documents and synthetic training examples alike) is hashed with the seed into train/val/test.
- Held-out synthetic templates always go to `test`.

A split that would be empty is an error. Nothing is duplicated to fill it. Duplicate ids are rejected.

## Synthetic families

Each family has a training template and a held-out template, and every answer is checked independently:
- `arithmetic`: fully parenthesized multi-operation expressions, checked by a separate parser.
- `logic`: "all A are B" deductions, with answers from enumerating the cases.
- `word_problem`: multi-step counting problems.

Instruction-following, coding, and reasoning evaluation sets are not generated yet.

## Chat format

```text
<|system|>...<|user|>...<|assistant|>...<|end_of_turn|
```

Each message is tokenized on its own and special tokens are added as whole ids. Loss covers only assistant reply tokens and the `<|end_of_turn|>` that closes each reply. Role markers and user/system content are masked out. A conversation must alternate user and assistant and end with an assistant reply.

## Tests

`cargo test -p lm-data` covers config validation, license checks, malformed input, determinism, split isolation, synthetic answer checks, template separation, and exact loss masks on real token ids, plus an end-to-end run on `fixtures/tiny.txt` and `fixtures/tiny.jsonl`.

@article{kandpal2025common,
  title={{The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text}},
  author={Nikhil Kandpal and Brian Lester and Colin Raffel and Sebastian Majstorovic and Stella Biderman and Baber Abbasi and Luca Soldaini and Enrico Shippole and A. Feder Cooper and Aviya Skowron and Shayne Longpre and Lintang Sutawika and Alon Albalak and Zhenlin Xu and Guilherme Penedo and Loubna Ben and Elie Bakouch and John David and Honglu Fan and Dashiell Stander and Guangyu Song and Aaron Gokaslan and John Kirchenbauer and Tom Goldstein and Brian R and Bhavya Kailkhura and Tyler Murray},
  journal={arXiv preprint},
  year={2025}
}