# Training data

Put training corpora in this directory and pass the file paths to `lm-cli` with `--inputs`.

## Formats

- **`.txt` / `.md`** — the whole file is one document.
- **`.jsonl`** — one JSON object per line with a string `text` field (other fields ignored):

```json
{"text": "One training document per line."}
```

- **a directory** — walked recursively for the three extensions above; anything else in it is
  ignored. So `--inputs data/speeches` and `--inputs data/wikipedia-core` both work.

Blank lines are skipped; a malformed line is an error naming the file and line. Documents are
trimmed, dropped if shorter than `min_chars` or longer than `max_chars`, and exact duplicates
are removed (first kept). Identical texts can never land in two splits.

## Declaring a source

Every external source must be declared in the config (`sources` array) with its origin and a
license on the allowlist (CC0-1.0, Public-Domain, Unlicense, MIT, ISC, BSD-2-Clause,
BSD-3-Clause, Apache-2.0, CC-BY-4.0, CC-BY-SA-4.0, CC-BY-NC-SA-4.0, No-License-Granted).
GPL and unknown licenses are rejected. `No-License-Granted` is the honest label for text you
collected yourself that carries no license at all; it grants no redistribution rights, so it is
for local training only. The pipeline never downloads anything — you bring the files.

## Workflow

```bash
# 1. Inspect the prepared splits (also writes records.jsonl + manifest.json)
cargo run -q -p lm-cli -- prepare --config data/train.json \
  --inputs data/corpus.jsonl --out runs/prep

# 2. Train (pretraining batches over the token stream)
cargo run -q -p lm-cli -- train --config data/train.json \
  --inputs data/corpus.jsonl --out runs/pretrain \
  --steps 200 --batch 8 --seq 128 --vocab 4096

# 3. Export for Soup / HuggingFace
cargo run -q -p lm-cli -- export \
  --weights runs/pretrain/weights.safetensors \
  --model-config runs/pretrain/model_config.json \
  --tokenizer runs/pretrain/tokenizer.json \
  --out runs/hf

# 4. SFT with Soup (see the walkthrough in the chat; soup.yaml points at runs/hf)
```

`--mode sft` trains on chat conversations (synthetic + tool-use trajectories) with
assistant-only loss masks instead of the plain-text stream.

### Training on the local corpus

```bash
cargo build --release -p lm-cli
./target/release/lm-cli train --config data/pretrain.json --inputs data/speeches data/forums data/wikipedia-core data/misc data/english-wordnet-2025-json --out runs/pretrain --steps 150680 --batch 8 --seq 256 --vocab 8192 --d-model 256 --layers 4 --heads 8 --lr 0.0003
```

Measured on this machine (RTX 5070 Ti, release build): 576 M characters, 169,842 documents
(167,734 after filtering), 75,340 batches per epoch at `--batch 8 --seq 256`, about 0.19 s per
step. So one epoch is roughly 4 hours, and `--steps 150680` is two epochs (about 8 hours).
`--steps 2000` is a 6 minute smoke run.

`--steps` also sets the LR schedule (warmup is 10% of it, then cosine down to 10% of `--lr`), so
increase it with the run length rather than leaving it at one epoch.

Before the first `step` line, the process trains the tokenizer and builds every batch in memory:
about 4 minutes and ~3.5 GB of RAM for this corpus. Nothing is hung, and weights are written only
at the end of the run, so a killed run saves nothing.

Steps consume batches in order from the start of the stream (`step % batches`), so a partial run
trains only on the beginning of the corpus — in `--inputs` order, so `data/speeches` first.
Shuffling is not implemented yet.

`data/misc` includes `libretexts.jsonl` (587 MB, ~62k documents); drop that path for a fast run,
but then change the split ratios — a 54-document corpus cannot fill a 0.98/0.01/0.01 split and
`prepare` fails rather than duplicating. Passing `data` itself is not equivalent: directory
walking ignores `.json`, so the configs are skipped, but this README is `.md` and would be
trained on as a document.

`data/example.jsonl` is a tiny synthetic sample so the commands above run out of the box; it
is not real training data. A split that would be empty is an error, so a corpus of one or two
documents needs explicit ratios (see `data/pretrain.json`, which sends every record to train).

## Credits

Attribution for everything in this directory. Keep this list in sync when a folder changes.

Note that `wikipedia-core/*.md` are raw MediaWiki HTML dumps, not prose: `<div>` wrappers, nav
boxes and embedded template JSON are part of the training text. Strip them before expecting the
model to learn anything but markup.

| Folder | Source | License |
|---|---|---|
| `wikipedia-core/` | English Wikipedia ([wikipedia.org](https://en.wikipedia.org)) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `misc/libretexts.jsonl` | [LibreTexts](https://libretexts.org) bookshelves | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) unless the page says otherwise |
| `forums/` | [Reddit](https://www.reddit.com) thread dumps (`r/apple`, `r/Android`, `r/rarepuppers`, `r/politics`, `r/relationship_advice`, `r/legaladvice`) | No license granted — local use only |
| `misc/hacker_news_2000.txt` | [Hacker News](https://news.ycombinator.com) comment dump | No license granted — local use only |
| `speeches/` | Auto-transcribed political speeches | No license granted — local use only |
| `misc/bible-full.md` | World English Bible ([blessedtxts.com](https://blessedtxts.com)) | Public domain |
| `english-wordnet-2025-json/wordnet.jsonl` | [Open English WordNet](https://en-word.net) 2025 edition ([globalwordnet/english-wordnet](https://github.com/globalwordnet/english-wordnet)), flattened by `scripts/wordnet_to_jsonl.py` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `example.jsonl` | Project-authored sample | CC0-1.0 |

What the licenses require:

- **Wikipedia (CC BY-SA 4.0)** — credit Wikipedia and link the license wherever this text or a
  model trained on it is written up or shared. Share-alike applies to the *text*; whether
  weights are a derivative work is unresolved, so do not redistribute weights from this corpus
  without deciding that question first.
- **LibreTexts (CC BY-NC-SA 4.0)** — same, plus **non-commercial**: no commercial redistribution.
- **No license granted** — Reddit, Hacker News and the speech transcripts grant no rights at
  all. Fine for local experimentation; not something to ship training artifacts from.
- **World English Bible** — public domain, no obligations.
- **Open English WordNet (CC BY 4.0)** — credit the Open English WordNet project and link the
  license. No share-alike, so weights trained on it carry only the attribution obligation.

Derived files live next to their source: `english-wordnet-2025-json/wordnet.jsonl` is generated
from the 70 MB of raw synset JSON by `scripts/wordnet_to_jsonl.py` (re-run it if the JSON is
replaced). The raw `.json` files are ignored by the loader, so pass the folder, not the files.

These are recorded in `data/pretrain.json` and end up in each run's `manifest.json`, so a
checkpoint can always be traced back to the sources and terms it was trained under.

The Common Pile paper is the reference for the licensing approach:

```bibtex
@article{kandpal2025common,
  title={{The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text}},
  author={Nikhil Kandpal and Brian Lester and Colin Raffel and Sebastian Majstorovic and Stella Biderman and Baber Abbasi and Luca Soldaini and Enrico Shippole and A. Feder Cooper and Aviya Skowron and Shayne Longpre and Lintang Sutawika and Alon Albalak and Zhenlin Xu and Guilherme Penedo and Loubna Ben and Elie Bakouch and John David and Honglu Fan and Dashiell Stander and Guangyu Song and Aaron Gokaslan and John Kirchenbauer and Tom Goldstein and Brian R and Bhavya Kailkhura and Tyler Murray},
  journal={arXiv preprint},
  year={2025}
}
```
