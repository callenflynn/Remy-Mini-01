# Remy-Mini-01 — corpus audit, cleaning and quality pass

Date: 2026-10-26. Everything below was measured on this checkout; commands are in the report so
they can be re-run. Numbers that are estimates are labelled as such.

---

## 1. Root causes, ranked by confidence and severity

### 1.1 The training stream was mostly HTML, not language — confirmed, severe

`data/wikipedia-core/*.md` is not Markdown: it is rendered MediaWiki HTML saved with a `.md`
extension. Measured over all 46 files (36.1 M characters):

| | raw | cleaned |
|---|---|---|
| characters | 36,083,280 | 4,895,537 (13.6%) |
| markup characters | 19,098,196 (52.9%) | 0 |
| HTML entities | 2,947,922 (8.2%) | 0 |
| URL characters | 1,611,039 (4.5%) | ~0 |
| `<div>`/`<span>`, `href=`, `mw-` attr, `data-mw` JSON | 547,944 / 111,001 / 100,244 / 58,191 | 0 |
| tokens per word (8k BPE) | **7.31** | **1.66** |
| markup-ish token share | **18.1%** | 2.4% |

Every one of the artifacts in the bad sample (`href="//en.wikipedia.org/wiki/`, `id%2222`,
`mw-…`) is a substring of this source. The model was not learning English from Wikipedia; it was
learning to reproduce tag soup, entity escapes and embedded JSON template payloads.

### 1.2 Initial loss was 7× too high — confirmed, severe, and fixed

The first training losses were ~68 instead of ~9. The cause was in `crates/lm-model/src/lib.rs`:
`candle_nn::embedding` initializes with `normal(0, 1)`, so with `d_model = 256` the tied output
logits start with standard deviation ≈ 16, giving a cross-entropy of ~68 instead of
`ln(8192) = 9.01`. Early steps were spent mostly resizing the embedding scale rather than
learning. Now initialized with GPT-2's `N(0, 0.02)`; a test (`fresh_model_starts_near_uniform`)
fails if the first loss drifts more than 1 nat from `ln(vocab)`. Both 2,000-step runs now start
at 9.07.

### 1.3 Runs shorter than one epoch only saw the first sources — confirmed, moderate

`train` packed documents in input order, so a 2,000-step run (4.1 M tokens) over
`speeches forums wikipedia-core misc wordnet` trained almost entirely on speeches and the start of
Wikipedia while validation drew from every source. `lm_data::split::stream_order` now orders the
training stream by `hash(seed, id)`: deterministic, reproducible, independent of the order inputs
are listed, and mixed across every source. Val loss was consequently uninterpretable (see 1.5).

### 1.4 Validation loss was one batch from the start of the val split — confirmed, moderate

`train` evaluated `val_batches.first()`. With the corpus listed as above, the first val batch is
speeches — the easiest, most repetitive text in the corpus — so validation read *lower* than
training (raw: 4.79 vs 7.15 at step 2000). It now averages 20 batches spread evenly across the
whole val split, and `metrics.json` records `val_batches_total`/`val_batches_evaluated`. The two
2,000-step runs in §7 were trained before that change, so their `val_loss` is the old single-batch
figure; the comparable numbers there come from the new `lm-cli eval` subcommand.

### 1.5 The corpus is 96.7% one source — confirmed, severe for output quality

After cleaning, character counts are: LibreTexts textbooks 501.9 M (96.7%), WordNet 11.6 M
(2.2%), Wikipedia 4.9 M (0.9%), forums 0.46 M, speeches 0.18 M (0.04%). The LibreTexts text is
mathematics/science prose full of LaTeX (`\(f(x)=\frac{…}\)`, `\PageIndex{3}`). A model trained
on this is a *textbook* model: that, not markup, is why sampled text is LaTeX soup. Balancing the
mix is a data-acquisition decision, not a cleaning one — see §9.

### 1.6 Everything is undertrained — confirmed by the loss curve

2,000 steps × 8 × 256 = 4.1 M tokens for a 4.46 M-parameter model. Loss is still falling steeply
at the end of both runs (raw 9.07 → 7.00 → 6.68 → 6.19 → 5.98; clean 9.07 → 7.01 → 6.54 → 6.26 →
5.53 → 5.63). Greedy decoding loops on `-` or `the same of the same` because the model has seen
2.7% of one epoch. This is why "it looks like garbage" cannot be read as a data verdict yet.

### 1.7 Checkpoint/config mismatches were possible — confirmed, moderate

`generate`/`harness` now require `model_config.json`, `tokenizer.json` and `weights.safetensors`
in the same run directory, refuse a missing one with "not a finished training run", reject a
tokenizer whose vocab differs from the model config, and turn a shape mismatch into
"weights … do not match … (a run that was interrupted, or a different run's files)". The earlier
`[8192, 256]` vs `[657, 64]` error was exactly that check doing its job on `runs/pretrain`
(a 657-token/64-wide experiment whose `model_config.json` a later run overwrote before it wrote
weights). It was a stale directory, not corruption — but note that `train` writes
`model_config.json` and `tokenizer.json` at the start and weights only at the end, so an
interrupted run leaves a directory that looks complete but is not. Treat any run without
`metrics.json` as unfinished; `runs/pretrain` is such a directory and should not be used.

### 1.8 Smaller findings

- `data/misc/libretexts.jsonl`: 2,108 exact duplicate documents (repeated book pages across
  editions); 2,051 removed by the cleaner, the rest were already filtered by length/prose rules.
- `data/misc/libretexts.jsonl` had 42 U+FFFD replacement characters and 6,011 control characters
  (soft hyphens, zero-width spaces, BOMs). Both are removed now.
- WordNet was a folder of 73 JSON files that the loader ignores plus a flattened `wordnet.jsonl`.
  Passing the folder silently contributed nothing. The renderer (`scripts/wordnet_to_jsonl.py`)
  is correct — it reads `members`/`partOfSpeech`/`definition`/`example` and produces
  `lemma, lemma (pos): gloss. Examples: …` — but it had no test and the docs did not say which
  path to pass. Now: `scripts/test_wordnet_to_jsonl.py`, and the docs say pass the `.jsonl`.
- The tokenizer was already trained on the training split only, and is correct: 8,192 ids,
  specials as whole ids, encode/decode round-trips, no id ever outside the embedding table
  (enforced by `lm_tokenizer` tests and by the vocab check on load).

---

## 2. Files changed

| File | Purpose |
|---|---|
| `crates/lm-data/src/clean.rs` | **new**: deterministic cleaner (`Policy`, `clean`, `CleanStats`, `Rejection`) and `Deduper` (exact documents + paragraphs), 20 unit tests |
| `crates/lm-data/src/lib.rs` | export the `clean` module |
| `crates/lm-data/src/split.rs` | `stream_order`: hash-ordered, reproducible training stream (+ test) |
| `crates/lm-data/tests/pipeline.rs` | tests that duplicates cannot straddle splits and that cleaning is not applied twice |
| `crates/lm-cli/src/cleanup.rs` | **new**: `lm-cli clean` — writes `<source>.jsonl`, `report.json`, `manifest.json`, refuses to write inside an input |
| `crates/lm-cli/src/generate.rs` | **new** (from the interrupted turn): `Checkpoint` loader with the three-file/shape/vocab checks, sampling, `batch_loss` |
| `crates/lm-cli/src/main.rs` | `clean`, `tokenize` and `eval` subcommands, `--help`, hash-ordered training stream, spread validation sampling |
| `crates/lm-cli/tests/clean_cli.rs` | **new**: end-to-end `clean` tests (loader-ready output, idempotence, byte-identical reruns, input-overlap refusal) |
| `crates/lm-model/src/lib.rs` | GPT-2 embedding init + `fresh_model_starts_near_uniform` test |
| `crates/lm-train/src/export.rs` | chat template matching the tokenizer's own special tokens, position map tests |
| `scripts/audit_corpus.py` | **new**: the audit that produced `reports/corpus-audit.json` |
| `scripts/test_wordnet_to_jsonl.py` | **new**: WordNet renderer tests |
| `data/README.md` | cleaning workflow, measured before/after stats, licensing table untouched |

Nothing was deleted. `runs/pretrain`, `runs/pretrain-real`, `runs/bible-*`, `runs/llama` are
untouched.

---

## 3. The cleaning command

```bash
cargo build --release -p lm-cli

./target/release/lm-cli clean \
  --config data/pretrain.json \
  --inputs data/speeches data/forums data/wikipedia-core data/misc \
           data/english-wordnet-2025-json/wordnet.jsonl \
  --out data/cleaned-pretrain
```

Takes ~20 s, writes `speeches.jsonl`, `forums.jsonl`, `wikipedia-core.jsonl`, `misc.jsonl`,
`wordnet.jsonl`, `report.json`, `manifest.json`. Source files are never written to; the command
refuses an `--out` that overlaps an input. Optional flags: `--min-chars`, `--alpha-ratio`,
`--keep-urls`, `--no-dedup-paragraphs`, `--paragraph-min`.

Re-running is byte-identical (there are no timestamps in the sidecars), and `clean(clean(x)) ==
clean(x)` is enforced by tests over hostile inputs.

---

## 4. Before and after

| Source | docs in → kept | chars in → out | kept | tokens ≈ (chars/4) |
|---|---|---|---|---|
| speeches | 3 → 3 | 184,499 → 184,382 | 99.9% | 46 k |
| forums | 3 → 3 | 467,188 → 460,685 | 98.6% | 115 k |
| wikipedia-core | 46 → 46 | 36,083,280 → 4,895,537 | 13.6% | 1.22 M |
| misc (bible, HN, LibreTexts) | 62,271 → 59,983 | 527,330,957 → 501,861,172 | 95.2% | 125.5 M |
| wordnet | 107,519 → 101,914 | 11,853,496 → 11,626,817 | 98.1% | 2.91 M |
| **total** | **169,842 → 161,949** | **575,919,420 → 519,028,593** | **90.1%** | **129.8 M** |

Measured token counts (not estimates) from the training stream: raw corpus 75,340 batches ×
8 × 256 = **154.3 M train tokens**, cleaned 68,232 × 8 × 256 = **139.7 M train tokens** (90% train
split). Cleaned documents rejected: 2,051 duplicate (misc), 237 not prose (table-only pages),
5,605 too short (WordNet glosses under 48 chars) — every rejection is counted with its reason in
`report.json`.

Residual artifacts in the cleaned corpus, measured by scanning every document: `href=` 26,
`<div` 29, `</` 234 — all inside LibreTexts pages that teach HTML or carry CKEditor
`data-cke-saved-href` attributes; 0 HTML tags, 0 entities, 0 `data-mw`, 0 `[edit]`, 0 mangled
LaTeX (`\frac[`) after the fixes below. That is ~289 occurrences in 519 M characters.

Two cleaner bugs found and fixed **while verifying this report** (both were corrupting real
content, so they are worth naming):

1. A lone `{` was rewritten to `[` (LaTeX `\frac{1}{2}` → `\frac[1}[2}`). Braces are now copied
   verbatim unless a complete `{{…}}` template is present.
2. When `{{` had no closing `}}`, the template branch fell through to the Markdown-link branch and
   emitted a `[`. `\frac{{(x+2)}^2…` in LibreTexts hit this. Now an unterminated `{{` is text.
3. Pandoc attribute blocks (`{#mwAfc .wikitable … mw="{\"parts\":…}"}`) are dropped whole; before,
   the template JSON leaked into prose.

---

## 5. Tests and results

```bash
cargo test --workspace --release
```

141 tests, **all passing** (0 failed): lm-cli 13 + 4 integration, lm-data 73 + 9 loader + 5
pipeline, lm-infer 10 + 5 security, lm-model 5 + 3 gradients, lm-tokenizer 4, lm-train 4 + 3
checkpoint + 1 export + 1 overfit + 1 smoke. Also:

```bash
python3 scripts/test_wordnet_to_jsonl.py   # 8 tests, pass
python3 scripts/audit_corpus.py data/speeches data/forums data/wikipedia-core \
        data/misc data/english-wordnet-2025-json > reports/corpus-audit.json   # 70 s
```

Coverage that matters for this task: HTML removal and entity decoding (incl. double-escaped and
numeric), Markdown/LaTeX preservation, broken markup and URL artifacts, BOM/NUL/control handling,
duplicate documents and paragraphs, empty/short/non-prose rejection with reasons, WordNet
rendering, determinism, idempotence, byte-identical reruns, loss masks on real token ids,
next-token targets, tiny-dataset overfitting, split isolation, checkpoint shape/vocab rejection,
quantized-free gradient checks. `tests/overfit.rs` trains a tiny model to near-zero loss on a
small corpus, which is the "can it learn at all" check.

---

## 6. What is still wrong

1. **Undertraining dominates output quality.** 4.1 M tokens for 4.46 M parameters. Nothing about
   the samples in §7 is evidence of corpus quality; the loss curve is still falling.
2. **The corpus is 96.7% LibreTexts.** LaTeX-dense maths dominates the token distribution.
3. **Greedy decoding loops** on any model this small and undertrained; it is a decoding artifact
   of a peaked, low-entropy distribution, not a data bug. Sample with `--temperature 0.8 --top-k 40`.
4. **`misc.jsonl` is a single 502 MB line-delimited file.** Fine for 62 k documents, but the loader
   holds the whole corpus in memory; a larger corpus needs streaming shards.
5. **Validation is 1,631 documents (3%)** and both runs' `metrics.json` `val_loss` came from the
   old single-batch code; use `lm-cli eval` for comparable numbers (§7).
6. **Language filtering is not implemented.** All five sources are English by construction, so
   nothing was excluded, but there is no detector if a non-English source is added.
7. **WordNet contributes 5,605 rejected entries** — glosses shorter than 48 characters. Lowering
   `--min-chars` keeps more but admits one-clause stubs; this was a deliberate cut-off.

---

## 7. Controlled comparison (fixed budget, identical architecture)

```bash
# raw
./target/release/lm-cli train --config data/pretrain.json \
  --inputs data/speeches data/forums data/wikipedia-core data/misc \
           data/english-wordnet-2025-json/wordnet.jsonl \
  --out runs/compare2-raw --steps 2000 --batch 8 --seq 256 --vocab 8192 \
  --d-model 256 --layers 4 --heads 8 --lr 0.0003

# cleaned
./target/release/lm-cli train --config data/pretrain.json --inputs data/cleaned-pretrain \
  --out runs/compare2-clean --steps 2000 --batch 8 --seq 256 --vocab 8192 \
  --d-model 256 --layers 4 --heads 8 --lr 0.0003

# comparable held-out loss (20 batches spread across the val split)
./target/release/lm-cli eval --run runs/compare2-raw --config data/pretrain.json \
  --inputs data/speeches data/forums data/wikipedia-core data/misc \
           data/english-wordnet-2025-json/wordnet.jsonl --batches 20
./target/release/lm-cli eval --run runs/compare2-clean --config data/pretrain.json \
  --inputs data/cleaned-pretrain --batches 20
```

| | raw | cleaned |
|---|---|---|
| first loss | 9.0720 | 9.0709 |
| loss @ 1000 / 1750 / 2000 | 6.19 / 5.98 / 7.15 | 6.26 / 5.53 / 5.63 |
| held-out loss (20 batches) | **5.9139** (ppl 370) | **5.7852** (ppl 325) |
| val docs scored | 1,664 | 1,631 |
| tokens/word on its own text | 7.31 (Wikipedia) | 1.66 |
| markup-ish tokens in Wikipedia | 18.1% | 2.4% |

Cleaned is 0.13 nats better on held-out text under an identical budget — a real but modest gain,
and the two runs use different tokenizers (each trained on its own train split), so treat it as
indicative rather than a clean A/B. The unambiguous result is the tokenizer one: 4.4× fewer tokens
to say the same Wiki sentence, and 18% → 2.4% of tokens describing markup. The last-step train
loss is noisier than the held-out figure (single batch, sawtooth LR) — hence `eval`.

Generated samples (greedy, 64 tokens, `reports/samples2/compare2-{raw,clean}.txt`) are degenerate
in both runs, e.g. raw `"Astronomy is the study of"` → `" the same-s of the same-s."` followed by
`-` lines, cleaned → `" the same of the same of the same of …"`. **This is undertraining, not
residual markup**, and the difference between the two is not measurable at this budget.

## 8. Generating text from a finished run

```bash
# deterministic (greedy) — the "did it learn anything" check
./target/release/lm-cli generate --run runs/compare2-clean \
  --prompt "The history of the Roman Empire" --max-tokens 80 --temperature 0

# sampled, reproducible for a seed
./target/release/lm-cli generate --run runs/compare2-clean \
  --prompt "A fraction is a number that" --max-tokens 64 \
  --temperature 0.8 --top-k 40 --seed 7

# tool-call harness, offline tool stubs
./target/release/lm-cli harness --run runs/compare2-clean \
  --prompt "search the web for the rust language" --max-steps 3
```

## 9. Recommended next run

Keep the architecture and the cleaned corpus; change the budget first, because the loss curve says
the model is data-starved, not broken:

```bash
./target/release/lm-cli train --config data/pretrain.json --inputs data/cleaned-pretrain \
  --out runs/pretrain-clean --steps 20000 --batch 16 --seq 512 --vocab 8192 \
  --d-model 256 --layers 6 --heads 8 --lr 0.0006
```

Reasoning: 20,000 × 16 × 512 ≈ 164 M tokens ≈ one epoch over the cleaned corpus, which is also
~37 tokens/parameter for a 4.5 M-parameter model — near the compute-optimal ratio for a model this
small. `seq 512` matches what 256-token windows waste (documents are long); `lr 0.0006` is the
usual scaling for batch 16. Expect perplexity in the low tens, real words, and — because the
corpus is 97% textbooks — a recognisably expository voice with LaTeX habits. Limitations: it is
still a 4.5 M-parameter model trained on 130 M tokens of which 97% is one source, so factual
recall will be near zero; do not read loss as knowledge. The 150,680-step config
(`runs/soup.yaml`) was not started.

Before that run, the highest-value data change is **rebalancing**: cap LibreTexts at roughly 40–50%
of the stream (it currently supplies 97%) so the model sees Wikipedia prose, dialogue and
definitions often enough to learn their style. That needs more non-textbook data, which is a
licensing decision (see `data/README.md`), not something the cleaning pipeline can invent.
