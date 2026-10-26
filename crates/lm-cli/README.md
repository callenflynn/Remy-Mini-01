# lm-cli

Command-line entry point for the whole pipeline: prepare data, train, inspect, export.

A finished `train` run leaves four files in `--out` (`runs/<name>/`):

| File | What it is |
|---|---|
| `model_config.json` | architecture the weights belong to (read by `generate` and `export`) |
| `tokenizer.json` | byte-level BPE tokenizer trained on that run's training split |
| `weights.safetensors` | model parameters, f32 — **written only after the final step** |
| `metrics.json` | steps, first/last train loss, val loss, device, weight count |

## Training

```bash
cargo build --release -p lm-cli

./target/release/lm-cli train --config data/pretrain.json \
  --inputs data/speeches data/forums data/wikipedia-core data/misc data/english-wordnet-2025-json \
  --out runs/pretrain --steps 150680 --batch 8 --seq 256 \
  --vocab 8192 --d-model 256 --layers 4 --heads 8 --lr 0.0003
```

`--steps` is the number of optimizer steps, not epochs: use `batches × epochs`, where
`train` prints the batch count before the first step. The learning-rate schedule is built
from the same number (10% warmup, cosine decay to 10% of `--lr`), so setting it to a small
number is safe but leaves you with a proportionally less-trained model.

There are no intermediate checkpoints: if the process dies at step 74,000 of 75,340, nothing
is saved. Run it under `nohup` and keep the machine awake.

## Sampling from a trained run

```bash
./target/release/lm-cli generate --run runs/pretrain --prompt "The history of" \
  --max-tokens 100 --temperature 0.8 --top-k 40 --seed 0
```

- `--run` is a directory containing `model_config.json`, `weights.safetensors` and
  `tokenizer.json`.
- `--temperature 0` is greedy decoding: deterministic, same output for every `--seed`, useful
  for checking whether a checkpoint learned anything at all.
- `--top-k 0` samples the full distribution. Same seed and same prompt give the same text.
- Generation stops early on `<|eos|>` or `<|end_of_turn|>`.
- There is no KV cache: each token costs one forward pass over the prompt so far, cropped to
  the newest `context_len` (256) tokens. Fine at this size; a larger model wants a cache.

## Running the tool-call harness

```bash
./target/release/lm-cli harness --run runs/pretrain \
  --prompt "<|user|>what is the population of Peru<|end_of_turn|><|assistant|>" \
  --max-steps 3 --max-tokens 64 --temperature 0.4
```

This drives `lm_infer::run` — generate a turn, parse `<tool_call>` envelopes, validate them
against the registry in `lm_data::tooluse::tool_registry()` (`web_search`, `fetch_page`),
"execute" them, append `<tool_result>` and go round again — with the trained model as the
generator. The executor is an offline stub: it returns the arguments it was handed and never
touches the network, the filesystem or a shell. It prints the full transcript and the number
of calls that actually executed.

A base pretrained model has never seen the tool protocol, so expect plain text and
`tool calls executed: 0`. Tool-call behaviour comes from `--mode sft`, which trains on the
synthetic trajectories in `lm-data`.

## Export as a Llama checkpoint

```bash
./target/release/lm-cli export --weights runs/pretrain/weights.safetensors \
  --model-config runs/pretrain/model_config.json \
  --tokenizer runs/pretrain/tokenizer.json --out runs/llama
```

Writes an HF `LlamaForCausalLM` directory — the same architecture (Llama-style RoPE, RMSNorm,
SwiGLU, GQA, tied embeddings), so this is a key rename and nothing else:

- `model.safetensors` in llama naming (`model.embed_tokens.weight`,
  `model.layers.N.self_attn.q_proj.weight`, `model.layers.N.mlp.gate_proj.weight`, …). The
  output projection is tied to the embedding, so there is no `lm_head.weight` and
  `config.json` says `tie_word_embeddings: true`.
- `config.json`, `tokenizer.json`, `tokenizer_config.json` (with a chat template using this
  tokenizer's `<|system|>`/`<|user|>`/`<|assistant|>`/`<|end_of_turn|>` markers, plus
  `{% generation %}` spans for SFT), `special_tokens_map.json`.

That directory loads with `transformers` (`AutoModelForCausalLM.from_pretrained`), with Soup
for LoRA fine-tuning, and converts to GGUF for llama.cpp/Ollama with llama.cpp's
`convert_hf_to_gguf.py` (needs its `requirements.txt`, i.e. Python plus the `gguf` package).
None of those tools are dependencies of this repo, so nothing here runs them for you.
