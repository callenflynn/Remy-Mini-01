//! Export adapter: HuggingFace `LlamaForCausalLM` layout.
//!
//! Our model (pre-norm RMSNorm, `rotate_half` RoPE, SwiGLU, GQA, tied embeddings) is
//! mathematically identical to HF Llama, so a pure key rename plus config/tokenizer files
//! produces an equivalent checkpoint. No tensor values are altered.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, anyhow};
use lm_tokenizer::{ASSISTANT, BOS, END_OF_TURN, EOS, LmTokenizer, PAD, SYSTEM, USER};

/// Rename one checkpoint key to the Llama layout. Returns `None` for no mapping.
fn map_key(k: &str) -> Option<String> {
    if k == "embed.weight" {
        return Some("model.embed_tokens.weight".into());
    }
    if k == "norm.weight" {
        return Some("model.norm.weight".into());
    }
    let rest = k.strip_prefix("layers.")?;
    let (idx, tail) = rest.split_once('.')?;
    let tail = tail
        .replace("attn.q.weight", "self_attn.q_proj.weight")
        .replace("attn.k.weight", "self_attn.k_proj.weight")
        .replace("attn.v.weight", "self_attn.v_proj.weight")
        .replace("attn.o.weight", "self_attn.o_proj.weight")
        .replace("attn_norm.weight", "input_layernorm.weight")
        .replace("mlp_norm.weight", "post_attention_layernorm.weight")
        .replace("mlp.gate.weight", "mlp.gate_proj.weight")
        .replace("mlp.up.weight", "mlp.up_proj.weight")
        .replace("mlp.down.weight", "mlp.down_proj.weight");
    Some(format!("model.layers.{idx}.{tail}"))
}

pub const CHATML_TEMPLATE: &str = "PLACEHOLDER";

/// ChatML template with `{% generation %}` blocks so SFT trainers can mask non-assistant tokens.
pub const CHATML_JINJA: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}{% if message.role == 'assistant' %}{% generation %}{{ message['content'] + 'im_end\n' }}{% endgeneration %}{% else %}{{ message['content'] + 'im_end\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

/// Write an HF-loadable LlamaForCausalLM directory from a raw checkpoint.
pub fn export_llama<P: AsRef<Path>, Q: AsRef<Path>>(
    raw_ckpt: P,
    out_dir: Q,
    cfg: &lm_model::ModelConfig,
    tok: &LmTokenizer,
    tokenizer_json: &Path,
) -> Result<()> {
    let out = out_dir.as_ref();
    std::fs::create_dir_all(out)?;
    let raw = candle_core::safetensors::load(raw_ckpt.as_ref(), &candle_core::Device::Cpu)?;
    let mut renamed: HashMap<String, candle_core::Tensor> = HashMap::new();
    for (k, v) in raw {
        let nk = map_key(&k).ok_or_else(|| anyhow!("unmapped key {k}"))?;
        renamed.insert(nk, v);
    }
    candle_core::safetensors::save(&renamed, out.join("model.safetensors"))?;
    write_configs(out, cfg, tok)?;
    std::fs::copy(tokenizer_json, out.join("tokenizer.json"))?;
    Ok(())
}

fn write_configs(out: &Path, cfg: &lm_model::ModelConfig, tok: &LmTokenizer) -> Result<()> {
    let id = |t: &str| tok.special_id(t);
    let config = serde_json::json!({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.ffn_hidden,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "max_position_embeddings": cfg.context_len,
        "rms_norm_eps": cfg.rms_eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": true,
        "torch_dtype": "float32",
        "bos_token_id": id(BOS)?,
        "eos_token_id": id(EOS)?,
        "pad_token_id": id(PAD)?,
    });
    std::fs::write(
        out.join("config.json"),
        serde_json::to_string_pretty(&config)?,
    )?;
    let tok_cfg = serde_json::json!({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": cfg.context_len,
        "chat_template": CHATML_JINJA,
    });
    std::fs::write(
        out.join("tokenizer_config.json"),
        serde_json::to_string_pretty(&tok_cfg)?,
    )?;
    let special = serde_json::json!({
        "pad_token": PAD, "bos_token": BOS, "eos_token": EOS,
        "unk_token": PAD, "additional_special_tokens": [SYSTEM, USER, ASSISTANT, END_OF_TURN],
    });
    std::fs::write(
        out.join("special_tokens_map.json"),
        serde_json::to_string_pretty(&special)?,
    )?;
    Ok(())
}
