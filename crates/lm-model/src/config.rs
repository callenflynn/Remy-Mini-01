use serde::{Deserialize, Serialize};

/// Hyperparameters of the decoder-only transformer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub context_len: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub ffn_hidden: usize,
    pub rope_theta: f64,
    pub rms_eps: f64,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Exact trainable parameter count (tied embeddings counted once).
    pub fn param_count(&self) -> usize {
        let hd = self.head_dim();
        let attn = self.d_model * self.n_heads * hd // q
            + 2 * self.d_model * self.n_kv_heads * hd // k, v
            + self.n_heads * hd * self.d_model; // o
        let mlp = 3 * self.d_model * self.ffn_hidden; // gate, up, down
        let norms = 2 * self.d_model;
        let per_layer = attn + mlp + norms;
        self.vocab_size * self.d_model + self.n_layers * per_layer + self.d_model
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.d_model.is_multiple_of(self.n_heads),
            "d_model must divide by n_heads"
        );
        anyhow::ensure!(
            self.n_heads.is_multiple_of(self.n_kv_heads),
            "n_heads must divide by n_kv_heads"
        );
        anyhow::ensure!(
            self.head_dim().is_multiple_of(2),
            "head_dim must be even for RoPE"
        );
        Ok(())
    }
}
