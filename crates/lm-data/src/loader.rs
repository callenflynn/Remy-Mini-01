//! Deterministic batch construction for pretraining and SFT.
//!
//! Two dataset paths:
//! - [`PretrainData`] tokenizes plain-text records into one stream and cuts it into
//!   fixed-length next-token rows (mask 1 everywhere).
//! - [`SftData`] serializes chat conversations with [`crate::chat`], so only assistant
//!   content and the end-of-turn closing each reply carry loss, and pads rows to a common
//!   length with the tokenizer's pad id (mask 0 on padding).
//!
//! There is no randomness anywhere: the same inputs always produce identical batches.

use anyhow::{Result, bail};
use lm_tokenizer::{EOS, LmTokenizer, PAD};

use crate::chat::{self, Message};

/// One batch of equal-length rows: `[batch][seq]`. `tokens` feeds the model, `targets` are
/// the next tokens, and `mask` is 1.0 where the loss counts a position.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    pub tokens: Vec<Vec<u32>>,
    pub targets: Vec<Vec<u32>>,
    pub mask: Vec<Vec<f32>>,
}

impl Batch {
    pub fn batch_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn seq_len(&self) -> usize {
        self.tokens.first().map_or(0, |r| r.len())
    }

    /// Number of supervised positions in the batch.
    pub fn supervised(&self) -> usize {
        self.mask
            .iter()
            .flat_map(|r| r.iter())
            .filter(|m| **m == 1.0)
            .count()
    }
}

/// Pretraining rows over one concatenated token stream. Each input text is followed by the
/// tokenizer's `<|eos|>` token, so documents never merge.
pub struct PretrainData {
    stream: Vec<u32>,
    seq_len: usize,
}

impl PretrainData {
    /// Tokenize `texts` in order, appending `<|eos|>` after each document.
    pub fn new(tok: &LmTokenizer, texts: &[String], seq_len: usize) -> Result<Self> {
        if seq_len == 0 {
            bail!("seq_len must be positive");
        }
        let eos = tok.special_id(EOS)?;
        let mut stream = Vec::new();
        for t in texts {
            stream.extend(tok.encode(t)?);
            stream.push(eos);
        }
        Ok(Self { stream, seq_len })
    }

    pub fn token_count(&self) -> usize {
        self.stream.len()
    }

    /// Cut the stream into consecutive rows of `seq_len` inputs with next-token targets and
    /// an all-ones mask. A trailing partial row is dropped. Batches are consecutive rows in
    /// generation order; the last batch may be smaller than `batch_size`.
    pub fn batches(&self, batch_size: usize) -> Result<Vec<Batch>> {
        if batch_size == 0 {
            bail!("batch_size must be positive");
        }
        let s = self.seq_len;
        let mut rows = Vec::new();
        let mut start = 0;
        while start + s < self.stream.len() {
            rows.push((
                self.stream[start..start + s].to_vec(),
                self.stream[start + 1..start + s + 1].to_vec(),
                vec![1.0f32; s],
            ));
            start += s;
        }
        if rows.is_empty() {
            bail!(
                "token stream has {} tokens; more than {s} are needed for one row",
                self.stream.len()
            );
        }
        Ok(chunk(rows, batch_size))
    }
}

/// SFT rows over serialized chat conversations with assistant-only loss masks.
pub struct SftData {
    rows: Vec<(Vec<u32>, Vec<u8>)>,
    seq_len: usize,
}

impl SftData {
    /// Serialize each conversation with [`crate::chat`]. Conversations longer than
    /// `seq_len + 1` tokens are truncated to that length. A conversation whose loss mask is
    /// all zeros is rejected: it would contribute nothing to training.
    pub fn new(tok: &LmTokenizer, conversations: &[Vec<Message>], seq_len: usize) -> Result<Self> {
        if seq_len == 0 {
            bail!("seq_len must be positive");
        }
        let mut rows = Vec::new();
        for (i, convo) in conversations.iter().enumerate() {
            let s = chat::serialize(tok, convo)
                .map_err(|e| anyhow::anyhow!("conversation {i}: {e}"))?;
            if !s.loss_mask.contains(&1) {
                bail!("conversation {i} has no supervised tokens");
            }
            let n = s.ids.len().min(seq_len + 1);
            if n < 2 {
                bail!("conversation {i} serializes to fewer than 2 tokens");
            }
            rows.push((s.ids[..n].to_vec(), s.loss_mask[..n].to_vec()));
        }
        if rows.is_empty() {
            bail!("no conversations given");
        }
        Ok(Self { rows, seq_len })
    }

    pub fn supervised_tokens(&self) -> usize {
        self.rows
            .iter()
            .flat_map(|(_, m)| m.iter())
            .filter(|m| **m == 1)
            .count()
    }

    /// Rows of `seq_len`: inputs, next-token targets, and the loss mask shifted with its
    /// target (position i supervises the prediction of `ids[i + 1]`). Short rows are padded
    /// with the pad id and mask 0.
    pub fn batches(&self, tok: &LmTokenizer, batch_size: usize) -> Result<Vec<Batch>> {
        if batch_size == 0 {
            bail!("batch_size must be positive");
        }
        let pad = tok.special_id(PAD)?;
        let s = self.seq_len;
        let mut rows = Vec::new();
        for (ids, mask) in &self.rows {
            let n = ids.len();
            let mut tokens = ids[..n - 1].to_vec();
            let mut targets = ids[1..n].to_vec();
            let mut m: Vec<f32> = mask[1..n].iter().map(|v| *v as f32).collect();
            tokens.resize(s, pad);
            targets.resize(s, pad);
            m.resize(s, 0.0);
            rows.push((tokens, targets, m));
        }
        Ok(chunk(rows, batch_size))
    }
}

fn chunk(rows: Vec<(Vec<u32>, Vec<u32>, Vec<f32>)>, batch_size: usize) -> Vec<Batch> {
    let mut out = Vec::new();
    let (mut tokens, mut targets, mut mask) = (Vec::new(), Vec::new(), Vec::new());
    for (t, g, m) in rows {
        tokens.push(t);
        targets.push(g);
        mask.push(m);
        if tokens.len() == batch_size {
            out.push(Batch {
                tokens: std::mem::take(&mut tokens),
                targets: std::mem::take(&mut targets),
                mask: std::mem::take(&mut mask),
            });
        }
    }
    if !tokens.is_empty() {
        out.push(Batch {
            tokens,
            targets,
            mask,
        });
    }
    out
}

/// Write conversations as JSONL: one conversation (a JSON array of messages) per line.
pub fn save_conversations(path: &std::path::Path, conversations: &[Vec<Message>]) -> Result<()> {
    let mut buf = String::new();
    for c in conversations {
        buf.push_str(&serde_json::to_string(c)?);
        buf.push('\n');
    }
    std::fs::write(path, buf)?;
    Ok(())
}

/// Read conversations written by [`save_conversations`].
pub fn load_conversations(path: &std::path::Path) -> Result<Vec<Vec<Message>>> {
    let raw = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("{}:{}: {e}", path.display(), i + 1))?,
        );
    }
    Ok(out)
}
