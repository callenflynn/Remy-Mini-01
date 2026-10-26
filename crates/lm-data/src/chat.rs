//! Chat serialization with assistant-only loss.
//!
//! Format (documented; uses the tokenizer's special tokens):
//!
//! ```text
//! <|system|>SYSTEM TEXT<|user|>USER TEXT<|assistant|>REPLY<|end_of_turn|>...
//! ```
//!
//! Only the assistant reply tokens and the `<|end_of_turn|>` that closes each assistant reply
//! carry loss. Role markers, the assistant header, and system/user content are masked out.
//! Every segment is tokenized on its own, so token boundaries are exact and the mask is built
//! from token ids, never from character offsets. Special tokens are added as whole ids, so
//! they never merge with neighbouring text.

use anyhow::{Result, bail};
use lm_tokenizer::{ASSISTANT, END_OF_TURN, LmTokenizer, SYSTEM, USER};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// Token ids of a serialized conversation, with a parallel loss mask (1 = supervised).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Serialized {
    pub ids: Vec<u32>,
    pub loss_mask: Vec<u8>,
}

/// Serialize a conversation. Rules:
/// - Each message starts with its role marker and then its content.
/// - Every assistant message is followed by `<|end_of_turn|>`.
/// - Roles must be system, then user/assistant turns in order. The conversation must end with
///   an assistant reply, so that the final supervised reply is a complete turn. A conversation
///   may contain no assistant reply, but then the loss mask is all zeros and the caller should
///   reject it for training.
/// - Empty content is allowed: an empty assistant reply yields only `<|end_of_turn|>` supervised.
pub fn serialize(tok: &LmTokenizer, messages: &[Message]) -> Result<Serialized> {
    validate_order(messages)?;
    let system_id = tok.special_id(SYSTEM)?;
    let user_id = tok.special_id(USER)?;
    let assistant_id = tok.special_id(ASSISTANT)?;
    let end_id = tok.special_id(END_OF_TURN)?;

    let mut ids = Vec::new();
    let mut loss_mask = Vec::new();
    let push = |segment: &[u32], supervised: u8, ids: &mut Vec<u32>, mask: &mut Vec<u8>| {
        ids.extend_from_slice(segment);
        mask.extend(std::iter::repeat_n(supervised, segment.len()));
    };

    for m in messages {
        let (marker, supervised_content) = match m.role {
            Role::System => (system_id, 0),
            Role::User => (user_id, 0),
            Role::Assistant => (assistant_id, 1),
        };
        // Role marker: never supervised.
        push(&[marker], 0, &mut ids, &mut loss_mask);
        let content_ids = tok.encode(&m.content)?;
        push(&content_ids, supervised_content, &mut ids, &mut loss_mask);
        if m.role == Role::Assistant {
            push(&[end_id], 1, &mut ids, &mut loss_mask);
        }
    }
    Ok(Serialized { ids, loss_mask })
}

fn validate_order(messages: &[Message]) -> Result<()> {
    if messages.is_empty() {
        bail!("conversation is empty");
    }
    let mut seen_non_system = false;
    for (i, m) in messages.iter().enumerate() {
        match m.role {
            Role::System => {
                if i != 0 {
                    bail!("system message at position {i}; system may only come first");
                }
            }
            Role::User | Role::Assistant => seen_non_system = true,
        }
    }
    if !seen_non_system {
        bail!("conversation has no user or assistant message");
    }
    let last = messages.last().expect("non-empty");
    if last.role != Role::Assistant {
        bail!("conversation must end with an assistant reply");
    }
    // Strict user/assistant alternation (system is ignored for this rule).
    let turns: Vec<Role> = messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(|m| m.role)
        .collect();
    for (i, pair) in turns.iter().enumerate() {
        let expected = if i % 2 == 0 {
            Role::User
        } else {
            Role::Assistant
        };
        if *pair != expected {
            bail!("turns must alternate user/assistant; turn {i} is {pair:?}");
        }
    }
    Ok(())
}
