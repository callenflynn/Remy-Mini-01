//! Versioned tool-call protocol.
//!
//! The model emits tool calls as one JSON envelope inside an assistant turn, between the
//! markers below. Ordinary assistant text never contains these markers unless it is
//! deliberately emitting a call, so parsing is unambiguous:
//!
//! ```text
//! <tool_call>{"v":1,"name":"web_search","arguments":{"query":"rust"}}</tool_call>
//! ```
//!
//! Tool results come back from the harness wrapped the same way:
//!
//! ```text
//! <tool_result>{"v":1,"name":"web_search","ok":true,"content":"..."}</tool_result>
//! ```
//!
//! The model never executes tools. The harness parses, validates, executes and replies.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PROTOCOL_VERSION: u32 = 1;
pub const CALL_OPEN: &str = "<tool_call>";
pub const CALL_CLOSE: &str = "</tool_call>";
pub const RESULT_OPEN: &str = "<tool_result>";
pub const RESULT_CLOSE: &str = "</tool_result>";

/// A tool call as emitted by the model, before validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub v: u32,
    pub name: String,
    pub arguments: Value,
}

/// A tool result sent back to the model by the harness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub v: u32,
    pub name: String,
    pub ok: bool,
    pub content: String,
}

/// Result of scanning assistant text for tool calls.
#[derive(Debug, Clone, PartialEq)]
pub enum Turn {
    /// Plain text with no tool call markers.
    Text(String),
    /// Text with one or more calls. `text` is everything outside the call markers.
    Calls { text: String, calls: Vec<ToolCall> },
}

/// Parse one assistant turn. Malformed envelopes are errors, never silently treated as text,
/// so the harness can return a controlled error result instead of guessing.
pub fn parse_turn(output: &str) -> Result<Turn> {
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut rest = output;
    while let Some(start) = rest.find(CALL_OPEN) {
        if rest[..start].contains(CALL_CLOSE) || rest[..start].contains(RESULT_OPEN) {
            bail!("stray marker before tool call");
        }
        text.push_str(&rest[..start]);
        let body_start = start + CALL_OPEN.len();
        let Some(len) = rest[body_start..].find(CALL_CLOSE) else {
            bail!("unterminated tool call");
        };
        let body = &rest[body_start..body_start + len];
        let call: ToolCall = serde_json::from_str(body.trim())
            .map_err(|e| anyhow::anyhow!("malformed tool call JSON: {e}"))?;
        if call.v != PROTOCOL_VERSION {
            bail!("unsupported tool protocol version {}", call.v);
        }
        calls.push(call);
        rest = &rest[body_start + len + CALL_CLOSE.len()..];
    }
    if rest.contains(CALL_CLOSE) || rest.contains(RESULT_OPEN) || rest.contains(RESULT_CLOSE) {
        bail!("stray tool marker in assistant text");
    }
    text.push_str(rest);
    if calls.is_empty() {
        Ok(Turn::Text(text))
    } else {
        Ok(Turn::Calls { text, calls })
    }
}

/// Encode a tool call in the model's output format. Kept next to the parser so emitters
/// (including the synthetic-data generator) cannot drift from it.
pub fn encode_call(call: &ToolCall) -> Result<String> {
    Ok(format!(
        "{CALL_OPEN}{}{CALL_CLOSE}",
        serde_json::to_string(call)?
    ))
}

/// Encode a tool result for the model's context.
pub fn encode_result(result: &ToolResult) -> Result<String> {
    Ok(format!(
        "{RESULT_OPEN}{}{RESULT_CLOSE}",
        serde_json::to_string(result)?
    ))
}
