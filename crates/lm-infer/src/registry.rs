//! Harness-side tool registry. The model never sees a tool that is not registered for the
//! current session, and every generated call is checked against its schema before execution.

use anyhow::{Result, bail};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::protocol::ToolCall;

/// A required argument and its expected JSON type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgType {
    String,
    Integer,
    Boolean,
}

#[derive(Debug, Clone, Default)]
pub struct ToolSchema {
    /// Required arguments. Unknown arguments are rejected.
    pub required: BTreeMap<&'static str, ArgType>,
}

#[derive(Debug, Clone, Default)]
pub struct Registry {
    tools: BTreeMap<String, ToolSchema>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, name: &str, schema: ToolSchema) {
        self.tools.insert(name.to_string(), schema);
    }

    /// Check a call's name exists and its arguments match the schema exactly.
    pub fn validate(&self, call: &ToolCall) -> Result<()> {
        let Some(schema) = self.tools.get(&call.name) else {
            bail!("unknown tool '{}'", call.name);
        };
        let Some(args) = call.arguments.as_object() else {
            bail!("arguments for '{}' must be an object", call.name);
        };
        for (key, ty) in &schema.required {
            match args.get(*key) {
                None => bail!("missing argument '{key}' for '{}'", call.name),
                Some(v) if !matches_type(v, *ty) => {
                    bail!("argument '{key}' for '{}' has wrong type", call.name)
                }
                Some(_) => {}
            }
        }
        for key in args.keys() {
            if !schema.required.contains_key(key.as_str()) {
                bail!("unexpected argument '{key}' for '{}'", call.name);
            }
        }
        Ok(())
    }
}

fn matches_type(v: &Value, ty: ArgType) -> bool {
    match ty {
        ArgType::String => v.is_string(),
        ArgType::Integer => v.is_i64() || v.is_u64(),
        ArgType::Boolean => v.is_boolean(),
    }
}
