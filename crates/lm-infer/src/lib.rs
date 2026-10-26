//! Inference-side tool protocol and a mock harness loop.
//!
//! The model crate, tokenizer and trainer do not depend on this crate's execution logic.
//! Tool execution happens only in a `ToolExecutor` supplied by the caller.

pub mod protocol;
pub mod registry;

use anyhow::Result;

use protocol::{PROTOCOL_VERSION, ToolCall, ToolResult, Turn, encode_result, parse_turn};
use registry::Registry;

/// Executes a validated call. Implementations live in the harness, not the model.
pub trait ToolExecutor {
    fn execute(&mut self, call: &ToolCall) -> Result<String>;
}

/// Produces assistant output given the transcript so far.
pub trait Generator {
    fn next_turn(&mut self, transcript: &str) -> Result<String>;
}

/// Run one conversation: generate, parse, validate, execute, feed results back.
/// Invalid calls and tool errors become controlled error results, never panics.
/// Returns the final transcript and the number of tool calls executed.
pub fn run<G: Generator, E: ToolExecutor>(
    generator: &mut G,
    registry: &Registry,
    executor: &mut E,
    prompt: &str,
    max_steps: usize,
) -> Result<(String, usize)> {
    let mut transcript = prompt.to_string();
    let mut executed = 0;
    for _ in 0..max_steps {
        let output = generator.next_turn(&transcript)?;
        transcript.push_str(&output);
        match parse_turn(&output) {
            Ok(Turn::Text(_)) => return Ok((transcript, executed)),
            Ok(Turn::Calls { calls, .. }) => {
                for call in &calls {
                    let result = match registry.validate(call) {
                        Err(e) => error_result(call, format!("invalid call: {e}")),
                        Ok(()) => {
                            executed += 1;
                            match executor.execute(call) {
                                Ok(content) => ToolResult {
                                    v: PROTOCOL_VERSION,
                                    name: call.name.clone(),
                                    ok: true,
                                    content,
                                },
                                Err(e) => error_result(call, format!("tool error: {e}")),
                            }
                        }
                    };
                    transcript.push_str(&encode_result(&result)?);
                }
            }
            Err(e) => {
                let result = ToolResult {
                    v: PROTOCOL_VERSION,
                    name: String::new(),
                    ok: false,
                    content: format!("malformed tool call: {e}"),
                };
                transcript.push_str(&encode_result(&result)?);
            }
        }
    }
    Ok((transcript, executed))
}

fn error_result(call: &ToolCall, content: String) -> ToolResult {
    ToolResult {
        v: PROTOCOL_VERSION,
        name: call.name.clone(),
        ok: false,
        content,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{CALL_CLOSE, CALL_OPEN};
    use registry::{ArgType, ToolSchema};
    use std::collections::BTreeMap;

    struct Scripted(Vec<String>);
    impl Generator for Scripted {
        fn next_turn(&mut self, _t: &str) -> Result<String> {
            Ok(self.0.remove(0))
        }
    }

    struct Echo(usize);
    impl ToolExecutor for Echo {
        fn execute(&mut self, call: &ToolCall) -> Result<String> {
            self.0 += 1;
            if call.name == "boom" {
                anyhow::bail!("network down");
            }
            Ok(format!("result for {}", call.arguments))
        }
    }

    fn registry() -> Registry {
        let mut r = Registry::new();
        let mut required = BTreeMap::new();
        required.insert("query", ArgType::String);
        r.register("web_search", ToolSchema { required });
        r.register("boom", ToolSchema::default());
        r
    }

    fn call(name: &str, args: &str) -> String {
        format!(r#"{CALL_OPEN}{{"v":1,"name":"{name}","arguments":{args}}}{CALL_CLOSE}"#)
    }

    #[test]
    fn direct_answer_runs_no_tools() {
        let mut g = Scripted(vec!["Paris.".into()]);
        let mut e = Echo(0);
        let (t, n) = run(&mut g, &registry(), &mut e, "Q?", 4).unwrap();
        assert_eq!((t.as_str(), n, e.0), ("Q?Paris.", 0, 0));
    }

    #[test]
    fn multi_step_search_then_answer() {
        let mut g = Scripted(vec![
            call("web_search", r#"{"query":"rust"}"#),
            "Rust is a language.".into(),
        ]);
        let mut e = Echo(0);
        let (t, n) = run(&mut g, &registry(), &mut e, "Q", 4).unwrap();
        assert_eq!(n, 1);
        assert!(t.contains("<tool_result>"));
        assert!(t.ends_with("Rust is a language."));
    }

    #[test]
    fn invalid_arguments_become_error_result_without_execution() {
        let mut g = Scripted(vec![call("web_search", r#"{"query":42}"#), "Sorry.".into()]);
        let mut e = Echo(0);
        let (t, n) = run(&mut g, &registry(), &mut e, "Q", 4).unwrap();
        assert_eq!((n, e.0), (0, 0));
        assert!(t.contains(r#""ok":false"#) && t.contains("wrong type"));
    }

    #[test]
    fn unknown_tool_is_rejected() {
        let mut g = Scripted(vec![call("rm_rf", "{}"), "Ok.".into()]);
        let mut e = Echo(0);
        let (t, _) = run(&mut g, &registry(), &mut e, "Q", 4).unwrap();
        assert!(t.contains("unknown tool 'rm_rf'"));
    }

    #[test]
    fn executor_failure_is_controlled() {
        let mut g = Scripted(vec![call("boom", "{}"), "Could not reach it.".into()]);
        let mut e = Echo(0);
        let (t, n) = run(&mut g, &registry(), &mut e, "Q", 4).unwrap();
        assert!(t.contains("tool error: network down"));
        assert_eq!(n, 1);
    }

    #[test]
    fn malformed_envelope_is_controlled() {
        let mut g = Scripted(vec![
            format!("{CALL_OPEN}not json{CALL_CLOSE}"),
            "Retrying.".into(),
        ]);
        let mut e = Echo(0);
        let (t, _) = run(&mut g, &registry(), &mut e, "Q", 4).unwrap();
        assert!(t.contains("malformed tool call"));
    }

    #[test]
    fn parse_keeps_text_around_calls() {
        let out = format!("Let me check. {}", call("web_search", r#"{"query":"x"}"#));
        match parse_turn(&out).unwrap() {
            Turn::Calls { text, calls } => {
                assert_eq!(text, "Let me check. ");
                assert_eq!(calls.len(), 1);
            }
            other => panic!("expected calls, got {other:?}"),
        }
    }

    #[test]
    fn unterminated_call_is_an_error() {
        assert!(parse_turn(&format!("{CALL_OPEN}{{\"v\":1")).is_err());
    }

    #[test]
    fn unsupported_version_is_an_error() {
        let out = format!(r#"{CALL_OPEN}{{"v":2,"name":"x","arguments":{{}}}}{CALL_CLOSE}"#);
        assert!(parse_turn(&out).is_err());
    }

    #[test]
    fn stray_result_marker_in_text_is_an_error() {
        assert!(parse_turn("see </tool_result> here").is_err());
    }
}
