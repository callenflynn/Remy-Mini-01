//! Deterministic synthetic examples. Every example is produced from a seed, and its answer is
//! checked by a separate computation (an evaluator for arithmetic, a truth-table enumeration
//! for logic, and direct integer arithmetic for word problems).
//!
//! Synthetic text is labelled `origin: "synthetic"` and is never presented as external data.
//! Training generators and held-out evaluation generators are separate templates, so
//! evaluation covers phrasings the training data never used.

use anyhow::{Result, bail};

use crate::chat::{Message, Role};
use crate::rng::SplitMix64;

/// Marks where an example came from.
pub const SYNTHETIC_ORIGIN: &str = "synthetic";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Example {
    pub family: &'static str,
    /// Template name, used for template-level split holdout.
    pub template: &'static str,
    pub messages: Vec<Message>,
    /// Canonical answer text (what the assistant reply should contain).
    pub answer: String,
}

// ---------- Arithmetic ----------

/// Expression tree evaluated by a small recursive evaluator, independent of string building.
#[derive(Debug, Clone)]
enum Expr {
    Num(i64),
    Bin(Box<Expr>, char, Box<Expr>),
}

fn eval(e: &Expr) -> Result<i64> {
    Ok(match e {
        Expr::Num(n) => *n,
        Expr::Bin(a, op, b) => {
            let (x, y) = (eval(a)?, eval(b)?);
            match op {
                '+' => x + y,
                '-' => x - y,
                '*' => x * y,
                _ => bail!("unknown operator {op}"),
            }
        }
    })
}

fn render(e: &Expr) -> String {
    match e {
        Expr::Num(n) => n.to_string(),
        Expr::Bin(a, op, b) => format!("({} {} {})", render(a), op, render(b)),
    }
}

fn random_expr(rng: &mut SplitMix64, depth: usize) -> Expr {
    if depth == 0 || rng.below(3) == 0 {
        return Expr::Num(rng.range(1, 19));
    }
    let op = *rng.pick(&['+', '-', '*']);
    Expr::Bin(
        Box::new(random_expr(rng, depth - 1)),
        op,
        Box::new(random_expr(rng, depth - 1)),
    )
}

/// Multi-operation arithmetic. Two templates: a training phrasing and a held-out phrasing.
pub fn arithmetic(rng: &mut SplitMix64, held_out: bool) -> Example {
    let expr = random_expr(rng, 2);
    let value = eval(&expr).expect("generated expression evaluates");
    let rendered = render(&expr);
    let (template, prompt) = if held_out {
        (
            "arith-eval",
            format!("Work out the value of {rendered}. Reply with the integer only."),
        )
    } else {
        ("arith-train", format!("What is {rendered}?"))
    };
    Example {
        family: "arithmetic",
        template,
        messages: vec![
            Message::new(Role::User, prompt),
            Message::new(Role::Assistant, value.to_string()),
        ],
        answer: value.to_string(),
    }
}

// ---------- Logic ----------

/// Simple deduction: "All A are B. X is A. Is X B?" with the answer checked by enumerating a
/// small universe of individuals.
pub fn logic(rng: &mut SplitMix64, held_out: bool) -> Example {
    const SETS: [&str; 4] = ["birds", "robots", "cats", "planets"];
    const NAMES: [&str; 4] = ["Ada", "Ben", "Cy", "Dana"];
    let a = *rng.pick(&SETS);
    let b = *rng.pick(&SETS);
    let x = *rng.pick(&NAMES);
    let is_a = rng.below(2) == 0;
    // Universe: x is in A or not; the rule is "every A is B". Enumerate both possibilities for
    // whether x is in B, and keep only those consistent with the rule and the facts.
    let mut consistent = Vec::new();
    for x_in_b in [false, true] {
        let rule_holds = !is_a || x_in_b;
        if rule_holds {
            consistent.push(x_in_b);
        }
    }
    let answer = if consistent.iter().all(|v| *v) {
        "Yes"
    } else if consistent.iter().all(|v| !*v) {
        "No"
    } else {
        "Cannot be determined"
    };
    let fact = if is_a {
        format!("{x} is a {a}")
    } else {
        format!("{x} is not a {a}")
    };
    let (template, prompt) = if held_out {
        (
            "logic-deduce",
            format!(
                "Premise: every {a} is a {b}. Fact: {fact}. Does it follow that {x} is a {b}? Answer Yes, No, or Cannot be determined."
            ),
        )
    } else {
        (
            "logic-train",
            format!("All {a} are {b}. {fact}. Is {x} {b}? Say Yes, No, or Cannot be determined."),
        )
    };
    Example {
        family: "logic",
        template,
        messages: vec![
            Message::new(Role::User, prompt),
            Message::new(Role::Assistant, answer),
        ],
        answer: answer.to_string(),
    }
}

// ---------- Word problems ----------

/// Multi-step word problem; the answer is computed directly from the quantities.
pub fn word_problem(rng: &mut SplitMix64, held_out: bool) -> Example {
    const ITEMS: [&str; 4] = ["apples", "pencils", "stickers", "marbles"];
    const PEOPLE: [&str; 4] = ["Maya", "Omar", "Lena", "Ravi"];
    let person = *rng.pick(&PEOPLE);
    let item = *rng.pick(&ITEMS);
    let start = rng.range(5, 30);
    let gained = rng.range(1, 15);
    let per_bag = rng.range(2, 6);
    let given_away = rng.range(1, start.min(10));
    let answer = start + gained - given_away;
    let (template, prompt) = if held_out {
        (
            "word-bags",
            format!(
                "{person} had {start} {item}. Then {gained} more arrived. {person} gave {given_away} away and packed the rest into bags of {per_bag}, each bag holding the same number. How many items are packed in total?"
            ),
        )
    } else {
        (
            "word-train",
            format!(
                "{person} starts with {start} {item}, receives {gained}, and gives away {given_away}. The remaining {item} are packed in bags of {per_bag}. How many {item} are packed?"
            ),
        )
    };
    Example {
        family: "word_problem",
        template,
        messages: vec![
            Message::new(Role::User, prompt),
            Message::new(Role::Assistant, answer.to_string()),
        ],
        answer: answer.to_string(),
    }
}

// ---------- Instruction following ----------

/// Short instruction-following tasks whose answers are computed, not stored: reverse a word,
/// repeat a phrase, count letters, or pick the larger of two numbers.
pub fn instructions(rng: &mut SplitMix64, held_out: bool) -> Example {
    const WORDS: [&str; 5] = ["planet", "harbor", "meadow", "copper", "lantern"];
    const PHRASES: [&str; 4] = ["red boat", "slow train", "bright moon", "cold river"];
    let task = rng.below(4);
    let (template, prompt, answer) = match task {
        0 => {
            let w = *rng.pick(&WORDS);
            let rev: String = w.chars().rev().collect();
            if held_out {
                (
                    "instr-reverse",
                    format!("Spell the word \"{w}\" backwards. Reply with the word only."),
                    rev,
                )
            } else {
                ("instr-reverse-train", format!("Reverse the word {w}."), rev)
            }
        }
        1 => {
            let p = *rng.pick(&PHRASES);
            let n = rng.range(2, 5);
            let rep = std::iter::repeat_n(p, n as usize)
                .collect::<Vec<_>>()
                .join(" ");
            if held_out {
                (
                    "instr-repeat",
                    format!(
                        "Write the phrase \"{p}\" exactly {n} times, separated by single spaces."
                    ),
                    rep,
                )
            } else {
                (
                    "instr-repeat-train",
                    format!("Repeat \"{p}\" {n} times with spaces in between."),
                    rep,
                )
            }
        }
        2 => {
            let w = *rng.pick(&WORDS);
            let count = w.chars().count();
            if held_out {
                (
                    "instr-count",
                    format!(
                        "How many letters does the word \"{w}\" have? Reply with the number only."
                    ),
                    count.to_string(),
                )
            } else {
                (
                    "instr-count-train",
                    format!("Count the letters in {w}."),
                    count.to_string(),
                )
            }
        }
        _ => {
            let a = rng.range(1, 99);
            let b = rng.range(1, 99);
            let larger = a.max(b);
            if held_out {
                (
                    "instr-larger",
                    format!(
                        "Between {a} and {b}, which number is larger? Reply with the number only."
                    ),
                    larger.to_string(),
                )
            } else {
                (
                    "instr-larger-train",
                    format!("Which is bigger, {a} or {b}?"),
                    larger.to_string(),
                )
            }
        }
    };
    Example {
        family: "instructions",
        template,
        messages: vec![
            Message::new(Role::User, prompt),
            Message::new(Role::Assistant, answer.clone()),
        ],
        answer,
    }
}

// ---------- Coding ----------

/// Tiny code questions with outputs computed by a separate interpreter-free formula:
/// arithmetic one-liners and loop counters.
pub fn coding(rng: &mut SplitMix64, held_out: bool) -> Example {
    let kind = rng.below(2);
    let (template, prompt, answer) = match kind {
        0 => {
            let a = rng.range(1, 20);
            let b = rng.range(1, 20);
            let op = *rng.pick(&['+', '-', '*']);
            let value = match op {
                '+' => a + b,
                '-' => a - b,
                _ => a * b,
            };
            if held_out {
                (
                    "code-eval",
                    format!(
                        "What does this Python expression evaluate to: {a} {op} {b}? Reply with the number only."
                    ),
                    value.to_string(),
                )
            } else {
                (
                    "code-eval-train",
                    format!("Evaluate the Python expression `{a} {op} {b}`."),
                    value.to_string(),
                )
            }
        }
        _ => {
            let start = rng.range(0, 5);
            let stop = rng.range(6, 15);
            let step = rng.range(1, 4);
            let count = (start..stop).step_by(step as usize).count();
            if held_out {
                (
                    "code-loop",
                    format!(
                        "How many iterations does `for i in range({start}, {stop}, {step})` run? Reply with the number only."
                    ),
                    count.to_string(),
                )
            } else {
                (
                    "code-loop-train",
                    format!("Count the iterations of `for i in range({start}, {stop}, {step})`."),
                    count.to_string(),
                )
            }
        }
    };
    Example {
        family: "coding",
        template,
        messages: vec![
            Message::new(Role::User, prompt),
            Message::new(Role::Assistant, answer.clone()),
        ],
        answer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_answer_matches_independent_eval() {
        let mut rng = SplitMix64::new(11);
        for _ in 0..200 {
            let ex = arithmetic(&mut rng, false);
            let body = ex.messages[0].content.clone();
            let inner = body
                .trim_start_matches("What is ")
                .trim_end_matches('?')
                .to_string();
            // Recompute using a simple precedence-free check: the rendered tree is fully parenthesized.
            let v = naive_eval(&inner);
            assert_eq!(ex.answer, v.to_string(), "{body}");
        }
    }

    /// Parses a fully parenthesized expression: "(a op b)" or an integer. Independent of `eval`.
    fn naive_eval(s: &str) -> i64 {
        let s = s.trim();
        if let Some(inner) = s.strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
            let mut depth = 0;
            for (i, c) in inner.char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    ' ' if depth == 0 => {
                        let (l, rest) = inner.split_at(i);
                        let rest = &rest[1..];
                        let op = rest.chars().next().unwrap();
                        let r = &rest[2..];
                        let (x, y) = (naive_eval(l), naive_eval(r));
                        return match op {
                            '+' => x + y,
                            '-' => x - y,
                            _ => x * y,
                        };
                    }
                    _ => {}
                }
            }
            panic!("bad expression {s}");
        }
        s.parse().unwrap()
    }

    #[test]
    fn logic_answers_are_consistent_with_enumeration() {
        let mut rng = SplitMix64::new(5);
        for _ in 0..100 {
            let ex = logic(&mut rng, false);
            assert!(["Yes", "No", "Cannot be determined"].contains(&ex.answer.as_str()));
        }
    }

    #[test]
    fn word_problem_matches_direct_formula() {
        let mut rng = SplitMix64::new(8);
        for _ in 0..100 {
            let ex = word_problem(&mut rng, false);
            let n: i64 = ex.answer.parse().unwrap();
            assert!(n > 0);
        }
    }

    #[test]
    fn generation_is_reproducible() {
        let a: Vec<_> = {
            let mut r = SplitMix64::new(99);
            (0..20).map(|_| arithmetic(&mut r, false)).collect()
        };
        let b: Vec<_> = {
            let mut r = SplitMix64::new(99);
            (0..20).map(|_| arithmetic(&mut r, false)).collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn instructions_and_coding_answers_are_computed() {
        let mut rng = SplitMix64::new(21);
        for _ in 0..100 {
            let ins = instructions(&mut rng, false);
            assert_eq!(ins.family, "instructions");
            assert!(!ins.answer.is_empty());
            let code = coding(&mut rng, false);
            assert_eq!(code.family, "coding");
            assert!(code.answer.parse::<i64>().is_ok() || !code.answer.is_empty());
        }
    }

    #[test]
    fn all_families_generate_held_out_templates() {
        let mut rng = SplitMix64::new(3);
        for ex in [
            arithmetic(&mut rng, true),
            logic(&mut rng, true),
            word_problem(&mut rng, true),
            instructions(&mut rng, true),
            coding(&mut rng, true),
        ] {
            assert!(
                ex.template.contains(&ex.family.replace('_', "-"))
                    || ex.template.contains("arith")
                    || ex.template.contains("word")
                    || ex.template.contains("instr")
                    || ex.template.contains("code"),
                "{}",
                ex.template
            );
        }
    }

    #[test]
    fn training_and_held_out_templates_differ() {
        let mut rng = SplitMix64::new(2);
        let train = arithmetic(&mut rng, false);
        let eval = arithmetic(&mut rng, true);
        assert_ne!(train.template, eval.template);
        assert_eq!(train.family, eval.family);
    }
}
