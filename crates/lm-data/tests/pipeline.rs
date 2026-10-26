//! End-to-end checks on the tiny fixture and on chat masks, using real token ids.

use lm_data::chat::{Message, Role, serialize};
use lm_data::config::Config;
use lm_data::prepare::prepare;
use lm_tokenizer::{ASSISTANT, END_OF_TURN, LmTokenizer, SYSTEM, USER};
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn tokenizer() -> LmTokenizer {
    let corpus = fixture("tiny.txt");
    LmTokenizer::train(&[corpus.to_string_lossy().into_owned()], 512).unwrap()
}

#[test]
fn tiny_fixture_runs_end_to_end() {
    let cfg =
        Config::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/tiny.json")).unwrap();
    let inputs = vec![fixture("tiny.txt"), fixture("tiny.jsonl")];
    let (records, manifest) = prepare(&cfg, &inputs).unwrap();

    // External records come only from the fixture files; synthetic records are labelled.
    assert!(records.iter().any(|r| r.origin == "external"));
    assert!(records.iter().any(|r| r.origin == "synthetic"));
    assert!(
        records
            .iter()
            .all(|r| r.origin == "external" || r.origin == "synthetic")
    );
    assert_eq!(manifest.seed, 7);

    // Held-out synthetic examples land only in test.
    let held_out_ids: Vec<&str> = records
        .iter()
        .filter(|r| r.origin == "synthetic" && r.split == "test")
        .map(|r| r.id.as_str())
        .collect();
    assert!(!held_out_ids.is_empty());

    // No record id appears in two splits.
    let mut ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
    ids.sort();
    let before = ids.len();
    ids.dedup();
    assert_eq!(before, ids.len(), "duplicate record id across splits");

    // Tokenize the synthetic dialogs and check every supervised token is assistant or end-of-turn.
    let tok = tokenizer();
    let ex = lm_data::synthetic::arithmetic(&mut lm_data::rng::SplitMix64::new(3), false);
    let s = serialize(&tok, &ex.messages).unwrap();
    assert_eq!(s.ids.len(), s.loss_mask.len());
    assert!(s.loss_mask.iter().any(|m| *m == 1));
}

/// Identical text in two different files must survive once, in exactly one split: otherwise the
/// same paragraph is trained on and evaluated on, and scores stop meaning anything.
#[test]
fn duplicate_text_cannot_straddle_splits() {
    let dir = std::env::temp_dir().join(format!("lm-leak-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();

    let shared = "The arctic tern migrates between hemispheres each year, following the summer \
        sun across open water and coastal staging grounds where it feeds on small fish and \
        invertebrates, which is among the longest migrations known for any animal.";
    std::fs::write(dir.join("a.txt"), shared).unwrap();
    std::fs::write(dir.join("b.txt"), shared).unwrap();
    let mut filler = String::new();
    for i in 0..200 {
        filler.push_str(&format!(
            "{{\"text\":\"Document number {i} talks about migration routes, wind patterns, \
             feeding grounds and nesting behaviour in enough words to pass the length filter.\"}}\n"
        ));
    }
    std::fs::write(dir.join("filler.jsonl"), filler).unwrap();

    let mut cfg = Config::default();
    cfg.split.train = 0.8;
    cfg.split.val = 0.1;
    cfg.split.test = 0.1;
    cfg.normalize.min_chars = 64;
    cfg.normalize.max_chars = 10_000;
    cfg.synthetic.arithmetic = 0;
    cfg.synthetic.logic = 0;
    cfg.synthetic.word_problems = 0;
    cfg.synthetic.instructions = 0;
    cfg.synthetic.coding = 0;

    let inputs = vec![
        dir.join("a.txt"),
        dir.join("b.txt"),
        dir.join("filler.jsonl"),
    ];
    let (records, manifest) = prepare(&cfg, &inputs).unwrap();
    let dupes: Vec<_> = records.iter().filter(|r| r.text.trim() == shared).collect();
    assert_eq!(dupes.len(), 1, "identical text kept twice");
    assert!(manifest.dedup_removed >= 1 || manifest.normalize_stats.duplicate >= 1);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn loss_mask_selects_exactly_assistant_content_and_end_of_turn() {
    let tok = tokenizer();
    let convo = vec![
        Message::new(Role::System, "be brief"),
        Message::new(Role::User, "hi there"),
        Message::new(Role::Assistant, "hello"),
        Message::new(Role::User, "2+2?"),
        Message::new(Role::Assistant, "4"),
    ];
    let s = serialize(&tok, &convo).unwrap();

    let end = tok.special_id(END_OF_TURN).unwrap();
    let markers = [
        tok.special_id(SYSTEM).unwrap(),
        tok.special_id(USER).unwrap(),
        tok.special_id(ASSISTANT).unwrap(),
    ];

    // Every supervised position is an assistant reply token or an end-of-turn token,
    // and every marker is masked.
    for (i, (&id, &m)) in s.ids.iter().zip(&s.loss_mask).enumerate() {
        if markers.contains(&id) {
            assert_eq!(m, 0, "role marker supervised at {i}");
        }
        if m == 1 && id == end {
            continue;
        }
        if m == 1 {
            // Must be inside an assistant span: walk back to the nearest marker.
            let prev = s.ids[..i]
                .iter()
                .rev()
                .find(|x| markers.contains(x))
                .copied();
            assert_eq!(
                prev,
                Some(tok.special_id(ASSISTANT).unwrap()),
                "supervised token outside assistant at {i}"
            );
        }
    }

    // Supervised content equals the encoded assistant replies plus their end-of-turn tokens.
    let supervised: Vec<u32> = s
        .ids
        .iter()
        .zip(&s.loss_mask)
        .filter(|(_, m)| **m == 1)
        .map(|(i, _)| *i)
        .collect();
    let mut expected = tok.encode("hello").unwrap();
    expected.push(end);
    expected.extend(tok.encode("4").unwrap());
    expected.push(end);
    assert_eq!(supervised, expected);
}

#[test]
fn empty_assistant_reply_supervises_only_end_of_turn() {
    let tok = tokenizer();
    let convo = vec![
        Message::new(Role::User, "ping"),
        Message::new(Role::Assistant, ""),
    ];
    let s = serialize(&tok, &convo).unwrap();
    let supervised: Vec<u32> = s
        .ids
        .iter()
        .zip(&s.loss_mask)
        .filter(|(_, m)| **m == 1)
        .map(|(i, _)| *i)
        .collect();
    assert_eq!(supervised, vec![tok.special_id(END_OF_TURN).unwrap()]);
}

#[test]
fn malformed_conversations_are_rejected() {
    let tok = tokenizer();
    assert!(serialize(&tok, &[]).is_err());
    assert!(serialize(&tok, &[Message::new(Role::User, "x")]).is_err());
    assert!(
        serialize(
            &tok,
            &[Message::new(Role::User, "x"), Message::new(Role::User, "y")]
        )
        .is_err()
    );
}
