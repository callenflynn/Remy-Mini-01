//! Loader tests: deterministic batching, correct target/mask alignment, cross-split dedup.

use lm_data::chat::{Message, Role};
use lm_data::loader::{PretrainData, SftData, load_conversations, save_conversations};
use lm_data::prepare::{Record, dedupe_by_text, prepare};
use lm_tokenizer::{END_OF_TURN, LmTokenizer, PAD};
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

fn texts() -> Vec<String> {
    vec![
        "the river bends past the old mill".to_string(),
        "a small model learns patterns from many sentences".to_string(),
        "careful readers check each claim before repeating it".to_string(),
    ]
}

#[test]
fn pretrain_batches_are_deterministic_and_aligned() {
    let tok = tokenizer();
    let a = PretrainData::new(&tok, &texts(), 8).unwrap();
    let b = PretrainData::new(&tok, &texts(), 8).unwrap();
    assert_eq!(a.token_count(), b.token_count());
    let ba = a.batches(2).unwrap();
    let bb = b.batches(2).unwrap();
    assert_eq!(ba, bb, "fixed inputs must give identical batches");

    for batch in &ba {
        assert_eq!(batch.seq_len(), 8);
        assert!(batch.batch_size() <= 2);
        for (row, (t, g)) in batch.tokens.iter().zip(&batch.targets).enumerate() {
            // Next-token alignment: targets are tokens shifted left by one within the stream.
            let _ = row;
            assert_eq!(t.len(), g.len());
        }
        assert_eq!(
            batch.supervised(),
            batch.batch_size() * batch.seq_len(),
            "pretraining supervises every position"
        );
    }
}

#[test]
fn pretrain_targets_are_the_following_tokens() {
    let tok = tokenizer();
    let data = PretrainData::new(&tok, &["hello world".into()], 4).unwrap();
    let mut stream = tok.encode("hello world").unwrap();
    stream.push(tok.special_id(lm_tokenizer::EOS).unwrap());
    let batches = data.batches(1).unwrap();
    let b = &batches[0];
    assert_eq!(b.tokens[0], stream[0..4].to_vec());
    assert_eq!(b.targets[0], stream[1..5].to_vec());
}

#[test]
fn sft_masks_only_assistant_content_and_end_of_turn() {
    let tok = tokenizer();
    let end = tok.special_id(END_OF_TURN).unwrap();
    let convo = vec![
        Message::new(Role::User, "2+2?"),
        Message::new(Role::Assistant, "4"),
    ];
    let data = SftData::new(&tok, &[convo], 16).unwrap();
    let batches = data.batches(&tok, 1).unwrap();
    let b = &batches[0];
    // Supervised targets are exactly the assistant tokens plus the end-of-turn.
    let supervised: Vec<u32> = b.targets[0]
        .iter()
        .zip(&b.mask[0])
        .filter(|(_, m)| **m == 1.0)
        .map(|(t, _)| *t)
        .collect();
    let mut expected = tok.encode("4").unwrap();
    expected.push(end);
    assert_eq!(supervised, expected);
}

#[test]
fn sft_batches_are_deterministic_and_padding_is_masked() {
    let tok = tokenizer();
    let pad = tok.special_id(PAD).unwrap();
    let convos = vec![
        vec![
            Message::new(Role::User, "say hi"),
            Message::new(Role::Assistant, "hi"),
        ],
        vec![
            Message::new(
                Role::User,
                "a much longer question that needs more tokens than the first",
            ),
            Message::new(Role::Assistant, "a longer reply as well"),
        ],
    ];
    let a = SftData::new(&tok, &convos, 32)
        .unwrap()
        .batches(&tok, 2)
        .unwrap();
    let b = SftData::new(&tok, &convos, 32)
        .unwrap()
        .batches(&tok, 2)
        .unwrap();
    assert_eq!(a, b);
    let batch = &a[0];
    assert_eq!(batch.batch_size(), 2);
    for r in 0..batch.batch_size() {
        for i in 0..batch.seq_len() {
            if batch.tokens[r][i] == pad {
                assert_eq!(batch.mask[r][i], 0.0, "pad position must be masked");
            }
        }
    }
    assert!(batch.supervised() > 0);
}

#[test]
fn sft_rejects_conversations_without_supervision() {
    let tok = tokenizer();
    // A conversation ending with an assistant reply is required by chat::serialize; a
    // user-only conversation is rejected before masking is even computed.
    let bad = vec![vec![Message::new(Role::User, "hello")]];
    assert!(SftData::new(&tok, &bad, 16).is_err());
}

#[test]
fn conversations_round_trip_through_jsonl() {
    let dir = std::env::temp_dir().join(format!("lmloader-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("conversations.jsonl");
    let convos = vec![vec![
        Message::new(Role::User, "q"),
        Message::new(Role::Assistant, "a"),
    ]];
    save_conversations(&path, &convos).unwrap();
    let back = load_conversations(&path).unwrap();
    assert_eq!(back, convos);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn dedupe_removes_exact_duplicate_texts() {
    let mut records = vec![
        Record {
            id: "a".into(),
            origin: "external".into(),
            family: "document".into(),
            split: "train".into(),
            text: "same text".into(),
        },
        Record {
            id: "b".into(),
            origin: "synthetic".into(),
            family: "arithmetic".into(),
            split: "val".into(),
            text: "same text".into(),
        },
        Record {
            id: "c".into(),
            origin: "external".into(),
            family: "document".into(),
            split: "test".into(),
            text: "different text".into(),
        },
    ];
    let removed = dedupe_by_text(&mut records);
    assert_eq!(removed, 1);
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].id, "a");
}

#[test]
fn prepare_output_has_no_text_in_two_splits() {
    let cfg = lm_data::config::Config::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/tiny.json"),
    )
    .unwrap();
    let inputs = vec![fixture("tiny.txt"), fixture("tiny.jsonl")];
    let (records, _manifest) = prepare(&cfg, &inputs).unwrap();
    let mut split_of: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for r in &records {
        if let Some(prev) = split_of.get(r.text.as_str()) {
            assert_eq!(
                *prev,
                r.split.as_str(),
                "text leaks across splits: {}",
                r.text
            );
        } else {
            split_of.insert(r.text.as_str(), r.split.as_str());
        }
    }
}

#[test]
fn fixed_seed_prepare_is_reproducible() {
    let cfg = lm_data::config::Config::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/tiny.json"),
    )
    .unwrap();
    let inputs = vec![fixture("tiny.txt"), fixture("tiny.jsonl")];
    let (a, ma) = prepare(&cfg, &inputs).unwrap();
    let (b, mb) = prepare(&cfg, &inputs).unwrap();
    let key = |r: &Record| (r.id.clone(), r.split.clone(), r.text.clone());
    assert_eq!(
        a.iter().map(key).collect::<Vec<_>>(),
        b.iter().map(key).collect::<Vec<_>>()
    );
    assert_eq!(ma.dedup_removed, mb.dedup_removed);
}
