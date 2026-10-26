//! Byte-level BPE tokenizer with explicit chat role special tokens.

use anyhow::{Context, Result, anyhow};
use std::path::Path;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BPE, BpeTrainer, BpeTrainerBuilder};
use tokenizers::models::{ModelWrapper, TrainerWrapper};
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{AddedToken, TokenizerBuilder, TokenizerImpl};

type Tokenizer = TokenizerImpl<
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

pub const PAD: &str = "<|pad|>";
pub const BOS: &str = "<|bos|>";
pub const EOS: &str = "<|eos|>";
pub const SYSTEM: &str = "<|system|>";
pub const USER: &str = "<|user|>";
pub const ASSISTANT: &str = "<|assistant|>";
pub const END_OF_TURN: &str = "<|end_of_turn|>";

/// Special tokens in a fixed order; their ids are 0..SPECIAL_TOKENS.len().
pub const SPECIAL_TOKENS: [&str; 7] = [PAD, BOS, EOS, SYSTEM, USER, ASSISTANT, END_OF_TURN];

pub struct LmTokenizer {
    inner: Tokenizer,
}

impl LmTokenizer {
    /// Train a byte-level BPE tokenizer on the given text files.
    pub fn train(files: &[String], vocab_size: usize) -> Result<Self> {
        let special: Vec<AddedToken> = SPECIAL_TOKENS
            .iter()
            .map(|s| AddedToken::from(*s, true))
            .collect();
        let trainer: BpeTrainer = BpeTrainerBuilder::new()
            .vocab_size(vocab_size)
            .min_frequency(2)
            .special_tokens(special)
            .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
            .show_progress(false)
            .build();
        let mut tok: Tokenizer = TokenizerBuilder::<
            ModelWrapper,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        >::new()
        .with_model(ModelWrapper::BPE(BPE::default()))
        .with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(ByteLevel::new(
            false, true, true,
        ))))
        .with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevelDecoder::default())))
        .build()
        .map_err(|e| anyhow!("{e}"))?;
        let mut trainer = TrainerWrapper::BpeTrainer(trainer);
        tok.train_from_files(&mut trainer, files.to_vec())
            .map_err(|e| anyhow!("{e}"))?;
        tok.add_special_tokens(
            &SPECIAL_TOKENS
                .iter()
                .map(|s| AddedToken::from(*s, true))
                .collect::<Vec<_>>(),
        );
        Ok(Self { inner: tok })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("loading tokenizer {}", path.display()))?;
        Ok(Self { inner })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.inner.save(path, false).map_err(|e| anyhow!("{e}"))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn special_id(&self, token: &str) -> Result<u32> {
        self.inner
            .token_to_id(token)
            .ok_or_else(|| anyhow!("missing special token {token}"))
    }

    /// Encode plain text without adding special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, false).map_err(|e| anyhow!("{e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode ids. Special tokens are kept as their literal text when `skip_special` is false.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| anyhow!("{e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn trained() -> LmTokenizer {
        let dir = std::env::temp_dir().join(format!("lmtok-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        for _ in 0..50 {
            writeln!(
                f,
                "The quick brown fox jumps over the lazy dog. 12 + 7 = 19. café naïve"
            )
            .unwrap();
        }
        let t = LmTokenizer::train(&[path.to_string_lossy().into_owned()], 512).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        t
    }

    #[test]
    fn special_tokens_have_fixed_low_ids() {
        let t = trained();
        for (i, s) in SPECIAL_TOKENS.iter().enumerate() {
            assert_eq!(t.special_id(s).unwrap(), i as u32, "{s}");
        }
    }

    #[test]
    fn round_trip_plain_and_unicode() {
        let t = trained();
        for text in [
            "The quick brown fox jumps over the lazy dog.",
            "café naïve 日本語 🙂",
            "  leading and  double  spaces\n\nnewline",
            "",
        ] {
            let ids = t.encode(text).unwrap();
            assert_eq!(t.decode(&ids, true).unwrap(), text);
        }
    }

    #[test]
    fn special_tokens_round_trip_as_single_ids() {
        let t = trained();
        let text = format!("{USER}hi{END_OF_TURN}");
        let ids = t.encode(&text).unwrap();
        assert_eq!(ids[0], t.special_id(USER).unwrap());
        assert_eq!(*ids.last().unwrap(), t.special_id(END_OF_TURN).unwrap());
        assert_eq!(t.decode(&ids, false).unwrap(), text);
    }

    #[test]
    fn save_and_load_preserve_encoding() {
        let t = trained();
        let path = std::env::temp_dir().join(format!("lmtok-save-{}.json", std::process::id()));
        t.save(&path).unwrap();
        let t2 = LmTokenizer::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let s = "The lazy dog, 12 + 7 = 19";
        assert_eq!(t.encode(s).unwrap(), t2.encode(s).unwrap());
    }
}
