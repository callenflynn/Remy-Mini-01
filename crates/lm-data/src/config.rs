//! Pipeline configuration, loaded from a JSON file. Every field has a documented default, and
//! `validate` rejects invalid combinations with a message that names the field.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::license::SourceMeta;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Seed for synthetic generation and splitting.
    pub seed: u64,
    pub split: SplitConfig,
    pub normalize: NormalizeConfig,
    pub synthetic: SyntheticConfig,
    /// Declared external sources. Every entry is validated by `license::SourceMeta::validate`.
    pub sources: Vec<SourceMeta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SplitConfig {
    /// `"hash"` assigns each record by hashing its id with the seed. `"template"` holds out
    /// whole synthetic templates for evaluation.
    pub strategy: String,
    pub train: f64,
    pub val: f64,
    pub test: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NormalizeConfig {
    /// Documents shorter than this (after trimming) are dropped.
    pub min_chars: usize,
    /// Documents longer than this are dropped, not truncated.
    pub max_chars: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SyntheticConfig {
    pub arithmetic: usize,
    pub logic: usize,
    pub word_problems: usize,
    pub instructions: usize,
    pub coding: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 1,
            split: SplitConfig::default(),
            normalize: NormalizeConfig::default(),
            synthetic: SyntheticConfig::default(),
            sources: Vec::new(),
        }
    }
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            strategy: "hash".into(),
            train: 0.98,
            val: 0.01,
            test: 0.01,
        }
    }
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self {
            min_chars: 20,
            max_chars: 100_000,
        }
    }
}

impl Default for SyntheticConfig {
    /// Small defaults for tests and smoke runs. These are not enough to train a capable model.
    fn default() -> Self {
        Self {
            arithmetic: 200,
            logic: 100,
            word_problems: 100,
            instructions: 100,
            coding: 100,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: Config = serde_json::from_str(&text)
            .with_context(|| format!("parsing config {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let s = &self.split;
        match s.strategy.as_str() {
            "hash" => {}
            "template" => {
                bail!("split.strategy \"template\" is not wired into prepare yet; use \"hash\"")
            }
            other => bail!("split.strategy must be \"hash\", got \"{other}\""),
        }
        for (name, v) in [("train", s.train), ("val", s.val), ("test", s.test)] {
            if !(0.0..=1.0).contains(&v) {
                bail!("split.{name} must be between 0 and 1, got {v}");
            }
        }
        let sum = s.train + s.val + s.test;
        if (sum - 1.0).abs() > 1e-9 {
            bail!("split ratios must sum to 1.0, got {sum}");
        }
        if s.train <= 0.0 {
            bail!("split.train must be greater than 0");
        }
        if self.normalize.min_chars > self.normalize.max_chars {
            bail!("normalize.min_chars must not exceed normalize.max_chars");
        }
        for src in &self.sources {
            src.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_use_98_1_1() {
        let c = Config::default();
        c.validate().unwrap();
        assert_eq!(c.split.train, 0.98);
        assert_eq!(c.split.val, 0.01);
        assert_eq!(c.split.test, 0.01);
    }

    #[test]
    fn bad_ratios_are_rejected() {
        let mut c = Config::default();
        c.split.train = 0.5;
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("sum to 1.0"), "{err}");
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let r: Result<Config, _> = serde_json::from_str(r#"{"sed": 3}"#);
        assert!(r.is_err());
    }

    #[test]
    fn unknown_strategy_is_rejected() {
        let mut c = Config::default();
        c.split.strategy = "random".into();
        assert!(c.validate().is_err());
    }

    #[test]
    fn template_strategy_is_rejected_until_wired() {
        let mut c = Config::default();
        c.split.strategy = "template".into();
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("not wired"), "{err}");
    }
}
