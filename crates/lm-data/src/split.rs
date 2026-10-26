//! Deterministic, seeded splits.
//!
//! - `hash`: each record goes to train/val/test from `hash(seed, id)`.
//! - `template`: whole templates are assigned to a split, so near-identical examples never
//!   straddle the train/evaluation boundary.
//!
//! Splits never duplicate records to fill a bucket. If a requested non-zero split would be
//! empty, the split fails with an error that names the split and the size of the input.

use anyhow::{Result, bail};
use std::collections::HashSet;

use crate::rng::hash_key;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bucket {
    Train,
    Val,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ratios {
    pub train: f64,
    pub val: f64,
    pub test: f64,
}

/// Reject duplicate keys. Returns the first duplicate found.
pub fn check_unique<'a>(keys: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut seen = HashSet::new();
    for k in keys {
        if !seen.insert(k) {
            bail!("duplicate record id '{k}'");
        }
    }
    Ok(())
}

/// Map a key to a bucket by hashing it into [0, 1) and comparing against the cumulative ratios.
pub fn bucket_for(seed: u64, key: &str, r: Ratios) -> Bucket {
    let u = (hash_key(seed, key) >> 11) as f64 / (1u64 << 53) as f64;
    if u < r.train {
        Bucket::Train
    } else if u < r.train + r.val {
        Bucket::Val
    } else {
        Bucket::Test
    }
}

/// Assign each key to a bucket with `hash` strategy. Fails when a non-zero ratio ends up empty.
pub fn split_hash(keys: &[String], seed: u64, r: Ratios) -> Result<Vec<Bucket>> {
    check_unique(keys.iter().map(String::as_str))?;
    let assigned: Vec<Bucket> = keys.iter().map(|k| bucket_for(seed, k, r)).collect();
    ensure_nonempty(&assigned, r, keys.len())?;
    Ok(assigned)
}

/// Assign whole templates to buckets. `template_of` maps each record to its template name.
/// Templates are ordered by hash, then assigned to buckets with the same cumulative rule.
pub fn split_by_template(template_of: &[String], seed: u64, r: Ratios) -> Result<Vec<Bucket>> {
    let mut names: Vec<&str> = template_of.iter().map(String::as_str).collect();
    names.sort();
    names.dedup();
    let assign: Vec<(String, Bucket)> = names
        .iter()
        .map(|n| (n.to_string(), bucket_for(seed, n, r)))
        .collect();
    let assigned: Vec<Bucket> = template_of
        .iter()
        .map(|t| {
            assign
                .iter()
                .find(|(n, _)| n == t)
                .map(|(_, b)| *b)
                .expect("template present")
        })
        .collect();
    ensure_nonempty(&assigned, r, template_of.len())?;
    Ok(assigned)
}

fn ensure_nonempty(assigned: &[Bucket], r: Ratios, total: usize) -> Result<()> {
    let count = |b: Bucket| assigned.iter().filter(|x| **x == b).count();
    for (name, bucket, ratio) in [
        ("train", Bucket::Train, r.train),
        ("val", Bucket::Val, r.val),
        ("test", Bucket::Test, r.test),
    ] {
        if ratio > 0.0 && count(bucket) == 0 {
            bail!(
                "split '{name}' is empty with {total} records; add more data or change the split configuration"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: Ratios = Ratios {
        train: 0.98,
        val: 0.01,
        test: 0.01,
    };

    fn keys(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("doc#{i}")).collect()
    }

    #[test]
    fn same_seed_same_assignment() {
        let k = keys(500);
        assert_eq!(split_hash(&k, 7, R).unwrap(), split_hash(&k, 7, R).unwrap());
    }

    #[test]
    fn ratios_are_roughly_honored() {
        let k = keys(20_000);
        let a = split_hash(&k, 3, R).unwrap();
        let train = a.iter().filter(|b| **b == Bucket::Train).count() as f64;
        let frac = train / a.len() as f64;
        assert!((frac - 0.98).abs() < 0.01, "train fraction {frac}");
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let k = vec!["a".to_string(), "a".to_string()];
        let err = split_hash(&k, 1, R).unwrap_err().to_string();
        assert!(err.contains("duplicate"), "{err}");
    }

    #[test]
    fn tiny_dataset_fails_instead_of_duplicating() {
        let k = keys(3);
        let err = split_hash(&k, 1, R).unwrap_err().to_string();
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn whole_templates_stay_in_one_split() {
        let templates: Vec<String> = (0..3000).map(|i| format!("t{}", i % 300)).collect();
        let a = split_by_template(
            &templates,
            9,
            Ratios {
                train: 0.5,
                val: 0.25,
                test: 0.25,
            },
        )
        .unwrap();
        for t in 0..300 {
            let name = format!("t{t}");
            let mut buckets = templates
                .iter()
                .zip(&a)
                .filter(|(n, _)| **n == name)
                .map(|(_, b)| *b);
            let first = buckets.next().unwrap();
            assert!(
                buckets.all(|b| b == first),
                "template {name} split across buckets"
            );
        }
    }
}
