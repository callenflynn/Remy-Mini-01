//! Deterministic randomness. SplitMix64 is fully specified, so output does not depend on the
//! platform or on the version of any external RNG crate.

/// SplitMix64 generator.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Integer in `0..n`. Panics when `n == 0`. The modulo bias is negligible for the small ranges used here.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0)");
        (self.next_u64() % n as u64) as usize
    }

    /// Integer in the inclusive range `lo..=hi`.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        assert!(lo <= hi, "empty range");
        lo + self.below((hi - lo + 1) as usize) as i64
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

/// Stable 64-bit hash of `key` under `seed` (FNV-1a, then a SplitMix finalizer).
pub fn hash_key(seed: u64, key: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in key.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    SplitMix64::new(h).next_u64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn range_is_inclusive_and_bounded() {
        let mut r = SplitMix64::new(1);
        let mut seen_lo = false;
        let mut seen_hi = false;
        for _ in 0..2000 {
            let v = r.range(-3, 3);
            assert!((-3..=3).contains(&v));
            seen_lo |= v == -3;
            seen_hi |= v == 3;
        }
        assert!(seen_lo && seen_hi);
    }

    #[test]
    fn hash_depends_on_seed_and_key() {
        assert_eq!(hash_key(1, "abc"), hash_key(1, "abc"));
        assert_ne!(hash_key(1, "abc"), hash_key(2, "abc"));
        assert_ne!(hash_key(1, "abc"), hash_key(1, "abd"));
    }
}
