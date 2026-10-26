/// Linear warmup to `max_lr` over `warmup_steps`, then cosine decay to `min_lr`
/// at `total_steps`. Steps are 0-indexed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LrSchedule {
    pub max_lr: f64,
    pub min_lr: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl LrSchedule {
    pub fn lr_at(&self, step: usize) -> f64 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.max_lr * (step + 1) as f64 / self.warmup_steps as f64;
        }
        if step >= self.total_steps {
            return self.min_lr;
        }
        let decay_steps = (self.total_steps - self.warmup_steps).max(1) as f64;
        let progress = (step - self.warmup_steps) as f64 / decay_steps;
        let cos = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        self.min_lr + (self.max_lr - self.min_lr) * cos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_is_warmup_fraction_and_final_step_is_min() {
        let s = LrSchedule {
            max_lr: 1.0,
            min_lr: 0.1,
            warmup_steps: 2,
            total_steps: 20,
        };
        assert!((s.lr_at(0) - 0.5).abs() < 1e-12);
        assert!((s.lr_at(1) - 1.0).abs() < 1e-12);
        assert!((s.lr_at(20) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn warmup_rises_then_cosine_decays_to_min() {
        let s = LrSchedule {
            max_lr: 1.0,
            min_lr: 0.1,
            warmup_steps: 10,
            total_steps: 110,
        };
        assert!((s.lr_at(0) - 0.1).abs() < 1e-12);
        assert!((s.lr_at(9) - 1.0).abs() < 1e-12);
        assert!((s.lr_at(10) - 1.0).abs() < 1e-9);
        let mid = s.lr_at(60);
        assert!(mid < 1.0 && mid > 0.1);
        assert!((s.lr_at(110) - 0.1).abs() < 1e-12);
        assert!((s.lr_at(500) - 0.1).abs() < 1e-12);
        for step in 10..109 {
            assert!(s.lr_at(step + 1) <= s.lr_at(step) + 1e-12);
        }
    }
}
