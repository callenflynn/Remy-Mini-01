use candle_core::backprop::GradStore;
use candle_core::{Result, Var};

/// Global L2 norm over all gradients of `vars`, computed in f32.
pub fn global_grad_norm(grads: &GradStore, vars: &[Var]) -> Result<f64> {
    let mut sq = 0f64;
    for v in vars {
        if let Some(g) = grads.get(v) {
            sq += g
                .sqr()?
                .sum_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?;
        }
    }
    Ok(sq.sqrt())
}

/// Scale all gradients in place (by replacement) so the global norm is at most `max_norm`.
/// Returns the pre-clip norm.
pub fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> Result<f64> {
    let norm = global_grad_norm(grads, vars)?;
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for v in vars {
            if let Some(g) = grads.remove(v) {
                grads.insert(v, (g * scale)?);
            }
        }
    }
    Ok(norm)
}
