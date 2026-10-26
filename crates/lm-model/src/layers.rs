use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Init, Module, VarBuilder};

/// RMSNorm with a learned scale and no bias, built from differentiable ops.
/// (candle_nn's fused rms_norm kernel has no backward pass, so it cannot train.)
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    // Scale starts at one; a zero scale would zero every block's output and block gradients.
    let weight = vb.get_with_hints(size, "weight", Init::Const(1.0))?;
    Ok(RmsNorm { weight, eps })
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = xf.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normed.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

/// Precomputed RoPE cos/sin tables of shape [context_len, head_dim / 2].
pub struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    pub fn new(
        head_dim: usize,
        context_len: usize,
        theta: f64,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
        let positions: Vec<f32> = (0..context_len).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (context_len, 1), dev)?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    /// Applies rotation to x of shape [batch, heads, seq, head_dim] for positions 0..seq.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let seq = x.dim(2)?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        let half = x.dim(D::Minus1)? / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let rot1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rot2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
        Tensor::cat(&[rot1, rot2], D::Minus1)
    }
}

/// SwiGLU MLP: down(silu(gate(x)) * up(x)). All linear layers are bias-free.
pub struct SwiGlu {
    gate: candle_nn::Linear,
    up: candle_nn::Linear,
    down: candle_nn::Linear,
}

impl SwiGlu {
    pub fn new(d_model: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: candle_nn::linear_no_bias(d_model, hidden, vb.pp("gate"))?,
            up: candle_nn::linear_no_bias(d_model, hidden, vb.pp("up"))?,
            down: candle_nn::linear_no_bias(hidden, d_model, vb.pp("down"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = candle_nn::ops::silu(&self.gate.forward(x)?)?;
        self.down.forward(&(g * self.up.forward(x)?)?)
    }
}

/// Grouped-query causal self-attention. Bias-free projections.
pub struct Attention {
    q: candle_nn::Linear,
    k: candle_nn::Linear,
    v: candle_nn::Linear,
    o: candle_nn::Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(d_model: usize, n_heads: usize, n_kv_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = d_model / n_heads;
        Ok(Self {
            q: candle_nn::linear_no_bias(d_model, n_heads * head_dim, vb.pp("q"))?,
            k: candle_nn::linear_no_bias(d_model, n_kv_heads * head_dim, vb.pp("k"))?,
            v: candle_nn::linear_no_bias(d_model, n_kv_heads * head_dim, vb.pp("v"))?,
            o: candle_nn::linear_no_bias(n_heads * head_dim, d_model, vb.pp("o"))?,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }

    /// x: [batch, seq, d_model]. Returns [batch, seq, d_model].
    pub fn forward(&self, x: &Tensor, rope: &Rope) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = self
            .q
            .forward(x)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k
            .forward(x)?
            .reshape((b, t, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v
            .forward(x)?
            .reshape((b, t, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = rope.apply(&q)?;
        let k = rope.apply(&k)?;
        let rep = self.n_heads / self.n_kv_heads;
        let k = repeat_kv(&k, rep)?;
        let v = repeat_kv(&v, rep)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let mask = causal_mask(t, x.dtype(), x.device())?;
        let att = att.broadcast_add(&mask)?;
        let att =
            candle_nn::ops::softmax(&att.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(v.dtype())?;
        let out = att
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t, self.n_heads * self.head_dim))?;
        self.o.forward(&out)
    }
}

/// Expands kv heads [b, kv, t, d] to [b, kv * rep, t, d].
fn repeat_kv(x: &Tensor, rep: usize) -> Result<Tensor> {
    if rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv, rep, t, d))?
        .reshape((b, kv * rep, t, d))
}

/// Additive causal mask [t, t]: 0 on and below the diagonal, -inf above it.
pub fn causal_mask(t: usize, dtype: DType, dev: &Device) -> Result<Tensor> {
    let allowed = Tensor::tril2(t, DType::U8, dev)?;
    let zeros = Tensor::zeros((t, t), dtype, dev)?;
    let neg = Tensor::full(f32::NEG_INFINITY, (t, t), dev)?.to_dtype(dtype)?;
    allowed.where_cond(&zeros, &neg)
}

/// Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x)).
pub struct Block {
    attn_norm: RmsNorm,
    attn: Attention,
    mlp_norm: RmsNorm,
    mlp: SwiGlu,
}

impl Block {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        ffn_hidden: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            attn_norm: rms_norm(d_model, eps, vb.pp("attn_norm"))?,
            attn: Attention::new(d_model, n_heads, n_kv_heads, vb.pp("attn"))?,
            mlp_norm: rms_norm(d_model, eps, vb.pp("mlp_norm"))?,
            mlp: SwiGlu::new(d_model, ffn_hidden, vb.pp("mlp"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, rope: &Rope) -> Result<Tensor> {
        let h = (x + self.attn.forward(&self.attn_norm.forward(x)?, rope)?)?;
        &h + self.mlp.forward(&self.mlp_norm.forward(&h)?)?
    }
}
