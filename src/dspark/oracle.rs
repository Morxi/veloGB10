//! The P8 reference oracle — a reference-exact CPU f32 implementation of one full DSpark
//! speculation round for the RadixArk/Qwen3.8-27B-DSpark anatomy.
//!
//! # Authority & port discipline
//!
//! The ONLY reference for this anatomy is `velogb10_tp4_dspark_addendum.md` §1 (the verified
//! pseudocode + Table A1) plus `PLAN/B8_S2_WORKDOC.md` §4.2. The vendor's `dflash.py`/`dspark.py`
//! are NOT on disk. This module ports that pseudocode line-by-line; each block below cites the
//! addendum line it implements. `src/dflash.rs` is a PORT-DISCIPLINE precedent only (per-head
//! q_norm/k_norm, ctx-concat KV, block q/k at `pos_start..pos_start+7`) — its norms/constants are
//! the Hy3 family and must NOT leak here (risk R9 / AGENTS §7).
//!
//! # Numerics
//!
//! Everything is f32, accumulated in fixed iteration order (no threading, no HashMap/map
//! iteration, no atomics). f64 is used ONLY where the addendum's YaRN reference computes in
//! python-f64 (`rope_table` correction-range math — cited below). The oracle is the DEFINITION for
//! S3–S5 kernel diffs; S7 re-validates every recorded interpretation against the real reference.
//!
//! # DECISIONS (mandatory — every place the pseudocode is ambiguous)
//!
//! Each entry: question / choice / reason / **REVALIDATE AT S7**.
//!
//! * **(A) Tap indexing.** "taps at trunk layers [4,16,28,40,52] reading post-layer outputs". The
//!   reference model reads `hidden_states[l+1]` for `l` in that list (addendum Table A1), i.e. the
//!   POST-layer output of trunk layer index `l` (0-based). Choice: the tap feature is the concat
//!   of `hidden_states[l+1]` for `l ∈ [4,16,28,40,52]`, treated here as an opaque `[L, 25600]`
//!   input (the real capture is S4). REVALIDATE AT S7 against the trunk's off-by-one convention.
//!
//! * **(B) Incremental vs batch tap materialization.** The SGLang implementation materializes the
//!   `th` projection "immediately" (addendum §1 interpretation / Table A3). Choice: the oracle
//!   computes `th` for ALL committed context positions in one batch (`tap_project`), and the
//!   incremental path (project only the newly committed row and append) is REQUIRED to equal the
//!   batch path bit-for-bit; the probe asserts this equality directly. REVALIDATE AT S7.
//!
//! * **(C) Block mask.** With the anchor at the block's first row and the whole context before it,
//!   "causal to context" is vacuous (every context row precedes the block). Choice: the mask
//!   reduces to "attend to ALL context rows + ALL block rows" (bidirectional within the 7-token
//!   block) — i.e. no row masking at all, matching dflash.rs's `is_causal=False` precedent.
//!   REVALIDATE AT S7 vs the vendor mask construction (which may still emit a causal block row).
//!
//! * **(D) RoPE positions for block rows.** Choice: dflash.rs convention — block q/k occupy
//!   `pos_start .. pos_start+7` with `pos_start = ctx_len` (the anchor sits at position `L`), and
//!   context k/v rotate at positions `0..L-1`. REVALIDATE AT S7 (the alternative — anchor at `L-1`
//!   — differs by one position across the whole block).
//!
//! * **(E) Greedy vs sampling in the Markov chain.** Choice: the oracle uses greedy argmax
//!   (deterministic, first-max tie-break); sampling is an engine/runtime concern and is out of
//!   scope for the correctness definition. REVALIDATE AT S7 (only for the sampling semantics, not
//!   the argmax math).
//!
//! * **(F) q_norm / k_norm form.** qwen3 RMSNorm is zero-centered `(1 + w)·x` (AGENTS §7 lesson;
//!   the Hy3/dflash port stores `(w−1)` because that family is plain `w·x` — NOT inherited here).
//!   Choice: every RMSNorm in this module (input_layernorm, post_attention_layernorm, final norm,
//!   hidden_norm, per-head q_norm/k_norm) computes `(x / rms(x)) * (1 + w)`. k_norm YES, v_norm NO
//!   (binding rule). REVALIDATE AT S7 (transformers' `Qwen3RMSNorm` is actually `w·x`, while
//!   `Qwen3_5RMSNorm` is `(1+w)·x`; the workdoc binding rule pins `(1+w)` — flag the naming trap).
//!
//! * **(G) k_verify convention.** `k_verify` is the verify WIDTH (`[1..8]`, 7 drafts + bonus), the
//!   same quantity the existing `agree_ext` folds into the hash word at bits `[27..31)` as a
//!   4-bit value (`(k_verify & 0xF) << 27`, see `src/net.rs::agree_ext` + `src/batch.rs::tp_agree_step`).
//!   Truncation is against the CUMULATIVE survival product (addendum §3.1 K-DSP4: "cumulative
//!   survival product and the truncation index"), NOT the per-position marginal (the addendum §1
//!   one-liner "first position below confidence threshold" is read as the survival curve). The
//!   anchor draft always survives (width ≥ 2 for τ ∈ (0,1)); width 1 (the "draft-gated" case) is
//!   representable in the field but not emitted here. REVALIDATE AT S7.
//!
//! * **(H) YaRN RoPE frequency mapping.** Two conventions exist in the wild: HF-canonical
//!   (low-frequency dims INTERPOLATED by `1/factor`, high-frequency EXTRAPOLATED) vs the opposite
//!   (e.g. the in-repo `src/dsv4_cpu.rs::rope_table`, a DIFFERENT family). Choice: HF-canonical
//!   (the drafter is `model_type: qwen3` served under transformers), correction-range math in
//!   python-f64 as the addendum cites. `mscale`/`mscale_all_dim` assumed 1.0 (net 1.0 — the
//!   addendum lists none). REVALIDATE AT S7.
//!
//! * **(I) Partial rotary.** The addendum/Table A1 give head_dim 128 and no
//!   `partial_rotary_factor`; dflash.rs rotates the full head_dim. Choice: full-dim rotary
//!   (rotary_dim = head_dim = 128). REVALIDATE AT S7 (the target itself uses partial rotary 0.25,
//!   but the DRAFTER config is what governs here).
//!
//! * **(J) f32 vs the reference's bf16 rounding.** The reference runs bf16 (cos/sin bf16-quantized,
//!   bf16 activations/weights). Choice: the oracle is pure f32 (the workdoc §4 mandate), so it does
//!   NOT replicate bf16 rounding. S3–S5 kernels diff against THIS f32 definition; S7 measures the
//!   oracle↔reference gap and records whether any piece needs an f64/bf16 refinement. REVALIDATE AT S7.
//!
//! * **(K) Tensor-name strings** (see `src/dspark/mod.rs::inventory`). No reference file is on
//!   disk; names follow the DFlash-backbone convention + Table A1 head names. REVALIDATE AT S7.
//!
//! * **(L) Markov chain `tok_0`.** The addendum §1 pseudocode's `for k in 1..7` implies `tok_0` is
//!   the COMMITTED anchor token; the workdoc §4.2 step 6 says "position 0 = anchor emits the first
//!   DRAFT token" and `tok_k = argmax(logits0_k + W1[tok_{k-1}] @ W2^T)`. Choice (workdoc wins):
//!   `d[0] = argmax(logits0_0)` is the first DRAFT token, and the Markov bias at position k uses
//!   the PREVIOUS DRAFT token `d[k-1]` (not the anchor). REVALIDATE AT S7.
//!
//! * **(M) Confidence latent.** `conf_k = w·[h_k ∥ latent_k] + b` with `latent_k = W1[tok_{k-1}]`
//!   = the previous draft token's 256-dim embedding. 6 confidences (positions 1..6; the anchor
//!   draft has none). REVALIDATE AT S7.
//!
//! * **(N) rms_norm_eps.** Unstated in the workdoc; qwen3 family default 1e-6 is used. REVALIDATE AT S7.
//!
//! * **(O) Embedding / LM head in the synthetic path.** The checkpoint has NEITHER (62 tensors);
//!   both are borrowed from the target. For S2 the oracle generates deterministic synthetic
//!   embed/head rows on the fly (`crate::dspark::synth::SyntheticTables`) with a fixed seed
//!   independent of the artifact. The real target binding is S4/S7. REVALIDATE AT S7.

use crate::dspark::synth::SyntheticTables;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The anatomy configuration (fully parameterized so the reduced-shape unit tests exercise the
/// exact same code paths as the full-size probe). Defaults = the P8 anatomy constants.
#[derive(Clone, Debug)]
pub struct DsparkConfig {
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub n_layers: usize,
    pub block: usize,
    pub mask_token_id: u32,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: u32,
    pub beta_slow: u32,
    pub orig_ctx: usize,
    pub max_positions: usize,
    pub rms_eps: f32,
    pub markov_rank: usize,
    /// Default confidence threshold (exposed as a parameter per workdoc §4.2 step 7).
    pub confidence_threshold: f32,
}

impl Default for DsparkConfig {
    fn default() -> Self {
        Self {
            hidden: crate::dspark::HIDDEN,
            num_heads: crate::dspark::NUM_HEADS,
            num_kv_heads: crate::dspark::NUM_KV_HEADS,
            head_dim: crate::dspark::HEAD_DIM,
            inter: crate::dspark::INTER,
            vocab: crate::dspark::VOCAB,
            n_layers: crate::dspark::N_LAYERS,
            block: crate::dspark::BLOCK,
            mask_token_id: crate::dspark::MASK_TOKEN_ID,
            rope_theta: crate::dspark::ROPE_THETA,
            rope_factor: crate::dspark::ROPE_FACTOR,
            beta_fast: crate::dspark::BETA_FAST,
            beta_slow: crate::dspark::BETA_SLOW,
            orig_ctx: crate::dspark::ORIG_CTX,
            max_positions: crate::dspark::MAX_POSITIONS,
            rms_eps: crate::dspark::RMS_EPS,
            markov_rank: crate::dspark::MARKOV_RANK,
            confidence_threshold: 0.5,
        }
    }
}

impl DsparkConfig {
    /// Verify the config's implied tensor inventory sums to the expected param count (a reduced
    /// config skips this — it only checks the full-shape reconciliation).
    pub fn n_params(&self) -> u64 {
        // 5 layers × 11 + 7 global, mirroring `inventory()`.
        let mut n = 0u64;
        n += (self.hidden * (5 * self.hidden)) as u64; // fc
        n += self.hidden as u64; // hidden_norm
        n += self.hidden as u64; // norm
        n += (self.vocab * self.markov_rank) as u64; // W1
        n += (self.vocab * self.markov_rank) as u64; // W2
        n += (self.hidden + self.markov_rank) as u64; // confidence.weight [1, h+rank]
        n += 1; // confidence.bias
        for _ in 0..self.n_layers {
            n += (self.hidden * self.hidden) as u64; // q_proj
            n += (self.num_kv_heads * self.head_dim * self.hidden) as u64; // k_proj
            n += (self.num_kv_heads * self.head_dim * self.hidden) as u64; // v_proj
            n += (self.hidden * self.hidden) as u64; // o_proj
            n += self.head_dim as u64; // q_norm
            n += self.head_dim as u64; // k_norm
            n += self.hidden as u64; // input_layernorm
            n += self.hidden as u64; // post_attention_layernorm
            n += (self.inter * self.hidden) as u64; // gate_proj
            n += (self.inter * self.hidden) as u64; // up_proj
            n += (self.hidden * self.inter) as u64; // down_proj
        }
        n
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// One draft-backbone layer's weights (all row-major `[out, in]` for linears, `[n]` for norms).
#[derive(Clone)]
pub struct LayerWeights {
    pub q_proj: Vec<f32>,   // [hidden, hidden]
    pub k_proj: Vec<f32>,   // [num_kv_heads*head_dim, hidden]
    pub v_proj: Vec<f32>,   // [num_kv_heads*head_dim, hidden]
    pub o_proj: Vec<f32>,   // [hidden, hidden]
    pub q_norm: Vec<f32>,   // [head_dim]
    pub k_norm: Vec<f32>,   // [head_dim]
    pub input_ln: Vec<f32>, // [hidden]
    pub post_ln: Vec<f32>,  // [hidden]
    pub gate_proj: Vec<f32>,// [inter, hidden]
    pub up_proj: Vec<f32>,  // [inter, hidden]
    pub down_proj: Vec<f32>,// [hidden, inter]
}

/// The full draft-model weight set (the 62 tensors, minus the target-borrowed embed/head).
#[derive(Clone)]
pub struct DsparkWeights {
    pub layers: Vec<LayerWeights>, // n_layers
    pub fc: Vec<f32>,        // [hidden, 5*hidden]
    pub hidden_norm: Vec<f32>, // [hidden]
    pub norm: Vec<f32>,      // [hidden]
    pub w1: Vec<f32>,        // [vocab, markov_rank]  (Embedding table)
    pub w2: Vec<f32>,        // [vocab, markov_rank]  (Linear [out, in])
    pub confidence_w: Vec<f32>, // [1, hidden + markov_rank]
    pub confidence_b: f32,   // [1]
}

// ---------------------------------------------------------------------------
// Round context / outputs
// ---------------------------------------------------------------------------

/// The per-round inputs. The tap hiddens and anchor come from the target path (S4); the synthetic
/// embed/head tables live in the oracle (fixed seed).
#[derive(Clone)]
pub struct RoundCtx {
    /// The target's tap hiddens for the L committed positions, `[L, 5*hidden]` (concat of the five
    /// tap-layer post-outputs per position), row-major.
    pub tap_hiddens: Vec<f32>,
    /// The anchor token id (last committed / bonus token).
    pub anchor: u32,
    /// The confidence threshold for `k_verify` truncation (workdoc §4.2 step 7).
    pub confidence_threshold: f32,
}

/// The context KV written by [`DsparkOracle::draft_kv_write`] and consumed by
/// [`DsparkOracle::block_forward`] (per layer).
#[derive(Clone)]
pub struct DraftKv {
    pub layers: Vec<CtxKv>,
}

/// One layer's injected context K/V (`[ctx_len, num_kv_heads, head_dim]`, row-major; k already
/// k_norm'd + RoPE'd, v raw).
#[derive(Clone)]
pub struct CtxKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Markov-chain result (K-DSP3).
#[derive(Clone)]
pub struct MarkovOut {
    /// The 7 draft tokens `d[0..6]`: `d[0]` from the anchor position (no bias), `d[k]` for k≥1 from
    /// position k corrected by `W1[d[k-1]] @ W2^T`.
    pub tokens: [u32; 7],
    /// The 6 latents `W1[d[k-1]]` for k=1..6, flattened `[6, markov_rank]` (for the confidence head).
    pub latents: Vec<f32>,
}

/// Confidence-head result (K-DSP4).
#[derive(Clone)]
pub struct ConfOut {
    /// Per-position acceptance probabilities `p_k = sigmoid(conf_k)` for k=1..6.
    pub p: [f32; 6],
    /// Cumulative survival `S_k = Π_{j≤k} p_j` for k=1..6 (monotone non-increasing).
    pub survival: [f32; 6],
    /// Verify width `k_verify ∈ [1..8]` at the configured threshold (DECISION G).
    pub k_verify: u8,
}

/// Full-round output (the convenience `run_round`).
#[derive(Clone)]
pub struct RoundOut {
    /// The tap projection `th` `[L, hidden]`.
    pub th: Vec<f32>,
    /// Final hidden `[block, hidden]` (post final-norm) — the piecewise surface for S3/S4 checks.
    pub h: Vec<f32>,
    /// The borrowed-head logits `[block, vocab]`.
    pub logits0: Vec<f32>,
    pub tokens: [u32; 7],
    pub latents: Vec<f32>,
    pub p: [f32; 6],
    pub survival: [f32; 6],
    pub k_verify: u8,
}

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

pub struct DsparkOracle {
    pub cfg: DsparkConfig,
    pub weights: DsparkWeights,
    /// YaRN frequency table (HF-canonical, DECISION H) — `[head_dim/2]` inverse freqs.
    freqs: Vec<f32>,
    /// Deterministic synthetic embed/head row generator (the target-borrowed surface, DECISION O).
    /// Rows are generated ON THE FLY (never materialized) so the oracle stays lean; the embed table
    /// needs only 7 rows/round and the head is streamed row-by-row in `lm_head`.
    synth: SyntheticTables,
}

impl DsparkOracle {
    pub fn from_weights(cfg: DsparkConfig, weights: DsparkWeights) -> Result<Self, anyhow::Error> {
        anyhow::ensure!(
            weights.layers.len() == cfg.n_layers,
            "weights layer count {} != config n_layers {}",
            weights.layers.len(),
            cfg.n_layers
        );
        anyhow::ensure!(cfg.block == 7, "the draft block is fixed at 7 positions (anchor + 6×MASK)");
        anyhow::ensure!(
            cfg.num_heads % cfg.num_kv_heads == 0,
            "GQA requires num_heads % num_kv_heads == 0"
        );
        anyhow::ensure!(cfg.head_dim % 2 == 0, "head_dim must be even (rotate_half)");
        anyhow::ensure!(
            cfg.hidden == cfg.num_heads * cfg.head_dim,
            "hidden ({}) must equal num_heads × head_dim ({}) — the o_proj input is the full head span",
            cfg.hidden,
            cfg.num_heads * cfg.head_dim
        );
        let freqs = yaarn_freqs(
            cfg.head_dim,
            cfg.rope_theta,
            cfg.rope_factor,
            cfg.orig_ctx,
            cfg.beta_fast,
            cfg.beta_slow,
        );
        Ok(Self {
            cfg,
            weights,
            freqs,
            synth: SyntheticTables::new(crate::dspark::SYNTH_EMBED_HEAD_SEED),
        })
    }

    /// The number of Q heads per KV head (GQA grouping).
    fn q_per_kv(&self) -> usize {
        self.cfg.num_heads / self.cfg.num_kv_heads
    }

    // -- primitive numerics (fixed order, deterministic) --------------------

    /// RMSNorm `(x / rms(x)) * w` over the LAST axis of each row — FULL GAINS (the artifact
    /// stores w≈1; SpecForge trains with HF `Qwen3RMSNorm` = plain `w·x`, and the checkpoint
    /// values confirm: every drafter norm weight has mean ≈ +1.0, while the TRUNK's are ≈ 0
    /// under its own `(1+w)` convention. The two checkpoints differ — §7 norm-convention trap).
    fn rms_norm_rows(&self, x: &[f32], w: &[f32], rows: usize, n: usize) -> Vec<f32> {
        let eps = self.cfg.rms_eps;
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let xr = &x[r * n..(r + 1) * n];
            let mut sum_sq = 0.0f32;
            for &v in xr {
                sum_sq += v * v;
            }
            let inv = 1.0f32 / (sum_sq / n as f32 + eps).sqrt();
            let or = &mut out[r * n..(r + 1) * n];
            for (i, &v) in xr.iter().enumerate() {
                or[i] = v * inv * w[i];
            }
        }
        out
    }

    /// Per-head RMSNorm over head_dim with a shared `[head_dim]` weight (q_norm / k_norm), in place.
    fn rms_norm_heads(&self, x: &mut [f32], rows: usize, heads: usize, w: &[f32]) {
        let hd = self.cfg.head_dim;
        let eps = self.cfg.rms_eps;
        debug_assert_eq!(x.len(), rows * heads * hd);
        for r in 0..rows {
            for h in 0..heads {
                let base = (r * heads + h) * hd;
                let mut sum_sq = 0.0f32;
                for d in 0..hd {
                    let v = x[base + d];
                    sum_sq += v * v;
                }
                let inv = 1.0f32 / (sum_sq / hd as f32 + eps).sqrt();
                for d in 0..hd {
                    x[base + d] *= inv * w[d];
                }
            }
        }
    }

    /// Row-batched linear: `out[r, o] = Σ_i W[o*inn + i] * x[r*inn + i]`. `W` is row-major `[outn, inn]`.
    fn linear(&self, w: &[f32], x: &[f32], outn: usize, inn: usize, rows: usize) -> Vec<f32> {
        debug_assert_eq!(w.len(), outn * inn);
        debug_assert_eq!(x.len(), rows * inn);
        let mut out = vec![0.0f32; rows * outn];
        for r in 0..rows {
            let xr = &x[r * inn..(r + 1) * inn];
            let or = &mut out[r * outn..(r + 1) * outn];
            for o in 0..outn {
                let wr = &w[o * inn..(o + 1) * inn];
                let mut acc = 0.0f32;
                for i in 0..inn {
                    acc += wr[i] * xr[i];
                }
                or[o] = acc;
            }
        }
        out
    }

    /// Apply rotary to the LAST `head_dim` dims of each `(row, head)` slice — NEOX/`rotate_half`
    /// pairing `(j, j+half)` with angle `pos * freq[j]` (SpecForge `dflash.py` imports
    /// `rotate_half` from HF qwen3; vLLM's `get_rope` default `is_neox_style=True` agrees —
    /// BOTH references are half-split, NOT interleaved).
    fn rope_apply(
        &self,
        x: &mut [f32],
        rows: usize,
        heads: usize,
        positions: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) {
        let hd = self.cfg.head_dim;
        let half = hd / 2;
        debug_assert_eq!(x.len(), rows * heads * hd);
        for r in 0..rows {
            let p = positions[r];
            for h in 0..heads {
                let base = (r * heads + h) * hd;
                for j in 0..half {
                    let c = cos[p * half + j];
                    let s = sin[p * half + j];
                    let x1 = x[base + j];
                    let x2 = x[base + half + j];
                    x[base + j] = x1 * c - x2 * s;
                    x[base + half + j] = x2 * c + x1 * s;
                }
            }
        }
    }

    /// Precompute (grow) the cos/sin tables up to `max_pos` positions (deterministic). The
    /// YaRN mscale (`0.1·ln(factor)+1` — vLLM `yarn_get_mscale`, folded into the cos/sin
    /// cache by `YaRNScalingRotaryEmbedding._compute_cos_sin_cache`) multiplies BOTH tables:
    /// q and k both scale by mscale, so attention scores scale by mscale². Without it the
    /// softmax runs too flat by ×1.813 at factor 32.
    fn rope_tables(&self, max_pos: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.cfg.head_dim / 2;
        let mscale = if self.cfg.rope_factor > 1.0 {
            0.1f32 * self.cfg.rope_factor.ln() + 1.0f32
        } else {
            1.0f32
        };
        let mut cos = vec![0.0f32; max_pos * half];
        let mut sin = vec![0.0f32; max_pos * half];
        for p in 0..max_pos {
            let pf = p as f32;
            for i in 0..half {
                let ang = pf * self.freqs[i];
                cos[p * half + i] = ang.cos() * mscale;
                sin[p * half + i] = ang.sin() * mscale;
            }
        }
        (cos, sin)
    }

    /// Full-visibility (maskless) GQA attention (DECISION C). `q` `[n_q, nh, hd]`, `k`/`v`
    /// `[n_kv_total, nkv, hd]`. Returns `[n_q, nh, hd]` (head-major), ready for o_proj.
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], n_q: usize, n_kv_total: usize) -> Vec<f32> {
        let nh = self.cfg.num_heads;
        let nkv = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let group = self.q_per_kv();
        let mut out = vec![0.0f32; n_q * nh * hd];
        let mut scores = vec![0.0f32; n_kv_total];
        for r in 0..n_q {
            for h in 0..nh {
                let kvh = h / group;
                // scores over the full visible set (ctx + block), fixed ascending order.
                let qrow = &q[(r * nh + h) * hd..(r * nh + h + 1) * hd];
                let mut m = f32::NEG_INFINITY;
                for j in 0..n_kv_total {
                    let krow = &k[(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd];
                    let mut s = 0.0f32;
                    for d in 0..hd {
                        s += qrow[d] * krow[d];
                    }
                    s *= scale;
                    scores[j] = s;
                    if s > m {
                        m = s;
                    }
                }
                // softmax (max-subtracted), fixed order.
                let mut sum = 0.0f32;
                for j in 0..n_kv_total {
                    let e = (scores[j] - m).exp();
                    scores[j] = e;
                    sum += e;
                }
                let o = &mut out[(r * nh + h) * hd..(r * nh + h + 1) * hd];
                for d in 0..hd {
                    o[d] = 0.0;
                }
                for j in 0..n_kv_total {
                    let w = scores[j] / sum;
                    let vrow = &v[(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd];
                    for d in 0..hd {
                        o[d] += w * vrow[d];
                    }
                }
            }
        }
        out
    }

    /// Borrowed-head logits: `logits0[r, o] = head[o] · h[r]`, `[rows, vocab]` row-major. The head
    /// row is generated on the fly (deterministic; DECISION O) and dotted against every row.
    pub fn lm_head(&self, h: &[f32], rows: usize) -> Vec<f32> {
        let hidden = self.cfg.hidden;
        let vocab = self.cfg.vocab;
        debug_assert_eq!(h.len(), rows * hidden);
        let scale = 1.0f32 / (hidden as f32).sqrt();
        let mut out = vec![0.0f32; rows * vocab];
        for o in 0..vocab {
            let hr = self.synth.row(SyntheticTables::TABLE_HEAD, o as u32, hidden, scale);
            for r in 0..rows {
                let xr = &h[r * hidden..(r + 1) * hidden];
                let mut acc = 0.0f32;
                for d in 0..hidden {
                    acc += hr[d] * xr[d];
                }
                out[r * vocab + o] = acc;
            }
        }
        out
    }

    // -- piecewise API (the S3–S5 kernel-diff contract) ---------------------

    /// K-DSP2 ref: `th = hidden_norm(fc(taps))` for `m` positions. `taps` is `[m, 5*hidden]`.
    /// (addendum §1: `th = hidden_norm(fc(ctx))`; DECISION B — batch path.)
    pub fn tap_project(&self, taps: &[f32], m: usize) -> Vec<f32> {
        let hidden = self.cfg.hidden;
        debug_assert_eq!(taps.len(), m * 5 * hidden);
        let fc = self.linear(&self.weights.fc, taps, hidden, 5 * hidden, m);
        self.rms_norm_rows(&fc, &self.weights.hidden_norm, m, hidden)
    }

    /// K-DSP2 ref: write the injected context KV for every layer. `th` is `[m, hidden]`. Returns
    /// per-layer `(k_ctx [m, nkv, hd] with k_norm + RoPE at 0..m-1, v_ctx [m, nkv, hd] raw)`.
    /// (addendum §1: `k_ctx = k_proj(th)` + k_norm + RoPE; `v_ctx = v_proj(th)`, no v_norm.)
    pub fn draft_kv_write(&self, th: &[f32], m: usize) -> DraftKv {
        let hidden = self.cfg.hidden;
        let nkv = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim;
        debug_assert_eq!(th.len(), m * hidden);
        let (cos, sin) = self.rope_tables(m.max(1));
        let positions: Vec<usize> = (0..m).collect();
        let mut layers = Vec::with_capacity(self.cfg.n_layers);
        for l in &self.weights.layers {
            let k = self.linear(&l.k_proj, th, nkv * hd, hidden, m);
            let mut k = k; // [m, nkv*hd]
            self.rms_norm_heads(&mut k, m, nkv, &l.k_norm);
            self.rope_apply(&mut k, m, nkv, &positions, &cos, &sin);
            let v = self.linear(&l.v_proj, th, nkv * hd, hidden, m);
            layers.push(CtxKv { k, v });
        }
        DraftKv { layers }
    }

    /// K-DSP1 ref: the 5-layer block forward. `emb` `[block, hidden]`; `ctx_kv` the injected
    /// context K/V; `pos_start` the block's first RoPE position (= ctx_len, DECISION D). Returns
    /// the FINAL hidden `[block, hidden]` (post final-norm). (addendum §1 steps 4 + final norm.)
    pub fn block_forward(&self, emb: &[f32], ctx_kv: &DraftKv, pos_start: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let nh = cfg.num_heads;
        let nkv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let block = cfg.block;
        let inter = cfg.inter;
        debug_assert_eq!(emb.len(), block * hidden);
        debug_assert_eq!(ctx_kv.layers.len(), cfg.n_layers);
        let ctx_len = ctx_kv.layers[0].k.len() / (nkv * hd);
        let block_pos: Vec<usize> = (pos_start..pos_start + block).collect();
        let (cos, sin) = self.rope_tables(pos_start + block);

        let mut h = emb.to_vec();
        for (li, l) in self.weights.layers.iter().enumerate() {
            let ln = self.rms_norm_rows(&h, &l.input_ln, block, hidden);
            // q
            let q = self.linear(&l.q_proj, &ln, nh * hd, hidden, block);
            let mut q = q;
            self.rms_norm_heads(&mut q, block, nh, &l.q_norm);
            self.rope_apply(&mut q, block, nh, &block_pos, &cos, &sin);
            // block k/v
            let kb = self.linear(&l.k_proj, &ln, nkv * hd, hidden, block);
            let mut kb = kb;
            self.rms_norm_heads(&mut kb, block, nkv, &l.k_norm);
            self.rope_apply(&mut kb, block, nkv, &block_pos, &cos, &sin);
            let vb = self.linear(&l.v_proj, &ln, nkv * hd, hidden, block);
            // concat [ctx; block]
            let ctx = &ctx_kv.layers[li];
            let ntot = ctx_len + block;
            let mut k = Vec::with_capacity(ntot * nkv * hd);
            let mut v = Vec::with_capacity(ntot * nkv * hd);
            k.extend_from_slice(&ctx.k);
            k.extend_from_slice(&kb);
            v.extend_from_slice(&ctx.v);
            v.extend_from_slice(&vb);
            // attention
            let attn = self.attention(&q, &k, &v, block, ntot);
            let attn = self.linear(&l.o_proj, &attn, hidden, nh * hd, block);
            // residual
            let mut h2 = Vec::with_capacity(block * hidden);
            for i in 0..block * hidden {
                h2.push(h[i] + attn[i]);
            }
            // post_attention_layernorm + SwiGLU
            let h2n = self.rms_norm_rows(&h2, &l.post_ln, block, hidden);
            let gate = self.linear(&l.gate_proj, &h2n, inter, hidden, block);
            let up = self.linear(&l.up_proj, &h2n, inter, hidden, block);
            let mut ffn = vec![0.0f32; block * inter];
            for i in 0..block * inter {
                ffn[i] = silu(gate[i]) * up[i];
            }
            let down = self.linear(&l.down_proj, &ffn, hidden, inter, block);
            let mut h3 = Vec::with_capacity(block * hidden);
            for i in 0..block * hidden {
                h3.push(h2[i] + down[i]);
            }
            h = h3;
        }
        // final norm
        self.rms_norm_rows(&h, &self.weights.norm, block, hidden)
    }

    /// The left→right Markov chain, REFERENCE semantics (vLLM `speculator.py::_sample_sequential`
    /// + `dspark.py::apply_block_logits`): the bias `W2 @ W1[prev]` is added at EVERY position
    /// 0..6, teacher-forced — `prev` is seeded with the ANCHOR token (the input id at query
    /// offset 0), then each sampled draft. `logits0` `[block, vocab]`.
    pub fn markov_chain(&self, logits0: &[f32], anchor: u32) -> MarkovOut {
        let vocab = self.cfg.vocab;
        let rank = self.cfg.markov_rank;
        let block = self.cfg.block;
        debug_assert_eq!(logits0.len(), block * vocab);
        let mut tokens = [0u32; 7];
        let mut latents = vec![0.0f32; (block - 1) * rank];
        let mut prev = anchor as usize;
        for k in 0..block {
            if k > 0 {
                // latents[k-1] = W1[d[k-1]] (the confidence head's rows — semantics unchanged).
                latents[(k - 1) * rank..k * rank]
                    .copy_from_slice(&self.weights.w1[prev * rank..(prev + 1) * rank]);
            }
            // logits_k = logits0_k + W2 @ W1[prev], argmax (greedy, DECISION E).
            let base = &logits0[k * vocab..(k + 1) * vocab];
            let w1row = &self.weights.w1[prev * rank..(prev + 1) * rank];
            let mut best = f32::NEG_INFINITY;
            let mut best_i = 0usize;
            for o in 0..vocab {
                let w2row = &self.weights.w2[o * rank..(o + 1) * rank];
                let mut s = base[o];
                for i in 0..rank {
                    s += w2row[i] * w1row[i];
                }
                if s > best {
                    best = s;
                    best_i = o;
                }
            }
            tokens[k] = best_i as u32;
            prev = best_i;
        }
        MarkovOut { tokens, latents }
    }

    /// K-DSP4 ref: per-position confidence + cumulative survival + `k_verify` at `threshold`.
    /// `h` `[block, hidden]`, `latents` `[block-1, markov_rank]` (from `markov_chain`).
    /// (addendum §1: `conf_k = Linear_5376→1([h_k ∥ W1[tok_{k-1}]])`; DECISION G/M.)
    pub fn confidence(&self, h: &[f32], latents: &[f32], threshold: f32) -> ConfOut {
        let hidden = self.cfg.hidden;
        let rank = self.cfg.markov_rank;
        let block = self.cfg.block;
        debug_assert_eq!(h.len(), block * hidden);
        debug_assert_eq!(latents.len(), (block - 1) * rank);
        let mut p = [0.0f32; 6];
        let mut survival = [0.0f32; 6];
        let mut prod = 1.0f32;
        for k in 1..block {
            let hk = &h[k * hidden..(k + 1) * hidden];
            let lk = &latents[(k - 1) * rank..k * rank];
            let mut c = self.weights.confidence_b;
            for i in 0..hidden {
                c += self.weights.confidence_w[i] * hk[i];
            }
            for i in 0..rank {
                c += self.weights.confidence_w[hidden + i] * lk[i];
            }
            let pk = sigmoid(c);
            p[k - 1] = pk;
            prod *= pk;
            survival[k - 1] = prod;
        }
        let k_verify = truncate(&survival, threshold);
        ConfOut { p, survival, k_verify }
    }

    /// Convenience: the whole §4.2 loop (tap_project → draft_kv_write → embed → block_forward →
    /// lm_head → markov_chain → confidence). (addendum §1 pseudocode, top to bottom.)
    pub fn run_round(&self, ctx: &RoundCtx) -> RoundOut {
        let hidden = self.cfg.hidden;
        let block = self.cfg.block;
        let l = ctx.tap_hiddens.len() / (5 * hidden);
        debug_assert_eq!(ctx.tap_hiddens.len(), l * 5 * hidden);
        let th = self.tap_project(&ctx.tap_hiddens, l);
        let kv = self.draft_kv_write(&th, l);
        // block input: [anchor, MASK×6], borrowed synthetic embeddings.
        let mut blk = Vec::with_capacity(block);
        blk.push(ctx.anchor);
        for _ in 1..block {
            blk.push(self.cfg.mask_token_id);
        }
        let scale = 1.0f32 / (hidden as f32).sqrt();
        let mut emb = Vec::with_capacity(block * hidden);
        for &t in &blk {
            emb.extend_from_slice(&self.synth.row(SyntheticTables::TABLE_EMBED, t, hidden, scale));
        }
        let h = self.block_forward(&emb, &kv, l);
        let logits0 = self.lm_head(&h, block);
        let mo = self.markov_chain(&logits0, ctx.anchor);
        let co = self.confidence(&h, &mo.latents, ctx.confidence_threshold);
        RoundOut {
            th,
            h,
            logits0,
            tokens: mo.tokens,
            latents: mo.latents,
            p: co.p,
            survival: co.survival,
            k_verify: co.k_verify,
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// First-index argmax over `[vocab]`, deterministic (ascending, strictly-greater tie-break).
pub fn argmax(x: &[f32]) -> usize {
    let mut best = f32::NEG_INFINITY;
    let mut bi = 0usize;
    for (i, &v) in x.iter().enumerate() {
        if v > best {
            best = v;
            bi = i;
        }
    }
    bi
}

/// Numerically stable SiLU: `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        x * e / (1.0 + e)
    }
}

/// Numerically stable logistic sigmoid.
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// k_verify (verify width) from the cumulative-survival curve (DECISION G).
pub fn truncate(survival: &[f32; 6], threshold: f32) -> u8 {
    for (k, &s) in survival.iter().enumerate() {
        if s < threshold {
            return (k + 2) as u8; // k+1 drafts survive (k is 0-based) → width = (k+1)+1
        }
    }
    8 // full: 7 drafts + bonus
}

/// HF-canonical YaRN inverse-frequency table (DECISION H). `rdim` = head_dim (full rotary). The
/// correction-range math is python-f64 (`find_correction_dim` / `find_correction_range`) exactly as
/// the transformers/vLLM reference computes it — `low = floor(find(beta_fast))`,
/// `high = ceil(find(beta_slow))` (vLLM `common.py::yarn_find_correction_range(low_rot=beta_fast,
/// high_rot=beta_slow)`); the earlier version had the two betas SWAPPED, which inverted the
/// ramp (interpolated dims extrapolated and vice versa). The ramp/mask are torch-f32 op order.
pub fn yaarn_freqs(
    rdim: usize,
    base: f32,
    factor: f32,
    orig_ctx: usize,
    beta_fast: u32,
    beta_slow: u32,
) -> Vec<f32> {
    let half = rdim / 2;
    let dim = rdim as f64;
    // inv_freq[i] = 1 / base^(2i/rdim)
    let mut freqs = vec![0.0f32; half];
    for i in 0..half {
        let e = (2 * i) as f32 / rdim as f32;
        freqs[i] = 1.0f32 / base.powf(e);
    }
    // find_correction_dim(num_rot) in python f64.
    let find = |num_rot: f64| -> f64 {
        dim * ((orig_ctx as f64) / (num_rot * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * (base as f64).ln())
    };
    // find_correction_range(beta_fast, beta_slow): low = floor(find(beta_fast)),
    // high = ceil(find(beta_slow)) — beta_fast(32) yields the SMALLER dim (it counts more
    // rotations), so low < high with this order (vLLM passes (low_rot=beta_fast, high_rot=beta_slow)).
    let low = find(beta_fast as f64).floor().max(0.0) as i32;
    let high = find(beta_slow as f64).ceil().min(dim - 1.0) as i32;
    let (minv, maxv) = if low == high {
        (low as f32, low as f32 + 0.001)
    } else {
        (low as f32, high as f32)
    };
    for (i, f) in freqs.iter_mut().enumerate() {
        let ramp = ((i as f32 - minv) / (maxv - minv)).clamp(0.0, 1.0);
        let mask = 1.0f32 - ramp;
        // inv_freq = interp*(1-mask) + extra*mask ; interp = inv_freq/factor, extra = inv_freq.
        *f = (*f / factor) * (1.0f32 - mask) + *f * mask;
    }
    freqs
}
