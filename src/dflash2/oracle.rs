//! The DFlash2 reference oracle — a reference-exact CPU f32 implementation of one full DFlash2
//! speculation round (incoai/Qwen3.8-27B-DFlash2), exposed PIECEWISE for the S3F–S5F kernel diffs.
//!
//! # Authority & port discipline
//!
//! THE binding reference is `ref/dflash/dflash/model.py` (z-lab, Apache-2.0), read
//! fully; every block below cites its line numbers. Cross-checks: SGLang #35371 (merged;
//! `_selector_walk_kernel`, `_grouped_conv`, `_radix_topk`), vLLM #52816 (`qwen3_dflash2.py` +
//! `test_dflash_causality.py`), llama.cpp #27342 (`dflash.cpp` graph ops + `conversion/qwen.py`
//! naming). Where `PLAN/B8_S2F_WORKDOC.md` and the code disagreed, THE CODE WON (entries F, K).
//! `src/dspark/` is the S2 template (piecewise API + ledger discipline) — the anatomy is NOT
//! inherited (risk R9); `src/dflash.rs` is a different family entirely (Hy3).
//!
//! # Numerics
//!
//! Everything is f32, accumulated in fixed iteration order (no threading, no map iteration, no
//! atomics). The oracle is the DEFINITION for S3F–S5F kernel diffs. The REAL checkpoint is BF16;
//! the loader upcasts to f32 exactly (bf16→f32 is lossless). The golden harness
//! (`tool_probe/dflash2_golden.py`) runs the vendor reference in BOTH f32 and bf16 on the same
//! deterministic inputs and retires the REVALIDATE markers below.
//!
//! # DECISIONS (mandatory — every place the reference left a choice)
//!
//! Each entry: question / choice / reason / revalidation status.
//!
//! * **(A) Tap indexing.** `extract_context_feature` reads `hidden_states[layer_id + 1]` for
//!   `layer_id ∈ [5,19,33,47,61]` (model.py:41-43, config `target_layer_ids`) and concats along
//!   the feature dim → `[C, 25600]`. Choice: the oracle takes the taps as an OPAQUE input (the
//!   trunk capture is S4F); `tap_project` implements `hidden_norm(fc(taps))` (model.py:584).
//!   RETIRED-BY-CONSTRUCTION (line-exact port).
//!
//! * **(B) Incremental vs batch materialization.** The reference accumulates the draft KV cache
//!   across rounds (`past_key_values.update` then `_crop_to(start)`, model.py:243-250): each round
//!   appends the newly committed ctx rows' K/V and drops the block's. Choice: the oracle computes
//!   ctx K/V per chunk with explicit positions (`draft_kv_write`) and REQUIRES chunked == batch
//!   bit-for-bit (all per-row ops with fixed reduction order → exact). The probe asserts it.
//!   RETIRED-BY-PROBE.
//!
//! * **(C) Block mask (the causality resolution).** Config has BOTH top-level `is_causal: false`
//!   AND all-`sliding_attention` layer_types. vLLM's `test_dflash_causality.py` documents the
//!   resolution: the explicit top-level flag WINS (all layers non-causal / block-mask governed);
//!   "only SWA layers causal" is the legacy fallback when the flag is ABSENT. model.py:363-367
//!   implements exactly that (`is_causal = bool(config.is_causal)` when present). The mask itself
//!   (model.py:157-171 `_attention_mask`): queries are the LAST `q_len` keys of the
//!   `[ctx; block]` key sequence; `visible = |q_pos − k_pos| < 2048` when non-causal (BOTH band
//!   sides, lines 167-170) — i.e. in-block fully bidirectional (8 ≪ 2048), ctx visible within a
//!   2048-row trailing band. Choice: port the predicate literally (`attn` below). The probe
//!   asserts the window boundary EXACTLY (perturbing ctx row 52 of C=2100 changes nothing, row 53
//!   changes the output — see `run_probe_dflash2`). GOLDEN-VALIDATED.
//!
//! * **(D) RoPE positions.** From `dflash_generate` (model.py:246): position_ids span
//!   `[start − ctx_len, start + 8)` — ctx rows at their true sequence positions, the block at
//!   `start..start+7` with the ANCHOR at `start` (block row 0). For one round with C ctx rows:
//!   ctx `0..C-1`, block `C..C+7`. `apply_rotary_pos_emb` (model.py:331-337) is ASYMMETRIC: q
//!   takes the LAST `q_len` cos/sin rows, k takes the full span; rotate_half is the GPT-NeoX
//!   half-split (`cat(−x₂, x₁)`, transformers `rotate_half`), NOT the interleaved convention
//!   `src/dspark/oracle.rs` used. RETIRED-BY-CONSTRUCTION + GOLDEN-VALIDATED.
//!
//! * **(E) Greedy vs sampling in the selector chain.** Choice: greedy argmax at temperature 0
//!   (model.py:540; deterministic first-index tie-break). The sampled path (`_sampling_probs`,
//!   model.py:535-538) is recorded, NOT implemented (runtime concern). RETIRED-BY-CONSTRUCTION.
//!
//! * **(F) Norm convention — PLAIN `w·x`, NOT `(1+w)·x`.** The workdoc §2 claimed "zero-centered
//!   (1+w) — qwen3 class". THE CODE + CHECKPOINT DISAGREE: transformers 5.8.1's `Qwen3RMSNorm`
//!   (imported by model.py:10-19) is T5-style `weight * x` (ones-init, "equivalent to
//!   T5LayerNorm"), and the real checkpoint's norm weights are NOT zero-centered (`norm.weight`
//!   mean 2.65, q_norm mean 1.57 — a (1+w) reading would imply ~250% average gain). Choice:
//!   EVERY RMSNorm here (input_layernorm, post_attention_layernorm, q_norm, k_norm, hidden_norm,
//!   final norm) computes `(x / rms(x)) * w` in f32 (variance in f32, `rsqrt(mean+eps)`) — the
//!   OPPOSITE of dspark's DECISION F (that family's workdoc pinned (1+w); do NOT cross-inherit,
//!   AGENTS §7). k_norm YES, v_norm NO (model.py:383-391). GOLDEN-VALIDATED.
//!
//! * **(G) Verify width is a CONSTANT 8.** DFlash2 has NO confidence head and NO k_verify
//!   truncation (the DSpark mechanism is gone): the block is always [anchor + 7 drafts], the
//!   verify width is always `block_size` (model.py:234 `verify_size = min(block_size, …)`).
//!   S5F note: `k_verify ≡ 8` folds into the `agree_ext` hash word at bits [27..31) as a CONSTANT
//!   — the wire format is unchanged. RETIRED-BY-CONSTRUCTION (the checkpoint inventory proves the
//!   head's absence: 81 tensors, no confidence).
//!
//! * **(H) RoPE frequencies.** `rope_type: "default"` → no interpolation/scaling
//!   (`attention_factor = 1.0`, transformers 5.8.1 `compute_default_rope_parameters`).
//!   `inv_freq[i] = 1 / θ^((2i)/128)` computed in f32 (torch: `base ** (arange(0,dim,2)/dim)` in
//!   f32), θ = 1e7 from `rope_parameters.rope_theta` (the 5.x config layout; there is NO
//!   top-level `rope_theta`). cos/sin = cos/sin(p·inv_freq) in f32, duplicated
//!   `cat(freqs, freqs)` along the head dim (transformers `Qwen3RotaryEmbedding.forward`).
//!   GOLDEN-VALIDATED.
//!
//! * **(I) Full rotary.** head_dim 128 rotated in full (no `partial_rotary_factor` in the
//!   config). RETIRED-BY-CONSTRUCTION.
//!
//! * **(J) f32 oracle vs the bf16 reference.** The checkpoint is BF16; the vendor reference
//!   typically runs bf16 (cos/sin cast to the activation dtype, transformers
//!   `Qwen3RotaryEmbedding.forward`). Choice: the oracle is pure f32 (the workdoc mandate) and
//!   does NOT replicate bf16 rounding; the golden harness runs the reference in f32 (tight gate)
//!   AND bf16 (the documented dtype gap, expected ~1e-2 class rel-L2 per the workdoc).
//!   GOLDEN-VALIDATED (numbers in `PLAN/B8_S2F_REPORT.md` R3).
//!
//! * **(K) Tensor names.** The REAL parsed header is authoritative (all 81 names in
//!   `src/dflash2/mod.rs::inventory`): `layers.{i}.…` (no `model.` prefix in the raw file;
//!   llama.cpp's converter shows the HF state-dict prefix), the selector codebooks stored WITHOUT
//!   `.weight` (model.py:633 `key_mapping` adds it for `nn.Embedding`), conv tensors
//!   `{attention,mlp}_conv.{base_kernel, kernel_projection.weight}`. The workdoc's "5×16=80+6=86"
//!   puzzle is RESOLVED: the workdoc said 16/layer but listed 15; the header has 15/layer × 5 + 6
//!   globals = 81 = 1,924,404,480 params exactly. RETIRED (parsed from the real header).
//!
//! * **(L) Selector candidate order + tie-breaks.** The reference's `torch.topk(logits, 16,
//!   sorted=False)` (model.py:525) leaves the candidate ORDER unspecified. SGLang #35371's
//!   `_radix_topk` (flashinfer `top_k(sorted=True, deterministic=True)`, falling back to
//!   `torch.topk` sorted=True) is the deterministic answer: candidates sorted by DESCENDING logit.
//!   Choice: the oracle's top-16 is a TOTAL order — (logit desc, token-id asc) — and the greedy
//!   chain argmax takes the FIRST maximal index (matching `_selector_walk_kernel`'s
//!   `min(where(scores == max))` tie-break). Scores: `unary[p][k] + Σ_r (pred_codebook[prev] ∘
//!   hidden_projection(h[p]))[r] · succ_codebook[cand][r]` (model.py:530-534, einsum br,bkr→bk),
//!   chained left→right from `predecessor = anchor` over the 7 MASK positions (model.py:528-542).
//!   The path positions are the model outputs at block rows 1..7 (the 7 MASK slots —
//!   `dflash_generate` slices `[:, 1-verify_size:, :]`, model.py:249). GOLDEN-VALIDATED
//!   (candidate-set exact + path exact; order convention recorded in R3).
//!
//! * **(M) Conv boundary.** `GroupedDynamicCausalConv` (model.py:492-512) is BLOCK-LOCAL with a
//!   causal zero-pad at block start: `_grouped_dynamic_convolve` (model.py:478-489) shifts right
//!   by `offset` with zero left-pad over the block rows only — there is NO cross-round conv state
//!   (SGLang's `_grouped_conv` confirms: it masks `position_in_block >= tap`). `prepare`
//!   convolves the sublayer INPUT with `base_kernel[0]` + `dynamic[…,0]`; `finish` convolves the
//!   sublayer OUTPUT with `base_kernel[1]` + the held `dynamic[…,1]` (model.py:449-473 call
//!   order: prepare BEFORE the sublayer on the normed hidden, finish AFTER). Per-group scalar
//!   dynamic taps: `out[r,c] = Σ_{o∈{0,1}} (base[o,c] + dyn[r,o,c/16]) · x[r−o, c]` (x[−1] ≡ 0).
//!   RETIRED-BY-CONSTRUCTION + GOLDEN-VALIDATED (via per-layer hiddens).
//!
//! * **(N) rms_norm_eps** = 1e-6 (config `rms_norm_eps`). RETIRED-BY-CONSTRUCTION.
//!
//! * **(O) Borrowed target tensors.** The checkpoint has NO embed table and NO lm_head
//!   (`tie_word_embeddings: false`; the reference borrows both from the target:
//!   `_raw_input_embeddings`/`_output_head`, model.py:135-143) and the trunk taps come from the
//!   target's hiddens. For S2F all three stand in as deterministic synthetic tables
//!   (`synth::SyntheticTables`, fixed seeds, ported EXACTLY by the golden harness so oracle and
//!   reference see bit-identical inputs). `input_embedding_scale` and `output_multiplier` are
//!   absent from `dflash_config` → default 1.0 (model.py:241, 601); `final_logit_softcapping`
//!   absent → none (model.py:602-604). The real bind is S4F. GOLDEN-VALIDATED for the stand-in.
//!
//! * **(P) Attention scaling** = `head_dim^-0.5` (model.py:347); GQA 32Q/8KV → 4 Q heads per KV
//!   head, kv_head = q_head / 4. RETIRED-BY-CONSTRUCTION.
//!
//! * **(Q) `max_window_layers: 5`** in the config is the HF sliding-window layer-count knob; with
//!   ALL 5 layers `sliding_attention` it changes nothing here. Recorded for completeness.

use crate::dflash2::synth::SyntheticTables;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The anatomy configuration (fully parameterized so the reduced-shape unit tests exercise the
/// exact same code paths as the full-size probe). Defaults = the DFlash2 anatomy constants.
#[derive(Clone, Debug)]
pub struct Dflash2Config {
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
    pub rms_eps: f32,
    /// Sliding window over the ctx KV band (all layers; DECISION C).
    pub sliding_window: usize,
    /// Top-level causality flag (DECISION C): false = block-mask governed (non-causal band).
    pub is_causal: bool,
    pub conv_kernel: usize,
    pub conv_group: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    /// Number of trunk taps concatenated per position (5 → 25600 in the real anatomy).
    pub n_taps: usize,
}

impl Default for Dflash2Config {
    fn default() -> Self {
        Self {
            hidden: crate::dflash2::HIDDEN,
            num_heads: crate::dflash2::NUM_HEADS,
            num_kv_heads: crate::dflash2::NUM_KV_HEADS,
            head_dim: crate::dflash2::HEAD_DIM,
            inter: crate::dflash2::INTER,
            vocab: crate::dflash2::VOCAB,
            n_layers: crate::dflash2::N_LAYERS,
            block: crate::dflash2::block(),
            mask_token_id: crate::dflash2::MASK_TOKEN_ID,
            rope_theta: crate::dflash2::ROPE_THETA,
            rms_eps: crate::dflash2::RMS_EPS,
            sliding_window: crate::dflash2::SLIDING_WINDOW,
            is_causal: crate::dflash2::IS_CAUSAL,
            conv_kernel: crate::dflash2::CONV_KERNEL,
            conv_group: crate::dflash2::CONV_GROUP,
            selector_rank: crate::dflash2::SELECTOR_RANK,
            selector_top_k: crate::dflash2::SELECTOR_TOP_K,
            n_taps: crate::dflash2::TAP_LAYERS.len(),
        }
    }
}

impl Dflash2Config {
    /// Implied tensor-inventory param count (a reduced config only checks structural consistency;
    /// the full-shape reconciliation lives in `crate::dflash2::reconcile_params`).
    pub fn n_params(&self) -> u64 {
        let groups = self.hidden / self.conv_group;
        let mut n = 0u64;
        n += (self.hidden * (self.n_taps * self.hidden)) as u64; // fc
        n += self.hidden as u64; // hidden_norm
        n += self.hidden as u64; // norm
        n += (self.selector_rank * self.hidden) as u64; // hidden_projection
        n += (self.vocab * self.selector_rank) as u64; // predecessor_codebook
        n += (self.vocab * self.selector_rank) as u64; // successor_codebook
        for _ in 0..self.n_layers {
            n += (self.num_heads * self.head_dim * self.hidden) as u64; // q_proj
            n += (self.num_kv_heads * self.head_dim * self.hidden) as u64; // k_proj
            n += (self.num_kv_heads * self.head_dim * self.hidden) as u64; // v_proj
            n += (self.hidden * self.num_heads * self.head_dim) as u64; // o_proj
            n += self.head_dim as u64; // q_norm
            n += self.head_dim as u64; // k_norm
            n += self.hidden as u64; // input_layernorm
            n += self.hidden as u64; // post_attention_layernorm
            n += (self.inter * self.hidden) as u64; // gate_proj
            n += (self.inter * self.hidden) as u64; // up_proj
            n += (self.hidden * self.inter) as u64; // down_proj
            n += (2 * self.conv_kernel * self.hidden) as u64; // attention_conv.base_kernel
            n += (2 * self.conv_kernel * groups * self.hidden) as u64; // attention_conv.kernel_projection
            n += (2 * self.conv_kernel * self.hidden) as u64; // mlp_conv.base_kernel
            n += (2 * self.conv_kernel * groups * self.hidden) as u64; // mlp_conv.kernel_projection
        }
        n
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// One dynamic-conv's weights: `base_kernel [2 sides, kernel, hidden]` (row-major
/// `[2*kernel, hidden]` — side-major) and `kernel_projection [2*kernel*groups, hidden]`.
#[derive(Clone)]
pub struct ConvWeights {
    /// `[2*conv_kernel, hidden]` row-major: rows `0..kernel` = side 0 (prepare), rows
    /// `kernel..2*kernel` = side 1 (finish) — the file's `[2, kernel, hidden]` layout flattened.
    pub base_kernel: Vec<f32>,
    /// `[2*conv_kernel*groups, hidden]` row-major: index `((side*kernel)+tap)*groups + g`.
    pub kernel_projection: Vec<f32>,
}

/// One draft-backbone layer's weights (all row-major `[out, in]` for linears, `[n]` for norms).
#[derive(Clone)]
pub struct LayerWeights {
    pub q_proj: Vec<f32>,   // [num_heads*head_dim, hidden]
    pub k_proj: Vec<f32>,   // [num_kv_heads*head_dim, hidden]
    pub v_proj: Vec<f32>,   // [num_kv_heads*head_dim, hidden]
    pub o_proj: Vec<f32>,   // [hidden, num_heads*head_dim]
    pub q_norm: Vec<f32>,   // [head_dim]
    pub k_norm: Vec<f32>,   // [head_dim]
    pub input_ln: Vec<f32>, // [hidden]
    pub post_ln: Vec<f32>,  // [hidden]
    pub gate_proj: Vec<f32>,// [inter, hidden]
    pub up_proj: Vec<f32>,  // [inter, hidden]
    pub down_proj: Vec<f32>,// [hidden, inter]
    pub attention_conv: ConvWeights,
    pub mlp_conv: ConvWeights,
}

/// The full draft-model weight set (the 81 tensors, minus the target-borrowed embed/head).
#[derive(Clone)]
pub struct Dflash2Weights {
    pub layers: Vec<LayerWeights>, // n_layers
    pub fc: Vec<f32>,              // [hidden, n_taps*hidden]
    pub hidden_norm: Vec<f32>,     // [hidden]
    pub norm: Vec<f32>,            // [hidden]
    pub hidden_projection: Vec<f32>,    // [selector_rank, hidden]
    pub predecessor_codebook: Vec<f32>, // [vocab, selector_rank]
    pub successor_codebook: Vec<f32>,   // [vocab, selector_rank]
}

// ---------------------------------------------------------------------------
// Round context / outputs
// ---------------------------------------------------------------------------

/// The per-round inputs. The tap hiddens and anchor come from the target path (S4F); the
/// synthetic embed/head tables live in the oracle (fixed seeds, DECISION O).
#[derive(Clone)]
pub struct RoundCtx {
    /// The target's tap hiddens for the C committed positions, `[C, n_taps*hidden]` (concat of
    /// the five tap-layer `hidden_states[l+1]` outputs per position), row-major. The committed
    /// positions are `0..C-1`; the block occupies `C..C+7` (DECISION D).
    pub tap_hiddens: Vec<f32>,
    /// The anchor token id (block row 0 = the last committed / bonus token, at position C).
    pub anchor: u32,
}

/// The context KV written by [`Dflash2Oracle::draft_kv_write`] and consumed by the attention
/// piece (per layer). Chunks concatenate (DECISION B).
#[derive(Clone)]
pub struct DraftKv {
    pub layers: Vec<CtxKv>,
}

/// One layer's injected context K/V (`[m, num_kv_heads, head_dim]`, row-major flattened to
/// `[m, nkv*hd]`; k already k_norm'd + RoPE'd at its true positions, v raw — model.py:384-396).
#[derive(Clone)]
pub struct CtxKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Number of ctx rows (m).
    pub m: usize,
}

impl DraftKv {
    /// Append another chunk (later positions) — DECISION B; bit-identical to a one-shot write.
    pub fn append(&mut self, other: DraftKv) {
        for (a, b) in self.layers.iter_mut().zip(other.layers.into_iter()) {
            a.k.extend_from_slice(&b.k);
            a.v.extend_from_slice(&b.v);
            a.m += b.m;
        }
    }
}

/// The conv `prepare` result (DECISION M): the convolved sublayer input + the held dynamic taps
/// for `finish` (`dynamic[…, 1]`, `[n, conv_kernel, groups]` flattened to `[n, kernel*groups]`).
#[derive(Clone)]
pub struct ConvPrepared {
    pub x_conv: Vec<f32>,
    pub dyn_hold: Vec<f32>,
}

/// Array capacity for the selector result: the largest supported block-1 (16-1 = 15).
pub const MAX_SELECT_LEVELS: usize = crate::dflash2::MAX_BLOCK - 1;

/// Selector result (DECISION L).
#[derive(Clone)]
pub struct SelectOut {
    /// DF2 block-16: the LIVE level count = `cfg.block - 1` (7 at block 8, 15 at block 16).
    /// The arrays below are sized to [`MAX_SELECT_LEVELS`] so every index stays in bounds at
    /// either block; **only the first `nlevels` rows are meaningful** — consumers must slice,
    /// never assume 7.
    pub nlevels: usize,
    /// The draft tokens (the greedy chain path over the MASK positions).
    pub tokens: [u32; MAX_SELECT_LEVELS],
    /// The top-16 candidate token ids per position (the deterministic order, DECISION L).
    pub candidates: [[u32; 16]; MAX_SELECT_LEVELS],
    /// The candidate unary logits per position (aligned with `candidates`).
    pub unary: [[f32; 16]; MAX_SELECT_LEVELS],
    /// The final chain scores per position (post codebook term; diagnostics for S4F).
    pub scores: [[f32; 16]; MAX_SELECT_LEVELS],
}

impl SelectOut {
    /// The meaningful prefix of each array (length `nlevels`).
    #[inline] pub fn tokens_v(&self) -> &[u32] { &self.tokens[..self.nlevels] }
    #[inline] pub fn candidates_v(&self) -> &[[u32; 16]] { &self.candidates[..self.nlevels] }
    #[inline] pub fn unary_v(&self) -> &[[f32; 16]] { &self.unary[..self.nlevels] }
    #[inline] pub fn scores_v(&self) -> &[[f32; 16]] { &self.scores[..self.nlevels] }
}

/// Full-round output (the convenience `run_round`).
#[derive(Clone)]
pub struct RoundOut {
    /// The tap projection `th` `[C, hidden]`.
    pub th: Vec<f32>,
    /// Per-layer POST-layer hiddens `[block, hidden]` (5 entries; pre-final-norm — the golden
    /// per-layer comparison surface).
    pub layer_hiddens: Vec<Vec<f32>>,
    /// Final hidden `[block, hidden]` (post final-norm).
    pub h: Vec<f32>,
    /// The borrowed-head logits over the 7 MASK positions `[7, vocab]`.
    pub logits: Vec<f32>,
    /// The selector result (7 draft tokens + candidates/unary/scores).
    pub select: SelectOut,
}

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

pub struct Dflash2Oracle {
    pub cfg: Dflash2Config,
    pub weights: Dflash2Weights,
    /// Default-rope inverse frequencies (DECISION H) — `[head_dim/2]`, f32.
    inv_freq: Vec<f32>,
    /// Deterministic synthetic embed/head row generator (the target-borrowed surface, DECISION O).
    synth: SyntheticTables,
}

impl Dflash2Oracle {
    pub fn from_weights(cfg: Dflash2Config, weights: Dflash2Weights) -> Result<Self, anyhow::Error> {
        anyhow::ensure!(
            weights.layers.len() == cfg.n_layers,
            "weights layer count {} != config n_layers {}",
            weights.layers.len(),
            cfg.n_layers
        );
        // DF2 block-16: the block is runtime now (8 default, 16 opt-in). The oracle is block-
        // generic (it iterates cfg.block / cfg.block - 1), so the gate only has to reject values
        // this build cannot verify, and it must agree with the live global or the oracle would
        // silently model a different block than the engine runs.
        anyhow::ensure!(
            cfg.block == crate::dflash2::block(),
            "oracle block {} != live block {} (--df2-block)",
            cfg.block, crate::dflash2::block()
        );
        anyhow::ensure!(
            cfg.block == crate::dflash2::BLOCK || cfg.block == crate::dflash2::MAX_BLOCK,
            "the draft block must be 8 or 16 (anchor + block-1 MASK); got {}",
            cfg.block
        );
        anyhow::ensure!(
            cfg.num_heads % cfg.num_kv_heads == 0,
            "GQA requires num_heads % num_kv_heads == 0"
        );
        anyhow::ensure!(cfg.head_dim % 2 == 0, "head_dim must be even (rotate_half)");
        // NOTE: unlike the DSpark anatomy, hidden (5120) != num_heads × head_dim (4096) here —
        // q_proj is [4096, 5120] and o_proj [5120, 4096] (the head span is narrower than the
        // residual stream). No hidden == heads×hd invariant exists; exact shapes are asserted by
        // the loader's inventory check instead.
        anyhow::ensure!(
            cfg.hidden % cfg.conv_group == 0,
            "conv groups require hidden % conv_group == 0"
        );
        anyhow::ensure!(cfg.conv_kernel == 2, "the anatomy pins conv_kernel_size = 2");
        let mut inv_freq = vec![0.0f32; cfg.head_dim / 2];
        for i in 0..cfg.head_dim / 2 {
            // transformers 5.x `compute_default_rope_parameters`: 1/(base ** (arange(0,dim,2)/dim)).
            // Computed in f64 and cast to f32 (correctly rounded; within 1 ulp of torch's f32 pow
            // either way — DECISION H) so the oracle↔reference RoPE gap is libm-ulp class only.
            let e = (2 * i) as f64 / cfg.head_dim as f64;
            inv_freq[i] = (1.0f64 / (cfg.rope_theta as f64).powf(e)) as f32;
        }
        Ok(Self {
            cfg,
            weights,
            inv_freq,
            synth: SyntheticTables::new(crate::dflash2::SYNTH_EMBED_HEAD_SEED),
        })
    }

    /// The number of Q heads per KV head (GQA grouping).
    fn q_per_kv(&self) -> usize {
        self.cfg.num_heads / self.cfg.num_kv_heads
    }

    /// The conv group count (hidden / conv_group).
    fn groups(&self) -> usize {
        self.cfg.hidden / self.cfg.conv_group
    }

    /// Expose the deterministic borrowed-table generator (the probe/golden share it, DECISION O).
    pub fn synth(&self) -> &SyntheticTables {
        &self.synth
    }

    // -- primitive numerics (fixed order, deterministic) --------------------

    /// PLAIN RMSNorm `(x / rms(x)) * w` over the LAST axis of each row (DECISION F — the qwen3
    /// `Qwen3RMSNorm` in transformers 5.x, NOT the (1+w) qwen3.5/dspark variant).
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

    /// Per-head RMSNorm over head_dim with a shared `[head_dim]` weight (q_norm / k_norm), in
    /// place (model.py:383/390 — applied AFTER the proj view, BEFORE RoPE).
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

    /// Row-batched linear: `out[r, o] = Σ_i W[o*inn + i] * x[r*inn + i]`. `W` is row-major
    /// `[outn, inn]`. Per-output the reduction is ascending-i (fixed order → bit-identical under
    /// row chunking, DECISION B); outputs are unrolled 8-wide purely for ILP (each output's
    /// order is unchanged). pub(crate): the S5F3 oracle replay applies the REAL trunk head
    /// (an external [vocab, hidden] table) with the same fixed-order reduction.
    pub fn linear(&self, w: &[f32], x: &[f32], outn: usize, inn: usize, rows: usize) -> Vec<f32> {
        debug_assert_eq!(w.len(), outn * inn);
        debug_assert_eq!(x.len(), rows * inn);
        let mut out = vec![0.0f32; rows * outn];
        for r in 0..rows {
            let xr = &x[r * inn..(r + 1) * inn];
            let or = &mut out[r * outn..(r + 1) * outn];
            let mut o = 0usize;
            while o + 8 <= outn {
                let mut acc = [0.0f32; 8];
                for i in 0..inn {
                    let xv = xr[i];
                    for u in 0..8 {
                        acc[u] += w[(o + u) * inn + i] * xv;
                    }
                }
                for u in 0..8 {
                    or[o + u] = acc[u];
                }
                o += 8;
            }
            while o < outn {
                let wr = &w[o * inn..(o + 1) * inn];
                let mut acc = 0.0f32;
                for i in 0..inn {
                    acc += wr[i] * xr[i];
                }
                or[o] = acc;
                o += 1;
            }
        }
        out
    }

    /// Apply rotary (GPT-NeoX half-split rotate_half, DECISION D) to each `(row, head)` slice.
    /// `positions` has one entry per row. cos/sin tables are `[max_pos, head_dim/2]`.
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
                    let re = x[base + j];
                    let im = x[base + half + j];
                    x[base + j] = re * c - im * s;
                    x[base + half + j] = im * c + re * s;
                }
            }
        }
    }

    /// Precompute the cos/sin tables up to `max_pos` positions (deterministic, f32 — DECISION H).
    fn rope_tables(&self, max_pos: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.cfg.head_dim / 2;
        let mut cos = vec![0.0f32; max_pos * half];
        let mut sin = vec![0.0f32; max_pos * half];
        for p in 0..max_pos {
            let pf = p as f32;
            for i in 0..half {
                let ang = pf * self.inv_freq[i];
                cos[p * half + i] = ang.cos();
                sin[p * half + i] = ang.sin();
            }
        }
        (cos, sin)
    }

    // -- the dynamic conv (model.py:478-512; DECISION M) ---------------------

    /// `_grouped_dynamic_convolve` (model.py:478): causal `conv_kernel`-tap grouped conv over the
    /// block rows, zero left-pad at row 0 (block-local, no cross-round state).
    /// `x` `[n, hidden]`; `dyn_taps` `[n, conv_kernel, groups]` flattened; `base`
    /// `[conv_kernel, hidden]` (ONE side of the base_kernel). Returns `[n, hidden]`.
    pub fn convolve(&self, x: &[f32], dyn_taps: &[f32], base: &[f32], n: usize) -> Vec<f32> {
        let hidden = self.cfg.hidden;
        let k = self.cfg.conv_kernel;
        let groups = self.groups();
        let gs = self.cfg.conv_group;
        debug_assert_eq!(x.len(), n * hidden);
        debug_assert_eq!(dyn_taps.len(), n * k * groups);
        debug_assert_eq!(base.len(), k * hidden);
        let mut out = vec![0.0f32; n * hidden];
        for r in 0..n {
            for o in 0..k {
                // values = x[r - o] if r >= o else 0 (causal zero-pad — model.py:485 F.pad).
                if r < o {
                    continue;
                }
                let src = &x[(r - o) * hidden..(r - o + 1) * hidden];
                let drow = &dyn_taps[(r * k + o) * groups..(r * k + o + 1) * groups];
                let brow = &base[o * hidden..(o + 1) * hidden];
                let orow = &mut out[r * hidden..(r + 1) * hidden];
                // out[c] += (base[o,c] + dyn[r,o,c/gs]) * x[r-o,c] — fixed ascending order.
                for g in 0..groups {
                    let d = drow[g];
                    for c in g * gs..(g + 1) * gs {
                        orow[c] += (brow[c] + d) * src[c];
                    }
                }
            }
        }
        out
    }

    /// `GroupedDynamicCausalConv.prepare` (model.py:501-509): project the (normed) sublayer input
    /// to the dynamic taps `[n, 2, kernel, groups]`, convolve the input with side 0
    /// (`base_kernel[0]` + `dynamic[…,0]`), hold `dynamic[…,1]` for `finish`.
    pub fn conv_prepare(&self, conv: &ConvWeights, x: &[f32], n: usize) -> ConvPrepared {
        let hidden = self.cfg.hidden;
        let k = self.cfg.conv_kernel;
        let groups = self.groups();
        let dyn_all = self.linear(&conv.kernel_projection, x, 2 * k * groups, hidden, n);
        // dyn_all[r, ((side*k)+o)*groups + g] (model.py:503-505 view [n, 2, k, groups]).
        let mut dyn0 = vec![0.0f32; n * k * groups];
        let mut dyn1 = vec![0.0f32; n * k * groups];
        for r in 0..n {
            for o in 0..k {
                for g in 0..groups {
                    dyn0[(r * k + o) * groups + g] = dyn_all[(r * 2 * k + o) * groups + g];
                    dyn1[(r * k + o) * groups + g] = dyn_all[(r * 2 * k + k + o) * groups + g];
                }
            }
        }
        let base0 = &conv.base_kernel[0..k * hidden];
        let x_conv = self.convolve(x, &dyn0, base0, n);
        ConvPrepared { x_conv, dyn_hold: dyn1 }
    }

    /// `GroupedDynamicCausalConv.finish` (model.py:511-512): convolve the sublayer OUTPUT with
    /// side 1 (`base_kernel[1]` + the held `dynamic[…,1]`).
    pub fn conv_finish(&self, conv: &ConvWeights, y: &[f32], dyn_hold: &[f32], n: usize) -> Vec<f32> {
        let hidden = self.cfg.hidden;
        let k = self.cfg.conv_kernel;
        let base1 = &conv.base_kernel[k * hidden..2 * k * hidden];
        self.convolve(y, dyn_hold, base1, n)
    }

    // -- the attention piece (model.py:340-420; DECISIONS C/D/P) -------------

    /// The `_attention_mask` predicate (model.py:157-171), ported literally. `q_pos`/`k_pos` are
    /// positions WITHIN the `[ctx; block]` key sequence (queries = the last `q_len` keys).
    fn visible(&self, q_pos: usize, k_pos: usize) -> bool {
        let mut v = true;
        if self.cfg.is_causal {
            v &= k_pos <= q_pos;
        }
        let w = self.cfg.sliding_window;
        // sliding_window is Some for every layer of this anatomy (all sliding_attention).
        v &= (q_pos as i64 - k_pos as i64) < w as i64;
        if !self.cfg.is_causal {
            v &= (k_pos as i64 - q_pos as i64) < w as i64;
        }
        v
    }

    /// One layer's attention over the dual-source KV: q from the (normed + conv'd) block rows;
    /// k/v = [ctx rows from `th` (cached in `ctx_kv`); block rows] (model.py:384-389). q_len =
    /// block; the queries sit at key positions `ctx_len..ctx_len+block` (DECISION C/D). Returns
    /// the o_proj output `[block, hidden]` (pre-`finish`-conv, pre-residual).
    pub fn attn(
        &self,
        l: &LayerWeights,
        x_conv: &[f32],
        ctx_kv: &CtxKv,
        block_pos: &[usize],
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let nh = cfg.num_heads;
        let nkv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let block = cfg.block;
        debug_assert_eq!(x_conv.len(), block * hidden);
        let ctx_len = ctx_kv.m;
        debug_assert_eq!(ctx_kv.k.len(), ctx_len * nkv * hd);
        let ntot = ctx_len + block;
        let max_pos = block_pos.iter().copied().max().unwrap_or(0) + 1;
        let (cos, sin) = self.rope_tables(max_pos.max(ntot));

        // q = q_proj(x_conv) → per-head q_norm → RoPE at the block's positions (model.py:381-383,
        // 392-393 — q takes the LAST q_len cos/sin rows, equivalent to roping at block_pos).
        let mut q = self.linear(&l.q_proj, x_conv, nh * hd, hidden, block);
        self.rms_norm_heads(&mut q, block, nh, &l.q_norm);
        self.rope_apply(&mut q, block, nh, block_pos, &cos, &sin);

        // block k/v from the same conv'd input (model.py:385/387); k_norm YES, v_norm NO.
        let mut kb = self.linear(&l.k_proj, x_conv, nkv * hd, hidden, block);
        self.rms_norm_heads(&mut kb, block, nkv, &l.k_norm);
        self.rope_apply(&mut kb, block, nkv, block_pos, &cos, &sin);
        let vb = self.linear(&l.v_proj, x_conv, nkv * hd, hidden, block);

        // concat [ctx; block] (model.py:388-389) — the ctx rows are already k_norm'd + RoPE'd.
        let mut k = Vec::with_capacity(ntot * nkv * hd);
        let mut v = Vec::with_capacity(ntot * nkv * hd);
        k.extend_from_slice(&ctx_kv.k);
        k.extend_from_slice(&kb);
        v.extend_from_slice(&ctx_kv.v);
        v.extend_from_slice(&vb);

        // Masked GQA attention (DECISION C): query row i sits at key position ctx_len + i.
        let scale = 1.0f32 / (hd as f32).sqrt();
        let group = self.q_per_kv();
        let mut attn = vec![0.0f32; block * nh * hd];
        let mut scores = vec![0.0f32; ntot];
        for r in 0..block {
            let qp = ctx_len + r;
            for h in 0..nh {
                let kvh = h / group;
                let qrow = &q[(r * nh + h) * hd..(r * nh + h + 1) * hd];
                let mut m = f32::NEG_INFINITY;
                for j in 0..ntot {
                    if !self.visible(qp, j) {
                        scores[j] = f32::NEG_INFINITY;
                        continue;
                    }
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
                // softmax over the visible set in fixed ascending order (masked stay -inf → e=0).
                let mut sum = 0.0f32;
                for j in 0..ntot {
                    let e = if scores[j] == f32::NEG_INFINITY { 0.0 } else { (scores[j] - m).exp() };
                    scores[j] = e;
                    sum += e;
                }
                let o = &mut attn[(r * nh + h) * hd..(r * nh + h + 1) * hd];
                for d in 0..hd {
                    o[d] = 0.0;
                }
                for j in 0..ntot {
                    if scores[j] == 0.0 {
                        continue;
                    }
                    let w = scores[j] / sum;
                    let vrow = &v[(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd];
                    for d in 0..hd {
                        o[d] += w * vrow[d];
                    }
                }
            }
        }
        self.linear(&l.o_proj, &attn, hidden, nh * hd, block)
    }

    /// One layer's SwiGLU MLP (transformers `Qwen3MLP`: down(silu(gate) * up); model.py:471).
    /// Input = the (normed + conv'd) hidden `[block, hidden]`; returns `[block, hidden]`
    /// (pre-`finish`-conv, pre-residual).
    pub fn mlp(&self, l: &LayerWeights, x_conv: &[f32]) -> Vec<f32> {
        let hidden = self.cfg.hidden;
        let inter = self.cfg.inter;
        let block = self.cfg.block;
        debug_assert_eq!(x_conv.len(), block * hidden);
        let gate = self.linear(&l.gate_proj, x_conv, inter, hidden, block);
        let up = self.linear(&l.up_proj, x_conv, inter, hidden, block);
        let mut ffn = vec![0.0f32; block * inter];
        for i in 0..block * inter {
            ffn[i] = silu(gate[i]) * up[i];
        }
        self.linear(&l.down_proj, &ffn, hidden, inter, block)
    }

    // -- piecewise API (the S3F–S5F kernel-diff contract) --------------------

    /// `th = hidden_norm(fc(taps))` for `m` positions (model.py:584; DECISION A/B).
    /// `taps` is `[m, n_taps*hidden]`. Row-chunkable (bit-identical — the probe asserts).
    pub fn tap_project(&self, taps: &[f32], m: usize) -> Vec<f32> {
        let hidden = self.cfg.hidden;
        let tap_dim = self.cfg.n_taps * hidden;
        debug_assert_eq!(taps.len(), m * tap_dim);
        let fc = self.linear(&self.weights.fc, taps, hidden, tap_dim, m);
        self.rms_norm_rows(&fc, &self.weights.hidden_norm, m, hidden)
    }

    /// Write the injected context KV for every layer for a CHUNK of `m` ctx rows whose first
    /// row's sequence position is `pos_start` (model.py:384-396 + RoPE at the ctx positions;
    /// DECISION B/D). k is k_norm'd + RoPE'd; v is raw. Chunks concatenate via `DraftKv::append`.
    pub fn draft_kv_write(&self, th: &[f32], m: usize, pos_start: usize) -> DraftKv {
        let hidden = self.cfg.hidden;
        let nkv = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim;
        debug_assert_eq!(th.len(), m * hidden);
        let (cos, sin) = self.rope_tables(pos_start + m.max(1));
        let positions: Vec<usize> = (pos_start..pos_start + m).collect();
        let mut layers = Vec::with_capacity(self.cfg.n_layers);
        for l in &self.weights.layers {
            let mut k = self.linear(&l.k_proj, th, nkv * hd, hidden, m);
            self.rms_norm_heads(&mut k, m, nkv, &l.k_norm);
            self.rope_apply(&mut k, m, nkv, &positions, &cos, &sin);
            let v = self.linear(&l.v_proj, th, nkv * hd, hidden, m);
            layers.push(CtxKv { k, v, m });
        }
        DraftKv { layers }
    }

    /// One full decoder layer (model.py:433-475 `Qwen3DFlashDecoderLayer.forward`):
    /// residual → input_layernorm → attention_conv.prepare → attn → attention_conv.finish →
    /// +residual → post_attention_layernorm → mlp_conv.prepare → mlp → mlp_conv.finish →
    /// +residual. `h` is `[block, hidden]`; returns `[block, hidden]` (post second residual).
    pub fn layer_forward(
        &self,
        li: usize,
        h: &[f32],
        ctx_kv: &CtxKv,
        block_pos: &[usize],
    ) -> Vec<f32> {
        let l = &self.weights.layers[li];
        let hidden = self.cfg.hidden;
        let block = self.cfg.block;
        debug_assert_eq!(h.len(), block * hidden);
        // attention sublayer
        let residual = h;
        let hn = self.rms_norm_rows(h, &l.input_ln, block, hidden);
        let prep = self.conv_prepare(&l.attention_conv, &hn, block);
        let attn_out = self.attn(l, &prep.x_conv, ctx_kv, block_pos);
        let fin = self.conv_finish(&l.attention_conv, &attn_out, &prep.dyn_hold, block);
        let mut h2 = vec![0.0f32; block * hidden];
        for i in 0..block * hidden {
            h2[i] = residual[i] + fin[i];
        }
        // mlp sublayer
        let residual2 = &h2;
        let hn2 = self.rms_norm_rows(&h2, &l.post_ln, block, hidden);
        let prep2 = self.conv_prepare(&l.mlp_conv, &hn2, block);
        let mlp_out = self.mlp(l, &prep2.x_conv);
        let fin2 = self.conv_finish(&l.mlp_conv, &mlp_out, &prep2.dyn_hold, block);
        let mut h3 = vec![0.0f32; block * hidden];
        for i in 0..block * hidden {
            h3[i] = residual2[i] + fin2[i];
        }
        h3
    }

    /// The 5-layer backbone (model.py:573-597 `DFlashDraftModel.forward`): noise embeddings in,
    /// per-layer forwards over the injected ctx KV, final `norm`. `emb` `[block, hidden]`;
    /// `pos_start` = the anchor's sequence position (= ctx_len, DECISION D). Returns the
    /// per-layer hiddens + the post-final-norm hidden (all `[block, hidden]`).
    pub fn backbone_forward(
        &self,
        emb: &[f32],
        ctx_kv: &DraftKv,
        pos_start: usize,
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let block = cfg.block;
        debug_assert_eq!(emb.len(), block * hidden);
        debug_assert_eq!(ctx_kv.layers.len(), cfg.n_layers);
        let block_pos: Vec<usize> = (pos_start..pos_start + block).collect();
        let mut h = emb.to_vec();
        let mut layer_hiddens = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            h = self.layer_forward(li, &h, &ctx_kv.layers[li], &block_pos);
            layer_hiddens.push(h.clone());
        }
        let h_final = self.rms_norm_rows(&h, &self.weights.norm, block, hidden);
        (layer_hiddens, h_final)
    }

    /// Borrowed-head logits: `logits[r, o] = head[o] · h[r]` over the requested rows
    /// (model.py:599-605 `compute_logits` with output_multiplier 1.0, no softcap — DECISION O).
    /// `h` is `[rows, hidden]`; returns `[rows, vocab]`. The head row is the deterministic
    /// synthetic stand-in (the golden harness generates the identical table).
    pub fn logits(&self, h: &[f32], rows: usize) -> Vec<f32> {
        let hidden = self.cfg.hidden;
        let vocab = self.cfg.vocab;
        debug_assert_eq!(h.len(), rows * hidden);
        let scale = 1.0f32 / (hidden as f32).sqrt();
        let mut out = vec![0.0f32; rows * vocab];
        for r in 0..rows {
            let xr = &h[r * hidden..(r + 1) * hidden];
            let orow = &mut out[r * vocab..(r + 1) * vocab];
            for o in 0..vocab {
                let hr = self.synth.row(SyntheticTables::TABLE_HEAD, o as u32, hidden, scale);
                let mut acc = 0.0f32;
                for d in 0..hidden {
                    acc += hr[d] * xr[d];
                }
                orow[o] = acc;
            }
        }
        out
    }

    /// The deterministic top-16 candidates for one logits row (DECISION L): the total order is
    /// (logit desc, token-id asc) — SGLang `_radix_topk(sorted=True, deterministic=True)` is the
    /// engine-side answer; ties break to the LOWER token id. Returns `(unary, candidates)`.
    pub fn top16(&self, logit_row: &[f32]) -> ([f32; 16], [u32; 16]) {
        let k = self.cfg.selector_top_k;
        debug_assert_eq!(k, 16);
        debug_assert_eq!(logit_row.len(), self.cfg.vocab);
        // Insertion-maintained sorted-desc list of (value, id); deterministic.
        let mut vals = [f32::NEG_INFINITY; 16];
        let mut ids = [u32::MAX; 16];
        for (i, &v) in logit_row.iter().enumerate() {
            // Skip if strictly worse than the current 16th (and not a tie-beating id).
            let worse = v < vals[k - 1] || (v == vals[k - 1] && (i as u32) >= ids[k - 1]);
            if worse {
                continue;
            }
            // find insertion point (stable: after equal values with smaller ids).
            let mut p = 0usize;
            while p < k && (vals[p] > v || (vals[p] == v && ids[p] < i as u32)) {
                p += 1;
            }
            if p >= k {
                continue;
            }
            let mut j = k - 1;
            while j > p {
                vals[j] = vals[j - 1];
                ids[j] = ids[j - 1];
                j -= 1;
            }
            vals[p] = v;
            ids[p] = i as u32;
        }
        (vals, ids)
    }

    /// `CandidateSelector.select` at temperature 0 (model.py:524-547; DECISION L). `h_sel` is the
    /// 7 MASK-position hiddens `[7, hidden]` (block rows 1..7 post final-norm — model.py:249);
    /// `logits` the matching `[7, vocab]`; `anchor` the predecessor token. Greedy chain:
    /// `scores[k] = unary[k] + Σ_r (pred_codebook[prev] ∘ hidden_proj(h[p]))[r] ·
    /// succ_codebook[cand[k]][r]`, first-index argmax, predecessor ← chosen candidate.
    pub fn select_path(&self, h_sel: &[f32], logits: &[f32], anchor: u32) -> SelectOut {
        let vocab = self.cfg.vocab;
        let n = self.cfg.block - 1;      // DF2 block-16: live levels, not a constant 7
        debug_assert_eq!(logits.len(), n * vocab);
        let mut candidates = [[0u32; 16]; MAX_SELECT_LEVELS];
        let mut unary = [[0.0f32; 16]; MAX_SELECT_LEVELS];
        for p in 0..n {
            let (vals, ids) = self.top16(&logits[p * vocab..(p + 1) * vocab]);
            unary[p] = vals;
            candidates[p] = ids;
        }
        self.select_chain(h_sel, &candidates, &unary, anchor)
    }

    /// The chain half of `select` (model.py:526-542), split out so the golden harness can feed
    /// the REFERENCE's own (candidates, unary) through our chain (isolating the chain mechanics
    /// from the topk order convention, DECISION L).
    pub fn select_chain(
        &self,
        h_sel: &[f32],
        candidates: &[[u32; 16]; MAX_SELECT_LEVELS],
        unary: &[[f32; 16]; MAX_SELECT_LEVELS],
        anchor: u32,
    ) -> SelectOut {
        let hidden = self.cfg.hidden;
        let rank = self.cfg.selector_rank;
        let k = self.cfg.selector_top_k;
        let n = self.cfg.block - 1;      // DF2 block-16: live levels
        debug_assert_eq!(h_sel.len(), n * hidden);
        // hidden_projection once for all n positions (model.py:526).
        let hp = self.linear(&self.weights.hidden_projection, h_sel, rank, hidden, n);
        let mut tokens = [0u32; MAX_SELECT_LEVELS];
        let mut scores_out = [[0.0f32; 16]; MAX_SELECT_LEVELS];
        let mut predecessor = anchor;
        for p in 0..n {
            let ids = &candidates[p];
            let vals = &unary[p];
            // a[r] = pred_codebook[prev][r] * hp[p][r] (model.py:532).
            let pred_row = &self.weights.predecessor_codebook
                [predecessor as usize * rank..(predecessor as usize + 1) * rank];
            let hp_row = &hp[p * rank..(p + 1) * rank];
            let mut a = vec![0.0f32; rank];
            for r in 0..rank {
                a[r] = pred_row[r] * hp_row[r];
            }
            // scores[k] = unary[k] + Σ_r a[r] * succ_codebook[cand[k]][r] (model.py:530-534).
            let mut best = f32::NEG_INFINITY;
            let mut best_i = 0usize;
            for kk in 0..k {
                let succ_row = &self.weights.successor_codebook
                    [ids[kk] as usize * rank..(ids[kk] as usize + 1) * rank];
                let mut s = vals[kk];
                for r in 0..rank {
                    s += a[r] * succ_row[r];
                }
                scores_out[p][kk] = s;
                if s > best {
                    best = s;
                    best_i = kk;
                }
            }
            let tok = ids[best_i];
            tokens[p] = tok;
            predecessor = tok;
        }
        SelectOut { nlevels: n, tokens, candidates: *candidates, unary: *unary, scores: scores_out }
    }

    /// Convenience: the whole round (model.py:573-605 + 243-258) — tap_project →
    /// draft_kv_write → embed [anchor, MASK×7] → backbone_forward → logits on the 7 MASK
    /// positions → select_path. DECISIONS A–Q apply.
    pub fn run_round(&self, ctx: &RoundCtx) -> RoundOut {
        let hidden = self.cfg.hidden;
        let block = self.cfg.block;
        let tap_dim = self.cfg.n_taps * hidden;
        let c = ctx.tap_hiddens.len() / tap_dim;
        debug_assert_eq!(ctx.tap_hiddens.len(), c * tap_dim);
        let th = self.tap_project(&ctx.tap_hiddens, c);
        let kv = self.draft_kv_write(&th, c, 0);
        // block input: [anchor, MASK×7], borrowed synthetic embeddings (DECISION O).
        let scale = 1.0f32 / (hidden as f32).sqrt();
        let mut emb = Vec::with_capacity(block * hidden);
        emb.extend_from_slice(&self.synth.row(SyntheticTables::TABLE_EMBED, ctx.anchor, hidden, scale));
        for _ in 1..block {
            emb.extend_from_slice(&self.synth.row(
                SyntheticTables::TABLE_EMBED,
                self.cfg.mask_token_id,
                hidden,
                scale,
            ));
        }
        let (layer_hiddens, h) = self.backbone_forward(&emb, &kv, c);
        // the MASK positions = block rows 1..block (model.py:249) — block-1 of them.
        let nl = block - 1;
        let h_sel = &h[hidden..block * hidden];
        let logits = self.logits(h_sel, nl);
        let select = self.select_path(h_sel, &logits, ctx.anchor);
        RoundOut { th, layer_hiddens, h, logits, select }
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// First-index argmax over a slice, deterministic (ascending, strictly-greater tie-break).
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
