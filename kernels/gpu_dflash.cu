// DFlash drafter kernels (E29-B1, src/dflash.rs). Three small kernels that reproduce the z-lab
// reference's BF16 rounding semantics EXACTLY — the serving kernels (gpu_batch.ptx) hard-code
// different conventions and cannot express these:
//
//   dflash_rmsnorm_b — transformers Qwen3RMSNorm: normalize in fp32, ROUND to bf16, then multiply
//                      by the bf16 weight (exact fp32 product) and round again. (The serving
//                      rmsnorm_b keeps fp32 through the weight multiply and hard-codes (1+w).)
//   dflash_rope_b    — apply_rotary_pos_emb with bf16 arithmetic: each (x * c) product and the
//                      final sum round to bf16 (the reference's `(q * cos) + (rotate_half(q)*sin)`
//                      on bf16 tensors). cos/sin arrive as f32-stored bf16 table values.
//   dflash_attn_b    — eager attention with the softmax WEIGHTS rounded to bf16 before the PV
//                      matmul (the reference's `softmax(..., dtype=fp32).to(query.dtype)`), fp32
//                      accumulation, bf16 output. Reads k/v from the rank-space cache rows 0..K-1
//                      (no indirection — the DFlash cache is rewritten fresh every forward).
//
// All three mirror the layout conventions of the kernels they replace (col-major [dim, B]
// activations, row-major weights). Built by build.rs into src/ptx/gpu_dflash.ptx and loaded only
// by the DFlash module; the serving kernels are untouched.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---- Build-ID stamp: makes a stale PTX impossible to run silently (see gpu_batch.cu) ----
#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

__device__ __forceinline__ float b2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 f2b(float x) { return __float2bfloat16(x); }

// ---- batched RMSNorm, transformers-Qwen3RMSNorm exact (one block per (b, head)) ----
// x: [nh*n, B] col-major; w: [n] RAW bf16 weights (f32-stored); out == x allowed (in place).
// nh=1 gives plain per-column normalization.
extern "C" __global__ void dflash_rmsnorm_b(__nv_bfloat16* out, const __nv_bfloat16* x, const float* w,
                                            int nh, int n, int B, float eps) {
    int blk = blockIdx.x;
    int b = blk / nh, head = blk % nh;
    if (b >= B) return;
    extern __shared__ float s[];
    int tid = threadIdx.x, bs = blockDim.x;
    long long base = (long long)b * (nh * n) + (long long)head * n;
    const __nv_bfloat16* xb = x + base;
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += bs) { float v = b2f(xb[i]); sum_sq += v * v; }
    s[tid] = sum_sq;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)n + eps);
    for (int i = tid; i < n; i += bs) {
        float v = b2f(xb[i]);
        __nv_bfloat16 vn = f2b(v * inv);            // hidden_states.to(input_dtype)
        out[base + i] = f2b(b2f(vn) * w[i]);        // weight(bf16) * vn(bf16), product exact in fp32
    }
}

// ---- batched rotate_half RoPE with bf16 rounding (reference apply_rotary_pos_emb on bf16) ----
// x: [nh*hd, B] col-major, in place; cos/sin: [B, rdim] f32 (bf16-quantized table values).
extern "C" __global__ void dflash_rope_b(__nv_bfloat16* x, const float* cos, const float* sin,
                                         int nh, int hd, int rdim, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = rdim / 2;
    int per_seq = nh * half;
    int total = B * per_seq;
    if (idx >= total) return;
    int b = idx / per_seq;
    int rem = idx % per_seq;
    int head = rem / half;
    int pair = rem % half;
    long long base = (long long)b * (nh * hd) + (long long)head * hd;
    long long cb = (long long)b * rdim + pair;
    float x1 = b2f(x[base + pair]);
    float x2 = b2f(x[base + pair + half]);
    float c = cos[cb], s = sin[cb];
    __nv_bfloat16 p1c = f2b(x1 * c);            // x1 * cos[pair]
    __nv_bfloat16 p2s = f2b(x2 * s);            // x2 * sin[pair]
    x[base + pair] = f2b(b2f(p1c) - b2f(p2s));  // [pair]:     x1*c - x2*s
    __nv_bfloat16 p2c = f2b(x2 * c);            // x2 * cos[pair+half] (table duplicated)
    __nv_bfloat16 p1s = f2b(x1 * s);            // x1 * sin[pair+half]
    x[base + pair + half] = f2b(b2f(p2c) + b2f(p1s)); // [pair+half]: x2*c + x1*s
}

// ---- eager attention, softmax weights rounded to bf16 (reference eager_attention_forward) ----
// q: [nh*hd, B] col-major; k/v caches: [nkv, stride, hd]; out: [nh*hd, B] bf16.
// Query (b, qh) attends to ALL keys at cache rows 0..K-1 (non-causal; the DFlash block attends to
// every context key and every block key). One block per (b, qh), blockDim = hd threads.
// Three passes over K: row max, normalization sum, weighted PV (each recomputes the dots).
// ---- eager attention, softmax WEIGHTS rounded to bf16, fp32 accumulation (reference
// `softmax(..., dtype=fp32).to(query.dtype)`). k/v are read from the rank-space cache rows
// [0, K) — ctx rows 0..ctx_len-1, block rows ctx_len..ctx_len+B-1.
//
// Per-query key RANGE (P14): the reference's `_attention_mask` is a per-layer mask derived from
// `config.layer_types` — `sliding_attention` gives is_causal=True + `sliding_window`, and
// `full_attention` gives no mask at all:
//   causal      -> visible r <= ctx_len + b            (query's own local position)
//   window > 0  -> visible r >= ctx_len + b - window + 1
//   window > 0 and NOT causal -> also r <= ctx_len + b + window - 1   (bidirectional window)
// causal == 0 && window == 0 reproduces the pre-P14 unrestricted block attention bit-for-bit.
// (ctx_len, causal, window) arrive PACKED in one u64: the drafter's launch helper is capped at
// 12 kernel arguments. bits[0:32) = ctx_len, bit 32 = causal, bits[33:46) = window.
extern "C" __global__ void dflash_attn_b(__nv_bfloat16* out, const __nv_bfloat16* q,
                                         const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                                         int stride, int nh, int nkv, int hd, int K, int B,
                                         unsigned long long mask_pack) {
    const int ctx_len = (int)(mask_pack & 0xFFFFFFFFull);
    const int causal  = (int)((mask_pack >> 32) & 1ull);
    const int window  = (int)((mask_pack >> 33) & 0x1FFFull);
    const int blk = blockIdx.x;
    const int b = blk / nh;
    const int qh = blk % nh;
    if (b >= B) return;
    const int kvh = qh / (nh / nkv);
    const int tid = threadIdx.x;
    const int NW = blockDim.x >> 5;
    const int warp = tid >> 5, lane = tid & 31;
    const float scale = 1.0f / sqrtf((float)hd);
    extern __shared__ float sm[];       // [NW] warp partials + [1] broadcast slot
    const __nv_bfloat16* qrow = q + (long long)b * (nh * hd) + (long long)qh * hd;
    const long long kvbase = (long long)kvh * stride;
    const __nv_bfloat16* kb = k_cache + kvbase * hd;
    const __nv_bfloat16* vb = v_cache + kvbase * hd;
    const float qv = b2f(qrow[tid]);

    // ---- visible key range for THIS query (the reference's per-layer mask, local indices) ----
    const int qpos = ctx_len + b;
    int lo = (window > 0) ? (qpos - window + 1) : 0;
    if (lo < 0) lo = 0;
    int hi = causal ? qpos : (K - 1);
    if (window > 0 && !causal) { int hw = qpos + window - 1; if (hw < hi) hi = hw; }
    if (hi > K - 1) hi = K - 1;
    if (hi < lo) hi = lo;   // cannot happen (r = qpos is always visible); defensive

    // dot(q, k_r) * scale, reduced across all threads, broadcast to every thread.
    #define DFLASH_DOT(scale_out) \
        do { \
            float part = qv * b2f(kb[(long long)r * hd + tid]); \
            for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off); \
            if (lane == 0) sm[warp] = part; \
            __syncthreads(); \
            if (warp == 0) { \
                float s0 = (tid < NW) ? sm[tid] : 0.0f; \
                for (int off = NW / 2; off > 0; off >>= 1) s0 += __shfl_xor_sync(0xffffffffu, s0, off); \
                if (tid == 0) sm[NW] = s0 * scale; \
            } \
            __syncthreads(); \
            scale_out = sm[NW]; \
        } while (0)

    // pass 1: row max
    float m = -1e30f;
    for (int r = lo; r <= hi; r++) {
        float s;
        DFLASH_DOT(s);
        m = fmaxf(m, s);
    }
    // pass 2: sum of exp (fp32)
    float l = 0.0f;
    for (int r = lo; r <= hi; r++) {
        float s;
        DFLASH_DOT(s);
        l += expf(s - m);
    }
    // pass 3: PV with the softmax weights ROUNDED TO BF16 (torch .to(query.dtype))
    float acc = 0.0f;
    for (int r = lo; r <= hi; r++) {
        float s;
        DFLASH_DOT(s);
        __nv_bfloat16 wb = f2b(expf(s - m) / l);
        acc += b2f(wb) * b2f(vb[(long long)r * hd + tid]);
    }
    out[(long long)b * (nh * hd) + (long long)qh * hd + tid] = f2b(acc);
}

// ===========================================================================
// P14 — TILED block attention: the draft round's cost fix.
//
// WHY. `dflash_attn_b` launches ONE BLOCK PER (row, q_head) and makes THREE passes over the key
// range (row max, sum of exp, PV), each pass re-reading K from global memory; the `nh/nkv` q-head
// blocks that share a KV head re-read the same rows again, and the B rows re-read them once more.
// On the 3.6-35B at a 7.2 K context that is
//     512 blocks x (3 K-passes + 1 V-pass) x ~4096 keys x 128 hd x 2 B ~= 14.4 GB per draft round
// -> ~57 ms at the 238 GB/s roofline, which the `[dflash-step]` split confirmed (round 57 ms +
// verify 62 ms). The kernel is bandwidth-saturated; only fewer BYTES can make it faster.
//
// WHAT. One block per (row-tile, kv_head): DFLASH_TQ rows x the G = nh/nkv q heads of that KV head.
// K/V are streamed through shared memory in tiles of TK keys, so each KV byte is read from DRAM
// about once per (row-tile, kv_head) instead of 3 x per (row, q_head):
//     14.4 GB -> ~0.46 GB per round at 7.2 K (5 sliding layers ~4096 keys, 1 full layer ~K keys).
// The key range is the UNION over the tile's rows (rows differ by 1 position), and each row masks
// its own window/causal bound inside the loop — byte-identical semantics to the eager kernel's
// per-row loop bounds, at a fraction of the traffic.
//
// NUMERICS. Online softmax: running max `m` with the accumulator rescaled when `m` grows, and the
// reference's bf16 weight rounding applied to the unnormalized `exp(s - m_running)`, normalized by
// the final `l`. bf16 rounding is scale-invariant (relative precision, 8-bit mantissa), so this
// reproduces the reference's `softmax(...).to(query.dtype)` contract up to a common factor while
// keeping fp32 accumulation. The dot-product reduction order also differs (butterfly over 32 lanes,
// 4 dims per lane, vs a 2-level shuffle+smem tree) — both are fp32 and the oracle tolerance is
// cos >= 0.9999. `GB10_DFLASH_ATTN_EAGER=1` restores the old kernel for the A/B and the first-line
// repro (the `GB10_NO_VERIFY_GRAPH` discipline).
//
// Layout: DPER = hd/32 dims per lane (contiguous), warp = row, block = DFLASH_TQ=4 warps = 128
// threads, dynamic smem = 2*TK*hd bf16 (K tile | V tile). Requires hd % 32 == 0.
// ===========================================================================
#define DFLASH_TQ 4
#define DFLASH_NT (DFLASH_TQ * 32)

template <int DPER> struct DfVec;
template <> struct DfVec<1> { using type = unsigned short; };   // 2 B
template <> struct DfVec<2> { using type = unsigned int; };     // 4 B
template <> struct DfVec<4> { using type = uint2; };            // 8 B
template <> struct DfVec<8> { using type = uint4; };            // 16 B

template <int DPER, int G>
__device__ __forceinline__ void dflash_attn_tiled_impl(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache,
    int stride, int nh, int nkv, int hd, int K, int B, int TK, unsigned long long mask_pack)
{
    const int ctx_len = (int)(mask_pack & 0xFFFFFFFFull);
    const int causal  = (int)((mask_pack >> 32) & 1ull);
    const int window  = (int)((mask_pack >> 33) & 0x1FFFull);
    extern __shared__ __nv_bfloat16 smem[];      // [TK*hd] K | [TK*hd] V
    __nv_bfloat16* Ksm = smem;
    __nv_bfloat16* Vsm = smem + TK * hd;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int RT = (B + DFLASH_TQ - 1) / DFLASH_TQ;
    const int rt = blockIdx.x % RT;
    const int kvh = blockIdx.x / RT;
    const int b = rt * DFLASH_TQ + warp;
    const bool active = (b < B);
    const int bq = active ? b : 0;               // clamped row for the (unused) register loads
    const float scale = rsqrtf((float)hd);

    // Per-warp state: all G q heads of this KV head, DPER dims per lane.
    __nv_bfloat16 qv[G][DPER];
    float acc[G][DPER];
    float m[G], l[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const __nv_bfloat16* qrow = q + (long long)bq * (nh * hd)
                                      + (long long)(kvh * G + g) * hd + lane * DPER;
        using V = typename DfVec<DPER>::type;
        V v = *reinterpret_cast<const V*>(qrow);
        const __nv_bfloat16* vv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int i = 0; i < DPER; ++i) { qv[g][i] = vv[i]; acc[g][i] = 0.0f; }
        m[g] = -INFINITY; l[g] = 0.0f;
    }

    // Key range: the UNION over this tile's rows (rows in a tile differ by 1 position, so the
    // union is the widest row's range); per-row masking happens inside the loop.
    const int qpos_min = ctx_len + rt * DFLASH_TQ;
    const int qpos_max = ctx_len + min(rt * DFLASH_TQ + DFLASH_TQ - 1, B - 1);
    int lo = (window > 0) ? (qpos_min - window + 1) : 0;
    if (lo < 0) lo = 0;
    int hi = causal ? min(qpos_max, K - 1) : (K - 1);
    if (hi < lo) hi = lo;

    const __nv_bfloat16* kb = k_cache + (long long)kvh * stride * hd;
    const __nv_bfloat16* vb = v_cache + (long long)kvh * stride * hd;

    const int qpos = ctx_len + bq;
    int lo_b = (window > 0) ? (qpos - window + 1) : 0;
    if (lo_b < 0) lo_b = 0;
    const int hi_b = causal ? min(qpos, K - 1) : (K - 1);

    for (int t0 = lo; t0 <= hi; t0 += TK) {
        const int nk = min(TK, hi - t0 + 1);
        __syncthreads();                       // every thread's reads of the previous tile are done
        {
            // 16-byte vectorized fill (nk*hd is a multiple of 8 for hd % 8 == 0)
            const int nvec = (nk * hd) >> 3;
            const uint4* ks = reinterpret_cast<const uint4*>(kb + (long long)t0 * hd);
            const uint4* vs = reinterpret_cast<const uint4*>(vb + (long long)t0 * hd);
            uint4* kd = reinterpret_cast<uint4*>(Ksm);
            uint4* vd = reinterpret_cast<uint4*>(Vsm);
            for (int i = tid; i < nvec; i += DFLASH_NT) { kd[i] = ks[i]; vd[i] = vs[i]; }
        }
        __syncthreads();
        if (!active) continue;

#pragma unroll
        for (int g = 0; g < G; ++g) {
            for (int kk = 0; kk < nk; ++kk) {
                const int r = t0 + kk;
                float part = 0.0f;
                using V = typename DfVec<DPER>::type;
                V kvec = *reinterpret_cast<const V*>(Ksm + (long long)kk * hd + lane * DPER);
                const __nv_bfloat16* kk8 = reinterpret_cast<const __nv_bfloat16*>(&kvec);
#pragma unroll
                for (int i = 0; i < DPER; ++i) part = fmaf(b2f(qv[g][i]), b2f(kk8[i]), part);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
                const float s = part * scale;
                // Masked keys contribute nothing. `s` is warp-uniform (it comes out of the shuffle
                // reduction), so these branches are uniform. The `continue` also keeps exp() away
                // from exp(-inf - -inf) = NaN when the leading keys of a row are all masked.
                if (r < lo_b || r > hi_b) continue;
                if (s > m[g]) {
                    const float corr = __expf(m[g] - s);
#pragma unroll
                    for (int i = 0; i < DPER; ++i) acc[g][i] *= corr;
                    l[g] *= corr;
                    m[g] = s;
                }
                const float w = __expf(s - m[g]);
                l[g] += w;
                const __nv_bfloat16 wb = f2b(w);            // reference `softmax(...).to(q.dtype)`
                const float wf = b2f(wb);
                V vvec = *reinterpret_cast<const V*>(Vsm + (long long)kk * hd + lane * DPER);
                const __nv_bfloat16* vv8 = reinterpret_cast<const __nv_bfloat16*>(&vvec);
#pragma unroll
                for (int i = 0; i < DPER; ++i) acc[g][i] = fmaf(wf, b2f(vv8[i]), acc[g][i]);
            }
        }
    }

    if (active) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float inv = (l[g] > 0.0f) ? (1.0f / l[g]) : 0.0f;
            __nv_bfloat16* orow = out + (long long)b * (nh * hd)
                                    + (long long)(kvh * G + g) * hd + lane * DPER;
#pragma unroll
            for (int i = 0; i < DPER; ++i) orow[i] = f2b(acc[g][i] * inv);
        }
    }
}

// Non-template entry points (a Rust launch site can only name a symbol, so each supported
// (DPER = hd/32, G = nh/nkv) combination gets a concrete kernel).
#define DFLASH_TILED_KERNEL(DPER, G)                                                      \
    extern "C" __global__ void dflash_attn_tiled_d##DPER##_g##G(                           \
        __nv_bfloat16* out, const __nv_bfloat16* q, const __nv_bfloat16* k_cache,          \
        const __nv_bfloat16* v_cache, int stride, int nh, int nkv, int hd, int K, int B,    \
        int TK, unsigned long long mask_pack) {                                            \
        dflash_attn_tiled_impl<DPER, G>(out, q, k_cache, v_cache, stride, nh, nkv, hd, K, B, \
                                        TK, mask_pack);                                    \
    }

DFLASH_TILED_KERNEL(1, 1) DFLASH_TILED_KERNEL(1, 2) DFLASH_TILED_KERNEL(1, 4)
DFLASH_TILED_KERNEL(2, 1) DFLASH_TILED_KERNEL(2, 2) DFLASH_TILED_KERNEL(2, 4)
DFLASH_TILED_KERNEL(2, 8) DFLASH_TILED_KERNEL(2, 16)
DFLASH_TILED_KERNEL(4, 1) DFLASH_TILED_KERNEL(4, 2) DFLASH_TILED_KERNEL(4, 4)
DFLASH_TILED_KERNEL(4, 8) DFLASH_TILED_KERNEL(4, 16)
DFLASH_TILED_KERNEL(8, 1) DFLASH_TILED_KERNEL(8, 2) DFLASH_TILED_KERNEL(8, 4)
DFLASH_TILED_KERNEL(8, 8)
