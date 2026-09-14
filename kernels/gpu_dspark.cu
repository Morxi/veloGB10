// gpu_dspark.cu — the Qwen3.8-27B-DSpark drafter's GPU-only kernels (WI1, 2026-09-06).
//
// DSpark = the DFlash backbone geometry at (block 7, 40q/8kv heads, MLP 10240, FULL attention
// — sliding_window: null) + a Markov bigram head. The backbone reuses the trunk kernels in
// gpu_batch.ptx / gpu_kernels.ptx (rmsnorm_b, rmsnorm_perhead_b, rope_b, gather_rope_b,
// embed_gather_b, silu_mul_b, add_residual_b, write_kv_b, gemm_tiled_b, gemm_dsp_b_m8_r4,
// gemm_binv_b, df2_top256_b, df2_head_rerank_b, top16_b) exactly the way dflash2's round
// does; only the two pieces DF2 cannot express live here:
//
//   1. dspark_attn_full_ring_b — the drafter's block attention over the FULL context ring.
//      gqa_attn_band_ring_b is window-bounded because its scores live in DYNAMIC SMEM sized
//      min(window+B, ntot); at window→ntot the smem would explode (the old ctx-12120 cap).
//      This variant keeps the scores in a GLOBAL per-(block-row, q-head) scratch row instead —
//      same three-pass structure, same ascending visit order, the same reduction TREES and the
//      same serial-ascending cross-warp combination as the band kernel, so the host mirror's
//      fixed order is preserved. smem is tiny (qs[hd] + red[32]) and ctx-independent.
//   2. dspark_chain_{a,b} — the left→right Markov chain (dspark/oracle.rs markov_chain,
//      DECISIONS L/E): d_0 = argmax(logits row 0); logits_k = logits0_k + W2 @ W1[d_{k-1}];
//      d_k = argmax. Step A fans the vocab scan over many blocks (each block owns a strided
//      vocab slice, emits its best (score, lowest-id) candidate); step B reduces the
//      candidates with the (score DESC, id ASC) total order and publishes the token. The
//      per-row dot is a single-thread ascending-i f32 accumulation — the oracle's exact
//      association — and the cross-thread reduction never mixes partial dots.

#include <cuda_bf16.h>

__device__ __forceinline__ float b2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f2b(float x) { return __float2bfloat16(x); }

// F8/B4 mma helpers (the gdn_chunk_tc idiom, verbatim semantics):
// bf16x2 pair as u32 register operand, and the m16n8k16 bf16 mma with f32 accumulate.
__device__ __forceinline__ unsigned __nv_bfloat162_b32(__nv_bfloat162 v) {
    return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ void mma_e(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- dspark_attn_full_ring_b ------------------------------------------------
// One block per (block-row b, q-head qh); hd = 128 threads. FULL visibility: every key
// j in [0, ntot) — ctx keys j<C read ring row (j % C_ring), block keys j>=C read ring row
// (C_ring + j - C) — the gqa_attn_band_ring_b mapping verbatim, with lo forced to 0.
//
// scores_g row (b*nh + qh) holds this block's score vector; the caller sizes it
// [B*nh, score_stride] f32. The three passes re-read/write the row exactly like the smem
// version (pass1 writes raw scores, pass2 rewrites exp(s-mx), pass3 reads them) — DRAM
// traffic instead of smem, same values, same order. smem = (hd + 32) floats.
// One-warp expander for the round's per-step scalars (F8/P3, 2026-09-07): the round used
// FOUR blocking htod_sync_copy_into per step (toks, pos, wrow, anchor) — each drains the
// round stream mid-flight (~20 ms/round of serialization at 17K ctx). Now the host makes
// ONE async pinned copy of {anchor, nprev, c_ring, shift} and this kernel expands all four
// buffers on device. Semantics identical to the old host paths (incl. the BLOCK_POS diag).
extern "C" __global__ void dspark_scalars_b(
    int* __restrict__ toks_blk, int* __restrict__ pos_blk, int* __restrict__ wrow_blk,
    unsigned* __restrict__ prev_dev, const int* __restrict__ consts,
    int mask_id, int block)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int anchor = consts[0], nprev = consts[1], c_ring = consts[2], shift = consts[3];
    for (int i = 0; i < 8; ++i) {
        toks_blk[i] = (i == 0) ? anchor : (i < block ? mask_id : 0);
        pos_blk[i]  = (i < block) ? (nprev + shift + i) : 0;
        wrow_blk[i] = (i < block) ? (c_ring + i) : 0;
    }
    prev_dev[0] = (unsigned)anchor;
}

extern "C" __global__ void __launch_bounds__(128) dspark_attn_full_ring_b(
    __nv_bfloat16* __restrict__ out,            // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ q,        // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ k_cache,  // [nkv, stride, hd]
    const __nv_bfloat16* __restrict__ v_cache,
    unsigned long long ntot_stride, const int* ntot_dev, int nh_packed,
    float* __restrict__ scores_g, int score_stride, float scale) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    // Packed geometry (F8 acceptance A2, 2026-09-06): ntot in bits [42..64) (22 bits),
    // C_ring in [21..42), stride in [0..21) — 21-bit fields carry the 512K context class
    // (524296 + slack << 2^21). The old 16-bit fields capped the ring at 65535 rows.
    const int ntot   = ntot_dev ? *ntot_dev : (int)(ntot_stride >> 42);
    const int C_ring = (int)((ntot_stride >> 21) & 0x1FFFFF);
    const int stride = (int)(ntot_stride & 0x1FFFFF);
    const int C = ntot - 7;                    // committed ctx rows (DSpark block = 7)
    // The band kernel's VOLATILE-smem note applies to its scores region; here scores live in
    // global memory and smem holds only qs + red, but we keep the volatile on red (the same
    // barrier-ordered cross-warp tree) — do not silently drop it either.
    extern __shared__ volatile float sm[];
    const int b  = blockIdx.x / nh;
    const int qh = blockIdx.x % nh;
    const int kvh = qh / (nh / nkv);
    const int d = threadIdx.x;

    volatile float* qs = sm;            // [hd]
    volatile float* red = sm + hd;      // [32]
    float* srow = scores_g + (long long)(b * nh + qh) * (long long)score_stride;

    qs[d] = b2f(q[(long long)b * (nh * hd) + (long long)qh * hd + d]);
    __syncthreads();

    const __nv_bfloat16* kbase = k_cache + (long long)kvh * (long long)stride * hd;
    const __nv_bfloat16* vbase = v_cache + (long long)kvh * (long long)stride * hd;

    // pass 1: scores over the FULL visit range, ascending key index (band-kernel schedule).
    float mx = -1e30f;
    for (int j = d; j < ntot; j += hd) {
        const int rj = (j < C) ? (j % C_ring) : (C_ring + (j - C));
        const __nv_bfloat16* kr = kbase + (long long)rj * hd;
        float s = 0.f;
        for (int dd = 0; dd < hd; ++dd) s = fmaf(qs[dd], b2f(kr[dd]), s);
        s *= scale;
        srow[j] = s;
        mx = fmaxf(mx, s);
    }
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if ((d & 31) == 0) red[d >> 5] = mx;
    __syncthreads();
    if (d == 0) {
        float v = red[0];
        for (int w = 1; w < hd / 32; ++w) v = fmaxf(v, red[w]);
        red[0] = v;
    }
    __syncthreads();
    mx = red[0];

    // pass 2: exp + sum (accurate expf, ~2 ulp — the S3F tail).
    float l = 0.f;
    for (int j = d; j < ntot; j += hd) { const float e = expf(srow[j] - mx); srow[j] = e; l += e; }
    for (int off = 16; off; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
    if ((d & 31) == 0) red[d >> 5] = l;
    __syncthreads();
    if (d == 0) {
        float v = red[0];
        for (int w = 1; w < hd / 32; ++w) v += red[w];
        red[0] = v;
    }
    __syncthreads();
    const float inv = 1.0f / red[0];

    // pass 3: PV, ascending over the full range.
    float acc = 0.f;
    for (int j = 0; j < ntot; ++j) {
        const int rj = (j < C) ? (j % C_ring) : (C_ring + (j - C));
        acc = fmaf(srow[j] * inv, b2f(vbase[(long long)rj * hd + d]), acc);
    }
    out[(long long)b * (nh * hd) + (long long)qh * hd + d] = __float2bfloat16(acc);
}

// ---- dspark_attn_full_ring_c ------------------------------------------------
// F8/B2 (2026-09-07): shared-memory-tiled rewrite of dspark_attn_full_ring_b.
//
// WHY: the _b kernel's pass 1 gave each thread a FULL key row (thread d owns keys
// j ≡ d mod 128) — at any instant 128 threads touched 128 DIFFERENT 256-B rows, i.e.
// every warp load fanned into ~32 cache lines (≈1/16 of peak bytes), and pass 3
// serialized ntot dependent global loads. Measured at ntot≈30K: 15.1 ms/call ×5
// layers ≈ 68 ms/step — the entire long-context step regression (122 ms @ short
// ctx → ~290 ms @ 30K). Byte model: per (b,qh) block the algorithm must read
// ntot·hd·2 B of K and of V (≈15.5 MB at 30K); _b demanded ~16× that in physical
// sectors. _c stages tiles of 128 rows through shared memory with fully coalesced
// 16-B vector loads, then runs the IDENTICAL arithmetic:
//   pass 1 — thread d still owns keys j ≡ d (mod 128), still accumulates the dot
//            ascending dd with fmaf on the same bf16 values;
//   pass 3 — still ascending j, fmaf(srow[j]·inv, V[rj][d]).
// Scores still round-trip through scores_g exactly as _b did. The output is
// BITWISE IDENTICAL to _b (same fmaf chains in the same order on the same
// values); only the DRAM/L2 path changed. smem = 32 KB tile + (hd+32) f32.
#define DSPARK_ATTN_TILE 128
extern "C" __global__ void __launch_bounds__(128) dspark_attn_full_ring_c(
    __nv_bfloat16* __restrict__ out,            // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ q,        // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ k_cache,  // [nkv, stride, hd]
    const __nv_bfloat16* __restrict__ v_cache,
    unsigned long long ntot_stride, const int* ntot_dev, int nh_packed,
    float* __restrict__ scores_g, int score_stride, float scale) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const int ntot   = ntot_dev ? *ntot_dev : (int)(ntot_stride >> 42);
    const int C_ring = (int)((ntot_stride >> 21) & 0x1FFFFF);
    const int stride = (int)(ntot_stride & 0x1FFFFF);
    const int C = ntot - 7;                    // committed ctx rows (DSpark block = 7)
    extern __shared__ volatile float smf[];
    // Layout: [hd] qs | [32] red | bf16 tile [TILE][hd+16]. Pitch 144 (hd=128): rows stay
    // 16-B aligned for the vectorized staging stores, and pitch mod 32 = 16 keeps the
    // pass-1 reads tile[d*144 + dd] to a 2-way bank conflict max ((16d+dd) mod 32).
    volatile float* qs = smf;                  // [hd]
    volatile float* red = smf + hd;            // [32]
    __nv_bfloat16* tile = (__nv_bfloat16*)(smf + hd + 32);   // [TILE][hd+16] bf16
    const int pitch = hd + 16;
    const int b  = blockIdx.x / nh;
    const int qh = blockIdx.x % nh;
    const int kvh = qh / (nh / nkv);
    const int d = threadIdx.x;

    qs[d] = b2f(q[(long long)b * (nh * hd) + (long long)qh * hd + d]);
    __syncthreads();

    const __nv_bfloat16* kbase = k_cache + (long long)kvh * (long long)stride * hd;
    const __nv_bfloat16* vbase = v_cache + (long long)kvh * (long long)stride * hd;
    float* srow = scores_g + (long long)(b * nh + qh) * (long long)score_stride;

    // ---- pass 1: tiled scores. Thread d owns key j = t0 + d of each tile — the same
    // (thread, key) incidence as _b, the same ascending-dd fmaf dot from tile smem.
    float mx = -1e30f;
    for (int t0 = 0; t0 < ntot; t0 += DSPARK_ATTN_TILE) {
        const int tvalid = min(DSPARK_ATTN_TILE, ntot - t0);
        // stage: 16-B vector loads, warp-contiguous. Vec v covers row v/16 (col (v%16)*8);
        // thread d loads vecs d, d+128, ... — consecutive threads → consecutive 16-B chunks.
        const int nvec_row = hd / 8;                      // 16 vecs per row
        const int nvec = tvalid * nvec_row;
        for (int v = d; v < nvec; v += hd) {
            const int row = v / nvec_row;
            const int col = (v % nvec_row) * 8;
            const int j = t0 + row;
            const int rj = (j < C) ? (j % C_ring) : (C_ring + (j - C));
            *(uint4*)&tile[row * pitch + col] =
                *(const uint4*)&kbase[(long long)rj * hd + col];
        }
        __syncthreads();
        const int j = t0 + d;
        if (j < ntot) {
            const __nv_bfloat16* kr = tile + d * pitch;      // thread d's OWN row
            float s = 0.f;
            for (int dd = 0; dd < hd; ++dd) s = fmaf(qs[dd], b2f(kr[dd]), s);
            s *= scale;
            srow[j] = s;
            mx = fmaxf(mx, s);
        }
        __syncthreads();
    }
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if ((d & 31) == 0) red[d >> 5] = mx;
    __syncthreads();
    if (d == 0) {
        float v = red[0];
        for (int w = 1; w < hd / 32; ++w) v = fmaxf(v, red[w]);
        red[0] = v;
    }
    __syncthreads();
    mx = red[0];

    // ---- pass 2: exp + sum (unchanged from _b — global scores, accurate expf).
    float l = 0.f;
    for (int j = d; j < ntot; j += hd) { const float e = expf(srow[j] - mx); srow[j] = e; l += e; }
    for (int off = 16; off; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
    if ((d & 31) == 0) red[d >> 5] = l;
    __syncthreads();
    if (d == 0) {
        float v = red[0];
        for (int w = 1; w < hd / 32; ++w) v += red[w];
        red[0] = v;
    }
    __syncthreads();
    const float inv = 1.0f / red[0];

    // ---- pass 3: tiled PV. Ascending j, thread d accumulates V column d — identical
    // fmaf chain to _b, values staged through smem (no dependent global loads).
    float acc = 0.f;
    for (int t0 = 0; t0 < ntot; t0 += DSPARK_ATTN_TILE) {
        const int tvalid = min(DSPARK_ATTN_TILE, ntot - t0);
        const int nvec_row = hd / 8;
        const int nvec = tvalid * nvec_row;
        for (int v = d; v < nvec; v += hd) {
            const int row = v / nvec_row;
            const int col = (v % nvec_row) * 8;
            const int j = t0 + row;
            const int rj = (j < C) ? (j % C_ring) : (C_ring + (j - C));
            *(uint4*)&tile[row * pitch + col] =
                *(const uint4*)&vbase[(long long)rj * hd + col];
        }
        __syncthreads();
        for (int jj = 0; jj < tvalid; ++jj) {
            const float p = srow[t0 + jj] * inv;
            acc = fmaf(p, b2f(tile[jj * pitch + d]), acc);
        }
        __syncthreads();
    }
    out[(long long)b * (nh * hd) + (long long)qh * hd + d] = __float2bfloat16(acc);
}

// ---- dspark_attn_full_ring_d (flash-decode, GQA-fused) -----------------------
// F8/B2 v2 (2026-09-07): the structural fix. _b/_c launch B*nh = 280 blocks, and every
// kv head's K/V is read by (nh/nkv)*B = 35 of them — at ntot≈30K that is ≥4 GB of
// demanded traffic per layer (15.1-18 ms measured) for 124 MB of unique bytes. _d
// reads each kv head's K/V tile ONCE and shares it across ALL (b, qh) queries of the
// group via shared memory, online-softmax (flash) style:
//
//   grid  = (nkv, SEG) — 8 kv heads × 8 key segments = 64 blocks, 256 threads.
//   block = 8 warps; warp w owns queries w, w+8, … of the (b, qg) list (≤5 each):
//           35 queries per kv head, each lane holding a 4-wide d-slice accumulator.
//   per K/V tile of 32 keys (both staged to smem coalesced): warp computes the dot for
//   each owned query (lane-partitioned, shfl tree — deterministic fixed order), folds
//   it into that query's running (mx, sum, acc) with flash rescaling.
//   Partials [nkv][SEG][35][2 + hd] are merged by dspark_attn_merge_d (ascending seg).
//
// NUMERICS: this is a reduction-order change vs _b (warp tree + online rescale instead
// of 3-pass global scores) — same math, ulp-level differences. The round probe gates it
// (chain tokens EXACT vs the offline reference, h/logits relL2 tolerances); it is NOT
// bitwise vs _b and does not claim to be. smem = qs[35][hd] f32 + K tile + V tile.
#define DSPARK_QSEG 8
extern "C" __global__ void __launch_bounds__(256) dspark_attn_full_ring_d(
    __nv_bfloat16* __restrict__ out,            // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ q,        // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ k_cache,  // [nkv, stride, hd]
    const __nv_bfloat16* __restrict__ v_cache,
    unsigned long long ntot_stride, const int* ntot_dev, int nh_packed,
    float* __restrict__ part,                   // [nkv][DSPARK_QSEG][B*QG][2+hd]
    int nq_packed, float scale) {               // NQ = B*QG in the low 16 bits
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const int ntot   = ntot_dev ? *ntot_dev : (int)(ntot_stride >> 42);
    const int C_ring = (int)((ntot_stride >> 21) & 0x1FFFFF);
    const int stride = (int)(ntot_stride & 0x1FFFFF);
    const int C = ntot - 7;
    const int kvh = blockIdx.x;
    const int seg = blockIdx.y;
    const int QG  = nh / nkv;                   // queries per kv head
    const int NQ  = nq_packed & 0xFFFF;         // B*QG queries this kv head serves
    // Layout: qs [NQ][hd] f32 | K tile [32][hd] | V tile [32][hd] (bf16)
    extern __shared__ float smd[];
    float* qs = smd;                                    // [NQ][hd]
    __nv_bfloat16* kt = (__nv_bfloat16*)(smd + NQ * hd);            // [32][hd]
    __nv_bfloat16* vt = kt + 32 * hd;                                // [32][hd]

    // Stage every query of this kv group: global query (b, qh = kvh*QG + qg).
    for (int i = threadIdx.x; i < NQ * hd; i += blockDim.x) {
        const int qi = i / hd, dd = i % hd;
        const int b = qi / QG, qg = qi % QG;
        qs[qi * hd + dd] = b2f(q[(long long)b * (nh * hd) + (long long)(kvh * QG + qg) * hd + dd]);
    }
    __syncthreads();

    const int jbegin = seg * ((ntot + DSPARK_QSEG - 1) / DSPARK_QSEG);
    const int jend   = min(ntot, jbegin + ((ntot + DSPARK_QSEG - 1) / DSPARK_QSEG));
    const __nv_bfloat16* kbase = k_cache + (long long)kvh * (long long)stride * hd;
    const __nv_bfloat16* vbase = v_cache + (long long)kvh * (long long)stride * hd;

    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;
    // This warp's queries: qi = w, w+8, ... < NQ (4-5 of them). Per query: mx, sum in
    // lane 0's values replicated across lanes for cheap shfl-free rescale; acc slice in
    // each lane (d = lane*4 .. lane*4+3, hd/32 = 4 wide).
    const int MYQ = (NQ - w + 7) / 8;           // queries owned by warp w (<= 5 at NQ=35)
    float mymx[5], mysum[5], myacc[5][4];       // fixed 5 — no runtime-sized locals (AGENTS §4)
    #pragma unroll
    for (int iq = 0; iq < 5; ++iq) { mymx[iq] = -1e30f; mysum[iq] = 0.f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) myacc[iq][k] = 0.f; }

    for (int t0 = jbegin; t0 < jend; t0 += 32) {
        const int tv = min(32, jend - t0);
        // Stage K and V tiles: coalesced 16-B vectors. 32 rows x hd x 2 B each.
        for (int i = threadIdx.x * 8; i < tv * hd; i += blockDim.x * 8) {
            const int row = i / hd, col = i % hd;
            const int j = t0 + row;
            const int rj = (j < C) ? (j % C_ring) : (C_ring + (j - C));
            *(uint4*)&kt[row * hd + col] = *(const uint4*)&kbase[(long long)rj * hd + col];
            *(uint4*)&vt[row * hd + col] = *(const uint4*)&vbase[(long long)rj * hd + col];
        }
        __syncthreads();
        for (int j = 0; j < tv; ++j) {
            // Per owned query: score = Σ_dd qs[dd]·K[j][dd], lane-partitioned + shfl tree.
            #pragma unroll
            for (int iq = 0; iq < 5; ++iq) {
                if (iq >= MYQ) continue;
                const int qi = w + iq * 8;
                float s = 0.f;
                #pragma unroll
                for (int k = 0; k < 4; ++k)
                    s = fmaf(qs[qi * hd + lane * 4 + k], b2f(kt[j * hd + lane * 4 + k]), s);
                #pragma unroll
                for (int off = 16; off; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
                s *= scale;
                // online softmax fold (all lanes hold s after the xor-butterfly).
                const float mn = fmaxf(mymx[iq], s);
                const float rs = expf(mymx[iq] - mn);   // accurate expf — the _b fidelity tail
                const float e  = expf(s - mn);
                mysum[iq] = mysum[iq] * rs + e;
                #pragma unroll
                for (int k = 0; k < 4; ++k)
                    myacc[iq][k] = fmaf(e, b2f(vt[j * hd + lane * 4 + k]), myacc[iq][k] * rs);
                mymx[iq] = mn;
            }
        }
        __syncthreads();
    }
    // Emit partials: part[((kvh*DSPARK_QSEG + seg)*NQ + qi) * (2+hd) + {0=mx,1=sum,2+d=acc}]
    const int ROW = 2 + hd;
    #pragma unroll
    for (int iq = 0; iq < 5; ++iq) {
        if (iq >= MYQ) continue;
        const int qi = w + iq * 8;
        float* prow = part + (((long long)kvh * DSPARK_QSEG + seg) * NQ + qi) * ROW;
        if (lane == 0) { prow[0] = mymx[iq]; prow[1] = mysum[iq]; }
        #pragma unroll
        for (int k = 0; k < 4; ++k) prow[2 + lane * 4 + k] = myacc[iq][k];
    }
}

// Merge the DSPARK_QSEG partials of every (kvh, query): ascending-segment flash merge,
// then out = acc/sum. Grid = nkv blocks; 256 threads cover (query, d) pairs.
// F8/B2b: SEG is now RUNTIME (nq_total bits 16..23) so _d (SEG=8) and _e (SEG=16) share
// this kernel; the ascending-segment accumulation order is unchanged for _d callers.
extern "C" __global__ void __launch_bounds__(256) dspark_attn_merge_d(
    __nv_bfloat16* __restrict__ out,            // [nh*hd, B] col-major
    const float* __restrict__ part, int nh_packed, int nq_total) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int SEG = (nq_total >> 16) & 0xFF;
    const int NQT = nq_total & 0xFFFF;
    const int kvh = blockIdx.x;
    const int QG  = nh / (nh_packed & 0x3FF);
    const int ROW = 2 + hd;
    for (int i = threadIdx.x; i < NQT * hd; i += blockDim.x) {
        const int qi = i / hd, dd = i % hd;
        const float* base = part + (long long)kvh * SEG * NQT * ROW;
        float m = -1e30f;
        for (int s = 0; s < SEG; ++s) m = fmaxf(m, base[((long long)s * NQT + qi) * ROW]);
        float l = 0.f, acc = 0.f;
        for (int s = 0; s < SEG; ++s) {
            const float* prow = base + ((long long)s * NQT + qi) * ROW;
            const float w = expf(prow[0] - m);
            l = fmaf(prow[1], w, l);
            acc = fmaf(prow[2 + dd], w, acc);
        }
        const int b = qi / QG, qg = qi % QG;
        out[(long long)b * (nh * hd) + (long long)(kvh * QG + qg) * hd + dd] = __float2bfloat16(acc / l);
    }
}

// ---- dspark_attn_full_ring_e (flash-decode on tensor cores) ---------------------
// F8/B4: the instruction-count collapse of _d. _d's per-key inner loop is scalar-issue
// bound (~4.8 ms/call at ntot=30K vs 0.52 ms KV roofline): per (key, query) it spends
// 4 FMA + a 5-level shfl butterfly + 2 expf + 4 PV FMA. _e moves BOTH matmuls onto
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (the gdn_chunk_tc idiom):
//
//   grid  = (nkv, SEG=16) — 8 kv heads × 16 key segments = 128 blocks, 96 threads (3 warps).
//   warp w owns a 16-row query tile of the 35 (b, qg) queries (rows padded to 48, zeros).
//   per 32-key tile: QK^T = 4 groups × 8 mma (Q row-fragments held permanently in
//   registers), online-softmax fold on the f32 score fragments (quad-tree max/sum via
//   2× shfl.xor — the whole 8-key group in 2 shuffles, vs _d's 5 shuffles per KEY),
//   probabilities parked as bf16 in a per-warp smem tile, then PV = 2×16 mma against a
//   transposed V tile. Partials identical to _d: [nkv][SEG][B*QG][2+hd] f32 via the same
//   merge kernel (SEG passed at launch).
//
// NUMERICS vs _d (both are reduction-order changes vs _b, gated by the round probe —
// chain tokens EXACT vs the offline reference + h/logits relL2):
//   - QK accumulates in f32 inside the mma (bf16 inputs, same input values as _d).
//   - PV multiplies bf16-rounded p (2^-9 relative per weight) — the ONE new rounding
//     class vs _d's f32 p·v; if the probe flips chain tokens this is the suspect.
//   - __expf (SFU) instead of expf: 2^-22 relative — below the bf16-P rounding above.
#define DSPARK_SEG_E 16
#define DSPARK_PT_PITCH 40            // 32 keys + 8 pad (bank-conflict free)
extern "C" __global__ void __launch_bounds__(96) dspark_attn_full_ring_e(
    __nv_bfloat16* __restrict__ out,            // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ q,        // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ k_cache,  // [nkv, stride, hd]
    const __nv_bfloat16* __restrict__ v_cache,
    unsigned long long ntot_stride, const int* ntot_dev, int nh_packed,
    float* __restrict__ part,                   // [nkv][gridDim.y][B*QG][2+hd]
    int nq_packed, float scale) {               // NQ = B*QG in the low 16 bits
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;  // host-guaranteed 128 (tile constants below)
    const int nkv = nh_packed & 0x3FF;
    const int ntot   = ntot_dev ? *ntot_dev : (int)(ntot_stride >> 42);
    const int C_ring = (int)((ntot_stride >> 21) & 0x1FFFFF);
    const int stride = (int)(ntot_stride & 0x1FFFFF);
    const int C = ntot - 7;
    const int kvh = blockIdx.x;
    const int seg = blockIdx.y;
    const int SEG = gridDim.y;
    const int QG  = nh / nkv;
    const int NQ  = nq_packed & 0xFFFF;
    const int QR  = 48;                          // padded query rows (3 tiles of 16)

    // smem: qb [48][hd] bf16 | kt [32][hd] bf16 | vt2 [hd][40] bf16 (transposed V)
    //       | pt [3 warps][16][40] bf16 (per-warp probability tile)
    __shared__ __nv_bfloat16 qb[48 * 128];
    __shared__ __nv_bfloat16 kt[32 * 128];
    __shared__ __nv_bfloat16 vt2[128 * DSPARK_PT_PITCH];
    __shared__ __nv_bfloat16 pt[3 * 16 * DSPARK_PT_PITCH];

    // Stage queries as bf16 (mma operand); pad rows NQ..47 zero (their scores are
    // garbage-but-finite and never emitted).
    for (int i = threadIdx.x * 2; i < QR * hd; i += blockDim.x * 2) {
        const int qi = i / hd, dd = i % hd;
        unsigned pair = 0u;
        if (qi < NQ) {
            const int b = qi / QG, qg = qi % QG;
            pair = *(const unsigned*)&q[(long long)b * (nh * hd) + (long long)(kvh * QG + qg) * hd + dd];
        }
        *(unsigned*)&qb[qi * hd + dd] = pair;
    }
    __syncthreads();

    const int jbegin = seg * ((ntot + SEG - 1) / SEG);
    const int jend   = min(ntot, jbegin + ((ntot + SEG - 1) / SEG));
    const __nv_bfloat16* kbase = k_cache + (long long)kvh * (long long)stride * hd;
    const __nv_bfloat16* vbase = v_cache + (long long)kvh * (long long)stride * hd;

    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;           // 0..2 — owns query rows w*16..w*16+15
    const int g    = lane >> 2, tq = lane & 3;   // mma fragment coords (gdn_chunk_tc layout)
    const int qbase = w * 16;

    // Persistent Q row-fragments: qa[ks8][4] — 8 k-steps of 16 over hd=128.
    unsigned qa[8][4];
    #pragma unroll
    for (int ks8 = 0; ks8 < 8; ++ks8) {
        const __nv_bfloat16* q0 = qb + (qbase + g) * hd + ks8 * 16 + 4 * tq;
        qa[ks8][0] = *(const unsigned*)q0;
        qa[ks8][1] = *(const unsigned*)(q0 + 8 * hd);
        qa[ks8][2] = *(const unsigned*)(q0 + 2);
        qa[ks8][3] = *(const unsigned*)(q0 + 8 * hd + 2);
    }
    float m[2] = { -1e30f, -1e30f }, l[2] = { 0.f, 0.f };
    float o[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) o[nt][e] = 0.f;

    for (int t0 = jbegin; t0 < jend; t0 += 32) {
        const int tv = min(32, jend - t0);
        // Stage K (row-major) and V (TRANSPOSED: vt2[d][key], pitch 40) — coalesced
        // 16-B reads; the transposed write is 8 strided u16 (bank-friendlier than a
        // post-stage transpose pass at this tile size).
        for (int i = threadIdx.x * 8; i < tv * hd; i += blockDim.x * 8) {
            const int row = i / hd, col = i % hd;
            const int j = t0 + row;
            const int rj = (j < C) ? (j % C_ring) : (C_ring + (j - C));
            *(uint4*)&kt[row * hd + col] = *(const uint4*)&kbase[(long long)rj * hd + col];
            const __nv_bfloat16* v8 = &vbase[(long long)rj * hd + col];
            #pragma unroll
            for (int k = 0; k < 8; ++k) vt2[(col + k) * DSPARK_PT_PITCH + row] = v8[k];
        }
        // Zero the V tile's INVALID key columns. The parked P is exactly 0 there, but
        // the PV mma still multiplies P·V: stale smem can hold NaN bit patterns, and
        // 0*NaN = NaN poisons the accumulator. (K needs no zeroing: masked scores are
        // REPLACED by -1e30, not multiplied away.) Engine-probe repro: seg-12 NaNs.
        if (tv < 32) {
            for (int i = threadIdx.x; i < (32 - tv) * hd; i += blockDim.x) {
                const int key = tv + i / hd, col = i % hd;
                vt2[col * DSPARK_PT_PITCH + key] = f2b(0.f);
            }
        }
        __syncthreads();
        __nv_bfloat16* mypt = pt + w * 16 * DSPARK_PT_PITCH;
        // ---- scores + online fold + P parking (4 key-groups of 8) ----
        // Single fold per 32-key tile (flash-2): park RAW mma scores in registers, track
        // the per-row tile max across groups, then ONE rescale + ONE P parking pass. (A
        // per-group fold would park P in stale softmax bases — the PV must see every
        // probability against the tile-final m.)
        float sraw[4][4];
        float rmx[2] = { m[0], m[1] };
        #pragma unroll
        for (int jg = 0; jg < 4; ++jg) {
            float s[4] = { 0.f, 0.f, 0.f, 0.f };
            #pragma unroll
            for (int ks8 = 0; ks8 < 8; ++ks8) {
                const int cc = ks8 * 16 + 4 * tq;
                const __nv_bfloat16* krow = kt + (jg * 8 + g) * hd + cc;
                unsigned bK[2] = {
                    __nv_bfloat162_b32(*(const __nv_bfloat162*)krow),
                    __nv_bfloat162_b32(*(const __nv_bfloat162*)(krow + 2)) };
                mma_e(s, qa[ks8], bK);
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = jg * 8 + 2 * tq + (e & 1);
                sraw[jg][e] = (key < tv) ? s[e] * scale : -1e30f;
            }
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                float rh = fmaxf(sraw[jg][2 * half], sraw[jg][2 * half + 1]);
                rh = fmaxf(rh, __shfl_xor_sync(0xffffffffu, rh, 1));
                rh = fmaxf(rh, __shfl_xor_sync(0xffffffffu, rh, 2));
                rmx[half] = fmaxf(rmx[half], rh);
            }
        }
        // One rescale to the tile-final basis (rmx seeded with the old m ⇒ m_new).
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const float rs = __expf(m[half] - rmx[half]);   // m=-1e30 (first) → 0, no NaN
            l[half] *= rs;
            m[half] = rmx[half];
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                o[nt][2 * half]     *= rs;
                o[nt][2 * half + 1] *= rs;
            }
        }
        float pacc[2] = { 0.f, 0.f };
        #pragma unroll
        for (int jg = 0; jg < 4; ++jg) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const float p0 = __expf(sraw[jg][2 * half] - rmx[half]);
                const float p1 = __expf(sraw[jg][2 * half + 1] - rmx[half]);
                pacc[half] += p0 + p1;
                mypt[(g + 8 * half) * DSPARK_PT_PITCH + jg * 8 + 2 * tq]     = __float2bfloat16(p0);
                mypt[(g + 8 * half) * DSPARK_PT_PITCH + jg * 8 + 2 * tq + 1] = __float2bfloat16(p1);
            }
        }
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            float ps = pacc[half];
            ps += __shfl_xor_sync(0xffffffffu, ps, 1);
            ps += __shfl_xor_sync(0xffffffffu, ps, 2);
            l[half] += ps;
        }
        __syncwarp();
        // ---- PV: 2 key-halves × 16 d-tiles of mma ----
        #pragma unroll
        for (int kp = 0; kp < 32; kp += 16) {
            const __nv_bfloat16* prow = mypt + g * DSPARK_PT_PITCH + kp + 4 * tq;
            unsigned aP[4] = {
                *(const unsigned*)prow,
                *(const unsigned*)(prow + 8 * DSPARK_PT_PITCH),
                *(const unsigned*)(prow + 2),
                *(const unsigned*)(prow + 8 * DSPARK_PT_PITCH + 2) };
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                const __nv_bfloat16* vrow = vt2 + (nt * 8 + g) * DSPARK_PT_PITCH + kp + 4 * tq;
                unsigned bV[2] = {
                    __nv_bfloat162_b32(*(const __nv_bfloat162*)vrow),
                    __nv_bfloat162_b32(*(const __nv_bfloat162*)(vrow + 2)) };
                mma_e(o[nt], aP, bV);
            }
        }
        __syncwarp();       // P tile reused next iteration (warp-local region)
        __syncthreads();    // kt/vt2 reused next tile
    }
    // Emit partials (identical layout to _d; SEG from gridDim.y):
    // part[((kvh*SEG + seg)*NQ + qi)*(2+hd) + {0=mx,1=sum,2+d=acc}]
    const int ROW = 2 + hd;
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int qi = qbase + g + 8 * half;
        if (qi < NQ) {
            float* prow = part + (((long long)kvh * SEG + seg) * NQ + qi) * ROW;
            if (tq == 0) { prow[0] = m[half]; prow[1] = l[half]; }
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                prow[2 + nt * 8 + 2 * tq]     = o[nt][2 * half];
                prow[2 + nt * 8 + 2 * tq + 1] = o[nt][2 * half + 1];
            }
        }
    }
}

// ---- dspark_markov_chain (steps A/B) ----------------------------------------
// Lexicographic draft order: (score DESC, id ASC) — the top16_b total order generalized to
// f32 scores. better() is a strict total order; serial cross-warp combination preserves the
// first-index-on-tie rule the oracle's ascending scan implements.
__device__ __forceinline__ void chain_better(float sa, int oa, float sb, int ob,
                                             float* s, int* o) {
    if (sa > sb || (sa == sb && oa < ob)) { *s = sa; *o = oa; }
    else { *s = sb; *o = ob; }
}

// Step A: grid = G blocks; block g owns vocab rows o ≡ g (mod G). Emits one candidate
// (score, token id) per block. prev is read from prev_dev (device u32) so the chain runs
// without host sync. latent[i] = f32(W1[prev*rank + i]), gathered per block (rank <= 256 =
// blockDim). smem = (rank + 64) floats.
extern "C" __global__ void __launch_bounds__(256) dspark_chain_a(
    float* __restrict__ cand_s, unsigned* __restrict__ cand_o,
    const __nv_bfloat16* __restrict__ logits,  // [vocab, B] col-major
    const __nv_bfloat16* __restrict__ w1,      // [vocab, rank] row-major
    const __nv_bfloat16* __restrict__ w2,      // [vocab, rank] row-major
    const unsigned* __restrict__ prev_dev, int vocab, int rank, int B, int k) {
    extern __shared__ volatile float sm[];     // [rank] latent + [32] red_s + [32] red_o
    volatile float* latent = sm;               // [rank]
    volatile float* red_s = sm + rank;         // [32]
    volatile unsigned* red_o = (volatile unsigned*)(sm + rank + 32);  // [32]
    const int tid = threadIdx.x;
    const int g = blockIdx.x;
    const int prev = (int)prev_dev[0];
    if (tid < rank) latent[tid] = b2f(w1[(long long)prev * rank + tid]);
    __syncthreads();

    float bs = -1e30f; int bo = -1;
    for (int o = g; o < vocab; o += gridDim.x) {
        // logits are token-major [B, vocab] (gemm_binv writes C[n*M + m], ld = vocab)
        float s = b2f(logits[(long long)k * vocab + o]);
        const __nv_bfloat16* w2r = w2 + (long long)o * rank;
        for (int i = 0; i < rank; ++i) s = fmaf(b2f(w2r[i]), latent[i], s);
        if (s > bs || bo < 0) { bs = s; bo = o; }
    }
    for (int off = 16; off; off >>= 1) {
        float s2 = __shfl_xor_sync(0xffffffffu, bs, off);
        int   o2 = __shfl_xor_sync(0xffffffffu, bo, off);
        float s; int o; chain_better(bs, bo, s2, o2, &s, &o); bs = s; bo = o;
    }
    if ((tid & 31) == 0) { red_s[tid >> 5] = bs; red_o[tid >> 5] = (unsigned)bo; }
    __syncthreads();
    if (tid == 0) {
        float s = red_s[0]; int o = (int)red_o[0];
        for (int w = 1; w < 256 / 32; ++w) {
            float s2; int o2; chain_better(s, o, red_s[w], (int)red_o[w], &s2, &o2); s = s2; o = o2;
        }
        cand_s[g] = s; cand_o[g] = (unsigned)o;
    }
}

// Step B: one block reduces the G candidates, publishes token k AND the device chain state,
// and writes the W1[token] latent row for the confidence head (out_latents[(k-1)*rank..]).
// smem = 64 floats.
extern "C" __global__ void __launch_bounds__(256) dspark_chain_b(
    unsigned* __restrict__ out_tokens,         // [B]
    unsigned* __restrict__ prev_dev,           // [1] device chain state
    float* __restrict__ out_latents,           // [(B-1)*rank] f32
    const float* __restrict__ cand_s, const unsigned* __restrict__ cand_o, int G,
    const __nv_bfloat16* __restrict__ w1, int rank, int B, int k) {
    extern __shared__ volatile float sm[];     // [32] red_s + [32] red_o
    volatile float* red_s = sm;
    volatile unsigned* red_o = (volatile unsigned*)(sm + 32);
    const int tid = threadIdx.x;
    float bs = -1e30f; int bo = -1;
    for (int g = tid; g < G; g += 256) {
        float s; int o; chain_better(bs, bo, cand_s[g], (int)cand_o[g], &s, &o); bs = s; bo = o;
    }
    for (int off = 16; off; off >>= 1) {
        float s2 = __shfl_xor_sync(0xffffffffu, bs, off);
        int   o2 = __shfl_xor_sync(0xffffffffu, bo, off);
        float s; int o; chain_better(bs, bo, s2, o2, &s, &o); bs = s; bo = o;
    }
    if ((tid & 31) == 0) { red_s[tid >> 5] = bs; red_o[tid >> 5] = (unsigned)bo; }
    __syncthreads();
    if (tid == 0) {
        float s = red_s[0]; int o = (int)red_o[0];
        for (int w = 1; w < 256 / 32; ++w) {
            float s2; int o2; chain_better(s, o, red_s[w], (int)red_o[w], &s2, &o2); s = s2; o = o2;
        }
        out_tokens[k] = (unsigned)o;
        prev_dev[0] = (unsigned)o;
    }
    __syncthreads();
    // out_latents row (k-1) = W1[token_{k-1}] — the latent that PRODUCED d_k (the W1[d_{k-1}]
    // bias input), which is what the confidence head consumes (rows W1[d_0..d_5]). k>=1 only:
    // the anchor-seeded chain now starts at k=0 (bias W2@W1[anchor] on logits row 0 — the
    // reference's teacher-forced prev), and the confidence input has no row for position 0.
    if (k > 0) {
        const int prev_tok = (int)out_tokens[k - 1];
        for (int i = tid; i < rank; i += 256)
            out_latents[(long long)(k - 1) * rank + i] = b2f(w1[(long long)prev_tok * rank + i]);
    }
}

// d0: the row-0 argmax (no bias) with the same total order — the chain's first step. The
// top16 candidate table also yields d0; this kernel is the standalone/probe path.
extern "C" __global__ void __launch_bounds__(256) dspark_row0_argmax_b(
    unsigned* __restrict__ out_tokens, unsigned* __restrict__ prev_dev,
    const __nv_bfloat16* __restrict__ logits, int vocab, int B) {
    extern __shared__ volatile float sm[];     // [32] red_s + [32] red_o
    volatile float* red_s = sm;
    volatile unsigned* red_o = (volatile unsigned*)(sm + 32);
    const int tid = threadIdx.x;
    float bs = -1e30f; int bo = -1;
    for (int o = tid; o < vocab; o += 256) {
        // token-major [B, vocab]; row 0 (the anchor position) starts at 0
        float s = b2f(logits[o]);
        if (s > bs) { bs = s; bo = o; }
    }
    for (int off = 16; off; off >>= 1) {
        float s2 = __shfl_xor_sync(0xffffffffu, bs, off);
        int   o2 = __shfl_xor_sync(0xffffffffu, bo, off);
        if (s2 > bs || (s2 == bs && o2 < bo)) { bs = s2; bo = o2; }
    }
    if ((tid & 31) == 0) { red_s[tid >> 5] = bs; red_o[tid >> 5] = (unsigned)bo; }
    __syncthreads();
    if (tid == 0) {
        float s = red_s[0]; int o = (int)red_o[0];
        for (int w = 1; w < 256 / 32; ++w)
            if (red_s[w] > s || (red_s[w] == s && (int)red_o[w] < o)) { s = red_s[w]; o = (int)red_o[w]; }
        out_tokens[0] = (unsigned)o;
        prev_dev[0] = (unsigned)o;
    }
}

// F8 (2026-09-06): the coalesced chain. dspark_chain_a broadcasts 2-byte w2 loads across
// 256 redundant threads (measured 19 ms/call — 2.8% of roofline — 58% of total GPU time,
// 7 calls/step = the dspark round's 147 ms). chain_c keeps the EXACT ascending-i fmaf
// order (bitwise-faithful to the oracle) but assigns ONE vocab row per thread with w2
// pre-transposed to [rank, vocab]: consecutive threads read consecutive addresses
// (64 B/warp transactions instead of one 32 B sector per element).
extern "C" __global__ void __launch_bounds__(256) dspark_chain_c(
    float* __restrict__ cand_s, unsigned* __restrict__ cand_o,
    const __nv_bfloat16* __restrict__ logits,  // [vocab, B] col-major (token-major [B,vocab])
    const __nv_bfloat16* __restrict__ w1,      // [vocab, rank] row-major
    const __nv_bfloat16* __restrict__ w2t,     // [rank, vocab] TRANSPOSED row-major
    const unsigned* __restrict__ prev_dev, int vocab, int rank, int B, int k) {
    extern __shared__ volatile float sm[];     // [rank] latent + [32] red_s + [32] red_o
    volatile float* latent = sm;               // [rank]
    volatile float* red_s = sm + rank;         // [32]
    volatile unsigned* red_o = (volatile unsigned*)(sm + rank + 32);
    const int tid = threadIdx.x;
    const int g = blockIdx.x;
    const int prev = (int)prev_dev[0];
    if (tid < rank) latent[tid] = b2f(w1[(long long)prev * rank + tid]);
    __syncthreads();

    float bs = -1e30f; int bo = -1;
    const long long lrow = (long long)k * vocab;
    for (int o = g * 256 + tid; o < vocab; o += gridDim.x * 256) {
        // token-major logits (gemm_binv writes C[n*M + m], ld = vocab)
        float s = b2f(logits[lrow + o]);
        const __nv_bfloat16* wcol = w2t + o;   // column o: w2t[i*vocab + o]
        for (int i = 0; i < rank; ++i)
            s = fmaf(b2f(wcol[(long long)i * vocab]), latent[i], s);
        if (s > bs || bo < 0) { bs = s; bo = o; }
    }
    for (int off = 16; off; off >>= 1) {
        float s2 = __shfl_xor_sync(0xffffffffu, bs, off);
        int   o2 = __shfl_xor_sync(0xffffffffu, bo, off);
        float s; int o; chain_better(bs, bo, s2, o2, &s, &o); bs = s; bo = o;
    }
    if ((tid & 31) == 0) { red_s[tid >> 5] = bs; red_o[tid >> 5] = (unsigned)bo; }
    __syncthreads();
    if (tid == 0) {
        float s = red_s[0]; int o = (int)red_o[0];
        for (int w = 1; w < 256 / 32; ++w) {
            float s2; int o2; chain_better(s, o, red_s[w], (int)red_o[w], &s2, &o2); s = s2; o = o2;
        }
        cand_s[g] = s; cand_o[g] = (unsigned)o;
    }
}

// deterministic per-tensor stamp (build.rs passes -DKERNEL_BUILD_ID; GpuModel::
// assert_kernel_build_id launches this on every module it loads)
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

// ---- interleaved-pair rope (the DSpark oracle's convention) --------------------
// The DF2 rope_b pairs (d, d+half) (half-split); the DSpark reference rope pairs
// (2j, 2j+1) (interleaved) with angle = pos * freq[j]. x is token-major [B, nh*hd];
// cos/sin are the gathered rows [B, rdim] (rdim = head_dim, duplicated-freqs table).
extern "C" __global__ void dspark_rope_b(__nv_bfloat16* x, const float* cos, const float* sin,
    int nh, int hd, int rdim, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = hd / 2;
    int per_seq = nh * half;
    int total = B * per_seq;
    if (idx >= total) return;
    int b = idx / per_seq;
    int rem = idx % per_seq;
    int head = rem / half;
    int j = rem % half;
    long long base = (long long)b * (nh * hd) + (long long)head * hd;
    long long cb = (long long)b * rdim + j;
    float x1 = b2f(x[base + 2 * j]);
    float x2 = b2f(x[base + 2 * j + 1]);
    float c = cos[cb], s = sin[cb];
    x[base + 2 * j] = f2b(x1 * c - x2 * s);
    x[base + 2 * j + 1] = f2b(x2 * c + x1 * s);
}
