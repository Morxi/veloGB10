// Elementwise + recurrent kernels for Qwen3.5-0.8B GPU forward (f32).
// All pointers are device pointers unless noted. Compiled to PTX for sm_121.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// DF2 block-16: window_B packs the sliding window and the draft block length. The low bits
// MUST hold BLOCK (8 today, 16 with --df2-block 16), so the field is 5 bits wide — at 4 bits
// BLOCK=16 would be truncated to 0 and every band-attention launch would read B=0.
#define DF2_WB_SHIFT 5
#define DF2_WB_MASK  0x1F

// ---- Build-ID stamp: makes a stale PTX impossible to run silently ----
// build.rs hashes the .cu bytes and passes the result as -DKERNEL_BUILD_ID. GpuModel::load reads this
// global back out of the loaded module and asserts it equals the ID compiled into the BINARY. A fresh
// binary next to old kernels then fails loudly at startup instead of launching a kernel whose ABI it no
// longer agrees with -- which is how we once got CUDA_ERROR_ILLEGAL_ADDRESS out of correct code.
#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }


#define WARP 32

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }

// ---- RMSNorm (Qwen3.5): out = x * rsqrt(mean(x^2)+eps) * (1+w) ----
// one block per vector of length n (n <= 1024, single block reduce)
extern "C" __global__ void rmsnorm_qwen(float* out, const float* x, const float* w, int n, float eps) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    float v = (tid < n) ? x[tid] : 0.0f;
    v = v * v;
    s[tid] = v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) s[tid] += s[tid + s2];
        __syncthreads();
    }
    float inv = rsqrtf(s[0] / (float)n + eps);
    if (tid < n) out[tid] = x[tid] * inv * (1.0f + w[tid]);
}

// ---- Gated RMSNorm (linear attn): out = rms(x) * w * silu(z) ----
// one block per head; normalize m elements at offset head*m.
extern "C" __global__ void rmsnorm_gated(float* out, const float* x, const float* z, const float* w, int m, float eps) {
    extern __shared__ float s[];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    float v = (tid < m) ? x[head * m + tid] : 0.0f;
    v = v * v;
    s[tid] = v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) s[tid] += s[tid + s2];
        __syncthreads();
    }
    float inv = rsqrtf(s[0] / (float)m + eps);
    if (tid < m) out[head * m + tid] = x[head * m + tid] * inv * w[tid] * silu_f(z[head * m + tid]);
}

// ---- per-head RMSNorm on q/k: normalize each head's hd-vector with shared weight w[hd] ----
extern "C" __global__ void rmsnorm_perhead(float* out, const float* x, const float* w, int nh, int hd, float eps) {
    // one block per head, blockDim.x >= hd (hd=256)
    int head = blockIdx.x;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    float v = (tid < hd) ? x[head * hd + tid] : 0.0f;
    v = v * v;
    s[tid] = v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) s[tid] += s[tid + s2];
        __syncthreads();
    }
    float inv = rsqrtf(s[0] / (float)hd + eps);
    if (tid < hd) out[head * hd + tid] = x[head * hd + tid] * inv * (1.0f + w[tid]);
}

// ---- silu(gate)*up for MLP ----
extern "C" __global__ void silu_mul(float* out, const float* gate, const float* up, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) out[i] = silu_f(gate[i]) * up[i];
}

// ---- residual add: out = a + b ----
extern "C" __global__ void add_residual(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// ---- apply sigmoid gate to attention output (in place) ----
extern "C" __global__ void sigmoid_gate(float* attn, const float* gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) attn[i] *= sigmoid_f(gate[i]);
}

// ---- rotate_half RoPE on first rdim of each head (in place) ----
// x layout: [nh, hd]; operates per head, first rdim dims.
extern "C" __global__ void rope_rot_half(float* x, const float* cos, const float* sin, int nh, int hd, int rdim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nh * (rdim / 2);
    if (idx >= total) return;
    int pair = idx % (rdim / 2);
    int head = idx / (rdim / 2);
    int half = rdim / 2;
    int base = head * hd;
    float x1 = x[base + pair];
    float x2 = x[base + pair + half];
    float c = cos[pair];
    float s = sin[pair];
    x[base + pair] = x1 * c - x2 * s;
    x[base + pair + half] = x2 * c + x1 * s;
}

// ---- GQA attention single-token decode ----
// q: [nh, hd]; k_cache/v_cache: [nkv, kv_stride, hd]; valid positions = pos_count (<= kv_stride).
// out: [nh, hd]; one block per query head, blockDim.x = hd (256).
extern "C" __global__ void gqa_attention(float* out, const float* q,
                                         const float* k_cache, const float* v_cache,
                                         int pos_count, int kv_stride, int nh, int nkv, int hd, float scale) {
    int qh = blockIdx.x;
    int kvh = qh / (nh / nkv);
    int d = threadIdx.x; // 0..hd-1
    extern __shared__ float sh[];
    float* scores = sh;          // pos_count
    float* red = sh + pos_count; // blockDim.x reduce scratch
    const float* qv = q + qh * hd;

    // scores[t] = scale * sum_d q[d]*k[t,d]
    for (int t = 0; t < pos_count; t++) {
        const float* kv = k_cache + (kvh * kv_stride + t) * hd;
        float dot = qv[d] * kv[d];
        red[d] = dot;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) scores[t] = red[0] * scale;
        __syncthreads();
    }
    float mx = -1e30f;
    if (d == 0) {
        for (int t = 0; t < pos_count; t++) mx = fmaxf(mx, scores[t]);
        red[0] = mx;
    }
    __syncthreads();
    mx = red[0];
    if (d == 0) {
        float se = 0.0f;
        for (int t = 0; t < pos_count; t++) { scores[t] = __expf(scores[t] - mx); se += scores[t]; }
        red[0] = se;
    }
    __syncthreads();
    float inv = 1.0f / red[0];
    float acc = 0.0f;
    for (int t = 0; t < pos_count; t++) {
        acc += scores[t] * inv * v_cache[(kvh * kv_stride + t) * hd + d];
    }
    out[qh * hd + d] = acc;
}

// ---- write current k,v into cache at position pos (stride = kv_stride) ----
extern "C" __global__ void write_kv(float* k_cache, float* v_cache,
                                    const float* k_new, const float* v_new,
                                    int pos, int kv_stride, int nkv, int hd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nkv * hd;
    if (idx >= total) return;
    int h = idx / hd;
    int d = idx % hd;
    k_cache[(h * kv_stride + pos) * hd + d] = k_new[idx];
    v_cache[(h * kv_stride + pos) * hd + d] = v_new[idx];
}

// ---- split q_proj output [nh, hd*2] into q[nh,hd] and gate[nh,hd] ----
extern "C" __global__ void split_qgate(float* q, float* gate, const float* qg, int nh, int hd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nh * hd;
    if (idx >= total) return;
    int head = idx / hd;
    int d = idx % hd;
    q[idx] = qg[head * hd * 2 + d];
    gate[idx] = qg[head * hd * 2 + hd + d];
}

// ---- conv1d depthwise causal step (in place): shift state, conv, silu ----
// x: [conv_dim] new sample (in/out); state: [conv_dim, k]; w: [conv_dim, k]
extern "C" __global__ void conv1d_step(float* x, float* state, const float* w, int conv_dim, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float* st = state + c * k;
    for (int j = 1; j < k; j++) st[j - 1] = st[j];
    st[k - 1] = x[c];
    float acc = 0.0f;
    for (int j = 0; j < k; j++) acc += w[c * k + j] * st[j];
    x[c] = silu_f(acc);
}

// ---- Gated delta-rule recurrent step (linear attention), one block per head ----
// Inputs (post-conv): qkv: [conv_dim] split q[nh*kd],k[nh*kd],v[nh*vd]; b[nh],a[nh]
// state: [nh, kd, vd]; out: [nh, vd] (core attention output, pre-norm)
// one block per head, blockDim = kd (128). kd==vd here.
extern "C" __global__ void delta_step(float* out,
                                      const float* q_in, const float* k_in, const float* v_in,
                                      const float* b_in, const float* a_in,
                                      float* state,
                                      int nh, int kd, int vd,
                                      const float* a_log, const float* dt_bias) {
    int head = blockIdx.x;
    int a = threadIdx.x; // 0..kd-1
    extern __shared__ float sh[];
    float* Srow = sh;                 // kd
    float* kv_mem = sh + kd;          // vd
    float* vbuf = sh + kd + vd;       // vd
    float* delta = sh + kd + 2 * vd;  // vd
    float* qrow = sh + kd + 3 * vd;   // kd
    float* krow = sh + kd + 3 * vd + kd; // kd

    float* S = state + head * kd * vd; // [kd, vd]
    float beta = sigmoid_f(b_in[head]);
    float sp = a_in[head] + dt_bias[head];
    sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
    float gt = __expf(-__expf(a_log[head]) * sp);

    // q,k: l2norm per head; q *= scale
    float qv = q_in[head * kd + a];
    float kv = k_in[head * kd + a];
    Srow[a] = qv * qv;
    __syncthreads();
    for (int s2 = kd / 2; s2 > 0; s2 >>= 1) { if (a < s2) Srow[a] += Srow[a + s2]; __syncthreads(); }
    float qn = rsqrtf(Srow[0] + 1e-6f);
    __syncthreads(); // ensure all read Srow[0] before we reuse Srow
    qv *= qn;
    Srow[a] = kv * kv;
    __syncthreads();
    for (int s2 = kd / 2; s2 > 0; s2 >>= 1) { if (a < s2) Srow[a] += Srow[a + s2]; __syncthreads(); }
    float kn = rsqrtf(Srow[0] + 1e-6f);
    __syncthreads();
    kv *= kn;                       // normalized k
    float scale = 1.0f / sqrtf((float)kd);
    qv *= scale;
    qrow[a] = qv;
    krow[a] = kv;                   // store normalized k
    __syncthreads();

    // S *= gt
    for (int bb = 0; bb < vd; bb++) S[a * vd + bb] *= gt;
    __syncthreads();

    // kv_mem[bb] = sum_a S[a,bb]*k[a]  (normalized k)
    int bb = a;
    float km = 0.0f;
    for (int aa = 0; aa < kd; aa++) km += S[aa * vd + bb] * krow[aa];
    kv_mem[bb] = km;
    vbuf[bb] = v_in[head * vd + bb];
    __syncthreads();

    // delta[bb] = (v[bb]-kv_mem[bb])*beta
    delta[bb] = (vbuf[bb] - kv_mem[bb]) * beta;
    __syncthreads();

    // S[a,bb] += k[a]*delta[bb]  (normalized k)
    float kk = krow[a];
    for (int bbb = 0; bbb < vd; bbb++) S[a * vd + bbb] += kk * delta[bbb];
    __syncthreads();

    // out[bb] = sum_a S[a,bb]*q[a]
    float o = 0.0f;
    for (int aa = 0; aa < kd; aa++) o += S[aa * vd + bb] * qrow[aa];
    out[head * vd + bb] = o;
}

// argmax is done on device (two-pass) below.

// ---- bf16<->f32 conversions ----
extern "C" __global__ void f32tobf16(__nv_bfloat16* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}
extern "C" __global__ void bf16tof32(float* dst, const __nv_bfloat16* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

// ---- fused: residual += mixer, then out = rmsnorm(residual, w) * (1+w) ----
// one block over vector length n.
extern "C" __global__ void fused_residual_rmsnorm(float* out, float* residual, const float* mixer,
                                                  const float* w, int n, float eps) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    float v = (tid < n) ? (residual[tid] + mixer[tid]) : 0.0f;
    residual[tid] = v;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)n + eps);
    if (tid < n) out[tid] = v * inv * (1.0f + w[tid]);
}

// ---- argmax pass 1: each block reduces a chunk of logits -> (val,idx) into globals ----
extern "C" __global__ void argmax_pass1(int* out_idx, float* out_val, const float* logits, int n) {
    int bid = blockIdx.x;
    int bs = blockDim.x;
    extern __shared__ float sv[];
    int* si = (int*)(sv + bs);
    int tid = threadIdx.x;
    int gid = bid * bs + tid;
    float val = -1e30f; int idx = -1;
    if (gid < n) { val = logits[gid]; idx = gid; }
    sv[tid] = val; si[tid] = idx;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) { if (sv[tid + s2] > sv[tid]) { sv[tid] = sv[tid + s2]; si[tid] = si[tid + s2]; } }
        __syncthreads();
    }
    if (tid == 0) { out_val[bid] = sv[0]; out_idx[bid] = si[0]; }
}

// ---- argmax pass 2: reduce per-block winners (m of them) into one index ----
extern "C" __global__ void argmax_pass2(int* token, const int* idxs, const float* vals, int m) {
    extern __shared__ float sv[];
    int* si = (int*)(sv + blockDim.x);
    int tid = threadIdx.x;
    float val = -1e30f; int idx = -1;
    if (tid < m) { val = vals[tid]; idx = idxs[tid]; }
    sv[tid] = val; si[tid] = idx;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) { if (sv[tid + s2] > sv[tid]) { sv[tid] = sv[tid + s2]; si[tid] = si[tid + s2]; } }
        __syncthreads();
    }
    if (tid == 0) token[0] = si[0];
}


// ===========================================================================================
// S3F — DFlash2 draft-block kernels (additive; the gpu_batch.cu serving family is untouched).
// bf16 activations (col-major [dim, B]), bf16 weights (row-major [out, in]), fp32 accumulate,
// fixed reduction orders (cross-rank bit-identity), no atomics. Reused serving kernels:
// rmsnorm_b / rmsnorm_perhead_b / rope_b / gather_rope_b / add_residual_b / silu_mul_b /
// write_kv_b (gpu_batch.cu). New here: the skinny GEMM, the ctx-side tiled GEMM, the grouped
// dynamic causal conv, and the band-masked dual-source GQA attention.
// ===========================================================================================

// ---- S3F skinny GEMM (gemm_dsp_b): one block per R output rows, 256 threads, 16 B vectorized
// W loads (uint4 = 8 bf16), x col-major [inn, M] read as M uint4 column loads (8 consecutive k
// each, 16 B-aligned since inn % 8 == 0) reused across the R rows. Templated on M (batch columns)
// and R (output rows/block) so the acc[R][M] register arrays never land in local memory
// (AGENTS.md §4). Fixed-order reduce: warp shuffle down, then the 8 warp partials in ascending
// warp order. W row-major [outn, inn]; x col-major [inn, M]; out col-major [outn, M].
template <int M, int R>
__device__ __forceinline__ void gemm_dsp_impl(__nv_bfloat16* __restrict__ out,
                                              const __nv_bfloat16* __restrict__ w,
                                              const __nv_bfloat16* __restrict__ x,
                                              int outn, int inn) {
    const int n0 = blockIdx.x * R;
    const int nvec = inn >> 3;                       // inn % 8 == 0 for every DFlash2 tensor
    const uint4* wrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int n = n0 + r;
        wrow[r] = reinterpret_cast<const uint4*>(w + (long long)(n < outn ? n : outn - 1) * (long long)inn);
    }
    const uint4* xcol[M];
#pragma unroll
    for (int m = 0; m < M; ++m)
        xcol[m] = reinterpret_cast<const uint4*>(x + (long long)m * inn);   // column m, k-major
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    for (int i = threadIdx.x; i < nvec; i += 256) {
        uint4 wp[R];
#pragma unroll
        for (int r = 0; r < R; ++r) wp[r] = wrow[r][i];
        uint4 xm[M];
#pragma unroll
        for (int m = 0; m < M; ++m) xm[m] = xcol[m][i];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const __nv_bfloat16* w8 = reinterpret_cast<const __nv_bfloat16*>(&wp[r]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float wv = __bfloat162float(w8[j]);
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const __nv_bfloat16* x8 = reinterpret_cast<const __nv_bfloat16*>(&xm[m]);
                    acc[r][m] = fmaf(wv, __bfloat162float(x8[j]), acc[r][m]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
            for (int off = 16; off; off >>= 1) acc[r][m] += __shfl_down_sync(0xffffffffu, acc[r][m], off);
    __shared__ float wsum[8][R][M];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int m = 0; m < M; ++m) wsum[wid][r][m] = acc[r][m];
    }
    __syncthreads();
    if (wid == 0 && lane < 8) {
        // Mask covers exactly lanes 0..7 (the 8 warp partials) — a full mask here would be UB.
        const unsigned mask8 = 0xFFu;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float t[M];
#pragma unroll
            for (int m = 0; m < M; ++m) t[m] = wsum[lane][r][m];
#pragma unroll
            for (int m = 0; m < M; ++m)
#pragma unroll
                for (int off = 4; off; off >>= 1) t[m] += __shfl_down_sync(mask8, t[m], off);
            if (lane == 0 && n0 + r < outn) {
#pragma unroll
                for (int m = 0; m < M; ++m) out[(long long)m * outn + n0 + r] = __float2bfloat16(t[m]);
            }
        }
    }
}

// The two R values measured (R=4 best — see PLAN/B8_S3F_REPORT.md R4). M=8 = the fixed draft block.
extern "C" __global__ void __launch_bounds__(256) gemm_dsp_b_m8_r2(
    __nv_bfloat16* out, const __nv_bfloat16* w, const __nv_bfloat16* x, int outn, int inn) {
    gemm_dsp_impl<8, 2>(out, w, x, outn, inn);
}
extern "C" __global__ void __launch_bounds__(256) gemm_dsp_b_m8_r4(
    __nv_bfloat16* out, const __nv_bfloat16* w, const __nv_bfloat16* x, int outn, int inn) {
    gemm_dsp_impl<8, 4>(out, w, x, outn, inn);
}

// ---- S3F ctx-side tiled GEMM (gemm_tiled_b): deterministic register-blocked bf16 GEMM for the
// large-M cases (fc [5120,25600] @ [25600,C], k/v_proj [1024,5120] @ [5120,C]). out [N, M]
// col-major; W [N, K] row-major; x [K, M] col-major (the engine convention, element (k,m) at
// m*K + k). 8x8 register blocking, block (16,16)=256 threads, TK=16 smem staging. Fixed
// ascending-k accumulation, no atomics.
#define DF2_TN 128
#define DF2_TM 128
#define DF2_TK 16
extern "C" __global__ void __launch_bounds__(256) gemm_tiled_b(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ x, int N, int K, int M) {
    __shared__ __nv_bfloat16 sW[DF2_TN][DF2_TK];
    __shared__ __nv_bfloat16 sX[DF2_TK][DF2_TM];
    const int ty = threadIdx.y, tx = threadIdx.x;      // 0..15
    const int n0 = blockIdx.y * DF2_TN;
    const int m0 = blockIdx.x * DF2_TM;
    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += DF2_TK) {
        // cooperative load sW[TN][TK]
        for (int i = ty * 16 + tx; i < DF2_TN * DF2_TK; i += 256) {
            const int r = i / DF2_TK, c = i % DF2_TK;
            const int nr = n0 + r, kc = k0 + c;
            sW[r][c] = (nr < N && kc < K) ? w[(long long)nr * K + kc] : __float2bfloat16(0.f);
        }
        // cooperative load sX[TK][TM] — x is col-major [K, M] (element (kc, mc) at mc*K + kc).
        for (int i = ty * 16 + tx; i < DF2_TK * DF2_TM; i += 256) {
            const int r = i / DF2_TM, c = i % DF2_TM;
            const int kc = k0 + r, mc = m0 + c;
            sX[r][c] = (kc < K && mc < M) ? x[(long long)mc * K + kc] : __float2bfloat16(0.f);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < DF2_TK; ++kk) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float wv = __bfloat162float(sW[ty * 8 + i][kk]);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[i][j] = fmaf(wv, __bfloat162float(sX[kk][tx * 8 + j]), acc[i][j]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int nr = n0 + ty * 8 + i;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int mc = m0 + tx * 8 + j;
            if (nr < N && mc < M) out[(long long)mc * N + nr] = __float2bfloat16(acc[i][j]);
        }
    }
}

// ---- S3F grouped dynamic causal 2-tap conv (DECISION M; model.py _grouped_dynamic_convolve).
// out[r,c] = (base[0,c] + dyn[r,0,g])*x[r,c] + (r>=1 ? (base[1,c] + dyn[r,1,g])*x[r-1,c] : 0)
// with g = c / gs (grouped channel). x/out col-major [hidden, n]; base [k, hidden] row-major for
// ONE side; dyn_all col-major [2*k*groups, n] (the full kernel_projection output); dyn_side is the
// row offset into dyn_all for this side (0 = prepare/side 0, k*groups = finish/side 1). k=2.
extern "C" __global__ void conv2_dynamic_b(__nv_bfloat16* __restrict__ out,
                                           const __nv_bfloat16* __restrict__ x,
                                           const __nv_bfloat16* __restrict__ dyn_all,
                                           const __nv_bfloat16* __restrict__ base,
                                           int hidden, int n, int groups, int gs,
                                           int dyn_side, int dyn_stride) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)n * hidden;
    if (idx >= total) return;
    const int r = (int)(idx / hidden);
    const int c = (int)(idx % hidden);
    const int g = c / gs;
    const float b0 = __bfloat162float(base[c]);
    const float b1 = __bfloat162float(base[hidden + c]);
    // dyn_all is [2*k*groups, n] col-major; `dyn_side` is the SIDE's row offset (side * k * groups)
    // and each tap o adds o*groups rows. dyn_all[r, dyn_side + o*groups + g].
    const float d0 = __bfloat162float(dyn_all[(long long)r * dyn_stride + dyn_side + g]);
    const float d1 = __bfloat162float(dyn_all[(long long)r * dyn_stride + dyn_side + groups + g]);
    float acc = (b0 + d0) * __bfloat162float(x[(long long)r * hidden + c]);
    if (r >= 1) acc += (b1 + d1) * __bfloat162float(x[(long long)(r - 1) * hidden + c]);
    out[(long long)r * hidden + c] = __float2bfloat16(acc);
}

// ---- S3F band-masked dual-source GQA attention (DECISION C). One block per (query row, q head),
// blockDim = hd (128). q [nh*hd, B] col-major (post q_norm + RoPE); k/v cache [nkv, stride, hd]
// (ctx rows 0..C-1, block rows C..C+7; stride is the allocated row pitch). Band: visible keys
// j in [lo, ntot) where lo = max(0, pos[b] - (window-1)); the upper band side is vacuous
// (in-block separation <= 7). Two-pass fp32 softmax (fixed ascending key order), deterministic.
// Args packed to fit the 12-arg cudarc launch cap: ntot_stride = (ntot<<16)|stride (both <2^13),
// nh_packed = (nh<<20)|(hd<<10)|nkv, window_B = (window<<4)|B.
extern "C" __global__ void __launch_bounds__(128) gqa_attn_band_b(
    __nv_bfloat16* __restrict__ out,          // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ q,      // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ k_cache,// [nkv, stride, hd]
    const __nv_bfloat16* __restrict__ v_cache,// [nkv, stride, hd]
    const int* pos,                            // [B] block key positions (C..C+7)
    long long ntot_stride, int nh_packed, int window_B, float scale) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const int ntot   = (int)(ntot_stride >> 16);
    const int stride = (int)(ntot_stride & 0xFFFF);
    const int window = window_B >> DF2_WB_SHIFT;
    // VOLATILE smem is LOAD-BEARING: ptxas -O2/-O3 (CUDA 13.0.88, sm_121) miscompiles this
    // kernel's barrier-ordered scores/red pattern — sporadically reorders a shared access
    // across bar.sync, so one block reads stale smem and emits a whole-block wrong output
    // (~1 bad launch per 1e2..1e4, always the tail blocks). volatile forbids the reorder;
    // ptxas -O0/-O1 and volatile-at-O3 are bit-stable over 4x10k launches (repro:
    // kernels/tests/sm121_ptxas_race_repro.cu). Do NOT silently drop the volatile.
    extern __shared__ volatile float sm[];
    const int b = blockIdx.x / nh;
    const int qh = blockIdx.x % nh;
    const int kvh = qh / (nh / nkv);
    const int d = threadIdx.x;                 // 0..hd-1 (blockDim == hd)
    const int qp = pos[b];
    int lo = qp - (window - 1);
    if (lo < 0) lo = 0;

    const int B = window_B & DF2_WB_MASK;             // S10R: unpacked (scores bound; see ring twin)
    volatile float* qs = sm;              // [hd]
    // S10R — WINDOW-BOUNDED scores region, relative index j - lo (same bound + proof as
    // gqa_attn_band_ring_b; PLAN/B8_S10R_DISSECTION.md §4): min(window+B, ntot) slots.
    const int scores_len = (window + B < ntot) ? (window + B) : ntot;
    volatile float* scores = qs + hd;     // [scores_len], relative index j - lo
    volatile float* red = scores + scores_len;  // [32]
    qs[d] = __bfloat162float(q[(long long)b * (nh * hd) + (long long)qh * hd + d]);
    __syncthreads();

    const __nv_bfloat16* kbase = k_cache + (long long)kvh * (long long)stride * hd;
    const __nv_bfloat16* vbase = v_cache + (long long)kvh * (long long)stride * hd;

    // pass 1: scores + block max over [lo, ntot) — thread d owns keys j == lo+d (mod hd).
    float mx = -1e30f;
    for (int j = lo + d; j < ntot; j += hd) {
        const __nv_bfloat16* kr = kbase + (long long)j * hd;
        float s = 0.f;
        for (int dd = 0; dd < hd; ++dd) s = fmaf(qs[dd], __bfloat162float(kr[dd]), s);
        s *= scale;
        scores[j - lo] = s;
        mx = fmaxf(mx, s);
    }
    // The cross-warp reduce runs on lanes 0..nw (nw = hd/32); its shuffle mask MUST cover exactly
    // those lanes (a full 0xffffffff mask with the rest of the warp outside the branch deadlocks).
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if ((d & 31) == 0) red[d >> 5] = mx;
    __syncthreads();
    // cross-warp reduce: SERIAL ASCENDING on thread 0 (fixed order = the host mirror's; no
    // partial-mask/runtime-offset shuffles under divergence — the sm_121 nondeterminism hunt)
    if (d == 0) {
        float v = red[0];
        for (int w = 1; w < hd / 32; ++w) v = fmaxf(v, red[w]);
        red[0] = v;
    }
    __syncthreads();
    mx = red[0];

    // pass 2: exp + sum over [lo, ntot). The accurate expf (~2 ulp) keeps the softmax close to
    // the oracle's correctly-rounded exp (the fast __expf is ~2+floor(1.16|x|) ulp).
    float l = 0.f;
    for (int j = lo + d; j < ntot; j += hd) { const float e = expf(scores[j - lo] - mx); scores[j - lo] = e; l += e; }
    for (int off = 16; off; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
    if ((d & 31) == 0) red[d >> 5] = l;
    __syncthreads();
    // serial ascending sum on thread 0 — the mirror's exact association
    if (d == 0) {
        float v = red[0];
        for (int w = 1; w < hd / 32; ++w) v += red[w];
        red[0] = v;
    }
    __syncthreads();
    const float inv = 1.0f / red[0];

    // pass 3: PV, ascending over [lo, ntot) — thread d reads every score, accumulates out[d].
    float acc = 0.f;
    for (int j = lo; j < ntot; ++j)
        acc = fmaf(scores[j - lo] * inv, __bfloat162float(vbase[(long long)j * hd + d]), acc);
    out[(long long)b * (nh * hd) + (long long)qh * hd + d] = __float2bfloat16(acc);
}

// ===========================================================================================
// S4F (K-DF2-2/3) — additive kernels: ring-KV band attention, radix top-16, selector walk.
// No existing kernel body modified (S3F discipline). All fixed-order, no atomics except the
// order-INDEPENDENT radix histogram (min/max/add on integers — commutative, deterministic).
// ===========================================================================================

// ---- S4F ring-KV band attention (gqa_attn_band_ring_b) -----------------------
// The production attention for the incremental ctx-injection path (workdoc §3.2): ctx KV lives
// in a C_ring-deep RING (ctx seq position j at cache row j % C_ring), the 8 block rows at
// physical rows [C_ring, C_ring+8) of the [nkv, stride=C_ring+8, hd] cache. A query at absolute
// position q_pos sees the trailing band [max(0, q_pos-2047), ntot) — <= 2048 ctx keys + the 8
// block keys — WINDOW-BOUNDED (no unbounded growth; the dossier §6.2 bound). The visit order,
// two-pass fp32 softmax tree and PV accumulation are IDENTICAL to gqa_attn_band_b, so a
// non-wrapping C (C <= C_ring) reproduces S3F's numbers bit-for-bit (the probe's control).
//   physical row of key j:  j < C (ctx)  -> j % C_ring
//                           j >= C (blk) -> C_ring + (j - C)
// Args packed: ntot_stride = ((u64)ntot<<32)|((u64)C_ring<<16)|stride; window_B=(window<<4)|B.
// S5F: `ntot_dev` (optional device i32) OVERRIDES the packed ntot when non-NULL — the CUDA-graph
// replay path needs ntot to vary per replay while the launch config (smem, args) is frozen at
// capture, so the graph writes the current ntot to a device int and passes its address here.
// Pass NULL/0 to keep the exact packed-arg behavior (bit-identical — the S4F probe path).
extern "C" __global__ void __launch_bounds__(128) gqa_attn_band_ring_b(
    __nv_bfloat16* __restrict__ out,          // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ q,      // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ k_cache,// [nkv, stride, hd]
    const __nv_bfloat16* __restrict__ v_cache,
    const int* pos,                            // [B] block key positions (ABSOLUTE C..C+7)
    unsigned long long ntot_stride, const int* ntot_dev, int nh_packed, int window_B, float scale) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const int ntot    = ntot_dev ? *ntot_dev : (int)(ntot_stride >> 32);
    const int C_ring  = (int)((ntot_stride >> 16) & 0xFFFF);
    const int stride  = (int)(ntot_stride & 0xFFFF);
    const int window  = window_B >> DF2_WB_SHIFT;
    const int B       = window_B & DF2_WB_MASK;
    const int C = ntot - B;                    // committed ctx rows
    // VOLATILE smem is LOAD-BEARING: ptxas -O2/-O3 (CUDA 13.0.88, sm_121) miscompiles this
    // kernel's barrier-ordered scores/red pattern — sporadically reorders a shared access
    // across bar.sync, so one block reads stale smem and emits a whole-block wrong output
    // (~1 bad launch per 1e2..1e4, always the tail blocks). volatile forbids the reorder;
    // ptxas -O0/-O1 and volatile-at-O3 are bit-stable over 4x10k launches (repro:
    // kernels/tests/sm121_ptxas_race_repro.cu). Do NOT silently drop the volatile.
    extern __shared__ volatile float sm[];
    const int b  = blockIdx.x / nh;
    const int qh = blockIdx.x % nh;
    const int kvh = qh / (nh / nkv);
    const int d = threadIdx.x;
    const int qp = pos[b];
    int lo = qp - (window - 1);
    if (lo < 0) lo = 0;

    volatile float* qs = sm;              // [hd]
    // S10R — the scores region is WINDOW-BOUNDED, indexed RELATIVELY (scores[j - lo]): the
    // visit range [lo, ntot) spans at most window+B-1 entries (lo = max(0,qp-(window-1)),
    // qp >= C, ntot = C+B), so min(window+B, ntot) slots always suffice. The old absolute
    // layout ([ntot] slots) sized smem by TOTAL context and capped ctx at 12120 (the 48 KiB
    // dynamic-smem default); this bound is ctx-INDEPENDENT (PLAN/B8_S10R_DISSECTION.md §4).
    // Values stored/loaded and every reduction tree are UNCHANGED — only the base offset.
    const int scores_len = (window + B < ntot) ? (window + B) : ntot;
    volatile float* scores = qs + hd;     // [scores_len], relative index j - lo
    volatile float* red = scores + scores_len;  // [32]
    qs[d] = __bfloat162float(q[(long long)b * (nh * hd) + (long long)qh * hd + d]);
    __syncthreads();

    const __nv_bfloat16* kbase = k_cache + (long long)kvh * (long long)stride * hd;
    const __nv_bfloat16* vbase = v_cache + (long long)kvh * (long long)stride * hd;

    // pass 1: scores + band max, ascending key index (same schedule as gqa_attn_band_b).
    float mx = -1e30f;
    for (int j = lo + d; j < ntot; j += hd) {
        const int rj = (j < C) ? (j % C_ring) : (C_ring + (j - C));
        const __nv_bfloat16* kr = kbase + (long long)rj * hd;
        float s = 0.f;
        for (int dd = 0; dd < hd; ++dd) s = fmaf(qs[dd], __bfloat162float(kr[dd]), s);
        s *= scale;
        scores[j - lo] = s;
        mx = fmaxf(mx, s);
    }
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if ((d & 31) == 0) red[d >> 5] = mx;
    __syncthreads();
    // cross-warp reduce: SERIAL ASCENDING on thread 0 (fixed order = the host mirror's; no
    // partial-mask/runtime-offset shuffles under divergence — the sm_121 nondeterminism hunt)
    if (d == 0) {
        float v = red[0];
        for (int w = 1; w < hd / 32; ++w) v = fmaxf(v, red[w]);
        red[0] = v;
    }
    __syncthreads();
    mx = red[0];

    // pass 2: exp + sum (accurate expf, ~2 ulp — the documented S3F tail).
    float l = 0.f;
    for (int j = lo + d; j < ntot; j += hd) { const float e = expf(scores[j - lo] - mx); scores[j - lo] = e; l += e; }
    for (int off = 16; off; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
    if ((d & 31) == 0) red[d >> 5] = l;
    __syncthreads();
    // serial ascending sum on thread 0 — the mirror's exact association
    if (d == 0) {
        float v = red[0];
        for (int w = 1; w < hd / 32; ++w) v += red[w];
        red[0] = v;
    }
    __syncthreads();
    const float inv = 1.0f / red[0];

    // pass 3: PV, ascending over the band.
    float acc = 0.f;
    for (int j = lo; j < ntot; ++j) {
        const int rj = (j < C) ? (j % C_ring) : (C_ring + (j - C));
        acc = fmaf(scores[j - lo] * inv, __bfloat162float(vbase[(long long)rj * hd + d]), acc);
    }
    out[(long long)b * (nh * hd) + (long long)qh * hd + d] = __float2bfloat16(acc);
}

// ---- S4F deterministic radix top-16 (top16_b) --------------------------------
// The total order (DECISION L): (logit DESC, token-id ASC) — the SGLang _radix_topk answer,
// bit-deterministic. Two-level radix over the 16-bit bf16 total-order key
//   key16 = (bits & 0x8000) ? (~bits & 0xFFFF) : (bits | 0x8000)
// (monotone in the float value; sign folded in). bf16 keys are VALUE-UNIQUE, so the level-2
// group is exactly the tie set (equal logits) — resolved by ascending id, which is precisely
// (logit desc, id asc). Histogram atomics are INTEGER adds — order-independent, so the output
// is a pure function of the input bytes (the probe asserts two runs identical).
// Documented divergence vs the oracle's f32 compare: -0.0 vs +0.0 compare equal in f32 but
// order +0 > -0 by key; the probe asserts no +-0 pair straddles a top-16 boundary.
// grid: one block per row; logits bf16 [vocab, rows] col-major; out_vals f32 + out_ids u32
// [16, rows] col-major (element (k, row) at k*rows + row).
#define TOP16_K 16
#define TOP16_MAX_ENT 24
__device__ __forceinline__ unsigned short df2_key16(unsigned short bits) {
    return (bits & 0x8000u) ? (unsigned short)((~bits) & 0xFFFFu) : (unsigned short)(bits | 0x8000u);
}
__device__ __forceinline__ float df2_key_to_f32(unsigned key) {
    // inverse of df2_key16, then bf16->f32 (low 16 bits zero)
    const unsigned b16 = (key & 0x8000u) ? (key & 0x7FFFu) : ((~key) & 0xFFFFu);
    return __uint_as_float(b16 << 16);
}
extern "C" __global__ void __launch_bounds__(256) top16_b(
    float* __restrict__ out_vals, unsigned* __restrict__ out_ids,
    const __nv_bfloat16* __restrict__ logits, int vocab, int rows) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* lg = logits + (long long)row * vocab;
    __shared__ int hist[256];
    __shared__ int hist2[256];
    __shared__ int sdata[256];
    __shared__ int sh_thr[2];
    __shared__ int sh_need;
    __shared__ int sh_nent;
    __shared__ unsigned tie_ids[256 * TOP16_K];
    __shared__ unsigned ent_id[TOP16_MAX_ENT * 16];   // strictly-above entries (<= 15 real)
    __shared__ unsigned ent_key[TOP16_MAX_ENT * 16];
    const int tid = threadIdx.x;

    // ---- level 1: histogram of key16 >> 8 ----
    hist[tid] = 0;
    __syncthreads();
    for (int i = tid; i < vocab; i += 256)
        atomicAdd(&hist[df2_key16(*reinterpret_cast<const unsigned short*>(lg + i)) >> 8], 1);
    __syncthreads();
    if (tid == 255) {
        int cum = 0, bin = 255;
        while (bin >= 0) { cum += hist[bin]; if (cum >= TOP16_K) break; --bin; }
        sh_thr[0] = bin;
    }
    __syncthreads();
    const int tbin = sh_thr[0];
    sdata[tid] = (tid > tbin) ? hist[tid] : 0;
    __syncthreads();
    for (int s2 = 128; s2 > 0; s2 >>= 1) {
        if (tid < s2) sdata[tid] += sdata[tid + s2];
        __syncthreads();
    }
    const int above1 = sdata[0];              // ids strictly above the threshold bin (< 16)

    // ---- level 2: sub-histogram (key16 & 0xFF) of the threshold bin ----
    hist2[tid] = 0;
    __syncthreads();
    for (int i = tid; i < vocab; i += 256) {
        const unsigned key = df2_key16(*reinterpret_cast<const unsigned short*>(lg + i));
        if ((int)(key >> 8) == tbin) atomicAdd(&hist2[key & 0xFF], 1);
    }
    __syncthreads();
    if (tid == 255) {
        int cum = above1, sub = 255;
        while (sub >= 0) { cum += hist2[sub]; if (cum >= TOP16_K) break; --sub; }
        sh_thr[1] = sub;
        sh_need = TOP16_K - (cum - hist2[sub]);   // ids to take from the tie sub-bin (>= 1)
    }
    __syncthreads();
    const int tsub = sh_thr[1];
    const unsigned thr_key16 = (((unsigned)tbin) << 8) | ((unsigned)tsub & 0xFFu);
    const int need = sh_need;

    // ---- collection pass ----
    int loc_key[TOP16_MAX_ENT]; int loc_id[TOP16_MAX_ENT]; int nloc = 0;
    unsigned loc_tie[TOP16_K]; int ntie = 0;
    for (int i = tid; i < vocab; i += 256) {
        const unsigned key = df2_key16(*reinterpret_cast<const unsigned short*>(lg + i));
        if (key > thr_key16) {
            int p = 0;
            while (p < nloc && ((unsigned)loc_key[p] > key ||
                   ((unsigned)loc_key[p] == key && (unsigned)loc_id[p] < (unsigned)i))) ++p;
            if (p < TOP16_MAX_ENT) {
                for (int q = nloc < TOP16_MAX_ENT ? nloc : TOP16_MAX_ENT-1; q > p; --q)
                    { loc_key[q] = loc_key[q-1]; loc_id[q] = loc_id[q-1]; }
                loc_key[p] = (int)key; loc_id[p] = i;
                if (nloc < TOP16_MAX_ENT) ++nloc;
            }
        } else if (key == thr_key16) {
            int p = 0;
            while (p < ntie && loc_tie[p] < (unsigned)i) ++p;
            if (p < need) {
                for (int q = ntie < TOP16_K ? ntie : TOP16_K-1; q > p; --q) loc_tie[q] = loc_tie[q-1];
                loc_tie[p] = (unsigned)i;
                if (ntie < TOP16_K) ++ntie;
            }
        }
    }
    if (tid == 0) sh_nent = 0;
    __syncthreads();
    for (int q = 0; q < nloc; ++q) {
        const int slot = atomicAdd(&sh_nent, 1);
        if (slot < TOP16_MAX_ENT * 16) { ent_key[slot] = (unsigned)loc_key[q]; ent_id[slot] = (unsigned)loc_id[q]; }
    }
    for (int q = 0; q < ntie; ++q) tie_ids[tid * TOP16_K + q] = loc_tie[q];
    for (int q = ntie; q < TOP16_K; ++q) tie_ids[tid * TOP16_K + q] = 0xFFFFFFFFu;
    __syncthreads();
    const int nent = sh_nent;

    // ---- final merge (thread 0, serial): the 16 winners by (key desc, id asc) ----
    if (tid == 0) {
        unsigned vk[TOP16_K]; unsigned id[TOP16_K]; int cnt = 0;
        for (int e = 0; e < nent && e < TOP16_MAX_ENT * 16; ++e) {
            const unsigned key = ent_key[e]; const unsigned i = ent_id[e];
            int p = 0;
            while (p < cnt && (vk[p] > key || (vk[p] == key && id[p] < i))) ++p;
            if (p < TOP16_K) {
                for (int q = cnt >= TOP16_K ? TOP16_K-1 : cnt; q > p; --q) { vk[q]=vk[q-1]; id[q]=id[q-1]; }
                vk[p] = key; id[p] = i;
                if (cnt < TOP16_K) ++cnt;
            }
        }
        for (int t = 0; t < need && cnt < TOP16_K; ++t) {
            unsigned best = 0xFFFFFFFFu;
            for (int th = 0; th < 256; ++th) {
                const unsigned cand = tie_ids[th * TOP16_K];
                if (cand < best) best = cand;
            }
            if (best == 0xFFFFFFFFu) break;
            for (int th = 0; th < 256; ++th) {
                if (tie_ids[th * TOP16_K] == best) {
                    for (int q = 0; q + 1 < TOP16_K; ++q)
                        tie_ids[th * TOP16_K + q] = tie_ids[th * TOP16_K + q + 1];
                    tie_ids[th * TOP16_K + TOP16_K - 1] = 0xFFFFFFFFu;
                    break;
                }
            }
            const unsigned key = thr_key16;
            int p = 0;
            while (p < cnt && (vk[p] > key || (vk[p] == key && id[p] < best))) ++p;
            if (p < TOP16_K) {
                for (int q = cnt >= TOP16_K ? TOP16_K-1 : cnt; q > p; --q) { vk[q]=vk[q-1]; id[q]=id[q-1]; }
                vk[p] = key; id[p] = best;
                if (cnt < TOP16_K) ++cnt;
            }
        }
        for (int k = 0; k < TOP16_K; ++k) {          // row-major [rows, 16] — the walk's layout
            out_vals[row * TOP16_K + k] = df2_key_to_f32(vk[k]);
            out_ids[row * TOP16_K + k] = id[k];
        }
    }
}

// ---- S4F fused selector walk (df2_sel_walk_b) --------------------------------
// The greedy chain (DECISION L/E) in SGLang _selector_walk_kernel's shape: ONE block walks the
// 7 positions serially and computes the codebook dot products IN the kernel (the walk IS the
// consumer — no intermediate scores round-trip). Per position: a = pred_codebook[prev][tid] *
// hp[p][tid] (one product per lane); per candidate k: s_k = unary[k] + sum_r a*succ[cand_k][r]
// via a 256-lane shuffle-DOWN tree (offsets 16..1) then a SERIAL ASCENDING sum of the 8 warp
// partials on thread 0 — a single fixed order, mirrored exactly on the host. All barriers are
// block-uniform. The argmax is a strictly-greater scan (FIRST maximal index = the oracle's
// tie-break) computed redundantly by every lane — no divergence, no atomics, deterministic.
//   hp [7,256] f32; cand [7,16] u32; unary [7,16] f32; pred/succ [vocab,256] bf16 row-major;
//   tokens [7] u32; scores_out [7,16] f32 row-major [p*16+k].
// ===== F8-CATCHUP two-stage draft head =====
// Stage 1 of the quality-neutral head cut (13.5 ms bf16 full-vocab pass -> ~3 ms NVFP4 coarse
// pass + exact bf16 re-rank of a 256-token shortlist). The COARSE ranking only has to keep the
// true bf16 top-16 inside the shortlist (recall@256); the FINAL logits the tree consumes are
// recomputed EXACTLY in bf16 from the original head rows and original hiddens, so downstream
// (top16_b, walk) is bit-identical to the plain bf16 path whenever the shortlist contains the
// bf16 top-16. Both kernels are deterministic (fixed orders, no float atomics).

// Coarse shortlist: per token-column j, the 256 ids with the largest bf16 keys. Histogram of the
// order-preserving key16>>8 (256 bins), threshold bin t* = highest t with count(>t) <= 255, all
// above-t ids collected by atomic smem order (the SET is what matters — stage 2 re-ranks), then
// filled to exactly 256 from bin t* in ASCENDING id order (deterministic subset).
//   logits [vocab, 7] col-major (bf16); out_idx [7, 256] u32.
extern "C" __global__ void __launch_bounds__(256) df2_top256_b(
    unsigned* __restrict__ out_idx, const __nv_bfloat16* __restrict__ logits,
    int vocab, int cols) {
    const int col = blockIdx.x;
    if (col >= cols) return;
    const __nv_bfloat16* lg = logits + (long long)col * vocab;
    unsigned* dst = out_idx + (long long)col * 256;
    __shared__ int hist[256];
    __shared__ int sh_t;
    __shared__ int sh_cnt;
    const int tid = threadIdx.x;
    hist[tid] = 0;
    __syncthreads();
    for (int i = tid; i < vocab; i += 256)
        atomicAdd(&hist[df2_key16(*reinterpret_cast<const unsigned short*>(lg + i)) >> 8], 1);
    __syncthreads();
    if (tid == 255) {
        int cum = 0, t = 255;
        while (t >= 0) { cum += hist[t]; if (cum > 256) break; --t; }
        // threshold bin: collect keys with bin > (t+1 clamped into range); the strictly-above
        // count is <= 256 by the walk invariant (255-fill covers the degenerate top-bin case)
        sh_t = t + 1 > 255 ? 255 : t + 1;
    }
    __syncthreads();
    const int t = sh_t;
    if (tid == 0) *reinterpret_cast<volatile int*>(&sh_cnt) = 0;
    __syncthreads();
    for (int i = tid; i < vocab; i += 256) {
        const unsigned key = df2_key16(*reinterpret_cast<const unsigned short*>(lg + i));
        if ((int)(key >> 8) > t) {
            const int p = atomicAdd(&sh_cnt, 1);
            if (p < 256) dst[p] = (unsigned)i;
        }
    }
    __syncthreads();
    int have = sh_cnt < 256 ? sh_cnt : 256;
    __syncthreads();
    if (have < 256) {
        // fill the remainder from the threshold bin — one strided pass per thread, atomic
        // cursor (SET semantics only: stage 2 re-ranks, so slot order is irrelevant)
        for (int i = tid; i < vocab && have < 256; i += 256) {
            const unsigned key = df2_key16(*reinterpret_cast<const unsigned short*>(lg + i));
            if ((int)(key >> 8) == t) {
                const int p = atomicAdd(&sh_cnt, 1);
                if (p < 256) dst[p] = (unsigned)i;
            }
        }
    }
}

// Stage 2: exact bf16 re-rank. Each block recomputes logits[col][idx] as the fp32 dot of the
// ORIGINAL bf16 head row with the ORIGINAL hidden column (identical accumulation order to
// gemm_binv_b's row-dot: 256 threads x 20 elements, ascending serial tree — bitwise equal is NOT
// required vs gemm_binv_b, only "the exact bf16 head applied to the exact hidden"; ties then
// resolve exactly as the plain path would on THESE logits).
//   logits [vocab, 7] col-major; idx [7,256] u32; W head [vocab, hidden] bf16 row-major;
//   x base = hidden col-0 pointer (col j at x + j*hidden).
extern "C" __global__ void __launch_bounds__(256) df2_head_rerank_b(
    __nv_bfloat16* __restrict__ logits, const unsigned* __restrict__ idx,
    const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x,
    int vocab, int hidden, int cols) {
    const int slot = blockIdx.x & 255;
    const int col = blockIdx.x >> 8;
    if (col >= cols) return;
    const unsigned row = idx[(long long)col * 256 + slot];
    const __nv_bfloat16* w = W + (long long)row * hidden;
    const __nv_bfloat16* xv = x + (long long)col * hidden;
    const int tid = threadIdx.x;
    float acc = 0.f;
    for (int i = tid; i < hidden; i += 256)
        acc += __bfloat162float(w[i]) * __bfloat162float(xv[i]);
    __shared__ float red[256];
    red[tid] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) logits[(long long)col * vocab + row] = __float2bfloat16(red[0]);
}

extern "C" __global__ void __launch_bounds__(256) df2_sel_walk_b(
    unsigned* __restrict__ tokens, float* __restrict__ scores_out,
    const float* __restrict__ hp, const unsigned* __restrict__ cand,
    const float* __restrict__ unary,
    const __nv_bfloat16* __restrict__ pred_codebook, const __nv_bfloat16* __restrict__ succ_codebook,
    unsigned anchor, const unsigned* anchor_dev, int rank, int nlevels) {
    __shared__ float sh_sc[16];
    __shared__ float sh_warp[8];
    const int tid = threadIdx.x;
    const int R = rank;                        // 256
    unsigned prev = anchor_dev ? *anchor_dev : anchor;
    // DF2 block-16: nlevels is the chain length (block-1: 7 today, 15 at block 16). The
    // per-position work below is unchanged; only the trip count is a parameter.
    for (int p = 0; p < nlevels; ++p) {
        const float a = __bfloat162float(pred_codebook[(long long)prev * R + tid]) * hp[p * R + tid];
        for (int k = 0; k < 16; ++k) {
            const float prod = a * __bfloat162float(succ_codebook[(long long)cand[p * 16 + k] * R + tid]);
            float t = prod;
            #pragma unroll
            for (int off = 16; off; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
            if ((tid & 31) == 0) sh_warp[tid >> 5] = t;      // 8 warp leaders write
            __syncthreads();                                  // ALL threads (uniform)
            if (tid == 0) {
                float w = 0.f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) w += sh_warp[i];  // serial ascending cross-warp sum
                sh_sc[k] = unary[p * 16 + k] + w;
            }
            __syncthreads();                                  // ALL threads (uniform)
        }
        float best = sh_sc[0]; int bi = 0;
        #pragma unroll
        for (int k = 1; k < 16; ++k) if (sh_sc[k] > best) { best = sh_sc[k]; bi = k; }
        const unsigned tok = cand[p * 16 + bi];
        if (tid == 0) {
            tokens[p] = tok;
            for (int k = 0; k < 16; ++k) scores_out[p * 16 + k] = sh_sc[k];
        }
        prev = tok;
        __syncthreads();
    }
}

// ---- S5F2 sampled selector walk (df2_sel_walk_sample_b) ----------------------
// The S4F walk's score computation (identical fixed-order reduction), but each position DRAWS
// a multinomial sample over the 16 candidates at `temperature` instead of the argmax — the
// SGLang `CandidateSelector.sample_path` semantics (softmax(scores/T), `uniforms.ge(cumsum)`
// multinomial pick). Outputs are packed to keep the launch at the 12-arg cudarc cap:
//   out_tok[0..7]           the DRAWN draft tokens; out_tok[7 + 16p + k] the candidate ids
//   out_q[0..7]             the drawn candidate's softmax weight per position (the q in the
//                           real-q RS accept u*q < p); out_q[7 + 16p + k] the candidate
//                           weights (the exact relu(p - q) verify residual's q table).
// RNG: per-position seeds (host-derived from the step key, RNG_DOM_DF2_SEL); the draw is the
// same LCG + (sr>>8) uniform class as spec_verify_b. The draw is a scalar computed by thread 0
// and broadcast through shared memory (the walk's predecessor must be warp-uniform).
extern "C" __global__ void __launch_bounds__(256) df2_sel_walk_sample_b(
    unsigned* __restrict__ out_tok, float* __restrict__ out_q,
    const float* __restrict__ hp, const unsigned* __restrict__ cand,
    const float* __restrict__ unary,
    const __nv_bfloat16* __restrict__ pred_codebook, const __nv_bfloat16* __restrict__ succ_codebook,
    unsigned anchor, const unsigned* anchor_dev,
    const unsigned int* seeds, float temperature, int rank_nlevels) {
    // cudarc caps a launch at 12 arguments, so rank (8 bits) and nlevels (8 bits) ride packed.
    const int rank = rank_nlevels >> 8;
    const int nlevels = rank_nlevels & 0xFF;
    __shared__ float sh_sc[16];
    __shared__ float sh_warp[8];
    __shared__ unsigned sh_prev;
    const int tid = threadIdx.x;
    const int R = rank;                        // 256
    unsigned prev = anchor_dev ? *anchor_dev : anchor;
    float inv_t = (temperature > 0.f) ? (1.0f / temperature) : 1.0f;
    // DF2 block-16: nlevels = block-1 (7 today, 15 at block 16). The output layout keeps the
    // `[nlevels]` drawn-token/q prefix and the `[nlevels + 16p + k]` candidate tables, so the
    // host-side strides are nlevels-derived rather than the constant 7.
    for (int p = 0; p < nlevels; ++p) {
        const float a = __bfloat162float(pred_codebook[(long long)prev * R + tid]) * hp[p * R + tid];
        for (int k = 0; k < 16; ++k) {
            const float prod = a * __bfloat162float(succ_codebook[(long long)cand[p * 16 + k] * R + tid]);
            float t = prod;
            #pragma unroll
            for (int off = 16; off; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
            if ((tid & 31) == 0) sh_warp[tid >> 5] = t;      // 8 warp leaders write
            __syncthreads();                                  // ALL threads (uniform)
            if (tid == 0) {
                float w = 0.f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) w += sh_warp[i];  // serial ascending cross-warp sum
                sh_sc[k] = unary[p * 16 + k] + w;
            }
            __syncthreads();                                  // ALL threads (uniform)
        }
        if (tid == 0) {
            // softmax over the 16 candidates at `temperature`
            float mx = sh_sc[0];
            #pragma unroll
            for (int k = 1; k < 16; ++k) if (sh_sc[k] > mx) mx = sh_sc[k];
            float sum = 0.f;
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                float e = __expf((sh_sc[k] - mx) * inv_t);
                out_q[nlevels + p * 16 + k] = e;                    // unnormalized scratch
                sum += e;
            }
            // P3(b) fix: normalize ALL 16 entries BEFORE the draw. The prior code normalized
            // in-place inside the multinomial loop and `break`ed on the chosen candidate, leaving
            // the tail entries (k > chosen) at their RAW exp values — the candidate q table fed to
            // spec_verify_realq_b's exact relu(p-q) residual was then a MIX of normalized and raw
            // weights (qsum != 1), biasing the reject-side distribution. The draw itself is
            // bit-identical (same cumsum, same ru, same chosen); only the table is corrected.
            #pragma unroll
            for (int k = 0; k < 16; ++k) out_q[nlevels + p * 16 + k] /= sum;
            // multinomial draw: first candidate whose cumulative mass strictly exceeds u
            // (the vendor's `uniforms.ge(cumsum)` pick). q_rows = the drawn candidate's weight.
            unsigned int sr = seeds[p];
            sr = sr * 1664525u + 1013904223u;
            float ru = (sr >> 8) * (1.0f / 16777216.0f);
            float cum = 0.f; int chosen = 15;
            for (int k = 0; k < 16; ++k) {
                cum += out_q[nlevels + p * 16 + k];
                if (ru < cum) { chosen = k; break; }
            }
            const unsigned tok = cand[p * 16 + chosen];
            out_tok[p] = tok;
            out_q[p] = out_q[nlevels + p * 16 + chosen];
            for (int k = 0; k < 16; ++k) out_tok[nlevels + p * 16 + k] = cand[p * 16 + k];
            sh_prev = tok;
        }
        __syncthreads();
        prev = sh_prev;
        __syncthreads();
    }
}
