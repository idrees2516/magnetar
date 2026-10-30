//! The Mikros kernel layer — runtime-dispatched AVX-512 / GFNI kernels for
//! the prover's hot loops (the paper's own host optimizations: "GFNI
//! provides F28 multiplication, VPCLMULQDQ provides carry-less
//! multiplication for extension-field elements, and AVX-512 vectorizes the
//! bitwise operations, transposes, and pack/unpack steps", §6).
//!
//! Every kernel is **bit-identical** to its scalar path (XOR/AND
//! commutativity or exact field arithmetic — no rounding anywhere), so the
//! Fiat–Shamir transcripts stay deterministic regardless of dispatch; the
//! cross-validation tests pin this.
//!
//! | Kernel | Replaces | Backend |
//! |---|---|---|
//! | [`xor_into`] / [`xor_into_e`] | the per-`c` selector plane XORs (`encode_k`, `encode_e_words`, the η layer encodings) | AVX-512 `VPXORQ` (8 words/op), AVX2 fallback, portable tail |
//! | [`PlaneSet::parity_dot`](crate::planes::PlaneSet::parity_dot) | the `d`-fold and `u`-claim parity dots | AVX-512 `VPOPCNTQ` accumulate (avx512vpopcntdq) or the vector XOR-fold |
//! | [`dot3_e`] | the cubic sum-check's eight triple-dots | `magnetar_field::avx::mulvv` (4-lane VPCLMULQDQ) + the XOR fold |
//! | [`sample_fold_products`] | the 65,536-entry `(a,b)`-bucket pass + 64K-element E-multiply fold of the Hadamard η pipeline | GFNI `VPGF2P8MULB` (64 byte-products/op) into a 256-entry **product** bucket (L1-resident) + a 256-term E-fold |
//!
//! The product-bucket interchange: `Σ_i l_i·(a_i·b_i)` over GF(2^8) with
//! `a_i·b_i` computed by GFNI regroups the aggregation by the *product*
//! byte `c = a_i·b_i`: `B[c] = ⊕_{i: a_i·b_i = c} l_i`, `fold =
//! Σ_c embed(c)·B[c]`. This is algebraically identical to the `(a,b)`-keyed
//! 64K table (`Σ_key T16[key]·bucket[key]`), but the random-access working
//! set drops from 1 MiB (L2) to 4 KiB (L1) and the E-multiply count from
//! ≤65,536 to 256 per node — the ~250× E-multiply reduction that makes the
//! f-sampling pipeline GFNI-bound instead of fold-bound.

use crate::gf8::Gf8;
use magnetar_field::Gf128;
use magnetar_field::{mul, ZERO};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Feature detection
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SimdLevel {
    /// AVX-512 (F + BW + VL) — 512-bit integer ops.
    Avx512,
    /// AVX2 — 256-bit integer ops.
    Avx2,
    /// Portable (compiler-default auto-vectorization).
    Scalar,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GfniLevel {
    /// GFNI + AVX-512 — `VPGF2P8MULB` on 64 bytes per instruction.
    Avx512,
    /// No GFNI — the scalar table route.
    None,
}

fn simd_level() -> SimdLevel {
    static LEVEL: OnceLock<SimdLevel> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        if let Ok(v) = std::env::var("MAGNETAR_MIKROS_SIMD") {
            return match v.as_str() {
                "avx512" => SimdLevel::Avx512,
                "avx2" => SimdLevel::Avx2,
                "scalar" => SimdLevel::Scalar,
                _ => SimdLevel::Scalar,
            };
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vl")
            {
                return SimdLevel::Avx512;
            }
            if std::is_x86_feature_detected!("avx2") {
                return SimdLevel::Avx2;
            }
        }
        SimdLevel::Scalar
    })
}

fn gfni_level() -> GfniLevel {
    static LEVEL: OnceLock<GfniLevel> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if simd_level() == SimdLevel::Avx512
                && std::is_x86_feature_detected!("gfni")
                && std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
            {
                return GfniLevel::Avx512;
            }
        }
        GfniLevel::None
    })
}

/// Whether the vectorized VPOPCNTQ parity route is available.
fn vpopcntq() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            simd_level() == SimdLevel::Avx512 && std::is_x86_feature_detected!("avx512vpopcntdq")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

// ---------------------------------------------------------------------------
// Raw-intrinsic bodies (target_feature-gated, the repo's dispatch pattern)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod arch {
    use super::Gf128;

    /// `dst ^= src` over u64 words, 512-bit lanes.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn xor_into_512(dst: *mut u64, src: *const u64, n8: usize) {
        // SAFETY: the caller guarantees 8·n8 words valid on both sides.
        unsafe {
            use std::arch::x86_64::{_mm512_loadu_si512, _mm512_storeu_si512, _mm512_xor_si512};
            for k in 0..n8 {
                let a = _mm512_loadu_si512(dst.add(8 * k).cast());
                let b = _mm512_loadu_si512(src.add(8 * k).cast());
                _mm512_storeu_si512(dst.add(8 * k).cast(), _mm512_xor_si512(a, b));
            }
        }
    }

    /// `dst ^= src` over u64 words, 256-bit lanes.
    #[target_feature(enable = "avx2")]
    pub unsafe fn xor_into_256(dst: *mut u64, src: *const u64, n4: usize) {
        // SAFETY: the caller guarantees 4·n4 words valid on both sides.
        unsafe {
            use std::arch::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256, _mm256_xor_si256};
            for k in 0..n4 {
                let a = _mm256_loadu_si256(dst.add(4 * k).cast());
                let b = _mm256_loadu_si256(src.add(4 * k).cast());
                _mm256_storeu_si256(dst.add(4 * k).cast(), _mm256_xor_si256(a, b));
            }
        }
    }

    /// `⊕_i a_i·b_i` — the VPOPCNTQ popcount-accumulate route.
    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    pub unsafe fn parity_dot_popcnt(a: *const u64, b: *const u64, n: usize) -> u64 {
        // SAFETY: the caller guarantees n words valid on both sides.
        unsafe {
            use std::arch::x86_64::{
                _mm512_add_epi64, _mm512_and_si512, _mm512_loadu_si512, _mm512_popcnt_epi64,
                _mm512_setzero_si512, _mm512_storeu_si512,
            };
            let mut acc = _mm512_setzero_si512();
            let mut k = 0;
            while n - k >= 8 {
                let x = _mm512_loadu_si512(a.add(k).cast());
                let y = _mm512_loadu_si512(b.add(k).cast());
                acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_and_si512(x, y)));
                k += 8;
            }
            let mut sums = [0u64; 8];
            _mm512_storeu_si512(sums.as_mut_ptr().cast(), acc);
            let mut total: u64 = sums.iter().sum();
            for i in k..n {
                total += (*a.add(i) & *b.add(i)).count_ones() as u64;
            }
            total & 1
        }
    }

    /// `⊕_i a_i·b_i` — the vector XOR-fold route (no VPOPCNTDQ).
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn parity_dot_xorfold(a: *const u64, b: *const u64, n: usize) -> u64 {
        // SAFETY: the caller guarantees n words valid on both sides.
        unsafe {
            use std::arch::x86_64::{
                _mm512_and_si512, _mm512_loadu_si512, _mm512_setzero_si512, _mm512_storeu_si512,
                _mm512_xor_si512,
            };
            let mut acc = _mm512_setzero_si512();
            let mut k = 0;
            while n - k >= 8 {
                let x = _mm512_loadu_si512(a.add(k).cast());
                let y = _mm512_loadu_si512(b.add(k).cast());
                acc = _mm512_xor_si512(acc, _mm512_and_si512(x, y));
                k += 8;
            }
            let mut words = [0u64; 8];
            _mm512_storeu_si512(words.as_mut_ptr().cast(), acc);
            let mut fold = 0u64;
            for w in words {
                fold ^= w;
            }
            for i in k..n {
                fold ^= *a.add(i) & *b.add(i);
            }
            fold.count_ones() as u64 & 1
        }
    }

    /// The GFNI product pass: per 64-row block, byte-spread both encodings,
    /// multiply with `VPGF2P8MULB`, and XOR the row weights into the
    /// 256-entry product bucket.
    #[target_feature(enable = "avx512f,avx512bw,gfni")]
    pub unsafe fn gfni_product_bucket(
        enc_a: &[Vec<u64>; 8],
        enc_b: &[Vec<u64>; 8],
        l: &[Gf128],
        bucket: &mut [Gf128; 256],
    ) {
        // SAFETY: the caller guarantees 8 planes of equal word counts,
        // matching weights, and a 256-entry bucket.
        unsafe {
            use std::arch::x86_64::{
                _mm512_gf2p8mul_epi8, _mm512_loadu_si512, _mm512_storeu_si512,
            };
            let wp = enc_a[0].len();
            for w in 0..wp {
                let wa: [u64; 8] = std::array::from_fn(|c| enc_a[c][w]);
                let wb: [u64; 8] = std::array::from_fn(|c| enc_b[c][w]);
                let ba = crate::planes::spread8(&wa);
                let bb = crate::planes::spread8(&wb);
                let va = _mm512_loadu_si512(ba.as_ptr().cast());
                let vb = _mm512_loadu_si512(bb.as_ptr().cast());
                let prod = _mm512_gf2p8mul_epi8(va, vb);
                let mut pb = [0u8; 64];
                _mm512_storeu_si512(pb.as_mut_ptr().cast(), prod);
                let base_row = w * 64;
                for rr in 0..64 {
                    let c = pb[rr] as usize;
                    bucket[c] += l[base_row + rr];
                }
            }
        }
    }

    /// The 64×64 bit-matrix transpose, AVX-512 form — the descending
    /// block-swap network (j = 32, 16, …, 1): each stage exchanges the
    /// anti-diagonal j×j blocks between the row pairs (i, i^j), exactly the
    /// recursion `planes::flip64` performs, flattened:
    ///
    /// * `j ≥ 8`: the pairs live in register-pairs `d = j/8` apart — the
    ///   aligned-lane masked exchange `X ^= (X ^ (Y << j)) & hi`,
    ///   `Y ^= (Y ^ (X >> j)) & lo` (both computed from the originals).
    /// * `j < 8`: the pairs are lane-mirrors inside each register — one
    ///   `VPERMQ` + a per-lane role blend (bit-j-clear lanes take the
    ///   `<< j / hi-mask` role, bit-j-set lanes the `>> j / lo-mask` role).
    ///
    /// Bit-identical to `planes::flip64` (cross-validated in tests).
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub unsafe fn flip64_512(a: &mut [u64; 64]) {
        // SAFETY: the slice has exactly 64 words; all loads/stores are
        // in-bounds through the __m512i view of the array.
        unsafe {
            use std::arch::x86_64::{
                _knot_mask8, _mm512_and_si512, _mm512_loadu_si512, _mm512_mask_mov_epi64,
                _mm512_permutexvar_epi64, _mm512_set1_epi64, _mm512_set_epi64, _mm512_slli_epi64,
                _mm512_srli_epi64, _mm512_storeu_si512, _mm512_xor_si512,
            };
            let mut regs: [std::arch::x86_64::__m512i; 8] =
                std::array::from_fn(|r| _mm512_loadu_si512(a[r * 8..].as_ptr().cast()));
            let alternating = |j: u64| -> u64 {
                // j ones every 2j bits, starting at bit 0
                match j {
                    1 => 0x5555_5555_5555_5555,
                    2 => 0x3333_3333_3333_3333,
                    4 => 0x0F0F_0F0F_0F0F_0F0F,
                    8 => 0x00FF_00FF_00FF_00FF,
                    16 => 0x0000_FFFF_0000_FFFF,
                    32 => 0x0000_0000_FFFF_FFFF,
                    _ => unreachable!("stage width"),
                }
            };
            // the cross-register stages (j ≥ 8): register pairs d = j/8
            // apart, aligned lanes.
            macro_rules! cross_stage {
                ($j:expr) => {{
                    const J: u32 = $j;
                    let lo = _mm512_set1_epi64(alternating(J as u64) as i64);
                    let hi = _mm512_set1_epi64((alternating(J as u64) << J) as i64);
                    let d = (J / 8) as usize;
                    let mut base = 0;
                    while base < 8 {
                        for k in 0..d {
                            let ra = base + k;
                            let rb = base + k + d;
                            let x = regs[ra];
                            let y = regs[rb];
                            // t1 = (x ^ (y << j)) & hi ; t2 = (y ^ (x >> j)) & lo
                            let t1 =
                                _mm512_and_si512(_mm512_xor_si512(x, _mm512_slli_epi64(y, J)), hi);
                            let t2 =
                                _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64(x, J)), lo);
                            regs[ra] = _mm512_xor_si512(x, t1);
                            regs[rb] = _mm512_xor_si512(y, t2);
                        }
                        base += 2 * d;
                    }
                }};
            }
            cross_stage!(32);
            cross_stage!(16);
            cross_stage!(8);
            // the in-register stages (j < 8): lane mirrors via VPERMQ + the
            // per-lane role blend
            macro_rules! lane_stage {
                ($j:expr) => {{
                    const J: u32 = $j;
                    const JU: u64 = $j;
                    let lo = _mm512_set1_epi64(alternating(JU) as i64);
                    let hi = _mm512_set1_epi64((alternating(JU) << J) as i64);
                    let idx = _mm512_set_epi64(
                        (7 ^ JU) as i64,
                        (6 ^ JU) as i64,
                        (5 ^ JU) as i64,
                        (4 ^ JU) as i64,
                        (3 ^ JU) as i64,
                        (2 ^ JU) as i64,
                        (1 ^ JU) as i64,
                        (0 ^ JU) as i64,
                    );
                    // bit-j-clear lanes take the << / hi role
                    const K_HI: u8 = {
                        let mut m = 0u8;
                        let mut p = 0u8;
                        while p < 8 {
                            if p & ($j as u8) == 0 {
                                m |= 1 << p;
                            }
                            p += 1;
                        }
                        m
                    };
                    let k_hi: std::arch::x86_64::__mmask8 = K_HI;
                    for r in 0..8 {
                        let x = regs[r];
                        let y = _mm512_permutexvar_epi64(idx, x);
                        // role A (bit-j-clear lanes): x ^ ((x ^ (y << j)) & hi)
                        // role B (bit-j-set lanes):   x ^ ((x ^ (y >> j)) & lo)
                        let t_hi =
                            _mm512_and_si512(_mm512_xor_si512(x, _mm512_slli_epi64(y, J)), hi);
                        let t_lo =
                            _mm512_and_si512(_mm512_xor_si512(x, _mm512_srli_epi64(y, J)), lo);
                        let t = _mm512_mask_mov_epi64(t_hi, _knot_mask8(k_hi), t_lo);
                        regs[r] = _mm512_xor_si512(x, t);
                    }
                }};
            }
            lane_stage!(4);
            lane_stage!(2);
            lane_stage!(1);
            for r in 0..8 {
                _mm512_storeu_si512(a[r * 8..].as_mut_ptr().cast(), regs[r]);
            }
        }
    }

    /// The scalar iterative form of the same network (the non-x86_64 and
    /// pre-AVX-512 path; also the cross-validation reference).
    pub fn flip64_iter(a: &mut [u64; 64]) {
        for &j in &[32u64, 16, 8, 4, 2, 1] {
            let lo = match j {
                1 => 0x5555_5555_5555_5555,
                2 => 0x3333_3333_3333_3333,
                4 => 0x0F0F_0F0F_0F0F_0F0F,
                8 => 0x00FF_00FF_00FF_00FF,
                16 => 0x0000_FFFF_0000_FFFF,
                32 => 0x0000_0000_FFFF_FFFF,
                _ => unreachable!("stage width"),
            };
            let hi = lo << j;
            let js = j as usize;
            for i in 0..64usize {
                if i & js == 0 {
                    let x = a[i];
                    let y = a[i | js];
                    let t1 = (x ^ (y << j)) & hi;
                    let t2 = (y ^ (x >> j)) & lo;
                    a[i] = x ^ t1;
                    a[i | js] = y ^ t2;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Word kernels (u64 planes)
// ---------------------------------------------------------------------------

/// The 64×64 bit-matrix transpose — the AVX-512 block-swap network, with
/// the iterative scalar network as the portable path. Bit-identical to
/// `planes::flip64` (the recursive reference; cross-validated in tests).
pub fn flip64_v(a: &mut [u64; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_level() == SimdLevel::Avx512 {
            #[allow(unsafe_code)]
            // SAFETY: the dispatch confirmed avx512f+bw+vl; the array has
            // exactly 64 words.
            unsafe {
                arch::flip64_512(a);
            }
            return;
        }
        arch::flip64_iter(a);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        crate::planes::flip64(a);
    }
}

/// `dst ^= src` over u64 words — the bit-plane XOR primitive.
pub fn xor_into(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len().min(src.len());
    match simd_level() {
        SimdLevel::Avx512 => {
            let n8 = n / 8;
            #[cfg(target_arch = "x86_64")]
            // SAFETY: the measured dispatch confirmed avx512f+bw; 8·n8 ≤ n
            // words on both slices.
            unsafe {
                arch::xor_into_512(dst.as_mut_ptr(), src.as_ptr(), n8);
            }
            for k in 8 * (n / 8)..n {
                dst[k] ^= src[k];
            }
        }
        SimdLevel::Avx2 => {
            let n4 = n / 4;
            #[cfg(target_arch = "x86_64")]
            // SAFETY: the dispatch confirmed avx2; 4·n4 ≤ n on both slices.
            unsafe {
                arch::xor_into_256(dst.as_mut_ptr(), src.as_ptr(), n4);
            }
            for k in 4 * (n / 4)..n {
                dst[k] ^= src[k];
            }
        }
        SimdLevel::Scalar => {
            for k in 0..n {
                dst[k] ^= src[k];
            }
        }
    }
}

/// `dst ^= src` over E-words (u128 XOR — the same primitive at 2× width).
pub fn xor_into_e(dst: &mut [Gf128], src: &[Gf128]) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len().min(src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if simd_level() != SimdLevel::Scalar {
            // SAFETY: [Gf128] and [u64; 2] have identical layout; both
            // slices are valid for 2·n words and disjoint.
            #[allow(unsafe_code)]
            unsafe {
                let d: &mut [u64] =
                    std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u64>(), 2 * n);
                let s: &[u64] = std::slice::from_raw_parts(src.as_ptr().cast::<u64>(), 2 * n);
                xor_into(&mut d[..2 * n], &s[..2 * n]);
            }
            return;
        }
    }
    for k in 0..n {
        dst[k] += src[k];
    }
}

/// The XOR reduction of an E-word run.
pub fn xor_fold_e(v: &[Gf128]) -> Gf128 {
    // Pair the halves: the reduction is a plain XOR tree — the scalar fold
    // is already auto-vectorizable, and correctness (not lanes) matters
    // here; the hot paths reduce through dot3_e's laned fold instead.
    let mut acc = ZERO;
    for &x in v {
        acc += x;
    }
    acc
}

/// The parity inner product of two bit-planes: `⊕_i a[i]·b[i]` — the `d`-fold
/// and `u`-claim kernel. VPOPCNTQ route: accumulate `Σ popcount(a&b)` and
/// take the low bit (the XOR of parities); else the vector XOR-fold.
pub fn parity_dot_v(a: &[u64], b: &[u64]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if vpopcntq() {
            // SAFETY: the dispatch confirmed the features; n words valid on
            // both slices.
            #[allow(unsafe_code)]
            unsafe {
                return arch::parity_dot_popcnt(a.as_ptr(), b.as_ptr(), n);
            }
        }
        if simd_level() == SimdLevel::Avx512 {
            // SAFETY: the dispatch confirmed avx512f+bw; n words valid.
            #[allow(unsafe_code)]
            unsafe {
                return arch::parity_dot_xorfold(a.as_ptr(), b.as_ptr(), n);
            }
        }
    }
    let mut acc = 0u64;
    for k in 0..n {
        acc ^= a[k] & b[k];
    }
    acc.count_ones() as u64 & 1
}

// ---------------------------------------------------------------------------
// E-vector kernels (through magnetar-field's VPCLMULQDQ lanes)
// ---------------------------------------------------------------------------

/// `〈x, y∘z〉` — the cubic sum-check's inner product, VPCLMULQDQ-laned.
/// `s1`/`s2` are caller-hoisted scratch (resized to the input length).
pub fn dot3_e(
    s1: &mut Vec<Gf128>,
    s2: &mut Vec<Gf128>,
    x: &[Gf128],
    y: &[Gf128],
    z: &[Gf128],
) -> Gf128 {
    let n = x.len();
    debug_assert_eq!(y.len(), n);
    debug_assert_eq!(z.len(), n);
    if s1.len() < n {
        s1.resize(n, ZERO);
    }
    if s2.len() < n {
        s2.resize(n, ZERO);
    }
    let (s1n, s2n) = (&mut s1[..n], &mut s2[..n]);
    let done = magnetar_field::avx::mulvv(s1n, y, z, n);
    for k in done..n {
        s1n[k] = mul(y[k], z[k]);
    }
    let done = magnetar_field::avx::mulvv(s2n, x, s1n, n);
    for k in done..n {
        s2n[k] = mul(x[k], s1n[k]);
    }
    xor_fold_e(s2n)
}

/// `〈x, y〉` over E — `mulvv` + the XOR fold (`s` is caller scratch).
pub fn dot_e(s: &mut Vec<Gf128>, x: &[Gf128], y: &[Gf128]) -> Gf128 {
    let n = x.len();
    debug_assert_eq!(y.len(), n);
    if s.len() < n {
        s.resize(n, ZERO);
    }
    let sn = &mut s[..n];
    let done = magnetar_field::avx::mulvv(sn, x, y, n);
    for k in done..n {
        sn[k] = mul(x[k], y[k]);
    }
    xor_fold_e(sn)
}

/// `x[i] *= y[i]` in place — the elementwise product of two E-runs through
/// the VPCLMULQDQ lanes (a 64-element stack staging buffer keeps the
/// kernel's load-before-store discipline without a heap clone).
pub fn mul_assign_e(x: &mut [Gf128], y: &[Gf128]) {
    debug_assert_eq!(x.len(), y.len());
    const CH: usize = 64;
    let mut tmp = [ZERO; CH];
    let n = x.len();
    let mut start = 0;
    while start < n {
        let len = CH.min(n - start);
        let (xs, ys) = (&mut x[start..start + len], &y[start..start + len]);
        tmp[..len].copy_from_slice(xs);
        let done = magnetar_field::avx::mulvv(xs, &tmp[..len], ys, len);
        for i in done..len {
            xs[i] = mul(xs[i], ys[i]);
        }
        start += len;
    }
}

// ---------------------------------------------------------------------------
// The GFNI product-bucket fold (the Hadamard η pipeline)
// ---------------------------------------------------------------------------

/// `Σ_i l_i · (a_i · b_i)` over GF(2^8), where `a_i` / `b_i` are the
/// byte-spread encodings (8 bit-planes of `m` rows) and `l_i ∈ E`. The
/// GFNI route multiplies the byte pairs with `VPGF2P8MULB`, buckets the
/// row weights by the 256 possible products (L1-resident) and folds with
/// 256 E-multiplies. Returns `None` when GFNI is unavailable (the caller
/// runs the scalar 64K-bucket route — byte-identical by XOR commutativity).
pub fn sample_fold_products(
    f8: &Gf8,
    enc_a: &[Vec<u64>; 8],
    enc_b: &[Vec<u64>; 8],
    l: &[Gf128],
) -> Option<Gf128> {
    if gfni_level() != GfniLevel::Avx512 {
        return None;
    }
    let wp = enc_a[0].len();
    debug_assert_eq!(enc_b[0].len(), wp);
    debug_assert_eq!(l.len(), wp * 64);
    let mut bucket = [ZERO; 256];
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the dispatch confirmed GFNI + AVX-512; the plane vectors and
    // weights are index-consistent (asserted by the caller's debug checks).
    unsafe {
        arch::gfni_product_bucket(enc_a, enc_b, l, &mut bucket);
    }
    // fold: Σ_c embed(c)·B[c] — 256 E-multiplies through the VPCLMULQDQ
    // lanes (embed8 is a [Gf128; 256] table).
    let embeds: Vec<Gf128> = (0..256).map(|c| f8.embed(c as u8)).collect();
    let mut scratch = vec![ZERO; 256];
    let mut acc = ZERO;
    let done = magnetar_field::avx::mulvv(&mut scratch, &embeds, &bucket, 256);
    for c in done..256 {
        acc += mul(f8.embed(c as u8), bucket[c]);
    }
    Some(acc + xor_fold_e(&scratch[..done]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planes::spread8;

    thread_local! {
        static RNG: std::cell::Cell<u64> = const { std::cell::Cell::new(0x51ed_270b_1a7f_bee5) };
    }

    fn word() -> u64 {
        RNG.with(|c| {
            let mut r = c.get();
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            c.set(r);
            r
        })
    }

    fn gf() -> Gf128 {
        Gf128(((word() as u128) << 64) | word() as u128)
    }

    #[test]
    fn xor_kernels_match_scalar() {
        for &n in &[1usize, 7, 8, 31, 64, 65, 200] {
            let a: Vec<u64> = (0..n).map(|_| word()).collect();
            let b: Vec<u64> = (0..n).map(|_| word()).collect();
            let mut want = a.clone();
            for k in 0..n {
                want[k] ^= b[k];
            }
            let mut got = a.clone();
            xor_into(&mut got, &b);
            assert_eq!(got, want, "xor_into n={n}");
            // the E-word form (2n u64 lanes)
            let ea: Vec<Gf128> = (0..n).map(|_| gf()).collect();
            let eb: Vec<Gf128> = (0..n).map(|_| gf()).collect();
            let mut ewant = ea.clone();
            for k in 0..n {
                ewant[k] += eb[k];
            }
            let mut egot = ea.clone();
            xor_into_e(&mut egot, &eb);
            assert_eq!(egot, ewant, "xor_into_e n={n}");
            assert_eq!(
                xor_fold_e(&ewant),
                ewant.iter().fold(ZERO, |acc, x| acc + *x)
            );
        }
    }

    #[test]
    fn parity_dot_matches_scalar() {
        for &n in &[1usize, 7, 8, 31, 64, 65, 200, 1024] {
            let a: Vec<u64> = (0..n).map(|_| word()).collect();
            let b: Vec<u64> = (0..n).map(|_| word()).collect();
            let mut acc = 0u64;
            for k in 0..n {
                acc ^= a[k] & b[k];
            }
            let want = acc.count_ones() as u64 & 1;
            assert_eq!(parity_dot_v(&a, &b), want, "parity n={n}");
            // agreement with the PlaneSet method (the production caller)
            assert_eq!(crate::planes::PlaneSet::parity_dot(&a, &b), want);
        }
    }

    #[test]
    fn dot_kernels_match_scalar() {
        for &n in &[3usize, 8, 17, 64, 100, 257] {
            let x: Vec<Gf128> = (0..n).map(|_| gf()).collect();
            let y: Vec<Gf128> = (0..n).map(|_| gf()).collect();
            let z: Vec<Gf128> = (0..n).map(|_| gf()).collect();
            let mut want = ZERO;
            let mut want2 = ZERO;
            for k in 0..n {
                want += mul(x[k], mul(y[k], z[k]));
                want2 += mul(x[k], y[k]);
            }
            let mut s1 = Vec::new();
            let mut s2 = Vec::new();
            assert_eq!(dot3_e(&mut s1, &mut s2, &x, &y, &z), want, "dot3 n={n}");
            let mut s = Vec::new();
            assert_eq!(dot_e(&mut s, &x, &y), want2, "dot n={n}");
            // the in-place elementwise product
            let mut ip = x.clone();
            mul_assign_e(&mut ip, &y);
            for k in 0..n {
                assert_eq!(ip[k], mul(x[k], y[k]), "mul_assign n={n} k={k}");
            }
        }
    }

    #[test]
    fn flip64_network_matches_recursive_reference() {
        let mut r = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            r
        };
        for _ in 0..16 {
            let a: [u64; 64] = std::array::from_fn(|_| next());
            let mut reference = a;
            crate::planes::flip64(&mut reference); // the recursive reference
            let mut iter_form = a;
            #[cfg(target_arch = "x86_64")]
            arch::flip64_iter(&mut iter_form);
            #[cfg(not(target_arch = "x86_64"))]
            crate::planes::flip64(&mut iter_form);
            assert_eq!(iter_form, reference, "the iterative network");
            let mut vect = a;
            flip64_v(&mut vect);
            assert_eq!(vect, reference, "the AVX-512 network");
            // transpose twice = identity
            flip64_v(&mut vect);
            assert_eq!(vect, a);
        }
    }

    #[test]
    fn gfni_fold_matches_bucket_route() {
        // the (a,b)-bucket reference (the scalar route it replaces)
        let f8 = crate::gf8::Gf8::new();
        for &log_m in &[6usize, 7, 8] {
            let m = 1usize << log_m;
            let wp = m / 64;
            let enc_a: [Vec<u64>; 8] = std::array::from_fn(|_| (0..wp).map(|_| word()).collect());
            let enc_b: [Vec<u64>; 8] = std::array::from_fn(|_| (0..wp).map(|_| word()).collect());
            let l: Vec<Gf128> = (0..m).map(|_| gf()).collect();
            // scalar reference: the 64K (a,b) bucket + fold
            let mut bucket: Vec<Gf128> = vec![ZERO; 65536];
            for w in 0..wp {
                let wa: [u64; 8] = std::array::from_fn(|c| enc_a[c][w]);
                let wb: [u64; 8] = std::array::from_fn(|c| enc_b[c][w]);
                let ba = spread8(&wa);
                let bb = spread8(&wb);
                let base_row = w * 64;
                for rr in 0..64 {
                    let key = ((ba[rr] as usize) << 8) | bb[rr] as usize;
                    bucket[key] += l[base_row + rr];
                }
            }
            let mut want = ZERO;
            for key in 0..65536usize {
                if bucket[key] != ZERO {
                    want += mul(bucket[key], f8.embed(f8.mul_tab[key]));
                }
            }
            match sample_fold_products(&f8, &enc_a, &enc_b, &l) {
                Some(got) => assert_eq!(got, want, "gfni fold m={m}"),
                None => eprintln!("GFNI unavailable — skipping the dispatch check"),
            }
        }
    }
}
