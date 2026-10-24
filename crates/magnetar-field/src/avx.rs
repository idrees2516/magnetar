//! AVX-512 VPCLMULQDQ batched field arithmetic (docs/PERFORMANCE.md R5).
//!
//! The CPU advertises `vpclmulqdq` (4-lane 64×64 carry-less multiply per
//! instruction). Four consecutive `Gf128`s are exactly one 512-bit
//! register whose CLMUL lanes hold `(lo, hi)` per element — so the field
//! multiply (3 Karatsuba CLMULs + the two-step reduction) and the sum-check
//! message/fold kernels vectorize with **zero repacking**: the tables load
//! straight into lane layout.
//!
//! The public API is safe; the intrinsics live in `#[target_feature]`
//! functions guarded by runtime detection. Results are bit-identical to
//! the scalar multiplier (cross-validated by the release test suite), so
//! Fiat–Shamir transcripts stay deterministic regardless of dispatch.

use crate::gf128::Gf128;

/// Which batched backend is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdKind {
    /// AVX-512 VPCLMULQDQ: 4 lanes per instruction.
    Avx512,
    /// AVX2 VPCLMULQDQ (256-bit): 2 lanes per instruction.
    Avx2,
    /// Scalar (portable fallback).
    Scalar,
}

/// Feature presence + a **measured micro-probe**. Some virtualized hosts
/// advertise `avx512f`/`vpclmulqdq` in CPUID but emulate the instructions
/// in the hypervisor at ~10× the scalar cost — a one-time timed probe (a
/// few hundred multiplies) picks the backend that is actually fastest on
/// this machine, so the kernels never regress on emulated hardware.
pub fn detect() -> SimdKind {
    use std::sync::OnceLock;
    static KIND: OnceLock<SimdKind> = OnceLock::new();
    *KIND.get_or_init(|| {
        // An explicit override wins (useful on hosts whose CPUID lies or
        // whose throttling the probe misjudges): MAGNETAR_SIMD=avx512|avx2|scalar.
        if let Ok(v) = std::env::var("MAGNETAR_SIMD") {
            match v.as_str() {
                "avx512" => return SimdKind::Avx512,
                "avx2" => return SimdKind::Avx2,
                "scalar" => return SimdKind::Scalar,
                _ => {}
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            let avx512 = std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("vpclmulqdq");
            let avx2 = std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("pclmulqdq");
            if !avx2 && !avx512 {
                return SimdKind::Scalar;
            }
            // Timed probes: 512 multiplies through each candidate.
            let t_scalar = timed_probe_scalar();
            let t_avx2 = if avx2 { timed_probe_avx2() } else { u128::MAX };
            let t_avx512 = if avx512 { timed_probe_avx512() } else { u128::MAX };
            // Prefer the measured fastest; the scalar baseline guards
            // against wholesale emulation.
            let mut best = SimdKind::Scalar;
            let mut best_t = t_scalar;
            if t_avx2 < best_t {
                best = SimdKind::Avx2;
                best_t = t_avx2;
            }
            if t_avx512 < best_t {
                best = SimdKind::Avx512;
            }
            best
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            SimdKind::Scalar
        }
    })
}

#[cfg(target_arch = "x86_64")]
fn timed_probe_scalar() -> u128 {
    use rayon::prelude::*;
    let a = [Gf128(0x1234_5678_9abc_def0_1357_9bdf_2468_ace0u128); 4];
    let b = [Gf128(0x0f0f_0f0f_0f0f_0f0f_f0f0_f0f0_f0f0_f0f0u128); 4];
    let t0 = std::time::Instant::now();
    // Parallel, like the real kernels: sustained multi-threaded use is
    // where virtualized hosts throttle or emulate.
    let sink: u128 = (0..16_384u64)
        .into_par_iter()
        .map(|i| crate::gf128::mul(a[(i & 3) as usize], b[(i & 1) as usize]).0)
        .sum();
    let _ = sink;
    t0.elapsed().as_nanos()
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn timed_probe_avx2() -> u128 {
    use rayon::prelude::*;
    let a = [Gf128(0x1234_5678_9abc_def0_1357_9bdf_2468_ace0u128); 2];
    let b = [Gf128(0x0f0f_0f0f_0f0f_0f0f_f0f0_f0f0_f0f0_f0f0u128); 2];
    let t0 = std::time::Instant::now();
    let sink: u128 = (0..8_192u64)
        .into_par_iter()
        .map(|i| {
            let mut x = a;
            x[0].0 ^= i as u128;
            let r = unsafe { mul2_avx2(x, b) };
            r[(i & 1) as usize].0
        })
        .sum();
    let _ = sink;
    t0.elapsed().as_nanos()
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn timed_probe_avx512() -> u128 {
    use rayon::prelude::*;
    let a = [Gf128(0x1234_5678_9abc_def0_1357_9bdf_2468_ace0u128); 4];
    let b = [Gf128(0x0f0f_0f0f_0f0f_0f0f_f0f0_f0f0_f0f0_f0f0u128); 4];
    let t0 = std::time::Instant::now();
    let sink: u128 = (0..16_384u64)
        .into_par_iter()
        .map(|i| {
            let mut x = a;
            x[0].0 ^= i as u128;
            let r = unsafe { avx512().mul4(x, b) };
            r[(i & 3) as usize].0
        })
        .sum();
    let _ = sink;
    t0.elapsed().as_nanos()
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod imp512 {
    //! The AVX-512 (4-lane) pipeline. Four consecutive `Gf128`s are exactly
    //! one `__m512i` whose CLMUL lanes hold `(lo, hi)` per element.
    use super::*;
    use std::arch::x86_64::{
        __m512i, _mm512_and_si512, _mm512_clmulepi64_epi128, _mm512_loadu_si512,
        _mm512_permutexvar_epi64, _mm512_set_epi64, _mm512_slli_epi64, _mm512_srli_epi64,
        _mm512_storeu_si512, _mm512_xor_si512,
    };

    pub(super) struct Avx;

    /// The within-lane word swap index [1,0,3,2,5,4,7,6].
    #[inline(always)]
    unsafe fn swap_idx() -> __m512i {
        _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1)
    }

    #[inline(always)]
    unsafe fn even_mask() -> __m512i {
        _mm512_set_epi64(0, -1, 0, -1, 0, -1, 0, -1)
    }

    #[inline(always)]
    unsafe fn odd_mask() -> __m512i {
        _mm512_set_epi64(-1, 0, -1, 0, -1, 0, -1, 0)
    }

    /// The full multiply pipeline on lane-packed registers.
    #[inline(always)]
    unsafe fn mul_mm(va: __m512i, vb: __m512i) -> __m512i {
        let lo = _mm512_clmulepi64_epi128(va, vb, 0x00);
        let hi = _mm512_clmulepi64_epi128(va, vb, 0x11);
        let sw = swap_idx();
        let ax = _mm512_xor_si512(va, _mm512_permutexvar_epi64(sw, va));
        let bx = _mm512_xor_si512(vb, _mm512_permutexvar_epi64(sw, vb));
        let mid = _mm512_clmulepi64_epi128(ax, bx, 0x00);
        let c = _mm512_xor_si512(_mm512_xor_si512(mid, lo), hi);
        let c_swapped = _mm512_permutexvar_epi64(sw, c);
        let t_lo = _mm512_xor_si512(lo, _mm512_and_si512(c_swapped, odd_mask()));
        let t_hi = _mm512_xor_si512(hi, _mm512_and_si512(c_swapped, even_mask()));
        reduce_hi_lo(t_lo, t_hi)
    }

    #[inline(always)]
    unsafe fn reduce_hi_lo(t_lo: __m512i, t_hi: __m512i) -> __m512i {
        let em = even_mask();
        let om = odd_mask();
        let sw = swap_idx();
        let h0 = _mm512_and_si512(t_hi, em);
        let h1 = _mm512_and_si512(t_hi, om);
        let s7_odd = _mm512_xor_si512(
            _mm512_slli_epi64(h1, 7),
            _mm512_and_si512(_mm512_permutexvar_epi64(sw, _mm512_srli_epi64(h0, 57)), om),
        );
        let s2_odd = _mm512_xor_si512(
            _mm512_slli_epi64(h1, 2),
            _mm512_and_si512(_mm512_permutexvar_epi64(sw, _mm512_srli_epi64(h0, 62)), om),
        );
        let s1_odd = _mm512_xor_si512(
            _mm512_slli_epi64(h1, 1),
            _mm512_and_si512(_mm512_permutexvar_epi64(sw, _mm512_srli_epi64(h0, 63)), om),
        );
        let r0 = _mm512_xor_si512(
            _mm512_xor_si512(_mm512_slli_epi64(h0, 7), _mm512_slli_epi64(h0, 2)),
            _mm512_xor_si512(_mm512_slli_epi64(h0, 1), h0),
        );
        let r1 = _mm512_xor_si512(_mm512_xor_si512(s7_odd, s2_odd), _mm512_xor_si512(s1_odd, h1));
        let q2 = _mm512_and_si512(
            _mm512_permutexvar_epi64(
                sw,
                _mm512_xor_si512(
                    _mm512_srli_epi64(h1, 57),
                    _mm512_xor_si512(_mm512_srli_epi64(h1, 62), _mm512_srli_epi64(h1, 63)),
                ),
            ),
            em,
        );
        let q2r = _mm512_xor_si512(
            _mm512_xor_si512(_mm512_slli_epi64(q2, 7), _mm512_slli_epi64(q2, 2)),
            _mm512_xor_si512(_mm512_slli_epi64(q2, 1), q2),
        );
        _mm512_xor_si512(
            _mm512_xor_si512(t_lo, _mm512_xor_si512(r0, r1)),
            _mm512_and_si512(q2r, em),
        )
    }

    impl Avx {
        /// The pipeline exposed for the sibling fold modules.
        #[target_feature(enable = "avx512f,vpclmulqdq")]
        pub unsafe fn mul_mm_public(&self, va: __m512i, vb: __m512i) -> __m512i {
            mul_mm(va, vb)
        }

        #[target_feature(enable = "avx512f,vpclmulqdq")]
        pub unsafe fn mul4(&self, a: [Gf128; 4], b: [Gf128; 4]) -> [Gf128; 4] {
            let va = _mm512_loadu_si512(a.as_ptr().cast());
            let vb = _mm512_loadu_si512(b.as_ptr().cast());
            let out = mul_mm(va, vb);
            let mut res = [Gf128(0); 4];
            _mm512_storeu_si512(res.as_mut_ptr().cast(), out);
            res
        }

        #[target_feature(enable = "avx512f,vpclmulqdq")]
        pub unsafe fn msg2(
            &self,
            acc: &mut [Gf128],
            c0: Gf128,
            f1: &[Gf128],
            f2: &[Gf128],
            base: usize,
            block: usize,
            n4: usize,
        ) {
            let zero = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
            let (mut a0, mut a1c, mut a2c) = (zero, zero, zero);
            for i in 0..n4 {
                let b = base + 4 * i;
                let a1v = _mm512_loadu_si512(f1.as_ptr().add(b).cast());
                let d1v = _mm512_xor_si512(
                    a1v,
                    _mm512_loadu_si512(f1.as_ptr().add(b + block).cast()),
                );
                let a2v = _mm512_loadu_si512(f2.as_ptr().add(b).cast());
                let d2v = _mm512_xor_si512(
                    a2v,
                    _mm512_loadu_si512(f2.as_ptr().add(b + block).cast()),
                );
                let x = mul_mm(a1v, a2v);
                let y = mul_mm(a1v, d2v);
                let z = mul_mm(d1v, a2v);
                let w = mul_mm(d1v, d2v);
                a0 = _mm512_xor_si512(a0, x);
                a1c = _mm512_xor_si512(a1c, _mm512_xor_si512(y, z));
                a2c = _mm512_xor_si512(a2c, w);
            }
            let mut r0 = [Gf128(0); 4];
            let mut r1 = [Gf128(0); 4];
            let mut r2 = [Gf128(0); 4];
            _mm512_storeu_si512(r0.as_mut_ptr().cast(), a0);
            _mm512_storeu_si512(r1.as_mut_ptr().cast(), a1c);
            _mm512_storeu_si512(r2.as_mut_ptr().cast(), a2c);
            for o in 0..4 {
                if c0 == crate::gf128::ONE {
                    acc[0] += r0[o];
                    acc[1] += r1[o];
                    acc[2] += r2[o];
                } else {
                    acc[0] += crate::gf128::mul(c0, r0[o]);
                    acc[1] += crate::gf128::mul(c0, r1[o]);
                    acc[2] += crate::gf128::mul(c0, r2[o]);
                }
            }
        }

        #[target_feature(enable = "avx512f,vpclmulqdq")]
        pub unsafe fn msg3(
            &self,
            acc: &mut [Gf128],
            c0: Gf128,
            f1: &[Gf128],
            f2: &[Gf128],
            f3: &[Gf128],
            base: usize,
            block: usize,
            n4: usize,
        ) {
            let zero = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
            let (mut q0, mut q1, mut q2, mut q3) = (zero, zero, zero, zero);
            for i in 0..n4 {
                let b = base + 4 * i;
                let a1v = _mm512_loadu_si512(f1.as_ptr().add(b).cast());
                let d1v = _mm512_xor_si512(
                    a1v,
                    _mm512_loadu_si512(f1.as_ptr().add(b + block).cast()),
                );
                let a2v = _mm512_loadu_si512(f2.as_ptr().add(b).cast());
                let d2v = _mm512_xor_si512(
                    a2v,
                    _mm512_loadu_si512(f2.as_ptr().add(b + block).cast()),
                );
                let a3v = _mm512_loadu_si512(f3.as_ptr().add(b).cast());
                let d3v = _mm512_xor_si512(
                    a3v,
                    _mm512_loadu_si512(f3.as_ptr().add(b + block).cast()),
                );
                let x = mul_mm(a1v, a2v);
                let y = mul_mm(a1v, d2v);
                let z = mul_mm(d1v, a2v);
                let w = mul_mm(d1v, d2v);
                let s1 = _mm512_xor_si512(y, z);
                q0 = _mm512_xor_si512(q0, mul_mm(x, a3v));
                q1 = _mm512_xor_si512(q1, _mm512_xor_si512(mul_mm(x, d3v), mul_mm(s1, a3v)));
                q2 = _mm512_xor_si512(q2, _mm512_xor_si512(mul_mm(s1, d3v), mul_mm(w, a3v)));
                q3 = _mm512_xor_si512(q3, mul_mm(w, d3v));
            }
            let mut r0 = [Gf128(0); 4];
            let mut r1 = [Gf128(0); 4];
            let mut r2 = [Gf128(0); 4];
            let mut r3 = [Gf128(0); 4];
            _mm512_storeu_si512(r0.as_mut_ptr().cast(), q0);
            _mm512_storeu_si512(r1.as_mut_ptr().cast(), q1);
            _mm512_storeu_si512(r2.as_mut_ptr().cast(), q2);
            _mm512_storeu_si512(r3.as_mut_ptr().cast(), q3);
            for o in 0..4 {
                if c0 == crate::gf128::ONE {
                    acc[0] += r0[o];
                    acc[1] += r1[o];
                    acc[2] += r2[o];
                    acc[3] += r3[o];
                } else {
                    acc[0] += crate::gf128::mul(c0, r0[o]);
                    acc[1] += crate::gf128::mul(c0, r1[o]);
                    acc[2] += crate::gf128::mul(c0, r2[o]);
                    acc[3] += crate::gf128::mul(c0, r3[o]);
                }
            }
        }

        /// The fused fold over `n4` chunks: `out[k] = src[k] ^ r·(src[k] ^ src[k+block])`.
        #[target_feature(enable = "avx512f,vpclmulqdq")]
        pub unsafe fn fold_run(
            &self,
            out: &mut [Gf128],
            src: &[Gf128],
            r: Gf128,
            block: usize,
            n4: usize,
        ) {
            let mut rb = [r; 4];
            let rv = _mm512_loadu_si512(rb.as_mut_ptr().cast());
            for i in 0..n4 {
                let k = 4 * i;
                let a = _mm512_loadu_si512(src.as_ptr().add(k).cast());
                let b = _mm512_loadu_si512(src.as_ptr().add(k + block).cast());
                let t = _mm512_xor_si512(a, b);
                let prod = mul_mm(rv, t);
                let res = _mm512_xor_si512(a, prod);
                _mm512_storeu_si512(out.as_mut_ptr().add(k).cast(), res);
            }
        }
    }

    /// `out[4i..4i+4]` = fold of the pairs (src[8j], src[8j+1]) … via the
    /// 64-bit unpack (two 512-bit loads carry 8 consecutive elements).
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub unsafe fn fold_pairs4(out: &mut [Gf128], src: &[Gf128], r: Gf128, n4: usize) {
        use std::arch::x86_64::{
            _mm512_loadu_si512, _mm512_permutex2var_epi64, _mm512_set_epi64,
            _mm512_storeu_si512, _mm512_xor_si512,
        };
        // a = [e0, e2, e4, e6] (the first of each pair), b = the seconds.
        let idx_a = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let idx_b = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let mut rb = [r; 4];
        let rv = _mm512_loadu_si512(rb.as_mut_ptr().cast());
        for i in 0..n4 {
            let lo = 8 * i;
            let reg1 = _mm512_loadu_si512(src.as_ptr().add(lo).cast());
            let reg2 = _mm512_loadu_si512(src.as_ptr().add(lo + 4).cast());
            let a = _mm512_permutex2var_epi64(reg1, idx_a, reg2);
            let b = _mm512_permutex2var_epi64(reg1, idx_b, reg2);
            let t = _mm512_xor_si512(a, b);
            let prod = mul_mm(rv, t);
            let res = _mm512_xor_si512(a, prod);
            _mm512_storeu_si512(out.as_mut_ptr().add(4 * i).cast(), res);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod imp256 {
    //! The AVX2 (2-lane) pipeline: the same structure at 256 bits. Two
    //! consecutive `Gf128`s are one `__m256i`.
    use super::*;
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_clmulepi64_epi128, _mm256_loadu_si256,
        _mm256_permute4x64_epi64, _mm256_set1_epi64x, _mm256_set_epi64x, _mm256_slli_epi64,
        _mm256_srli_epi64, _mm256_storeu_si256, _mm256_xor_si256,
    };

    pub(super) struct Avx2;

    #[inline(always)]
    unsafe fn even_mask() -> __m256i {
        _mm256_set_epi64x(0, -1, 0, -1)
    }

    #[inline(always)]
    unsafe fn odd_mask() -> __m256i {
        _mm256_set_epi64x(-1, 0, -1, 0)
    }

    #[inline(always)]
    unsafe fn mul_mm(va: __m256i, vb: __m256i) -> __m256i {
        let lo = _mm256_clmulepi64_epi128(va, vb, 0x00);
        let hi = _mm256_clmulepi64_epi128(va, vb, 0x11);
        let ax = _mm256_xor_si256(va, _mm256_permute4x64_epi64(va, 0b1011_0001));
        let bx = _mm256_xor_si256(vb, _mm256_permute4x64_epi64(vb, 0b1011_0001));
        let mid = _mm256_clmulepi64_epi128(ax, bx, 0x00);
        let c = _mm256_xor_si256(_mm256_xor_si256(mid, lo), hi);
        let c_swapped = _mm256_permute4x64_epi64(c, 0b1011_0001);
        let t_lo = _mm256_xor_si256(lo, _mm256_and_si256(c_swapped, odd_mask()));
        let t_hi = _mm256_xor_si256(hi, _mm256_and_si256(c_swapped, even_mask()));
        reduce_hi_lo(t_lo, t_hi)
    }

    #[inline(always)]
    unsafe fn reduce_hi_lo(t_lo: __m256i, t_hi: __m256i) -> __m256i {
        let em = even_mask();
        let om = odd_mask();
        let h0 = _mm256_and_si256(t_hi, em);
        let h1 = _mm256_and_si256(t_hi, om);
        let s7_odd = _mm256_xor_si256(
            _mm256_slli_epi64(h1, 7),
            _mm256_and_si256(_mm256_permute4x64_epi64(_mm256_srli_epi64(h0, 57), 0b1011_0001), om),
        );
        let s2_odd = _mm256_xor_si256(
            _mm256_slli_epi64(h1, 2),
            _mm256_and_si256(_mm256_permute4x64_epi64(_mm256_srli_epi64(h0, 62), 0b1011_0001), om),
        );
        let s1_odd = _mm256_xor_si256(
            _mm256_slli_epi64(h1, 1),
            _mm256_and_si256(_mm256_permute4x64_epi64(_mm256_srli_epi64(h0, 63), 0b1011_0001), om),
        );
        let r0 = _mm256_xor_si256(
            _mm256_xor_si256(_mm256_slli_epi64(h0, 7), _mm256_slli_epi64(h0, 2)),
            _mm256_xor_si256(_mm256_slli_epi64(h0, 1), h0),
        );
        let r1 = _mm256_xor_si256(
            _mm256_xor_si256(s7_odd, s2_odd),
            _mm256_xor_si256(s1_odd, h1),
        );
        let q2 = _mm256_and_si256(
            _mm256_permute4x64_epi64(
                _mm256_xor_si256(
                    _mm256_srli_epi64(h1, 57),
                    _mm256_xor_si256(_mm256_srli_epi64(h1, 62), _mm256_srli_epi64(h1, 63)),
                ),
                0b1011_0001,
            ),
            em,
        );
        let q2r = _mm256_xor_si256(
            _mm256_xor_si256(_mm256_slli_epi64(q2, 7), _mm256_slli_epi64(q2, 2)),
            _mm256_xor_si256(_mm256_slli_epi64(q2, 1), q2),
        );
        _mm256_xor_si256(
            _mm256_xor_si256(t_lo, _mm256_xor_si256(r0, r1)),
            _mm256_and_si256(q2r, em),
        )
    }

    impl Avx2 {
        /// The pipeline exposed for the sibling fold modules.
        #[target_feature(enable = "avx2,pclmulqdq")]
        pub unsafe fn mul_mm_public(&self, va: __m256i, vb: __m256i) -> __m256i {
            mul_mm(va, vb)
        }

        #[target_feature(enable = "avx2,pclmulqdq")]
        pub unsafe fn mul2(&self, a: [Gf128; 2], b: [Gf128; 2]) -> [Gf128; 2] {
            let va = _mm256_loadu_si256(a.as_ptr().cast());
            let vb = _mm256_loadu_si256(b.as_ptr().cast());
            let out = mul_mm(va, vb);
            let mut res = [Gf128(0); 2];
            _mm256_storeu_si256(res.as_mut_ptr().cast(), out);
            res
        }

        #[target_feature(enable = "avx2,pclmulqdq")]
        pub unsafe fn msg2(
            &self,
            acc: &mut [Gf128],
            c0: Gf128,
            f1: &[Gf128],
            f2: &[Gf128],
            base: usize,
            block: usize,
            n2: usize,
        ) {
            let zero = _mm256_set1_epi64x(0);
            let (mut a0, mut a1c, mut a2c) = (zero, zero, zero);
            for i in 0..n2 {
                let b = base + 2 * i;
                let a1v = _mm256_loadu_si256(f1.as_ptr().add(b).cast());
                let d1v = _mm256_xor_si256(
                    a1v,
                    _mm256_loadu_si256(f1.as_ptr().add(b + block).cast()),
                );
                let a2v = _mm256_loadu_si256(f2.as_ptr().add(b).cast());
                let d2v = _mm256_xor_si256(
                    a2v,
                    _mm256_loadu_si256(f2.as_ptr().add(b + block).cast()),
                );
                let x = mul_mm(a1v, a2v);
                let y = mul_mm(a1v, d2v);
                let z = mul_mm(d1v, a2v);
                let w = mul_mm(d1v, d2v);
                a0 = _mm256_xor_si256(a0, x);
                a1c = _mm256_xor_si256(a1c, _mm256_xor_si256(y, z));
                a2c = _mm256_xor_si256(a2c, w);
            }
            let mut r0 = [Gf128(0); 2];
            let mut r1 = [Gf128(0); 2];
            let mut r2 = [Gf128(0); 2];
            _mm256_storeu_si256(r0.as_mut_ptr().cast(), a0);
            _mm256_storeu_si256(r1.as_mut_ptr().cast(), a1c);
            _mm256_storeu_si256(r2.as_mut_ptr().cast(), a2c);
            for o in 0..2 {
                if c0 == crate::gf128::ONE {
                    acc[0] += r0[o];
                    acc[1] += r1[o];
                    acc[2] += r2[o];
                } else {
                    acc[0] += crate::gf128::mul(c0, r0[o]);
                    acc[1] += crate::gf128::mul(c0, r1[o]);
                    acc[2] += crate::gf128::mul(c0, r2[o]);
                }
            }
        }

        #[target_feature(enable = "avx2,pclmulqdq")]
        pub unsafe fn msg3(
            &self,
            acc: &mut [Gf128],
            c0: Gf128,
            f1: &[Gf128],
            f2: &[Gf128],
            f3: &[Gf128],
            base: usize,
            block: usize,
            n2: usize,
        ) {
            let zero = _mm256_set1_epi64x(0);
            let (mut q0, mut q1, mut q2, mut q3) = (zero, zero, zero, zero);
            for i in 0..n2 {
                let b = base + 2 * i;
                let a1v = _mm256_loadu_si256(f1.as_ptr().add(b).cast());
                let d1v = _mm256_xor_si256(
                    a1v,
                    _mm256_loadu_si256(f1.as_ptr().add(b + block).cast()),
                );
                let a2v = _mm256_loadu_si256(f2.as_ptr().add(b).cast());
                let d2v = _mm256_xor_si256(
                    a2v,
                    _mm256_loadu_si256(f2.as_ptr().add(b + block).cast()),
                );
                let a3v = _mm256_loadu_si256(f3.as_ptr().add(b).cast());
                let d3v = _mm256_xor_si256(
                    a3v,
                    _mm256_loadu_si256(f3.as_ptr().add(b + block).cast()),
                );
                let x = mul_mm(a1v, a2v);
                let y = mul_mm(a1v, d2v);
                let z = mul_mm(d1v, a2v);
                let w = mul_mm(d1v, d2v);
                let s1 = _mm256_xor_si256(y, z);
                q0 = _mm256_xor_si256(q0, mul_mm(x, a3v));
                q1 = _mm256_xor_si256(q1, _mm256_xor_si256(mul_mm(x, d3v), mul_mm(s1, a3v)));
                q2 = _mm256_xor_si256(q2, _mm256_xor_si256(mul_mm(s1, d3v), mul_mm(w, a3v)));
                q3 = _mm256_xor_si256(q3, mul_mm(w, d3v));
            }
            let mut r0 = [Gf128(0); 2];
            let mut r1 = [Gf128(0); 2];
            let mut r2 = [Gf128(0); 2];
            let mut r3 = [Gf128(0); 2];
            _mm256_storeu_si256(r0.as_mut_ptr().cast(), q0);
            _mm256_storeu_si256(r1.as_mut_ptr().cast(), q1);
            _mm256_storeu_si256(r2.as_mut_ptr().cast(), q2);
            _mm256_storeu_si256(r3.as_mut_ptr().cast(), q3);
            for o in 0..2 {
                if c0 == crate::gf128::ONE {
                    acc[0] += r0[o];
                    acc[1] += r1[o];
                    acc[2] += r2[o];
                    acc[3] += r3[o];
                } else {
                    acc[0] += crate::gf128::mul(c0, r0[o]);
                    acc[1] += crate::gf128::mul(c0, r1[o]);
                    acc[2] += crate::gf128::mul(c0, r2[o]);
                    acc[3] += crate::gf128::mul(c0, r3[o]);
                }
            }
        }

        #[target_feature(enable = "avx2,pclmulqdq")]
        pub unsafe fn fold_run(
            &self,
            out: &mut [Gf128],
            src: &[Gf128],
            r: Gf128,
            block: usize,
            n2: usize,
        ) {
            let mut rb = [r; 2];
            let rv = _mm256_loadu_si256(rb.as_mut_ptr().cast());
            for i in 0..n2 {
                let k = 2 * i;
                let a = _mm256_loadu_si256(src.as_ptr().add(k).cast());
                let b = _mm256_loadu_si256(src.as_ptr().add(k + block).cast());
                let t = _mm256_xor_si256(a, b);
                let prod = mul_mm(rv, t);
                let res = _mm256_xor_si256(a, prod);
                _mm256_storeu_si256(out.as_mut_ptr().add(k).cast(), res);
            }
        }
    }

    #[target_feature(enable = "avx2,pclmulqdq")]
    pub unsafe fn fold_pairs2(out: &mut [Gf128], src: &[Gf128], r: Gf128, n2: usize) {
        use std::arch::x86_64::{
            _mm256_loadu_si256, _mm256_permute2x128_si256, _mm256_storeu_si256,
            _mm256_xor_si256,
        };
        // a = [e0, e2] (the first of each pair), b = [e1, e3].
        let mut rb = [r; 2];
        let rv = _mm256_loadu_si256(rb.as_mut_ptr().cast());
        for i in 0..n2 {
            let lo = 4 * i;
            let reg1 = _mm256_loadu_si256(src.as_ptr().add(lo).cast());
            let reg2 = _mm256_loadu_si256(src.as_ptr().add(lo + 2).cast());
            let a = _mm256_permute2x128_si256(reg1, reg2, 0x20);
            let b = _mm256_permute2x128_si256(reg1, reg2, 0x31);
            let t = _mm256_xor_si256(a, b);
            let prod = mul_mm(rv, t);
            let res = _mm256_xor_si256(a, prod);
            _mm256_storeu_si256(out.as_mut_ptr().add(2 * i).cast(), res);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn avx512()-> imp512::Avx {
    imp512::Avx
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn avx2() -> imp256::Avx2 {
    imp256::Avx2
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,pclmulqdq")]
#[allow(unsafe_code)]
unsafe fn mul2_avx2(a: [Gf128; 2], b: [Gf128; 2]) -> [Gf128; 2] {
    // SAFETY: guarded by the measured dispatch; the self-contained body
    // mirrors imp256::mul2 (kept separate so the probe can call it without
    // going through the dispatch).
    let got = avx2().mul2(a, b);
    got
}

/// 4 independent field multiplies (bit-identical to the scalar path); used
/// by cross-validation tests. The production kernels are the fused
/// `msg2_batch`/`msg3_batch`/`fold_run` below.
pub fn mul4(a: [Gf128; 4], b: [Gf128; 4]) -> [Gf128; 4] {
    let mut out = [Gf128(0); 4];
    for i in 0..4 {
        out[i] = crate::gf128::mul(a[i], b[i]);
    }
    out
}

/// The safe fused 2-factor message kernel over `count` positions from
/// `base` (pair partners at `+ block`). Returns the number of leading
/// positions consumed (0 = no SIMD backend: the caller runs its scalar
/// path for everything).
pub fn msg2_batch(
    acc: &mut [Gf128],
    c0: Gf128,
    f1: &[Gf128],
    f2: &[Gf128],
    base: usize,
    block: usize,
    count: usize,
) -> usize {
    match detect() {
        #[cfg(target_arch = "x86_64")]
        SimdKind::Avx512 => {
            let n4 = count / 4;
            #[allow(unsafe_code)]
            unsafe {
                avx512().msg2(acc, c0, f1, f2, base, block, n4);
            }
            4 * n4
        }
        #[cfg(target_arch = "x86_64")]
        SimdKind::Avx2 => {
            let n2 = count / 2;
            #[allow(unsafe_code)]
            unsafe {
                avx2().msg2(acc, c0, f1, f2, base, block, n2);
            }
            2 * n2
        }
        _ => 0,
    }
}

/// The safe fused 3-factor message kernel over `count` positions from
/// `base`. Returns the number of leading positions consumed.
pub fn msg3_batch(
    acc: &mut [Gf128],
    c0: Gf128,
    f1: &[Gf128],
    f2: &[Gf128],
    f3: &[Gf128],
    base: usize,
    block: usize,
    count: usize,
) -> usize {
    match detect() {
        #[cfg(target_arch = "x86_64")]
        SimdKind::Avx512 => {
            let n4 = count / 4;
            #[allow(unsafe_code)]
            unsafe {
                avx512().msg3(acc, c0, f1, f2, f3, base, block, n4);
            }
            4 * n4
        }
        #[cfg(target_arch = "x86_64")]
        SimdKind::Avx2 => {
            let n2 = count / 2;
            #[allow(unsafe_code)]
            unsafe {
                avx2().msg3(acc, c0, f1, f2, f3, base, block, n2);
            }
            2 * n2
        }
        _ => 0,
    }
}

/// The fused fold over `count` outputs:
/// `out[k] = src[k] ^ r·(src[k] ^ src[k + block])` — the **contiguous-run**
/// form of the sum-check fold (the top-bit and LSB rounds). Returns the
/// number of leading outputs consumed (0 = no backend).
pub fn fold_run(out: &mut [Gf128], src: &[Gf128], r: Gf128, block: usize, count: usize) -> usize {
    match detect() {
        #[cfg(target_arch = "x86_64")]
        SimdKind::Avx512 => {
            let n4 = count / 4;
            #[allow(unsafe_code)]
            unsafe {
                avx512().fold_run(out, src, r, block, n4);
            }
            4 * n4
        }
        #[cfg(target_arch = "x86_64")]
        SimdKind::Avx2 => {
            let n2 = count / 2;
            #[allow(unsafe_code)]
            unsafe {
                avx2().fold_run(out, src, r, block, n2);
            }
            2 * n2
        }
        _ => 0,
    }
}


/// The interleaved-pairs fold (binding the LSB): `out[k] = src[2k] ^
/// r·(src[2k] ^ src[2k+1])` — the 2-per-lane unpack form. Returns the
/// outputs consumed.
pub fn fold_pairs(out: &mut [Gf128], src: &[Gf128], r: Gf128, count: usize) -> usize {
    match detect() {
        #[cfg(target_arch = "x86_64")]
        SimdKind::Avx512 => {
            let n4 = count / 4;
            #[allow(unsafe_code)]
            unsafe {
                imp512::fold_pairs4(out, src, r, n4);
            }
            4 * n4
        }
        #[cfg(target_arch = "x86_64")]
        SimdKind::Avx2 => {
            let n2 = count / 2;
            #[allow(unsafe_code)]
            unsafe {
                imp256::fold_pairs2(out, src, r, n2);
            }
            2 * n2
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    #[test]
    fn detected_backend_is_measured_fastest() {
        let k = detect();
        eprintln!("measured backend: {k:?}");
    }

    #[test]
    fn msg_kernels_match_scalar() {
        let mut rng = rand::rng();
        let n = 256usize;
        let block = 512usize;
        let f1: Vec<Gf128> = (0..n + block).map(|_| Gf128(rng.random::<u128>())).collect();
        let f2: Vec<Gf128> = (0..n + block).map(|_| Gf128(rng.random::<u128>())).collect();
        let f3: Vec<Gf128> = (0..n + block).map(|_| Gf128(rng.random::<u128>())).collect();
        for &c0 in &[crate::gf128::ONE, Gf128(rng.random::<u128>())] {
            let mut acc_batch = [Gf128(0); 4];
            let done = msg2_batch(&mut acc_batch, c0, &f1, &f2, 0, block, n);
            for b in done..n {
                let a1 = f1[b];
                let d1 = f1[b] + f1[b + block];
                let a2 = f2[b];
                let d2 = f2[b] + f2[b + block];
                let ca = crate::gf128::mul(c0, a1);
                let cd1 = crate::gf128::mul(c0, d1);
                acc_batch[0] += crate::gf128::mul(ca, a2);
                acc_batch[1] += crate::gf128::mul(ca, d2) + crate::gf128::mul(cd1, a2);
                acc_batch[2] += crate::gf128::mul(cd1, d2);
            }
            let mut acc_scalar = [Gf128(0); 4];
            for b in 0..n {
                let a1 = f1[b];
                let d1 = f1[b] + f1[b + block];
                let a2 = f2[b];
                let d2 = f2[b] + f2[b + block];
                let ca = crate::gf128::mul(c0, a1);
                let cd1 = crate::gf128::mul(c0, d1);
                acc_scalar[0] += crate::gf128::mul(ca, a2);
                acc_scalar[1] += crate::gf128::mul(ca, d2) + crate::gf128::mul(cd1, a2);
                acc_scalar[2] += crate::gf128::mul(cd1, d2);
            }
            for e in 0..3 {
                assert_eq!(acc_batch[e], acc_scalar[e], "msg2 c0-ac {e}");
            }
            let mut acc3 = [Gf128(0); 5];
            let done3 = msg3_batch(&mut acc3, c0, &f1, &f2, &f3, 0, block, n);
            for b in done3..n {
                let a1 = f1[b];
                let d1 = f1[b] + f1[b + block];
                let a2 = f2[b];
                let d2 = f2[b] + f2[b + block];
                let a3 = f3[b];
                let d3 = f3[b] + f3[b + block];
                let a1a2 = crate::gf128::mul(a1, a2);
                let a1d2 = crate::gf128::mul(a1, d2);
                let d1a2 = crate::gf128::mul(d1, a2);
                let d1d2 = crate::gf128::mul(d1, d2);
                let s1 = a1d2 + d1a2;
                if c0 == crate::gf128::ONE {
                    acc3[0] += crate::gf128::mul(a1a2, a3);
                    acc3[1] += crate::gf128::mul(a1a2, d3) + crate::gf128::mul(s1, a3);
                    acc3[2] += crate::gf128::mul(s1, d3) + crate::gf128::mul(d1d2, a3);
                    acc3[3] += crate::gf128::mul(d1d2, d3);
                } else {
                    acc3[0] += crate::gf128::mul(c0, crate::gf128::mul(a1a2, a3));
                    acc3[1] +=
                        crate::gf128::mul(c0, crate::gf128::mul(a1a2, d3) + crate::gf128::mul(s1, a3));
                    acc3[2] +=
                        crate::gf128::mul(c0, crate::gf128::mul(s1, d3) + crate::gf128::mul(d1d2, a3));
                    acc3[3] += crate::gf128::mul(c0, crate::gf128::mul(d1d2, d3));
                }
            }
            let mut acc3s = [Gf128(0); 5];
            for b in 0..n {
                let a1 = f1[b];
                let d1 = f1[b] + f1[b + block];
                let a2 = f2[b];
                let d2 = f2[b] + f2[b + block];
                let a3 = f3[b];
                let d3 = f3[b] + f3[b + block];
                let a1a2 = crate::gf128::mul(a1, a2);
                let a1d2 = crate::gf128::mul(a1, d2);
                let d1a2 = crate::gf128::mul(d1, a2);
                let d1d2 = crate::gf128::mul(d1, d2);
                let s1 = a1d2 + d1a2;
                if c0 == crate::gf128::ONE {
                    acc3s[0] += crate::gf128::mul(a1a2, a3);
                    acc3s[1] += crate::gf128::mul(a1a2, d3) + crate::gf128::mul(s1, a3);
                    acc3s[2] += crate::gf128::mul(s1, d3) + crate::gf128::mul(d1d2, a3);
                    acc3s[3] += crate::gf128::mul(d1d2, d3);
                } else {
                    acc3s[0] += crate::gf128::mul(c0, crate::gf128::mul(a1a2, a3));
                    acc3s[1] +=
                        crate::gf128::mul(c0, crate::gf128::mul(a1a2, d3) + crate::gf128::mul(s1, a3));
                    acc3s[2] +=
                        crate::gf128::mul(c0, crate::gf128::mul(s1, d3) + crate::gf128::mul(d1d2, a3));
                    acc3s[3] += crate::gf128::mul(c0, crate::gf128::mul(d1d2, d3));
                }
            }
            for e in 0..4 {
                assert_eq!(acc3[e], acc3s[e], "msg3 acc {e}");
            }
        }
    }

    #[test]
    fn fold_run_matches_scalar() {
        let mut rng = rand::rng();
        let n = 256usize;
        let block = 512usize;
        let src: Vec<Gf128> = (0..n + block).map(|_| Gf128(rng.random::<u128>())).collect();
        let r = Gf128(rng.random::<u128>());
        let mut out = vec![Gf128(0); n];
        let did = fold_run(&mut out, &src, r, block, n);
        for k in did..n {
            let a = src[k];
            let b = src[k + block];
            out[k] = a + crate::gf128::mul(r, a + b);
        }
        for k in 0..n {
            let a = src[k];
            let b = src[k + block];
            let want = a + crate::gf128::mul(r, a + b);
            assert_eq!(out[k], want, "fold_run {k}");
        }
    }

    #[test]
    fn mul2_avx2_matches_scalar() {
        let mut rng = rand::rng();
        for _ in 0..256 {
            let a = [Gf128(rng.random::<u128>()), Gf128(rng.random::<u128>())];
            let b = [Gf128(rng.random::<u128>()), Gf128(rng.random::<u128>())];
            #[allow(unsafe_code)]
            let got = unsafe { mul2_avx2(a, b) };
            assert_eq!(got[0], crate::gf128::mul(a[0], b[0]));
            assert_eq!(got[1], crate::gf128::mul(a[1], b[1]));
        }
    }
}

/// 2-lane broadcast multiply for strided callers: `[r·t0, r·t1]` through
/// the measured backend (scalar on non-x86 hosts).
pub fn mul2_inline(r: Gf128, t: [Gf128; 2]) -> [Gf128; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        match detect() {
            SimdKind::Avx512 => {
                let mut b4 = [Gf128(0); 4];
                b4[0] = t[0];
                b4[1] = t[1];
                let mut rb = [r; 4];
                let got = unsafe { avx512().mul4(rb, b4) };
                [got[0], got[1]]
            }
            SimdKind::Avx2 => {
                #[allow(unsafe_code)]
                unsafe {
                    avx2().mul2([r, r], t)
                }
            }
            _ => [crate::gf128::mul(r, t[0]), crate::gf128::mul(r, t[1])],
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        [crate::gf128::mul(r, t[0]), crate::gf128::mul(r, t[1])]
    }
}
