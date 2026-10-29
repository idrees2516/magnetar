//! GF(2^32) — the position-encoding field `K_pos` of the Mikros copy
//! constraint (ePrint 2026/2646, §5.1 "Proof for the PLONK Copy
//! Constraint").
//!
//! `K_pos` is *the* unique order-2^32 subfield of `E = GF(2^128)` (the
//! paper's tower `F ⊂ K_pos ⊂ E` with `[K_pos : F] = d = 32`,
//! `[E : K_pos] = κ_pos = 4`).
//!
//! **Code representation.** A `K_pos`-element is stored as a `u32` *code*
//! whose bits are the coefficients over the GF(2)-power basis
//! `(1, h, h², …, h³¹)` of a root `h ∈ K_pos` of an irreducible degree-32
//! polynomial `p(X)` found at init. Multiplication of codes is carry-less
//! 32×32 followed by reduction mod `p`:
//!
//! * **PCLMULQDQ fast path**: one 64×64 CLMUL (operands zero-extended)
//!   produces the ≤62-bit product; a two-step **Barrett reduction**
//!   (`q̂ = t_hi ^ ((t_hi·mu_low) >> 32)`, `r = t ⊕ q̂·p`, with
//!   `μ = ⌊X⁶⁴/p⌋` precomputed) folds it in two more CLMULs — for *any*
//!   shape of `p` (no sparse-modulus search needed).
//! * **VPCLMULQDQ lane form** ([`Gf32::mul4`]): four independent
//!   products per ZMM — the coefficient-tree and encoding kernels of
//!   [`crate::copy`] run 4 rows at a time.
//! * **Portable fallback**: the schoolbook carry-less loop + the
//!   shift-and-fold reduction (also the cross-validation reference).
//!
//! **The position encoding** (§5.1 / Appendix B.3): with the power basis as
//! the `θ`-basis, the joint value–position label of position `n` with
//! witness bit `w` is *literally* `code = n | (w << 31)`:
//! `code(ι_n) = n`, `code(ς_n) = σ₀(n)` — the low `d_L = ⌈log₂ L⌉` bits are
//! the position digits and the top coordinate `θ_32 = h³¹` carries the
//! witness (Lemma 2's injectivity: distinct `(position, w)` give distinct
//! codes — the coordinates never collide because `d_L + 1 ≤ 32`).
//!
//! The parameter correspondence for every single-instance size the
//! Theorem-1 interface admits (`N ≥ κ² = 2¹⁴`, so `L = 4N ≤ 2²⁷` and
//! `d_L + 1 ≤ 28 ≤ d = 32`): `d = 32`, `κ_pos = 4`.

use magnetar_field::Gf128;
use magnetar_field::{mul as mul_e, ONE as E_ONE, ZERO as E_ZERO};

/// The intermediate field's tabulated arithmetic, built once per process.
pub struct Gf32 {
    /// `embed32[b][byte]` — the E-element of the coded value with byte
    /// `byte` in byte-position `b` (a GF(2)-linear map: the code's bits are
    /// the `h^i` coordinates).
    pub embed32: [[Gf128; 256]; 4],
    /// The power basis `(1, h, …, h³¹)` as E-elements.
    pub basis: [Gf128; 32],
    /// The reduction polynomial's low part: `p = X³² + p_low`.
    pub p_low: u32,
    /// The Barrett constant: `μ = X³² + mu_low` with `μ = ⌊X⁶⁴/p⌋`.
    pub mu_low: u32,
}

/// Carry-less product of two 32-bit polynomials (the ≤62-bit result).
pub(crate) fn clmul32(a: u32, b: u32) -> u64 {
    // the kernel helper shared with `copy`
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("pclmulqdq") {
            #[allow(unsafe_code)]
            // SAFETY: guarded by the feature check; both operands are
            // zero-extended 32-bit values, so the 64-bit CLMUL's low half
            // is the full ≤62-bit product.
            unsafe {
                use std::arch::x86_64::{_mm_clmulepi64_si128, _mm_cvtsi128_si64, _mm_set_epi64x};
                let va = _mm_set_epi64x(0, a as i64);
                let vb = _mm_set_epi64x(0, b as i64);
                let r = _mm_clmulepi64_si128(va, vb, 0x00);
                _mm_cvtsi128_si64(r) as u64
            }
        } else {
            clmul32_portable(a, b)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        clmul32_portable(a, b)
    }
}

fn clmul32_portable(a: u32, b: u32) -> u64 {
    let mut acc = 0u64;
    let mut b = b as u64;
    let mut i = 0;
    while b != 0 {
        if b & 1 == 1 {
            acc ^= (a as u64) << i;
        }
        b >>= 1;
        i += 1;
    }
    acc
}

/// Polynomial long division: `μ = ⌊X⁶⁴/p⌋ = X³² + mu_low` (the Barrett
/// constant; the monic quotient of dividing `X⁶⁴` by `p = X³² + p_low`).
fn barrett_constant(p_low: u32) -> u32 {
    let mut rem: u128 = 1u128 << 64;
    let mut quotient: u128 = 0;
    for deg in (32..=64).rev() {
        if (rem >> deg) & 1 == 1 {
            rem ^= (1u128 << deg) | ((p_low as u128) << (deg - 32));
            quotient |= 1u128 << (deg - 32);
        }
    }
    debug_assert_eq!(quotient >> 32, 1, "μ is monic of degree 32");
    (quotient & 0xFFFF_FFFF) as u32
}

/// The Barrett reduction of a ≤62-bit carry-less product mod
/// `p = X³² + p_low` (the two-CLMUL form; unit-gated against the loop).
#[inline(always)]
fn reduce_barrett(t: u64, p_low: u32, mu_low: u32) -> u32 {
    // q̂ = (t_hi·μ) >> 32 with μ = X³² + mu_low — the t_hi·X³² term is the
    // plain t_hi after the shift, so q̂ = t_hi ^ ((t_hi·mu_low) >> 32).
    // All CLMULs keep the low 64 bits (the operands are ≤ 31/33 bits, so
    // each product fits in 63).
    let t_hi = (t >> 32) as u32;
    let q = ((clmul32(t_hi, mu_low) >> 32) as u32) ^ t_hi;
    let qp = clmul32(q, p_low);
    (t ^ qp) as u32
}

/// The shift-and-fold reference reduction (correct for any p; slow).
fn reduce_loop(mut t: u64, p_low: u32) -> u32 {
    // fold while degree ≥ 32: the top bit removed each step converges
    // because p is monic (X^32 ≡ p_low).
    while t >> 32 != 0 {
        let hi = (t >> 32) as u32;
        t = (t & 0xFFFF_FFFF) ^ clmul32_portable(hi, p_low);
    }
    t as u32
}

impl Gf32 {
    pub fn new() -> Self {
        // K_pos* = <α^((2^128−1)/(2^32−1))>, α = the GCM root Gf128(2).
        let alpha = Gf128(2);
        let m: u128 = u128::MAX; // 2^128 − 1
        debug_assert_eq!(m % ((1u64 << 32) - 1) as u128, 0);
        let g = pow_u128(alpha, m / (((1u64 << 32) - 1) as u128));

        // Find h with degree exactly 32 over GF(2) (in no proper subfield).
        let mut h = E_ZERO;
        {
            let mut x = E_ONE;
            for _ in 0..1 << 20 {
                if degree_is_32(x) {
                    h = x;
                    break;
                }
                x = mul_e(x, g);
            }
        }
        assert_ne!(h, E_ZERO, "K_pos contains degree-32 elements");

        // The power basis and the embed tables.
        let mut basis = [E_ZERO; 32];
        let mut acc = E_ONE;
        for b in basis.iter_mut() {
            *b = acc;
            acc = mul_e(acc, h);
        }
        // h has multiplicative order dividing 2^32 − 1 (K_pos*), i.e. the
        // Frobenius^32 fixes it; spot-checked by the tests.
        let mut embed32 = [[E_ZERO; 256]; 4];
        for byte_pos in 0..4 {
            for code in 1..256usize {
                let low = code & (code - 1);
                let bit = (code ^ low).trailing_zeros() as usize;
                embed32[byte_pos][code] = embed32[byte_pos][low] + basis[byte_pos * 8 + bit];
            }
        }

        // The minimal polynomial p = ∏(X + h^{2^i}) — coefficients over
        // GF(2) (E[X] arithmetic at init; the non-leading coefficients
        // must collapse to GF(2) elements).
        let p_low = {
            // poly as Vec<Gf128>, little-endian coefficients.
            let mut poly: Vec<Gf128> = vec![E_ZERO, E_ONE]; // X + h
            poly[0] = h;
            let mut frob = mul_e(h, h); // h^{2^i}
            for _ in 1..32 {
                // poly *= (X + frob)
                let mut next = vec![E_ZERO; poly.len() + 1];
                for (i, &c) in poly.iter().enumerate() {
                    next[i] += mul_e(c, frob); // c·frob (constant shift)
                    next[i + 1] += c; // c·X
                }
                poly = next;
                frob = mul_e(frob, frob);
            }
            debug_assert_eq!(poly.len(), 33);
            debug_assert_eq!(poly[32], E_ONE, "p is monic of degree 32");
            let mut bits = 0u32;
            for (i, &c) in poly.iter().take(32).enumerate() {
                if c != E_ZERO {
                    debug_assert_eq!(c, E_ONE, "minpoly coefficients are in GF(2)");
                    bits |= 1 << i;
                }
            }
            bits
        };

        let mu_low = barrett_constant(p_low);
        let f = Gf32 {
            embed32,
            basis,
            p_low,
            mu_low,
        };
        // The init-time self-check: the Barrett route == the loop route on
        // a deterministic sample (the test module covers random samples).
        debug_assert_eq!(f.mul(0x1234_5678, 0x9abc_def0), {
            let t = clmul32_portable(0x1234_5678, 0x9abc_def0);
            reduce_loop(t, p_low)
        });
        f
    }

    /// The `E`-embedding of a code (4 byte-table lookups).
    #[inline(always)]
    pub fn embed(&self, code: u32) -> Gf128 {
        self.embed32[0][(code & 0xFF) as usize]
            + self.embed32[1][((code >> 8) & 0xFF) as usize]
            + self.embed32[2][((code >> 16) & 0xFF) as usize]
            + self.embed32[3][((code >> 24) & 0xFF) as usize]
    }

    /// Code-level product (PCLMULQDQ + Barrett; portable fallback).
    #[inline(always)]
    pub fn mul(&self, a: u32, b: u32) -> u32 {
        let t = clmul32(a, b);
        debug_assert_eq!(
            reduce_barrett(t, self.p_low, self.mu_low),
            reduce_loop(t, self.p_low),
            "the Barrett route matches the shift-and-fold reference"
        );
        reduce_barrett(t, self.p_low, self.mu_low)
    }

    /// The `h`-power: `h^k` as a code (`h^31` is the witness coordinate).
    #[inline(always)]
    pub fn h_pow(&self, k: usize) -> u32 {
        debug_assert!(k < 32);
        1u32 << k
    }

    /// Four independent products, VPCLMULQDQ-laned when available (the
    /// coefficient-tree and encoding kernels' inner op; bit-identical to
    /// four [`Self::mul`] calls — unit-gated).
    #[inline(always)]
    pub fn mul4(&self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        #[cfg(target_arch = "x86_64")]
        {
            if magnetar_field::avx::detect() == magnetar_field::avx::SimdKind::Avx512 {
                #[allow(unsafe_code)]
                // SAFETY: the measured dispatch confirmed avx512f +
                // vpclmulqdq; the lane arrays are zero-extended [u64; 8]s
                // built on the stack.
                unsafe {
                    use std::arch::x86_64::{
                        _mm512_clmulepi64_epi128, _mm512_loadu_si512, _mm512_set1_epi64,
                        _mm512_srli_epi64, _mm512_storeu_si512, _mm512_xor_si512,
                    };
                    let aa: [u64; 8] = [
                        a[0] as u64,
                        a[1] as u64,
                        a[2] as u64,
                        a[3] as u64,
                        0,
                        0,
                        0,
                        0,
                    ];
                    let bb: [u64; 8] = [
                        b[0] as u64,
                        b[1] as u64,
                        b[2] as u64,
                        b[3] as u64,
                        0,
                        0,
                        0,
                        0,
                    ];
                    let va = _mm512_loadu_si512(aa.as_ptr().cast());
                    let vb = _mm512_loadu_si512(bb.as_ptr().cast());
                    let t = _mm512_clmulepi64_epi128(va, vb, 0x00); // ≤62-bit per lane
                    let t_hi = _mm512_srli_epi64(t, 32);
                    let mu = _mm512_set1_epi64(self.mu_low as i64);
                    let q_full = _mm512_clmulepi64_epi128(t_hi, mu, 0x00);
                    // q̂ = t_hi ^ (t_hi·mu_low >> 32) (the μ = X³² + mu_low split)
                    let q = _mm512_xor_si512(_mm512_srli_epi64(q_full, 32), t_hi);
                    let p = _mm512_set1_epi64(self.p_low as i64);
                    let qp = _mm512_clmulepi64_epi128(q, p, 0x00);
                    let r = _mm512_xor_si512(t, qp);
                    let mut lanes = [0u64; 8];
                    _mm512_storeu_si512(lanes.as_mut_ptr().cast(), r);
                    return [
                        lanes[0] as u32,
                        lanes[1] as u32,
                        lanes[2] as u32,
                        lanes[3] as u32,
                    ];
                }
            }
        }
        let mut out = [0u32; 4];
        for i in 0..4 {
            out[i] = self.mul(a[i], b[i]);
        }
        out
    }

    /// The 4-term K_pos combination `⊕_j w_j·c_j`, VPCLMULQDQ-laned when
    /// available (one CLMUL per term plus the horizontal lane XOR).
    #[inline(always)]
    pub fn dot4(&self, w: [u32; 4], c: [u32; 4]) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            if magnetar_field::avx::detect() == magnetar_field::avx::SimdKind::Avx512 {
                let prod = self.mul4(w, c);
                #[allow(unsafe_code)]
                // SAFETY: the dispatch confirmed the features; the lane
                // array is a stack [u64; 8].
                unsafe {
                    use std::arch::x86_64::{
                        _mm512_loadu_si512, _mm512_permutexvar_epi64, _mm512_set_epi64,
                        _mm512_storeu_si512, _mm512_xor_si512,
                    };
                    let lanes: [u64; 8] = [
                        prod[0] as u64,
                        prod[1] as u64,
                        prod[2] as u64,
                        prod[3] as u64,
                        0,
                        0,
                        0,
                        0,
                    ];
                    let v = _mm512_loadu_si512(lanes.as_ptr().cast());
                    // XOR the mirrored lane pairs: lanes 0,1 = p0^p2, p1^p3
                    let idx = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
                    let x = _mm512_xor_si512(v, _mm512_permutexvar_epi64(idx, v));
                    let mut out = [0u64; 8];
                    _mm512_storeu_si512(out.as_mut_ptr().cast(), x);
                    return (out[0] ^ out[1]) as u32;
                }
            }
        }
        let mut acc = 0u32;
        for j in 0..4 {
            acc ^= self.mul(w[j], c[j]);
        }
        acc
    }
}

impl Default for Gf32 {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether `x ∈ K_pos` has degree exactly 32 over GF(2) (member of no
/// proper subfield GF(2^k), k | 32, k < 32).
fn degree_is_32(x: Gf128) -> bool {
    let mut x32 = x;
    for _ in 0..32 {
        x32 = mul_e(x32, x32);
    }
    if x32 != x {
        return false; // not in K_pos
    }
    // proper-subfield membership: x^{2^k} = x for some k ∈ {1,2,4,8,16}
    let mut cur = x;
    for _ in 1..=16 {
        cur = mul_e(cur, cur); // cur = x^{2^k}
        if cur == x {
            return false; // degree divides k < 32
        }
    }
    true
}

fn pow_u128(a: Gf128, k: u128) -> Gf128 {
    let mut result = E_ONE;
    let mut base = a;
    let mut e = k;
    while e > 0 {
        if e & 1 == 1 {
            result = mul_e(result, base);
        }
        base = mul_e(base, base);
        e >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        static RNG: std::cell::Cell<u64> = const { std::cell::Cell::new(0xa5a5_1234_9876_f00d) };
    }

    fn word32() -> u32 {
        RNG.with(|c| {
            let mut r = c.get();
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            c.set(r);
            r as u32
        })
    }

    #[test]
    fn field_axioms_and_barrett() {
        let f = Gf32::new();
        // identity + inverse-by-order (K_pos* has order 2^32 − 1)
        assert_eq!(f.mul(1, 0xdead_beef), 0xdead_beef);
        // commutativity / associativity / distributivity on samples
        for _ in 0..512 {
            let a = word32();
            let b = word32();
            let c = word32();
            assert_eq!(f.mul(a, b), f.mul(b, a));
            assert_eq!(f.mul(f.mul(a, b), c), f.mul(a, f.mul(b, c)));
            assert_eq!(f.mul(a, b ^ c), f.mul(a, b) ^ f.mul(a, c));
            // the Barrett route == the shift-and-fold reference
            let t = clmul32_portable(a, b);
            assert_eq!(f.mul(a, b), reduce_loop(t, f.p_low), "barrett vs loop");
        }
        // the lane forms match the scalar
        for _ in 0..512 {
            let a = [word32(), word32(), word32(), word32()];
            let b = [word32(), word32(), word32(), word32()];
            let got = f.mul4(a, b);
            for i in 0..4 {
                assert_eq!(got[i], f.mul(a[i], b[i]), "mul4 lane {i}");
            }
            let d = f.dot4(a, b);
            let mut want = 0u32;
            for i in 0..4 {
                want ^= f.mul(a[i], b[i]);
            }
            assert_eq!(d, want, "dot4");
        }
    }

    #[test]
    fn subfield_closure_and_embed() {
        let f = Gf32::new();
        for _ in 0..256 {
            let a = word32();
            let b = word32();
            let ea = f.embed(a);
            let eb = f.embed(b);
            // embed is a ring map: K_pos ⊂ E
            assert_eq!(f.embed(a ^ b), ea + eb);
            assert_eq!(f.embed(f.mul(a, b)), mul_e(ea, eb));
            // closure: products stay in the subfield (Frobenius^32 fixes K_pos)
            let mut k = f.embed(f.mul(a, b));
            for _ in 0..32 {
                k = mul_e(k, k);
            }
            assert_eq!(k, f.embed(f.mul(a, b)), "Frobenius^32 fixes K_pos");
        }
        // the power basis codes: h^k ↔ 1 << k
        for k in 0..32 {
            assert_eq!(f.embed(1u32 << k), f.basis[k]);
        }
    }

    #[test]
    fn position_encoding_is_injective() {
        // Lemma 2's requirement on the joint encoding: distinct
        // (position, witness) pairs give distinct K_pos elements.
        // code = pos | (w << 31) — injective while pos < 2^31.
        let mut seen = std::collections::HashSet::new();
        let mut r = 0x1234u64;
        let mut next = move || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            r
        };
        for _ in 0..4096 {
            let pos = (next() as u32) & 0x7FFF_FFFF;
            let w = (next() & 1) as u32;
            assert!(seen.insert(pos | (w << 31)));
        }
    }
}
