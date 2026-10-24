//! `K = F_{2^128} = F_2[x] / (x^128 + x^7 + x^2 + x + 1)`.
//!
//! Representation: a single `u128`; bit `i` is the coefficient of `x^i`.
//! Multiplication: carry-less 128×128 → 256-bit product (Karatsuba over two
//! `PCLMULQDQ` limbs, portable fallback otherwise), then folded through
//! `x^128 ≡ x^7+x^2+x+1` (GCM-style two-step reduction).
//!
//! Bit extraction is free in this basis: the bits of the `u128` *are* the
//! polynomial coefficients, which is the property ring switching exploits.

// The workspace's single `unsafe` exception (the crate denies
// `unsafe_code`): guarded, audited clmul intrinsics. This inner attribute
// must sit at the top of the module.
#![cfg_attr(target_arch = "x86_64", allow(unsafe_code))]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm_clmulepi64_si128, _mm_extract_epi64, _mm_set_epi64x};

/// The additive identity.
pub const ZERO: Gf128 = Gf128(0);
/// The multiplicative identity.
pub const ONE: Gf128 = Gf128(1);

/// A `F_{2^128}` element.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Gf128(pub u128);

impl std::fmt::Debug for Gf128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "K({:#034x})", self.0)
    }
}

// Addition in characteristic two is XOR.
#[allow(clippy::suspicious_arithmetic_impl)]
impl std::iter::Sum for Gf128 {
    fn sum<I: Iterator<Item = Gf128>>(iter: I) -> Gf128 {
        // Characteristic-two addition is XOR — order-independent, so this
        // is usable inside parallel reductions (docs/PERFORMANCE.md R1).
        iter.fold(Gf128(0), |a, b| a + b)
    }
}

impl<'a> std::iter::Sum<&'a Gf128> for Gf128 {
    fn sum<I: Iterator<Item = &'a Gf128>>(iter: I) -> Gf128 {
        iter.fold(Gf128(0), |a, b| a + *b)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)] // XOR is the characteristic-two +
impl std::ops::Add for Gf128 {
    type Output = Gf128;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Gf128(self.0 ^ rhs.0)
    }
}
#[allow(clippy::suspicious_op_assign_impl)] // XOR is the characteristic-two +=
impl std::ops::AddAssign for Gf128 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
// Subtraction equals addition in characteristic two.
#[allow(clippy::suspicious_arithmetic_impl)]
impl std::ops::Sub for Gf128 {
    type Output = Gf128;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl std::ops::Neg for Gf128 {
    type Output = Gf128;
    #[inline(always)]
    fn neg(self) -> Self {
        self
    }
}

#[inline(always)]
fn lo64(x: Gf128) -> u64 {
    x.0 as u64
}
#[inline(always)]
fn hi64(x: Gf128) -> u64 {
    (x.0 >> 64) as u64
}

#[inline(always)]
fn fast_path_available() -> bool {
    use std::sync::OnceLock;
    static FAST: OnceLock<bool> = OnceLock::new();
    *FAST.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            // `_mm_extract_epi64` needs SSE4.1; every PCLMULQDQ CPU in
            // practice ships SSE4.1, but verify both to stay correct.
            std::is_x86_feature_detected!("pclmulqdq") && std::is_x86_feature_detected!("sse4.1")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// 64×64 carry-less multiply → 128-bit.
#[inline(always)]
fn clmul64(a: u64, b: u64) -> u128 {
    #[cfg(target_arch = "x86_64")]
    if fast_path_available() {
        // SAFETY: guarded by cached `is_x86_feature_detected!` checks for
        // `pclmulqdq` and `sse4.1` on x86-64; operands are plain integers
        // and the intrinsic has no preconditions beyond availability.
        #[allow(unsafe_code)]
        unsafe {
            let m = _mm_clmulepi64_si128(
                _mm_set_epi64x(0, a as i64),
                _mm_set_epi64x(0, b as i64),
                0x00,
            );
            (_mm_extract_epi64(m, 0) as u64 as u128)
                | ((_mm_extract_epi64(m, 1) as u64 as u128) << 64)
        }
    } else {
        clmul64_portable(a, b)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        clmul64_portable(a, b)
    }
}

/// Portable branchless 64×64 carry-less multiply (Russian peasant).
#[inline]
fn clmul64_portable(mut a: u64, mut b: u64) -> u128 {
    let mut p: u128 = 0;
    for _ in 0..64 {
        p ^= (a as u128) & ((b & 1).wrapping_neg() as u128);
        a = a.wrapping_shl(1);
        b >>= 1;
    }
    p
}

/// Carry-less 128×128 → 256-bit product as four little-endian `u64` limbs
/// (Karatsuba: 3 half-limb multiplications).
#[inline]
fn clmul128(a: Gf128, b: Gf128) -> [u64; 4] {
    let a0 = lo64(a);
    let a1 = hi64(a);
    let b0 = lo64(b);
    let b1 = hi64(b);
    let lo = clmul64(a0, b0);
    let hi = clmul64(a1, b1);
    let mid = clmul64(a0 ^ a1, b0 ^ b1) ^ lo ^ hi;
    let lo_l = lo as u64;
    let lo_h = (lo >> 64) as u64;
    let mid_l = mid as u64;
    let mid_h = (mid >> 64) as u64;
    let hi_l = hi as u64;
    let hi_h = (hi >> 64) as u64;
    [lo_l, lo_h ^ mid_l, mid_h ^ hi_l, hi_h]
}

/// Reduce a 256-bit carry-less product mod `x^128+x^7+x^2+x+1`.
///
/// `v = t0 + t1·x^64 + t2·x^128 + t3·x^192`; with `R = x^7+x^2+x+1`,
/// `x^128 ≡ R` gives `v ≡ L + H·R` where `H = t2 + t3·x^64` and
/// `L = t0 + t1·x^64`. `H·R` has degree ≤ 134, so split once more:
/// `H·R = Q2·x^128 + R2` with `deg Q2 ≤ 6`, and `Q2·x^128 ≡ Q2·R` (degree
/// ≤ 13, no further overflow).
#[inline]
fn reduce256(t: [u64; 4]) -> Gf128 {
    let lo = (t[0] as u128) | ((t[1] as u128) << 64);
    let hi = (t[2] as u128) | ((t[3] as u128) << 64);
    // HR = hi · 0x87 = (hi<<7) ^ (hi<<2) ^ (hi<<1) ^ hi, over three limbs.
    let a0 = hi as u64;
    let a1 = (hi >> 64) as u64;
    let s7_0 = a0 << 7;
    let s7_1 = (a1 << 7) | (a0 >> 57);
    let s7_2 = a1 >> 57;
    let s2_0 = a0 << 2;
    let s2_1 = (a1 << 2) | (a0 >> 62);
    let s2_2 = a1 >> 62;
    let s1_0 = a0 << 1;
    let s1_1 = (a1 << 1) | (a0 >> 63);
    let s1_2 = a1 >> 63;
    let r0 = s7_0 ^ s2_0 ^ s1_0 ^ a0;
    let r1 = s7_1 ^ s2_1 ^ s1_1 ^ a1;
    let q2 = (s7_2 ^ s2_2 ^ s1_2) as u128; // deg ≤ 6
    let r2 = (r0 as u128) | ((r1 as u128) << 64);
    let q2r = (q2 << 7) ^ (q2 << 2) ^ (q2 << 1) ^ q2;
    Gf128(lo ^ r2 ^ q2r)
}

/// Multiply two field elements (constant time, branchless).
#[inline]
pub fn mul(a: Gf128, b: Gf128) -> Gf128 {
    reduce256(clmul128(a, b))
}

/// Multiply by a **bit-valued** factor (`b ∈ {0, 1}`, the small-value
/// setting of the sum-check papers): `r · b` is `r` or `0` — a branchless
/// select, no CLMUL. Caller guarantees `b.0 ∈ {0, 1}` (the engine detects
/// bit-valued tables once and uses this path while they last).
#[inline]
pub fn mul_bit(r: Gf128, b: Gf128) -> Gf128 {
    debug_assert!(b.0 <= 1, "mul_bit requires a bit-valued factor");
    Gf128(r.0 & b.0.wrapping_neg())
}

/// `true` iff the element is `0` or `1` (bit-valued).
#[inline]
pub fn is_bit(a: Gf128) -> bool {
    a.0 <= 1
}

/// Frobenius: `a → a²` (the paper's `s_h = r_h²` in BinaryTwist §4.4).
#[inline]
pub fn square(a: Gf128) -> Gf128 {
    mul(a, a)
}

/// Multiplicative inverse via square-and-multiply with the fixed public
/// exponent `2^128 − 2` (no secret-dependent control flow of concern: the
/// exponent is a public constant).
pub fn inv(a: Gf128) -> Gf128 {
    assert_ne!(a, ZERO, "inverse of zero");
    let e: u128 = u128::MAX - 1; // 2^128 - 2
    let mut result = ONE;
    for i in (0..128).rev() {
        result = square(result);
        if (e >> i) & 1 == 1 {
            result = mul(result, a);
        }
    }
    result
}

/// Exponentiation with a public exponent.
pub fn pow(a: Gf128, k: u128) -> Gf128 {
    let mut result = ONE;
    let mut base = a;
    for i in 0..128 {
        if (k >> i) & 1 == 1 {
            result = mul(result, base);
        }
        base = square(base);
    }
    result
}

#[cfg(test)]
impl rand::distr::Distribution<Gf128> for rand::distr::StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Gf128 {
        Gf128(rand::RngExt::random(rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slow_mul(mut a: u128, mut b: u128) -> u128 {
        // Russian peasant with branchless fold; reference implementation.
        let mut p = 0u128;
        for _ in 0..128 {
            p ^= a & ((b & 1).wrapping_neg());
            let carry = a >> 127;
            a <<= 1;
            a ^= carry.wrapping_neg() & 0x87;
            b >>= 1;
        }
        p
    }

    #[test]
    fn mul_matches_reference() {
        use rand::RngExt;
        let mut rng = rand::rng();
        for _ in 0..512 {
            let a: Gf128 = rng.random();
            let b: Gf128 = rng.random();
            assert_eq!(mul(a, b).0, slow_mul(a.0, b.0));
        }
        // Directed: x^128 ≡ 0x87.
        let x127 = Gf128(1u128 << 127);
        assert_eq!(mul(x127, Gf128(2)).0, 0x87);
        assert_eq!(mul(Gf128(0x87), x127).0, mul(x127, Gf128(0x87)).0);
        // (x^127)^2 = x^254 ≡ x^126·(x^128) ≡ x^126·0x87 ... spot check via slow.
        assert_eq!(square(x127).0, slow_mul(x127.0, x127.0));
    }

    #[test]
    fn clmul_portable_reference() {
        for (a, b) in [
            (0u64, 0u64),
            (1, 1),
            (u64::MAX, 2),
            (0xDEAD_BEEF_CAFE_F00Du64, 0x1234_5678_9ABC_DEF0u64),
        ] {
            let want = {
                let mut p = 0u128;
                let (mut x, mut y) = (a, b);
                for _ in 0..64 {
                    p ^= (x as u128) & ((y & 1).wrapping_neg() as u128);
                    x = x.wrapping_shl(1);
                    y >>= 1;
                }
                p
            };
            assert_eq!(clmul64_portable(a, b), want);
        }
    }

    #[test]
    fn karatsuba_matches_schoolbook() {
        fn schoolbook(a: Gf128, b: Gf128) -> [u64; 4] {
            let mut t = [0u64; 4];
            for i in 0..128 {
                if (b.0 >> i) & 1 == 1 {
                    let lo = if i >= 128 { 0 } else { a.0 << i };
                    let hi = if i == 0 { 0 } else { a.0 >> (128 - i) };
                    t[0] ^= lo as u64;
                    t[1] ^= (lo >> 64) as u64;
                    t[2] ^= hi as u64;
                    t[3] ^= (hi >> 64) as u64;
                }
            }
            t
        }
        for (a, b) in [
            (Gf128(0), Gf128(12345)),
            (Gf128(1), Gf128(u128::MAX)),
            (Gf128(0x87), Gf128(1u128 << 127)),
            (Gf128(u128::MAX), Gf128(u128::MAX)),
        ] {
            assert_eq!(clmul128(a, b), schoolbook(a, b));
        }
    }

    #[test]
    fn inverse_works() {
        use rand::RngExt;
        let mut rng = rand::rng();
        for _ in 0..16 {
            let a: Gf128 = rng.random();
            if a == ZERO {
                continue;
            }
            assert_eq!(mul(a, inv(a)), ONE);
        }
    }
}
