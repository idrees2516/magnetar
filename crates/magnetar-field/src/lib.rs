//! # magnetar-field
//!
//! Binary-field arithmetic for the Magnetar proof stack (paper §3).
//!
//! * `B = F_2` — bits, embedded canonically in `K`.
//! * `K = F_{2^128} = F_2[x] / (x^128 + x^7 + x^2 + x + 1)` — the GCM
//!   polynomial, in the *polynomial basis* `β_b = x^b`.
//!
//! The polynomial basis is the packing basis of BinarySpartan (paper §4.3):
//! a field element's 128-bit representation *is* the β-coefficient vector, so
//! bit extraction is a plain bit read. This is what makes the `φ_c` functional
//! of ring switching (paper §5.2) verifier-efficient:
//! `φ_c(P) = Σ_b c_b · bit_b(P)` costs 128 field ops, not a linear system.
//!
//! All operations are constant-time (no secret-dependent branches); the
//! multiplication fast path uses `PCLMULQDQ` with a portable fallback.

#![deny(unsafe_code)]
// Proof-system style: loops indexed over Boolean-cube coordinates and
// multi-input reductions are the domain's normal shape.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![forbid(clippy::undocumented_unsafe_blocks)]
// The two exceptions: `gf128::clmul` and `avx` contain guarded `unsafe`
// intrinsics. They are dedicated modules with `#[allow(unsafe_code)]` and
// audited SAFETY comments; every other module in the workspace stays
// `forbid`-clean.
#![cfg_attr(target_arch = "x86_64", allow(unsafe_code))]

pub mod avx;
mod eq;
mod gf128;
mod multilinear;
mod pack;
pub mod parallel;

pub use eq::{eq_char2, eq_table, EqBits};
pub use gf128::{inv, is_bit, mul, mul_bit, pow, square, Gf128, ONE, ZERO};
pub use multilinear::{bind_var, eval_mle, fold_pairs, Multilinear};
pub use pack::{
    c_fold_table, eq_coefficients, pack_bits, pack_bytes_le, pack_table, phi_c, unpack_bits,
    PHIBITS,
};
pub use parallel::{
    par_bind_var, par_bind_var_bit, par_dot_bit_xor, par_dot_xor, par_gather_bit, par_gather_bit_at,
    PAR_CHUNK, PAR_THRESHOLD,
};

/// A small helper trait for field elements used across the stack.
pub trait Field: Copy + PartialEq + std::fmt::Debug {
    const ZERO: Self;
    const ONE: Self;
    fn add(self, rhs: Self) -> Self;
    fn double(self) -> Self;
}

impl Field for Gf128 {
    const ZERO: Gf128 = Gf128(0);
    const ONE: Gf128 = Gf128(1);
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline(always)]
    fn double(self) -> Self {
        self // characteristic two
    }
}

/// Number of variables of a table of `len` entries (panic-free for len==1 → 0).
#[inline]
pub fn num_vars_for(len: usize) -> usize {
    debug_assert!(
        len.is_power_of_two() || len == 1,
        "table length must be a power of two"
    );
    len.trailing_zeros() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_axioms() {
        use rand::RngExt;
        let mut rng = rand::rng();
        for _ in 0..64 {
            let a: Gf128 = rng.random();
            let b: Gf128 = rng.random();
            let c: Gf128 = rng.random();
            assert_eq!(a + a, ZERO);
            assert_eq!(a + ZERO, a);
            assert_eq!(mul(a, b), mul(b, a), "commutativity");
            assert_eq!(mul(mul(a, b), c), mul(a, mul(b, c)), "associativity");
            assert_eq!(mul(a, b + c), mul(a, b) + mul(a, c), "distributivity");
            if a != ZERO {
                assert_eq!(mul(a, inv(a)), ONE, "inverse");
            }
            assert_eq!(mul(a, ONE), a);
            assert_eq!(square(a), mul(a, a));
        }
    }

    #[test]
    fn frobenius_is_multiplicative() {
        use rand::RngExt;
        let mut rng = rand::rng();
        for _ in 0..16 {
            let a: Gf128 = rng.random();
            let b: Gf128 = rng.random();
            // (a*b)^2 == a^2 * b^2 (Frobenius)
            assert_eq!(square(mul(a, b)), mul(square(a), square(b)));
            // (a+b)^2 == a^2 + b^2 (char 2)
            assert_eq!(square(a + b), square(a) + square(b));
        }
    }

    #[test]
    fn reduction_matches_slow_reference() {
        // Reference: Russian-peasant multiplication over the full polynomial.
        fn slow_mul(mut a: u128, mut b: u128) -> u128 {
            let mut p = 0u128;
            while b != 0 {
                if b & 1 == 1 {
                    p ^= a;
                }
                let carry = a >> 127;
                a <<= 1;
                a ^= carry.wrapping_neg() & 0x87;
                b >>= 1;
            }
            p
        }
        use rand::RngExt;
        let mut rng = rand::rng();
        for _ in 0..256 {
            let a: Gf128 = rng.random();
            let b: Gf128 = rng.random();
            assert_eq!(mul(a, b).0, slow_mul(a.0, b.0));
        }
        // Edge cases.
        assert_eq!(mul(Gf128(1), Gf128(1)), ONE);
        assert_eq!(mul(Gf128(0x87), ONE).0, 0x87);
        assert_eq!(mul(Gf128(1u128 << 127), ONE).0, 1u128 << 127);
        // x^128 == x^7+x^2+x+1 = 0x87.
        let x127 = Gf128(1u128 << 127);
        assert_eq!(mul(x127, Gf128(2)).0, 0x87, "x^128 must reduce to 0x87");
    }

    #[test]
    fn packing_roundtrip_and_phi() {
        use rand::RngExt;
        let mut rng = rand::rng();
        for _ in 0..32 {
            let bits: [bool; 128] = std::array::from_fn(|_| rng.random());
            let p = pack_bits(&bits);
            let back = unpack_bits(p);
            assert_eq!(&back, &bits);
            // phi_c = dot product of c with the bits.
            let c: [Gf128; 128] = std::array::from_fn(|_| rng.random());
            let expect: Gf128 = c
                .iter()
                .zip(bits.iter())
                .map(|(ci, bi)| if *bi { *ci } else { ZERO })
                .fold(ZERO, |x, y| x + y);
            assert_eq!(phi_c(p, &c), expect);
            // phi with c = basis selector returns the bit.
            for b in [0usize, 1, 63, 64, 127] {
                let mut sel = [ZERO; 128];
                sel[b] = ONE;
                assert_eq!(phi_c(p, &sel), Gf128(bits[b] as u128));
            }
        }
    }
}
