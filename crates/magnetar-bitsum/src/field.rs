//! GF(2^128) in the *tower basis*, the field family used by binary-field zkVMs
//! (Binius, Flock, Magnetar's F2/F2^128 dual-mode prover).
//!
//! # Tower construction (computationally verified)
//!
//! The field is built as a chain of quadratic extensions
//!
//! ```text
//! F2 = F_{2^{2^0}}  <  F4  <  F16  <  F256  <  F65536  <  F_{2^32}  <  F_{2^64}  <  F_{2^128}
//! ```
//!
//! where the level-`k` extension adds the generator `g_k`:
//!
//! ```text
//! F_{2^{2^{k+1}}} = F_{2^{2^k}}[g_k] / (g_k^2 + g_k + NUS[k]),   NUS[k] in F_{2^{2^k}}.
//! ```
//!
//! An element is a 128-bit *coordinate vector*: bit `m` is the coefficient of
//! the basis element `B_m = prod_{j in m} g_j` (the product of the generators
//! whose bits are set in `m`). Sub-field `F_{2^{2^k}}` = elements with
//! coordinates in `[0, 2^k)`.
//!
//! The irreducibility criterion for `x^2 + x + nu` over an even-characteristic
//! field `F_q` is `Tr_{F_q/F2}(nu) = 1`. A exhaustive basis-trace search
//! (`scripts/tower_search.rs` in the repository) proves that the sparsest
//! valid constants are, at every level, the full generator product
//!
//! ```text
//! NUS[k] = B_{2^k - 1} = g_{k-1} * g_{k-2} * ... * g_0     (single coordinate!)
//! ```
//!
//! i.e. `NUS[k] = 1 << (2^k - 1)` as a coordinate vector. Single-coordinate
//! constants keep the multiplication tree cheap (see the `tree` module).
//!
//! # Verification performed on these constants
//!
//! * every `NUS[k]` has absolute trace 1 (irreducibility at each level);
//! * field axioms on 300 random triples (commutativity, associativity,
//!   distributivity, identities);
//! * `a^(2^128 - 1) == 1` for random `a != 0` (multiplicative group order,
//!   which fails if *any* level were reducible);
//! * the `tree` module's multiplier (the exact algorithm the GPU kernel runs)
//!   agrees with the scalar multiplier on random inputs.

#![forbid(unsafe_code)]

/// Tower constants: `NUS[k] = 1 << (2^k - 1)`; `g_k^2 = g_k + NUS[k]`.
///
/// The tower constants were found by an exhaustive basis-trace search and
/// are re-verified by the tests below at build time; see the
/// [module](self) docs.
pub const TOWER_NUS: [u128; 8] = [
    0x0000_0000_0000_0000_0000_0000_0000_0001,
    0x0000_0000_0000_0000_0000_0000_0000_0002,
    0x0000_0000_0000_0000_0000_0000_0000_0008,
    0x0000_0000_0000_0000_0000_0000_0000_0080,
    0x00000000000000000000000008000,
    0x0000_0000_0000_0000_0000_0000_8000_0000,
    0x0000_0000_0000_0000_8000_0000_0000_0000,
    0x8000_0000_0000_0000_0000_0000_0000_0000,
];

/// Multiplication table for the 8-coordinate sub-field F_{2^{2^3}} = F_256,
/// used to short-circuit the recursion (64 KiB, built once, lazily).
static F256_MUL_TABLE: std::sync::OnceLock<Box<[[u8; 256]; 256]>> = std::sync::OnceLock::new();

fn f256_table() -> &'static [[u8; 256]; 256] {
    F256_MUL_TABLE.get_or_init(|| {
        let mut table: Box<[[u8; 256]; 256]> = Box::new(core::array::from_fn(|_| [0u8; 256]));
        for a in 0usize..256 {
            for b in 0usize..256 {
                table[a][b] = mul_slow(a as u128, b as u128, 3, &TOWER_NUS) as u8;
            }
        }
        table
    })
}

/// An element of GF(2^128), stored as its 128-bit tower coordinate vector.
///
/// Addition is XOR of coordinate vectors. See the [module](self) docs for the
/// basis and the tower constants.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct F128(pub u128);

impl F128 {
    /// The additive / multiplicative identity pair.
    pub const ZERO: F128 = F128(0);
    /// The multiplicative identity (empty generator product).
    pub const ONE: F128 = F128(1);

    /// Little-endian byte decoding.
    #[must_use]
    pub fn from_le_bytes(bytes: [u8; 16]) -> Self {
        F128(u128::from_le_bytes(bytes))
    }

    /// Little-endian byte encoding (used by the transcript).
    #[must_use]
    pub fn to_le_bytes(self) -> [u8; 16] {
        self.0.to_le_bytes()
    }

    /// Embeds `x` as the sub-field element with coordinate vector `x`.
    #[must_use]
    pub fn from_u64(x: u64) -> Self {
        F128(x as u128)
    }

    /// Returns true if this is the additive identity.
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Squaring.
    #[must_use]
    pub fn square(self) -> F128 {
        self * self
    }

    /// Exponentiation by square-and-multiply.
    #[must_use]
    pub fn pow(self, mut exp: u128) -> F128 {
        let mut result = F128::ONE;
        let mut base = self;
        while exp != 0 {
            if exp & 1 == 1 {
                result *= base;
            }
            base = base.square();
            exp >>= 1;
        }
        result
    }

    /// Multiplicative inverse (None for zero). `a^(2^128 - 2)`.
    #[must_use]
    pub fn invert(self) -> Option<F128> {
        if self.is_zero() {
            None
        } else {
            Some(self.pow(u128::MAX - 1)) // 2^128 - 2
        }
    }

    /// Deterministic pseudo-random element for tests and benches.
    #[must_use]
    pub fn random<R: crate::Rng>(rng: &mut R) -> Self {
        F128(rng.next_u128())
    }
}

// Characteristic two: field addition is coordinate XOR and subtraction
// equals addition, so the `^` / `+` bodies below are definitions, not
// accidents.
#[allow(clippy::suspicious_arithmetic_impl)]
impl core::ops::Add for F128 {
    type Output = F128;
    fn add(self, rhs: F128) -> F128 {
        F128(self.0 ^ rhs.0)
    }
}
#[allow(clippy::suspicious_op_assign_impl)]
impl core::ops::AddAssign for F128 {
    fn add_assign(&mut self, rhs: F128) {
        self.0 ^= rhs.0;
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl core::ops::Sub for F128 {
    type Output = F128;
    fn sub(self, rhs: F128) -> F128 {
        self + rhs // char 2: subtraction is addition
    }
}
#[allow(clippy::suspicious_op_assign_impl)]
impl core::ops::SubAssign for F128 {
    fn sub_assign(&mut self, rhs: F128) {
        *self += rhs;
    }
}
impl core::ops::Mul for F128 {
    type Output = F128;
    /// Field multiplication (tower recursion, table-boosted below 16
    /// coordinates). Cost: ~2000 scalar ops after the table is built.
    fn mul(self, rhs: F128) -> F128 {
        F128(mul_fast(self.0, rhs.0))
    }
}
impl core::ops::MulAssign for F128 {
    fn mul_assign(&mut self, rhs: F128) {
        *self = *self * rhs;
    }
}
impl core::ops::Neg for F128 {
    type Output = F128;
    fn neg(self) -> F128 {
        self // char 2
    }
}

impl core::fmt::Debug for F128 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "F128(0x{:032x})", self.0)
    }
}
impl core::fmt::Display for F128 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// Reference tower multiplication (no tables). Coordinates in `[0, 2^k)`.
fn mul_slow(a: u128, b: u128, k: usize, nus: &[u128; 8]) -> u128 {
    if k == 0 {
        return a & b & 1;
    }
    let split: u32 = 1u32 << (k - 1);
    let half_mask: u128 = (1u128 << split) - 1;
    let a_lo = a & half_mask;
    let a_hi = a >> split;
    let b_lo = b & half_mask;
    let b_hi = b >> split;
    let m0 = mul_slow(a_lo, b_lo, k - 1, nus);
    let m1 = mul_slow(a_hi, b_hi, k - 1, nus);
    let m2 = mul_slow(a_lo ^ a_hi, b_lo ^ b_hi, k - 1, nus);
    let nu_m1 = mul_slow(nus[k - 1], m1, k - 1, nus);
    let out_lo = m0 ^ nu_m1;
    let out_hi = m2 ^ m0;
    out_lo | (out_hi << split)
}

/// Table-boosted full-size multiplication.
fn mul_fast(a: u128, b: u128) -> u128 {
    let table = f256_table();
    mul_table(a, b, 7, table)
}

fn mul_table(a: u128, b: u128, k: usize, table: &[[u8; 256]; 256]) -> u128 {
    if k == 3 {
        return table[(a & 0xFF) as usize][(b & 0xFF) as usize] as u128;
    }
    let split: u32 = 1u32 << (k - 1);
    let half_mask: u128 = (1u128 << split) - 1;
    let a_lo = a & half_mask;
    let a_hi = a >> split;
    let b_lo = b & half_mask;
    let b_hi = b >> split;
    let m0 = mul_table(a_lo, b_lo, k - 1, table);
    let m1 = mul_table(a_hi, b_hi, k - 1, table);
    let m2 = mul_table(a_lo ^ a_hi, b_lo ^ b_hi, k - 1, table);
    let nu_m1 = mul_table(TOWER_NUS[k - 1], m1, k - 1, table);
    let out_lo = m0 ^ nu_m1;
    let out_hi = m2 ^ m0;
    out_lo | (out_hi << split)
}

/// Absolute trace `Tr_{F_{2^{2^k}}/F2}(z)` (z in coordinates `[0, 2^k)`).
///
/// Exposed for tower re-verification: `trace(TOWER_NUS[k], k) == 1` for
/// every k is exactly the irreducibility condition of each tower level.
#[must_use]
pub fn trace(z: u128, k: usize) -> u32 {
    let mut acc = z;
    let mut cur = z;
    for _ in 1..(1usize << k) {
        cur = mul_fast(cur, cur);
        acc ^= cur;
    }
    debug_assert!(acc >> 1 == 0);
    (acc & 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    struct SplitMix(u64);
    impl crate::Rng for SplitMix {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x09E3_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn rng() -> SplitMix {
        SplitMix(0x6d61_676e_6574_6172)
    }

    #[test]
    fn tower_constants_have_trace_one() {
        // Irreducibility of every level, re-checked at runtime.
        for (k, &nu) in TOWER_NUS.iter().enumerate() {
            assert_eq!(
                trace(nu, k),
                1,
                "NUS[{k}] must have trace 1 (level irreducible)"
            );
        }
    }

    #[test]
    fn nus_are_single_coordinates() {
        for (k, &nu) in TOWER_NUS.iter().enumerate() {
            assert_eq!(nu.count_ones(), 1, "NUS[{k}] sparse");
            assert_eq!(nu.trailing_zeros() as usize, (1 << k) - 1);
        }
    }

    #[test]
    fn field_axioms() {
        let mut r = rng();
        for _ in 0..200 {
            let a = F128::random(&mut r);
            let b = F128::random(&mut r);
            let c = F128::random(&mut r);
            assert_eq!(a + b, b + a);
            assert_eq!((a + b) + c, a + (b + c));
            assert_eq!(a * b, b * a);
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
            assert_eq!(a + F128::ZERO, a);
            assert_eq!(a * F128::ONE, a);
            assert_eq!(a * F128::ZERO, F128::ZERO);
            assert_eq!(a - b, a + b);
        }
    }

    #[test]
    fn table_matches_reference() {
        let mut r = rng();
        for _ in 0..200 {
            let a: u128 = r.next_u128();
            let b: u128 = r.next_u128();
            assert_eq!(mul_fast(a, b), mul_slow(a, b, 7, &TOWER_NUS));
        }
    }

    #[test]
    fn group_order() {
        // a^(2^128 - 1) == 1 for a != 0 fails if any tower level is reducible.
        let mut r = rng();
        for _ in 0..2 {
            let mut a = F128::ZERO;
            while a.is_zero() {
                a = F128::random(&mut r);
            }
            assert_eq!(a.pow(u128::MAX), F128::ONE);
        }
    }

    #[test]
    fn inverses() {
        let mut r = rng();
        for _ in 0..20 {
            let mut a = F128::ZERO;
            while a.is_zero() {
                a = F128::random(&mut r);
            }
            let inv = a.invert().expect("nonzero");
            assert_eq!(a * inv, F128::ONE);
        }
    }

    #[test]
    fn frobenius_is_square() {
        let mut r = rng();
        for _ in 0..20 {
            let a = F128::random(&mut r);
            // squaring is linear in char 2: (a+b)^2 = a^2 + b^2
            let b = F128::random(&mut r);
            assert_eq!((a + b).square(), a.square() + b.square());
        }
    }
}
