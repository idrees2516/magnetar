//! Bit packing over the polynomial basis (paper §4.3, step 1).
//!
//! `P(v) = Σ_{b<128} β_b · w(u_b, v)` with `β_b = x^b`: in the polynomial
//! basis the packed element's `u128` representation has bit `b` equal to
//! `w(u_b, v)`. Packing and unpacking are therefore trivial, and the
//! `φ_c` functional of ring switching (paper §5.2 / §4.3) is a plain
//! dot product with the bits:
//!
//! `φ_c(P) = Σ_b c_b · bit_b(P)` — 128 field additions, verifier-side.
//!
//! The bit-index convention matches the multilinear layer: within a packed
//! element, bit `b` is variable `b` (little-endian) of the 7 packing
//! variables.

use crate::gf128::{mul, Gf128, ONE, ZERO};

/// Number of bits packed per field element (2^7 = 128 → 7 packing vars).
pub const PHIBITS: usize = 7;

/// Pack 128 bits (bit `b` of the slice is coefficient of `x^b`).
#[inline]
pub fn pack_bits(bits: &[bool; 128]) -> Gf128 {
    let mut v = 0u128;
    for (b, &bit) in bits.iter().enumerate() {
        v |= (bit as u128) << b;
    }
    Gf128(v)
}

/// Unpack a field element into its 128 polynomial-basis bits.
#[inline]
pub fn unpack_bits(p: Gf128) -> [bool; 128] {
    let mut bits = [false; 128];
    for (b, slot) in bits.iter_mut().enumerate() {
        *slot = (p.0 >> b) & 1 == 1;
    }
    bits
}

/// A tabulated `φ_c` for one fixed coefficient vector `c`: 32 nibble
/// tables (`T[j][v] = Σ_{b ∈ v} c[4j + b]`), so one application is 32
/// table lookups + 31 additions — the per-bit branch loop is ~100x
/// slower on the packed-table scales (the ring switch's A-factor
/// materialization runs it over every packed word).
pub struct PhiTab {
    tabs: [[Gf128; 16]; 32],
}

impl PhiTab {
    pub fn build(c: &[Gf128; 128]) -> Self {
        let mut tabs = [[ZERO; 16]; 32];
        for j in 0..32 {
            for v in 0..16usize {
                let mut acc = ZERO;
                for b in 0..4usize {
                    if (v >> b) & 1 == 1 {
                        acc += c[4 * j + b];
                    }
                }
                tabs[j][v] = acc;
            }
        }
        PhiTab { tabs }
    }

    /// `φ_c(p)` via the nibble tables.
    #[inline]
    pub fn apply(&self, p: Gf128) -> Gf128 {
        let lo = p.0 as u64;
        let hi = (p.0 >> 64) as u64;
        let mut acc = ZERO;
        for j in 0..16 {
            acc += self.tabs[j][(lo >> (4 * j)) as usize & 15];
        }
        for j in 0..16 {
            acc += self.tabs[16 + j][(hi >> (4 * j)) as usize & 15];
        }
        acc
    }
}

/// The ring-switching functional: `φ_c(P) = Σ_b c_b · bit_b(P)`.
///
/// Given a packed element `P` and coefficients `c` (typically `c_b =
/// eq(r_B, b)` from an equality table), this is the exact F₂-linear
/// combination of the packed bits — computable by anyone who knows `P`.
#[inline]
pub fn phi_c(p: Gf128, c: &[Gf128; 128]) -> Gf128 {
    let mut acc = ZERO;
    for (b, &cb) in c.iter().enumerate() {
        if (p.0 >> b) & 1 == 1 {
            acc += cb;
        }
    }
    acc
}

/// Pack the *bit-major* table `w: {0,1}^n × {0,1}^7 → B` (row `u` of 128
/// bits, little-endian bit = packing variable) into `2^n` field elements:
/// `P(u) = Σ_b β_b w(u, b)`.
pub fn pack_table(w: &[u8]) -> Vec<Gf128>
where
    [u8]: PartialEq,
{
    debug_assert_eq!(w.len() % 128, 0);
    let n = w.len() / 128;
    (0..n)
        .map(|u| pack_bytes_le(&w[u * 128..(u + 1) * 128]))
        .collect()
}

/// Pack one 128-byte 0/1 slice (little-endian bit order).
#[inline]
pub fn pack_bytes_le(bytes: &[u8]) -> Gf128 {
    debug_assert_eq!(bytes.len(), 128);
    let mut v = 0u128;
    for (b, &x) in bytes.iter().enumerate() {
        debug_assert!(x <= 1);
        v |= (x as u128) << b;
    }
    Gf128(v)
}

/// Batched eq-weighted combination helper used by MosaicPCS: given the
/// packed table `P`, the c-folded table is `C(u) = φ_c(P(u))` for
/// `c_b = eq(r_B, b)`.
pub fn c_fold_table(packed: &[Gf128], c: &[Gf128; 128]) -> Vec<Gf128> {
    packed.iter().map(|&p| phi_c(p, c)).collect()
}

/// `eq(r_B, b)` coefficients over all 128 Boolean bit-points.
pub fn eq_coefficients(r_bit: &[Gf128; 7]) -> [Gf128; 128] {
    let mut c = [ZERO; 128];
    for b in 0..128usize {
        let mut acc = ONE;
        for (j, &r) in r_bit.iter().enumerate() {
            let bj = (b >> j) & 1 == 1;
            acc = mul(acc, if bj { r } else { ONE + r });
        }
        c[b] = acc;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        use rand::RngExt;
        let mut rng = rand::rng();
        for _ in 0..32 {
            let bits: [bool; 128] = std::array::from_fn(|_| rng.random());
            let p = pack_bits(&bits);
            assert_eq!(&unpack_bits(p)[..], &bits[..]);
            let bytes: Vec<u8> = bits.iter().map(|&b| b as u8).collect();
            assert_eq!(pack_bytes_le(&bytes), p);
        }
    }

    #[test]
    fn phi_c_is_linear_in_bits() {
        use rand::RngExt;
        let mut rng = rand::rng();
        for _ in 0..16 {
            let bits: [bool; 128] = std::array::from_fn(|_| rng.random());
            let p = pack_bits(&bits);
            let c: [Gf128; 128] = std::array::from_fn(|_| rng.random());
            let mut want = ZERO;
            for (b, &bit) in bits.iter().enumerate() {
                if bit {
                    want += c[b];
                }
            }
            assert_eq!(phi_c(p, &c), want);
        }
    }

    #[test]
    fn eq_coefficients_match_eq_table() {
        use crate::eq::eq_table;
        use rand::RngExt;
        let mut rng = rand::rng();
        let r: [Gf128; 7] = std::array::from_fn(|_| rng.random());
        let c = eq_coefficients(&r);
        for b in 0..128usize {
            let bits: Vec<bool> = (0..7).map(|j| (b >> j) & 1 == 1).collect();
            assert_eq!(c[b], eq_table(&r, &bits), "b = {b}");
        }
    }

    #[test]
    fn c_fold_matches_manual() {
        use rand::RngExt;
        let mut rng = rand::rng();
        let rows = 12;
        let w: Vec<bool> = (0..rows * 128).map(|_| rng.random()).collect();
        let bytes: Vec<u8> = w.iter().map(|&b| b as u8).collect();
        let packed = pack_table(&bytes);
        let c: [Gf128; 128] = std::array::from_fn(|_| rng.random());
        let folded = c_fold_table(&packed, &c);
        for u in 0..rows {
            let want: Gf128 = (0..128)
                .map(|b| if w[u * 128 + b] { c[b] } else { ZERO })
                .fold(ZERO, |a, x| a + x);
            assert_eq!(folded[u], want, "row {u}");
        }
    }
}
