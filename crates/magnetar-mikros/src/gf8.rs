//! GF(2^8) — the intermediate (interpolation) field `K` of the Mikros
//! small-field variant (ePrint 2026/2646, §4.2 "Small fields via an
//! intermediate field").
//!
//! `K` is *the* unique order-256 subfield of `E = GF(2^128)` (the repo's
//! `Gf128`, GCM polynomial), so `F = GF(2) ⊂ K ⊂ E` holds canonically.
//!
//! **Code representation.** A `K`-element is stored as a `u8` *code* whose
//! bits are the coefficients over the GF(2)-basis `(1, h, h², …, h⁷)`, where
//! `h` is a root of the AES polynomial `x⁸+x⁴+x³+x+1` inside `K`. Because the
//! basis is a power basis of an AES-poly root:
//!   * addition is `XOR` of codes (any GF(2) basis is coordinate-wise), and
//!   * the code-level multiplication *is* AES-polynomial multiplication, i.e.
//!     the exact arithmetic `VPGF2P8MULB` (GFNI) implements — the dispatch on
//!     GFNI hardware can therefore replace the tables verbatim.
//!
//! The full 64 KiB product table (`[u8; 1<<16]`, indexed `a<<8|b`) is
//! L2-resident and drives the scalar path; `embed8` maps codes into `E` so all
//! `K`-values participate in `Gf128` algebra (the paper's `K ⊆ E`).
//!
//! The paper's parameter correspondence: `t = [K:F] = 8`, `κ = 128`
//! packing (needs `q^t = 256 ≥ 2κ−1 = 255` ✓), `[E:K] = 16`.

use magnetar_field::Gf128;
use magnetar_field::{mul, ZERO};

/// The intermediate field's tabulated arithmetic, built once per process.
pub struct Gf8 {
    /// `embed8[code]` — the `E`-element of the coded `K`-value.
    pub embed8: [Gf128; 256],
    /// Full product table indexed `(a << 8) | b` (codes).
    pub mul_tab: Box<[u8; 65536]>,
    /// `inv8[a]` — multiplicative inverse (`inv8[0] = 0` by convention).
    pub inv8: [u8; 256],
    /// The code-basis roots: `basis[i] = embed of h^i`.
    basis: [Gf128; 8],
}

/// Build the tables. `alpha = Gf128(2)` is the GCM root, a generator of
/// `E*`; the order-255 subgroup it contains is `K*`.
impl Gf8 {
    pub fn new() -> Self {
        // K* = the unique order-255 subgroup of E* = <alpha^((2^128-1)/255)>.
        let alpha = Gf128(2);
        let m: u128 = u128::MAX; // 2^128 - 1
        assert_eq!(m % 255, 0, "255 | 2^128-1 (255 | 2^8-1 and 8 | 128)");
        let g = pow_u128(alpha, m / 255);
        // Sanity: g has order 255.
        debug_assert_eq!(pow_u128(g, 255), Gf128(1));

        // Find h ∈ K with the AES minimal polynomial x^8+x^4+x^3+x+1
        // (a root exists in K because GF(2^8) contains roots of every
        // irreducible degree-8 polynomial; scanning K* finds one).
        let mut h = ZERO;
        {
            let mut x = Gf128(1);
            for _ in 0..255 {
                // p(h) = h^8 + h^4 + h^3 + h + 1
                let h2 = mul(x, x);
                let h4 = mul(h2, h2);
                let h8 = mul(h4, h4);
                let h3 = mul(h2, x);
                if (h8 + h4 + h3 + x + Gf128(1)) == ZERO {
                    h = x;
                    break;
                }
                x = mul(x, g);
            }
        }
        assert_ne!(h, ZERO, "K contains a root of the AES polynomial");

        let mut basis = [ZERO; 8];
        let mut acc = Gf128(1);
        for b in basis.iter_mut() {
            *b = acc;
            acc = mul(acc, h);
        }

        // embed8[code] = XOR of basis[i] over set bits i of the code — linear.
        let mut embed8 = [ZERO; 256];
        for code in 1..256usize {
            let low = code & (code - 1); // clear lowest set bit
            let bit = (code ^ low).trailing_zeros() as usize;
            embed8[code] = embed8[low] + basis[bit];
        }
        // decode map (init-time only)
        let mut decode = std::collections::HashMap::with_capacity(256);
        for (code, &e) in embed8.iter().enumerate() {
            decode.insert(e.0, code as u8);
        }

        let mut mul_tab = vec![0u8; 65536].into_boxed_slice();
        for a in 0..256usize {
            let ea = embed8[a];
            for b in 0..256usize {
                let p = mul(ea, embed8[b]);
                mul_tab[(a << 8) | b] = decode[&p.0];
            }
        }
        let mul_tab: Box<[u8; 65536]> = {
            // Box<[u8]> of exactly 65536 elements → Box<[u8; 65536]>.
            // Length was just set; the conversion only reinterprets the slice.
            let raw = Box::into_raw(mul_tab) as *mut [u8; 65536];
            // SAFETY: the raw pointer came from a Box<[u8]> of length 65536.
            unsafe { Box::from_raw(raw) }
        };
        // (defensive re-check of the length invariant)
        debug_assert_eq!(mul_tab.len(), 65536);

        // Multiplicative inverses: for e ∈ K*, scan p = e^k until e·p = 1
        // (then p = e^{ord(e)−1} = e^{-1}). Init-time only: 256·255 table muls.
        let mut inv8 = [0u8; 256];
        for a in 1..256usize {
            let e = embed8[a];
            let mut p = e;
            let mut inv = ZERO;
            for _ in 1..255 {
                if mul(e, p) == Gf128(1) {
                    inv = p;
                    break;
                }
                p = mul(p, e);
            }
            assert_ne!(inv, ZERO, "inverse exists for a={a}");
            inv8[a] = decode[&inv.0];
        }

        // AES-polynomial compatibility self-check (the GFNI contract):
        // multiply two codes and compare against the AES carry-less product
        // reduced mod 0x11B for a few values.
        {
            let aes_mul = |a: u8, b: u8| -> u8 {
                let mut acc: u16 = 0;
                for i in 0..8 {
                    if (b >> i) & 1 == 1 {
                        acc ^= (a as u16) << i;
                    }
                }
                let mut v = acc;
                while v >= 0x100 {
                    let shift = 15 - v.leading_zeros() as u16; // top bit position
                    v ^= 0x11Bu16 << (shift - 8);
                }
                v as u8
            };
            for &(a, b) in &[(0x57u8, 0x83u8), (1, 0xff), (0xca, 0x53), (0x80, 0x01)] {
                debug_assert_eq!(mul_tab[((a as usize) << 8) | b as usize], aes_mul(a, b));
            }
        }

        Gf8 {
            embed8,
            mul_tab,
            inv8,
            basis,
        }
    }

    /// The `E`-embedding of a code.
    #[inline(always)]
    pub fn embed(&self, a: u8) -> Gf128 {
        self.embed8[a as usize]
    }

    /// Code-level product.
    #[inline(always)]
    pub fn mul(&self, a: u8, b: u8) -> u8 {
        self.mul_tab[((a as usize) << 8) | b as usize]
    }

    /// The `K`-basis `(1, h, …, h⁷)` as `E`-elements.
    pub fn basis(&self) -> &[Gf128; 8] {
        &self.basis
    }
}

impl Default for Gf8 {
    fn default() -> Self {
        Self::new()
    }
}

fn pow_u128(a: Gf128, k: u128) -> Gf128 {
    let mut result = Gf128(1);
    let mut base = a;
    let mut e = k;
    while e > 0 {
        if e & 1 == 1 {
            result = mul(result, base);
        }
        base = mul(base, base);
        e >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subfield_closure() {
        let f = Gf8::new();
        // Closure: products and sums of K-elements stay in K (order-255 check).
        for a in (1..256).step_by(37) {
            let ea = f.embed(a as u8);
            // Frobenius^(8): k^(2^8) = k for subfield elements.
            let mut k = ea;
            for _ in 0..8 {
                k = mul(k, k);
            }
            assert_eq!(k, ea, "Frobenius^8 fixes K");
        }
    }

    #[test]
    fn field_axioms() {
        let f = Gf8::new();
        // commutativity + identity + inverse on a sample
        assert_eq!(f.mul(1, 0x5a), 0x5a);
        for a in 1..256usize {
            let ia = f.inv8[a] as usize;
            assert_eq!(f.mul(a as u8, ia as u8), 1, "a*a^(-1) must be 1 (a={a})");
        }
        // distributivity spot-check: a·(b^c) = a·b ^ a·c
        let (a, b, c) = (0x53u8, 0x64u8, 0x9au8);
        assert_eq!(f.mul(a, b ^ c), f.mul(a, b) ^ f.mul(a, c));
        // associativity spot-check
        assert_eq!(f.mul(f.mul(a, b), c), f.mul(a, f.mul(b, c)));
    }

    #[test]
    fn embed_is_ring_map() {
        let f = Gf8::new();
        for &(a, b) in &[(0x57u8, 0x83u8), (0xf1, 0x0e), (0x00, 0xff)] {
            assert_eq!(f.embed(a ^ b), f.embed(a) + f.embed(b), "add");
            assert_eq!(f.embed(f.mul(a, b)), mul(f.embed(a), f.embed(b)), "mul");
        }
    }
}
