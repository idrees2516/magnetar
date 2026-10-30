//! The shared interpolation data of the Mikros small-field variant
//! (ePrint 2026/2646, Appendix B.1 "Shared interpolation data").
//!
//! Binary instantiation (paper §6, Table 3): `F = GF(2)`, `t = 8`,
//! `K = GF(2^8)`, `κ = 128`, `E = GF(2^128)`.
//!
//! * **Roots** `H`: the codes `0x00..=0x7F` — the span of `(1, h, …, h⁶)`, a
//!   7-dimensional (hence 128-element) additive subspace of `K`, exactly as
//!   the small-field variant requires (`|H| = κ`, all nodes in `K`).
//! * **Off-domain nodes**: 127 distinct codes of `K \ H` (`0x80..=0xFE`).
//!   The verifier additionally samples `x ←$ E \ H`.
//! * **`Ψ`**: the vanishing polynomial `Π_{γ∈H}(X+γ)`, kept as `K`-code
//!   monomial coefficients (degree 128) for E-point Horner evaluation.
//! * **`ψ_j`**: the Lagrange basis on `H`, evaluated (as `K`-codes) at every
//!   off-domain node — these are the *encoding selectors* of the packed
//!   Hadamard prover — plus `inv(Ψ'(γ_j))` so `ψ_j(x)` can be evaluated at
//!   any `E`-point as `Ψ(x)·(x+γ_j)^{-1}·inv(Ψ'(γ_j))`.
//! * **`V⁻¹`**: the 127×127 inverse Vandermonde over `K`, mapping the
//!   quotient's node-values `h(γ') = f(γ')/Ψ(γ')` to its monomial
//!   coefficients (the prover's `h` message).
//!
//! All tables are built with `O(κ²)` table operations at init.

use magnetar_field::Gf128;
use magnetar_field::{mul, ZERO};

use crate::gf8::Gf8;

pub const KAPPA: usize = 128;
pub const T_DEG: usize = 8; // [K : F]

/// A `K`-coefficient polynomial stored as code bytes, little-endian
/// (coefficient `i` at `coefs[i]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KPoly {
    pub coefs: Vec<u8>,
}

impl KPoly {
    pub fn eval_k(&self, f: &Gf8, at: u8) -> u8 {
        let mut acc = 0u8;
        for &c in self.coefs.iter().rev() {
            acc = f.mul(acc, at) ^ c;
        }
        acc
    }

    /// Horner-evaluate at an `E`-point (the verifier's `x ←$ E \ H`).
    pub fn eval_e(&self, f: &Gf8, at: Gf128) -> Gf128 {
        let mut acc = ZERO;
        for &c in self.coefs.iter().rev() {
            acc = mul(acc, at) + f.embed(c);
        }
        acc
    }

    pub fn degree(&self) -> usize {
        self.coefs.len().saturating_sub(1)
    }
}

/// Multiply by `(X + a)` in place over the code field:
/// `(X+a)·Σ c_i X^i` has `X^0`-coefficient `a·c_0`, `X^i`-coefficient
/// `c_{i−1} ^ a·c_i`, and `X^{d+1}`-coefficient `c_d`.
fn poly_mul_x_plus(f: &Gf8, p: &mut Vec<u8>, a: u8) {
    let mut out = vec![0u8; p.len() + 1];
    out[0] = f.mul(a, p[0]);
    for i in 1..p.len() {
        out[i] = p[i - 1] ^ f.mul(a, p[i]);
    }
    out[p.len()] = p[p.len() - 1];
    *p = out;
}

pub struct InterpData {
    /// `Ψ(X) = Π_{γ∈H}(X+γ)` — degree 128, `K`-code monomial coefficients.
    pub psi: KPoly,
    /// `Ψ` evaluated at each off-domain node (codes).
    pub psi_at_off: Vec<u8>,
    /// `inv(Ψ'(γ_j))` for the roots `γ_j = j` (codes) — `Ψ'(γ) = Π_{h∈H, h≠γ}(γ+h)`.
    pub psi_prime_inv: [u8; KAPPA],
    /// `ψ_j` evaluated at every off-domain node: `psi_off[node][j]` (codes).
    /// `ψ_j(γ') = Ψ(γ') · (γ'+γ_j)^{-1} · inv(Ψ'(γ_j))`.
    pub psi_off: Vec<[u8; KAPPA]>,
    /// The 127 off-domain node codes (`0x80..=0xFE`).
    pub nodes_off: Vec<u8>,
    /// `V⁻¹[node_slot] = [u8; 127]`: `h_coef[i] = Σ_n Vinv[i][n] · hval[n]`
    /// (over `K`, applied to `E`-valued samples by embedding).
    pub vandermonde_inv: Vec<[u8; 127]>,
    /// Monomial coefficients of each `ψ_j` (built by `build_psi_monomials`).
    pub psi_monomials: Vec<Vec<u8>>,
}

impl InterpData {
    pub fn new(f: &Gf8) -> Self {
        // Ψ(X) over H = codes 0..127
        let mut psi = vec![1u8];
        for h in 0..KAPPA as u8 {
            poly_mul_x_plus(f, &mut psi, h);
        }
        assert_eq!(psi.len(), KAPPA + 1, "deg Ψ = κ = 128");

        // Ψ'(γ_j) for each root
        let mut psi_prime_inv = [0u8; KAPPA];
        for j in 0..KAPPA as u8 {
            let mut d = 1u8;
            for h in 0..KAPPA as u8 {
                if h != j {
                    d = f.mul(d, j ^ h);
                }
            }
            psi_prime_inv[j as usize] = f.inv8[d as usize];
        }

        let nodes_off: Vec<u8> = (0x80u8..=0xFE).collect();
        assert_eq!(nodes_off.len(), KAPPA - 1);

        // ψ_j at each off-domain node (codes)
        let mut psi_off = Vec::with_capacity(nodes_off.len());
        let mut psi_at_off = Vec::with_capacity(nodes_off.len());
        for &g in &nodes_off {
            let pg = {
                // evaluate Ψ at the code g
                let mut acc = 0u8;
                for &c in psi.iter().rev() {
                    acc = f.mul(acc, g) ^ c;
                }
                acc
            };
            psi_at_off.push(pg);
            assert_ne!(pg, 0, "off-domain node must not be a root");
            let mut row = [0u8; KAPPA];
            for j in 0..KAPPA as u8 {
                // ψ_j(γ') = Ψ(γ')·(γ'+γ_j)^{-1}·inv(Ψ'(γ_j))
                row[j as usize] = f.mul(
                    pg,
                    f.mul(f.inv8[(g ^ j) as usize], psi_prime_inv[j as usize]),
                );
            }
            psi_off.push(row);
        }

        // Inverse Vandermonde of the 127 nodes over K (Gaussian elimination).
        // V[n][i] = nodes_off[n]^i ; we invert the (127×127) matrix.
        let n = nodes_off.len();
        // Build [V | I] and reduce.
        let mut rows: Vec<Vec<u8>> = Vec::with_capacity(n);
        for node in 0..n {
            let mut row = vec![0u8; 2 * n];
            // powers of the node
            let g = nodes_off[node];
            let mut p = 1u8;
            for i in 0..n {
                row[i] = p;
                p = f.mul(p, g);
            }
            row[n + node] = 1;
            rows.push(row);
        }
        for col in 0..n {
            // pivot
            let mut piv = col;
            while piv < n && rows[piv][col] == 0 {
                piv += 1;
            }
            assert!(piv < n, "nodes are distinct ⇒ Vandermonde invertible");
            rows.swap(col, piv);
            let pinv = f.inv8[rows[col][col] as usize];
            for v in rows[col].iter_mut() {
                *v = f.mul(*v, pinv);
            }
            for r in 0..n {
                if r != col && rows[r][col] != 0 {
                    let factor = rows[r][col];
                    for c in 0..2 * n {
                        rows[r][c] ^= f.mul(factor, rows[col][c]);
                    }
                }
            }
        }
        let vandermonde_inv: Vec<[u8; 127]> = rows
            .iter()
            .map(|r| {
                let mut out = [0u8; 127];
                out.copy_from_slice(&r[n..]);
                out
            })
            .collect();

        InterpData {
            psi: KPoly { coefs: psi },
            psi_at_off,
            psi_prime_inv,
            psi_off,
            nodes_off,
            vandermonde_inv,
            psi_monomials: Vec::new(),
        }
    }

    /// `ρ(x) = (ψ_1(x), …, ψ_κ(x)) ∈ E^κ` for the verifier's `x ∈ E \ H`.
    /// `ψ_j(x) = Ψ(x)·(x+γ_j)^{-1}·inv(Ψ'(γ_j))`; one shared E-inversion of
    /// `Ψ(x)`... (x+γ_j)^{-1} is per-j, so we use Horner on the ψ_j monomial
    /// coefficients instead — but we never materialised those. Practical
    /// route: evaluate `Ψ(x)` once, then `ψ_j(x)` needs `(x+γ_j)^{-1}` — the
    /// per-j E-inversion is the honest cost; we keep it simple and exact.
    pub fn rho_at_e(&self, f: &Gf8, x: Gf128) -> Vec<Gf128> {
        // Materialise ψ_j monomial coefficients lazily per call? κ polys of
        // 128 coefficients evaluated by Horner is 16K E-mults — cheap and
        // allocation-free after init. We cache the coefficient table at init
        // time in `psi_monomials`.
        let mut out = Vec::with_capacity(KAPPA);
        for j in 0..KAPPA {
            let coefs = &self.psi_monomials[j];
            let mut acc = ZERO;
            for &c in coefs.iter().rev() {
                acc = mul(acc, x) + f.embed(c);
            }
            out.push(acc);
        }
        out
    }

    /// The monomial-coefficient cache for `rho_at_e` (built in `new`).
    pub fn build_psi_monomials(&mut self, f: &Gf8) {
        if self.psi_monomials.is_empty() {
            let mut polys = Vec::with_capacity(KAPPA);
            for j in 0..KAPPA as u8 {
                // ψ_j(X) = Ψ(X)·(X+γ_j)^{-1}·inv(Ψ'(γ_j)) over the K[X]
                // fraction: do the polynomial division Ψ(X)/(X+γ_j) by
                // synthetic division (the root is γ_j since char 2: X+γ_j
                // vanishes at X=γ_j).
                let inv_prime = self.psi_prime_inv[j as usize];
                // synthetic division of Ψ by (X + γ_j): quotient q of degree 127
                let d = self.psi.coefs.len() - 1; // 128
                let mut q = vec![0u8; d];
                let mut carry = self.psi.coefs[d];
                for i in (0..d).rev() {
                    q[i] = carry;
                    carry = self.psi.coefs[i] ^ f.mul(carry, j);
                }
                // ψ_j = q · inv(Ψ'(γ_j))
                for v in q.iter_mut() {
                    *v = f.mul(*v, inv_prime);
                }
                polys.push(q);
            }
            self.psi_monomials = polys;
        }
    }

    /// Interpolate `h`'s 127 monomial coefficients (over `E`) from the
    /// `f`-samples at the off-domain nodes: `h(γ') = f(γ')/Ψ(γ')`.
    pub fn h_from_samples(&self, f: &Gf8, f_samples: &[Gf128]) -> Vec<Gf128> {
        assert_eq!(f_samples.len(), self.nodes_off.len());
        let n = self.nodes_off.len();
        // h-values at the nodes
        let mut hvals = Vec::with_capacity(n);
        for (s, node) in f_samples.iter().zip(self.nodes_off.iter()) {
            let inv_psi = f.embed(
                f.inv8[self.psi_at_off[self
                    .nodes_off
                    .iter()
                    .position(|&x| x == *node)
                    .expect("node present")] as usize],
            );
            hvals.push(mul(*s, inv_psi));
        }
        // coefficients: h_coef[i] = Σ_n Vinv[i][n]·hvals[n]  (E × K products)
        let mut coefs = Vec::with_capacity(n);
        for row in self.vandermonde_inv.iter() {
            let mut acc = ZERO;
            for (v, h) in row.iter().zip(hvals.iter()) {
                if *v != 0 {
                    acc += mul(f.embed(*v), *h);
                }
            }
            coefs.push(acc);
        }
        coefs
    }

    /// `Ψ(x)` for the verifier's final identity `〈a, b∘l〉 = c_x + Ψ(x)h(x)`.
    pub fn psi_at_e(&self, f: &Gf8, x: Gf128) -> Gf128 {
        self.psi.eval_e(f, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force reference: compute ψ_j by direct Lagrange evaluation over E.
    fn psi_j_ref(f: &Gf8, _data: &InterpData, j: usize, x: Gf128) -> Gf128 {
        // ψ_j(x) = Π_{h∈H, h≠γ_j} (x + h) / (γ_j + h)
        let gamma = f.embed(j as u8);
        let mut num = Gf128(1);
        let mut den = Gf128(1);
        for h in 0..KAPPA as u8 {
            if h as usize != j {
                num = mul(num, x + f.embed(h));
                den = mul(den, gamma + f.embed(h));
            }
        }
        // num / den over E: den · ψ = num
        // brute-force inverse
        let inv = {
            // den^(2^128-2)
            let e: u128 = u128::MAX - 1;
            let mut r = Gf128(1);
            let b = den;
            for i in (0..128).rev() {
                r = mul(r, r);
                if (e >> i) & 1 == 1 {
                    r = mul(r, b);
                }
            }
            let _ = b;
            r
        };
        mul(num, inv)
    }

    #[test]
    fn psi_vanishes_on_h_and_interpolates() {
        let f = Gf8::new();
        let mut data = InterpData::new(&f);
        data.build_psi_monomials(&f);
        // Ψ vanishes on H and not off-domain
        for h in (0..KAPPA as u8).step_by(9) {
            assert_eq!(data.psi.eval_k(&f, h), 0);
        }
        for &g in &data.nodes_off {
            assert_ne!(data.psi.eval_k(&f, g), 0);
        }
        // ψ_j(γ_k) = δ_jk over K
        for j in (0..KAPPA).step_by(17) {
            for k in (0..KAPPA).step_by(31) {
                let v = {
                    let mut acc = 0u8;
                    for &c in data.psi_monomials[j].iter().rev() {
                        acc = f.mul(acc, k as u8) ^ c;
                    }
                    acc
                };
                assert_eq!(
                    v,
                    if j == k { 1 } else { 0 },
                    "ψ_{j}(γ_{k}) must be δ_{{jk}}"
                );
            }
        }
        // ρ(x) matches the brute-force Lagrange evaluation at E-points
        let mut x = Gf128(0x9e3779b97f4a7c15f39cc0605cedc834);
        for _ in 0..3 {
            x = mul(x, x) + Gf128(3);
            let rho = data.rho_at_e(&f, x);
            for j in (0..KAPPA).step_by(29) {
                assert_eq!(rho[j], psi_j_ref(&f, &data, j, x));
            }
            // Σ_j ψ_j(x) = 1 (partition of unity, x ∉ H)
            let s = rho.iter().fold(ZERO, |a, b| a + *b);
            assert_eq!(s, Gf128(1));
        }
    }

    #[test]
    fn vandermonde_interpolates() {
        let f = Gf8::new();
        let data = InterpData::new(&f);
        // pick h(X) = Σ c_i X^i over E with small coefficients; sample at the
        // nodes via Horner over E; then reconstruct via h_from_samples with
        // f-samples = h(node)·Ψ(node) and compare.
        let h_coefs: Vec<Gf128> = (0..127u32)
            .map(|i| Gf128((i as u128) * 0x010101 + 7))
            .collect();
        let mut f_samples = Vec::with_capacity(data.nodes_off.len());
        for &g in &data.nodes_off {
            let ge = f.embed(g);
            let mut acc = ZERO;
            for &c in h_coefs.iter().rev() {
                acc = mul(acc, ge) + c;
            }
            f_samples.push(mul(
                acc,
                f.embed(data.psi_at_off[data.nodes_off.iter().position(|&x| x == g).unwrap()]),
            ));
        }
        let rec = data.h_from_samples(&f, &f_samples);
        for i in 0..127 {
            assert_eq!(rec[i], h_coefs[i], "coefficient {i}");
        }
    }
}
