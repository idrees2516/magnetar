//! Algorithm 3 at the §5.1 copy-constraint geometry (ePrint 2026/2646):
//! the packed permutation proof with base field `K_pos = GF(2³²)`,
//! extension `E = GF(2¹²⁸)`, and packing width `κ_pos = [E : K_pos] = 4`.
//!
//! The single-instance PLONK protocol (Algorithm 5) reduces its copy
//! constraint `σ(w) = w` to a permutation relation over `K_pos` via the
//! joint value–position encoding (Lemma 2): with the GF(2)-power basis of
//! [`crate::gf32`] as the `θ`-basis, the operand entries are *literally*
//! `code_a[n] = σ₀(n) | (w[n] << 31)` and `code_b[n] = n | (w[n] << 31)`
//! (zero beyond `L = 3N + n_io`), and
//!
//! ```text
//! σ(w) = w  ⟺  a ≅_perm b        (Lemma 2)
//! ```
//!
//! The two operands are padded to length `4N = m_pos·κ_pos` (the padding
//! entries are 0 on both sides, contributing the shared factor
//! `z^{4N−L}` to both grand products — hence `z ←$ E \ {0}`), arranged
//! column-major `D[r][j] = operand[j·m_pos + r]` — **column `j` is exactly
//! the position segment `[jN, (j+1)N)`** of the augmented witness
//! `(w_L ‖ w_R ‖ w_O ‖ io)`, the structure the terminal expansion of
//! Algorithm 5 exploits.
//!
//! **The tree** (Eq. 11 over `K_pos`): `C₀(Z) = Z·1 + D` (2 coefficient
//! layers), `C₁` (3 layers, half the rows), `C₂` (5 layers, a quarter of
//! the rows) — `log₂ κ_pos = 2` levels, each entry a `u32` K_pos code.
//! The top level is z-evaluated (embedded into `E`, Horner against the
//! `z^d`), flattened column-major to `c₂(z) ∈ E^{m_pos}`, and the
//! remaining `log₂ m_pos` levels are pointwise products over `E` up to
//! the claimed common root `Θ` — one grand product per operand.
//!
//! **The proof unfolds in reverse** (the §4.3 engine instantiated at
//! width 4): `log₂ m_pos` cubic weighted sum-checks over `E` (the shared
//! [`crate::perm::cubic_prove`] engine) down to `v₂ = l₂^⊤ C₂(z) r₂`,
//! then **two coefficient transitions** `τ = 1, 0`: the column claims
//! `u_{τ+1} = l_{τ+1}^⊤ C_{τ+1}(z)` (aggregate-then-evaluate against the
//! E-row-weight), the quotient `h = f/Ψ` of the width-4 Lagrange residual
//! (3 off-domain samples over `K_pos`, 3 coefficients), the `g_τ(z,x)`
//! contraction against `ρ(x) = ψ(x) ∈ E⁴`, and one more cubic sum-check
//! per transition. The terminal is the packed claim
//!
//! ```text
//! v₀ − z = l₀^⊤ D ρ(x₀)
//! ```
//!
//! with `l₀` a tensor row weight over the `m_pos` rows and `ρ(x₀) ∈ E⁴` —
//! the interface Algorithm 5's grouped openings consume (restricted to the
//! wire segments and regrouped to the width-κ witness packing there).
//!
//! The verifier's identity-label contribution (Appendix B.3, Proposition 5)
//! — the `ι` side of the terminal — is evaluated directly from the
//! position bits in `O(κ_pos + log L)` by [`iota_eval_padded`]: the label
//! codes split as `code(n) = (j ≪ log₂ m_pos) | r` (the column/segment
//! part plus the row part), so the weighted evaluation factors into
//! full-column sums and one `n_io`-restricted partial column.

use crate::gf32::{clmul32, Gf32};
use crate::perm::{cubic_prove, cubic_verify, msbf_to_factors, ts_msbf, CubicProof};
use magnetar_field::Gf128;
use magnetar_field::{mul, ONE, ZERO};
use magnetar_hash::Transcript;

pub const KPOS: usize = 4; // κ_pos = [E : K_pos]
pub const LOG_KPOS: usize = 2;

// ---------------------------------------------------------------------------
// The width-4 interpolation data over K_pos
// ---------------------------------------------------------------------------

/// The shared interpolation data of the copy argument: `H = {0,1,2,3}`
/// (codes; distinct as K_pos elements in the power basis), the 3 off-domain
/// nodes `{4,5,6}`, the Lagrange `ψ_j` (as `K_pos`-code monomial
/// coefficients and their node evaluations), `Ψ`, and the 3×3 inverse
/// Vandermonde for the quotient interpolation.
pub struct PosInterp {
    /// `Ψ(X)` — degree 4, K_pos-code monomial coefficients (monic).
    pub psi: [u32; 5],
    /// `Ψ` evaluated at each off-domain node (K_pos codes, nonzero).
    pub psi_at_off: [u32; 3],
    /// `ψ_j` evaluated at every off-domain node: `psi_off[node][j]` (codes).
    pub psi_off: [[u32; KPOS]; 3],
    /// `ψ_j` monomial coefficients (K_pos codes), for the E-point Horner.
    pub psi_monomials: [[u32; 4]; KPOS],
    /// The 3×3 inverse Vandermonde over K_pos (codes).
    pub vandermonde_inv: [[u32; 3]; 3],
    /// The 3 off-domain node codes.
    pub nodes_off: [u32; 3],
}

/// Inversion in `K_pos` by the binary extended-Euclid over `GF(2)[X]`
/// (init-time only; the inverses are cached as E-elements afterwards).
fn inv32_euclid(f: &Gf32, a: u32) -> u32 {
    assert_ne!(a, 0, "0 has no inverse");
    let modulus: u64 = (f.p_low as u64) | (1u64 << 32);
    let (mut r0, mut s0): (u64, u64) = (a as u64, 1);
    let (mut r1, mut s1): (u64, u64) = (modulus, 0);
    while r1 != 0 {
        if r0.leading_zeros() > r1.leading_zeros() {
            std::mem::swap(&mut r0, &mut r1);
            std::mem::swap(&mut s0, &mut s1);
        }
        // deg(r0) ≥ deg(r1) here, so the cancellation shift is
        // deg(r0) − deg(r1) = lz(r1) − lz(r0) ≥ 0.
        let shift = r1.leading_zeros() - r0.leading_zeros();
        r0 ^= r1 << shift;
        s0 ^= s1 << shift; // GF(2)[X] arithmetic
    }
    debug_assert_eq!(r0, 1, "a is invertible mod p");
    // The Bezout coefficient satisfies s0·a ≡ 1 (mod p) but may carry
    // degree ≥ 32 — reduce it to the canonical representative.
    let s = reduce32(s0, f.p_low);
    assert_eq!(reduce32(clmul32(a, s), f.p_low), 1, "a·a⁻¹ = 1");
    s
}

fn reduce32(mut t: u64, p_low: u32) -> u32 {
    while t >> 32 != 0 {
        let hi = (t >> 32) as u32;
        t = (t & 0xFFFF_FFFF) ^ clmul32(hi, p_low);
    }
    t as u32
}

impl PosInterp {
    pub fn new(f: &Gf32) -> Self {
        // Ψ(X) = Π_{γ ∈ H}(X + γ) over the codes
        let mut psi: Vec<u32> = vec![0, 1]; // X
        for gamma in 1u32..4 {
            // multiply by (X + gamma)
            let mut next = vec![0u32; psi.len() + 1];
            for (i, &c) in psi.iter().enumerate() {
                next[i] ^= f.mul(c, gamma);
                next[i + 1] ^= c;
            }
            psi = next;
        }
        assert_eq!(psi.len(), 5);
        let psi: [u32; 5] = psi.try_into().unwrap();

        // Ψ'(γ_j) = Π_{h ∈ H, h ≠ γ}(γ + h) for each root (the code form is
        // needed at several places below)
        let prime_inv_code = |f: &Gf32, j: usize| -> u32 {
            let gj = j as u32;
            let mut d = 1u32;
            for h in 0..4u32 {
                if h != gj {
                    d = f.mul(d, gj ^ h);
                }
            }
            inv32_euclid(f, d)
        };

        // the off-domain nodes {4, 5, 6}
        let nodes_off = [4u32, 5, 6];
        let mut psi_at_off = [0u32; 3];
        let mut psi_off = [[0u32; KPOS]; 3];
        for (n, &g) in nodes_off.iter().enumerate() {
            let mut acc = 0u32;
            for &c in psi.iter().rev() {
                acc = f.mul(acc, g) ^ c;
            }
            assert_ne!(acc, 0, "off-domain node is not a root");
            psi_at_off[n] = acc;
            for j in 0..4 {
                // ψ_j(g) = Ψ(g)·(g+γ_j)^{-1}·inv(Ψ'(γ_j))
                let inv_diff = inv32_euclid(f, g ^ (j as u32));
                psi_off[n][j] = f.mul(f.mul(acc, inv_diff), prime_inv_code(f, j));
            }
        }

        // the ψ_j monomial coefficients by synthetic division of Ψ by
        // (X + γ_j), scaled by inv(Ψ'(γ_j))
        let mut psi_monomials = [[0u32; 4]; KPOS];
        for j in 0..4 {
            let inv_prime = prime_inv_code(f, j);
            let mut q = [0u32; 4];
            let mut carry = psi[4];
            for i in (0..4).rev() {
                q[i] = carry;
                carry = psi[i] ^ f.mul(carry, j as u32);
            }
            debug_assert_eq!(carry, 0, "γ_j is a root of Ψ");
            for v in q.iter_mut() {
                *v = f.mul(*v, inv_prime);
            }
            psi_monomials[j] = q;
        }

        // the 3×3 inverse Vandermonde of the off-domain nodes (Gaussian
        // elimination over K_pos)
        let n = 3;
        let mut rows: Vec<Vec<u32>> = Vec::with_capacity(n);
        for &g in nodes_off.iter() {
            let mut row = vec![0u32; 2 * n];
            let mut p = 1u32;
            for i in 0..n {
                row[i] = p;
                p = f.mul(p, g);
            }
            row[n + rows.len()] = 1;
            rows.push(row);
        }
        for col in 0..n {
            let mut piv = col;
            while piv < n && rows[piv][col] == 0 {
                piv += 1;
            }
            assert!(piv < n, "distinct nodes ⇒ invertible");
            rows.swap(col, piv);
            let pinv = inv32_euclid(f, rows[col][col]);
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
        let mut vandermonde_inv = [[0u32; 3]; 3];
        for (i, r) in rows.iter().enumerate() {
            vandermonde_inv[i].copy_from_slice(&r[n..]);
        }

        PosInterp {
            psi,
            psi_at_off,
            psi_off,
            psi_monomials,
            vandermonde_inv,
            nodes_off,
        }
    }

    /// `ρ(x) = (ψ_1(x), …, ψ_4(x)) ∈ E⁴` for `x ∈ E \ H` (Horner on the
    /// ψ_j monomials over the embedded codes).
    pub fn rho_at_e(&self, f: &Gf32, x: Gf128) -> [Gf128; KPOS] {
        let mut out = [ZERO; KPOS];
        for j in 0..KPOS {
            let mut acc = ZERO;
            for &c in self.psi_monomials[j].iter().rev() {
                acc = mul(acc, x) + f.embed(c);
            }
            out[j] = acc;
        }
        out
    }

    /// `Ψ(x)` for the identity `〈l, g_U ∘ g_D〉 = gC(x) + Ψ(x)h(x)`.
    pub fn psi_at_e(&self, f: &Gf32, x: Gf128) -> Gf128 {
        let mut acc = ZERO;
        for &c in self.psi.iter().rev() {
            acc = mul(acc, x) + f.embed(c);
        }
        acc
    }

    /// Is `x` outside `H` (i.e., `Ψ(x) ≠ 0`)?
    pub fn outside_h(&self, f: &Gf32, x: Gf128) -> bool {
        self.psi_at_e(f, x) != ZERO
    }

    /// Interpolate `h`'s 3 monomial coefficients (over `E`) from the
    /// `f`-samples at the off-domain nodes: `h(γ') = f(γ')/Ψ(γ')`.
    pub fn h_from_samples(&self, f: &Gf32, f_samples: &[Gf128; 3]) -> [Gf128; 3] {
        let mut hvals = [ZERO; 3];
        for n in 0..3 {
            let inv_psi = inv32_euclid(f, self.psi_at_off[n]);
            hvals[n] = mul(f_samples[n], f.embed(inv_psi));
        }
        let mut coefs = [ZERO; 3];
        for i in 0..3 {
            let mut acc = ZERO;
            for n in 0..3 {
                let v = self.vandermonde_inv[i][n];
                if v != 0 {
                    acc += mul(f.embed(v), hvals[n]);
                }
            }
            coefs[i] = acc;
        }
        coefs
    }
}

// ---------------------------------------------------------------------------
// The coefficient tree over K_pos
// ---------------------------------------------------------------------------

/// One level: `layers[j][d][r]` — the `Z^d`-coefficient (a K_pos code) of
/// column `j` at row `r`.
pub struct CopyTree {
    pub levels: Vec<Vec<Vec<Vec<u32>>>>, // [τ][j][d][r]
    pub m_pos: usize,
}

fn ncoef_at(tau: usize) -> usize {
    (1 << tau) + 1
}

impl CopyTree {
    /// Build from the operand codes `d[j·m_pos + r]` (`D[r][j]`).
    pub fn build(f: &Gf32, d: &[u32], m_pos: usize) -> Self {
        assert_eq!(d.len(), m_pos * KPOS);
        assert!(m_pos.is_multiple_of(1 << LOG_KPOS));
        let mut levels = Vec::with_capacity(LOG_KPOS + 1);
        // C₀(Z) = Z·1 + D
        let mut c0: Vec<Vec<Vec<u32>>> = Vec::with_capacity(KPOS);
        for j in 0..KPOS {
            let col = &d[j * m_pos..(j + 1) * m_pos];
            c0.push(vec![col.to_vec(), vec![1u32; m_pos]]);
        }
        levels.push(c0);
        for tau in 0..LOG_KPOS {
            let prev = &levels[tau];
            let rows_in = m_pos >> tau;
            let ncoef_in = ncoef_at(tau);
            let rows_out = rows_in / 2;
            let ncoef_out = ncoef_in * 2 - 1;
            let mut cur: Vec<Vec<Vec<u32>>> = (0..KPOS)
                .map(|_| vec![vec![0u32; rows_out]; ncoef_out])
                .collect();
            // The row pairing is the CONTIGUOUS halves (C_{τ+1}[r] =
            // C_τ[r] ∘ C_τ[r + rows_out]) — the cubic engine's U/D
            // convention (perm.rs's CoefTree does the same).
            for j in 0..KPOS {
                for d1 in 0..ncoef_in {
                    for d2 in 0..ncoef_in {
                        let pu = &prev[j][d1];
                        let pd = &prev[j][d2];
                        let po = &mut cur[j][d1 + d2];
                        // 4-row batches through the VPCLMULQDQ lanes:
                        // lane k multiplies the sources of output row r+k
                        // (the tail lanes clamp to the last valid row and
                        // are discarded)
                        let mut r = 0;
                        while r < rows_out {
                            let take = 4.min(rows_out - r);
                            let mut a = [0u32; 4];
                            let mut b = [0u32; 4];
                            for k in 0..4 {
                                let rr = r + k.min(take - 1);
                                a[k] = pu[rr];
                                b[k] = pd[rr + rows_out];
                            }
                            let prod = f.mul4(a, b);
                            for k in 0..take {
                                po[r + k] ^= prod[k];
                            }
                            r += take;
                        }
                    }
                }
            }
            levels.push(cur);
        }
        CopyTree { levels, m_pos }
    }

    /// `C₂(z)` flattened column-major: `out[j·(m_pos/4) + r] =
    /// Σ_d embed(C₂[j][d][r])·z^d` — the E-tree's base vector. The layers
    /// are embedded once each and mixed by broadcast-multiplied runs.
    pub fn flatten_top(&self, f: &Gf32, z: Gf128) -> Vec<Gf128> {
        let top = &self.levels[LOG_KPOS];
        let rows = self.m_pos >> LOG_KPOS;
        let ncoef = ncoef_at(LOG_KPOS);
        let mut out = vec![ZERO; self.m_pos];
        for j in 0..KPOS {
            let base = j * rows;
            let mut zd = ONE;
            for d in 0..ncoef {
                let emb: Vec<Gf128> = top[j][d].iter().map(|&c| f.embed(c)).collect();
                magnetar_field::avx::axpy_run(&mut out[base..base + rows], &emb, zd, rows);
                zd = mul(zd, z);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// The protocol
// ---------------------------------------------------------------------------

/// One coefficient transition `τ = 1, 0` (§4.3 at width 4).
#[derive(Clone)]
pub struct CopyFTrans {
    /// `u_{τ+1} = l_{τ+1}^⊤ C_{τ+1}(z)` — the 4 column claims.
    pub u: [Gf128; KPOS],
    /// The 3 quotient coefficients `h = f/Ψ`.
    pub h: [Gf128; 3],
    pub cubic: CubicProof,
}

#[derive(Clone)]
pub struct CopyUnfold {
    pub e_tree: Vec<CubicProof>,
    pub f_trans: Vec<CopyFTrans>,
}

#[derive(Clone)]
pub struct CopyProof {
    /// The claimed common fingerprint root.
    pub theta: Gf128,
    pub unfolds: [CopyUnfold; 2],
}

/// A deep-clone helper for the proof structure (the tamper tests).
pub fn clone_copy_proof(p: &CopyProof) -> CopyProof {
    p.clone()
}

/// One operand's terminal packed claim:
/// `(v₀ − z, the l₀ row factors (little-endian), ρ(x₀) ∈ E⁴)`.
pub type CopyClaim = (Gf128, Vec<(Gf128, Gf128)>, [Gf128; KPOS]);

pub struct CopyTerminal {
    pub claims: [CopyClaim; 2],
}

/// A transcript challenge `z ∈ E \ {0}` (the shared padding factor
/// `z^{4N−L}` must be nonzero).
fn challenge_nonzero(t: &mut Transcript) -> Gf128 {
    loop {
        let z = t.challenge_field("copy-z");
        if z != ZERO {
            return z;
        }
    }
}

/// A transcript challenge `x ∈ E \ H`.
fn challenge_outside_h(f: &Gf32, interp: &PosInterp, t: &mut Transcript) -> Gf128 {
    loop {
        let x = t.challenge_field("copy-x");
        if interp.outside_h(f, x) {
            return x;
        }
    }
}

fn dot_gf(a: &[Gf128], b: &[Gf128]) -> Gf128 {
    let mut acc = ZERO;
    for i in 0..a.len() {
        acc += mul(a[i], b[i]);
    }
    acc
}

/// `u_{τ+1} = l_{τ+1}^⊤ C_{τ+1}(z)` — aggregate-then-evaluate: per (j, d),
/// the E-embedding of the layer dotted against the row weight, scaled by
/// `z^d` (the dots run through the VPCLMULQDQ lanes).
fn column_claims_u(
    f: &Gf32,
    tree: &CopyTree,
    tau1: usize,
    l: &[Gf128],
    zp: &[Gf128],
    scratch: &mut Vec<Gf128>,
) -> [Gf128; KPOS] {
    let level = &tree.levels[tau1];
    let ncoef = ncoef_at(tau1);
    let rows = tree.m_pos >> tau1;
    debug_assert_eq!(l.len(), rows);
    let mut u = [ZERO; KPOS];
    for j in 0..KPOS {
        let mut acc = ZERO;
        for d in 0..ncoef {
            let emb: Vec<Gf128> = level[j][d].iter().map(|&c| f.embed(c)).collect();
            acc += mul(crate::simd::dot_e(scratch, l, &emb), zp[d]);
        }
        u[j] = acc;
    }
    u
}

/// The off-domain `f`-samples of one transition: per node `γ'`, the
/// K_pos-encoded half-row products of level τ's layers, z-evaluated over E
/// and aggregated against `l_{τ+1}`:
/// `f(γ') = Σ_r l_r·ĜU[r]·ĜD[r] + gC(γ')` with
/// `ĜU[r] = Σ_d z^d·embed(encU[d][r])`.
#[allow(clippy::too_many_arguments)]
fn f_samples_copy(
    f: &Gf32,
    interp: &PosInterp,
    tree: &CopyTree,
    tau: usize,
    l: &[Gf128],
    u: &[Gf128; KPOS],
    zp: &[Gf128],
) -> [Gf128; 3] {
    let rows_half = tree.m_pos >> (tau + 1);
    let ncoef = ncoef_at(tau);
    let level = &tree.levels[tau];
    let mut gu: Vec<Gf128> = vec![ZERO; rows_half];
    let mut gd: Vec<Gf128> = vec![ZERO; rows_half];
    let mut s1: Vec<Gf128> = Vec::new();
    let mut s2: Vec<Gf128> = Vec::new();
    let mut out = [ZERO; 3];
    for node in 0..3 {
        let psi = interp.psi_off[node];
        // the encodings of each half: enc[d][r] = Σ_j ψ_j·C_τ[j][d][r+half]
        // (the contiguous U/D row halves — the tree's pairing)
        let mut enc_u: Vec<Vec<u32>> = Vec::with_capacity(ncoef);
        let mut enc_d: Vec<Vec<u32>> = Vec::with_capacity(ncoef);
        for d in 0..ncoef {
            let mut eu = vec![0u32; rows_half];
            let mut ed = vec![0u32; rows_half];
            for r in 0..rows_half {
                let cu = [
                    level[0][d][r],
                    level[1][d][r],
                    level[2][d][r],
                    level[3][d][r],
                ];
                let cd = [
                    level[0][d][r + rows_half],
                    level[1][d][r + rows_half],
                    level[2][d][r + rows_half],
                    level[3][d][r + rows_half],
                ];
                eu[r] = f.dot4(psi, cu);
                ed[r] = f.dot4(psi, cd);
            }
            enc_u.push(eu);
            enc_d.push(ed);
        }
        // ĝ = Σ_d z^d·embed(enc_d), then the triple dot against l
        gu.iter_mut().for_each(|v| *v = ZERO);
        gd.iter_mut().for_each(|v| *v = ZERO);
        for d in 0..ncoef {
            let emb_u: Vec<Gf128> = enc_u[d].iter().map(|&c| f.embed(c)).collect();
            let emb_d: Vec<Gf128> = enc_d[d].iter().map(|&c| f.embed(c)).collect();
            magnetar_field::avx::axpy_run(&mut gu, &emb_u, zp[d], rows_half);
            magnetar_field::avx::axpy_run(&mut gd, &emb_d, zp[d], rows_half);
        }
        let f_val = crate::simd::dot3_e(&mut s1, &mut s2, l, &gu, &gd);
        let mut g_c = ZERO;
        for j in 0..KPOS {
            g_c += mul(u[j], f.embed(psi[j]));
        }
        out[node] = f_val + g_c; // char 2
    }
    out
}

/// `g_τ(z,x)[r] = Σ_{d,j} C_τ[j][d][r]·z^d·ρ_j(x)` — per (j, d) the
/// embedded layer scaled by `ρ_j·z^d`, accumulated over the full rows.
fn contract_g_copy(
    f: &Gf32,
    tree: &CopyTree,
    tau: usize,
    rho: &[Gf128; KPOS],
    zp: &[Gf128],
) -> Vec<Gf128> {
    let level = &tree.levels[tau];
    let ncoef = ncoef_at(tau);
    let rows = tree.m_pos >> tau;
    let mut g = vec![ZERO; rows];
    for j in 0..KPOS {
        let scale_j = rho[j];
        let mut zd = zp[0];
        for d in 0..ncoef {
            let emb: Vec<Gf128> = level[j][d].iter().map(|&c| f.embed(c)).collect();
            let scaled: Vec<Gf128> = emb.iter().map(|&e| mul(e, scale_j)).collect();
            magnetar_field::avx::axpy_run(&mut g, &scaled, zd, rows);
            if d + 1 < ncoef {
                zd = zp[d + 1];
            }
        }
    }
    g
}

fn expand_factors(factors: &[(Gf128, Gf128)]) -> Vec<Gf128> {
    let mut v = vec![ONE];
    for &(f0, f1) in factors {
        let half = v.len();
        let mut nv = vec![ZERO; half * 2];
        for i in 0..half {
            nv[i] = mul(v[i], f0);
            nv[half + i] = mul(v[i], f1);
        }
        v = nv;
    }
    v
}

/// Prove `a ≅_perm b` over `K_pos` at width `κ_pos = 4` (the copy
/// relation of §5.1). The caller absorbs the commitments before this call.
/// `a`/`b` are the padded operand code arrays (length `m_pos·4`,
/// column-major `operand[j·m_pos + r]`).
pub fn prove_copy(
    f: &Gf32,
    interp: &PosInterp,
    a: &[u32],
    b: &[u32],
    m_pos: usize,
    t: &mut Transcript,
) -> (CopyProof, CopyTerminal) {
    let z = challenge_nonzero(t);
    let mut zp = Vec::with_capacity(6);
    let mut p = ONE;
    for _ in 0..ncoef_at(LOG_KPOS) {
        zp.push(p);
        p = mul(p, z);
    }
    let log_m = m_pos.trailing_zeros() as usize;
    assert!(log_m >= LOG_KPOS, "m_pos ≥ κ_pos (got {m_pos})");

    struct Side {
        tree: CopyTree,
        c2: Vec<Gf128>,
        e_vecs: Vec<Vec<Gf128>>,
        theta: Gf128,
    }
    let mut sides: Vec<Side> = Vec::with_capacity(2);
    for operand in [a, b] {
        let tree = CopyTree::build(f, operand, m_pos);
        let c2 = tree.flatten_top(f, z);
        let mut e_vecs: Vec<Vec<Gf128>> = Vec::with_capacity(log_m);
        let mut cur = c2.clone();
        for _ in 0..log_m {
            let half = cur.len() / 2;
            let mut next = vec![ZERO; half];
            let done = magnetar_field::avx::mulvv(&mut next, &cur[..half], &cur[half..], half);
            for i in done..half {
                next[i] = mul(cur[i], cur[half + i]);
            }
            e_vecs.push(next.clone());
            cur = next;
        }
        let theta = cur[0];
        sides.push(Side {
            tree,
            c2,
            e_vecs,
            theta,
        });
    }
    // NOTE: no prover-side assert Θ_a = Θ_b — the tamper tests deliberately
    // prove broken copy relations (the verifier's unfolds against the single
    // absorbed θ are the soundness gate; a dishonest root breaks them).
    t.absorb_field("copy-theta", &[sides[0].theta]);

    let mut unfolds: Vec<CopyUnfold> = Vec::with_capacity(2);
    let mut claims: Vec<CopyClaim> = Vec::new();
    let mut scratch: Vec<Gf128> = Vec::new();

    for side in &sides {
        // ---- the E-tree unfold: log2 m_pos levels, top-first.
        let mut e_proofs = Vec::with_capacity(log_m);
        let mut cur_pt: Vec<Gf128> = Vec::new();
        let mut cur_v = side.theta;
        for level in 0..log_m {
            let l_len = m_pos >> (log_m - level);
            let vec: &Vec<Gf128> = if level == log_m - 1 {
                &side.c2
            } else {
                &side.e_vecs[log_m - level - 2]
            };
            debug_assert_eq!(vec.len(), 2 * l_len);
            let xi = ts_msbf(&cur_pt);
            let mut cu: Vec<Gf128> = vec[..l_len].to_vec();
            let mut cd: Vec<Gf128> = vec[l_len..].to_vec();
            let mut xi_v = xi;
            let cubic = cubic_prove(&mut xi_v, &mut cu, &mut cd, cur_v, t);
            let y = cubic.challenges.clone();
            let y_p = t.challenge_field("copy-yp");
            cur_v = mul(ONE + y_p, cubic.v_child_u) + mul(y_p, cubic.v_child_d);
            let mut pt = Vec::with_capacity(1 + y.len());
            pt.push(y_p);
            pt.extend_from_slice(&y);
            cur_pt = pt;
            e_proofs.push(cubic);
        }

        // ---- factor the final E-tree point: the first 2 coords are the
        // column bits, the rest the row bits.
        debug_assert_eq!(cur_pt.len(), log_m);
        let col_pt: Vec<Gf128> = cur_pt[..LOG_KPOS].to_vec();
        let row_pt: Vec<Gf128> = cur_pt[LOG_KPOS..].to_vec();
        let r_2 = expand_factors(&msbf_to_factors(&col_pt));
        let mut v_next = cur_v;
        let mut l_next = ts_msbf(&row_pt);
        let mut l_pt_next = row_pt;
        let mut r_next: Vec<Gf128> = r_2;

        // ---- the 2 coefficient transitions, τ = 1 … 0.
        let mut f_trans = Vec::with_capacity(LOG_KPOS);
        for tau in (0..LOG_KPOS).rev() {
            let rows_half = m_pos >> (tau + 1);
            debug_assert_eq!(l_next.len(), rows_half);

            // u_{τ+1}
            let u = column_claims_u(f, &side.tree, tau + 1, &l_next, &zp, &mut scratch);
            debug_assert_eq!(dot_gf(&u, &r_next), v_next, "v_{{τ+1}} = 〈u, r_{{τ+1}}〉");
            t.absorb_field("copy-u", &u);

            // the η samples → h
            let f_samples = f_samples_copy(f, interp, &side.tree, tau, &l_next, &u, &zp);
            let h = interp.h_from_samples(f, &f_samples);
            t.absorb_field("copy-h", &h);

            // x ←$ E \ H, then g_τ(z,x) and the cubic
            let x = challenge_outside_h(f, interp, t);
            let rho = interp.rho_at_e(f, x);
            let g = contract_g_copy(f, &side.tree, tau, &rho, &zp);
            let c_x = dot_gf(&u, &rho);
            let psi_x = interp.psi_at_e(f, x);
            let mut hv = ZERO;
            for &c in h.iter().rev() {
                hv = mul(hv, x) + c;
            }
            let claim = c_x + mul(psi_x, hv);
            debug_assert_eq!(
                dot_gf(&l_next, &{
                    let prod: Vec<Gf128> = g[..rows_half]
                        .iter()
                        .zip(g[rows_half..].iter())
                        .map(|(u_, d_)| mul(*u_, *d_))
                        .collect();
                    prod
                }),
                claim,
                "〈l, g_U ∘ g_D〉 = gC(x) + Ψ(x)h(x)"
            );
            let mut cu: Vec<Gf128> = g[..rows_half].to_vec();
            let mut cd: Vec<Gf128> = g[rows_half..].to_vec();
            let mut xi_v = l_next.clone();
            let cubic = cubic_prove(&mut xi_v, &mut cu, &mut cd, claim, t);
            let y = cubic.challenges.clone();
            let y_p = t.challenge_field("copy-yp");
            v_next = mul(ONE + y_p, cubic.v_child_u) + mul(y_p, cubic.v_child_d);
            let mut pt = Vec::with_capacity(1 + y.len());
            pt.push(y_p);
            pt.extend_from_slice(&y);
            l_pt_next = pt;
            l_next = ts_msbf(&l_pt_next);
            r_next = rho.to_vec();
            f_trans.push(CopyFTrans { u, h, cubic });
        }

        // ---- the terminal: v₀ − z = l₀^⊤ D ρ(x₀).
        debug_assert_eq!(l_next.len(), m_pos);
        let claim = v_next + z; // char 2: v₀ − z
        let l0_factors = msbf_to_factors(&l_pt_next);
        claims.push((claim, l0_factors, {
            let mut r4 = [ZERO; KPOS];
            r4.copy_from_slice(&r_next[..KPOS]);
            r4
        }));
        unfolds.push(CopyUnfold {
            e_tree: e_proofs,
            f_trans,
        });
    }

    let claim_b = claims.pop().expect("two operands");
    let claim_a = claims.pop().expect("two operands");
    let unfold_b = unfolds.pop().expect("two operands");
    let unfold_a = unfolds.pop().expect("two operands");
    (
        CopyProof {
            theta: sides[0].theta,
            unfolds: [unfold_a, unfold_b],
        },
        CopyTerminal {
            claims: [claim_a, claim_b],
        },
    )
}

/// Verify (Algorithm 3 at width 4). Returns the two terminal packed claims
/// for Algorithm 5's grouped openings.
pub fn verify_copy(
    f: &Gf32,
    interp: &PosInterp,
    m_pos: usize,
    proof: &CopyProof,
    t: &mut Transcript,
) -> Result<CopyTerminal, &'static str> {
    let z = challenge_nonzero(t);
    t.absorb_field("copy-theta", &[proof.theta]);
    let log_m = m_pos.trailing_zeros() as usize;
    if log_m < LOG_KPOS {
        return Err("copy: operand too small");
    }
    let mut out_claims: Vec<CopyClaim> = Vec::new();
    for unfold in &proof.unfolds {
        if unfold.e_tree.len() != log_m {
            return Err("copy: e-tree level count");
        }
        if unfold.f_trans.len() != LOG_KPOS {
            return Err("copy: transition count");
        }
        // ---- the E-tree.
        let mut cur_pt: Vec<Gf128> = Vec::new();
        let mut cur_v = proof.theta;
        for (level, cubic) in unfold.e_tree.iter().enumerate() {
            let l_len = m_pos >> (log_m - level);
            let xi = ts_msbf(&cur_pt);
            if xi.len() != l_len {
                return Err("copy: e-tree level length");
            }
            let (y, v_u, v_d) = cubic_verify(cur_v, &xi, cubic, t)?;
            let y_p = t.challenge_field("copy-yp");
            cur_v = mul(ONE + y_p, v_u) + mul(y_p, v_d);
            let mut pt = Vec::with_capacity(1 + y.len());
            pt.push(y_p);
            pt.extend_from_slice(&y);
            cur_pt = pt;
        }
        let col_pt: Vec<Gf128> = cur_pt[..LOG_KPOS].to_vec();
        let row_pt: Vec<Gf128> = cur_pt[LOG_KPOS..].to_vec();
        let r_2 = expand_factors(&msbf_to_factors(&col_pt));

        // ---- the transitions (τ = 1 … 0, top-first in f_trans).
        let mut v_next = cur_v;
        let mut l_next = ts_msbf(&row_pt);
        let mut l_pt_next = row_pt;
        let mut r_next = r_2;
        for (idx, trans) in unfold.f_trans.iter().enumerate() {
            let tau = LOG_KPOS - 1 - idx;
            let rows_half = m_pos >> (tau + 1);
            if l_next.len() != rows_half {
                return Err("copy: l length");
            }
            t.absorb_field("copy-u", &trans.u);
            if dot_gf(&trans.u, &r_next) != v_next {
                return Err("copy: v_{τ+1} = 〈u, r_{τ+1}〉 failed");
            }
            t.absorb_field("copy-h", &trans.h);
            let x = challenge_outside_h(f, interp, t);
            let rho = interp.rho_at_e(f, x);
            let c_x = dot_gf(&trans.u, &rho);
            let psi_x = interp.psi_at_e(f, x);
            let mut hv = ZERO;
            for &c in trans.h.iter().rev() {
                hv = mul(hv, x) + c;
            }
            let claim = c_x + mul(psi_x, hv);
            let xi = ts_msbf(&l_pt_next);
            if xi.len() != l_next.len() {
                return Err("copy: l materialisation");
            }
            let (y, v_u, v_d) = cubic_verify(claim, &xi, &trans.cubic, t)?;
            let y_p = t.challenge_field("copy-yp");
            v_next = mul(ONE + y_p, v_u) + mul(y_p, v_d);
            let mut pt = Vec::with_capacity(1 + y.len());
            pt.push(y_p);
            pt.extend_from_slice(&y);
            l_pt_next = pt;
            l_next = ts_msbf(&l_pt_next);
            r_next = rho.to_vec();
        }
        if l_next.len() != m_pos {
            return Err("copy: terminal l₀ length");
        }
        // partition-of-unity hygiene
        let l_sum = l_next.iter().fold(ZERO, |acc, &v| acc + v);
        let r_sum = r_next.iter().fold(ZERO, |acc, &v| acc + v);
        if l_sum != ONE || r_sum != ONE {
            return Err("copy: weight vectors must sum to one");
        }
        let claim = v_next + z;
        let l0_factors = msbf_to_factors(&l_pt_next);
        let mut r4 = [ZERO; KPOS];
        r4.copy_from_slice(&r_next[..KPOS]);
        out_claims.push((claim, l0_factors, r4));
    }
    let claim_b = out_claims.pop().expect("two operands");
    let claim_a = out_claims.pop().expect("two operands");
    Ok(CopyTerminal {
        claims: [claim_a, claim_b],
    })
}

// ---------------------------------------------------------------------------
// Proposition 5 — the identity-label weighted evaluation (verifier side)
// ---------------------------------------------------------------------------

/// `〈ι ⊔ 0, r ⊗ l〉` — the padded identity labels' weighted evaluation,
/// computed directly from the position bits in `O(κ_pos + log L)`:
/// `code(ι_n) = n` splits into the column part `(j ≪ log_m)` and the row
/// part `r`, so
///
/// ```text
/// 〈ι, r⊗l〉 = 〈ι_row, l〉·(Σ_{j<3} r_j) + (Σ_r l_r)·〈ι_col, r〉
///           + ρ_3·(〈ι_row, l⟩_{io} + ι_col(3)·(Σ_{i<n_io} l_i))
/// ```
///
/// with the row part summed by the Prop-5 bit formula and column 3 (the
/// public-I/O segment) restricted to its `n_io`-long prefix.
///
/// * `l_factors` — the terminal `l₀`'s little-endian two-entry factors
///   (factor `k` ↔ row bit `k`).
/// * `rho` — the terminal column weight `ρ(x₀) ∈ E⁴`.
/// * `n_io` — the number of public-I/O positions (the retained prefix of
///   column 3; positions beyond are zero-padded).
pub fn iota_eval_padded(
    f: &Gf32,
    l_factors: &[(Gf128, Gf128)],
    rho: &[Gf128; KPOS],
    n_io: usize,
) -> Gf128 {
    let log_m = l_factors.len();
    // Position n = j·m_pos + r: the top 2 bits are the column (segment),
    // the low log_m bits the row — code(n) = (j ≪ log_m) | r.
    // ι_col[j] = embed(j ≪ log_m) — the column part of the label.
    let col_label = |j: usize| -> Gf128 {
        if j == 0 {
            ZERO
        } else {
            f.embed((j as u32) << log_m)
        }
    };
    // The row part: 〈ι_row, l〉 where ι_row[r] = embed(r). By Prop 5's
    // bit formula, Σ_r l_r·embed(r) = Σ_b h^b·(Σ_{r: bit_b(r)=1} l_r) with
    // the bit-sums = f_{b,1}·(prefix_b)·(suffix_{b+1}) over the factors.
    let row_dot_full = {
        let mut suffix = vec![ONE; log_m + 1];
        for b in (0..log_m).rev() {
            suffix[b] = mul(suffix[b + 1], l_factors[b].0 + l_factors[b].1);
        }
        let mut acc = ZERO;
        let mut prefix = ONE;
        for b in 0..log_m {
            let partial = mul(mul(l_factors[b].1, prefix), suffix[b + 1]);
            if partial != ZERO {
                acc += mul(f.embed(1u32 << b), partial);
            }
            prefix = mul(prefix, l_factors[b].0 + l_factors[b].1);
        }
        acc
    };
    // l's total sum (the partition-of-unity value 1 for honest weights)
    let l_sum: Gf128 = l_factors
        .iter()
        .fold(ONE, |acc, &(f0, f1)| mul(acc, f0 + f1));

    // the FULL columns 0..3 (j < 3): [Σ_{j<3} ρ_j]·row_dot + l_sum·[Σ_{j<3}
    // ρ_j·ι_col(j)]
    let r_sum: Gf128 = rho[..3].iter().fold(ZERO, |acc, &v| acc + v);
    let mut col_dot = ZERO;
    for j in 0..3 {
        col_dot += mul(rho[j], col_label(j));
    }
    let mut total = mul(row_dot_full, r_sum) + mul(l_sum, col_dot);

    // column 3 (the public-I/O segment): positions 3·m_pos + r, r < n_io
    // (the rest is zero padding); the label is col_label(3) + embed(r).
    if n_io > 0 && rho[3] != ZERO {
        // l[r] for r < n_io by the truncated tensor expansion
        let mut l_vals = vec![ONE];
        for b in 0..log_m {
            let half = l_vals.len();
            let keep = n_io.min(half * 2);
            let mut next = vec![ZERO; keep];
            for i in 0..keep.min(half) {
                next[i] = mul(l_vals[i], l_factors[b].0);
            }
            for i in half..keep {
                next[i] = mul(l_vals[i - half], l_factors[b].1);
            }
            l_vals = next;
        }
        let mut io_row = ZERO;
        let mut io_sum = ZERO;
        for (r, &lv) in l_vals.iter().enumerate() {
            io_row += mul(lv, f.embed(r as u32));
            io_sum += lv;
        }
        total += mul(rho[3], mul(io_sum, col_label(3)) + io_row);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_hash::Transcript;

    thread_local! {
        static RNG: std::cell::Cell<u64> = const { std::cell::Cell::new(0x1bad_c0de_f00d_bee5) };
    }

    fn word() -> u32 {
        RNG.with(|c| {
            let mut r = c.get();
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            c.set(r);
            r as u32
        })
    }

    /// The dense E-arithmetic reference for the grand product: Θ =
    /// Π_n (z + embed(code_n)) over the operand (any length).
    fn grand_product_dense(f: &Gf32, codes: &[u32], z: Gf128) -> Gf128 {
        let mut acc = ONE;
        for &c in codes {
            acc = mul(acc, z + f.embed(c));
        }
        acc
    }

    #[test]
    fn interp_tables_are_sound() {
        let f = Gf32::new();
        let interp = PosInterp::new(&f);
        // Ψ vanishes on H
        for j in 0..4u32 {
            let mut acc = 0u32;
            for &c in interp.psi.iter().rev() {
                acc = f.mul(acc, j) ^ c;
            }
            assert_eq!(acc, 0);
        }
        // ψ_j(γ_k) = δ_jk over K_pos
        for j in 0..4 {
            for k in 0..4u32 {
                let v = {
                    let mut acc = 0u32;
                    for &c in interp.psi_monomials[j].iter().rev() {
                        acc = f.mul(acc, k) ^ c;
                    }
                    acc
                };
                assert_eq!(v, u32::from(j as u32 == k));
            }
        }
        // the off-domain ψ evaluations satisfy the partition of unity
        for node in 0..3 {
            let mut sum = ZERO;
            for j in 0..4 {
                sum += f.embed(interp.psi_off[node][j]);
            }
            assert_eq!(sum, ONE);
        }
        // ρ(x) at K_pos points matches the codes
        {
            let x = f.embed(interp.nodes_off[0]);
            let rho = interp.rho_at_e(&f, x);
            assert_eq!(rho[0], f.embed(interp.psi_off[0][0]));
        }
    }

    #[test]
    fn tree_flatten_matches_dense() {
        let f = Gf32::new();
        for &log_m in &[2usize, 3, 5] {
            let m_pos = 1usize << log_m;
            let codes: Vec<u32> = (0..m_pos * KPOS).map(|_| word()).collect();
            let tree = CopyTree::build(&f, &codes, m_pos);
            let z = Gf128(0x9e3779b97f4a7c15u128.wrapping_mul(3));
            let flat = tree.flatten_top(&f, z);
            // dense reference: per (r, j), Σ_d embed(C₂[j][d][r])·z^d
            let rows = m_pos >> LOG_KPOS;
            for r in 0..rows {
                for j in 0..KPOS {
                    let mut acc = ZERO;
                    for d in 0..5usize {
                        acc += mul(f.embed(tree.levels[LOG_KPOS][j][d][r]), pow_z(z, d));
                    }
                    assert_eq!(flat[j * rows + r], acc, "log_m={log_m} r={r} j={j}");
                }
            }
            // the grand-product invariant: Θ(tree) = Π(z + embed(codes))
            let theta = {
                let mut cur = flat;
                while cur.len() > 1 {
                    let half = cur.len() / 2;
                    let mut next = Vec::with_capacity(half);
                    for i in 0..half {
                        next.push(mul(cur[i], cur[half + i]));
                    }
                    cur = next;
                }
                cur[0]
            };
            assert_eq!(
                theta,
                grand_product_dense(&f, &codes, z),
                "the grand product at log_m={log_m}"
            );
        }
    }

    fn pow_z(z: Gf128, d: usize) -> Gf128 {
        let mut acc = ONE;
        for _ in 0..d {
            acc = mul(acc, z);
        }
        acc
    }

    #[test]
    fn copy_roundtrip_and_tamper() {
        let f = Gf32::new();
        let interp = PosInterp::new(&f);
        for &log_m in &[2usize, 4, 6] {
            let m_pos = 1usize << log_m;
            // a genuine permutation of the multiset: permute the operand
            let base: Vec<u32> = (0..m_pos * KPOS).map(|_| word()).collect();
            let mut perm: Vec<usize> = (0..m_pos * KPOS).collect();
            for i in (1..perm.len()).rev() {
                let j = (word() as usize) % (i + 1);
                perm.swap(i, j);
            }
            let b_codes: Vec<u32> = perm.iter().map(|&p| base[p]).collect();
            let mut t = Transcript::new("mikros-copy-test", &[]);
            let (proof, terminal) = prove_copy(&f, &interp, &base, &b_codes, m_pos, &mut t);
            let mut tv = Transcript::new("mikros-copy-test", &[]);
            let got = verify_copy(&f, &interp, m_pos, &proof, &mut tv).expect("verify");
            assert_eq!(got.claims[0].0, terminal.claims[0].0);
            assert_eq!(got.claims[1].0, terminal.claims[1].0);
            assert_eq!(got.claims[0].1, terminal.claims[0].1);
            assert_eq!(got.claims[0].2, terminal.claims[0].2);
            assert_eq!(got.claims[1].2, terminal.claims[1].2);
        }
        // tamper: corrupted messages must fail
        let m_pos = 16;
        let base: Vec<u32> = (0..m_pos * KPOS).map(|_| word()).collect();
        let mut perm: Vec<usize> = (0..m_pos * KPOS).collect();
        for i in (1..perm.len()).rev() {
            let j = (word() as usize) % (i + 1);
            perm.swap(i, j);
        }
        let b_codes: Vec<u32> = perm.iter().map(|&p| base[p]).collect();
        let mut t = Transcript::new("mikros-copy-test", &[]);
        let (proof, _) = prove_copy(&f, &interp, &base, &b_codes, m_pos, &mut t);
        // corrupt the first transition's u claims
        let mut bad = clone_copy_proof(&proof);
        bad.unfolds[0].f_trans[0].u[1] += ONE;
        let mut tv = Transcript::new("mikros-copy-test", &[]);
        assert!(verify_copy(&f, &interp, m_pos, &bad, &mut tv).is_err());
        // corrupt h
        let mut bad = clone_copy_proof(&proof);
        bad.unfolds[1].f_trans[1].h[0] += ONE;
        let mut tv = Transcript::new("mikros-copy-test", &[]);
        assert!(verify_copy(&f, &interp, m_pos, &bad, &mut tv).is_err());
        // corrupt a cubic round message (pick a level that has rounds —
        // the E-tree's deepest levels reduce length-1 vectors)
        let mut bad = clone_copy_proof(&proof);
        let mut tampered = false;
        for level in bad.unfolds[0].e_tree.iter_mut() {
            if !level.rounds.is_empty() {
                level.rounds[0][1] += ONE;
                tampered = true;
                break;
            }
        }
        assert!(tampered, "some E-tree level has rounds");
        let mut tv = Transcript::new("mikros-copy-test", &[]);
        assert!(verify_copy(&f, &interp, m_pos, &bad, &mut tv).is_err());
    }

    #[test]
    fn iota_eval_matches_dense() {
        let f = Gf32::new();
        for &log_m in &[3usize, 5, 7] {
            let m_pos = 1usize << log_m;
            // random tensor factors for l₀ (over the log_m row bits)
            let factors: Vec<(Gf128, Gf128)> = (0..log_m)
                .map(|_| (Gf128(word() as u128), Gf128(word() as u128)))
                .collect();
            let l_vals = expand_factors(&factors);
            assert_eq!(l_vals.len(), m_pos);
            let rho: [Gf128; KPOS] = std::array::from_fn(|_| Gf128(word() as u128));
            for &n_io in &[0usize, 3, m_pos.min(17)] {
                // dense: code(ι_n) = n for n < L = 3·m_pos + n_io, else 0;
                // position n = j·m_pos + r over the padded length 4·m_pos
                let l_full = 3 * m_pos + n_io;
                let mut want = ZERO;
                for j in 0..KPOS {
                    for r in 0..m_pos {
                        let n = j * m_pos + r;
                        if n < l_full {
                            want += mul(mul(l_vals[r], rho[j]), f.embed(n as u32));
                        }
                    }
                }
                let got = iota_eval_padded(&f, &factors, &rho, n_io);
                assert_eq!(got, want, "log_m={log_m} n_io={n_io}");
            }
        }
    }
}
