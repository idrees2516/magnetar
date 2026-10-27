//! Algorithm 3 — the packed permutation proof (ePrint 2026/2646, §4.3).
//!
//! Proves that two committed packed matrices `A, B ∈ F^{m×κ}` contain the
//! same multiset of entries (`a ≅_perm b`) via **grand-product
//! fingerprints**: after the commitments are fixed the verifier samples
//! `z ←$ E` and the prover proves
//!
//! ```text
//! Θ = GrandProd(z + a) = GrandProd(z + b)
//! ```
//!
//! **The product tree** (Eq. 11): `C₀(Z) = Z·1 + D` over `F`, then
//! `C_{τ+1}(Z) = C_{τ,U}(Z) ∘ C_{τ,D}(Z)` halving the rows at each level —
//! `C_τ ∈ F[Z]^{(m/2^τ)×κ}` of degree `2^τ`. At `τ = log2 κ = 7` each entry
//! is the monic degree-κ row product; the matrix is flattened column-major
//! into `c₇(z) ∈ E^m` and the remaining levels are pointwise products over
//! `E` up to the root `Θ`.
//!
//! **The proof unfolds the tree in reverse.** The `E`-tree levels reduce
//! `Θ` through `log2 m` *cubic weighted sum-checks* `v = 〈ξ, c_U ∘ c_D〉`
//! to the claim `v₇ = l₇^⊤ C₇(z) r₇` — the accumulated tensor weight
//! factoring into the column weight `r₇` (the flatten's high bits) and
//! the row weight `l₇`. Each coefficient-tree transition `τ = 6 … 0` then
//! sends the column claims `u_{τ+1} = l_{τ+1}^⊤ C_{τ+1}(z)` and the
//! quotient `h = f/Ψ` of the §4.2 Lagrange residual — the off-domain
//! samples come from the `η_{τ+1,j}(Z)` half-row polynomial products of
//! the encodings, aggregated against `l_{τ+1}` through the GF(2^8)
//! product table — then samples `x ←$ E\H` and applies one more cubic
//! sum-check to
//!
//! ```text
//! 〈l_{τ+1}, g_{τ,U}(z,x) ∘ g_{τ,D}(z,x)〉 = g_C(x) + Ψ(x)h(x)
//! ```
//!
//! At level zero both weight vectors sum to one and `C₀(z) = z + D`,
//! leaving the terminal packed-PCS claims `v₀ − z = l₀^⊤ D ρ(x)` (one per
//! operand), proved by Algorithm 4.
//!
//! Prover cost: `O(Nκ log2 κ + κ⁴)` base-field operations (Prop. 3); the
//! coefficient tensors cost ~10·(N/8) bytes across all levels.

use crate::gf8::Gf8;
use crate::interp::{InterpData, KAPPA};
use crate::planes::PlaneSet;
use magnetar_field::Gf128;
use magnetar_field::{mul, ONE, ZERO};
use magnetar_hash::Transcript;

/// `log2 κ = 7` — the coefficient-tree depth.
pub const LOG_KAPPA: usize = 7;

// ---------------------------------------------------------------------------
// The cubic weighted sum-check (the §4.3 reduction engine)
// ---------------------------------------------------------------------------

/// One cubic sum-check: `v = 〈ξ, cu ∘ cd〉` over `E^L`.
///
/// The round message is `q(χ) = Σ ξ_r(χ)·cu_r(χ)·cd_r(χ)` (individual
/// degree 3) with the characteristic-2 Boolean check `Σ_{i≥1} q_i =
/// running`; the challenges bind the MSB first (Fact 1's convention). At
/// the terminal the prover sends the two child MLEs and the verifier
/// checks `running = ξ̂(y)·v_U·v_D`.
#[derive(Clone)]
pub struct CubicProof {
    pub rounds: Vec<[Gf128; 4]>,
    pub v_child_u: Gf128,
    pub v_child_d: Gf128,
    /// The fold challenges in bind order (the verifier re-derives them;
    /// carried for the prover's point bookkeeping).
    pub challenges: Vec<Gf128>,
}

fn triple_dot(x: &[Gf128], y: &[Gf128], z: &[Gf128]) -> Gf128 {
    let mut acc = ZERO;
    for i in 0..x.len() {
        acc += mul(mul(x[i], y[i]), z[i]);
    }
    acc
}

/// Prove `v = 〈ξ, cu ∘ cd〉` (all three length `L = 2^d`), folding in
/// place. Absorbs the messages, samples the challenges, and returns the
/// proof.
pub fn cubic_prove(
    xi: &mut Vec<Gf128>,
    cu: &mut Vec<Gf128>,
    cd: &mut Vec<Gf128>,
    v: Gf128,
    t: &mut Transcript,
) -> CubicProof {
    let mut running = v;
    let mut rounds = Vec::new();
    let mut y_local = Vec::new();
    let mut len = xi.len();
    debug_assert_eq!(cu.len(), len);
    debug_assert_eq!(cd.len(), len);
    while len > 1 {
        let half = len / 2;
        let (xi_lo, xi_hi) = xi.split_at_mut(half);
        let (cu_lo, cu_hi) = cu.split_at_mut(half);
        let (cd_lo, cd_hi) = cd.split_at_mut(half);
        // q(χ) with x_χ = x_lo + χ·(x_lo + x_hi):
        //   q0 = 〈ξ_lo, cu_lo ∘ cd_lo〉,    q3 = 〈ξ̃, c̃u ∘ c̃d〉,
        //   q1 = 〈ξ̃,cu_lo∘cd_lo〉+〈ξ_lo,cũ∘cd_lo〉+〈ξ_lo,cu_lo∘c̃d〉,
        //   q2 = 〈ξ̃,cũ∘cd_lo〉+〈ξ̃,cu_lo∘c̃d〉+〈ξ_lo,cũ∘c̃d〉.
        let xit: Vec<Gf128> = xi_lo
            .iter()
            .zip(xi_hi.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let cut: Vec<Gf128> = cu_lo
            .iter()
            .zip(cu_hi.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let cdt: Vec<Gf128> = cd_lo
            .iter()
            .zip(cd_hi.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let q0 = triple_dot(xi_lo, cu_lo, cd_lo);
        let q3 = triple_dot(&xit, &cut, &cdt);
        let q1 = triple_dot(&xit, cu_lo, cd_lo)
            + triple_dot(xi_lo, &cut, cd_lo)
            + triple_dot(xi_lo, cu_lo, &cdt);
        let q2 = triple_dot(&xit, &cut, cd_lo)
            + triple_dot(&xit, cu_lo, &cdt)
            + triple_dot(xi_lo, &cut, &cdt);
        t.absorb_field("perm-cub", &[q0, q1, q2, q3]);
        let lambda = t.challenge_field("perm-cub-r");
        for i in 0..half {
            xi_lo[i] += mul(lambda, xit[i]);
            cu_lo[i] += mul(lambda, cut[i]);
            cd_lo[i] += mul(lambda, cdt[i]);
        }
        xi.truncate(half);
        cu.truncate(half);
        cd.truncate(half);
        let l2 = mul(lambda, lambda);
        running = q0 + mul(lambda, q1) + mul(l2, q2) + mul(mul(l2, lambda), q3);
        rounds.push([q0, q1, q2, q3]);
        y_local.push(lambda);
        len = half;
    }
    debug_assert_eq!(
        running,
        mul(mul(xi[0], cu[0]), cd[0]),
        "cubic terminal (the prover's own fold)"
    );
    let challenges = y_local.clone();
    let proof = CubicProof {
        rounds,
        v_child_u: cu[0],
        v_child_d: cd[0],
        challenges,
    };
    t.absorb_field("perm-child", &[proof.v_child_u, proof.v_child_d]);
    proof
}

/// Verify: returns `(y, v_U, v_D)`. `xi_init` is the verifier's own
/// materialisation of the public weight; it is folded with the transcript
/// challenges to check the terminal product.
pub fn cubic_verify(
    v: Gf128,
    xi_init: &[Gf128],
    proof: &CubicProof,
    t: &mut Transcript,
) -> Result<(Vec<Gf128>, Gf128, Gf128), &'static str> {
    if !xi_init.len().is_power_of_two() {
        return Err("perm-cubic: weight length");
    }
    let expected_rounds = xi_init.len().trailing_zeros() as usize;
    if proof.rounds.len() != expected_rounds {
        return Err("perm-cubic: round count");
    }
    let mut running = v;
    let mut y = Vec::with_capacity(expected_rounds);
    let mut cur = xi_init.to_vec();
    for &[q0, q1, q2, q3] in &proof.rounds {
        if q1 + q2 + q3 != running {
            return Err("perm-cubic: round sum");
        }
        t.absorb_field("perm-cub", &[q0, q1, q2, q3]);
        let lambda = t.challenge_field("perm-cub-r");
        let l2 = mul(lambda, lambda);
        running = q0 + mul(lambda, q1) + mul(l2, q2) + mul(mul(l2, lambda), q3);
        y.push(lambda);
        let half = cur.len() / 2;
        for i in 0..half {
            cur[i] = cur[i] + mul(lambda, cur[i] + cur[half + i]);
        }
        cur.truncate(half);
    }
    t.absorb_field("perm-child", &[proof.v_child_u, proof.v_child_d]);
    let xi_hat = cur[0];
    if running != mul(mul(xi_hat, proof.v_child_u), proof.v_child_d) {
        return Err("perm-cubic: terminal product");
    }
    Ok((y, proof.v_child_u, proof.v_child_d))
}

/// The tensor-sum weight with the coordinates in **MSB-first bind order**
/// (coord 0 ↔ the index MSB — Fact 1's halving convention). The growing
/// recursion applies the LAST coordinate first (it is the index LSB).
pub fn ts_msbf(pt: &[Gf128]) -> Vec<Gf128> {
    let mut words = vec![ONE];
    for &p in pt.iter().rev() {
        let half = words.len();
        let mut next = vec![ZERO; half * 2];
        for i in 0..half {
            next[i] = mul(words[i], ONE + p);
            next[half + i] = mul(words[i], p);
        }
        words = next;
    }
    words
}

/// The little-endian two-entry factors of an MSB-first point (for
/// `RowWeightTensor`-style expansion and Algorithm 4).
pub fn msbf_to_factors(pt: &[Gf128]) -> Vec<(Gf128, Gf128)> {
    pt.iter().rev().map(|&p| (ONE + p, p)).collect()
}

/// The little-endian expansion of MSB-first factors into a weight vector.
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

// ---------------------------------------------------------------------------
// The coefficient tensors C_τ(Z)
// ---------------------------------------------------------------------------

/// One tree level's coefficient planes: `nplanes = κ·(2^τ+1)` planes of
/// `wp = ceil(rows/64)` words each (the row tail bits stay zero).
#[derive(Clone)]
struct LevelPlanes {
    rows: usize,
    wp: usize,
    planes: Vec<u64>,
}

impl LevelPlanes {
    fn zeros(rows: usize, nplanes: usize) -> Self {
        let wp = rows.div_ceil(64);
        LevelPlanes {
            rows,
            wp,
            planes: vec![0u64; nplanes * wp],
        }
    }

    #[inline(always)]
    fn plane(&self, i: usize) -> &[u64] {
        &self.planes[i * self.wp..(i + 1) * self.wp]
    }

    #[inline(always)]
    fn plane_mut(&mut self, i: usize) -> &mut [u64] {
        let wp = self.wp;
        &mut self.planes[i * wp..(i + 1) * wp]
    }

    #[inline(always)]
    fn bit(&self, i: usize, r: usize) -> bool {
        let p = self.plane(i);
        (p[r / 64] >> (r % 64)) & 1 == 1
    }
}

/// The `k`-th word of the UPPER half (`rows [0, rows_half)`) of a plane.
#[inline(always)]
fn half_word_u(pl: &[u64], rows_half: usize, k: usize) -> u64 {
    if 64 * k >= rows_half {
        return 0;
    }
    let raw = pl.get(k).copied().unwrap_or(0);
    if 64 * k + 64 <= rows_half {
        raw
    } else {
        raw & ((1u64 << (rows_half - 64 * k)) - 1)
    }
}

/// The `k`-th word of the LOWER half (`rows [rows_half, rows_total)`).
#[inline(always)]
fn half_word_d(pl: &[u64], rows_total: usize, rows_half: usize, k: usize) -> u64 {
    let base = rows_half + 64 * k;
    if base >= rows_total {
        return 0;
    }
    let w0 = base / 64;
    let sh = (base % 64) as u32;
    let lo = pl.get(w0).copied().unwrap_or(0);
    if sh == 0 {
        lo
    } else {
        let hi = pl.get(w0 + 1).copied().unwrap_or(0);
        (lo >> sh) | (hi << (64 - sh))
    }
}

/// The GF(2) coefficient tensors of the product tree, levels 0..=7.
pub struct CoefTree {
    levels: Vec<LevelPlanes>,
    pub m: usize,
}

impl CoefTree {
    /// Build from the operand planes `D` (the `m×κ` packed witness, its
    /// column planes). Plane `j·(2^τ+1) + d` of level `τ` holds the
    /// `Z^d`-coefficient of column `j`.
    pub fn build(d: &PlaneSet) -> Self {
        let m = d.m;
        assert_eq!(d.kappa, KAPPA);
        assert_eq!(m % (1 << LOG_KAPPA), 0, "m must be a multiple of κ");
        let mut levels = Vec::with_capacity(LOG_KAPPA + 1);
        // C₀(Z) = Z·1 + D: coefficient 0 = D, coefficient 1 = the ones row.
        let mut c0 = LevelPlanes::zeros(m, KAPPA * 2);
        for j in 0..KAPPA {
            let src = d.plane(j);
            c0.plane_mut(j * 2)[..src.len()].copy_from_slice(src);
            c0.plane_mut(j * 2 + 1)
                .iter_mut()
                .for_each(|w| *w = u64::MAX);
        }
        levels.push(c0);
        for tau in 0..LOG_KAPPA {
            let prev = &levels[tau];
            let rows_in = prev.rows; // m / 2^τ
            let ncoef_in = (1 << tau) + 1;
            let rows_out = rows_in / 2;
            let ncoef_out = ncoef_in * 2 - 1; // 2^{τ+1} + 1
            let mut cur = LevelPlanes::zeros(rows_out, KAPPA * ncoef_out);
            let wp_out = cur.wp;
            for j in 0..KAPPA {
                for d1 in 0..ncoef_in {
                    let pu = prev.plane(j * ncoef_in + d1);
                    for d2 in 0..ncoef_in {
                        let pd = prev.plane(j * ncoef_in + d2);
                        let po = cur.plane_mut(j * ncoef_out + d1 + d2);
                        for k in 0..wp_out {
                            po[k] ^= half_word_u(pu, rows_out, k)
                                & half_word_d(pd, rows_in, rows_out, k);
                        }
                    }
                }
            }
            levels.push(cur);
        }
        CoefTree { levels, m }
    }

    #[inline(always)]
    fn ncoef(tau: usize) -> usize {
        (1 << tau) + 1
    }

    #[inline(always)]
    fn plane_idx(tau: usize, j: usize, dd: usize) -> usize {
        j * Self::ncoef(tau) + dd
    }

    /// The plane bit `(column j, coefficient d)` at level `tau`, row `r`.
    #[inline]
    pub fn level_bit(&self, tau: usize, j: usize, d: usize, r: usize) -> bool {
        self.levels[tau].bit(Self::plane_idx(tau, j, d), r)
    }

    /// `C_7(z)` flattened column-major: `c7[j·(m/κ) + r] = Σ_d C₇[d][r][j]·z^d`.
    pub fn flatten_top(&self, z: Gf128) -> Vec<Gf128> {
        let top = &self.levels[LOG_KAPPA];
        let rows = top.rows; // m / κ
        let ncoef = Self::ncoef(LOG_KAPPA);
        let mut zp = Vec::with_capacity(ncoef);
        let mut p = ONE;
        for _ in 0..ncoef {
            zp.push(p);
            p = mul(p, z);
        }
        let mut out = vec![ZERO; rows * KAPPA];
        for j in 0..KAPPA {
            for r in 0..rows {
                let mut acc = ZERO;
                for d in 0..ncoef {
                    if top.bit(Self::plane_idx(LOG_KAPPA, j, d), r) {
                        acc += zp[d];
                    }
                }
                out[j * rows + r] = acc;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// The per-transition prover kernels
// ---------------------------------------------------------------------------

/// The l-planes for the bitsliced aggregation: coordinate `e`'s plane of
/// the weight vector `l` (any length; tail-safe).
fn weight_planes(l: &[Gf128]) -> (Vec<u64>, usize) {
    let wp = l.len().div_ceil(64);
    let mut planes = vec![0u64; 128 * wp];
    for (r, &lv) in l.iter().enumerate() {
        let mut bits = lv.0;
        while bits != 0 {
            let e = bits.trailing_zeros() as usize;
            planes[e * wp + r / 64] |= 1u64 << (r % 64);
            bits &= bits - 1;
        }
    }
    (planes, wp)
}

/// `u_{τ+1,j} = Σ_r l_r·C_{τ+1}[d][r][j]·z^d` — the aggregate-then-evaluate
/// column claims (§4.3: "aggregating Z-coefficients before evaluating").
#[allow(clippy::too_many_arguments)]
fn column_claims_u(
    tree: &CoefTree,
    tau1: usize, // the level τ+1 index
    l: &[Gf128],
    l_planes: &[u64],
    l_wp: usize,
    zp: &[Gf128],
) -> Vec<Gf128> {
    let level = &tree.levels[tau1];
    let ncoef = ncoef_at(tau1);
    debug_assert_eq!(level.rows, l.len());
    debug_assert_eq!(l_wp, level.wp);
    let mut u = vec![ZERO; KAPPA];
    for j in 0..KAPPA {
        let mut acc = ZERO;
        for d in 0..ncoef {
            let pl = level.plane(CoefTree::plane_idx(tau1, j, d));
            let mut bits = 0u128;
            for e in 0..128 {
                let lp = &l_planes[e * l_wp..(e + 1) * l_wp];
                if PlaneSet::parity_dot(pl, lp) == 1 {
                    bits |= 1u128 << e;
                }
            }
            acc += mul(Gf128(bits), zp[d]);
        }
        u[j] = acc;
    }
    u
}

#[inline]
fn ncoef_at(tau: usize) -> usize {
    (1 << tau) + 1
}

/// `g_τ(z,x)[r] = Σ_{d,j} C_τ[d][r][j]·z^d·ρ_j(x)` — contract the columns
/// with `ρ(x)` per coefficient layer, then evaluate at `z`.
fn contract_g(tree: &CoefTree, tau: usize, rho: &[Gf128], zp: &[Gf128]) -> Vec<Gf128> {
    let level = &tree.levels[tau];
    let rows = level.rows;
    let ncoef = ncoef_at(tau);
    let wp = level.wp;
    let mut g = vec![ZERO; rows];
    for d in 0..ncoef {
        // the E-selector build: coordinate-e plane = XOR of the column
        // planes j with bit_e(ρ_j) set
        let mut planes = vec![0u64; 128 * wp];
        for e in 0..128 {
            let out = &mut planes[e * wp..(e + 1) * wp];
            for j in 0..KAPPA {
                if (rho[j].0 >> e) & 1 == 1 {
                    let sp = level.plane(CoefTree::plane_idx(tau, j, d));
                    for k in 0..wp {
                        out[k] ^= sp[k];
                    }
                }
            }
        }
        // g += (the transposed words)·z^d — tail-safe word assembly
        for r in 0..rows {
            let mut word = 0u128;
            for e in 0..128 {
                if (planes[e * wp + r / 64] >> (r % 64)) & 1 == 1 {
                    word |= 1u128 << e;
                }
            }
            g[r] += mul(Gf128(word), zp[d]);
        }
    }
    g
}

/// The off-domain `f`-samples of one transition. Per node `γ′`:
///
/// ```text
/// f(γ′) = Σ_{d1,d2} z^{d1+d2}·(Σ_r l_r·encU[d1][r]·encD[d2][r]) − gC(γ′)
///       = Σ_r l_r·(Σ_{d1} z^{d1}·encU[d1][r])·(Σ_{d2} z^{d2}·encD[d2][r]) − gC(γ′)
/// ```
///
/// The second form (the interchange) evaluates the K-coded encodings at
/// `z` FIRST — one GF(2^128) multiply per (coefficient, row) and then two
/// per row — instead of one per (row, coefficient-pair): ~8× fewer
/// extension-field multiplies per node. The pair-product route through the
/// GF(2^8) tables (Algorithm 2's step-4 shape) wins only on GFNI hardware
/// where byte products are single instructions; this route is the honest
/// scalar-optimized form, byte-identical in output.
#[allow(clippy::too_many_arguments)]
fn f_samples_eta(
    f8: &Gf8,
    data: &InterpData,
    tree: &CoefTree,
    tau: usize,
    l: &[Gf128],
    u: &[Gf128],
    zp: &[Gf128],
) -> Vec<Gf128> {
    let rows_half = l.len();
    let ncoef = ncoef_at(tau);
    let wp = rows_half.div_ceil(64);
    // hoisted scratch: the K-coded encodings and their byte spreads
    let mut enc_u: [Vec<u64>; 8] = std::array::from_fn(|_| vec![0u64; wp]);
    let mut enc_d: [Vec<u64>; 8] = std::array::from_fn(|_| vec![0u64; wp]);
    let mut spread_u: Vec<[u8; 64]> = vec![[0u8; 64]; wp];
    let mut spread_d: Vec<[u8; 64]> = vec![[0u8; 64]; wp];
    // the z-evaluated half encodings over the rows
    let mut gu: Vec<Gf128> = vec![ZERO; rows_half];
    let mut gd: Vec<Gf128> = vec![ZERO; rows_half];
    let mut out = Vec::with_capacity(data.nodes_off.len());
    for node in 0..data.nodes_off.len() {
        let row = &data.psi_off[node];
        let mut sel = [[0u64; 2]; 8];
        for j in 0..KAPPA {
            for c in 0..8 {
                let bit = ((row[j] >> c) & 1) as u64;
                sel[c][j / 64] |= bit << (j % 64);
            }
        }
        // ĜU[r] = Σ_d z^d·encU[d][r] (and the D half), the encodings
        // selector-XORed in K and spread to bytes once per coefficient.
        for v in gu.iter_mut() {
            *v = ZERO;
        }
        for v in gd.iter_mut() {
            *v = ZERO;
        }
        for d in 0..ncoef {
            encode_layer_into_u(tree, tau, d, &sel, rows_half, &mut enc_u);
            encode_layer_into_d(tree, tau, d, &sel, rows_half, &mut enc_d);
            for w in 0..wp {
                let words: [u64; 8] = std::array::from_fn(|c| enc_u[c][w]);
                spread_u[w] = crate::planes::spread8(&words);
                let words: [u64; 8] = std::array::from_fn(|c| enc_d[c][w]);
                spread_d[w] = crate::planes::spread8(&words);
            }
            let zd = zp[d];
            for w in 0..wp {
                let base = w * 64;
                let su = &spread_u[w];
                let sd = &spread_d[w];
                for rr in 0..64 {
                    let idx = base + rr;
                    if idx >= rows_half {
                        break;
                    }
                    gu[idx] += mul(f8.embed8[su[rr] as usize], zd);
                    gd[idx] += mul(f8.embed8[sd[rr] as usize], zd);
                }
            }
        }
        // f(γ′) = Σ_r l_r·ĜU[r]·ĜD[r] − gC(γ′)
        let mut f_val = ZERO;
        for r in 0..rows_half {
            f_val += mul(l[r], mul(gu[r], gd[r]));
        }
        let mut g_c = ZERO;
        for j in 0..KAPPA {
            g_c += mul(u[j], f8.embed(row[j]));
        }
        out.push(f_val + g_c); // char 2: minus = plus
    }
    out
}

/// `encode_layer_u` writing into caller-managed buffers (the per-node
/// allocation hoist).
fn encode_layer_into_u(
    tree: &CoefTree,
    tau: usize,
    d: usize,
    sel: &[[u64; 2]; 8],
    rows_half: usize,
    out: &mut [Vec<u64>; 8],
) {
    let level = &tree.levels[tau];
    let wp = rows_half.div_ceil(64);
    for c in 0..8 {
        let selc = sel[c];
        let oc = &mut out[c];
        for v in oc.iter_mut() {
            *v = 0;
        }
        for j in 0..KAPPA {
            if (selc[j / 64] >> (j % 64)) & 1 == 1 {
                let sp = level.plane(CoefTree::plane_idx(tau, j, d));
                for k in 0..wp {
                    oc[k] ^= half_word_u(sp, rows_half, k);
                }
            }
        }
    }
}

/// `encode_layer_d` writing into caller-managed buffers.
fn encode_layer_into_d(
    tree: &CoefTree,
    tau: usize,
    d: usize,
    sel: &[[u64; 2]; 8],
    rows_half: usize,
    out: &mut [Vec<u64>; 8],
) {
    let level = &tree.levels[tau];
    let rows_total = level.rows;
    let wp = rows_half.div_ceil(64);
    for c in 0..8 {
        let selc = sel[c];
        let oc = &mut out[c];
        for v in oc.iter_mut() {
            *v = 0;
        }
        for j in 0..KAPPA {
            if (selc[j / 64] >> (j % 64)) & 1 == 1 {
                let sp = level.plane(CoefTree::plane_idx(tau, j, d));
                for k in 0..wp {
                    oc[k] ^= half_word_d(sp, rows_total, rows_half, k);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The protocol
// ---------------------------------------------------------------------------

/// Per-phase timings (ms) — the grand-product cost anatomy (both operands).
#[derive(Default, Clone, Copy, Debug)]
pub struct PermStats {
    pub tree_build_ms: f64,
    pub e_vec_ms: f64,
    pub e_tree_ms: f64,
    pub u_ms: f64,
    pub eta_ms: f64,
    pub g_ms: f64,
    pub f_cubic_ms: f64,
    pub total_ms: f64,
}

/// The per-operand unfold data.
#[derive(Clone)]
pub struct OperandUnfold {
    /// `log2 m` E-tree levels, top-first.
    pub e_tree: Vec<CubicProof>,
    /// The 7 coefficient-tree transitions, `τ = 6 … 0`.
    pub f_trans: Vec<FTransition>,
}

#[derive(Clone)]
pub struct FTransition {
    /// `u_{τ+1} = l_{τ+1}^⊤ C_{τ+1}(z)` — the κ column claims.
    pub u: Vec<Gf128>,
    /// The 127 quotient coefficients `h = f/Ψ`.
    pub h: Vec<Gf128>,
    /// The transition's cubic sum-check.
    pub cubic: CubicProof,
}

pub struct PermProof {
    /// The claimed common fingerprint root.
    pub theta: Gf128,
    /// The unfold data per operand: `[a, b]`.
    pub unfolds: [OperandUnfold; 2],
}

/// One operand's terminal packed-PCS claim (an Algorithm-4 opening):
/// `(v₀ − z, the l₀ row factors (little-endian), ρ(x₀))`.
pub type OperandClaim = (Gf128, Vec<(Gf128, Gf128)>, Vec<Gf128>);

/// The terminal packed-PCS claims (Algorithm 4 openings), one per operand.
pub struct PermTerminal {
    pub claims: [OperandClaim; 2],
}

/// A transcript challenge outside `H` (`x ←$ E\H`, i.e. `Ψ(x) ≠ 0`).
fn challenge_outside_h(f8: &Gf8, data: &InterpData, t: &mut Transcript) -> Gf128 {
    loop {
        let x = t.challenge_field("perm-x");
        if data.psi_at_e(f8, x) != ZERO {
            return x;
        }
    }
}

/// Prove `a ≅_perm b` (Algorithm 3). The caller absorbs the commitments
/// before this call.
#[allow(clippy::type_complexity)]
pub fn prove_perm(
    f8: &Gf8,
    data: &InterpData,
    a: &PlaneSet,
    b: &PlaneSet,
    t: &mut Transcript,
) -> (PermProof, PermTerminal, PermStats) {
    use std::time::Instant;
    let t_all = Instant::now();
    let mut stats = PermStats::default();
    let m = a.m;
    debug_assert_eq!(b.m, m);
    let log_m = m.trailing_zeros() as usize;
    assert!(log_m >= LOG_KAPPA, "m ≥ κ (got m = {m})");

    let z = t.challenge_field("perm-z");
    let mut zp = Vec::with_capacity(KAPPA + 1);
    {
        let mut p = ONE;
        for _ in 0..=KAPPA {
            zp.push(p);
            p = mul(p, z);
        }
    }

    // ---- build both trees, both E-vector ladders, and the common root.
    struct Side {
        tree: CoefTree,
        c7: Vec<Gf128>,
        e_vecs: Vec<Vec<Gf128>>,
        theta: Gf128,
    }
    let mut sides: Vec<Side> = Vec::with_capacity(2);
    for operand in [a, b] {
        let t0 = Instant::now();
        let tree = CoefTree::build(operand);
        stats.tree_build_ms += t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let c7 = tree.flatten_top(z);
        let mut e_vecs: Vec<Vec<Gf128>> = Vec::with_capacity(log_m);
        let mut cur = c7.clone();
        for _ in 0..log_m {
            let half = cur.len() / 2;
            let mut next = Vec::with_capacity(half);
            for i in 0..half {
                next.push(mul(cur[i], cur[half + i]));
            }
            e_vecs.push(next.clone());
            cur = next;
        }
        let theta = cur[0];
        stats.e_vec_ms += t1.elapsed().as_secs_f64() * 1e3;
        sides.push(Side {
            tree,
            c7,
            e_vecs,
            theta,
        });
    }
    // the honest prover's invariant: the two grand products share the root
    debug_assert_eq!(
        sides[0].theta, sides[1].theta,
        "the permutation relation must hold (Θ_a = Θ_b)"
    );
    t.absorb_field("perm-theta", &[sides[0].theta]);

    let mut unfolds: Vec<OperandUnfold> = Vec::with_capacity(2);
    let mut terminal_claims: Vec<OperandClaim> = Vec::new();

    for side in &sides {
        // ---- the E-tree unfold: log2 m levels, top-first.
        let t_e = Instant::now();
        let mut e_proofs = Vec::with_capacity(log_m);
        let mut cur_pt: Vec<Gf128> = Vec::new(); // MSB-first
        let mut cur_v = side.theta;
        for level in 0..log_m {
            // level ℓ: c_{7+S−ℓ} → c_{7+S−ℓ−1}; the operands are the halves
            // of the LOWER (longer) vector, of length L = m >> (S−ℓ).
            let l_len = m >> (log_m - level);
            let vec: &Vec<Gf128> = if level == log_m - 1 {
                &side.c7
            } else {
                &side.e_vecs[log_m - level - 2]
            };
            debug_assert_eq!(vec.len(), 2 * l_len);
            let xi = ts_msbf(&cur_pt);
            debug_assert_eq!(xi.len(), l_len);
            let mut cu: Vec<Gf128> = vec[..l_len].to_vec();
            let mut cd: Vec<Gf128> = vec[l_len..].to_vec();
            let mut xi_v = xi;
            let cubic = cubic_prove(&mut xi_v, &mut cu, &mut cd, cur_v, t);
            let y = cubic.challenges.clone();
            let y_p = t.challenge_field("perm-yp");
            cur_v = mul(ONE + y_p, cubic.v_child_u) + mul(y_p, cubic.v_child_d);
            let mut pt = Vec::with_capacity(1 + y.len());
            pt.push(y_p);
            pt.extend_from_slice(&y);
            cur_pt = pt;
            e_proofs.push(cubic);
        }
        stats.e_tree_ms += t_e.elapsed().as_secs_f64() * 1e3;

        // ---- factor the final E-tree point (column-major flatten: the
        // first 7 coords are the column bits, the rest the row bits).
        debug_assert_eq!(cur_pt.len(), log_m);
        let col_pt: Vec<Gf128> = cur_pt[..LOG_KAPPA].to_vec();
        let row_pt: Vec<Gf128> = cur_pt[LOG_KAPPA..].to_vec();
        let r_7 = expand_factors(&msbf_to_factors(&col_pt));
        let mut v_next = cur_v;
        let mut l_next = ts_msbf(&row_pt);
        let mut l_pt_next = row_pt;
        let mut r_next = r_7;

        // ---- the F-transitions τ = 6 … 0.
        let mut f_trans = Vec::with_capacity(LOG_KAPPA);
        for tau in (0..LOG_KAPPA).rev() {
            let rows_half = m >> (tau + 1);
            debug_assert_eq!(l_next.len(), rows_half);
            let (l_planes, l_wp) = weight_planes(&l_next);

            // u_{τ+1}
            let t_u = Instant::now();
            let u = column_claims_u(&side.tree, tau + 1, &l_next, &l_planes, l_wp, &zp);
            stats.u_ms += t_u.elapsed().as_secs_f64() * 1e3;
            debug_assert_eq!(
                crate::r1cs::dot_gf(&u, &r_next),
                v_next,
                "v_{{τ+1}} = 〈u, r_{{τ+1}}〉 (the recursion's invariant)"
            );
            t.absorb_field("perm-u", &u);

            // the η samples → h
            let t_eta = Instant::now();
            let f_samples = f_samples_eta(f8, data, &side.tree, tau, &l_next, &u, &zp);
            let h = data.h_from_samples(f8, &f_samples);
            stats.eta_ms += t_eta.elapsed().as_secs_f64() * 1e3;
            t.absorb_field("perm-h", &h);

            // x ←$ E\H, then g_τ(z,x) and the cubic
            let x = challenge_outside_h(f8, data, t);
            let t_g = Instant::now();
            let rho = data.rho_at_e(f8, x);
            let g = contract_g(&side.tree, tau, &rho, &zp);
            stats.g_ms += t_g.elapsed().as_secs_f64() * 1e3;
            let c_x = crate::r1cs::dot_gf(&u, &rho);
            let psi_x = data.psi_at_e(f8, x);
            let mut hv = ZERO;
            for &c in h.iter().rev() {
                hv = mul(hv, x) + c;
            }
            let claim = c_x + mul(psi_x, hv);
            debug_assert_eq!(
                crate::r1cs::dot_gf(&l_next, &hadamard_words(&g[..rows_half], &g[rows_half..])),
                claim,
                "〈l, g_U ∘ g_D〉 = gC(x) + Ψ(x)h(x)"
            );
            let mut cu = g[..rows_half].to_vec();
            let mut cd = g[rows_half..].to_vec();
            let mut xi_v = l_next.clone();
            let t_cub = Instant::now();
            let cubic = cubic_prove(&mut xi_v, &mut cu, &mut cd, claim, t);
            stats.f_cubic_ms += t_cub.elapsed().as_secs_f64() * 1e3;
            let y = cubic.challenges.clone();
            let y_p = t.challenge_field("perm-yp");
            v_next = mul(ONE + y_p, cubic.v_child_u) + mul(y_p, cubic.v_child_d);
            let mut pt = Vec::with_capacity(1 + y.len());
            pt.push(y_p);
            pt.extend_from_slice(&y);
            l_pt_next = pt;
            l_next = ts_msbf(&l_pt_next);
            r_next = rho;
            f_trans.push(FTransition { u, h, cubic });
        }

        // ---- the terminal: v₀ − z = l₀^⊤ D ρ(x) (the packed-PCS claim).
        debug_assert_eq!(l_next.len(), m);
        let claim = v_next + z; // char 2: v₀ − z = v₀ + z
        let l0_factors = msbf_to_factors(&l_pt_next);
        terminal_claims.push((claim, l0_factors, r_next.clone()));
        unfolds.push(OperandUnfold {
            e_tree: e_proofs,
            f_trans,
        });
    }

    stats.total_ms = t_all.elapsed().as_secs_f64() * 1e3;
    let claim_b = terminal_claims.pop().expect("two operands");
    let claim_a = terminal_claims.pop().expect("two operands");
    let unfold_b = unfolds.pop().expect("two operands");
    let unfold_a = unfolds.pop().expect("two operands");
    (
        PermProof {
            theta: sides[0].theta,
            unfolds: [unfold_a, unfold_b],
        },
        PermTerminal {
            claims: [claim_a, claim_b],
        },
        stats,
    )
}

/// The pointwise product of two E-vectors (the g_U ∘ g_D identity check).
fn hadamard_words(u: &[Gf128], d: &[Gf128]) -> Vec<Gf128> {
    let mut out = Vec::with_capacity(u.len());
    for i in 0..u.len() {
        out.push(mul(u[i], d[i]));
    }
    out
}

/// Verify (Algorithm 3). Returns the two terminal packed claims for the
/// caller's Algorithm-4 openings.
#[allow(clippy::type_complexity)]
pub fn verify_perm(
    f8: &Gf8,
    data: &InterpData,
    proof: &PermProof,
    t: &mut Transcript,
) -> Result<PermTerminal, &'static str> {
    let z = t.challenge_field("perm-z");
    t.absorb_field("perm-theta", &[proof.theta]);
    let mut out_claims: Vec<OperandClaim> = Vec::new();
    let mut log_m_seen: Option<usize> = None;

    for unfold in &proof.unfolds {
        let log_m = unfold.e_tree.len();
        if log_m < LOG_KAPPA {
            return Err("perm: operand too small");
        }
        if let Some(seen) = log_m_seen {
            if seen != log_m {
                return Err("perm: operand shapes differ");
            }
        }
        log_m_seen = Some(log_m);
        let m = 1usize << log_m;

        // ---- the E-tree.
        let mut cur_pt: Vec<Gf128> = Vec::new();
        let mut cur_v = proof.theta;
        for (level, cubic) in unfold.e_tree.iter().enumerate() {
            if level >= log_m {
                return Err("perm: e-tree level count");
            }
            let l_len = m >> (log_m - level);
            let xi = ts_msbf(&cur_pt);
            if xi.len() != l_len {
                return Err("perm: e-tree level length");
            }
            let (y, v_u, v_d) = cubic_verify(cur_v, &xi, cubic, t)?;
            let y_p = t.challenge_field("perm-yp");
            cur_v = mul(ONE + y_p, v_u) + mul(y_p, v_d);
            let mut pt = Vec::with_capacity(1 + y.len());
            pt.push(y_p);
            pt.extend_from_slice(&y);
            cur_pt = pt;
        }
        if cur_pt.len() != log_m {
            return Err("perm: final E-tree point arity");
        }
        let col_pt: Vec<Gf128> = cur_pt[..LOG_KAPPA].to_vec();
        let row_pt: Vec<Gf128> = cur_pt[LOG_KAPPA..].to_vec();
        let r_7 = expand_factors(&msbf_to_factors(&col_pt));

        // ---- the F-transitions.
        if unfold.f_trans.len() != LOG_KAPPA {
            return Err("perm: transition count");
        }
        let mut v_next = cur_v;
        let mut l_next = ts_msbf(&row_pt);
        let mut l_pt_next = row_pt;
        let mut r_next = r_7;
        for (idx, trans) in unfold.f_trans.iter().enumerate() {
            let tau = LOG_KAPPA - 1 - idx;
            let rows_half = m >> (tau + 1);
            if trans.u.len() != KAPPA || trans.h.len() != KAPPA - 1 {
                return Err("perm: malformed transition");
            }
            if l_next.len() != rows_half {
                return Err("perm: l length");
            }
            t.absorb_field("perm-u", &trans.u);
            if crate::r1cs::dot_gf(&trans.u, &r_next) != v_next {
                return Err("perm: v_{τ+1} = 〈u, r_{τ+1}〉 failed");
            }
            t.absorb_field("perm-h", &trans.h);
            let x = challenge_outside_h(f8, data, t);
            let rho = data.rho_at_e(f8, x);
            let c_x = crate::r1cs::dot_gf(&trans.u, &rho);
            let psi_x = data.psi_at_e(f8, x);
            let mut hv = ZERO;
            for &c in trans.h.iter().rev() {
                hv = mul(hv, x) + c;
            }
            let claim = c_x + mul(psi_x, hv);
            let xi = ts_msbf(&l_pt_next);
            if xi.len() != l_next.len() {
                return Err("perm: l materialisation");
            }
            let (y, v_u, v_d) = cubic_verify(claim, &xi, &trans.cubic, t)?;
            let y_p = t.challenge_field("perm-yp");
            v_next = mul(ONE + y_p, v_u) + mul(y_p, v_d);
            let mut pt = Vec::with_capacity(1 + y.len());
            pt.push(y_p);
            pt.extend_from_slice(&y);
            l_pt_next = pt;
            l_next = ts_msbf(&l_pt_next);
            r_next = rho;
        }
        if l_next.len() != m {
            return Err("perm: terminal l₀ length");
        }
        // partition-of-unity hygiene: both weight vectors are ts/ψ sums
        let l_sum = l_next.iter().fold(ZERO, |acc, &v| acc + v);
        let r_sum = r_next.iter().fold(ZERO, |acc, &v| acc + v);
        if l_sum != ONE || r_sum != ONE {
            return Err("perm: weight vectors must sum to one");
        }
        let claim = v_next + z; // v₀ − z (char 2)
        let l0_factors = msbf_to_factors(&l_pt_next);
        out_claims.push((claim, l0_factors, r_next));
    }
    let claim_b = out_claims.pop().expect("two operands");
    let claim_a = out_claims.pop().expect("two operands");
    Ok(PermTerminal {
        claims: [claim_a, claim_b],
    })
}

// (the row-weight tensor is the terminal factors' consumer — Algorithm 4)

#[cfg(test)]
mod tests;
