//! Algorithm 4 — the packed PCS opening (ePrint 2026/2646, §4.4).
//!
//! Proves `v = 〈Aρ, ξ〉 = ξ^⊤ A ρ` for the fixed-basis commitment
//! `a = Aβ ∈ E^m`, with `ξ = ⊗_r (ξ^{(r)}_0, ξ^{(r)}_1)` a two-entry-factor
//! row weight and `ρ ∈ E^κ` arbitrary column weights:
//!
//! 1. `U = Ξ^⊤ A` (the coordinate claims) → message `v_msg = U^⊤ β`;
//!    the verifier checks `v = 〈v_msg, ρ⟩` and recovers `U` from the
//!    message (the poly basis makes coordinates = bits).
//! 2. A fresh `r ← E^κ` projects `U β = Ξ^⊤ a` to
//!    `c = r^⊤ U β = 〈a, Ξ r〉` — reduced by Fact 1 to the witness MLE claim
//!    `w_com = 〈a, ts(y)〉` and the public-side claim
//!    `w = ts(y)^⊤ Ξ r`.
//! 3. The verifier checks `w` by the char-2 form of the Eq.-(25) recursion
//!    (`W_j = M_{1−y_j}^⊤ W_{j+1} M_{ξ^{(j)}_0} + M_{y_j}^⊤ W_{j+1}
//!    M_{ξ^{(j)}_1}` — derived: `= W_{j+1}·M_{1+x_j}`-style row/column
//!    scalings in the poly basis), then `w = β^⊤ W r`.
//! 4. `pc.Open(cm, w_com, y)` — this repo's fold-chain PCS (the
//!    "Ligerito-role" backend).

use crate::fact1::{self, Fact1Proof};
use crate::interp::KAPPA;
use crate::planes::PlaneSet;
use magnetar_field::Gf128;
use magnetar_field::{mul, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_pcs::eval::{
    commit_table, prove_eval, verify_eval, PcsParams, PcsProof, PcsStatement,
};

pub struct PackedOpenProof {
    /// `U^⊤ β` — the packed U columns (κ E-elements).
    pub v_msg: Vec<Gf128>,
    pub fact1: Fact1Proof,
    /// The public-side terminal `w = ts(y)^⊤ Ξ r` (also derivable by the
    /// W-recursion; sent for transcript binding).
    pub w_pub: Gf128,
    pub pcs: PcsProof,
}

/// `U = Ξ^⊤ A` — κ×κ over GF(2), bitsliced: `U[c][j] =
/// parity(Ξplane_c ∧ Aplane_j)`. `ξ` enters as its 128 coordinate planes.
pub fn compute_u(a: &PlaneSet, xi_planes: &[u64], xi_wp: usize) -> Vec<[u64; 2]> {
    // out[c] = the packed U row c (the coordinate-claim row)
    let mut rows = vec![[0u64; 2]; KAPPA];
    for c in 0..KAPPA {
        let xp = &xi_planes[c * xi_wp..(c + 1) * xi_wp];
        let mut acc = [0u64; 2];
        for j in 0..KAPPA {
            if PlaneSet::parity_dot(xp, a.plane(j)) == 1 {
                acc[j / 64] |= 1u64 << (j % 64);
            }
        }
        rows[c] = acc;
    }
    rows
}

/// The char-2 W-recursion (Eq. 25 in the paper's general form): given the
/// ξ two-entry factors and the ts(y) point, compute `W` from
/// `W_{ℓ+1} = e_1 e_1^⊤` with
/// `W_j = M_{1−y_j}^⊤ W_{j+1} M_{ξ^{(j)}_0} + M_{y_j}^⊤ W_{j+1} M_{ξ^{(j)}_1}`
/// (in the poly basis: left-multiplication scales coordinate ROWS,
/// right-multiplication scales coordinate COLUMNS).
/// Returns `w = β^⊤ W r`.
pub fn w_recursion(xi_factors: &[(Gf128, Gf128)], y: &[Gf128], r: &[Gf128]) -> Gf128 {
    assert_eq!(xi_factors.len(), y.len());
    // W as 128 packed rows (E-words) — W[k]'s coordinates = row k.
    let mut w_rows = vec![ZERO; KAPPA];
    w_rows[0] = ONE; // e_1 e_1^⊤: row 0 has its 0th coordinate set
                     // (in the packed-row form: the E-element of row 0 = β_0 = 1.)
    for (j, (&(xi0, xi1), &yj)) in xi_factors.iter().zip(y.iter()).enumerate().rev() {
        // W_j = M_{1−y_j}^⊤ W M_{ξ^{(j)}_0} + M_{y_j}^⊤ W M_{ξ^{(j)}_1}
        // The right factor M scales coordinate ROWS (by ξ0/ξ1); the left
        // factor M^⊤ scales coordinate COLUMNS (by 1−y / y — char 2: 1+y).
        let one_my = ONE + yj; // 1 − y in char 2
        let mut term_lo = w_rows.clone();
        let mut term_hi = w_rows.clone();
        for k in 0..KAPPA {
            term_lo[k] = mul(term_lo[k], xi0);
            term_hi[k] = mul(term_hi[k], xi1);
        }
        term_lo = col_scale(&term_lo, one_my);
        term_hi = col_scale(&term_hi, yj);
        for k in 0..KAPPA {
            w_rows[k] = term_lo[k] + term_hi[k];
        }
        let _ = j;
    }
    // w = β^⊤ W r = Σ_k W_col(k)·r_k — the column values
    let mut acc = ZERO;
    for k in 0..KAPPA {
        if r[k] != ZERO {
            let col = col_value(&w_rows, k);
            acc += mul(col, r[k]);
        }
    }
    acc
}

/// Scale the coordinate-COLUMN k of W (rows-packed) by α: implemented via
/// the transposed view — column k's E-element × α.
fn col_scale(rows: &[Gf128], alpha: Gf128) -> Vec<Gf128> {
    let mut out = rows.to_vec();
    for m in 0..KAPPA {
        // column m's value = Σ_k bit_m(rows[k])·β_k
        let mut col = ZERO;
        for k in 0..KAPPA {
            if (rows[k].0 >> m) & 1 == 1 {
                col += Gf128(1u128 << k);
            }
        }
        // scale and write back into every row's bit m
        let scaled = mul(col, alpha);
        for k in 0..KAPPA {
            let bit = (scaled.0 >> k) & 1;
            if bit == 1 {
                out[k].0 |= 1u128 << m;
            } else {
                out[k].0 &= !(1u128 << m);
            }
        }
    }

    out
}

fn col_value(rows: &[Gf128], k: usize) -> Gf128 {
    let mut col = ZERO;
    for m in 0..KAPPA {
        if (rows[m].0 >> k) & 1 == 1 {
            col += Gf128(1u128 << m);
        }
    }
    col
}

pub struct PackedOpenOutput {
    pub stmt: PcsStatement,
    pub a_table: Vec<Gf128>,
}

/// Commit `a = Aβ` (Algorithm 4's precondition; shared across openings).
pub fn commit_packed(params: &PcsParams, a: &PlaneSet) -> PackedOpenOutput {
    let m = a.m;
    // a_i = the packed row i = Σ_j A[i][j]·2^j: transpose to (plane = row)
    let tr = crate::r1cs::transpose_ps(a);
    let mut a_table = Vec::with_capacity(m);
    for i in 0..m {
        let lo = tr.plane(i)[0];
        let hi = tr.plane(i)[1];
        a_table.push(Gf128((lo as u128) | ((hi as u128) << 64)));
    }
    let (_, tree, stmt) = commit_table(params, &a_table);
    let root = tree.root();
    let _ = root;
    PackedOpenOutput { stmt, a_table }
}

#[allow(clippy::too_many_arguments)]
pub fn prove_packed_open(
    a: &PlaneSet,
    a_table: &[Gf128],
    xi_factors: &[(Gf128, Gf128)],
    rho: &[Gf128],
    v: Gf128,
    params: &PcsParams,
    t: &mut Transcript,
) -> PackedOpenProof {
    let m = a.m;
    // --- the coordinate claims: U = Ξ^⊤ A
    // materialize ξ as E-words (the tensor product), then its planes:
    let mut words = vec![ONE];
    for &(f0, f1) in xi_factors {
        let half = words.len();
        let mut nv = vec![ZERO; half * 2];
        for i in 0..half {
            nv[i] = mul(words[i], f0);
            nv[half + i] = mul(words[i], f1);
        }
        words = nv;
    }
    assert_eq!(words.len(), m);
    let xi_planes = crate::planes::words_to_planes(&words, m);
    let xi_wp = m / 64;
    let rows = compute_u(a, &xi_planes, xi_wp);
    let mut v_msg = vec![ZERO; KAPPA];
    for j in 0..KAPPA {
        // v_msg[j] = Σ_c U[c][j]·β_c = the packed U column j
        let mut acc = 0u128;
        for c in 0..KAPPA {
            if (rows[c][j / 64] >> (j % 64)) & 1 == 1 {
                acc |= 1u128 << c;
            }
        }
        v_msg[j] = Gf128(acc);
    }
    t.absorb_field("po-vmsg", &v_msg);
    debug_assert_eq!(crate::r1cs::dot_gf(&v_msg, rho), v, "v = <v_msg, rho>");
    let r = {
        let mut rv = Vec::with_capacity(KAPPA);
        for _ in 0..KAPPA {
            rv.push(t.challenge_field("po-r"));
        }
        rv
    };
    // c = r^⊤ U β = 〈a, Ξ r〉
    // (Uβ)_c = the packed row c; c = Σ_c r_c·(Uβ)_c
    let mut c = ZERO;
    for k in 0..KAPPA {
        if r[k] != ZERO {
            let mut acc = ZERO;
            for cj in 0..KAPPA {
                if (rows[k][cj / 64] >> (cj % 64)) & 1 == 1 {
                    acc += Gf128(1u128 << cj);
                }
            }
            c += mul(acc, r[k]);
        }
    }
    // operand Ξ r: per i: Σ_c bit_c(ξ_i)·r_c (the φ_r fold)
    let mut operand = vec![ZERO; m];
    for i in 0..m {
        let xi = words[i].0;
        let mut acc = ZERO;
        let mut bits = xi;
        while bits != 0 {
            let cpos = bits.trailing_zeros() as usize;
            acc += r[cpos];
            bits &= bits - 1;
        }
        operand[i] = acc;
    }
    debug_assert_eq!(crate::r1cs::dot_gf(a_table, &operand), c);
    let mut av = a_table.to_vec();
    let mut op = operand;
    let fact1 = fact1::prove(&mut av, &mut op, c, t);
    let (v1, v2) = fact1.terminals;
    let y = fact1.challenges.clone();
    // public-side terminal: w = ts(y)^⊤ Ξ r = 〈Ξr, ts(y)〉 — the Fact-1
    // second terminal (bound for transcript consistency)
    let w_pub = v2;
    t.absorb_field("po-wpub", &[w_pub]);
    // Fact 1 binds the MSB first; the PCS binds LSB-first — reverse.
    let point: Vec<Gf128> = y.iter().rev().copied().collect();
    let pcs = prove_eval(params, a_table, &point, v1, t);
    PackedOpenProof {
        v_msg,
        fact1,
        w_pub,
        pcs,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_packed_open(
    stmt: &PcsStatement,
    params: &PcsParams,
    xi_factors: &[(Gf128, Gf128)],
    rho: &[Gf128],
    v: Gf128,
    proof: &PackedOpenProof,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    if proof.v_msg.len() != KAPPA {
        return Err("packed-open: v_msg length");
    }
    if crate::r1cs::dot_gf(&proof.v_msg, rho) != v {
        return Err("packed-open: v != <v_msg, rho>");
    }
    t.absorb_field("po-vmsg", &proof.v_msg);
    let r: Vec<Gf128> = (0..KAPPA).map(|_| t.challenge_field("po-r")).collect();
    // recover U from the message: U[c][j] = bit c of v_msg[j]
    // c = Σ_c r_c · (Uβ)_c
    let mut c = ZERO;
    for k in 0..KAPPA {
        if r[k] != ZERO {
            let mut acc = ZERO;
            for cj in 0..KAPPA {
                if (proof.v_msg[cj].0 >> k) & 1 == 1 {
                    acc += Gf128(1u128 << cj);
                }
            }
            c += mul(acc, r[k]);
        }
    }
    let (y, v1, v2) = fact1::verify(c, &proof.fact1, t)?;
    // the W-recursion check: w = ts(y)^⊤ Ξ r
    let w_check = w_recursion(xi_factors, &y, &r);
    if w_check != v2 {
        return Err("packed-open: W-recursion mismatch");
    }
    t.absorb_field("po-wpub", &[proof.w_pub]);
    if proof.w_pub != v2 {
        return Err("packed-open: w_pub binding");
    }
    verify_eval(stmt, params, &y, v1, &proof.pcs, t)
}

/// Expose the packed witness words (`a = Aβ`) from the transposed layout.
pub fn packed_words_from_transposed(tr: &PlaneSet, m: usize) -> Vec<Gf128> {
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let lo = tr.plane(i)[0];
        let hi = tr.plane(i)[1];
        out.push(Gf128((lo as u128) | ((hi as u128) << 64)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_recursion_matches_direct() {
        // direct: w = ts(y)^⊤ Ξ r where Ξ = the coordinate matrix of
        // ξ = ⊗ factors; both sides computed densely over E.
        let mut seed = 0x1234u128;
        let mut gf = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            Gf128(seed)
        };
        let l = 6usize;
        let xis: Vec<(Gf128, Gf128)> = (0..l).map(|_| (gf(), gf())).collect();
        let y: Vec<Gf128> = (0..l).map(|_| gf()).collect();
        let r: Vec<Gf128> = (0..KAPPA).map(|_| gf()).collect();
        // ξ words and ts(y) words
        let mut xi = vec![ONE];
        for &(f0, f1) in &xis {
            let half = xi.len();
            let mut nv = vec![ZERO; half * 2];
            for i in 0..half {
                nv[i] = mul(xi[i], f0);
                nv[half + i] = mul(xi[i], f1);
            }
            xi = nv;
        }
        let mut tsy = vec![ONE];
        for &yk in &y {
            let half = tsy.len();
            let mut nv = vec![ZERO; half * 2];
            for i in 0..half {
                nv[i] = mul(tsy[i], ONE + yk);
                nv[half + i] = mul(tsy[i], yk);
            }
            tsy = nv;
        }
        // direct w = Σ_i tsy_i · (Σ_c bit_c(xi_i)·r_c)
        let mut want = ZERO;
        for i in 0..1 << l {
            let mut phi = ZERO;
            let mut bits = xi[i].0;
            while bits != 0 {
                let cpos = bits.trailing_zeros() as usize;
                phi += r[cpos];
                bits &= bits - 1;
            }
            want += mul(tsy[i], phi);
        }
        assert_eq!(w_recursion(&xis, &y, &r), want);
    }
}
