//! Algorithm 2 — the packed Hadamard-product proof (ePrint 2026/2646, §4.2).
//!
//! Proves the weighted relation `c = 〈ξ, a∘b〉` with `ξ = r⊗l`,
//! `A,B ∈ F^{m×κ}` (the packed witness planes), row weight
//! `l = ⊗_r (l_{r,0}, l_{r,1})` and column weight `r ∈ E^κ`:
//!
//! ```text
//! d^(j) = 〈l, a^(j)∘b^(j)〉          (the κ column claims — bitsliced GF(2)
//!                                     parity dots against l's coordinate planes)
//! f(X) = 〈l, gA(X)∘gB(X)〉 − gC(X)   (deg ≤ 2κ−2; vanishes on H for valid d)
//! h    = f / Ψ                       (127 coefficients via f's samples at the
//!                                     off-domain K-nodes and division by Ψ)
//! 〈a, b∘l〉 = c_x + Ψ(x)h(x)        (a length-m inner product over E,
//!                                     x ←$ E\H, ρ = ψ(x))
//! ```
//!
//! **The f-sampling pipeline** (the prover's dominant cost, O(Nκ) bitsliced):
//! per off-domain node `γ'`, the encodings `gA(γ') = Σ_j a^(j)·ψ_j(γ')` are
//! formed as 8 GF(2) output planes (selector XORs over the witness planes —
//! the "precomputed base-polynomial evaluations"), spread to bytes with
//! `PDEP`, and the row products `gA_i(γ')·gB_i(γ')` are aggregated against
//! the E-row-weight through a 65536-entry `(a,b)`-keyed bucket table —
//! `Σ_i l_i·(gA_i·gB_i) = Σ_key T16[key]·(⊕_{i: key_i=key} l_i)` — one
//! random L2 read-modify-write per row, then one E-multiply per key. On
//! GFNI hardware the `T16` lookups are a single `VPGF2P8MULB` per 64 rows
//! (our code-basis is AES-polynomial by construction — see `gf8`).

use crate::fact1::{self, Fact1Proof};
use crate::gf8::Gf8;
use crate::interp::{InterpData, KAPPA};
use crate::planes::{planes_to_words, spread8, PlaneSet, RowWeightTensor};
use magnetar_field::Gf128;
use magnetar_field::{mul, ONE, ZERO};
use magnetar_hash::Transcript;

pub struct HadamardProof {
    /// The κ column claims `d^(j)` (`E`-elements).
    pub d: Vec<Gf128>,
    /// The 127 quotient coefficients `h = f/Ψ` (`E`-elements).
    pub h: Vec<Gf128>,
    /// The inner-product reduction.
    pub fact1: Fact1Proof,
}

/// Per-phase prover timings (ms) — the Table-3 cost anatomy.
#[derive(Default, Clone, Copy, Debug)]
pub struct HadamardStats {
    /// The bitsliced `d = l^T(A∘B)` folds.
    pub d_fold_ms: f64,
    /// The per-node encodings `gA(γ'), gB(γ')` (plane XORs).
    pub encodings_ms: f64,
    /// The bit→byte spreads feeding the GF(2^8) products.
    pub spreads_ms: f64,
    /// The 64K `(a,b)`-bucket aggregation pass.
    pub bucket_ms: f64,
    /// The `Σ_key T16[key]·bucket[key]` folds + `gC` terms.
    pub fold_ms: f64,
    /// `h = f/Ψ` (division at nodes + the Vandermonde application).
    pub h_ms: f64,
    /// `a = Aρ`, `b = Bρ` (the E-weighted selector builds + transposes).
    pub ab_ms: f64,
    /// The Fact-1 inner-product reduction.
    pub fact1_ms: f64,
    /// Total `prove_hadamard` wall time.
    pub total_ms: f64,
}

/// The terminal data a packed-PCS opening (Algorithm 4) consumes: the two
/// MLE claims with their tensor row weights and the Lagrange column weights.
pub struct HadamardTerminal {
    pub y: Vec<Gf128>,
    pub v1: Gf128,
    pub v2: Gf128,
    /// `ρ(x) = ψ(x) ∈ E^κ` — the column weights of both claims.
    pub rho: Vec<Gf128>,
}

fn dot(a: &[Gf128], b: &[Gf128]) -> Gf128 {
    let mut acc = ZERO;
    for i in 0..a.len() {
        acc += mul(a[i], b[i]);
    }
    acc
}

fn horner(coefs: &[Gf128], x: Gf128) -> Gf128 {
    let mut acc = ZERO;
    for &c in coefs.iter().rev() {
        acc = mul(acc, x) + c;
    }
    acc
}

/// Prove `c = 〈ξ, a∘b〉` with `ξ = r⊗l` (Algorithm 2).
#[allow(clippy::too_many_arguments)]
pub fn prove_hadamard(
    f8: &Gf8,
    data: &InterpData,
    a: &PlaneSet,
    b: &PlaneSet,
    l: &RowWeightTensor,
    r: &[Gf128],
    c: Gf128,
    t: &mut Transcript,
) -> HadamardProof {
    let (_, proof, _) = prove_hadamard_timed(f8, data, a, b, l, r, c, t);
    proof
}

/// The prover-side terminal data (what Algorithm 6's steps 9–11 consume).
pub struct HadamardProverTerminal {
    pub y: Vec<Gf128>,
    pub v1: Gf128,
    pub v2: Gf128,
    pub rho: Vec<Gf128>,
}

#[allow(clippy::too_many_arguments)]
pub fn prove_hadamard_timed(
    f8: &Gf8,
    data: &InterpData,
    a: &PlaneSet,
    b: &PlaneSet,
    l: &RowWeightTensor,
    r: &[Gf128],
    c: Gf128,
    t: &mut Transcript,
) -> (HadamardStats, HadamardProof, HadamardProverTerminal) {
    use std::time::Instant;
    let t_all = Instant::now();
    let mut stats = HadamardStats::default();

    let m = a.m;
    assert_eq!(b.m, m);
    assert_eq!(a.kappa, KAPPA);
    assert_eq!(r.len(), KAPPA);
    assert_eq!(l.m, m);
    // NOTE: the caller's claim `c` enters only through the VERIFIER's
    // `c = 〈d, r〉` check; the prover derives its reduced claim from `d`
    // and `h` (tamper tests feed claims inconsistent with the witness).
    let _ = c;

    // --- Step 1: the column claims d (bitsliced over l's coordinate planes).
    let t0 = Instant::now();
    let mut d = Vec::with_capacity(KAPPA);
    {
        let mut prod = PlaneSet::zeros(m, KAPPA);
        for j in 0..KAPPA {
            prod.and_into(j, a, b);
        }
        for j in 0..KAPPA {
            let pj = prod.plane(j);
            let mut bits = 0u128;
            for e in 0..128 {
                bits |= (PlaneSet::parity_dot(pj, l.plane(e)) as u128) << e;
            }
            d.push(Gf128(bits));
        }
        drop(prod); // free the product planes before the sampling pipeline
    }
    stats.d_fold_ms = t0.elapsed().as_secs_f64() * 1e3;
    // NOTE: no prover-side assert `c = 〈d, r〉` — the claim `c` is supplied
    // by the caller and tamper tests deliberately pass claims inconsistent
    // with (A, B); the VERIFIER's `c = 〈d, r〉` check is the soundness gate.
    t.absorb_field("had-d", &d);

    // --- Step 2: the f-samples at the 127 off-domain nodes.
    let t1 = Instant::now();
    let nodes = data.nodes_off.len();
    let wp = m / 64;
    let mk_enc = || -> [Vec<u64>; 8] { std::array::from_fn(|_| vec![0u64; wp]) };
    let mut enc_a = mk_enc();
    let mut enc_b = mk_enc();
    let mut bucket: Vec<Gf128> = vec![Gf128(0); 65536];
    let mut f_samples = Vec::with_capacity(nodes);
    let mut enc_ms = 0.0f64;
    let mut buck_ms = 0.0f64;
    let mut fold_ms = 0.0f64;
    for node in 0..nodes {
        // selectors: bit c of each ψ_j(γ') (shared by gA and gB — the same
        // Lagrange basis drives both encodings)
        let mut sel = [[0u64; 2]; 8];
        let row = &data.psi_off[node];
        for j in 0..KAPPA {
            for c in 0..8 {
                let bit = ((row[j] >> c) & 1) as u64;
                sel[c][j / 64] |= bit << (j % 64);
            }
        }
        let te = Instant::now();
        a.encode_k(&sel, &mut enc_a);
        b.encode_k(&sel, &mut enc_b);
        enc_ms += te.elapsed().as_secs_f64() * 1e3;
        // bucket pass
        for v in bucket.iter_mut() {
            *v = Gf128(0);
        }
        let tb1 = Instant::now();
        for w in 0..wp {
            let mut wa = [0u64; 8];
            let mut wb = [0u64; 8];
            for c in 0..8 {
                wa[c] = enc_a[c][w];
                wb[c] = enc_b[c][w];
            }
            let ba = spread8(&wa);
            let bb = spread8(&wb);
            let base_row = w * 64;
            for rr in 0..64 {
                let key = ((ba[rr] as usize) << 8) | bb[rr] as usize;
                bucket[key] += l.words[base_row + rr];
            }
        }
        buck_ms += tb1.elapsed().as_secs_f64() * 1e3; // includes the spreads
                                                      // gC(γ') = 〈d, ψ(γ')〉 and the fold Σ_key T16[key]·bucket[key]
        let tf = Instant::now();
        let mut g_c = ZERO;
        for j in 0..KAPPA {
            g_c += mul(d[j], f8.embed(row[j]));
        }
        let mut total = ZERO;
        for key in 0..65536usize {
            if bucket[key] != ZERO {
                total += mul(bucket[key], f8.embed(f8.mul_tab[key]));
            }
        }
        f_samples.push(total + g_c); // char 2: subtract = add
        fold_ms += tf.elapsed().as_secs_f64() * 1e3;
    }
    stats.encodings_ms = enc_ms;
    stats.bucket_ms = buck_ms; // (spread+bucket; the split reported together)
    stats.fold_ms = fold_ms;
    let _ = t1;

    // --- Step 3: h = f/Ψ from the samples; absorb; sample x.
    let t2 = Instant::now();
    let h = data.h_from_samples(f8, &f_samples);
    assert_eq!(h.len(), KAPPA - 1);
    stats.h_ms = t2.elapsed().as_secs_f64() * 1e3;
    t.absorb_field("had-h", &h);
    let x = t.challenge_field("had-x");
    let rho = data.rho_at_e(f8, x);

    // --- Step 4: a = Aρ, b = Bρ as E-words, then b∘l in place.
    let t3 = Instant::now();
    let a_vec = encode_e_words(a, &rho, m);
    let mut b_vec = encode_e_words(b, &rho, m);
    for i in 0..m {
        b_vec[i] = mul(b_vec[i], l.words[i]);
    }
    stats.ab_ms = t3.elapsed().as_secs_f64() * 1e3;

    // --- Step 5: the reduced claim and Fact 1.
    let c_x = dot(&d, &rho);
    let psi_x = data.psi_at_e(f8, x);
    let v = c_x + mul(psi_x, horner(&h, x));
    let mut av = a_vec;
    let mut bv = b_vec;
    debug_assert_eq!(dot(&av, &bv), v, "〈a, b∘l〉 = c_x + Ψ(x)h(x)");
    let t4 = Instant::now();
    let fact1 = fact1::prove(&mut av, &mut bv, v, t);
    stats.fact1_ms = t4.elapsed().as_secs_f64() * 1e3;
    stats.total_ms = t_all.elapsed().as_secs_f64() * 1e3;

    let terminal = HadamardProverTerminal {
        y: fact1.challenges.clone(),
        v1: fact1.terminals.0,
        v2: fact1.terminals.1,
        rho,
    };
    (stats, HadamardProof { d, h, fact1 }, terminal)
}

/// `A·ρ` as E-words: coordinate-e plane = XOR of witness planes j with
/// bit_e(ρ_j) set (the E-generalisation of `encode_k`'s selector XOR).
/// Shared by the Hadamard, inner-product, and permutation provers.
pub(crate) fn encode_e_words(ps: &PlaneSet, rho: &[Gf128], m: usize) -> Vec<Gf128> {
    let wp = m / 64;
    let mut planes = vec![0u64; 128 * wp];
    let mut sel = [[0u64; 2]; 128];
    for e in 0..128 {
        let mut lo = 0u64;
        let mut hi = 0u64;
        for j in 0..KAPPA {
            let bit = ((rho[j].0 >> e) & 1) as u64;
            if j < 64 {
                lo |= bit << j;
            } else {
                hi |= bit << (j - 64);
            }
        }
        sel[e] = [lo, hi];
    }
    for e in 0..128 {
        let (lo, hi) = (sel[e][0], sel[e][1]);
        let out = &mut planes[e * wp..(e + 1) * wp];
        for j in 0..KAPPA {
            if ((if j < 64 { lo >> j } else { hi >> (j - 64) }) & 1) == 1 {
                let sp = ps.plane(j);
                for k in 0..wp {
                    out[k] ^= sp[k];
                }
            }
        }
    }
    planes_to_words(&planes, m)
}

/// The public/verifier-side ξ factors of the second terminal:
/// `ξ2 = l∘ts(y)` with per-factor entries `(l_{j,0}·(1−y_j), l_{j,1}·y_j)`
/// (Algorithm 2, step 14).
pub fn xi2_factors(l: &RowWeightTensor, y: &[Gf128]) -> Vec<(Gf128, Gf128)> {
    assert_eq!(l.factors.len(), y.len());
    // ξ2 = l∘ts(y): the element-wise product of the two tensors — the
    // factor pair is (l_{k,0}·(1+y_k), l_{k,1}·y_k) in char 2.
    l.factors
        .iter()
        .zip(y.iter())
        .map(|(&(f0, f1), &yj)| (mul(f0, ONE + yj), mul(f1, yj)))
        .collect()
}

/// The ξ2 factors from raw factor pairs (the verifier's path — it holds
/// `l`'s factors, not the materialised tensor).
pub fn xi2_factors_from(l: &[(Gf128, Gf128)], y: &[Gf128]) -> Vec<(Gf128, Gf128)> {
    assert_eq!(l.len(), y.len());
    l.iter()
        .zip(y.iter())
        .map(|(&(f0, f1), &yj)| (mul(f0, ONE + yj), mul(f1, yj)))
        .collect()
}

/// Verify (Algorithm 2): returns the terminal data for the packed PCS.
pub fn verify_hadamard(
    f8: &Gf8,
    data: &InterpData,
    r: &[Gf128],
    c: Gf128,
    proof: &HadamardProof,
    t: &mut Transcript,
) -> Result<HadamardTerminal, &'static str> {
    if proof.d.len() != KAPPA || proof.h.len() != KAPPA - 1 {
        return Err("hadamard: malformed message sizes");
    }
    if dot(&proof.d, r) != c {
        return Err("hadamard: c != <d, r>");
    }
    t.absorb_field("had-d", &proof.d);
    t.absorb_field("had-h", &proof.h);
    let x = t.challenge_field("had-x");
    let rho = data.rho_at_e(f8, x);
    let c_x = dot(&proof.d, &rho);
    let psi_x = data.psi_at_e(f8, x);
    let v = c_x + mul(psi_x, horner(&proof.h, x));
    let (y, v1, v2) = fact1::verify(v, &proof.fact1, t)?;
    Ok(HadamardTerminal { y, v1, v2, rho })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf8::Gf8;
    use crate::interp::InterpData;
    use magnetar_field::ONE;

    thread_local! {
        static RNG: std::cell::Cell<u128> = const { std::cell::Cell::new(0xDEADBEEFCAFEF00D) };
    }

    fn gf() -> Gf128 {
        RNG.with(|c| {
            let mut r = c.get();
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            c.set(r);
            Gf128(r)
        })
    }

    /// The dense E-arithmetic reference for the whole protocol at small m.
    fn reference(
        f8: &Gf8,
        data: &InterpData,
        a: &PlaneSet,
        b: &PlaneSet,
        l: &RowWeightTensor,
        r: &[Gf128],
    ) -> (Vec<Gf128>, Vec<Gf128>, Gf128) {
        let m = a.m;
        // d_j = Σ_i l_i a_ij b_ij over E
        let mut d = vec![ZERO; KAPPA];
        for j in 0..KAPPA {
            let mut acc = ZERO;
            for i in 0..m {
                let ai = if a.bit(j, i) { ONE } else { ZERO };
                let bi = if b.bit(j, i) { ONE } else { ZERO };
                acc += mul(mul(ai, bi), l.words[i]);
            }
            d[j] = acc;
        }
        // h: f(X) = Σ_{jj'} S_{jj'} ψ_j ψ_j' − Σ_j d_j ψ_j ; S_{jj'} = Σ_i l_i a_ij b_ij'
        // (compute S densely, then the f-coefficients, then divide by Ψ.)
        let mut s = vec![ZERO; KAPPA * KAPPA];
        for j in 0..KAPPA {
            for j2 in 0..KAPPA {
                let mut acc = ZERO;
                for i in 0..m {
                    let ai = if a.bit(j, i) { ONE } else { ZERO };
                    let bi = if b.bit(j2, i) { ONE } else { ZERO };
                    if ai == ONE && bi == ONE {
                        acc += l.words[i];
                    }
                }
                s[j * KAPPA + j2] = acc;
            }
        }
        // ψ_j as E-coefficient polynomials (from data.psi_monomials)
        let psi_e: Vec<Vec<Gf128>> = data
            .psi_monomials
            .iter()
            .map(|coefs| coefs.iter().map(|&c| f8.embed(c)).collect())
            .collect();
        // f's coefficients (deg ≤ 254)
        let mut fc = vec![ZERO; 2 * KAPPA - 1];
        for j in 0..KAPPA {
            for j2 in 0..KAPPA {
                let sv = s[j * KAPPA + j2];
                if sv == ZERO {
                    continue;
                }
                let pj = &psi_e[j];
                let pj2 = &psi_e[j2];
                for u in 0..pj.len() {
                    for v in 0..pj2.len() {
                        if pj[u] != ZERO && pj2[v] != ZERO {
                            fc[u + v] += mul(sv, mul(pj[u], pj2[v]));
                        }
                    }
                }
            }
        }
        for j in 0..KAPPA {
            let dj = d[j];
            if dj == ZERO {
                continue;
            }
            for u in 0..psi_e[j].len() {
                if psi_e[j][u] != ZERO {
                    fc[u] += mul(dj, psi_e[j][u]);
                }
            }
        }
        // divide fc (deg 254) by Ψ (deg 128, E-embedded coefficients)
        let psi_c: Vec<Gf128> = data.psi.coefs.iter().map(|&c| f8.embed(c)).collect();
        let mut rem = fc.clone();
        let mut h_out = vec![ZERO; KAPPA - 1];
        assert_eq!(psi_c[KAPPA], ONE, "Ψ is monic");
        for i in (0..KAPPA - 1).rev() {
            // coefficient of X^{i+128} in rem
            let top = rem[i + KAPPA];
            if top != ZERO {
                h_out[i] = top;
                for u in 0..=KAPPA {
                    rem[i + u] += mul(top, psi_c[u]);
                }
            }
        }
        assert!(rem[..KAPPA].iter().all(|&v| v == ZERO), "Ψ | f exactly");
        // the claim c
        let c_claim = dot(&d, r);
        (d, h_out, c_claim)
    }

    #[test]
    fn hadamard_matches_reference_and_verifies() {
        let f8 = Gf8::new();
        let mut data = InterpData::new(&f8);
        data.build_psi_monomials(&f8);
        let m = 64usize; // 1 chunk
        let log_m = 6usize;
        // random planes
        let mut a = PlaneSet::zeros(m, KAPPA);
        let mut b = PlaneSet::zeros(m, KAPPA);
        for j in 0..KAPPA {
            for i in 0..m {
                a.set_bit(j, i, gf().0 & 1 == 1);
                b.set_bit(j, i, gf().0 & 1 == 1);
            }
        }
        let factors: Vec<(Gf128, Gf128)> = (0..log_m).map(|_| (gf(), gf())).collect();
        let l = RowWeightTensor::new(factors, true);
        let r: Vec<Gf128> = (0..KAPPA).map(|_| gf()).collect();

        let (d_ref, h_ref, c) = reference(&f8, &data, &a, &b, &l, &r);

        let mut t = Transcript::new("mikros-hadamard-test", &[]);
        let proof = prove_hadamard(&f8, &data, &a, &b, &l, &r, c, &mut t);
        assert_eq!(proof.d, d_ref, "d must match the dense reference");
        assert_eq!(proof.h, h_ref, "h must match the reference quotient");

        let mut tv = Transcript::new("mikros-hadamard-test", &[]);
        let term = verify_hadamard(&f8, &data, &r, c, &proof, &mut tv).expect("verify");
        // terminal claims against direct dense evaluation:
        // v1 = <A·ρ, ts(y)> ; v2 = <(B·ρ)∘l, ts(y)>
        let rho = term.rho;
        let mut a_vec = vec![ZERO; m];
        let mut b_vec = vec![ZERO; m];
        for i in 0..m {
            for j in 0..KAPPA {
                if a.bit(j, i) {
                    a_vec[i] += rho[j];
                }
                if b.bit(j, i) {
                    b_vec[i] += rho[j];
                }
            }
            b_vec[i] = mul(b_vec[i], l.words[i]);
        }
        let ts_eval = |v: &[Gf128], pts: &[Gf128]| -> Gf128 {
            let mut cur = v.to_vec();
            for &p in pts {
                let half = cur.len() / 2;
                for k in 0..half {
                    cur[k] = cur[k] + mul(p, cur[k] + cur[k + half]);
                }
                cur.truncate(half);
            }
            cur[0]
        };
        assert_eq!(term.v1, ts_eval(&a_vec, &term.y), "v1 = <a, ts(y)>");
        assert_eq!(term.v2, ts_eval(&b_vec, &term.y), "v2 = <b∘l, ts(y)>");
    }

    #[test]
    fn hadamard_tamper_rejected() {
        let f8 = Gf8::new();
        let mut data = InterpData::new(&f8);
        data.build_psi_monomials(&f8);
        let m = 128usize;
        let log_m = 7usize;
        let mut a = PlaneSet::zeros(m, KAPPA);
        let mut b = PlaneSet::zeros(m, KAPPA);
        for j in 0..KAPPA {
            for i in 0..m {
                a.set_bit(j, i, gf().0 & 1 == 1);
                b.set_bit(j, i, gf().0 & 1 == 1);
            }
        }
        let factors: Vec<(Gf128, Gf128)> = (0..log_m).map(|_| (gf(), gf())).collect();
        let l = RowWeightTensor::new(factors, true);
        let r: Vec<Gf128> = (0..KAPPA).map(|_| gf()).collect();
        // honest claim
        let mut c = ZERO;
        for j in 0..KAPPA {
            let mut dj = ZERO;
            for i in 0..m {
                if a.bit(j, i) && b.bit(j, i) {
                    dj += l.words[i];
                }
            }
            c += mul(dj, r[j]);
        }
        let mut t = Transcript::new("mikros-hadamard-test", &[]);
        let proof = prove_hadamard(&f8, &data, &a, &b, &l, &r, c, &mut t);

        // tamper d (one column claim): must fail c = <d, r>
        {
            let mut bad = HadamardProof {
                d: proof.d.clone(),
                h: proof.h.clone(),
                fact1: clone_fact1(&proof.fact1),
            };
            bad.d[3] += Gf128(1);
            let mut tv = Transcript::new("mikros-hadamard-test", &[]);
            assert!(verify_hadamard(&f8, &data, &r, c, &bad, &mut tv).is_err());
        }
        // tamper h: the final identity must fail (x fixed by transcript…
        // the tampered h changes the absorb → different x; the identity
        // 〈a,b∘l〉 = c_x + Ψ(x)h(x) fails at the new x except w.p. ~2^-128)
        {
            let mut bad = HadamardProof {
                d: proof.d.clone(),
                h: proof.h.clone(),
                fact1: clone_fact1(&proof.fact1),
            };
            bad.h[5] += Gf128(1);
            let mut tv = Transcript::new("mikros-hadamard-test", &[]);
            assert!(verify_hadamard(&f8, &data, &r, c, &bad, &mut tv).is_err());
        }
        // tamper the witness (a bit flip where the product is live) with the
        // same claim: the proof regenerated for the tampered witness must
        // fail against the stale c
        {
            let mut a2 = a.clone();
            // choose a flip that changes a∘b: find a row where b's bit is 1
            let mut flipped = false;
            'outer: for j in 0..KAPPA {
                for i in 0..m {
                    if b.bit(j, i) && a.bit(j, i) {
                        a2.set_bit(j, i, false);
                        flipped = true;
                        break 'outer;
                    }
                }
            }
            assert!(flipped, "a random witness has a live product bit");
            // the honest claim for the tampered witness:
            let mut c2 = ZERO;
            for j in 0..KAPPA {
                let mut dj = ZERO;
                for i in 0..m {
                    if a2.bit(j, i) && b.bit(j, i) {
                        dj += l.words[i];
                    }
                }
                c2 += mul(dj, r[j]);
            }
            assert_ne!(c2, c, "the flip must change the claim");
            let mut t2 = Transcript::new("mikros-hadamard-test", &[]);
            let bad_proof = prove_hadamard(&f8, &data, &a2, &b, &l, &r, c2, &mut t2);
            let mut tv = Transcript::new("mikros-hadamard-test", &[]);
            assert!(
                verify_hadamard(&f8, &data, &r, c, &bad_proof, &mut tv).is_err(),
                "tampered witness with stale claim must reject"
            );
        }
    }

    fn clone_fact1(p: &Fact1Proof) -> Fact1Proof {
        Fact1Proof {
            rounds: p.rounds.clone(),
            terminals: p.terminals,
            challenges: p.challenges.clone(),
        }
    }
}
