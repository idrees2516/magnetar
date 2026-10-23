//! The **cross-step transition argument** (the paper's Remark 4: the
//! cross-step constraints "enforced with the next multilinear extension
//! and one additional sum-check argument over the instance coordinates").
//!
//! The constraint: for every cycle `j`,
//! `fetch(j+1) = TT[fetch(j), taken(j), halt(j)]` — the next cycle's
//! fetch address is the public transition table applied to this cycle's
//! fetch, the branch-taken flag, and the halt latch (the halt freezes).
//!
//! Two chained sum-checks, both on the sparse engine (factors stay in
//! their own variable spaces — the prover's work is linear in the trace):
//!
//! * **S1 (the transition Shout)** over the joint `(k, k', t, h, j)`
//!   space: `Σ eq(r_k)·eq(r_k')·eq(r_j)·A(k,j)·ASH(k',j)·L_t·L_h·
//!   [α₀ + γ·TT(k,k',t,h)] = α₀ + γ`, where `A` is the committed fetch
//!   one-hot, `ASH(k',j) = A(k', j+1)` is the prover's shifted table
//!   (frozen at the end / the public boundary pc), and `L_t = 1 + t +
//!   t̂(j)`, `L_h = 1 + h + ĥ(j)` are the eq-links to the committed flag
//!   columns. The `α₀` term forces `A` and `ASH` sum-to-one per cycle;
//!   the `γ` term forces every transition to hit `TT`.
//! * **S2 (the shift discharge)** over `(k', j')`:
//!   `Σ eq(r_k')·g(j')·A(k',j') = v_ASH (+ P)` with the public
//!   predecessor-weight table `g(j') = eq(r_j, j'−1)` (plus the freeze
//!   `eq(r_j, n_cols−1)`, or the public boundary one-hot term `P` for
//!   tree-IVC segments) — discharging S1's `ASH` terminal against the
//!   **committed** `A`.
//!
//! Terminals: `A` at S1's and S2's points (Mosaic claims on the fetch
//! component), the flag columns `t̂/ĥ` at S1's cycle point (Mosaic
//! point-claims on the flags component), recovered from the eq-link
//! terminals exactly as the store router recovers its offset bits.

use crate::claims::PcsPoint;
use crate::AddressComponent;
use magnetar_field::{mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_sumcheck::sparse::{SparseFactor, SparseProver, SparseSpec};
use magnetar_sumcheck::{BindOrder, Term};

/// The transition argument's spec.
#[derive(Clone, Debug)]
pub struct CrossStepSpec {
    /// The fetch address width.
    pub a: usize,
    /// Cycle variable count.
    pub n: usize,
    /// The transition table over `(k, k', t, h)` (indexed as in
    /// `RomTables::tt`).
    pub tt: Vec<Gf128>,
    /// The next segment's public initial pc (tree-IVC segments); `None`
    /// for a full trace (the halt freeze applies at the last cycle).
    pub next_pc_public: Option<usize>,
}

/// The transition argument's proof.
#[derive(Clone, Debug)]
pub struct CrossStepProof {
    /// S1's round messages.
    pub s1_rounds: Vec<Vec<Gf128>>,
    /// S1's `A` terminal claim.
    pub s1_a_claim: Gf128,
    /// S1's `ASH` terminal claim (`v_ASH`).
    pub s1_ash_claim: Gf128,
    /// S1's eq-link terminals (the `L_t`, `L_h` values at `(r_t/r_h, r_j)`).
    pub s1_link_claims: [Gf128; 2],
    /// S2's round messages.
    pub s2_rounds: Vec<Vec<Gf128>>,
    /// S2's `A` terminal claim.
    pub s2_a_claim: Gf128,
}

/// The claims the caller must discharge against the commitment.
#[derive(Clone, Debug)]
pub struct CrossStepClaims {
    /// `A(r_k, r_j)` from S1.
    pub s1_a: PcsPoint,
    /// `A(r_k'', r_j'')` from S2.
    pub s2_a: PcsPoint,
    /// `t̂(r_j)` (the branch-taken flag at S1's cycle point).
    pub taken_value: Gf128,
    /// `ĥ(r_j)` (the halt latch at S1's cycle point).
    pub halt_value: Gf128,
    /// S1's cycle point (the flags claims' cycle coordinates).
    pub r_j: Vec<Gf128>,
}

/// The joint layout of S1: `k | k' | t | h | j`.
fn s1_layout(a: usize, _n: usize) -> [usize; 5] {
    let k = 0;
    let k2 = k + a;
    let t = k2 + a;
    let h = t + 1;
    let j = h + 1;
    [k, k2, t, h, j]
}

/// Prove the transition argument. `fetch` is the committed one-hot
/// component; `taken`/`halt` are the per-cycle flag bits (length `2^n`).
pub fn prove_cross_step(
    spec: &CrossStepSpec,
    fetch: &AddressComponent,
    taken: &[Gf128],
    halt: &[Gf128],
    t: &mut Transcript,
) -> (CrossStepProof, CrossStepClaims) {
    let a = spec.a;
    let n = spec.n;
    let n_cols = 1usize << n;
    let [k0, k1, t0, h0, j0] = s1_layout(a, n);
    let jv = 2 * a + 2 + n;
    debug_assert_eq!(fetch.table.len(), (1 << a) * n_cols);
    debug_assert_eq!(taken.len(), n_cols);
    debug_assert_eq!(halt.len(), n_cols);
    // The shifted fetch table ASH(k', j) = A(k', j+1) with the last cycle
    // frozen (or the public boundary one-hot).
    let ash: Vec<Gf128> = {
        let mut v = vec![ZERO; (1 << a) * n_cols];
        for j in 0..n_cols - 1 {
            for k in 0..(1usize << a) {
                v[k | (j << a)] = fetch.table[k | ((j + 1) << a)];
            }
        }
        let last = n_cols - 1;
        match spec.next_pc_public {
            Some(pc) => {
                for k in 0..(1usize << a) {
                    v[k | (last << a)] = Gf128((k == pc) as u128);
                }
            }
            None => {
                for k in 0..(1usize << a) {
                    v[k | (last << a)] = fetch.table[k | (last << a)];
                }
            }
        }
        v
    };
    // The eq-links L_t(t, j) = 1 + t + t̂(j), L_h(h, j) = 1 + h + ĥ(j).
    let l_t: Vec<Gf128> = (0..2 * n_cols)
        .map(|i| {
            let bit = i & 1;
            let j = i >> 1;
            ONE + Gf128(bit as u128) + taken[j]
        })
        .collect();
    let l_h: Vec<Gf128> = (0..2 * n_cols)
        .map(|i| {
            let bit = i & 1;
            let j = i >> 1;
            ONE + Gf128(bit as u128) + halt[j]
        })
        .collect();
    // The coefficients and the claim.
    let alpha0 = loop {
        let x = t.challenge_field("cs-alpha0");
        if x != ZERO {
            break x;
        }
    };
    let gamma = t.challenge_field("cs-gamma");
    let claim = alpha0 + gamma;
    // The sparse spec: dense [A, ASH, L_t, L_h], public [eq(r_j), α₀, TT]
    // — the k/k' spaces are SUMMED (the one-hot selects), so they carry
    // no eq factors; only the cycle space is eq-weighted.
    let r_j = t.challenge_field_vec("cs-rj", n);
    let dense = vec![
        SparseFactor::new(fetch.table.clone(), (0..a).chain(j0..j0 + n).collect()),
        SparseFactor::new(ash.clone(), (k1..k1 + a).chain(j0..j0 + n).collect()),
        SparseFactor::new(l_t, std::iter::once(t0).chain(j0..j0 + n).collect()),
        SparseFactor::new(l_h, std::iter::once(h0).chain(j0..j0 + n).collect()),
    ];
    let public = vec![
        SparseFactor::new(
            magnetar_field::EqBits::build(&r_j).values,
            (j0..j0 + n).collect(),
        ),
        SparseFactor::new(vec![alpha0], vec![]),
        SparseFactor::new(
            spec.tt.clone(),
            (k0..k0 + a)
                .chain(k1..k1 + a)
                .chain(std::iter::once(t0))
                .chain(std::iter::once(h0))
                .collect(),
        ),
    ];
    let spec_sc = SparseSpec {
        num_vars: jv,
        dense,
        public,
        terms: vec![
            Term {
                coeff: ONE,
                dense: vec![0, 1, 2, 3],
                public: vec![0, 1],
            },
            Term {
                coeff: gamma,
                dense: vec![0, 1, 2, 3],
                public: vec![0, 2],
            },
        ],
        claim,
    };
    let mut prover = SparseProver::new(spec_sc, BindOrder::Increasing);
    let mut s1_rounds = Vec::with_capacity(jv);
    for _ in 0..jv {
        let msg = prover.message();
        s1_rounds.push(msg.0.clone());
        prover.absorb_and_fold(t, "cs-s1");
    }
    let challenges = prover.challenges();
    let (terminals, _) = prover.finish();
    let r_joint = BindOrder::Increasing.point(jv, &challenges);
    let a_claim = terminals[0];
    let ash_claim = terminals[1];
    // Recover t̂/ĥ at the cycle point from the link terminals.
    let r_j_term: Vec<Gf128> = r_joint[j0..j0 + n].to_vec();
    let r_t = r_joint[t0];
    let r_h = r_joint[h0];
    let taken_value = terminals[2] + ONE + r_t;
    let halt_value = terminals[3] + ONE + r_h;
    // ---- S2: the shift discharge.
    let r_k2_s1: Vec<Gf128> = r_joint[k1..k1 + a].to_vec();
    let g: Vec<Gf128> = {
        let eqj = magnetar_field::EqBits::build(&r_j_term).values;
        let mut g = vec![ZERO; n_cols];
        // eq(r_j, j'−1): the predecessor of j' is j'−1.
        g[1..n_cols].copy_from_slice(&eqj[..n_cols - 1]);
        // The last cycle: the freeze contributes eq(r_j, n_cols−1) when the
        // public case is absent (ASH(n_cols−1) = A(n_cols−1)); with a public
        // boundary pc the last column's weight moves to the claim side.
        if spec.next_pc_public.is_none() {
            g[n_cols - 1] += eqj[n_cols - 1];
        }
        g
    };
    // claim_S2 = v_ASH + P.
    let p_term = match spec.next_pc_public {
        Some(pc) => {
            let mut e = magnetar_field::EqBits::build(&r_j_term).values[n_cols - 1];
            // eq(r_k2_s1, pc).
            for (i, &rk) in r_k2_s1.iter().enumerate() {
                let bit = (pc >> i) & 1 == 1;
                e = mul(e, if bit { rk } else { ONE + rk });
            }
            e
        }
        None => ZERO,
    };
    let s2_claim = ash_claim + p_term;
    let jv2 = a + n;
    let spec2 = SparseSpec {
        num_vars: jv2,
        dense: vec![SparseFactor::new(
            fetch.table.clone(),
            (0..a).chain(a..a + n).collect(),
        )],
        public: vec![
            SparseFactor::new(
                magnetar_field::EqBits::build(&r_k2_s1).values,
                (0..a).collect(),
            ),
            SparseFactor::new(g, (a..a + n).collect()),
        ],
        terms: vec![Term {
            coeff: ONE,
            dense: vec![0],
            public: vec![0, 1],
        }],
        claim: s2_claim,
    };
    let mut prover2 = SparseProver::new(spec2, BindOrder::Increasing);
    let mut s2_rounds = Vec::with_capacity(jv2);
    for _ in 0..jv2 {
        let msg = prover2.message();
        s2_rounds.push(msg.0.clone());
        prover2.absorb_and_fold(t, "cs-s2");
    }
    let ch2 = prover2.challenges();
    let (terms2, _) = prover2.finish();
    let r2 = BindOrder::Increasing.point(jv2, &ch2);
    let s2_a_point: Vec<Gf128> = r2[..a + n].to_vec();
    let s1_a_point: Vec<Gf128> = {
        let mut pt = r_joint[k0..k0 + a].to_vec();
        pt.extend_from_slice(&r_j_term);
        pt
    };
    (
        CrossStepProof {
            s1_rounds,
            s1_a_claim: a_claim,
            s1_ash_claim: ash_claim,
            s1_link_claims: [terminals[2], terminals[3]],
            s2_rounds,
            s2_a_claim: terms2[0],
        },
        CrossStepClaims {
            s1_a: PcsPoint {
                point: s1_a_point,
                value: a_claim,
            },
            s2_a: PcsPoint {
                point: s2_a_point,
                value: terms2[0],
            },
            taken_value,
            halt_value,
            r_j: r_j_term,
        },
    )
}

/// Verify the transition argument. `fetch` provides the component's shape
/// only (the verifier never sees the table). Returns the claims to
/// discharge.
#[allow(clippy::too_many_arguments)]
pub fn verify_cross_step(
    spec: &CrossStepSpec,
    a_vars: usize,
    n: usize,
    taken: Gf128,
    halt: Gf128,
    proof: &CrossStepProof,
    t: &mut Transcript,
) -> Result<CrossStepClaims, &'static str> {
    let a = a_vars;
    let n_cols = 1usize << n;
    let [k0, k1, t0, h0, j0] = s1_layout(a, n);
    let jv = 2 * a + 2 + n;
    // Coefficients and points.
    let alpha0 = loop {
        let x = t.challenge_field("cs-alpha0");
        if x != ZERO {
            break x;
        }
    };
    let gamma = t.challenge_field("cs-gamma");
    let claim = alpha0 + gamma;
    let r_j = t.challenge_field_vec("cs-rj", n);
    // The public factors in their own variable spaces (PaddedFactor —
    // the verifier evaluates them at the terminal point without any
    // 2^jv materialization). The k/k' spaces are summed (no eq factors).
    let pub_full: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> = vec![
        Box::new(magnetar_sumcheck::public::PaddedFactor::new(
            magnetar_field::EqBits::build(&r_j).values,
            (j0..j0 + n).collect(),
            jv,
        )),
        Box::new(magnetar_sumcheck::public::PaddedFactor::new(
            vec![alpha0],
            vec![],
            jv,
        )),
        Box::new(magnetar_sumcheck::public::PaddedFactor::new(
            spec.tt.clone(),
            (k0..k0 + a)
                .chain(k1..k1 + a)
                .chain(std::iter::once(t0))
                .chain(std::iter::once(h0))
                .collect(),
            jv,
        )),
    ];
    let terms = vec![
        Term {
            coeff: ONE,
            dense: vec![0, 1, 2, 3],
            public: vec![0, 1],
        },
        Term {
            coeff: gamma,
            dense: vec![0, 1, 2, 3],
            public: vec![0, 2],
        },
    ];
    let mut v = magnetar_sumcheck::engine::SumcheckVerifier::new(
        jv,
        claim,
        &pub_full,
        terms,
        BindOrder::Increasing,
    );
    let mut r_joint = Vec::with_capacity(jv);
    if proof.s1_rounds.len() != jv {
        return Err("cs: S1 round count");
    }
    for msg in &proof.s1_rounds {
        r_joint.push(v.round_capturing(&magnetar_sumcheck::RoundMsg(msg.clone()), t, "cs-s1")?);
    }
    let mut all = vec![proof.s1_a_claim, proof.s1_ash_claim];
    all.extend_from_slice(&proof.s1_link_claims);
    v.finish(&all)?;
    // Recover t̂/ĥ at the cycle point.
    let r_j_term: Vec<Gf128> = r_joint[j0..j0 + n].to_vec();
    let r_t = r_joint[t0];
    let r_h = r_joint[h0];
    let taken_value = proof.s1_link_claims[0] + ONE + r_t;
    let halt_value = proof.s1_link_claims[1] + ONE + r_h;
    // ---- S2.
    let r_k2_s1: Vec<Gf128> = r_joint[k1..k1 + a].to_vec();
    let g: Vec<Gf128> = {
        let eqj = magnetar_field::EqBits::build(&r_j_term).values;
        let mut g = vec![ZERO; n_cols];
        g[1..n_cols].copy_from_slice(&eqj[..n_cols - 1]);
        if spec.next_pc_public.is_none() {
            g[n_cols - 1] += eqj[n_cols - 1];
        }
        g
    };
    let p_term = match spec.next_pc_public {
        Some(pc) => {
            let mut e = magnetar_field::EqBits::build(&r_j_term).values[n_cols - 1];
            for (i, &rk) in r_k2_s1.iter().enumerate() {
                let bit = (pc >> i) & 1 == 1;
                e = mul(e, if bit { rk } else { ONE + rk });
            }
            e
        }
        None => ZERO,
    };
    let s2_claim = proof.s1_ash_claim + p_term;
    let jv2 = a + n;
    if proof.s2_rounds.len() != jv2 {
        return Err("cs: S2 round count");
    }
    let pub2: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> = vec![
        Box::new(magnetar_sumcheck::public::PaddedFactor::new(
            magnetar_field::EqBits::build(&r_k2_s1).values,
            (0..a).collect(),
            jv2,
        )),
        Box::new(magnetar_sumcheck::public::PaddedFactor::new(
            g,
            (a..a + n).collect(),
            jv2,
        )),
    ];
    let terms2 = vec![Term {
        coeff: ONE,
        dense: vec![0],
        public: vec![0, 1],
    }];
    let mut v2 = magnetar_sumcheck::engine::SumcheckVerifier::new(
        jv2,
        s2_claim,
        &pub2,
        terms2,
        BindOrder::Increasing,
    );
    let mut r2 = Vec::with_capacity(jv2);
    for msg in &proof.s2_rounds {
        r2.push(v2.round_capturing(&magnetar_sumcheck::RoundMsg(msg.clone()), t, "cs-s2")?);
    }
    v2.finish(&[proof.s2_a_claim])?;
    let s2_a_point: Vec<Gf128> = r2[..a + n].to_vec();
    let s1_a_point: Vec<Gf128> = {
        let mut pt = r_joint[k0..k0 + a].to_vec();
        pt.extend_from_slice(&r_j_term);
        pt
    };
    let _ = (taken, halt);
    Ok(CrossStepClaims {
        s1_a: PcsPoint {
            point: s1_a_point,
            value: proof.s1_a_claim,
        },
        s2_a: PcsPoint {
            point: s2_a_point,
            value: proof.s2_a_claim,
        },
        taken_value,
        halt_value,
        r_j: r_j_term,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bit(v: bool) -> Gf128 {
        Gf128(v as u128)
    }

    /// A small transition system: 4 rows, 8 cycles. Row 0 branches to
    /// row 1 or back to itself; row 1 falls to 2; row 2 to the halt row
    /// 3, which freezes.
    #[test]
    fn cross_step_roundtrip() {
        let a = 2usize;
        let n = 3usize;
        let n_cols = 1usize << n;
        let mut tt = vec![ZERO; 1usize << (2 * a + 2)];
        let set = |tt: &mut Vec<Gf128>, k: usize, n0: usize, n1: usize| {
            for t in 0..2usize {
                let target = if t == 0 { n0 } else { n1 };
                tt[k | (target << a) | (t << (2 * a))] = ONE;
                tt[k | (k << a) | (t << (2 * a)) | (1usize << (2 * a + 1))] = ONE;
            }
        };
        set(&mut tt, 0, 1, 0);
        set(&mut tt, 1, 2, 2);
        set(&mut tt, 2, 3, 3);
        set(&mut tt, 3, 3, 3);
        let path: Vec<usize> = vec![0, 1, 2, 3, 3, 3, 3, 3];
        let taken: Vec<Gf128> = vec![bit(false); n_cols];
        let halt: Vec<Gf128> = path.iter().map(|&k| bit(k == 3)).collect();
        let mut fetch_bits = vec![false; (1 << a) * n_cols];
        for (j, &k) in path.iter().enumerate() {
            fetch_bits[k | (j << a)] = true;
        }
        let fetch = AddressComponent::from_bits(a, n, &fetch_bits);
        let spec = CrossStepSpec {
            a,
            n,
            tt,
            next_pc_public: None,
        };
        let mut t = Transcript::new("cs", &[b"t"]);
        let (proof, claims) = prove_cross_step(&spec, &fetch, &taken, &halt, &mut t);
        let mut tv = Transcript::new("cs", &[b"t"]);
        let vclaims = verify_cross_step(
            &spec,
            a,
            n,
            claims.taken_value,
            claims.halt_value,
            &proof,
            &mut tv,
        )
        .expect("cross-step verifies");
        assert_eq!(claims.s1_a.point, vclaims.s1_a.point);
        assert_eq!(claims.s1_a.value, vclaims.s1_a.value);
        assert_eq!(claims.s2_a.point, vclaims.s2_a.point);
        assert_eq!(claims.s2_a.value, vclaims.s2_a.value);
        assert_eq!(
            claims.s1_a.value,
            magnetar_field::eval_mle(&fetch.table, &claims.s1_a.point)
        );
        assert_eq!(
            claims.s2_a.value,
            magnetar_field::eval_mle(&fetch.table, &claims.s2_a.point)
        );
        assert_eq!(
            claims.taken_value,
            magnetar_field::eval_mle(&taken, &claims.r_j)
        );
        assert_eq!(
            claims.halt_value,
            magnetar_field::eval_mle(&halt, &claims.r_j)
        );

        // A jumped pc must be rejected.
        {
            let bad_path: Vec<usize> = vec![0, 2, 3, 3, 3, 3, 3, 3];
            let mut fb = vec![false; (1 << a) * n_cols];
            for (j, &k) in bad_path.iter().enumerate() {
                fb[k | (j << a)] = true;
            }
            let bad_fetch = AddressComponent::from_bits(a, n, &fb);
            let mut t = Transcript::new("cs", &[b"t"]);
            let (p, _) = prove_cross_step(&spec, &bad_fetch, &taken, &halt, &mut t);
            let mut tv = Transcript::new("cs", &[b"t"]);
            assert!(
                verify_cross_step(&spec, a, n, ZERO, ZERO, &p, &mut tv).is_err(),
                "a jumped pc must be rejected"
            );
        }
        // The public-boundary mode: the right pc (the transition-dictated
        // successor — here the frozen halt row 3) verifies, a wrong one
        // fails.
        {
            let spec2 = CrossStepSpec {
                a,
                n,
                tt: spec.tt.clone(),
                next_pc_public: Some(3),
            };
            let mut t = Transcript::new("cs", &[b"t"]);
            let (p, c) = prove_cross_step(&spec2, &fetch, &taken, &halt, &mut t);
            let mut tv = Transcript::new("cs", &[b"t"]);
            assert!(
                verify_cross_step(&spec2, a, n, c.taken_value, c.halt_value, &p, &mut tv).is_ok()
            );
            let spec3 = CrossStepSpec {
                a,
                n,
                tt: spec.tt.clone(),
                next_pc_public: Some(1),
            };
            let mut tv = Transcript::new("cs", &[b"t"]);
            assert!(
                verify_cross_step(&spec3, a, n, c.taken_value, c.halt_value, &p, &mut tv).is_err()
            );
        }
        // Tampered rounds and claims.
        {
            let mut p = proof.clone();
            p.s1_rounds[0][1] += ONE;
            let mut tv = Transcript::new("cs", &[b"t"]);
            assert!(verify_cross_step(
                &spec,
                a,
                n,
                claims.taken_value,
                claims.halt_value,
                &p,
                &mut tv
            )
            .is_err());
        }
        {
            let mut p = proof.clone();
            p.s1_ash_claim += ONE;
            let mut tv = Transcript::new("cs", &[b"t"]);
            assert!(verify_cross_step(
                &spec,
                a,
                n,
                claims.taken_value,
                claims.halt_value,
                &p,
                &mut tv
            )
            .is_err());
        }
        {
            let mut p = proof.clone();
            p.s2_a_claim += ONE;
            let mut tv = Transcript::new("cs", &[b"t"]);
            assert!(verify_cross_step(
                &spec,
                a,
                n,
                claims.taken_value,
                claims.halt_value,
                &p,
                &mut tv
            )
            .is_err());
        }
    }
}
