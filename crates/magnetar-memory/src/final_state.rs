//! The **final-state binding** (the paper's Remark 4: "the binding of the
//! public initial and final states … enforced with the next multilinear
//! extension and one additional sum-check argument over the instance
//! coordinates").
//!
//! BinaryTwist proves that a memory evolves *from its public initial
//! contents* `m0` by the committed increments, but the final contents are
//! never claimed. This module binds a **public final-state string** to the
//! evolution with two chained sum-checks:
//!
//! * **Check A (the zero-check)** over the cell/bit space `(k, o)`:
//!   `Σ eq(ρ, k, o)·(D + Ev)(k, o) = 0`, where `D = PubFinal + m0` is
//!   public and `Ev` is the evolution table
//!   `Ev(k, o) = Σ_j W(k, j)·Inc(o, j)` (the sum over the one-hot write
//!   address components `W` and the committed increments `Inc`). Its
//!   terminal claim reduces to `Êv(ρ)` by the verifier's own evaluation
//!   of `D̂(ρ)`.
//! * **Check B (the evolution reduction)** over the joint `(k, j)` space
//!   with the sparse engine: `Σ eq(r_k, k)·Π_i A^w_i(k^(i), j)·Inc′(j)`,
//!   where `Inc′(j) = Σ_o eq(r_o, o)·Inc(o, j)` is the o-pre-folded
//!   increment column. Its terminals are the **existing committed
//!   claims**: the write-address components at `(r_k`-slices, `r_j)` and
//!   the increment at `(r_o, r_j)` — all discharged by MosaicPCS.
//!
//! Used twice per segment in tree-IVC: once for the register file and
//! once for RAM.

use crate::{AddressComponent, TwistWitness};
use magnetar_field::{eval_mle, mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_sumcheck::{
    engine::{SumcheckProver, SumcheckVerifier},
    public::TableFactor,
    sparse::{SparseFactor, SparseProver, SparseSpec},
    BindOrder, RoundMsg, SumcheckSpec, Term,
};

/// One final-state binding instance.
#[derive(Clone, Debug)]
pub struct FinalStateInstance {
    /// Cell address variables (a): 6 for registers, 8 for RAM.
    pub a: usize,
    /// Bit variables (ell): 6.
    pub ell: usize,
    /// Cycle variables (n).
    pub n: usize,
    /// The public initial memory `m0: {0,1}^{a+ell} → B` (address low,
    /// bit high).
    pub m0: Vec<Gf128>,
    /// The public claimed final memory, same layout.
    pub pub_final: Vec<Gf128>,
}

/// The binding's proof: Check A's rounds + terminal, Check B's sparse
/// rounds + the terminal claims (component evaluations + the increment
/// evaluation), which the caller discharges against the commitment.
#[derive(Clone, Debug)]
pub struct FinalStateProof {
    /// Check A's round messages (degree 2).
    pub zero_rounds: Vec<Vec<Gf128>>,
    /// Check A's terminal: the (D + Ev) MLE at `ρ`.
    pub diff_terminal: Gf128,
    /// Check B's round messages.
    pub evo_rounds: Vec<Vec<Gf128>>,
    /// The per-component terminal claims (in component order).
    pub component_claims: Vec<Gf128>,
    /// The o-pre-folded increment column's terminal claim.
    pub inc_claim: Gf128,
}

/// The claims Check B produces, for the caller's Mosaic discharge.
#[derive(Clone, Debug)]
pub struct FinalStateClaims {
    /// The write-address components' claims: `Â^w_i(r_k^(i), r_j)` per
    /// component `i`.
    pub component_points: Vec<(Vec<Gf128>, Gf128)>,
    /// The increment claim `Înc(r_o, r_j)`.
    pub inc_point: (Vec<Gf128>, Gf128),
}

/// Materialize the evolution table `Ev(k, o) = Σ_j W(k, j)·Inc(o, j)`.
fn materialize_ev(
    a: usize,
    ell: usize,
    n: usize,
    wfa_comps: &[AddressComponent],
    inc: &[Gf128],
) -> Vec<Gf128> {
    let cells = 1usize << a;
    let bits = 1usize << ell;
    let cycles = 1usize << n;
    let d = wfa_comps.len();
    debug_assert_eq!(d, 1, "the current instances use d = 1");
    let mut ev = vec![ZERO; cells * bits];
    for j in 0..cycles {
        // The one-hot write address of cycle j (skip all-zero columns).
        if let Some(k) = wfa_comps[0].addr_of(j) {
            for o in 0..bits {
                let v = inc[o | (j << ell)];
                if v != ZERO {
                    ev[k | (o << a)] += v;
                }
            }
        }
    }
    ev
}

impl AddressComponent {
    /// The one-hot address of cycle `j` (the single hot cell), or `None`
    /// if the column is all-zero.
    fn addr_of(&self, j: usize) -> Option<usize> {
        // Address part low, cycle part high: cycle j's block is
        // [j·2^addr_vars, (j+1)·2^addr_vars).
        let stride = 1usize << self.addr_vars;
        let col = &self.table[j * stride..(j + 1) * stride];
        col.iter().position(|&v| v == ONE)
    }
}

/// Prove the final-state binding. Returns the proof and the claims for
/// the Mosaic discharge.
pub fn prove_final_state(
    inst: &FinalStateInstance,
    wit: &TwistWitness,
    t: &mut Transcript,
) -> (FinalStateProof, FinalStateClaims) {
    let a = inst.a;
    let ell = inst.ell;
    let n = inst.n;
    let wport = wit.ports.last().expect("the write port");
    debug_assert_eq!(wport.len(), 1, "the current instances use d = 1");
    // The evolution table and the difference table D + Ev.
    let ev = materialize_ev(a, ell, n, wport, &wit.inc);
    if std::env::var("MAGNETAR_FS_DBG").is_ok() {
        for (x, (&m, &p)) in inst.m0.iter().zip(inst.pub_final.iter()).enumerate() {
            if m + p + ev[x] != ZERO {
                eprintln!(
                    "[fs-dbg] honest diff nonzero at {x} (k={}, o={})",
                    x & ((1usize << a) - 1),
                    x >> a
                );
                break;
            }
        }
    }
    let diff: Vec<Gf128> = inst
        .m0
        .iter()
        .zip(inst.pub_final.iter())
        .zip(ev.iter())
        .map(|((&m, &p), &e)| m + p + e)
        .collect();
    // ---- Check A: the zero-check over (k, o) — a + ell variables.
    let ko_vars = a + ell;
    let rho = t.challenge_field_vec("fs-rho", ko_vars);
    let eq_rho = magnetar_field::EqBits::build(&rho).values;
    // Claim: Σ eq(ρ, x)·diff(x) = 0 (the honest diff is all-zero).
    let spec = SumcheckSpec {
        num_vars: ko_vars,
        dense: vec![diff],
        public: vec![Box::new(TableFactor::new(eq_rho.clone()))],
        terms: vec![Term {
            coeff: ONE,
            dense: vec![0],
            public: vec![0],
        }],
        claim: ZERO,
    };
    let mut prover = SumcheckProver::new(&spec, BindOrder::Increasing);
    let mut zero_rounds = Vec::with_capacity(ko_vars);
    for _ in 0..ko_vars {
        let msg = prover.message();
        zero_rounds.push(msg.0.clone());
        prover.absorb_and_fold(t, "fs-zero");
    }
    let (dense_claims, _) = prover.finish();
    let diff_terminal = dense_claims[0];
    t.absorb_field("fs-terminal", &[diff_terminal]);
    // The verifier recovers Êv(ρ) = terminal + D̂(ρ).
    // D̂(ρ) = Σ eq(ρ, x)·(m0 + PubFinal)(x) — public, verifier-computable.
    let d_hat = {
        let mut dh = ZERO;
        for (x, (&m, &p)) in inst.m0.iter().zip(inst.pub_final.iter()).enumerate() {
            dh += mul(eq_rho[x], m + p);
        }
        dh
    };
    let ev_claim = diff_terminal + d_hat;
    // ---- Check B: the evolution reduction over the joint (k, j) space.
    // Êv(r_k, r_o) = Σ_{k,j} eq(r_k, k)·A^w(k, j)·Inc′(j).
    let (r_k, r_o): (Vec<Gf128>, Vec<Gf128>) = (rho[..a].to_vec(), rho[a..].to_vec());
    // The o-pre-folded increment column Inc′(j) = Σ_o eq(r_o, o)·Inc(o, j).
    let eq_o = magnetar_field::EqBits::build(&r_o).values;
    let mut incp = vec![ZERO; 1usize << n];
    for j in 0..(1usize << n) {
        let mut acc = ZERO;
        for o in 0..(1usize << ell) {
            acc += mul(eq_o[o], wit.inc[o | (j << ell)]);
        }
        incp[j] = acc;
    }
    // The joint (k, j) layout: k | j (k low).
    let jv = a + n;
    let comps = wport;
    let mut dense: Vec<SparseFactor> = comps
        .iter()
        .map(|c| {
            let mut active: Vec<usize> = (0..c.addr_vars).collect();
            active.extend((a..a + n).collect::<Vec<usize>>());
            SparseFactor::new(c.table.clone(), active)
        })
        .collect();
    dense.push(SparseFactor::new(incp, (a..a + n).collect()));
    let spec = SparseSpec {
        num_vars: jv,
        dense,
        public: vec![SparseFactor::new(
            magnetar_field::EqBits::build(&r_k).values,
            (0..a).collect(),
        )],
        terms: vec![Term {
            coeff: ONE,
            dense: (0..comps.len() + 1).collect(),
            public: vec![0],
        }],
        claim: ev_claim,
    };
    let mut prover = SparseProver::new(spec, BindOrder::Increasing);
    let mut evo_rounds = Vec::with_capacity(jv);
    for _ in 0..jv {
        let msg = prover.message();
        evo_rounds.push(msg.0.clone());
        prover.absorb_and_fold(t, "fs-evo");
    }
    let challenges = prover.challenges();
    let (claims, _) = prover.finish();
    let component_claims = claims[..comps.len()].to_vec();
    let inc_claim = claims[comps.len()];
    // The terminal point from the sparse prover's own challenges (bind
    // order = layout order).
    let r_joint = BindOrder::Increasing.point(jv, &challenges);
    let r_j: Vec<Gf128> = r_joint[a..].to_vec();
    // The components' claim points come from the Check-B challenges
    // (r_joint's k-slice — NOT ρ's k-part, which only weights the eq
    // public factor).
    let component_points = comps
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let slice = &r_joint[i * c.addr_vars..(i + 1) * c.addr_vars];
            let mut pt = slice.to_vec();
            pt.extend_from_slice(&r_j);
            (pt, component_claims[i])
        })
        .collect();
    let mut inc_pt = r_o.clone();
    inc_pt.extend_from_slice(&r_j);
    (
        FinalStateProof {
            zero_rounds,
            diff_terminal,
            evo_rounds,
            component_claims,
            inc_claim,
        },
        FinalStateClaims {
            component_points,
            inc_point: (inc_pt, inc_claim),
        },
    )
}

/// Verify the final-state binding. On success returns the claims the
/// caller must discharge against the commitment (the caller checks them
/// against its own expected component/Inc Mosaic claims).
pub fn verify_final_state(
    inst: &FinalStateInstance,
    proof: &FinalStateProof,
    t: &mut Transcript,
) -> Result<FinalStateClaims, &'static str> {
    let a = inst.a;
    let ell = inst.ell;
    let n = inst.n;
    let ko_vars = a + ell;
    // ---- Check A.
    let rho = t.challenge_field_vec("fs-rho", ko_vars);
    let eq_rho = magnetar_field::EqBits::build(&rho).values;
    let d_hat = {
        let mut dh = ZERO;
        for (x, (&m, &p)) in inst.m0.iter().zip(inst.pub_final.iter()).enumerate() {
            dh += mul(eq_rho[x], m + p);
        }
        dh
    };
    if proof.zero_rounds.len() != ko_vars {
        return Err("fs: zero-check round count");
    }
    let public: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(TableFactor::new(eq_rho))];
    let terms = vec![Term {
        coeff: ONE,
        dense: vec![0],
        public: vec![0],
    }];
    let mut v = SumcheckVerifier::new(ko_vars, ZERO, &public, terms, BindOrder::Increasing);
    for msg in &proof.zero_rounds {
        v.round_capturing(&RoundMsg(msg.clone()), t, "fs-zero")?;
    }
    v.finish(&[proof.diff_terminal])?;
    t.absorb_field("fs-terminal", &[proof.diff_terminal]);
    let ev_claim = proof.diff_terminal + d_hat;
    // ---- Check B.
    let (r_k, r_o): (Vec<Gf128>, Vec<Gf128>) = (rho[..a].to_vec(), rho[a..].to_vec());
    let jv = a + n;
    if proof.evo_rounds.len() != jv {
        return Err("fs: evolution round count");
    }
    // The public factor: eq(r_k) over the k dims; the dense factors are
    // witness tables — the verifier only checks the round sums and the
    // terminal identity with the CLAIMED evaluations.
    // eq(r_k) over the k dims, padded across the j dims of the joint space.
    let public_b: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(magnetar_sumcheck::public::PaddedFactor::new(
            magnetar_field::EqBits::build(&r_k).values,
            (0..a).collect(),
            jv,
        ))];
    let n_comp = proof.component_claims.len();
    let terms_b = vec![Term {
        coeff: ONE,
        dense: (0..n_comp + 1).collect(),
        public: vec![0],
    }];
    let mut vb = SumcheckVerifier::new(jv, ev_claim, &public_b, terms_b, BindOrder::Increasing);
    let mut r_joint = Vec::with_capacity(jv);
    for msg in &proof.evo_rounds {
        r_joint.push(vb.round_capturing(&RoundMsg(msg.clone()), t, "fs-evo")?);
    }
    let mut all = proof.component_claims.clone();
    all.push(proof.inc_claim);
    vb.finish(&all)?;
    let r_j: Vec<Gf128> = r_joint[a..].to_vec();
    let mut inc_pt = r_o.clone();
    inc_pt.extend_from_slice(&r_j);
    Ok(FinalStateClaims {
        component_points: (0..n_comp)
            .map(|i| {
                // d = 1 in the current instances: the single component's
                // k-slice is all of r_joint's k-part.
                let mut pt = r_joint[..a].to_vec();
                pt.extend_from_slice(&r_j);
                (pt, proof.component_claims[i])
            })
            .collect(),
        inc_point: (inc_pt, proof.inc_claim),
    })
}

/// Evaluate the MLE of a table at a point (a public helper for callers).
pub fn mle(table: &[Gf128], point: &[Gf128]) -> Gf128 {
    eval_mle(table, point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_field::Gf128;

    fn bit(v: bool) -> Gf128 {
        Gf128(v as u128)
    }

    /// A small memory: a = 2 (4 cells), ell = 2 (4 bits), n = 3 (8 cycles).
    /// The write port writes a sequence of addresses; the increments XOR.
    /// The final state must bind to the true evolution.
    #[test]
    fn final_state_roundtrip() {
        let a = 2usize;
        let ell = 2usize;
        let n = 3usize;
        let cells = 1usize << a;
        let bits = 1usize << ell;
        let cycles = 1usize << n;
        // The write plan: (address, new word) per cycle.
        let plan: Vec<(usize, u64)> = vec![
            (0, 0b0101),
            (2, 0b1111),
            (0, 0b1010),
            (1, 0b0001),
            (3, 0b1100),
            (0, 0b0101),
            (2, 0b0000),
            (1, 0b0011),
        ];
        // m0: the initial words.
        let init: Vec<u64> = vec![0b0000, 0b0011, 0b1110, 0b0100];
        // The true final state.
        let mut mem = init.clone();
        for &(k, w) in &plan {
            mem[k] = w;
        }
        // The witness: the one-hot write component (a = 2) and Inc.
        let mut wfa_bits = vec![false; cells * cycles];
        let mut inc = vec![ZERO; bits * cycles];
        for (j, &(k, w)) in plan.iter().enumerate() {
            wfa_bits[k | (j * cells)] = true;
            let old = init_or_mem(&mut mem_dummy(), j, &plan, &init);
            for o in 0..bits {
                let b = ((w >> o) & 1) ^ ((old >> o) & 1);
                inc[o | (j << ell)] = bit(b == 1);
            }
            let _ = old;
        }
        let wfa = AddressComponent::from_bits(a, n, &wfa_bits);
        // The read port + write port structure the twist witness needs:
        // ports = [read₁, read₂, write]; we only need the write port.
        let wit = TwistWitness {
            ports: vec![vec![], vec![], vec![wfa]],
            inc,
        };
        // The instances.
        // The m0/pub_final layout: address low, bit high (k | o << a).
        let m0: Vec<Gf128> = {
            let mut v = Vec::with_capacity(cells * bits);
            for o in 0..bits {
                for k in 0..cells {
                    v.push(bit(((init[k] >> o) & 1) == 1));
                }
            }
            v
        };
        let pub_final: Vec<Gf128> = {
            let mut v = Vec::with_capacity(cells * bits);
            for o in 0..bits {
                for k in 0..cells {
                    v.push(bit(((mem[k] >> o) & 1) == 1));
                }
            }
            v
        };
        let inst = FinalStateInstance {
            a,
            ell,
            n,
            m0: m0.clone(),
            pub_final: pub_final.clone(),
        };
        // Debug: the true Check-B sum vs the claim the prover derives.
        {
            let ev = materialize_ev(a, ell, n, &wit.ports[2], &wit.inc);
            let d: Vec<Gf128> = inst
                .m0
                .iter()
                .zip(inst.pub_final.iter())
                .zip(ev.iter())
                .map(|((&m, &p), &e)| m + p + e)
                .collect();
            assert!(d.iter().all(|&v| v == ZERO), "the honest diff must vanish");
        }
        let mut t = Transcript::new("fs", &[b"t"]);
        let (proof, claims) = prove_final_state(&inst, &wit, &mut t);
        let mut tv = Transcript::new("fs", &[b"t"]);
        let vclaims = verify_final_state(&inst, &proof, &mut tv).expect("fs verifies");
        // The claims match (the points and values).
        assert_eq!(
            claims.component_points.len(),
            vclaims.component_points.len()
        );
        for (a, b) in claims
            .component_points
            .iter()
            .zip(vclaims.component_points.iter())
        {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
        }
        assert_eq!(claims.inc_point.0, vclaims.inc_point.0);
        assert_eq!(claims.inc_point.1, vclaims.inc_point.1);
        // The component claim is the true one-hot MLE at the point.
        let (pt, val) = &claims.component_points[0];
        let want = eval_mle(&wit.ports[2][0].table, pt);
        assert_eq!(*val, want);
        // The increment claim is the true Inc MLE.
        let (ipt, ival) = &claims.inc_point;
        let wanti = eval_mle(&wit.inc, ipt);
        if *ival != wanti {
            eprintln!("inc pt: {ipt:?}");
            eprintln!("ival: {ival:?} wanti: {wanti:?}");
        }
        assert_eq!(*ival, wanti);

        // Tamper: a wrong public final string must fail the zero-check.
        let mut pf2 = pub_final.clone();
        pf2[0] += ONE;
        let inst2 = FinalStateInstance {
            a,
            ell,
            n,
            m0,
            pub_final: pf2,
        };
        let mut tv = Transcript::new("fs", &[b"t"]);
        assert!(verify_final_state(&inst2, &proof, &mut tv).is_err());
        // Tamper: the diff terminal.
        let mut p = proof.clone();
        p.diff_terminal += ONE;
        let mut tv = Transcript::new("fs", &[b"t"]);
        assert!(verify_final_state(&inst, &p, &mut tv).is_err());
        // Tamper: a component claim.
        let mut p = proof.clone();
        p.component_claims[0] += ONE;
        let mut tv = Transcript::new("fs", &[b"t"]);
        assert!(verify_final_state(&inst, &p, &mut tv).is_err());
        // Tamper: the Inc claim.
        let mut p = proof.clone();
        p.inc_claim += ONE;
        let mut tv = Transcript::new("fs", &[b"t"]);
        assert!(verify_final_state(&inst, &p, &mut tv).is_err());
    }

    fn mem_dummy() -> Vec<u64> {
        vec![0; 16]
    }

    fn init_or_mem(_m: &mut Vec<u64>, j: usize, plan: &[(usize, u64)], init: &[u64]) -> u64 {
        // The memory content just before cycle j.
        let mut mem = init.to_vec();
        for &(k, w) in plan.iter().take(j) {
            mem[k] = w;
        }
        mem[plan[j].0]
    }
}
