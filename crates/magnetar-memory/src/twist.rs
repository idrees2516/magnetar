//! BinaryTwist (paper §4.4): a mutable memory of `K = 2^a` cells of
//! `L = 2^ℓ` bits, `T = 2^n` cycles, `p` reads + one write per cycle,
//! checked via the three identities:
//!
//! * **reconstruction (strict)** `Val(k,o,j) = M0(k,o) + Σ_{j'<j} wfa(k,j')·Inc(o,j')`,
//! * **read values** `rv_s(o,j) = Σ_k ra_s(k,j)·Val(k,o,j)`,
//! * **write value** `wv(o,j) = Σ_k wfa(k,j)·(Val(k,o,j)+Inc(o,j))`,
//!   where `Val+Inc` is the post-write memory
//!   `M0 + Σ_{j'≤j} wfa·Inc` (the **inclusive** LT).
//!
//! Plus the sum-to-one argument (the identities are homogeneous) and the
//! Δ* collision check per component (shared with Shout). Value sum-checks
//! run over the joint (k, o, j) space; factors are padded across variables
//! they do not depend on (padding leaves the MLE unchanged).

use crate::claims::{PcsPoint, TwistPcsClaims};
use crate::linear::{addr_bits, prove_port_linear, suffix_sums, TwistShared};
use crate::{AddressComponent, OneHotProof};
use magnetar_field::{mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_sumcheck::{
    engine::{SumcheckProver, SumcheckVerifier},
    public::{EqProdFactor, TableFactor},
    BindOrder, SumcheckSpec, Term,
};

/// A drop-guard wall-clock phase timer (MAGNETAR_TIME=1 instrumentation).
struct Pt {
    label: &'static str,
    start: std::time::Instant,
}
impl Pt {
    fn new(label: &'static str) -> Self {
        Pt {
            label,
            start: std::time::Instant::now(),
        }
    }
}
impl Drop for Pt {
    fn drop(&mut self) {
        if std::env::var("MAGNETAR_TIME").is_ok() {
            eprintln!(
                "[time] {:<24} {:>9.1} ms",
                self.label,
                self.start.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}

/// A BinaryTwist instance.
#[derive(Clone, Debug)]
pub struct TwistInstance {
    pub a: usize,
    pub ell: usize,
    pub n: usize,
    pub p: usize,
    /// Initial memory `M0: {0,1}^{a+ell} → B` (address low bits, bit index
    /// high).
    pub m0: Vec<Gf128>,
    /// Claimed read values `rv_s(r_val, r_cycle)` per port.
    pub rv_claims: Vec<Gf128>,
    /// Claimed write value `wv(r_val, r_cycle)`.
    pub wv_claim: Gf128,
    /// The common value point.
    pub r_val: Vec<Gf128>,
    /// The common cycle point.
    pub r_cycle: Vec<Gf128>,
}

/// The committed witness data.
#[derive(Clone, Debug)]
pub struct TwistWitness {
    /// `p` read sequences + 1 write sequence, each with `d` components.
    pub ports: Vec<Vec<AddressComponent>>,
    /// Increments `Inc: {0,1}^{ell+n} → B` (bit low, cycle high).
    pub inc: Vec<Gf128>,
}

impl TwistWitness {
    pub fn d(&self) -> usize {
        self.ports[0].len()
    }
}

/// One port's read/write sum-check output.
#[derive(Clone, Debug)]
pub struct PortRun {
    pub rounds: Vec<Vec<Gf128>>,
    /// The joint point (original coordinates).
    pub r_joint: Vec<Gf128>,
    /// The full-address terminal claim (product of component claims).
    pub ra_claim: Gf128,
    /// The Val (or Val+Inc) terminal claim.
    pub val_claim: Gf128,
}

/// One reconstruction sum-check output.
#[derive(Clone, Debug)]
pub struct ReconRun {
    pub rounds: Vec<Vec<Gf128>>,
    /// The full-address product terminal claim (checked against the
    /// product of `wfa_component_claims` — address virtualization).
    pub wfa_claim: Gf128,
    /// The per-component claims at the recon's point: `Â^w_i(r_k^{(i)},
    /// r_c)` (proven against the commitment by the zkvm layer).
    pub wfa_component_claims: Vec<Gf128>,
    pub inc_claim: Gf128,
    /// The recon's random cycle point (binding order = variable order).
    pub r_cycle: Vec<Gf128>,
}

/// A BinaryTwist proof.
#[derive(Clone, Debug)]
pub struct TwistProof {
    pub read_runs: Vec<PortRun>,
    pub write_run: PortRun,
    pub read_recons: Vec<ReconRun>,
    pub write_recon: ReconRun,
    /// Per-port component claims (address virtualization): `d` values per
    /// port at the port's read point.
    pub read_component_claims: Vec<Vec<Gf128>>,
    /// The write port's component claims.
    pub write_component_claims: Vec<Gf128>,
    /// Sum-to-one rounds per (port, component) and the terminal claims.
    pub st1: Vec<Vec<Vec<Vec<Gf128>>>>,
    pub st1_claims: Vec<Vec<Gf128>>,
    /// Δ* proofs per (port, component).
    pub onehot: Vec<Vec<OneHotProof>>,
}

/// `LT_strict(j', j) = [j' < j]`, `LT_le(j', j) = [j' ≤ j]`.
/// (Kept for the dense cross-check tests; the protocol paths use the
/// O(T) suffix sums.)
#[cfg(test)]
fn lt_table(n: usize, inclusive: bool) -> Vec<Gf128> {
    let s = 1usize << n;
    let mut t = vec![ZERO; s * s];
    for j in 0..s {
        for jp in 0..j + usize::from(inclusive) {
            t[jp + j * s] = ONE;
        }
    }
    t
}

/// Materialize `Val` (strict) or `Val+Inc` (inclusive) incrementally.
/// (Kept for the dense cross-check tests; the protocol prover is the
/// linear-in-T `prove_port_linear`.)
#[cfg(test)]
fn materialize_val(
    a: usize,
    ell: usize,
    n: usize,
    m0: &[Gf128],
    wfa_comps: &[AddressComponent],
    inc: &[Gf128],
    inclusive: bool,
) -> Vec<Gf128> {
    let d = wfa_comps.len();
    let ad = a / d;
    let cells = 1usize << a;
    let bits = 1usize << ell;
    let cycles = 1usize << n;
    let mut val = vec![ZERO; cells * bits * cycles];
    let wfa_at = |k: usize, j: usize| -> bool {
        for (i, comp) in wfa_comps.iter().enumerate() {
            let slice = (k >> (i * ad)) & ((1usize << ad) - 1);
            if comp.table[comp.index_with(slice, j)] != ONE {
                return false;
            }
        }
        true
    };
    for k in 0..cells {
        for o in 0..bits {
            for j in 0..cycles {
                val[k + o * cells + j * cells * bits] = m0[k | (o << a)];
            }
        }
        for jp in 0..cycles {
            if wfa_at(k, jp) {
                for o in 0..bits {
                    let inc_bit = inc[o | (jp << ell)];
                    for j in (jp + usize::from(!inclusive))..cycles {
                        let idx = k + o * cells + j * cells * bits;
                        val[idx] += inc_bit;
                    }
                }
            }
        }
    }
    val
}

/// The joint-space padded table of the full address (product of components).
#[cfg(test)]
fn full_addr_joint(comps: &[AddressComponent], a: usize, ell: usize, n: usize) -> Vec<Gf128> {
    let d = comps.len();
    let ad = a / d;
    let joint_vars = a + ell + n;
    let mut out = vec![ZERO; 1usize << joint_vars];
    for idx in 0..out.len() {
        let k = idx & ((1usize << a) - 1);
        let j = idx >> (a + ell);
        let mut acc = ONE;
        for (i, comp) in comps.iter().enumerate() {
            let slice = (k >> (i * ad)) & ((1usize << ad) - 1);
            acc = mul(acc, comp.table[comp.index_with(slice, j)]);
        }
        out[idx] = acc;
    }
    out
}

/// Run one value sum-check (read or write shape) capturing the point —
/// the **dense** reference prover (kept for the byte-identity
/// cross-check against the linear prover).
#[cfg(test)]
fn run_port_sumcheck(
    comps: &[AddressComponent],
    val_table: &[Gf128],
    claim: Gf128,
    joint_vars: usize,
    pub_table: &[Gf128],
    a: usize,
    ell: usize,
    n: usize,
    label: &str,
    t: &mut Transcript,
) -> PortRun {
    let ra_joint = full_addr_joint(comps, a, ell, n);
    let spec = SumcheckSpec {
        num_vars: joint_vars,
        dense: vec![ra_joint, val_table.to_vec()],
        public: vec![Box::new(TableFactor::new(pub_table.to_vec()))],
        terms: vec![Term {
            coeff: ONE,
            dense: vec![0, 1],
            public: vec![0],
        }],
        claim,
    };
    let mut p = SumcheckProver::new(&spec, BindOrder::Increasing);
    let mut rounds = Vec::with_capacity(joint_vars);
    for _ in 0..joint_vars {
        let msg = p.message();
        rounds.push(msg.0.clone());
        p.absorb_and_fold(t, label);
    }
    let r_joint = p.challenges();
    let (claims, _) = p.finish();
    PortRun {
        rounds,
        r_joint,
        ra_claim: claims[0],
        val_claim: claims[1],
    }
}

/// Run one reconstruction sum-check at a joint point.
#[allow(clippy::too_many_arguments)]
fn run_reconstruction(
    wfa_comps: &[AddressComponent],
    wfa_comp_addrs: &[Vec<usize>],
    inc: &[Gf128],
    val_claim: Gf128,
    r_joint: &[Gf128],
    a: usize,
    ell: usize,
    n: usize,
    m0: &[Gf128],
    inclusive: bool,
    label: &str,
    t: &mut Transcript,
) -> ReconRun {
    let d = wfa_comps.len();
    let ad = a / d;
    let r_k: Vec<Gf128> = r_joint[..a].to_vec();
    let r_v: Vec<Gf128> = r_joint[a..a + ell].to_vec();
    let r_c: Vec<Gf128> = r_joint[a + ell..].to_vec();
    let m0_at = magnetar_field::eval_mle(m0, &[r_k.clone(), r_v.clone()].concat());
    // LT column at r_c: the suffix sums of eq(r_c, ·) — O(T), not the
    // O(T²) less-than table.
    let eq_j = magnetar_field::EqBits::build(&r_c).values;
    let lt_col = suffix_sums(&eq_j, inclusive);
    // wfa row at r_k: the one-hot address product — O(T·a), not O(T·K).
    // The full per-cycle write address = the OR of the component traces.
    let w_addr: Vec<usize> = {
        let mut acc = vec![0usize; 1usize << n];
        for tr in wfa_comp_addrs {
            for (j, &a) in tr.iter().enumerate() {
                acc[j] |= a;
            }
        }
        acc
    };
    let wfa_row: Vec<Gf128> = (0..(1usize << n))
        .map(|jp| {
            let mut acc = ONE;
            for (bit_idx, &rk) in r_k.iter().enumerate() {
                acc = mul(acc, ONE + rk + Gf128(((w_addr[jp] >> bit_idx) & 1) as u128));
            }
            acc
        })
        .collect();
    // inc row at r_v: fold the ell low (bit) variables.
    let mut inc_row = inc.to_vec();
    for b in 0..ell {
        inc_row = {
            let mut v = inc_row;
            magnetar_field::par_bind_var(&mut v, 0, r_v[b]);
            v
        };
    }
    debug_assert_eq!(inc_row.len(), 1usize << n);
    let spec = SumcheckSpec {
        num_vars: n,
        dense: vec![wfa_row, inc_row],
        public: vec![Box::new(TableFactor::new(lt_col))],
        terms: vec![Term {
            coeff: ONE,
            dense: vec![0, 1],
            public: vec![0],
        }],
        claim: val_claim + m0_at,
    };
    let mut p = SumcheckProver::new(&spec, BindOrder::Increasing);
    let mut rounds = Vec::with_capacity(n);
    for _ in 0..n {
        let msg = p.message();
        rounds.push(msg.0.clone());
        p.absorb_and_fold(t, label);
    }
    let r_c = p.challenges();
    let (claims, _) = p.finish();
    // The per-component claims at the recon's point (address
    // virtualization): `Â^w_i(r_k^{(i)}, r_c)`.
    let wfa_component_claims: Vec<Gf128> = wfa_comps
        .iter()
        .enumerate()
        .map(|(i, comp)| {
            let slice: Vec<Gf128> = (0..ad).map(|b| r_k[i * ad + b]).collect();
            comp.eval_at_sparse(&slice, &r_c, &wfa_comp_addrs[i])
        })
        .collect();
    for (i, &cv) in wfa_component_claims.iter().enumerate() {
        t.absorb_field(&format!("{label}-wfacomp-{i}"), &[cv]);
    }
    ReconRun {
        rounds,
        wfa_claim: claims[0],
        wfa_component_claims,
        inc_claim: claims[1],
        r_cycle: r_c,
    }
}

/// Prove BinaryTwist. Returns the proof and the PCS claims it leaves open
/// (the caller proves them against the Mosaic commitment).
pub fn prove_twist(
    inst: &TwistInstance,
    wit: &TwistWitness,
    t: &mut Transcript,
) -> (TwistProof, TwistPcsClaims) {
    let (a, ell, n, p) = (inst.a, inst.ell, inst.n, inst.p);
    let d = wit.d();
    // The per-(port, component) cycle-address traces — computed once,
    // threaded through the claims, st1, onehot, and reconstruction paths.
    let port_addrs: Vec<Vec<Vec<usize>>> = wit
        .ports
        .iter()
        .map(|comps| {
            comps
                .iter()
                .map(|c| crate::linear::addr_bits(std::slice::from_ref(c)))
                .collect()
        })
        .collect();
    // The shared linear-prover precomputation (the value-pre-folded
    // increments and initial memory, the write-address trace).
    let w_addr = addr_bits(&wit.ports[p]);
    let shared = TwistShared::new(inst, w_addr.clone(), &wit.inc);
    // Read sum-checks (the linear-in-T prover: cycle enumeration with
    // the one-hot kill, never the joint materialization).
    let mut read_runs = Vec::with_capacity(p);
    let mut read_recons = Vec::with_capacity(p);
    let mut read_component_claims = Vec::with_capacity(p);
    for s in 0..p {
        let _pt = Pt::new("tw-port-read");
        let run = prove_port_linear(
            inst,
            &shared,
            &wit.inc,
            &wit.ports[s],
            inst.rv_claims[s],
            false,
            &format!("twist-read-{s}"),
            t,
        );
        drop(_pt);
        // Component claims (address virtualization) at the run's point.
        let ad = a / d;
        let comps = &wit.ports[s];
        let comp_claims: Vec<Gf128> = comps
            .iter()
            .enumerate()
            .map(|(i, comp)| {
                let slice: Vec<Gf128> = (0..ad).map(|b| run.r_joint[i * ad + b]).collect();
                let r_j: Vec<Gf128> = run.r_joint[a + ell..].to_vec();
                comp.eval_at_sparse(&slice, &r_j, &port_addrs[s][i])
            })
            .collect();
        for (i, &v) in comp_claims.iter().enumerate() {
            t.absorb_field(&format!("twist-comp-{s}-{i}"), &[v]);
        }
        let _pt2 = Pt::new("tw-recon");
        let recon = run_reconstruction(
            &wit.ports[p],
            &port_addrs[p],
            &wit.inc,
            run.val_claim,
            &run.r_joint,
            a,
            ell,
            n,
            &inst.m0,
            false,
            &format!("twist-recon-{s}"),
            t,
        );
        drop(_pt2);
        t.absorb_field(
            &format!("twist-recon-claims-{s}"),
            &[recon.wfa_claim, recon.inc_claim],
        );
        read_component_claims.push(comp_claims);
        read_recons.push(recon);
        read_runs.push(run);
    }
    // Write sum-check (Val + Inc).
    let _ptw = Pt::new("tw-port-write");
    let write_run = prove_port_linear(
        inst,
        &shared,
        &wit.inc,
        &wit.ports[p],
        inst.wv_claim,
        true,
        "twist-write",
        t,
    );
    drop(_ptw);
    let ad = a / d;
    let wcomps = &wit.ports[p];
    let write_component_claims: Vec<Gf128> = wcomps
        .iter()
        .enumerate()
        .map(|(i, comp)| {
            let slice: Vec<Gf128> = (0..ad).map(|b| write_run.r_joint[i * ad + b]).collect();
            let r_j: Vec<Gf128> = write_run.r_joint[a + ell..].to_vec();
            comp.eval_at_sparse(&slice, &r_j, &port_addrs[p][i])
        })
        .collect();
    for (i, &v) in write_component_claims.iter().enumerate() {
        t.absorb_field(&format!("twist-wcomp-{i}"), &[v]);
    }
    let write_recon = run_reconstruction(
        wcomps,
        &port_addrs[p],
        &wit.inc,
        write_run.val_claim,
        &write_run.r_joint,
        a,
        ell,
        n,
        &inst.m0,
        true,
        "twist-wrecon",
        t,
    );
    t.absorb_field(
        "twist-wrecon-claims",
        &[write_recon.wfa_claim, write_recon.inc_claim],
    );
    // Sum-to-one per (port, component).
    let _ptst1 = Pt::new("tw-st1");
    let mut st1 = Vec::new();
    let mut st1_claims = Vec::new();
    let mut st1_pcs: Vec<Vec<PcsPoint>> = Vec::new();
    for (port, comps) in wit.ports.iter().enumerate() {
        let mut rounds_per = Vec::new();
        let mut claims_per = Vec::new();
        let mut pcs_per = Vec::new();
        for (i, comp) in comps.iter().enumerate() {
            let rho = t.challenge_field_vec(&format!("twist-st1-{port}-{i}-rho"), n);
            // The linear sum-to-one prover (the one-hot kill): O(T) per
            // round over the per-cycle address trace — no 2^{ad+n} table
            // clone, no dense engine pass. Messages byte-identical to the
            // generic engine's (cross-validated by tests).
            let (rounds, st1_pt, claim) = crate::linear::prove_st1_linear(
                &port_addrs[port][i],
                ad,
                &rho,
                t,
                &format!("twist-st1-{port}-{i}"),
            );
            rounds_per.push(rounds);
            claims_per.push(claim);
            pcs_per.push(PcsPoint::new(st1_pt, claim));
            let _ = comp;
        }
        st1.push(rounds_per);
        st1_claims.push(claims_per);
        st1_pcs.push(pcs_per);
    }
    drop(_ptst1);
    for (port, claims) in st1_claims.iter().enumerate() {
        for (i, v) in claims.iter().enumerate() {
            t.absorb_field(&format!("twist-st1-claim-{port}-{i}"), &[*v]);
        }
    }
    // Δ* per (port, component).
    let _ptoh = Pt::new("tw-onehot");
    let mut onehot = Vec::new();
    let mut onehot_pts = Vec::new();
    for (port, comps) in wit.ports.iter().enumerate() {
        let mut per = Vec::new();
        let mut per_pts = Vec::new();
        for (i, comp) in comps.iter().enumerate() {
            let (oh, cyc_pt) = crate::shout::prove_onehot(
                comp,
                &port_addrs[port][i],
                t,
                &format!("twist-onehot-{port}-{i}"),
            );
            per.push(oh);
            per_pts.push(cyc_pt);
        }
        onehot.push(per);
        onehot_pts.push(per_pts);
    }
    drop(_ptoh);
    // Assemble the PCS claims.
    let mk_comp_pt = |r_joint: &[Gf128], i: usize, ad: usize, a: usize, n: usize| {
        let mut pt = r_joint[i * ad..(i + 1) * ad].to_vec();
        pt.extend_from_slice(&r_joint[a + ell..a + ell + n]);
        pt
    };
    let read_addr: Vec<Vec<PcsPoint>> = (0..p)
        .map(|s| {
            let run = &read_runs[s];
            (0..d)
                .map(|i| {
                    PcsPoint::new(
                        mk_comp_pt(&run.r_joint, i, ad, a, n),
                        read_component_claims[s][i],
                    )
                })
                .collect()
        })
        .collect();
    let write_addr: Vec<PcsPoint> = (0..d)
        .map(|i| {
            PcsPoint::new(
                mk_comp_pt(&write_run.r_joint, i, ad, a, n),
                write_component_claims[i],
            )
        })
        .collect();
    let all_recons: Vec<&ReconRun> = read_recons
        .iter()
        .chain(std::iter::once(&write_recon))
        .collect();
    let all_runs: Vec<&PortRun> = read_runs
        .iter()
        .chain(std::iter::once(&write_run))
        .collect();
    let recon_wfa: Vec<Vec<PcsPoint>> = all_recons
        .iter()
        .zip(all_runs.iter())
        .map(|(recon, run)| {
            (0..d)
                .map(|i| {
                    let mut pt = run.r_joint[i * ad..(i + 1) * ad].to_vec();
                    pt.extend_from_slice(&recon.r_cycle);
                    PcsPoint::new(pt, recon.wfa_component_claims[i])
                })
                .collect()
        })
        .collect();
    let inc: Vec<PcsPoint> = all_recons
        .iter()
        .zip(all_runs.iter())
        .map(|(recon, run)| {
            let mut pt = run.r_joint[a..a + ell].to_vec();
            pt.extend_from_slice(&recon.r_cycle);
            PcsPoint::new(pt, recon.inc_claim)
        })
        .collect();
    let onehot_claims: Vec<Vec<[PcsPoint; 2]>> = onehot
        .iter()
        .zip(onehot_pts.iter())
        .map(|(per, pts)| {
            per.iter()
                .zip(pts.iter())
                .map(|(oh, cyc)| {
                    let mut pt_r = oh.r.clone();
                    pt_r.extend_from_slice(cyc);
                    let mut pt_rp = oh.r_prime.clone();
                    pt_rp.extend_from_slice(cyc);
                    [
                        PcsPoint::new(pt_r, oh.v_at_r),
                        PcsPoint::new(pt_rp, oh.v_at_rp),
                    ]
                })
                .collect()
        })
        .collect();
    (
        TwistProof {
            read_runs,
            write_run,
            read_recons,
            write_recon,
            read_component_claims,
            write_component_claims,
            st1,
            st1_claims,
            onehot,
        },
        TwistPcsClaims {
            read_addr,
            write_addr,
            recon_wfa,
            inc,
            st1: st1_pcs,
            onehot: onehot_claims,
        },
    )
}

/// Verify a BinaryTwist proof. Returns the claims left for the PCS (the
/// components and increments), which the caller proves against the Mosaic
/// commitment.
pub fn verify_twist(
    inst: &TwistInstance,
    d: usize,
    proof: &TwistProof,
    t: &mut Transcript,
) -> Result<TwistPcsClaims, &'static str> {
    let (a, ell, n, p) = (inst.a, inst.ell, inst.n, inst.p);
    let joint_vars = a + ell + n;
    let ad = a / d;
    // The public factor `eq(r_val, o)·eq(r_cycle, j)` in structured form:
    // the verifier evaluates it at the terminal point in O(ell + n) —
    // no joint-cube materialization.
    let pubf: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(EqProdFactor::new(
            vec![
                (inst.r_val.clone(), (a..a + ell).collect()),
                (inst.r_cycle.clone(), (a + ell..a + ell + n).collect()),
            ],
            joint_vars,
        ))];
    let terms = vec![Term {
        coeff: ONE,
        dense: vec![0, 1],
        public: vec![0],
    }];
    // Read runs.
    for (s, run) in proof.read_runs.iter().enumerate() {
        // The dense claims: the address claim = the product of the
        // component claims; the Val claim comes from the reconstruction.
        let ra_claim: Gf128 = proof.read_component_claims[s]
            .iter()
            .copied()
            .fold(ONE, mul);
        let recon = &proof.read_recons[s];
        // The reconstruction's Val claim: recompute via its own sum-check
        // below; here we need the terminal value: the recon proves
        // v_val + M0 = Σ LT·wfa·inc — the Val claim is the recon's claim
        // minus M0 at the point — the verifier recomputes M0 at the run's
        // point (public!).
        let r_k: Vec<Gf128> = run.r_joint[..a].to_vec();
        let r_v: Vec<Gf128> = run.r_joint[a..a + ell].to_vec();
        let _m0_at = magnetar_field::eval_mle(&inst.m0, &[r_k, r_v].concat());
        // Replay the read sum-check; the transmitted Val claim is checked
        // against the terminal identity and then reconstructed.
        let mut v = SumcheckVerifier::new(
            joint_vars,
            inst.rv_claims[s],
            &pubf,
            terms.clone(),
            BindOrder::Increasing,
        );
        for msg in &run.rounds {
            v.round_capturing(
                &magnetar_sumcheck::RoundMsg(msg.clone()),
                t,
                &format!("twist-read-{s}"),
            )?;
        }
        v.finish(&[ra_claim, run.val_claim])
            .map_err(|_| "twist: read terminal failed")?;
        for (i, &cv) in proof.read_component_claims[s].iter().enumerate() {
            t.absorb_field(&format!("twist-comp-{s}-{i}"), &[cv]);
        }
        // Verify the reconstruction of this Val claim.
        verify_recon(
            &proof.read_recons[s],
            &inst.m0,
            &run.r_joint,
            run.val_claim,
            a,
            ell,
            n,
            false,
            &format!("twist-recon-{s}"),
            t,
        )?;
        t.absorb_field(
            &format!("twist-recon-claims-{s}"),
            &[recon.wfa_claim, recon.inc_claim],
        );
    }
    // Write run.
    {
        let ra_claim: Gf128 = proof.write_component_claims.iter().copied().fold(ONE, mul);
        let mut v = SumcheckVerifier::new(
            joint_vars,
            inst.wv_claim,
            &pubf,
            terms,
            BindOrder::Increasing,
        );
        for msg in &proof.write_run.rounds {
            v.round_capturing(&magnetar_sumcheck::RoundMsg(msg.clone()), t, "twist-write")?;
        }
        v.finish(&[ra_claim, proof.write_run.val_claim])
            .map_err(|_| "twist: write terminal failed")?;
        for (i, &cv) in proof.write_component_claims.iter().enumerate() {
            t.absorb_field(&format!("twist-wcomp-{i}"), &[cv]);
        }
        verify_recon(
            &proof.write_recon,
            &inst.m0,
            &proof.write_run.r_joint,
            proof.write_run.val_claim,
            a,
            ell,
            n,
            true,
            "twist-wrecon",
            t,
        )?;
        t.absorb_field(
            "twist-wrecon-claims",
            &[proof.write_recon.wfa_claim, proof.write_recon.inc_claim],
        );
    }
    // Sum-to-one.
    let mut st1_pcs: Vec<Vec<PcsPoint>> = Vec::new();
    for (port, rounds_per) in proof.st1.iter().enumerate() {
        let mut pcs_per = Vec::new();
        for (i, rounds) in rounds_per.iter().enumerate() {
            let rho = t.challenge_field_vec(&format!("twist-st1-{port}-{i}-rho"), n);
            let kj_vars = ad + n;
            let pubf2: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
                vec![Box::new(EqProdFactor::new(
                    vec![(rho.clone(), (ad..ad + n).collect())],
                    kj_vars,
                ))];
            let terms2 = vec![Term {
                coeff: ONE,
                dense: vec![0],
                public: vec![0],
            }];
            let mut v = SumcheckVerifier::new(kj_vars, ONE, &pubf2, terms2, BindOrder::Increasing);
            let mut chs = Vec::with_capacity(kj_vars);
            for msg in rounds {
                chs.push(v.round_capturing(
                    &magnetar_sumcheck::RoundMsg(msg.clone()),
                    t,
                    &format!("twist-st1-{port}-{i}"),
                )?);
            }
            v.finish(&[proof.st1_claims[port][i]])
                .map_err(|_| "twist: st1 terminal failed")?;
            pcs_per.push(PcsPoint::new(chs, proof.st1_claims[port][i]));
        }
        st1_pcs.push(pcs_per);
    }
    for (port, claims) in proof.st1_claims.iter().enumerate() {
        for (i, v) in claims.iter().enumerate() {
            t.absorb_field(&format!("twist-st1-claim-{port}-{i}"), &[*v]);
        }
    }
    // Δ*.
    let mut onehot_claims = Vec::new();
    for (port, per) in proof.onehot.iter().enumerate() {
        let mut per_claims = Vec::new();
        for (i, oh) in per.iter().enumerate() {
            let cyc_pt = crate::shout::verify_onehot(oh, t, &format!("twist-onehot-{port}-{i}"))?;
            let mut pt_r = oh.r.clone();
            pt_r.extend_from_slice(&cyc_pt);
            let mut pt_rp = oh.r_prime.clone();
            pt_rp.extend_from_slice(&cyc_pt);
            per_claims.push([
                PcsPoint::new(pt_r, oh.v_at_r),
                PcsPoint::new(pt_rp, oh.v_at_rp),
            ]);
        }
        onehot_claims.push(per_claims);
    }
    // Assemble the claims (mirroring the prover's layout).
    let mk_comp_pt = |r_joint: &[Gf128], i: usize, ad: usize, a: usize, n: usize| {
        let mut pt = r_joint[i * ad..(i + 1) * ad].to_vec();
        pt.extend_from_slice(&r_joint[a + ell..a + ell + n]);
        pt
    };
    let read_addr: Vec<Vec<PcsPoint>> = (0..p)
        .map(|s| {
            let run = &proof.read_runs[s];
            (0..d)
                .map(|i| {
                    PcsPoint::new(
                        mk_comp_pt(&run.r_joint, i, ad, a, n),
                        proof.read_component_claims[s][i],
                    )
                })
                .collect()
        })
        .collect();
    let write_addr: Vec<PcsPoint> = (0..d)
        .map(|i| {
            PcsPoint::new(
                mk_comp_pt(&proof.write_run.r_joint, i, ad, a, n),
                proof.write_component_claims[i],
            )
        })
        .collect();
    let recon_wfa: Vec<Vec<PcsPoint>> = proof
        .read_recons
        .iter()
        .chain(std::iter::once(&proof.write_recon))
        .zip(
            proof
                .read_runs
                .iter()
                .chain(std::iter::once(&proof.write_run)),
        )
        .map(|(recon, run)| {
            (0..d)
                .map(|i| {
                    let mut pt = run.r_joint[i * ad..(i + 1) * ad].to_vec();
                    pt.extend_from_slice(&recon.r_cycle);
                    PcsPoint::new(pt, recon.wfa_component_claims[i])
                })
                .collect()
        })
        .collect();
    let inc: Vec<PcsPoint> = proof
        .read_recons
        .iter()
        .chain(std::iter::once(&proof.write_recon))
        .zip(
            proof
                .read_runs
                .iter()
                .chain(std::iter::once(&proof.write_run)),
        )
        .map(|(recon, run)| {
            let mut pt = run.r_joint[a..a + ell].to_vec();
            pt.extend_from_slice(&recon.r_cycle);
            PcsPoint::new(pt, recon.inc_claim)
        })
        .collect();
    Ok(TwistPcsClaims {
        read_addr,
        write_addr,
        recon_wfa,
        inc,
        st1: st1_pcs,
        onehot: onehot_claims,
    })
}

/// Verify one reconstruction sum-check: the claim is
/// `val_claim + M̂0(point) = Σ_{j'} LT(j', r_c)·ŵfa·Înc`.
fn verify_recon(
    recon: &ReconRun,
    m0: &[Gf128],
    r_joint: &[Gf128],
    val_claim: Gf128,
    a: usize,
    ell: usize,
    n: usize,
    inclusive: bool,
    label: &str,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    let r_k: Vec<Gf128> = r_joint[..a].to_vec();
    let r_v: Vec<Gf128> = r_joint[a..a + ell].to_vec();
    let r_c: Vec<Gf128> = r_joint[a + ell..].to_vec();
    let m0_at = magnetar_field::eval_mle(m0, &[r_k, r_v].concat());
    // The LT column at r_c via the O(T) suffix sums (the verifier's
    // structured evaluation — no 2^{2n} less-than table).
    let eq_j = magnetar_field::EqBits::build(&r_c).values;
    let lt_col = suffix_sums(&eq_j, inclusive);
    let claim = val_claim + m0_at;
    let pubf: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(TableFactor::new(lt_col))];
    let terms = vec![Term {
        coeff: ONE,
        dense: vec![0, 1],
        public: vec![0],
    }];
    let mut v = SumcheckVerifier::new(n, claim, &pubf, terms, BindOrder::Increasing);
    for msg in &recon.rounds {
        v.round_capturing(&magnetar_sumcheck::RoundMsg(msg.clone()), t, label)?;
    }
    v.finish(&[recon.wfa_claim, recon.inc_claim])
        .map_err(|_| "twist: recon terminal failed")?;
    // Address virtualization: the full-address terminal must equal the
    // product of the per-component claims at the recon's point.
    let product = recon.wfa_component_claims.iter().copied().fold(ONE, mul);
    if product != recon.wfa_claim {
        return Err("twist: recon address virtualization failed");
    }
    for (i, &cv) in recon.wfa_component_claims.iter().enumerate() {
        t.absorb_field(&format!("{label}-wfacomp-{i}"), &[cv]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lt_tables() {
        let n = 3;
        let strict = lt_table(n, false);
        let incl = lt_table(n, true);
        for j in 0..(1usize << n) {
            for jp in 0..(1usize << n) {
                assert_eq!(strict[jp + j * 8], if jp < j { ONE } else { ZERO });
                assert_eq!(incl[jp + j * 8], if jp <= j { ONE } else { ZERO });
            }
        }
    }

    /// The linear-in-T port prover emits **byte-identical** round
    /// messages, challenges, and terminal claims to the dense engine's
    /// for the same statement — the cross-validation that gates the
    /// linear prover's use in the protocol.
    #[test]
    fn linear_port_prover_matches_dense() {
        use rand::RngExt;
        let mut rng = rand::rng();
        for &(a, d, ell, n, p) in &[
            (4usize, 2usize, 3usize, 3usize, 2usize),
            (6, 2, 4, 4, 1),
            (6, 3, 2, 3, 2),
        ] {
            let ad = a / d;
            let cycles = 1usize << n;
            let cells = 1usize << a;
            let bits = 1usize << ell;
            // The write port's components and increments.
            let mut wcomps = Vec::new();
            for _ in 0..d {
                let mut b = vec![false; (1usize << ad) * cycles];
                for j in 0..cycles {
                    b[rng.random_range(0..(1usize << ad)) + j * (1usize << ad)] = true;
                }
                wcomps.push(AddressComponent::from_bits(ad, n, &b));
            }
            let inc: Vec<Gf128> = (0..bits * cycles)
                .map(|_| Gf128(rng.random::<bool>() as u128))
                .collect();
            let m0: Vec<Gf128> = (0..cells * bits)
                .map(|_| Gf128(rng.random::<bool>() as u128))
                .collect();
            let r_val: Vec<Gf128> = (0..ell).map(|_| Gf128(rng.random::<u128>())).collect();
            let r_cycle: Vec<Gf128> = (0..n).map(|_| Gf128(rng.random::<u128>())).collect();
            let shared = crate::linear::TwistShared::new(
                &TwistInstance {
                    a,
                    ell,
                    n,
                    p,
                    m0: m0.clone(),
                    rv_claims: vec![],
                    wv_claim: ZERO,
                    r_val: r_val.clone(),
                    r_cycle: r_cycle.clone(),
                },
                crate::linear::addr_bits(&wcomps),
                &inc,
            );
            let eq_val = magnetar_field::EqBits::build(&r_val).values;
            let eq_cyc = magnetar_field::EqBits::build(&r_cycle).values;
            let joint_vars = a + ell + n;
            let mut pub_table = vec![ZERO; 1usize << joint_vars];
            for idx in 0..pub_table.len() {
                let o = (idx >> a) & ((1usize << ell) - 1);
                let j = idx >> (a + ell);
                pub_table[idx] = mul(eq_val[o], eq_cyc[j]);
            }
            let ra_joint = full_addr_joint(&wcomps, a, ell, n);
            let val_strict = materialize_val(a, ell, n, &m0, &wcomps, &inc, false);
            let val_inc = materialize_val(a, ell, n, &m0, &wcomps, &inc, true);
            for &(inclusive, val_table) in &[(false, &val_strict), (true, &val_inc)] {
                // The honest claim.
                let mut claim = ZERO;
                for idx in 0..(1usize << joint_vars) {
                    claim += mul(pub_table[idx], mul(ra_joint[idx], val_table[idx]));
                }
                // Dense.
                let mut t1 = Transcript::new("lin-x", &[b"dense"]);
                let dense = run_port_sumcheck(
                    &wcomps, val_table, claim, joint_vars, &pub_table, a, ell, n, "port", &mut t1,
                );
                // Linear.
                let mut t2 = Transcript::new("lin-x", &[b"dense"]);
                let lin = crate::linear::prove_port_linear(
                    &TwistInstance {
                        a,
                        ell,
                        n,
                        p,
                        m0: m0.clone(),
                        rv_claims: vec![],
                        wv_claim: ZERO,
                        r_val: r_val.clone(),
                        r_cycle: r_cycle.clone(),
                    },
                    &shared,
                    &inc,
                    &wcomps,
                    claim,
                    inclusive,
                    "port",
                    &mut t2,
                );
                assert_eq!(
                    dense.rounds, lin.rounds,
                    "rounds a={a} d={d} inc={inclusive}"
                );
                assert_eq!(dense.r_joint, lin.r_joint, "point");
                assert_eq!(dense.ra_claim, lin.ra_claim, "ra claim");
                assert_eq!(dense.val_claim, lin.val_claim, "val claim");
            }
        }
    }

    #[test]
    fn val_materialization_strict_and_inclusive() {
        use rand::RngExt;
        let mut rng = rand::rng();
        let (a, d, ell, n) = (4usize, 2, 2, 3);
        let ad = a / d;
        let cycles = 1usize << n;
        let cells = 1usize << a;
        let bits = 1usize << ell;
        let mut wcomps = Vec::new();
        for _ in 0..d {
            let mut b = vec![false; (1usize << ad) * cycles];
            for j in 0..cycles {
                b[rng.random_range(0..(1usize << ad)) + j * (1usize << ad)] = true;
            }
            wcomps.push(AddressComponent::from_bits(ad, n, &b));
        }
        let inc: Vec<Gf128> = (0..bits * cycles)
            .map(|_| Gf128(rng.random::<bool>() as u128))
            .collect();
        let m0: Vec<Gf128> = (0..cells * bits)
            .map(|_| Gf128(rng.random::<bool>() as u128))
            .collect();
        let vs = materialize_val(a, ell, n, &m0, &wcomps, &inc, false);
        let vi = materialize_val(a, ell, n, &m0, &wcomps, &inc, true);
        for k in 0..cells {
            for o in 0..bits {
                for j in 0..cycles {
                    let mut want_s = m0[k | (o << a)];
                    let mut want_i = m0[k | (o << a)];
                    for jp in 0..cycles {
                        let mut hot = true;
                        for (i, comp) in wcomps.iter().enumerate() {
                            let slice = (k >> (i * ad)) & ((1usize << ad) - 1);
                            if comp.table[comp.index_with(slice, jp)] != ONE {
                                hot = false;
                                break;
                            }
                        }
                        if hot {
                            let bit = inc[o | (jp << ell)];
                            if jp < j {
                                want_s += bit;
                            }
                            if jp <= j {
                                want_i += bit;
                            }
                        }
                    }
                    let idx = k + o * cells + j * cells * bits;
                    assert_eq!(vs[idx], want_s, "strict k={k} o={o} j={j}");
                    assert_eq!(vi[idx], want_i, "incl k={k} o={o} j={j}");
                }
            }
        }
        let _ = 0;
    }
}
