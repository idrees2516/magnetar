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
use crate::{AddressComponent, OneHotProof};
use magnetar_field::{mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_sumcheck::{
    engine::{SumcheckProver, SumcheckVerifier},
    public::TableFactor,
    BindOrder, SumcheckSpec, Term,
};

/// A BinaryTwist instance.
#[derive(Clone, Debug)]
pub struct TwistInstance {
    pub a: usize,
    pub ell: usize,
    pub n: usize,
    pub p: usize,
    /// Initial memory `M0: {0,1}^{a+ell} → B` (address low bits, bit index
    /// high). The prover always evaluates the true table; the verifier
    /// uses it only when [`Self::m0_public`] — the tree-IVC chunks carry a
    /// **committed** entry image (the `ram-init` Mosaic component) and the
    /// verifier instead takes the reconstruction's `M̂0` values as
    /// transmitted claims, discharged against that component.
    pub m0: Vec<Gf128>,
    /// Whether the initial memory is verifier-known (public). When
    /// `false`, every reconstruction run transmits its `M̂0(point)` value
    /// (absorbed into the transcript) and leaves a PCS claim at its
    /// `(r_k ‖ r_v)` point.
    pub m0_public: bool,
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
    /// The transmitted `M̂0(r_k, r_v)` value — meaningful (and bound into
    /// the transcript) only when the instance's memory is not public.
    pub m0_at: Gf128,
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

/// Run one value sum-check (read or write shape) capturing the point.
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

/// Run one reconstruction sum-check at a joint point. When the memory is
/// not public the `M̂0(r_k, r_v)` value is absorbed before the rounds and
/// returned as a transmitted claim (discharged against the committed
/// entry-image component by the zkvm layer).
#[allow(clippy::too_many_arguments)]
fn run_reconstruction(
    wfa_comps: &[AddressComponent],
    inc: &[Gf128],
    val_claim: Gf128,
    r_joint: &[Gf128],
    a: usize,
    ell: usize,
    n: usize,
    m0: &[Gf128],
    m0_public: bool,
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
    if !m0_public {
        // Bind the claimed initial-memory value before any challenge that
        // depends on it (it enters the reconstruction's round checks).
        t.absorb_field(&format!("{label}-m0"), &[m0_at]);
    }
    // LT column at r_c.
    let lt = lt_table(n, inclusive);
    let eq_j = magnetar_field::EqBits::build(&r_c).values;
    let mut lt_col = vec![ZERO; 1usize << n];
    for jp in 0..(1usize << n) {
        let mut acc = ZERO;
        for j in 0..(1usize << n) {
            acc += mul(eq_j[j], lt[jp + j * (1usize << n)]);
        }
        lt_col[jp] = acc;
    }
    // wfa row at r_k; inc row at r_v.
    let eq_k = magnetar_field::EqBits::build(&r_k).values;
    let mut wfa_row = vec![ZERO; 1usize << n];
    for jp in 0..(1usize << n) {
        let mut acc = ZERO;
        for k in 0..(1usize << a) {
            let mut prod = eq_k[k];
            for (i, comp) in wfa_comps.iter().enumerate() {
                let slice = (k >> (i * ad)) & ((1usize << ad) - 1);
                prod = mul(prod, comp.table[comp.index_with(slice, jp)]);
            }
            acc += prod;
        }
        wfa_row[jp] = acc;
    }
    let eq_v = magnetar_field::EqBits::build(&r_v).values;
    let mut inc_row = vec![ZERO; 1usize << n];
    for jp in 0..(1usize << n) {
        let mut acc = ZERO;
        for o in 0..(1usize << ell) {
            acc += mul(eq_v[o], inc[o | (jp << ell)]);
        }
        inc_row[jp] = acc;
    }
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
            comp.eval_at(&slice, &r_c)
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
        m0_at,
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
    let joint_vars = a + ell + n;
    let d = wit.d();
    // The public factor over the joint space: eq(r_val, o)·eq(r_cycle, j).
    let eq_val = magnetar_field::EqBits::build(&inst.r_val).values;
    let eq_cyc = magnetar_field::EqBits::build(&inst.r_cycle).values;
    let mut pub_table = vec![ZERO; 1usize << joint_vars];
    for idx in 0..pub_table.len() {
        let o = (idx >> a) & ((1usize << ell) - 1);
        let j = idx >> (a + ell);
        pub_table[idx] = mul(eq_val[o], eq_cyc[j]);
    }
    // Val (strict) for reads; Val+Inc (inclusive) for the write.
    let val_strict = materialize_val(a, ell, n, &inst.m0, &wit.ports[p], &wit.inc, false);
    let val_inc = materialize_val(a, ell, n, &inst.m0, &wit.ports[p], &wit.inc, true);
    // Read sum-checks.
    let mut read_runs = Vec::with_capacity(p);
    let mut read_recons = Vec::with_capacity(p);
    let mut read_component_claims = Vec::with_capacity(p);
    for s in 0..p {
        let run = run_port_sumcheck(
            &wit.ports[s],
            &val_strict,
            inst.rv_claims[s],
            joint_vars,
            &pub_table,
            a,
            ell,
            n,
            &format!("twist-read-{s}"),
            t,
        );
        // Component claims (address virtualization) at the run's point.
        let ad = a / d;
        let comps = &wit.ports[s];
        let comp_claims: Vec<Gf128> = comps
            .iter()
            .enumerate()
            .map(|(i, comp)| {
                let slice: Vec<Gf128> = (0..ad).map(|b| run.r_joint[i * ad + b]).collect();
                let r_j: Vec<Gf128> = run.r_joint[a + ell..].to_vec();
                comp.eval_at(&slice, &r_j)
            })
            .collect();
        for (i, &v) in comp_claims.iter().enumerate() {
            t.absorb_field(&format!("twist-comp-{s}-{i}"), &[v]);
        }
        let recon = run_reconstruction(
            &wit.ports[p],
            &wit.inc,
            run.val_claim,
            &run.r_joint,
            a,
            ell,
            n,
            &inst.m0,
            inst.m0_public,
            false,
            &format!("twist-recon-{s}"),
            t,
        );
        t.absorb_field(
            &format!("twist-recon-claims-{s}"),
            &[recon.wfa_claim, recon.inc_claim],
        );
        read_component_claims.push(comp_claims);
        read_recons.push(recon);
        read_runs.push(run);
    }
    // Write sum-check (Val + Inc).
    let write_run = run_port_sumcheck(
        &wit.ports[p],
        &val_inc,
        inst.wv_claim,
        joint_vars,
        &pub_table,
        a,
        ell,
        n,
        "twist-write",
        t,
    );
    let ad = a / d;
    let wcomps = &wit.ports[p];
    let write_component_claims: Vec<Gf128> = wcomps
        .iter()
        .enumerate()
        .map(|(i, comp)| {
            let slice: Vec<Gf128> = (0..ad).map(|b| write_run.r_joint[i * ad + b]).collect();
            let r_j: Vec<Gf128> = write_run.r_joint[a + ell..].to_vec();
            comp.eval_at(&slice, &r_j)
        })
        .collect();
    for (i, &v) in write_component_claims.iter().enumerate() {
        t.absorb_field(&format!("twist-wcomp-{i}"), &[v]);
    }
    let write_recon = run_reconstruction(
        wcomps,
        &wit.inc,
        write_run.val_claim,
        &write_run.r_joint,
        a,
        ell,
        n,
        &inst.m0,
        inst.m0_public,
        true,
        "twist-wrecon",
        t,
    );
    t.absorb_field(
        "twist-wrecon-claims",
        &[write_recon.wfa_claim, write_recon.inc_claim],
    );
    // Sum-to-one per (port, component).
    let mut st1 = Vec::new();
    let mut st1_claims = Vec::new();
    let mut st1_pcs: Vec<Vec<PcsPoint>> = Vec::new();
    for (port, comps) in wit.ports.iter().enumerate() {
        let mut rounds_per = Vec::new();
        let mut claims_per = Vec::new();
        let mut pcs_per = Vec::new();
        for (i, comp) in comps.iter().enumerate() {
            let rho = t.challenge_field_vec(&format!("twist-st1-{port}-{i}-rho"), n);
            let eq_rho = magnetar_field::EqBits::build(&rho).values;
            let kj_vars = ad + n;
            let mut pubf = vec![ZERO; 1usize << kj_vars];
            for idx in 0..pubf.len() {
                let j = idx >> ad;
                pubf[idx] = eq_rho[j];
            }
            let spec = SumcheckSpec {
                num_vars: kj_vars,
                dense: vec![comp.table.clone()],
                public: vec![Box::new(TableFactor::new(pubf))],
                terms: vec![Term {
                    coeff: ONE,
                    dense: vec![0],
                    public: vec![0],
                }],
                claim: ONE,
            };
            let mut pr = SumcheckProver::new(&spec, BindOrder::Increasing);
            let mut rounds = Vec::new();
            for _ in 0..kj_vars {
                let msg = pr.message();
                rounds.push(msg.0.clone());
                pr.absorb_and_fold(t, &format!("twist-st1-{port}-{i}"));
            }
            let st1_pt = pr.challenges();
            let (cclaims, _) = pr.finish();
            rounds_per.push(rounds);
            claims_per.push(cclaims[0]);
            pcs_per.push(PcsPoint::new(st1_pt, cclaims[0]));
        }
        st1.push(rounds_per);
        st1_claims.push(claims_per);
        st1_pcs.push(pcs_per);
    }
    for (port, claims) in st1_claims.iter().enumerate() {
        for (i, v) in claims.iter().enumerate() {
            t.absorb_field(&format!("twist-st1-claim-{port}-{i}"), &[*v]);
        }
    }
    // Δ* per (port, component).
    let mut onehot = Vec::new();
    let mut onehot_pts = Vec::new();
    for (port, comps) in wit.ports.iter().enumerate() {
        let mut per = Vec::new();
        let mut per_pts = Vec::new();
        for (i, comp) in comps.iter().enumerate() {
            let (oh, cyc_pt) =
                crate::shout::prove_onehot(comp, t, &format!("twist-onehot-{port}-{i}"));
            per.push(oh);
            per_pts.push(cyc_pt);
        }
        onehot.push(per);
        onehot_pts.push(per_pts);
    }
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
    // The transmitted initial-memory claims (only when not public): one
    // per reconstruction run, at its `(r_k ‖ r_v)` point.
    let m0: Vec<PcsPoint> = if inst.m0_public {
        Vec::new()
    } else {
        all_recons
            .iter()
            .zip(all_runs.iter())
            .map(|(recon, run)| PcsPoint::new(run.r_joint[..a + ell].to_vec(), recon.m0_at))
            .collect()
    };
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
            m0,
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
    let eq_val = magnetar_field::EqBits::build(&inst.r_val).values;
    let eq_cyc = magnetar_field::EqBits::build(&inst.r_cycle).values;
    let pub_table: Vec<Gf128> = {
        let mut pt = vec![ZERO; 1usize << joint_vars];
        for idx in 0..pt.len() {
            let o = (idx >> a) & ((1usize << ell) - 1);
            let j = idx >> (a + ell);
            pt[idx] = mul(eq_val[o], eq_cyc[j]);
        }
        pt
    };
    let terms = vec![Term {
        coeff: ONE,
        dense: vec![0, 1],
        public: vec![0],
    }];
    let pubf: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(TableFactor::new(pub_table.clone()))];
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
            inst.m0_public,
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
            inst.m0_public,
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
            let eq_rho = magnetar_field::EqBits::build(&rho).values;
            let kj_vars = ad + n;
            let mut pubf2 = vec![ZERO; 1usize << kj_vars];
            for idx in 0..pubf2.len() {
                let j = idx >> ad;
                pubf2[idx] = eq_rho[j];
            }
            let pubf2: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
                vec![Box::new(TableFactor::new(pubf2))];
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
    // The transmitted initial-memory claims — only when not public.
    let m0: Vec<PcsPoint> = if inst.m0_public {
        Vec::new()
    } else {
        proof
            .read_recons
            .iter()
            .chain(std::iter::once(&proof.write_recon))
            .zip(
                proof
                    .read_runs
                    .iter()
                    .chain(std::iter::once(&proof.write_run)),
            )
            .map(|(recon, run)| PcsPoint::new(run.r_joint[..a + ell].to_vec(), recon.m0_at))
            .collect()
    };
    Ok(TwistPcsClaims {
        read_addr,
        write_addr,
        recon_wfa,
        inc,
        m0,
        st1: st1_pcs,
        onehot: onehot_claims,
    })
}

/// Verify one reconstruction sum-check: the claim is
/// `val_claim + M̂0(point) = Σ_{j'} LT(j', r_c)·ŵfa·Înc`. The `M̂0`
/// value is recomputed when the memory is public, otherwise taken from
/// the transmitted `recon.m0_at` (absorbed at the same position as the
/// prover).
fn verify_recon(
    recon: &ReconRun,
    m0: &[Gf128],
    m0_public: bool,
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
    let m0_at = if m0_public {
        magnetar_field::eval_mle(m0, &[r_k, r_v].concat())
    } else {
        t.absorb_field(&format!("{label}-m0"), &[recon.m0_at]);
        recon.m0_at
    };
    let lt = lt_table(n, inclusive);
    let eq_j = magnetar_field::EqBits::build(&r_c).values;
    let mut lt_col = vec![ZERO; 1usize << n];
    for jp in 0..(1usize << n) {
        let mut acc = ZERO;
        for j in 0..(1usize << n) {
            acc += mul(eq_j[j], lt[jp + j * (1usize << n)]);
        }
        lt_col[jp] = acc;
    }
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

    #[test]
    fn val_materialization_strict_and_inclusive() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let (a, d, ell, n) = (4usize, 2, 2, 3);
        let ad = a / d;
        let cycles = 1usize << n;
        let cells = 1usize << a;
        let bits = 1usize << ell;
        let mut wcomps = Vec::new();
        for _ in 0..d {
            let mut b = vec![false; (1usize << ad) * cycles];
            for j in 0..cycles {
                b[rng.gen_range(0..(1usize << ad)) + j * (1usize << ad)] = true;
            }
            wcomps.push(AddressComponent::from_bits(ad, n, &b));
        }
        let inc: Vec<Gf128> = (0..bits * cycles)
            .map(|_| Gf128(rng.gen::<bool>() as u128))
            .collect();
        let m0: Vec<Gf128> = (0..cells * bits)
            .map(|_| Gf128(rng.gen::<bool>() as u128))
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
