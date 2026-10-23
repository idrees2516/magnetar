//! The virtual-claim discharge helpers (paper §5.5, steps 3–6): the
//! Ψ-sum-checks that bind memory-address claims to the one-hot components,
//! the **joint-space BinaryShout** for the bytecode (one Φ sum-check over
//! the (address, value-bit, cycle) space checking the fetched bus, the
//! register-index slices and the folded non-emptiness at once), and the
//! selector checks that authenticate the routers' one-hot selector claims
//! against the bytecode ROM.
//!
//! All three reductions consume the **committed one-hot address
//! components** as dense factors and public ROM-derived tables as
//! verifier-evaluable factors, and output per-component evaluation claims
//! for the PCS — exactly the paper's "each of these arguments discharges
//! its claim by reducing it, through further sum-check arguments, to
//! evaluations of polynomials that are committed".
//!
//! Summand structure: every public factor is a *multilinear* table over a
//! subset of the joint variables (replication over the idle variables
//! leaves the MLE unchanged, because each eq-fold over an idle variable
//! sums to one). The degree-2+ summands arise as *products* of factors in
//! the term structure, which the engine's round interpolation handles.

use crate::claims::{PcsPoint, SelectorPcsClaims, ShoutPcsClaims};
use crate::shout::{prove_onehot, verify_onehot};
use crate::{AddressComponent, OneHotProof};
use magnetar_field::{eval_mle, mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_sumcheck::{
    engine::{SumcheckProver, SumcheckVerifier},
    public::PublicFactor,
    BindOrder, RoundMsg, SumcheckSpec, Term,
};

/// Pad one address component's table into the joint `(k, mid, j)` space
/// (address part low, `mid_vars` middle variables, cycle part high).
fn pad_component(comp: &AddressComponent, i: usize, d: usize, mid_vars: usize) -> Vec<Gf128> {
    let ad = comp.addr_vars; // a/d
    let n = comp.cycle_vars;
    let a = ad * d;
    let joint_vars = a + mid_vars + n;
    let mut out = vec![ZERO; 1usize << joint_vars];
    for idx in 0..out.len() {
        let k = idx & ((1usize << a) - 1);
        let j = idx >> (a + mid_vars);
        let slice = (k >> (i * ad)) & ((1usize << ad) - 1);
        out[idx] = comp.table[comp.index_with(slice, j)];
    }
    out
}

/// Split a joint `(k, mid, j)` point into the component's local coords:
/// `(k-slice_i, r_j)`.
fn component_point(r_joint: &[Gf128], i: usize, d: usize, mid_vars: usize, n: usize) -> Vec<Gf128> {
    let a = r_joint.len() - mid_vars - n;
    let ad = a / d;
    let mut pt = Vec::with_capacity(ad + n);
    pt.extend_from_slice(&r_joint[i * ad..(i + 1) * ad]);
    pt.extend_from_slice(&r_joint[a + mid_vars..]);
    pt
}

/// A public multilinear over `active` joint bit positions, given as a dense
/// source table (in the table's own variable order). Materialization
/// replicates over the idle variables; evaluation takes the active slice.
#[derive(Clone, Debug)]
pub struct JointTableFactor {
    pub table: Vec<Gf128>,
    /// Joint bit position of the table's variable `v`, in table order.
    pub active: Vec<usize>,
    pub joint_vars: usize,
}

impl JointTableFactor {
    pub fn new(table: Vec<Gf128>, active: Vec<usize>, joint_vars: usize) -> Self {
        debug_assert_eq!(table.len(), 1usize << active.len());
        JointTableFactor {
            table,
            active,
            joint_vars,
        }
    }
}

impl PublicFactor for JointTableFactor {
    fn num_vars(&self) -> usize {
        self.joint_vars
    }
    fn materialize(&self) -> Vec<Gf128> {
        let mut out = vec![ZERO; 1usize << self.joint_vars];
        for idx in 0..out.len() {
            let mut ti = 0usize;
            for (tv, &av) in self.active.iter().enumerate() {
                if (idx >> av) & 1 == 1 {
                    ti |= 1usize << tv;
                }
            }
            out[idx] = self.table[ti];
        }
        out
    }
    fn eval_at(&self, point: &[Gf128]) -> Gf128 {
        let sub: Vec<Gf128> = self.active.iter().map(|&av| point[av]).collect();
        eval_mle(&self.table, &sub)
    }
}

// ---------------------------------------------------------------------------
// The Ψ-sum-check (address-claim binding).
// ---------------------------------------------------------------------------

/// A Ψ-sum-check proof: `c = Σ_{k,j} eq(r_cycle,j)·Ψ(k)·Π_i ra_i(k^{(i)},j)`
/// with `Ψ(k) = Σ_t w_t·k_t` (public weights), reduced to per-component
/// claims.
#[derive(Clone, Debug)]
pub struct PsiProof {
    pub rounds: Vec<Vec<Gf128>>,
    pub component_claims: Vec<PcsPoint>,
}

/// The Ψ weights: `Ψ(k) = Σ_t w_t·k_t` as a table over the address cube.
fn psi_table(weights: &[Gf128], a: usize) -> Vec<Gf128> {
    let mut t = vec![ZERO; 1usize << a];
    for (k, v) in t.iter_mut().enumerate() {
        let mut acc = ZERO;
        for (tbit, &w) in weights.iter().enumerate() {
            if (k >> tbit) & 1 == 1 {
                acc += w;
            }
        }
        *v = acc;
    }
    t
}

/// Prove the Ψ-claim for a set of address components; returns the honest
/// claim value `c` (the caller forwards it to the consuming argument) and
/// the proof.
pub fn prove_psi_claim(
    components: &[AddressComponent],
    r_cycle: &[Gf128],
    weights: &[Gf128],
    label: &str,
    t: &mut Transcript,
) -> (Gf128, PsiProof) {
    let d = components.len();
    let a = components[0].addr_vars * d;
    let n = components[0].cycle_vars;
    debug_assert_eq!(r_cycle.len(), n);
    debug_assert_eq!(weights.len(), a);
    // The honest claim value.
    let eq_j = magnetar_field::EqBits::build(r_cycle).values;
    let psi = psi_table(weights, a);
    let mut claim = ZERO;
    for idx in 0..(1usize << (a + n)) {
        let k = idx & ((1usize << a) - 1);
        let j = idx >> a;
        let mut prod = mul(eq_j[j], psi[k]);
        for (i, comp) in components.iter().enumerate() {
            let ad = comp.addr_vars;
            let slice = (k >> (i * ad)) & ((1usize << ad) - 1);
            prod = mul(prod, comp.table[comp.index_with(slice, j)]);
        }
        claim += prod;
    }
    t.absorb_field(&format!("{label}-claim"), &[claim]);
    // Sum-check over the joint (k, j) space: Ψ(k)·eq(r_cycle, j)·Π comps.
    let dense: Vec<Vec<Gf128>> = (0..d)
        .map(|i| pad_component(&components[i], i, d, 0))
        .collect();
    let public: Vec<Box<dyn PublicFactor + Send + Sync>> = vec![
        Box::new(JointTableFactor::new(
            psi_table(weights, a),
            (0..a).collect(),
            a + n,
        )),
        Box::new(JointTableFactor::new(
            magnetar_field::EqBits::build(r_cycle).values,
            (a..a + n).collect(),
            a + n,
        )),
    ];
    let spec = SumcheckSpec {
        num_vars: a + n,
        dense,
        public,
        terms: vec![Term {
            coeff: ONE,
            dense: (0..d).collect(),
            public: vec![0, 1],
        }],
        claim,
    };
    let mut prover = SumcheckProver::new_owned(spec, BindOrder::Increasing);
    let mut rounds = Vec::with_capacity(a + n);
    for _ in 0..(a + n) {
        let msg = prover.message_cached();
        rounds.push(msg.0.clone());
        prover.absorb_and_fold(t, label);
    }
    let r_joint = prover.challenges();
    let (terminals, _) = prover.finish();
    let component_claims: Vec<PcsPoint> = (0..d)
        .map(|i| {
            let pt = component_point(&r_joint, i, d, 0, n);
            PcsPoint::new(pt, terminals[i])
        })
        .collect();
    for c in &component_claims {
        t.absorb_field(&format!("{label}-comp"), &[c.value]);
    }
    (
        claim,
        PsiProof {
            rounds,
            component_claims,
        },
    )
}

/// Verify a Ψ-sum-check; returns the accepted per-component claims.
pub fn verify_psi_claim(
    d: usize,
    addr_vars: usize,
    cycle_vars: usize,
    r_cycle: &[Gf128],
    weights: &[Gf128],
    claim: Gf128,
    proof: &PsiProof,
    label: &str,
    t: &mut Transcript,
) -> Result<Vec<PcsPoint>, &'static str> {
    let a = addr_vars * d;
    let n = cycle_vars;
    if proof.rounds.len() != a + n || proof.component_claims.len() != d {
        return Err("psi: shape mismatch");
    }
    t.absorb_field(&format!("{label}-claim"), &[claim]);
    let public: Vec<Box<dyn PublicFactor + Send + Sync>> = vec![
        Box::new(JointTableFactor::new(
            psi_table(weights, a),
            (0..a).collect(),
            a + n,
        )),
        Box::new(JointTableFactor::new(
            magnetar_field::EqBits::build(r_cycle).values,
            (a..a + n).collect(),
            a + n,
        )),
    ];
    let terms = vec![Term {
        coeff: ONE,
        dense: (0..d).collect(),
        public: vec![0, 1],
    }];
    let mut v = SumcheckVerifier::new(a + n, claim, &public, terms, BindOrder::Increasing);
    let mut r_joint = Vec::with_capacity(a + n);
    for msg in &proof.rounds {
        r_joint.push(v.round_capturing(&RoundMsg(msg.clone()), t, label)?);
    }
    let values: Vec<Gf128> = proof.component_claims.iter().map(|c| c.value).collect();
    v.finish(&values)
        .map_err(|_| "psi: terminal identity failed")?;
    let out = (0..d)
        .map(|i| {
            let pt = component_point(&r_joint, i, d, 0, n);
            PcsPoint::new(pt, proof.component_claims[i].value)
        })
        .collect();
    for c in &proof.component_claims {
        t.absorb_field(&format!("{label}-comp"), &[c.value]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The joint-space bytecode BinaryShout (the Φ check).
// ---------------------------------------------------------------------------

/// The bytecode Shout specification (all public).
#[derive(Clone, Debug)]
pub struct BusShoutSpec {
    /// The fetch address width (the ROM has `2^a` rows).
    pub a: usize,
    /// The bus width in bits (the value-bit variable count).
    pub ell: usize,
    /// Cycle variable count.
    pub n: usize,
    /// The ROM bus bits, indexed `(o low, k high)`.
    pub rom_bus: Vec<Gf128>,
    /// The claimed bus evaluation at `(r_val, r_cycle)` (VS's fetch claim).
    pub bus_claim: Gf128,
    /// The register-index slice claims: `(position, width, value)` — from
    /// the register Twist's Ψ-sum-checks.
    pub index_claims: Vec<(usize, usize, Gf128)>,
    /// The value point (VS's `r_local`).
    pub r_val: Vec<Gf128>,
    /// The cycle point (VS's `r_inst`).
    pub r_cycle: Vec<Gf128>,
    /// The micro-PC segment's local point (VS's `r_local` for the pc).
    pub pc_r_local: Vec<Gf128>,
    /// VS's claim on the micro-PC segment.
    pub pc_claim: Gf128,
}

/// The per-`o` weight table `w(o) = γ_bus + Σ_s γ_s·[o ∈ slice_s]`.
fn bus_w_table(spec: &BusShoutSpec, gamma_bus: Gf128, gammas: &[Gf128]) -> Vec<Gf128> {
    let mut w = vec![gamma_bus; 1usize << spec.ell];
    for ((pos, width, _), &g) in spec.index_claims.iter().zip(gammas.iter()) {
        for o in *pos..(*pos + *width) {
            w[o] += g;
        }
    }
    w
}

/// The bus value table `busbit(o, k)` over the `(o, k)` cube — indexed
/// `(o low, k high)` — as the prover/verifier both need it.
fn bus_rom_table(spec: &BusShoutSpec) -> Vec<Gf128> {
    spec.rom_bus.clone()
}

/// A bytecode Shout proof: the Φ sum-check, the pc Ψ-sum-check and the
/// Δ\* one-hotness proofs.
#[derive(Clone, Debug)]
pub struct BusShoutProof {
    pub phi_rounds: Vec<Vec<Gf128>>,
    /// The Φ sum-check's per-component terminal claims.
    pub phi_component_claims: Vec<PcsPoint>,
    /// The pc Ψ-sum-check.
    pub pc_psi: PsiProof,
    /// Δ\* per component.
    pub onehot: Vec<OneHotProof>,
}

/// Sample the Φ coefficients (α₀ non-zero via resampling, as in §4.4).
fn sample_bus_coeffs(t: &mut Transcript, n_slices: usize) -> (Gf128, Gf128, Vec<Gf128>) {
    let alpha0 = loop {
        let x = t.challenge_field("bus-alpha0");
        if x != ZERO {
            break x;
        }
    };
    let gamma_bus = t.challenge_field("bus-gamma");
    let gammas = t.challenge_field_vec("bus-slice", n_slices);
    (alpha0, gamma_bus, gammas)
}

/// The Ψ weights for the micro-PC claim: `w_t = eq(pc_r_local, t)` for the
/// address bit `t < a` (the pc segment's columns are the address bits;
/// bits at or beyond `a` are forced to zero by the claim equality — a
/// nonzero high bit shifts `v_pc` off the Ψ-bound combination).
pub fn pc_weights(pc_r_local: &[Gf128], a: usize) -> Vec<Gf128> {
    let lv = pc_r_local.len();
    assert!(
        a <= (1usize << lv),
        "the pc segment must cover all ROM address bits"
    );
    let eq = magnetar_field::EqBits::build(pc_r_local).values;
    eq[..a].to_vec()
}

/// Build the Φ sum-check's public factors and terms.
fn bus_phi_factors(
    spec: &BusShoutSpec,
    alpha0: Gf128,
    gamma_bus: Gf128,
    gammas: &[Gf128],
) -> (Vec<Box<dyn PublicFactor + Send + Sync>>, Vec<Term>) {
    let (a, ell, n) = (spec.a, spec.ell, spec.n);
    let jv = a + ell + n;
    // Factor 0: eq(r_val, o) over the o-block.
    let eq_v = JointTableFactor::new(
        magnetar_field::EqBits::build(&spec.r_val).values,
        (a..a + ell).collect(),
        jv,
    );
    // Factor 1: eq(r_cycle, j) over the j-block.
    let eq_c = JointTableFactor::new(
        magnetar_field::EqBits::build(&spec.r_cycle).values,
        (a + ell..a + ell + n).collect(),
        jv,
    );
    // Factor 2: the constant α₀ (an empty-table constant).
    let const_a0 = JointTableFactor::new(vec![alpha0], vec![], jv);
    // Factor 3: w(o) over the o-block.
    let w = JointTableFactor::new(
        bus_w_table(spec, gamma_bus, gammas),
        (a..a + ell).collect(),
        jv,
    );
    // Factor 4: busbit(o, k) — the table's vars (o, k) with the joint
    // positions [o-block] ++ [k-block].
    let mut bus_active = (a..a + ell).collect::<Vec<usize>>();
    bus_active.extend(0..a);
    let bus = JointTableFactor::new(bus_rom_table(spec), bus_active, jv);
    (
        vec![
            Box::new(eq_v),
            Box::new(eq_c),
            Box::new(const_a0),
            Box::new(w),
            Box::new(bus),
        ],
        // Terms: [comps·eq_v·eq_c·α₀] + [comps·eq_v·eq_c·w·bus].
        vec![
            Term {
                coeff: ONE,
                dense: Vec::new(), // filled by the caller
                public: vec![0, 1, 2],
            },
            Term {
                coeff: ONE,
                dense: Vec::new(),
                public: vec![0, 1, 3, 4],
            },
        ],
    )
}

/// Prove the bytecode Shout. `components` are the committed fetch-address
/// components (over `(k, j)`). Returns the proof and the PCS claims it
/// leaves open (symmetric with `verify_bus_shout`).
pub fn prove_bus_shout(
    spec: &BusShoutSpec,
    components: &[AddressComponent],
    t: &mut Transcript,
) -> (BusShoutProof, ShoutPcsClaims) {
    let d = components.len();
    let (a, ell, n) = (spec.a, spec.ell, spec.n);
    // Absorb the input claims (they precede the Φ coefficients).
    t.absorb_field("bus-in-bus", &[spec.bus_claim]);
    t.absorb_field("bus-in-pc", &[spec.pc_claim]);
    for (i, (_, _, c)) in spec.index_claims.iter().enumerate() {
        t.absorb_field(&format!("bus-in-idx-{i}"), &[*c]);
    }
    let (alpha0, gamma_bus, gammas) = sample_bus_coeffs(t, spec.index_claims.len());
    // The Φ claim: α₀·1 (the folded non-emptiness) + γ_bus·v_bus + Σ γ_s·c_s.
    let mut claim = alpha0 + mul(gamma_bus, spec.bus_claim);
    for ((_, _, c), &g) in spec.index_claims.iter().zip(gammas.iter()) {
        claim += mul(g, *c);
    }
    // The honest sum check (defensive): compute directly.
    let dense: Vec<Vec<Gf128>> = (0..d)
        .map(|i| pad_component(&components[i], i, d, ell))
        .collect();
    let (public, mut terms) = bus_phi_factors(spec, alpha0, gamma_bus, &gammas);
    let dense_idx: Vec<usize> = (0..d).collect();
    for tm in terms.iter_mut() {
        tm.dense = dense_idx.clone();
    }
    let spec_sc = SumcheckSpec {
        num_vars: a + ell + n,
        dense,
        public,
        terms,
        claim,
    };
    let mut prover = SumcheckProver::new_owned(spec_sc, BindOrder::Increasing);
    let mut phi_rounds = Vec::with_capacity(a + ell + n);
    for _ in 0..(a + ell + n) {
        let msg = prover.message_cached();
        phi_rounds.push(msg.0.clone());
        prover.absorb_and_fold(t, "bus-phi");
    }
    let r_joint = prover.challenges();
    let (terminals, _) = prover.finish();
    let phi_component_claims: Vec<PcsPoint> = (0..d)
        .map(|i| {
            let pt = component_point(&r_joint, i, d, ell, n);
            PcsPoint::new(pt, terminals[i])
        })
        .collect();
    for c in &phi_component_claims {
        t.absorb_field("bus-phi-comp", &[c.value]);
    }
    // The pc Ψ-sum-check (weights = eq(pc_r_local, ·) over the address bits).
    let weights = pc_weights(&spec.pc_r_local, a);
    let (_, pc_psi) = prove_psi_claim(components, &spec.r_cycle, &weights, "bus-pc-psi", t);
    let psi_claims = pc_psi.component_claims.clone();
    // Δ\* per component.
    let mut onehot = Vec::with_capacity(d);
    let mut onehot_claims = Vec::with_capacity(d);
    for (i, comp) in components.iter().enumerate() {
        let (oh, cyc_pt) = prove_onehot(comp, t, &format!("bus-onehot-{i}"));
        let mut pt_r = oh.r.clone();
        pt_r.extend_from_slice(&cyc_pt);
        let mut pt_rp = oh.r_prime.clone();
        pt_rp.extend_from_slice(&cyc_pt);
        onehot_claims.push([
            PcsPoint::new(pt_r, oh.v_at_r),
            PcsPoint::new(pt_rp, oh.v_at_rp),
        ]);
        onehot.push(oh);
    }
    let proof = BusShoutProof {
        phi_rounds,
        phi_component_claims: phi_component_claims.clone(),
        pc_psi,
        onehot,
    };
    let claims = ShoutPcsClaims {
        phi: phi_component_claims,
        psi: psi_claims,
        onehot: onehot_claims,
    };
    (proof, claims)
}

/// Verify the bytecode Shout; returns the claims left for the PCS.
pub fn verify_bus_shout(
    spec: &BusShoutSpec,
    d: usize,
    proof: &BusShoutProof,
    t: &mut Transcript,
) -> Result<ShoutPcsClaims, &'static str> {
    let (a, ell, n) = (spec.a, spec.ell, spec.n);
    if proof.phi_rounds.len() != a + ell + n
        || proof.phi_component_claims.len() != d
        || proof.onehot.len() != d
    {
        return Err("bus-shout: shape mismatch");
    }
    t.absorb_field("bus-in-bus", &[spec.bus_claim]);
    t.absorb_field("bus-in-pc", &[spec.pc_claim]);
    for (i, (_, _, c)) in spec.index_claims.iter().enumerate() {
        t.absorb_field(&format!("bus-in-idx-{i}"), &[*c]);
    }
    let (alpha0, gamma_bus, gammas) = sample_bus_coeffs(t, spec.index_claims.len());
    let mut claim = alpha0 + mul(gamma_bus, spec.bus_claim);
    for ((_, _, c), &g) in spec.index_claims.iter().zip(gammas.iter()) {
        claim += mul(g, *c);
    }
    let (public, mut terms) = bus_phi_factors(spec, alpha0, gamma_bus, &gammas);
    let dense_idx: Vec<usize> = (0..d).collect();
    for tm in terms.iter_mut() {
        tm.dense = dense_idx.clone();
    }
    let mut v = SumcheckVerifier::new(a + ell + n, claim, &public, terms, BindOrder::Increasing);
    let mut r_joint = Vec::with_capacity(a + ell + n);
    for msg in &proof.phi_rounds {
        r_joint.push(v.round_capturing(&RoundMsg(msg.clone()), t, "bus-phi")?);
    }
    let values: Vec<Gf128> = proof.phi_component_claims.iter().map(|c| c.value).collect();
    v.finish(&values)
        .map_err(|_| "bus-shout: phi terminal identity failed")?;
    for c in &proof.phi_component_claims {
        t.absorb_field("bus-phi-comp", &[c.value]);
    }
    // The pc Ψ verification (the claim value passed here is VS's pc claim —
    // a divergent honest value fails the round/terminal replay).
    let weights = pc_weights(&spec.pc_r_local, a);
    let psi_claims = verify_psi_claim(
        d,
        a / d,
        n,
        &spec.r_cycle,
        &weights,
        spec.pc_claim,
        &proof.pc_psi,
        "bus-pc-psi",
        t,
    )?;
    // Δ\* per component, capturing the cycle points for the claims.
    let mut onehot_claims = Vec::with_capacity(d);
    for (i, oh) in proof.onehot.iter().enumerate() {
        let cyc_pt = verify_onehot(oh, t, &format!("bus-onehot-{i}"))?;
        let mut pt_r = oh.r.clone();
        pt_r.extend_from_slice(&cyc_pt);
        let mut pt_rp = oh.r_prime.clone();
        pt_rp.extend_from_slice(&cyc_pt);
        onehot_claims.push([
            PcsPoint::new(pt_r, oh.v_at_r),
            PcsPoint::new(pt_rp, oh.v_at_rp),
        ]);
    }
    let phi = (0..d)
        .map(|i| {
            let pt = component_point(&r_joint, i, d, ell, n);
            PcsPoint::new(pt, proof.phi_component_claims[i].value)
        })
        .collect();
    Ok(ShoutPcsClaims {
        phi,
        onehot: onehot_claims,
        psi: psi_claims,
    })
}

// ---------------------------------------------------------------------------
// The selector checks (the routers' one-hot selectors vs. the ROM).
// ---------------------------------------------------------------------------

/// One selector factor to check: the router's terminal claim on a one-hot
/// selector field, authenticated against the ROM.
///
/// Each factor runs its **own** sum-check over the joint `(k, h_f, j)`
/// space — every variable is covered by at least one factor of the
/// summand (a factor replicated over an idle variable would contribute
/// `2^idle · v = 0` over the Boolean cube in characteristic two, so
/// per-factor checks keep the summand exact).
#[derive(Clone, Debug)]
pub struct SelectorCheckSpec {
    /// The factor's variable count (its slice of the router's `h` block).
    pub h_vars: usize,
    /// The eq point over the factor's vars (from the router's terminal).
    pub r_h: Vec<Gf128>,
    /// The router's terminal claim for this factor.
    pub claim: Gf128,
    /// The ROM's selector bits, indexed `(h_f low, k high)`.
    pub rom_sel: Vec<Gf128>,
}

/// The selector-check specification.
#[derive(Clone, Debug)]
pub struct SelectorShoutSpec {
    pub a: usize,
    pub n: usize,
    /// The router's cycle point.
    pub r_j: Vec<Gf128>,
    pub factors: Vec<SelectorCheckSpec>,
}

/// A selector-check proof: one run per factor.
#[derive(Clone, Debug)]
pub struct SelectorShoutProof {
    pub per_factor: Vec<SelectorFactorRun>,
}

/// One factor's sum-check run.
#[derive(Clone, Debug)]
pub struct SelectorFactorRun {
    pub rounds: Vec<Vec<Gf128>>,
    /// The fetch components' terminal claims (one per component).
    pub component_claims: Vec<PcsPoint>,
}

/// Prove the selector checks (one sum-check per factor). Returns the
/// proof and the per-factor component claims.
pub fn prove_selector_shout(
    spec: &SelectorShoutSpec,
    components: &[AddressComponent],
    label: &str,
    t: &mut Transcript,
) -> (SelectorShoutProof, SelectorPcsClaims) {
    let d = components.len();
    let (a, n) = (spec.a, spec.n);
    let mut per_factor = Vec::with_capacity(spec.factors.len());
    let mut all_claims = Vec::new();
    for (fi, f) in spec.factors.iter().enumerate() {
        let hv = f.h_vars;
        debug_assert_eq!(f.r_h.len(), hv);
        debug_assert_eq!(f.rom_sel.len(), 1usize << (hv + a));
        t.absorb_field(&format!("{label}-in-{fi}"), &[f.claim]);
        let jv = a + hv + n;
        // Dense: the fetch components padded over (k, h_f, j).
        let dense: Vec<Vec<Gf128>> = (0..d)
            .map(|i| pad_component(&components[i], i, d, hv))
            .collect();
        // Public: eq(r_j, j) over the j block; eq(r_h, h_f) over the h
        // block; sel_f(h_f, k) over (h_f, k).
        let sel_active: Vec<usize> = (a..a + hv).chain(0..a).collect();
        let public: Vec<Box<dyn PublicFactor + Send + Sync>> = vec![
            Box::new(JointTableFactor::new(
                magnetar_field::EqBits::build(&spec.r_j).values,
                (a + hv..a + hv + n).collect(),
                jv,
            )),
            Box::new(JointTableFactor::new(
                magnetar_field::EqBits::build(&f.r_h).values,
                (a..a + hv).collect(),
                jv,
            )),
            Box::new(JointTableFactor::new(f.rom_sel.clone(), sel_active, jv)),
        ];
        let terms = vec![Term {
            coeff: ONE,
            dense: (0..d).collect(),
            public: vec![0, 1, 2],
        }];
        let sc_spec = SumcheckSpec {
            num_vars: jv,
            dense,
            public,
            terms,
            claim: f.claim,
        };
        let mut prover = SumcheckProver::new_owned(sc_spec, BindOrder::Increasing);
        let mut rounds = Vec::with_capacity(jv);
        for _ in 0..jv {
            let msg = prover.message_cached();
            rounds.push(msg.0.clone());
            prover.absorb_and_fold(t, &format!("{label}-{fi}"));
        }
        let r_joint = prover.challenges();
        let (terminals, _) = prover.finish();
        let component_claims: Vec<PcsPoint> = (0..d)
            .map(|i| {
                let pt = component_point(&r_joint, i, d, hv, n);
                PcsPoint::new(pt, terminals[i])
            })
            .collect();
        for c in &component_claims {
            t.absorb_field(&format!("{label}-{fi}-comp"), &[c.value]);
            all_claims.push(c.clone());
        }
        per_factor.push(SelectorFactorRun {
            rounds,
            component_claims,
        });
    }
    let claims = SelectorPcsClaims {
        components: all_claims,
    };
    (SelectorShoutProof { per_factor }, claims)
}

/// Verify the selector checks; returns the per-factor component claims.
pub fn verify_selector_shout(
    spec: &SelectorShoutSpec,
    d: usize,
    proof: &SelectorShoutProof,
    label: &str,
    t: &mut Transcript,
) -> Result<SelectorPcsClaims, &'static str> {
    let (a, n) = (spec.a, spec.n);
    if proof.per_factor.len() != spec.factors.len() {
        return Err("selector: factor count mismatch");
    }
    let mut all_claims = Vec::new();
    for (fi, f) in spec.factors.iter().enumerate() {
        let hv = f.h_vars;
        let run = &proof.per_factor[fi];
        let jv = a + hv + n;
        if run.rounds.len() != jv || run.component_claims.len() != d {
            return Err("selector: shape mismatch");
        }
        t.absorb_field(&format!("{label}-in-{fi}"), &[f.claim]);
        let sel_active: Vec<usize> = (a..a + hv).chain(0..a).collect();
        let public: Vec<Box<dyn PublicFactor + Send + Sync>> = vec![
            Box::new(JointTableFactor::new(
                magnetar_field::EqBits::build(&spec.r_j).values,
                (a + hv..a + hv + n).collect(),
                jv,
            )),
            Box::new(JointTableFactor::new(
                magnetar_field::EqBits::build(&f.r_h).values,
                (a..a + hv).collect(),
                jv,
            )),
            Box::new(JointTableFactor::new(f.rom_sel.clone(), sel_active, jv)),
        ];
        let terms = vec![Term {
            coeff: ONE,
            dense: (0..d).collect(),
            public: vec![0, 1, 2],
        }];
        let mut v = SumcheckVerifier::new(jv, f.claim, &public, terms, BindOrder::Increasing);
        let mut r_joint = Vec::with_capacity(jv);
        for msg in &run.rounds {
            r_joint.push(v.round_capturing(&RoundMsg(msg.clone()), t, &format!("{label}-{fi}"))?);
        }
        let values: Vec<Gf128> = run.component_claims.iter().map(|c| c.value).collect();
        v.finish(&values)
            .map_err(|_| "selector: terminal identity failed")?;
        for (i, _) in (0..d).enumerate() {
            let pt = component_point(&r_joint, i, d, hv, n);
            all_claims.push(PcsPoint::new(pt, run.component_claims[i].value));
        }
        for c in &run.component_claims {
            t.absorb_field(&format!("{label}-{fi}-comp"), &[c.value]);
        }
    }
    Ok(SelectorPcsClaims {
        components: all_claims,
    })
}

// ---------------------------------------------------------------------------
// The register-shift stage-selector sum-check (the stage-select extension).
// ---------------------------------------------------------------------------

/// The stage-selector sum-check's public specification.
///
/// A register-shift stage row offers the shift router the *conditional*
/// amount pair `{identity, shift-by-2^k}` — `k` the row's stage, fixed by
/// the ROM's `sel_stage` one-hot — and the witness amount bit `b` (bit `k`
/// of the amount register) picks within it. This sum-check proves, over
/// the joint `(k', j)` space (stage index low, cycle high):
///
/// `q_stage = Σ_{k',j} eq(r_j, j)·oh(k', j)·[(1+b(j))·E₀ + b(j)·E(k')]`
/// `        + γ · Σ_{k',j} eq(r_j, j)·oh(k', j)·(b(j) + amt(k', j))`
///
/// where `oh` is the fetched `sel_stage` one-hot (discharged against the
/// ROM by the caller's selector check, exactly like the routers' other
/// selectors), `b` the committed shamt flag, `amt` the amount register's
/// low-byte bits (the `store-src` copy segment), `E₀ = eq(ρ_h, 0)` the
/// identity-routing constant, `E(k') = eq(ρ_h, 2^{k'})` the
/// shift-by-`2^{k'}` structure constants, and `γ` a transcript challenge
/// folding in the bit-binding `b(j) = bit_{k}(amount)` at the one-hot's
/// stage (the honest binding sum is zero).
///
/// The caller composes the claim additively in characteristic two: the
/// shift router proves `Ê_Qlo(ρ_h, r_j) = q_lo` for its *witness* amount
/// selector, this argument proves `q_stage = Ê_STAGElo(ρ_h, r_j)` for the
/// *structured* stage part, and the `sel_lo` selector check runs with
/// claim `q_lo + q_stage`, forcing `Q_lo = ROM(sel_lo) + STAGElo` at the
/// router's random point — hence, by Schwartz–Zippel over the multilinear
/// extensions, everywhere.
#[derive(Clone, Debug)]
pub struct StageSelectSpec {
    /// The stage one-hot's index width (3 bits, 8 slots, stages 0..=5 live).
    pub stage_bits: usize,
    /// Cycle variables.
    pub n: usize,
    /// The shift routing's amount-space width (`ρ_h.len()`).
    pub amount_bits: usize,
    /// The router's amount point `ρ_h` (the structure constants).
    pub r_amount: Vec<Gf128>,
    /// The router's cycle point (the `eq(r_j, ·)` weights).
    pub r_j: Vec<Gf128>,
}

impl StageSelectSpec {
    /// `E₀ = eq(ρ_h, 0) = Π_i (1 + ρ_{h,i})` — the identity routing's
    /// constant (an amount of zero).
    pub fn e0(&self) -> Gf128 {
        let mut acc = ONE;
        for &r in &self.r_amount {
            acc = mul(acc, ONE + r);
        }
        acc
    }

    /// `E[k'] = eq(ρ_h, 2^{k'})` — the shift-by-`2^{k'}` routing's
    /// structure constant (zero for slots the amount space cannot index;
    /// the one-hot is never live there).
    pub fn e_table(&self) -> Vec<Gf128> {
        let mut e = vec![ZERO; 1usize << self.stage_bits];
        for (k, v) in e.iter_mut().enumerate() {
            if k < self.amount_bits {
                // eq(ρ_h, e_k) = ρ_{h,k} · Π_{i≠k} (1 + ρ_{h,i}).
                let mut acc = self.r_amount[k];
                for (i, &r) in self.r_amount.iter().enumerate() {
                    if i != k {
                        acc = mul(acc, ONE + r);
                    }
                }
                *v = acc;
            }
        }
        e
    }
}

/// A stage-selector sum-check proof.
#[derive(Clone, Debug)]
pub struct StageSelectProof {
    /// The claim `q_stage` (the structured stage part's contribution to
    /// the router's amount selector).
    pub claim: Gf128,
    /// Round messages (the `(k', j)` space, `k'` bound first).
    pub rounds: Vec<Vec<Gf128>>,
    /// The fetched `sel_stage` one-hot's terminal claim (the caller
    /// discharges it through the `sel_stage` selector check).
    pub oh_terminal: Gf128,
    /// The shamt flag's terminal claim at the bound cycle point.
    pub b_terminal: Gf128,
    /// The amount low-byte's terminal claim at the bound point.
    pub amt_terminal: Gf128,
}

/// The symmetric sum-check driver: builds the spec's factors and terms.
fn stage_select_spec(
    spec: &StageSelectSpec,
    gamma: Gf128,
    dense: Vec<Vec<Gf128>>,
    claim: Gf128,
) -> SumcheckSpec {
    let sb = spec.stage_bits;
    let n = spec.n;
    let jv = sb + n;
    debug_assert_eq!(dense.len(), 3);
    // oh and amt over the joint (k', j); b replicated over the stage block.
    debug_assert_eq!(dense[0].len(), 1usize << jv);
    debug_assert_eq!(dense[1].len(), 1usize << jv);
    debug_assert_eq!(dense[2].len(), 1usize << jv);
    // Public: eq(r_j, j) over the cycle block; E(k') over the stage block.
    let public: Vec<Box<dyn PublicFactor + Send + Sync>> = vec![
        Box::new(JointTableFactor::new(spec.e_table(), (0..sb).collect(), jv)),
        Box::new(JointTableFactor::new(
            magnetar_field::EqBits::build(&spec.r_j).values,
            (sb..sb + n).collect(),
            jv,
        )),
    ];
    // Terms: oh·eq_j·[(1+b)E0 + b·E + γ(b + amt)].
    SumcheckSpec {
        num_vars: jv,
        dense,
        public,
        terms: vec![
            Term {
                coeff: spec.e0(),
                dense: vec![0],
                public: vec![1],
            },
            Term {
                coeff: spec.e0(),
                dense: vec![0, 1],
                public: vec![1],
            },
            Term {
                coeff: ONE,
                dense: vec![0, 1],
                public: vec![0, 1],
            },
            Term {
                coeff: gamma,
                dense: vec![0, 1],
                public: vec![1],
            },
            Term {
                coeff: gamma,
                dense: vec![0, 2],
                public: vec![1],
            },
        ],
        claim,
    }
}

/// The dense tables' padding: `oh` and `amt` live over the joint `(k', j)`
/// space; `b` is replicated over the stage block (its fold over an idle
/// variable is the identity in characteristic two, so its terminal claim
/// lands exactly at the cycle point).
fn stage_dense_tables(
    stage_bits: usize,
    n: usize,
    oh: &[Gf128],
    b: &[Gf128],
    amt: &[Gf128],
) -> Vec<Vec<Gf128>> {
    let jv = stage_bits + n;
    let mut b_pad = vec![ZERO; 1usize << jv];
    for j in 0..(1usize << n) {
        for k in 0..(1usize << stage_bits) {
            b_pad[k | (j << stage_bits)] = b[j];
        }
    }
    vec![oh.to_vec(), b_pad, amt.to_vec()]
}

/// The bound points a stage-selector proof leaves behind (the prover
/// needs them to build the `sel_stage` selector check's spec; the
/// verifier recomputes them from the transcript).
#[derive(Clone, Debug)]
pub struct StageSelectPoints {
    /// The bound stage-space point `ρ_{k'}`.
    pub rho_k: Vec<Gf128>,
    /// The bound cycle point `r_j*`.
    pub r_j: Vec<Gf128>,
}

/// Prove the stage-selector sum-check. `oh` is the fetched `sel_stage`
/// table over `(k', j)`, `b` the per-cycle shamt flags, `amt` the amount
/// register's low-byte bits over `(k', j)`. Returns the proof and the
/// bound points.
pub fn prove_stage_select(
    spec: &StageSelectSpec,
    oh: &[Gf128],
    b: &[Gf128],
    amt: &[Gf128],
    t: &mut Transcript,
) -> (StageSelectProof, StageSelectPoints) {
    let (sb, n) = (spec.stage_bits, spec.n);
    let gamma = t.challenge_field("stage-gamma");
    let e0 = spec.e0();
    let e = spec.e_table();
    let eq_j = magnetar_field::EqBits::build(&spec.r_j).values;
    // The honest claim.
    let mut claim = ZERO;
    for j in 0..(1usize << n) {
        for k in 0..(1usize << sb) {
            let ohv = oh[k | (j << sb)];
            if ohv == ZERO {
                continue;
            }
            // eq(r_j, j)·oh·[(1+b)·E0 + b·E(k') + γ·(b + amt(k', j))].
            let structure = e0 + mul(b[j], e0 + e[k]);
            let binding = b[j] + amt[k | (j << sb)];
            claim += mul(mul(ohv, eq_j[j]), structure + mul(gamma, binding));
        }
    }
    let dense = stage_dense_tables(sb, n, oh, b, amt);
    let sc = stage_select_spec(spec, gamma, dense, claim);
    let mut prover = SumcheckProver::new(&sc, BindOrder::Increasing);
    let mut rounds = Vec::with_capacity(sb + n);
    for _ in 0..(sb + n) {
        let msg = prover.message();
        rounds.push(msg.0.clone());
        prover.absorb_and_fold(t, "stage-select");
    }
    let challenges = prover.challenges();
    let (terminals, _) = prover.finish();
    let out = StageSelectProof {
        claim,
        rounds,
        oh_terminal: terminals[0],
        b_terminal: terminals[1],
        amt_terminal: terminals[2],
    };
    t.absorb_field(
        "stage-terminals",
        &[out.oh_terminal, out.b_terminal, out.amt_terminal],
    );
    let point = BindOrder::Increasing.point(sb + n, &challenges);
    let points = StageSelectPoints {
        rho_k: point[..sb].to_vec(),
        r_j: point[sb..].to_vec(),
    };
    (out, points)
}

/// Verify the stage-selector sum-check; returns the terminal claims and the
/// bound points for the caller's Mosaic/selector discharges.
#[derive(Clone, Debug)]
pub struct StageSelectClaims {
    /// The bound stage-space point `ρ_{k'}`.
    pub rho_k: Vec<Gf128>,
    /// The bound cycle point `r_j*`.
    pub r_j: Vec<Gf128>,
    /// The fetched one-hot's terminal (the `sel_stage` selector check's
    /// claim input).
    pub oh: Gf128,
    /// The shamt flag's value at `r_j*`.
    pub b: Gf128,
    /// The amount low-byte's value at `(ρ_{k'}, r_j*)`.
    pub amt: Gf128,
}

pub fn verify_stage_select(
    spec: &StageSelectSpec,
    proof: &StageSelectProof,
    t: &mut Transcript,
) -> Result<StageSelectClaims, &'static str> {
    let (sb, n) = (spec.stage_bits, spec.n);
    if proof.rounds.len() != sb + n {
        return Err("stage-select: round count mismatch");
    }
    let gamma = t.challenge_field("stage-gamma");
    // The verifier replays with the public factors only (the dense tables'
    // terminal claims arrive as the proof's claimed values).
    let jv = sb + n;
    let e0 = spec.e0();
    let public: Vec<Box<dyn PublicFactor + Send + Sync>> = vec![
        Box::new(JointTableFactor::new(spec.e_table(), (0..sb).collect(), jv)),
        Box::new(JointTableFactor::new(
            magnetar_field::EqBits::build(&spec.r_j).values,
            (sb..sb + n).collect(),
            jv,
        )),
    ];
    let terms = vec![
        Term {
            coeff: e0,
            dense: vec![0],
            public: vec![1],
        },
        Term {
            coeff: e0,
            dense: vec![0, 1],
            public: vec![1],
        },
        Term {
            coeff: ONE,
            dense: vec![0, 1],
            public: vec![0, 1],
        },
        Term {
            coeff: gamma,
            dense: vec![0, 1],
            public: vec![1],
        },
        Term {
            coeff: gamma,
            dense: vec![0, 2],
            public: vec![1],
        },
    ];
    let mut v = SumcheckVerifier::new(jv, proof.claim, &public, terms, BindOrder::Increasing);
    let mut challenges = Vec::with_capacity(jv);
    for msg in &proof.rounds {
        challenges.push(v.round_capturing(&RoundMsg(msg.clone()), t, "stage-select")?);
    }
    v.finish(&[proof.oh_terminal, proof.b_terminal, proof.amt_terminal])
        .map_err(|_| "stage-select: terminal identity failed")?;
    t.absorb_field(
        "stage-terminals",
        &[proof.oh_terminal, proof.b_terminal, proof.amt_terminal],
    );
    let point = BindOrder::Increasing.point(jv, &challenges);
    Ok(StageSelectClaims {
        rho_k: point[..sb].to_vec(),
        r_j: point[sb..].to_vec(),
        oh: proof.oh_terminal,
        b: proof.b_terminal,
        amt: proof.amt_terminal,
    })
}

#[cfg(test)]
mod multi_factor_tests {
    use super::*;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rng.gen::<u128>())
    }

    #[test]
    fn selector_two_factors_roundtrip() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let (a, n, d) = (2usize, 2, 1);
        let ad = a / d;
        let fetch_seq = [0usize, 1, 2, 3];
        let mut bits = vec![false; (1usize << ad) * (1usize << n)];
        for (j, &k) in fetch_seq.iter().enumerate() {
            bits[k | (j << ad)] = true;
        }
        let comps = vec![crate::AddressComponent::from_bits(ad, n, &bits)];
        let hv_lo = 2usize;
        let hv_hi = 2usize;
        let rom_sel_lo: Vec<Gf128> = {
            let mut sel = vec![ZERO; (1usize << hv_lo) * (1usize << a)];
            for k in 0..(1usize << a) {
                if rng.gen::<bool>() {
                    let hot = rng.gen_range(0..(1usize << hv_lo));
                    sel[hot | (k << hv_lo)] = ONE;
                }
            }
            sel
        };
        let rom_sel_hi: Vec<Gf128> = {
            let mut sel = vec![ZERO; (1usize << hv_hi) * (1usize << a)];
            for k in 0..(1usize << a) {
                if rng.gen::<bool>() {
                    let hot = rng.gen_range(0..(1usize << hv_hi));
                    sel[hot | (k << hv_hi)] = ONE;
                }
            }
            sel
        };
        let r_j: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
        let r_h_lo: Vec<Gf128> = (0..hv_lo).map(|_| rg(&mut rng)).collect();
        let r_h_hi: Vec<Gf128> = (0..hv_hi).map(|_| rg(&mut rng)).collect();
        let q_tables: [Vec<Gf128>; 2] = {
            let mk = |rom_sel: &Vec<Gf128>, hv: usize| {
                let mut q = vec![ZERO; (1usize << hv) * (1usize << n)];
                for (j, &k) in fetch_seq.iter().enumerate() {
                    for h in 0..(1usize << hv) {
                        q[h | (j << hv)] = rom_sel[h | (k << hv)];
                    }
                }
                q
            };
            [mk(&rom_sel_lo, hv_lo), mk(&rom_sel_hi, hv_hi)]
        };
        let q_claims = [
            magnetar_field::eval_mle(&q_tables[0], &[r_h_lo.clone(), r_j.clone()].concat()),
            magnetar_field::eval_mle(&q_tables[1], &[r_h_hi.clone(), r_j.clone()].concat()),
        ];
        let spec = SelectorShoutSpec {
            a,
            n,
            r_j: r_j.clone(),
            factors: vec![
                SelectorCheckSpec {
                    h_vars: hv_lo,
                    r_h: r_h_lo.clone(),
                    claim: q_claims[0],
                    rom_sel: rom_sel_lo.clone(),
                },
                SelectorCheckSpec {
                    h_vars: hv_hi,
                    r_h: r_h_hi.clone(),
                    claim: q_claims[1],
                    rom_sel: rom_sel_hi.clone(),
                },
            ],
        };
        let mut t = Transcript::new("sel2", &[b"t"]);
        let (proof, _) = prove_selector_shout(&spec, &comps, "sel2", &mut t);
        let mut tv = Transcript::new("sel2", &[b"t"]);
        let out = verify_selector_shout(&spec, 1, &proof, "sel2", &mut tv)
            .expect("two-factor selector verifies");
        assert_eq!(out.components.len(), 2); // one per factor
    }
}
